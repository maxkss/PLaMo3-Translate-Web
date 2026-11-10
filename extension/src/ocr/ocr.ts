/**
 * PP-OCRv5 pipeline: DB text detection → CTC line recognition → paragraph grouping → colour sampling.
 * Mirrors PaddleOCR's pre/post-processing (inference.yml): BGR input, det ImageNet normalisation with long side
 * resized and rounded to /32, DB threshold 0.3 / box 0.6 / unclip 1.5; rec height 48, (x/255-0.5)/0.5, greedy CTC.
 * Boxes are axis-aligned (no rotated text).
 */

import type { InferenceSession, Tensor } from "onnxruntime-common";
import type { OcrLine, OcrParagraph, Rgb } from "../shared/common.ts";

type Ort = { Tensor: typeof Tensor };

export interface Pixels {
  width: number;
  height: number;
  /** RGBA */
  data: Uint8ClampedArray;
}

type Box = [x: number, y: number, w: number, h: number];

const DET_THRESH = 0.3;
const BOX_THRESH = 0.6;
const UNCLIP = 1.5;
const REC_H = 48;
const REC_MAX_W = 3200;
const DROP_SCORE = 0.5;

// ---------------------------------------------------------------- detection

/** Long side of the detector input: small images are upscaled (small text), large ones capped for speed. */
function detScale(w: number, h: number): number {
  const long = Math.max(w, h);
  const target = Math.min(Math.max(long, 960), 1920);
  return target / long;
}

export async function detect(ort: Ort, session: InferenceSession, img: Pixels): Promise<Box[]> {
  const s = detScale(img.width, img.height);
  const W = Math.max(32, Math.round((img.width * s) / 32) * 32);
  const H = Math.max(32, Math.round((img.height * s) / 32) * 32);
  const sx = img.width / W;
  const sy = img.height / H;
  const input = new Float32Array(3 * W * H);
  const resized = resize(img, 0, 0, img.width, img.height, W, H);
  // BGR, (x/255 - mean) / std with mean/std listed in BGR channel order (as PaddleOCR applies them)
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const plane = W * H;
  for (let i = 0; i < plane; i++) {
    const r = resized[i * 4] / 255, g = resized[i * 4 + 1] / 255, b = resized[i * 4 + 2] / 255;
    input[i] = (b - mean[0]) / std[0];
    input[plane + i] = (g - mean[1]) / std[1];
    input[2 * plane + i] = (r - mean[2]) / std[2];
  }
  const feeds = { [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, H, W]) };
  const out = await session.run(feeds);
  const t = out[session.outputNames[0]];
  const prob = (await t.getData()) as Float32Array;
  t.dispose();
  return dbBoxes(prob, W, H).map(([x, y, w, h]): Box => {
    const bx = Math.max(0, x * sx), by = Math.max(0, y * sy);
    return [bx, by, Math.min(img.width, (x + w) * sx) - bx, Math.min(img.height, (y + h) * sy) - by];
  });
}

/** Connected components of the thresholded probability map → unclipped axis-aligned boxes (map coordinates). */
function dbBoxes(prob: Float32Array, W: number, H: number): Box[] {
  const n = W * H;
  const label = new Int32Array(n);
  const stack = new Int32Array(n);
  const boxes: Box[] = [];
  let next = 0;
  for (let start = 0; start < n; start++) {
    if (label[start] || prob[start] <= DET_THRESH) continue;
    next++;
    let sp = 0;
    stack[sp++] = start;
    label[start] = next;
    let x0 = W, y0 = H, x1 = -1, y1 = -1, sum = 0, count = 0;
    while (sp) {
      const p = stack[--sp];
      const x = p % W, y = (p - x) / W;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      sum += prob[p];
      count++;
      // 4-connectivity keeps adjacent lines apart
      if (x > 0 && !label[p - 1] && prob[p - 1] > DET_THRESH) (label[p - 1] = next), (stack[sp++] = p - 1);
      if (x < W - 1 && !label[p + 1] && prob[p + 1] > DET_THRESH) (label[p + 1] = next), (stack[sp++] = p + 1);
      if (y > 0 && !label[p - W] && prob[p - W] > DET_THRESH) (label[p - W] = next), (stack[sp++] = p - W);
      if (y < H - 1 && !label[p + W] && prob[p + W] > DET_THRESH) (label[p + W] = next), (stack[sp++] = p + W);
    }
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    if (Math.min(w, h) < 3 || sum / count < BOX_THRESH) continue;
    // DB unclip: grow the shrunk text kernel by area * ratio / perimeter
    const d = (w * h * UNCLIP) / (2 * (w + h));
    const bx = Math.max(0, x0 - d), by = Math.max(0, y0 - d);
    boxes.push([bx, by, Math.min(W, x1 + 1 + d) - bx, Math.min(H, y1 + 1 + d) - by]);
  }
  return boxes;
}

// ---------------------------------------------------------------- recognition

/** A crop is read as a vertical column (rotated before recognition) when it is clearly taller than wide. */
const isVertical = (b: Box) => b[3] >= b[2] * 1.5;

export async function recognize(ort: Ort, session: InferenceSession, dict: string[], img: Pixels, boxes: Box[]): Promise<OcrLine[]> {
  const items = boxes.map((box, i) => {
    const vertical = isVertical(box);
    const [w, h] = vertical ? [box[3], box[2]] : [box[2], box[3]];
    return { i, box, vertical, ratio: w / Math.max(1, h) };
  });
  // batches of similar aspect ratio waste little padding
  items.sort((a, b) => a.ratio - b.ratio);
  const out: (OcrLine | null)[] = new Array(boxes.length).fill(null);
  const BATCH = 8;
  for (let k = 0; k < items.length; k += BATCH) {
    const batch = items.slice(k, k + BATCH);
    const maxW = Math.min(REC_MAX_W, Math.max(320, ...batch.map((b) => Math.ceil(REC_H * b.ratio))));
    const plane = REC_H * maxW;
    const input = new Float32Array(batch.length * 3 * plane); // zero padding on the right, as PaddleOCR
    batch.forEach((it, bi) => {
      const [x, y, w, h] = it.box;
      const tw = Math.max(1, Math.min(maxW, Math.ceil(REC_H * it.ratio)));
      const px = it.vertical ? rotateCCW(resize(img, x, y, w, h, REC_H, tw), REC_H, tw) : resize(img, x, y, w, h, tw, REC_H);
      const base = bi * 3 * plane;
      for (let yy = 0; yy < REC_H; yy++) {
        for (let xx = 0; xx < tw; xx++) {
          const s = (yy * tw + xx) * 4;
          const d = base + yy * maxW + xx;
          input[d] = px[s + 2] / 127.5 - 1; // B
          input[d + plane] = px[s + 1] / 127.5 - 1; // G
          input[d + 2 * plane] = px[s] / 127.5 - 1; // R
        }
      }
    });
    const res = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [batch.length, 3, REC_H, maxW]) });
    const t = res[session.outputNames[0]];
    const probs = (await t.getData()) as Float32Array;
    const [, T, C] = t.dims as number[];
    t.dispose();
    batch.forEach((it, bi) => {
      const { text, score } = ctcDecode(probs, bi * T * C, T, C, dict);
      if (text.trim() && score >= DROP_SCORE) out[it.i] = { box: it.box, text, score };
    });
  }
  return out.filter((l): l is OcrLine => !!l);
}

/** Greedy CTC: argmax per step, collapse repeats, drop blanks (0). The last class is a space. */
function ctcDecode(p: Float32Array, off: number, T: number, C: number, dict: string[]): { text: string; score: number } {
  let text = "";
  let prev = -1;
  let sum = 0;
  let n = 0;
  for (let t = 0; t < T; t++) {
    let best = 0;
    let bv = -Infinity;
    const o = off + t * C;
    for (let c = 0; c < C; c++) if (p[o + c] > bv) (bv = p[o + c]), (best = c);
    if (best !== 0 && best !== prev) {
      text += best === C - 1 ? " " : (dict[best - 1] ?? "");
      sum += bv;
      n++;
    }
    prev = best;
  }
  return { text: text.trim(), score: n ? sum / n : 0 };
}

// ---------------------------------------------------------------- image helpers

/** Bilinear resize of the (x, y, w, h) region to tw × th, RGBA. */
function resize(img: Pixels, x: number, y: number, w: number, h: number, tw: number, th: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(tw * th * 4);
  const { width: W, height: H, data } = img;
  const fx = w / tw, fy = h / th;
  for (let ty = 0; ty < th; ty++) {
    const syf = Math.min(H - 1, Math.max(0, y + (ty + 0.5) * fy - 0.5));
    const y0 = Math.floor(syf), y1 = Math.min(H - 1, y0 + 1), wy = syf - y0;
    for (let tx = 0; tx < tw; tx++) {
      const sxf = Math.min(W - 1, Math.max(0, x + (tx + 0.5) * fx - 0.5));
      const x0 = Math.floor(sxf), x1 = Math.min(W - 1, x0 + 1), wx = sxf - x0;
      const a = (y0 * W + x0) * 4, b = (y0 * W + x1) * 4, c = (y1 * W + x0) * 4, d = (y1 * W + x1) * 4;
      const o = (ty * tw + tx) * 4;
      for (let k = 0; k < 4; k++) {
        out[o + k] = (data[a + k] * (1 - wx) + data[b + k] * wx) * (1 - wy) + (data[c + k] * (1 - wx) + data[d + k] * wx) * wy;
      }
    }
  }
  return out;
}

/** Rotate a w × h RGBA image 90° counter-clockwise (→ h × w), like np.rot90 in PaddleOCR. */
function rotateCCW(px: Uint8ClampedArray, w: number, h: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(px.length);
  // source (x, y) → destination (y, w - 1 - x) in an h-wide image
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const s = (y * w + x) * 4;
      const d = ((w - 1 - x) * h + y) * 4;
      out[d] = px[s];
      out[d + 1] = px[s + 1];
      out[d + 2] = px[s + 2];
      out[d + 3] = px[s + 3];
    }
  }
  return out;
}

// ---------------------------------------------------------------- layout

const CJK = /[　-ヿ㐀-鿿豈-﫿＀-￯]/;

/** Join lines of one paragraph: no space between CJK text, a space otherwise; a trailing hyphen joins words. */
function joinLines(lines: string[]): string {
  let out = "";
  for (const l of lines) {
    if (!out) out = l;
    else if (out.endsWith("-") && /^[a-z]/.test(l)) out = out.slice(0, -1) + l;
    else if (CJK.test(out.at(-1)!) || CJK.test(l[0])) out += l;
    else out += " " + l;
  }
  return out;
}

const overlap = (a0: number, a1: number, b0: number, b1: number) => Math.min(a1, b1) - Math.max(a0, b0);
const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)] ?? 0;
};
const union = (bs: Box[]): Box => {
  const x0 = Math.min(...bs.map((b) => b[0])), y0 = Math.min(...bs.map((b) => b[1]));
  const x1 = Math.max(...bs.map((b) => b[0] + b[2])), y1 = Math.max(...bs.map((b) => b[1] + b[3]));
  return [x0, y0, x1 - x0, y1 - y0];
};

/** Raw colour measurements of some text lines (see `measure`). */
export interface Style {
  bg: Rgb;
  fg: Rgb;
  /** RMS distance of the background ring from its median colour: high on photos and gradients */
  noise: number;
  /** share of the line boxes covered by text-coloured pixels */
  ink: number;
  /** typical stroke thickness relative to the line height: higher for bold text */
  stroke: number;
}

export type StyledLine = OcrLine & { style: Style; ink?: [number, number]; span?: [number, number] };

const cdist = (a: Rgb, b: Rgb) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

/** Lines of one paragraph share the text colour and (on plain backgrounds) the background colour. */
function sameStyle(a: Style, b: Style): boolean {
  if (cdist(a.fg, b.fg) > 70) return false;
  // a busy ring makes the background estimate unreliable only when both are busy (text on one photo); a badge next
  // to a label on a plain background is a different element
  const busy = (x: Style) => x.noise > 30;
  return busy(a) && busy(b) ? true : !busy(a) && !busy(b) && cdist(a.bg, b.bg) < 50;
}

/**
 * Group recognised lines into paragraphs: first merge pieces of the same line split by the detector (same row, small
 * gap, same colours), then stack lines of similar size and colour that are close and overlap along the other axis.
 * Vertical lines (columns) are grouped the same way with the axes swapped and read right to left.
 */
export function group(lines: StyledLine[]): (Omit<OcrParagraph, "bg" | "fg" | "busy" | "bold" | "room" | "fontPx" | "inkBox"> & { lines: StyledLine[] })[] {
  const paras: (Omit<OcrParagraph, "bg" | "fg" | "busy" | "bold" | "room" | "fontPx" | "inkBox"> & { lines: StyledLine[] })[] = [];
  for (const vertical of [false, true]) {
    // work in "line coordinates": u along the line, v across it
    const uv = (b: Box) => (vertical ? { u0: b[1], u1: b[1] + b[3], v0: b[0], v1: b[0] + b[2] } : { u0: b[0], u1: b[0] + b[2], v0: b[1], v1: b[1] + b[3] });
    // a tall box with a single character is just a character, not a column
    const isColumn = (l: OcrLine) => isVertical(l.box) && l.text.length >= 2;
    const items = lines.filter((l) => isColumn(l) === vertical).map((l) => ({ ...l, ...uv(l.box) }));
    if (!items.length) continue;

    // 1) same-row pieces → one line (UI elements in a row — tabs, a label and its badge — are further apart)
    items.sort((a, b) => a.u0 - b.u0);
    const merged: typeof items = [];
    for (const it of items) {
      const size = it.v1 - it.v0;
      const m = merged.find((p) => {
        const ps = p.v1 - p.v0;
        const gap = it.u0 - p.u1;
        return overlap(p.v0, p.v1, it.v0, it.v1) > 0.6 * Math.min(size, ps) && gap < 0.4 * Math.max(size, ps) && gap > -0.5 * size && sameStyle(p.style, it.style);
      });
      if (m) {
        const sep = CJK.test(m.text.at(-1)!) || CJK.test(it.text[0]) ? "" : " ";
        m.text += sep + it.text;
        m.box = union([m.box, it.box]);
        m.score = Math.min(m.score, it.score);
        // the merged line's ink is the union of both pieces' ink
        const hull = (a?: [number, number], b?: [number, number]): [number, number] | undefined =>
          a && b ? [Math.min(a[0], b[0]), Math.max(a[1], b[1])] : (a ?? b);
        m.ink = hull(m.ink, it.ink);
        m.span = hull(m.span, it.span);
        Object.assign(m, uv(m.box));
      } else merged.push({ ...it });
    }

    // 2) stack lines into paragraphs (union-find)
    const parent = merged.map((_, i) => i);
    const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    for (let i = 0; i < merged.length; i++) {
      for (let j = i + 1; j < merged.length; j++) {
        const a = merged[i], b = merged[j];
        const sa = a.v1 - a.v0, sb = b.v1 - b.v0;
        // a title and the body under it are separate paragraphs
        if (Math.max(sa, sb) > 1.25 * Math.min(sa, sb)) continue;
        const gap = Math.max(a.v0, b.v0) - Math.min(a.v1, b.v1);
        // line spacing inside a paragraph is tight; a larger gap starts a new block
        if (gap > 0.5 * Math.min(sa, sb) || gap < -0.4 * Math.min(sa, sb)) continue;
        const ov = overlap(a.u0, a.u1, b.u0, b.u1);
        // lines of a paragraph overlap along the line, or start at the same place
        if (ov <= 0 && Math.abs(a.u0 - b.u0) > sa) continue;
        // a name above a grey preview, a date next to a message: different styles, different blocks
        if (!sameStyle(a.style, b.style)) continue;
        parent[find(i)] = find(j);
      }
    }
    const groups = new Map<number, typeof merged>();
    merged.forEach((m, i) => {
      const r = find(i);
      groups.set(r, [...(groups.get(r) ?? []), m]);
    });
    for (const g of groups.values()) {
      // reading order: top to bottom, or right to left for columns
      g.sort((a, b) => (vertical ? b.v0 - a.v0 : a.v0 - b.v0));
      paras.push({
        box: union(g.map((l) => l.box)),
        lines: g.map(({ box, text, score, style, ink, span }) => ({ box, text, score, style, ink, span })),
        text: joinLines(g.map((l) => l.text)),
        vertical,
        lineSize: median(g.map((l) => l.v1 - l.v0)),
      });
    }
  }
  // reading order of paragraphs: top to bottom, then left to right
  return paras.sort((a, b) => a.box[1] - b.box[1] || a.box[0] - b.box[0]);
}

// ---------------------------------------------------------------- colours

const dist2 = (d: Uint8ClampedArray, i: number, c: Rgb) => (d[i] - c[0]) ** 2 + (d[i + 1] - c[1]) ** 2 + (d[i + 2] - c[2]) ** 2;
const lum = ([r, g, b]: Rgb) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

/**
 * Background: per-channel median of a thin ring just outside the line boxes. Text: mean of the pixels inside the lines
 * that differ most from the background (top 15%). Also how busy the background is and how thick the strokes are.
 */
export function measure(img: Pixels, boxes: Box[]): Style {
  const { width: W, height: H, data } = img;
  const ring: number[] = [];
  const step = (n: number) => Math.max(1, Math.floor(n / 60));
  const add = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    ring.push((Math.floor(y) * W + Math.floor(x)) * 4);
  };
  for (const [x, y, w, h] of boxes) {
    const pad = Math.max(2, h * 0.08);
    for (let xx = x; xx < x + w; xx += step(w)) add(xx, y - pad), add(xx, y + h + pad);
    for (let yy = y; yy < y + h; yy += step(h)) add(x - pad, yy), add(x + w + pad, yy);
  }
  const bg: Rgb = ring.length ? [0, 1, 2].map((k) => median(ring.map((i) => data[i + k]))) as Rgb : [255, 255, 255];
  // 80th percentile of the ring's distance from the background: a small icon or emoji touching the ring does not make
  // the background "busy", a photo or gradient does
  const rd = ring.map((i) => Math.sqrt(dist2(data, i, bg))).sort((a, b) => a - b);
  const noise = rd.length ? rd[Math.floor(rd.length * 0.8)] : 0;

  const samples: { i: number; d: number }[] = [];
  for (const [x, y, w, h] of boxes) {
    const sx = Math.max(1, Math.floor(w / 120)), sy = Math.max(1, Math.floor(h / 32));
    for (let yy = Math.floor(y); yy < Math.min(H, y + h); yy += sy) {
      for (let xx = Math.floor(x); xx < Math.min(W, x + w); xx += sx) {
        const i = (yy * W + xx) * 4;
        samples.push({ i, d: dist2(data, i, bg) });
      }
    }
  }
  samples.sort((a, b) => b.d - a.d);
  const top = samples.slice(0, Math.max(1, Math.floor(samples.length * 0.15)));
  let fg: Rgb = [0, 0, 0];
  let ink = 0;
  if (top.length && top[0].d > 0) {
    const s = [0, 0, 0];
    for (const { i } of top) (s[0] += data[i]), (s[1] += data[i + 1]), (s[2] += data[i + 2]);
    fg = [Math.round(s[0] / top.length), Math.round(s[1] / top.length), Math.round(s[2] / top.length)];
    // pixels closer to the text colour than to the background
    const half = (cdist(fg, bg) / 2) ** 2;
    ink = samples.filter((p) => p.d > half && dist2(data, p.i, fg) < p.d).length / samples.length;
  }
  return { bg, fg, noise, ink, stroke: strokeWidth(img, boxes, bg, fg) };
}

/**
 * Mean stroke thickness in pixels, sub-pixel: for thin strokes the ink area is about thickness × length and the
 * outline about 2 × length, so thickness ≈ 2 · area / perimeter. The area uses antialiased coverage (where each pixel
 * lies between the background and text colour); the perimeter counts mask edges at 50% coverage.
 */
function strokeWidth(img: Pixels, boxes: Box[], bg: Rgb, fg: Rgb): number {
  const { width: W, height: H, data } = img;
  const d = [fg[0] - bg[0], fg[1] - bg[1], fg[2] - bg[2]];
  const dd = d[0] ** 2 + d[1] ** 2 + d[2] ** 2;
  if (dd < 20 ** 2) return 0;
  const alpha = (i: number) => Math.min(1, Math.max(0, ((data[i] - bg[0]) * d[0] + (data[i + 1] - bg[1]) * d[1] + (data[i + 2] - bg[2]) * d[2]) / dd));
  let area = 0, edges = 0;
  for (const [bx, by, bw, bh] of boxes) {
    const x0 = Math.max(0, Math.floor(bx)), x1 = Math.min(W, Math.ceil(bx + bw));
    const y0 = Math.max(0, Math.floor(by)), y1 = Math.min(H, Math.ceil(by + bh));
    for (let y = y0; y < y1; y++) {
      let prev = false;
      for (let x = x0; x < x1; x++) {
        const i = (y * W + x) * 4;
        const a = alpha(i);
        area += a;
        const on = a >= 0.5;
        if (on !== prev && x > x0) edges++;
        prev = on;
        if (y > y0 && on !== alpha(i - W * 4) >= 0.5) edges++;
      }
    }
  }
  return edges ? (2 * area) / edges : 0;
}

/** WCAG relative luminance / contrast ratio. */
function luminance([r, g, b]: Rgb): number {
  const c = (v: number) => ((v /= 255) <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  return 0.2126 * c(r) + 0.7152 * c(g) + 0.0722 * c(b);
}
const contrastRatio = (a: Rgb, b: Rgb) => {
  const [l1, l2] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (l1 + 0.05) / (l2 + 0.05);
};

/**
 * Colours to draw with. Muted secondary text (light grey on white) keeps its colour; only a text colour that is
 * practically indistinguishable from the background (a failed estimate) falls back to black/white.
 */
export function drawColours({ bg, fg }: Style): { bg: Rgb; fg: Rgb } {
  if (contrastRatio(fg, bg) < 1.35) fg = lum(bg) > 128 ? [0, 0, 0] : [255, 255, 255];
  return { bg, fg };
}

/**
 * How far the background continues unchanged on each side of the text's ink rectangle (image pixels) — left, right,
 * up, down — checked along several lines across that side: room a longer translation can take without covering an
 * icon, a button edge or other text.
 */
export function freeRoom(img: Pixels, ink: Box, bg: Rgb, limit: number, skip: number): [number, number, number, number] {
  const { width: W, height: H, data } = img;
  // the ink rectangle can miss a faint edge of the outermost glyph: start a little outside it
  const [x, y, w, h] = [ink[0] - skip, ink[1] - skip, ink[2] + 2 * skip, ink[3] + 2 * skip];
  // a faint icon (light grey on a tinted background) is still an obstacle
  const tol = 20 ** 2;
  const scan = (horizontal: boolean, dir: -1 | 1) => {
    let best = Infinity;
    for (const f of [0.05, 0.3, 0.5, 0.7, 0.95]) {
      let n = 0;
      if (horizontal) {
        const yy = Math.min(H - 1, Math.max(0, Math.floor(y + h * f)));
        for (let xx = dir < 0 ? Math.floor(x) - 1 : Math.ceil(x + w); xx >= 0 && xx < W && n < limit; xx += dir, n++) {
          if (dist2(data, (yy * W + xx) * 4, bg) > tol) break;
        }
      } else {
        const xx = Math.min(W - 1, Math.max(0, Math.floor(x + w * f)));
        for (let yy = dir < 0 ? Math.floor(y) - 1 : Math.ceil(y + h); yy >= 0 && yy < H && n < limit; yy += dir, n++) {
          if (dist2(data, (yy * W + xx) * 4, bg) > tol) break;
        }
      }
      best = Math.min(best, n);
    }
    return best + skip;
  };
  return [scan(true, -1), scan(true, 1), scan(false, -1), scan(false, 1)];
}

// ---------------------------------------------------------------- glyph size

/**
 * Ink height of a full-height glyph as a share of the font size (em), by what the text contains. Ideographs, kana and
 * Hangul fill most of the em square; Latin depends on which letters occur (ascenders/capitals/digits, descenders).
 */
export function inkRatio(text: string): number {
  if (/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(text)) return EM_CJK;
  const asc = /[A-Z0-9bdfhklt]/.test(text), desc = /[gjpqy]/.test(text);
  return asc && desc ? 0.93 : asc || desc ? 0.72 : 0.52;
}
// calibrated on rendered CJK text (test/typo-eval.mjs): ideographs and kana span ~93% of the em vertically
const EM_CJK = 0.93;

/**
 * Where the text's ink is across the line (rows for horizontal text, columns for vertical), found from the projection
 * of text-coloured pixels: the contiguous ink band nearest the box centre (small gaps inside glyphs are bridged; ink of
 * a neighbouring line that the detector box overlaps is not part of it). Returns [start, end] in image pixels.
 */
export function inkBand(img: Pixels, box: Box, vertical: boolean, s: Pick<Style, "bg" | "fg">): [number, number] {
  const { width: W, height: H, data } = img;
  const [x, y, w, h] = box;
  const across = Math.max(1, Math.round(vertical ? w : h));
  const along = vertical ? h : w;
  const step = Math.max(1, Math.floor(along / 400));
  const contrast = (cdist(s.fg, s.bg) / 2) ** 2;
  const prof = new Float32Array(across);
  for (let a = 0; a < across; a++) {
    let n = 0, tot = 0;
    for (let t = 0; t < along; t += step) {
      const px = Math.floor(vertical ? x + a : x + t), py = Math.floor(vertical ? y + t : y + a);
      if (px < 0 || py < 0 || px >= W || py >= H) continue;
      const i = (py * W + px) * 4;
      const db = dist2(data, i, s.bg);
      tot++;
      if (db > contrast && dist2(data, i, s.fg) < db) n++;
    }
    prof[a] = tot ? n / tot : 0;
  }
  // rows with real strokes: a thin stroke reaching past the glyph body (a slash, a bracket, an accent) does not count
  const peak = Math.max(...prof);
  const on = (a: number) => prof[a] >= Math.max(0.01, 0.12 * peak);
  // contiguous bands, bridging gaps up to 12% of the box thickness
  const bands: [number, number][] = [];
  const bridge = Math.max(1, Math.round(across * 0.12));
  for (let a = 0; a < across; a++) {
    if (!on(a)) continue;
    const last = bands.at(-1);
    if (last && a - last[1] <= bridge + 1) last[1] = a;
    else bands.push([a, a]);
  }
  if (!bands.length) return vertical ? [x, x + w] : [y, y + h];
  const c = across / 2;
  const dist = ([a, b]: [number, number]) => (c < a ? a - c : c > b ? c - b : 0);
  // the band at the centre, or else the thickest one
  bands.sort((p, q) => dist(p) - dist(q) || q[1] - q[0] - (p[1] - p[0]));
  const [a, b] = bands[0];
  const o = vertical ? x : y;
  return [o + a, o + b + 1];
}

/**
 * Where the text's ink starts and ends along the line (x for horizontal text, y for a column), inside its ink band:
 * the detector box is padded by an amount that depends on its aspect ratio, the ink is not.
 */
export function inkSpan(img: Pixels, box: Box, vertical: boolean, band: [number, number], s: Pick<Style, "bg" | "fg">): [number, number] {
  const { width: W, height: H, data } = img;
  const [x, y, w, h] = box;
  const contrast = (cdist(s.fg, s.bg) / 2) ** 2;
  const a0 = Math.floor(band[0]), a1 = Math.ceil(band[1]);
  const t0 = Math.floor(vertical ? y : x), t1 = Math.ceil(vertical ? y + h : x + w);
  const across = Math.max(1, a1 - a0);
  const step = Math.max(1, Math.floor(across / 40));
  let first = -1, last = -1;
  for (let t = t0; t < t1; t++) {
    let n = 0;
    for (let a = a0; a < a1; a += step) {
      const px = vertical ? a : t, py = vertical ? t : a;
      if (px < 0 || py < 0 || px >= W || py >= H) continue;
      const i = (py * W + px) * 4;
      const db = dist2(data, i, s.bg);
      if (db > contrast && dist2(data, i, s.fg) < db) n++;
    }
    if (n * step >= Math.max(1, across * 0.04)) {
      if (first < 0) first = t;
      last = t;
    }
  }
  return first < 0 ? [t0, t1] : [first, last + 1];
}

/**
 * Background along each side of the ink rectangle (left, right, top, bottom): median colour and 80th-percentile
 * spread. A background that is busy around the text as a whole but uniform along each side is a smooth gradient,
 * which can be painted over opaquely instead of blurred.
 */
export function sideBackgrounds(img: Pixels, ink: Box, pad: number): { colour: Rgb; noise: number }[] {
  const { width: W, height: H, data } = img;
  const [x, y, w, h] = [ink[0] - pad, ink[1] - pad, ink[2] + 2 * pad, ink[3] + 2 * pad];
  const side = (pts: [number, number][]) => {
    const idx = pts.filter(([px, py]) => px >= 0 && py >= 0 && px < W && py < H).map(([px, py]) => (Math.floor(py) * W + Math.floor(px)) * 4);
    if (!idx.length) return { colour: [255, 255, 255] as Rgb, noise: 0 };
    const colour = [0, 1, 2].map((k) => median(idx.map((i) => data[i + k]))) as Rgb;
    const d = idx.map((i) => Math.sqrt(dist2(data, i, colour))).sort((a, b) => a - b);
    return { colour, noise: d[Math.floor(d.length * 0.8)] };
  };
  const n = 40;
  const along = (a0: number, a1: number) => Array.from({ length: n }, (_, k) => a0 + ((a1 - a0) * (k + 0.5)) / n);
  return [
    side(along(y, y + h).map((v) => [x, v])),
    side(along(y, y + h).map((v) => [x + w, v])),
    side(along(x, x + w).map((v) => [v, y])),
    side(along(x, x + w).map((v) => [v, y + h])),
  ];
}
