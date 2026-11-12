/**
 * From recognised lines to styled paragraphs and the plan for erasing them — the part of image translation between the
 * OCR models and the erasers. Pure functions of the pixels (no models, no DOM), shared by the OCR worker and the
 * evaluation harness (eval/erase.mjs).
 */

import type { OcrLine, OcrParagraph, OcrResult } from "../shared/common.ts";
import { textBlocks } from "../shared/blocks.ts";
import { markGlyphs, markText, patchRect, type Rect } from "./erase.ts";
import { drawColours, freeRoom, glyphGaps, group, inkBand, inkRatio, inkSpan, measure, sideBackgrounds, type Pixels } from "./ocr.ts";

/**
 * Background noise (80th percentile distance from its median colour) above which it counts as busy, and the stroke
 * score (thickness against a regular face's at that size) above which text counts as bold.
 */
const BUSY_NOISE = 28;
const BOLD_SCORE = 1.2;

const busy0 = (s: { noise: number }) => s.noise > BUSY_NOISE;
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;

/**
 * Group the lines into paragraphs and measure each one's style. `px` is what the OCR read; `k` scales its coordinates
 * back to the image's natural size (the result's coordinates).
 */
export function describe(px: Pixels, lines: OcrLine[], k: number): OcrParagraph[] {
  const sc = (b: [number, number, number, number]): [number, number, number, number] => [b[0] * k, b[1] * k, b[2] * k, b[3] * k];
  const all = lines.map((l) => l.box);
  const styled = lines.map((l) => {
    const style = measure(px, [l.box], all);
    const vertical = l.box[3] >= l.box[2] * 1.5 && l.text.length >= 2;
    const ink = inkBand(px, l.box, vertical, style);
    return { ...l, style, ink, span: inkSpan(px, l.box, vertical, ink, style) };
  });
  return group(styled).map((p) => {
    const style = measure(px, p.lines.map((l) => l.box), all);
    // font size from the ink band of each line (the detector box height depends on its aspect ratio)
    const fontPx = median(p.lines.map((l) => (l.ink ? l.ink[1] - l.ink[0] : p.lineSize * 0.8))) / inkRatio(p.text);
    // rectangle covered by the glyph ink: lines' bands across, spans along
    const across = p.lines.flatMap((l) => l.ink ?? []), along = p.lines.flatMap((l) => l.span ?? []);
    const a0 = Math.min(...across), a1 = Math.max(...across), t0 = Math.min(...along), t1 = Math.max(...along);
    const inkBox: [number, number, number, number] = p.vertical ? [a0, t0, a1 - a0, t1 - t0] : [t0, a0, t1 - t0, a1 - a0];
    const room = freeRoom(px, inkBox, style.bg, Math.max(px.width, px.height), Math.max(2, Math.round(fontPx * 0.12)));
    // letter-spacing: the typical gap between glyphs beyond a face's own (about 0.1 em). Only clear tracking counts —
    // spaced-out capitals, not a loosely drawn face — and only where the gaps could be seen (not on a photo).
    const gaps = busy0(style) ? [] : p.lines.flatMap((l) => (l.ink && l.span ? glyphGaps(px, p.vertical, l.ink, l.span, style) : []));
    const gap = gaps.length >= 3 ? median(gaps) / fontPx : 0;
    const tracking = gap > 0.25 ? Math.min(1.5, Math.round((gap - 0.1) * 100) / 100) : 0;
    // stroke thickness against what a regular face has at this size: about 0.062 × ink height, plus ~0.55 px of
    // antialiasing blur (which matters for small text); bold faces come out ~1.3× or more
    const inkH = fontPx * 0.93;
    const boldScore = style.stroke / (0.55 + 0.062 * inkH);
    const busy = style.noise > BUSY_NOISE;
    // a gradient: each side uniform, the colour changing across the text
    let gradient: OcrResult["paragraphs"][number]["gradient"];
    if (busy) {
      const [l, r, t, b] = sideBackgrounds(px, inkBox, Math.max(2, fontPx * 0.2));
      if (Math.max(l.noise, r.noise, t.noise, b.noise) < BUSY_NOISE) {
        const dist = (a: number[], c: number[]) => Math.hypot(a[0] - c[0], a[1] - c[1], a[2] - c[2]);
        gradient = dist(l.colour, r.colour) >= dist(t.colour, b.colour) ? { to: "right", from: l.colour, end: r.colour } : { to: "bottom", from: t.colour, end: b.colour };
      }
    }
    return {
      ...p,
      ...drawColours(style),
      busy,
      gradient,
      bold: boldScore > BOLD_SCORE,
      tracking,
      room: room.map((r) => r * k) as [number, number, number, number],
      inkBox: sc(inkBox),
      noise: Math.round(style.noise),
      ink: Math.round(boldScore * 100) / 100,
      box: sc(p.box),
      lineSize: p.lineSize * k,
      fontPx: fontPx * k,
      lines: p.lines.map(({ box, text, score, ink, span }) => ({
        box: sc(box),
        text,
        score,
        ink: ink && ([ink[0] * k, ink[1] * k] as [number, number]),
        span: span && ([span[0] * k, span[1] * k] as [number, number]),
      })),
    };
  });
}

export interface ErasePlan {
  /** 1 = to erase, over the pixels the OCR read */
  mask: Uint8Array;
  /** which paragraph's hole each masked pixel is (index + 1; 0 = none) */
  owner: Int16Array;
  /**
   * holes as thin as the strokes — on plain and gradient backgrounds, and on photos where the glyphs could be told
   * from what is behind them (`glyphs`): filled by diffusion
   */
  thin: { i: number; rect: Rect; fontPx: number; glyphs?: boolean }[];
  /** paragraphs on photos whose whole box is the hole: for the inpainting model */
  box: { i: number; rect: Rect; fontPx: number }[];
}

/**
 * Which paragraphs get erased, with what mask and by which eraser. Text left untranslated is still erased where a
 * translated paragraph's area overlaps it (a "40%" inside a translated headline would otherwise show through).
 * Sets the text colour of paragraphs masked by their glyphs.
 */
export function planErase(px: Pixels, paragraphs: OcrParagraph[], k: number, targetLang: string, imageHeight: number, opts: { glyphs?: boolean } = {}): ErasePlan {
  const mask = new Uint8Array(px.width * px.height);
  const owner = new Int16Array(px.width * px.height);
  const plan: ErasePlan = { mask, owner, thin: [], box: [] };
  let scratch: Uint8Array | undefined;
  // (a line of a phrase block is replaced with its block, however short it is)
  const translated = textBlocks(paragraphs, targetLang, imageHeight).translate;
  const areas = paragraphs.map((p) => {
    const g = p.fontPx * 0.5;
    return [p.inkBox[0] - g, p.inkBox[1] - g, p.inkBox[2] + 2 * g, p.inkBox[3] + 2 * g];
  });
  const overlaps = (a: number[], b: number[]) => a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];
  paragraphs.forEach((p, i) => {
    // untranslated text is erased as part of the translated paragraph it lies in: its hole shows in that one's patch
    const host = translated[i] ? i : paragraphs.findIndex((_, j) => translated[j] && j !== i && overlaps(areas[i], areas[j]));
    if (host < 0) return;
    // back to the pixels the OCR read (the result is in the image's natural size)
    const pp = { inkBox: p.inkBox.map((v) => v / k) as Rect, fontPx: p.fontPx / k, bg: p.bg, fg: p.fg };
    const photo = p.busy && !p.gradient;
    // on a photo: the glyphs alone when every line's text colour can be told from what is behind it, else the box
    const lineBoxes = p.lines.map((l) => l.box.map((v) => v / k) as Rect);
    let glyphs = false;
    if (photo && opts.glyphs !== false) {
      // into a scratch mask first (every line has to succeed), read back over the paragraph's own area only: one
      // buffer for the image, not one per paragraph
      scratch ??= new Uint8Array(mask.length);
      const own = scratch;
      const colours = lineBoxes.map((b) => markGlyphs(px, own, b, pp.fontPx, p.vertical));
      glyphs = colours.every((c) => c);
      const reach = Math.ceil(0.3 * pp.fontPx) + 4;
      const bx0 = Math.max(0, Math.floor(Math.min(...lineBoxes.map((b) => b[0])) - reach)), by0 = Math.max(0, Math.floor(Math.min(...lineBoxes.map((b) => b[1])) - reach));
      const bx1 = Math.min(px.width, Math.ceil(Math.max(...lineBoxes.map((b) => b[0] + b[2])) + reach)), by1 = Math.min(px.height, Math.ceil(Math.max(...lineBoxes.map((b) => b[1] + b[3])) + reach));
      for (let y = by0; y < by1; y++) {
        for (let x = bx0; x < bx1; x++) {
          const n = y * px.width + x;
          if (!own[n]) continue;
          own[n] = 0;
          if (glyphs) mask[n] = 1;
        }
      }
      if (glyphs) {
        // the glyphs' own colour is the text colour (the colour measured against the surroundings of a label is the
        // label's): the translation is written in it
        p.fg = [0, 1, 2].map((ch) => median(colours.map((c) => c![ch]))) as OcrParagraph["fg"];
      }
    }
    if (!glyphs) markText(px, mask, pp, photo);
    // the hole is this paragraph's, where no earlier one has it
    const rect = patchRect(px, pp);
    for (let y = rect[1]; y < rect[1] + rect[3]; y++) for (let x = rect[0]; x < rect[0] + rect[2]; x++) if (mask[y * px.width + x] && !owner[y * px.width + x]) owner[y * px.width + x] = host + 1;
    if (photo && !glyphs) plan.box.push({ i, rect: patchRect(px, pp), fontPx: pp.fontPx });
    else plan.thin.push({ i, rect: patchRect(px, pp), fontPx: pp.fontPx, glyphs });
  });
  return plan;
}
