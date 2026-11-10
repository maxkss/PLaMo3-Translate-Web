/**
 * Erasing the original text before the translation is drawn, like Google Lens: only the glyph pixels are removed and
 * filled from their surroundings, so the background (gradients, borders, room the translation grows into) stays.
 *
 *  - mask: on a plain or gradient background, the pixels inside the text's ink box that look like text — between the
 *    background and text colour (coverage > 12%, which includes antialiased edges) or clearly unlike the background —
 *    dilated by ~0.08 em. On a photo the colours are unreliable, so the whole ink box (+0.1 em) is the hole.
 *  - plain/gradient: push-pull fill — a mask-weighted image pyramid is built down, the holes are filled from coarser
 *    levels on the way up, then smoothed. Exact colours, no model, a few ms per paragraph.
 *  - photo: MI-GAN inpainting (a mobile inpainting GAN, ~7M parameters) on a square crop around the hole (resized to
 *    512²), composited back into the hole only, with the colour offset between the model's output and the image on a
 *    ring around the hole removed.
 */

import type { InferenceSession, Tensor } from "onnxruntime-common";
import type { OcrParagraph } from "../shared/common.ts";
import type { Pixels } from "./ocr.ts";

type Ort = { Tensor: typeof Tensor };
export type Rect = [x: number, y: number, w: number, h: number];

/** The area a paragraph's patch covers: its ink box with a margin (image pixels, clipped to the image). */
export function patchRect(img: Pixels, p: Pick<OcrParagraph, "inkBox" | "fontPx">): Rect {
  const pad = Math.max(2, 0.25 * p.fontPx);
  const x0 = Math.max(0, Math.floor(p.inkBox[0] - pad)), y0 = Math.max(0, Math.floor(p.inkBox[1] - pad));
  const x1 = Math.min(img.width, Math.ceil(p.inkBox[0] + p.inkBox[2] + pad)), y1 = Math.min(img.height, Math.ceil(p.inkBox[1] + p.inkBox[3] + pad));
  return [x0, y0, x1 - x0, y1 - y0];
}

/** Mark the paragraph's text pixels in `mask` (1 = to erase). */
export function markText(img: Pixels, mask: Uint8Array, p: Pick<OcrParagraph, "inkBox" | "fontPx" | "bg" | "fg">, wholeBox: boolean): void {
  const { width: W, height: H, data } = img;
  const [ix, iy, iw, ih] = p.inkBox;
  if (wholeBox) {
    const g = Math.max(2, 0.1 * p.fontPx);
    for (let y = Math.max(0, Math.floor(iy - g)); y < Math.min(H, Math.ceil(iy + ih + g)); y++) {
      for (let x = Math.max(0, Math.floor(ix - g)); x < Math.min(W, Math.ceil(ix + iw + g)); x++) mask[y * W + x] = 1;
    }
    return;
  }
  const [br, bg, bb] = p.bg;
  const dr = p.fg[0] - br, dg = p.fg[1] - bg, db = p.fg[2] - bb;
  const dd = Math.max(1, dr * dr + dg * dg + db * db);
  const x0 = Math.max(0, Math.floor(ix) - 1), x1 = Math.min(W, Math.ceil(ix + iw) + 2);
  const y0 = Math.max(0, Math.floor(iy) - 1), y1 = Math.min(H, Math.ceil(iy + ih) + 2);
  const hit = new Uint8Array((x1 - x0) * (y1 - y0));
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = (y * W + x) * 4;
      const r = data[i] - br, g = data[i + 1] - bg, b = data[i + 2] - bb;
      const a = (r * dr + g * dg + b * db) / dd;
      if (a > 0.12 || r * r + g * g + b * b > 40 * 40) hit[(y - y0) * (x1 - x0) + (x - x0)] = 1;
    }
  }
  // dilate (square neighbourhood) into the mask
  const rad = Math.max(2, Math.round(0.08 * p.fontPx));
  const w = x1 - x0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (!hit[(y - y0) * w + (x - x0)]) continue;
      for (let yy = Math.max(0, y - rad); yy <= Math.min(H - 1, y + rad); yy++) {
        mask.fill(1, yy * W + Math.max(0, x - rad), yy * W + Math.min(W - 1, x + rad) + 1);
      }
    }
  }
}

/** Fill the masked pixels of `rect` (in place, RGBA) from the unmasked ones, with `margin` pixels around it as context. */
export function pushPull(img: Pixels, mask: Uint8Array, rect: Rect, margin: number): void {
  const { width: W, height: H, data } = img;
  const cx0 = Math.max(0, rect[0] - margin), cy0 = Math.max(0, rect[1] - margin);
  const cx1 = Math.min(W, rect[0] + rect[2] + margin), cy1 = Math.min(H, rect[1] + rect[3] + margin);
  const w = cx1 - cx0, h = cy1 - cy0;
  // level 0: colour × weight and weight
  type Level = { w: number; h: number; c: Float32Array; k: Float32Array };
  const l0: Level = { w, h, c: new Float32Array(w * h * 3), k: new Float32Array(w * h) };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const gi = (y + cy0) * W + (x + cx0), li = y * w + x;
      if (mask[gi]) continue;
      l0.k[li] = 1;
      l0.c[li * 3] = data[gi * 4];
      l0.c[li * 3 + 1] = data[gi * 4 + 1];
      l0.c[li * 3 + 2] = data[gi * 4 + 2];
    }
  }
  const levels = [l0];
  while (Math.min(levels.at(-1)!.w, levels.at(-1)!.h) > 2) {
    const a = levels.at(-1)!;
    const nw = Math.ceil(a.w / 2), nh = Math.ceil(a.h / 2);
    const b: Level = { w: nw, h: nh, c: new Float32Array(nw * nh * 3), k: new Float32Array(nw * nh) };
    for (let y = 0; y < a.h; y++) {
      for (let x = 0; x < a.w; x++) {
        const s = y * a.w + x, d = (y >> 1) * nw + (x >> 1);
        b.k[d] += a.k[s];
        b.c[d * 3] += a.c[s * 3];
        b.c[d * 3 + 1] += a.c[s * 3 + 1];
        b.c[d * 3 + 2] += a.c[s * 3 + 2];
      }
    }
    levels.push(b);
  }
  // pull: each level's estimate = its own mean where known, blended with the coarser estimate where weight is low
  let est = new Float32Array(levels.at(-1)!.w * levels.at(-1)!.h * 3);
  {
    const top = levels.at(-1)!;
    for (let i = 0; i < top.w * top.h; i++) for (let ch = 0; ch < 3; ch++) est[i * 3 + ch] = top.k[i] ? top.c[i * 3 + ch] / top.k[i] : 0;
  }
  for (let li = levels.length - 2; li >= 0; li--) {
    const a = levels[li], up = levels[li + 1];
    const next = new Float32Array(a.w * a.h * 3);
    for (let y = 0; y < a.h; y++) {
      for (let x = 0; x < a.w; x++) {
        const i = y * a.w + x, u = (y >> 1) * up.w + (x >> 1);
        const k = Math.min(1, a.k[i]);
        for (let ch = 0; ch < 3; ch++) {
          const mean = a.k[i] ? a.c[i * 3 + ch] / a.k[i] : 0;
          next[i * 3 + ch] = k * mean + (1 - k) * est[u * 3 + ch];
        }
      }
    }
    est = next;
  }
  // smooth inside the holes (Jacobi passes over hole pixels only), then write back
  const holes: number[] = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (mask[(y + cy0) * W + (x + cx0)]) holes.push(y * w + x);
  for (let pass = 0; pass < 20; pass++) {
    for (const i of holes) {
      const x = i % w, y = (i - x) / w;
      for (let ch = 0; ch < 3; ch++) {
        let s = 0, n = 0;
        if (x > 0) (s += est[(i - 1) * 3 + ch]), n++;
        if (x < w - 1) (s += est[(i + 1) * 3 + ch]), n++;
        if (y > 0) (s += est[(i - w) * 3 + ch]), n++;
        if (y < h - 1) (s += est[(i + w) * 3 + ch]), n++;
        est[i * 3 + ch] = s / n;
      }
    }
  }
  for (const i of holes) {
    const x = i % w, y = (i - x) / w;
    const gi = ((y + cy0) * W + (x + cx0)) * 4;
    data[gi] = est[i * 3];
    data[gi + 1] = est[i * 3 + 1];
    data[gi + 2] = est[i * 3 + 2];
  }
}

/** Bilinear resample of an RGBA region to tw × th. */
function resample(src: Uint8ClampedArray, sw: number, sh: number, tw: number, th: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(tw * th * 4);
  const fx = sw / tw, fy = sh / th;
  for (let y = 0; y < th; y++) {
    const syf = Math.min(sh - 1, Math.max(0, (y + 0.5) * fy - 0.5));
    const y0 = Math.floor(syf), y1 = Math.min(sh - 1, y0 + 1), wy = syf - y0;
    for (let x = 0; x < tw; x++) {
      const sxf = Math.min(sw - 1, Math.max(0, (x + 0.5) * fx - 0.5));
      const x0 = Math.floor(sxf), x1 = Math.min(sw - 1, x0 + 1), wx = sxf - x0;
      const a = (y0 * sw + x0) * 4, b = (y0 * sw + x1) * 4, c = (y1 * sw + x0) * 4, d = (y1 * sw + x1) * 4;
      for (let k = 0; k < 4; k++) out[(y * tw + x) * 4 + k] = (src[a + k] * (1 - wx) + src[b + k] * wx) * (1 - wy) + (src[c + k] * (1 - wx) + src[d + k] * wx) * wy;
    }
  }
  return out;
}

const SIDE = 512;

/**
 * Inpaint the masked pixels of `rect` with MI-GAN (in place). The crop is a square around the hole, at least 2.2× its
 * size (context) and 256 px, so the model sees enough background. Returns false (and changes nothing) when the holes
 * cover too much of the crop to be reconstructed — e.g. a small thumbnail covered in text.
 */
export async function inpaintFill(ort: Ort, session: InferenceSession, img: Pixels, mask: Uint8Array, rect: Rect): Promise<boolean> {
  const { width: W, height: H, data } = img;
  const side = Math.min(Math.max(W, H), Math.max(256, Math.round(2.2 * Math.max(rect[2], rect[3]))));
  const cx = rect[0] + rect[2] / 2, cy = rect[1] + rect[3] / 2;
  const x0 = Math.round(Math.max(0, Math.min(W - Math.min(side, W), cx - side / 2)));
  const y0 = Math.round(Math.max(0, Math.min(H - Math.min(side, H), cy - side / 2)));
  const cw = Math.min(side, W - x0), ch = Math.min(side, H - y0);
  const crop = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) crop.set(data.subarray(((y0 + y) * W + x0) * 4, ((y0 + y) * W + x0 + cw) * 4), y * cw * 4);
  // how much of the hole's own neighbourhood (the rect grown by half its size) is hole: text packed over a small
  // picture leaves nothing to reconstruct it from
  const g = Math.max(rect[2], rect[3]) / 2;
  const nx0 = Math.max(0, Math.floor(rect[0] - g)), ny0 = Math.max(0, Math.floor(rect[1] - g));
  const nx1 = Math.min(W, Math.ceil(rect[0] + rect[2] + g)), ny1 = Math.min(H, Math.ceil(rect[1] + rect[3] + g));
  let holes = 0;
  for (let y = ny0; y < ny1; y++) for (let x = nx0; x < nx1; x++) holes += mask[y * W + x];
  if (holes > 0.35 * (nx1 - nx0) * (ny1 - ny0)) return false;
  const small = resample(crop, cw, ch, SIDE, SIDE);
  const plane = SIDE * SIDE;
  const holeMask = new Uint8Array(plane);
  for (let y = 0; y < SIDE; y++) {
    for (let x = 0; x < SIDE; x++) {
      // nearest source pixel; any hole pixel nearby counts (the model must not see the text's antialiased edge)
      const sx = Math.min(cw - 1, Math.floor(((x + 0.5) * cw) / SIDE)), sy = Math.min(ch - 1, Math.floor(((y + 0.5) * ch) / SIDE));
      holeMask[y * SIDE + x] = mask[(y0 + sy) * W + x0 + sx];
    }
  }
  // the hole as seen by the model, grown by one model pixel
  const grown = holeMask.slice();
  for (let y = 1; y < SIDE - 1; y++) for (let x = 1; x < SIDE - 1; x++) if (holeMask[y * SIDE + x]) for (const d of [-1, 1, -SIDE, SIDE]) grown[y * SIDE + x + d] = 1;
  // MI-GAN input: [known − 0.5, image × known] with known = 1 outside the hole and the image in [-1, 1]
  const input = new Float32Array(4 * plane);
  for (let i = 0; i < plane; i++) {
    const keep = grown[i] ? 0 : 1;
    input[i] = keep - 0.5;
    for (let k = 0; k < 3; k++) input[(k + 1) * plane + i] = (small[i * 4 + k] / 127.5 - 1) * keep;
  }
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 4, SIDE, SIDE]) });
  const t = res[session.outputNames[0]];
  const out = (await t.getData()) as Float32Array;
  t.dispose();
  const rgba = new Uint8ClampedArray(plane * 4);
  for (let i = 0; i < plane; i++) {
    for (let k = 0; k < 3; k++) rgba[i * 4 + k] = (out[k * plane + i] * 0.5 + 0.5) * 255;
    rgba[i * 4 + 3] = 255;
  }
  const back = resample(rgba, SIDE, SIDE, cw, ch);
  // colour offset between the model's output and the image on a ring around the hole (the model shifts colours slightly)
  const off = [0, 0, 0];
  let n = 0;
  for (let y = 1; y < ch - 1; y++) {
    for (let x = 1; x < cw - 1; x++) {
      const gi = (y0 + y) * W + x0 + x;
      if (mask[gi]) continue;
      if (!(mask[gi - 1] || mask[gi + 1] || mask[gi - W] || mask[gi + W] || mask[gi - 2] || mask[gi + 2])) continue;
      for (let k = 0; k < 3; k++) off[k] += data[gi * 4 + k] - back[(y * cw + x) * 4 + k];
      n++;
    }
  }
  if (n) for (let k = 0; k < 3; k++) off[k] /= n;
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const gi = (y0 + y) * W + x0 + x;
      if (!mask[gi]) continue;
      for (let k = 0; k < 3; k++) data[gi * 4 + k] = back[(y * cw + x) * 4 + k] + off[k];
    }
  }
  return true;
}

/** PNG data: URL of `rect` of the image. */
export async function encodePatch(img: Pixels, rect: Rect): Promise<string> {
  const [x, y, w, h] = rect;
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext("2d")!;
  const id = ctx.createImageData(w, h);
  for (let yy = 0; yy < h; yy++) id.data.set(img.data.subarray(((y + yy) * img.width + x) * 4, ((y + yy) * img.width + x + w) * 4), yy * w * 4);
  ctx.putImageData(id, 0, 0);
  const blob = await canvas.convertToBlob({ type: "image/png" });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = "";
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return "data:image/png;base64," + btoa(bin);
}
