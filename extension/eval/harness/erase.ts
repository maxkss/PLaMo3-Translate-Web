/**
 * Browser side of eval/erase.mjs: runs the extension's own OCR → paragraph → mask → erase code on one image and
 * scores every eraser against the true text-free background (Crello's clean composite).
 */
import * as ort from "onnxruntime-web/webgpu";
import { describe, planErase } from "../../src/ocr/analyse.ts";
import { inpaintFill, markText, patchRect, pushPull, type Rect } from "../../src/ocr/erase.ts";
import { detect, measure, recognize, type Pixels } from "../../src/ocr/ocr.ts";
import type { OcrParagraph } from "../../src/shared/common.ts";
import { VARIANTS, type Variant } from "./variants.ts";

type Sess = ort.InferenceSession;
const state: { det?: Sess; rec?: Sess; dict?: string[]; inpaint: Map<string, Sess> } = { inpaint: new Map() };

async function bytes(url: string): Promise<Uint8Array> {
  return new Uint8Array(await (await fetch(url)).arrayBuffer());
}

async function init(): Promise<string> {
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  const opts: ort.InferenceSession.SessionOptions = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
  state.det = await ort.InferenceSession.create(await bytes("/models/det_mobile.onnx"), opts);
  state.rec = await ort.InferenceSession.create(await bytes("/models/rec_mobile.onnx"), opts);
  state.dict = await (await fetch("/ocr/ppocrv5_dict.json")).json();
  return `threads=${ort.env.wasm.numThreads} isolated=${self.crossOriginIsolated}`;
}

/** An inpainting session, by model file and backend. */
async function inpaintSession(file: string, backend: "webgpu" | "wasm"): Promise<Sess> {
  const key = `${file}/${backend}`;
  if (!state.inpaint.has(key)) state.inpaint.set(key, await ort.InferenceSession.create(await bytes(`/models/${file}`), { executionProviders: [backend], graphOptimizationLevel: "all" }));
  return state.inpaint.get(key)!;
}

async function pixels(url: string): Promise<Pixels> {
  const bmp = await createImageBitmap(await (await fetch(url)).blob());
  const ctx = new OffscreenCanvas(bmp.width, bmp.height).getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0);
  return { width: bmp.width, height: bmp.height, data: ctx.getImageData(0, 0, bmp.width, bmp.height).data };
}

async function png(px: Pixels): Promise<string> {
  const c = new OffscreenCanvas(px.width, px.height);
  c.getContext("2d")!.putImageData(new ImageData(px.data as Uint8ClampedArray<ArrayBuffer>, px.width, px.height), 0, 0);
  const b = new Uint8Array(await (await c.convertToBlob({ type: "image/png" })).arrayBuffer());
  let bin = "";
  for (let i = 0; i < b.length; i += 0x8000) bin += String.fromCharCode(...b.subarray(i, i + 0x8000));
  return btoa(bin);
}

const lum = (d: Uint8ClampedArray, i: number) => 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];

export interface ParaScore {
  i: number;
  text: string;
  /** what the pipeline took the background for */
  kind: "plain" | "gradient" | "photo";
  /** mean gradient magnitude of the true background under the text: how much there is to reconstruct */
  texture: number;
  fontPx: number;
  /** share of the patch that is hole */
  hole: number;
  /** share of the true text pixels inside the mask (the rest stays visible) */
  maskRecall: number;
  /** mask area over true text area */
  maskExcess: number;
  /** mean absolute error against the true background over the hole and the true text (0–255), and leaving it as is */
  l1: number;
  l1None: number;
  /** mean absolute difference of gradient magnitudes there: blur and invented texture both cost */
  grad: number;
  /** worst 5% of pixel errors: leftover strokes and seams */
  p95: number;
  ms: number;
  filled: boolean;
  /** a photo paragraph that got a glyph mask instead of its whole box */
  glyphs?: boolean;
}

/**
 * Score what an eraser left in `rect` against the clean background. Counted are the pixels it was allowed to change
 * (its mask) and the pixels it had to change (the true text, grown by a pixel for antialiasing).
 */
function score(out: Pixels, clean: Pixels, orig: Pixels, mask: Uint8Array, truth: Uint8Array, rect: Rect): Pick<ParaScore, "hole" | "maskRecall" | "maskExcess" | "l1" | "l1None" | "grad" | "p95" | "texture"> {
  const W = out.width, H = out.height;
  const [x0, y0, w, h] = rect;
  let n = 0, l1 = 0, l1None = 0, grad = 0, holes = 0, text = 0, hit = 0, tex = 0;
  const errs: number[] = [];
  const gm = (d: Uint8ClampedArray, x: number, y: number) => {
    const i = (y * W + x) * 4;
    const gx = lum(d, x < W - 1 ? i + 4 : i) - lum(d, x > 0 ? i - 4 : i), gy = lum(d, y < H - 1 ? i + W * 4 : i) - lum(d, y > 0 ? i - W * 4 : i);
    return Math.hypot(gx, gy) / 2;
  };
  for (let y = y0; y < y0 + h; y++) {
    for (let x = x0; x < x0 + w; x++) {
      const p = y * W + x;
      const t = truth[p];
      if (t === 2) text++, (hit += mask[p] ? 1 : 0);
      if (mask[p]) holes++;
      if (!mask[p] && !t) continue;
      const i = p * 4;
      const e = (Math.abs(out.data[i] - clean.data[i]) + Math.abs(out.data[i + 1] - clean.data[i + 1]) + Math.abs(out.data[i + 2] - clean.data[i + 2])) / 3;
      l1 += e;
      errs.push(e);
      l1None += (Math.abs(orig.data[i] - clean.data[i]) + Math.abs(orig.data[i + 1] - clean.data[i + 1]) + Math.abs(orig.data[i + 2] - clean.data[i + 2])) / 3;
      const gc = gm(clean.data, x, y);
      grad += Math.abs(gm(out.data, x, y) - gc);
      tex += gc;
      n++;
    }
  }
  errs.sort((a, b) => a - b);
  const r = (v: number) => +v.toFixed(3);
  return { hole: r(holes / Math.max(1, w * h)), maskRecall: r(text ? hit / text : 1), maskExcess: r(holes / Math.max(1, text)), l1: r(l1 / Math.max(1, n)), l1None: r(l1None / Math.max(1, n)), grad: r(grad / Math.max(1, n)), p95: r(errs[Math.floor(errs.length * 0.95)] ?? 0), texture: r(tex / Math.max(1, n)) };
}

export interface Request {
  image: string;
  clean: string;
  mask: string;
  variants: string[];
  /** return the erased images (PNG, base64) of these variants */
  save?: boolean;
}

async function run(req: Request) {
  const [orig, clean, alpha] = await Promise.all([pixels(req.image), pixels(req.clean), pixels(req.mask)]);
  const W = orig.width, H = orig.height;
  // true text: 2 = ink, 1 = within a pixel of it (antialiasing)
  const truth = new Uint8Array(W * H);
  for (let p = 0; p < W * H; p++) if (alpha.data[p * 4 + 3] > 40) truth[p] = 2;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const p = y * W + x;
    if (truth[p] !== 2) continue;
    for (const d of [-1, 1, -W, W]) if (!truth[p + d]) truth[p + d] = 1;
  }
  const t0 = performance.now();
  const boxes = await detect(ort, state.det!, orig);
  const lines = await recognize(ort, state.rec!, state.dict!, orig, boxes);
  const paragraphs: OcrParagraph[] = describe(orig, lines, 1);
  const ocrMs = performance.now() - t0;
  const out: Record<string, { paras: ParaScore[]; png?: string }> = {};
  for (const name of req.variants) {
    const v: Variant = VARIANTS[name];
    const px: Pixels = { width: W, height: H, data: orig.data.slice() };
    const plan = v.plan ? v.plan(px, paragraphs, truth) : planErase(px, paragraphs, 1, "ko", H);
    const jobs = [...plan.thin.map((j) => ({ ...j, box: false })), ...plan.box.map((j) => ({ ...j, box: true, glyphs: false }))];
    const paras: ParaScore[] = [];
    for (const j of jobs) {
      const p = paragraphs[j.i];
      const kind = p.busy && !p.gradient ? "photo" : p.gradient ? "gradient" : "plain";
      const t = performance.now();
      const filled = await v.fill({ ort, px, mask: plan.mask, rect: j.rect, fontPx: j.fontPx, kind, glyphs: j.glyphs, inpaint: inpaintSession, paragraph: p });
      const ms = performance.now() - t;
      paras.push({ i: j.i, text: p.text.slice(0, 40), kind, fontPx: Math.round(p.fontPx), ms: +ms.toFixed(1), filled, glyphs: j.glyphs, ...score(px, clean, orig, plan.mask, truth, j.rect) });
    }
    out[name] = { paras, png: req.save ? await png(px) : undefined };
  }
  return { ocrMs: Math.round(ocrMs), paragraphs: paragraphs.length, variants: out };
}

/** The paragraphs the extension finds in an image (for looking into single cases). */
async function ocr(url: string): Promise<OcrParagraph[]> {
  const px = await pixels(url);
  return describe(px, await recognize(ort, state.rec!, state.dict!, px, await detect(ort, state.det!, px)), 1);
}

/** The recognised lines of an image with their measured colours, before grouping. */
async function lines(url: string) {
  const px = await pixels(url);
  const ls = await recognize(ort, state.rec!, state.dict!, px, await detect(ort, state.det!, px));
  return ls.map((l) => ({ text: l.text, box: l.box.map(Math.round), ...measure(px, [l.box]) }));
}

Object.assign(globalThis, { evalInit: init, evalErase: run, evalOcr: ocr, evalLines: lines, markText, patchRect, pushPull, inpaintFill });
