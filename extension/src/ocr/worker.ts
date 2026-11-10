/**
 * OCR worker for image translation. It has its own onnxruntime instance (and GPU device), separate from the
 * translation engine worker, so OCR sessions never interleave with the engine's captured decode graph.
 */

import * as ort from "onnxruntime-web/webgpu";
import { INPAINT_MODEL, OCR_MODELS, type ErasePatch, type OcrModelId, type OcrResult } from "../shared/common.ts";
import { getStoredFiles } from "../engine/storage.ts";
import { skipParagraph } from "../shared/skip.ts";
import { encodePatch, inpaintFill, markText, patchRect, pushPull, type Rect } from "./erase.ts";
import { detect, drawColours, freeRoom, group, inkBand, inkRatio, inkSpan, measure, recognize, sideBackgrounds, type Pixels } from "./ocr.ts";

/**
 * Background noise (80th percentile distance from its median colour) above which it counts as busy, and the stroke
 * score (thickness against a regular face's at that size) above which text counts as bold.
 */
const BUSY_NOISE = 28;
const BOLD_SCORE = 1.2;

export type OcrWorkerIn =
  | { t: "ocr"; job: number; image: ImageBitmap; extBase: string; targetLang: string; model: OcrModelId }
  /** erase the job's pending paragraphs (on photos) with the inpainting model, installed by now */
  | { t: "inpaint"; job: number }
  | { t: "unload" };
export type OcrWorkerOut =
  | { t: "done"; job: number; result: OcrResult }
  | { t: "patches"; job: number; patches: ErasePatch[] }
  | { t: "error"; job: number; message: string };

const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] ?? 0;
const post = (m: OcrWorkerOut, transfer: Transferable[] = []) => (self as unknown as Worker).postMessage(m, transfer);

interface Sessions {
  det: ort.InferenceSession;
  rec: ort.InferenceSession;
  dict: string[];
  backend: string;
}
let sessions: Promise<Sessions> | null = null;
let sessionsModel: OcrModelId | null = null;
let inpaintSession: Promise<ort.InferenceSession> | null = null;
/** the last images' pixels and erase masks, for the inpainting pass that follows OCR */
const erasing = new Map<number, { px: Pixels; mask: Uint8Array; k: number; pending: { i: number; rect: Rect }[] }>();

/**
 * Sessions (and the cached pixels) are released after a minute without work, so the models do not stay in memory
 * between translations — loading them again takes a fraction of a second.
 */
const IDLE_MS = 60_000;
let idleTimer: ReturnType<typeof setTimeout> | null = null;
async function releaseAll(): Promise<void> {
  const s = await sessions?.catch(() => null);
  const ip = await inpaintSession?.catch(() => null);
  sessions = null;
  sessionsModel = null;
  inpaintSession = null;
  erasing.clear();
  await s?.det.release();
  await s?.rec.release();
  await ip?.release();
}
function touch(): void {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = setTimeout(() => (chain = chain.then(releaseAll)), IDLE_MS);
}

/** one image at a time */
let chain: Promise<unknown> = Promise.resolve();

/** Largest side the image is read at (bigger images are downscaled first; boxes are mapped back). */
const MAX_SIDE = 4096;

async function load(extBase: string, model: OcrModelId): Promise<Sessions> {
  ort.env.wasm.wasmPaths = extBase + "ort/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  const [files, dict] = await Promise.all([getStoredFiles(OCR_MODELS[model]), fetch(extBase + "ocr/ppocrv5_dict.json").then((r) => r.json() as Promise<string[]>)]);
  const [detBytes, recBytes] = await Promise.all(files.map(async (f) => new Uint8Array(await f.arrayBuffer())));
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  const hasGpu = !!gpu && !!(await gpu.requestAdapter().catch(() => null));
  let lastErr: unknown;
  for (const backend of hasGpu ? (["webgpu", "wasm"] as const) : (["wasm"] as const)) {
    try {
      const opts: ort.InferenceSession.SessionOptions = { executionProviders: [backend], graphOptimizationLevel: "all" };
      const det = await ort.InferenceSession.create(detBytes, opts);
      const rec = await ort.InferenceSession.create(recBytes, opts);
      return { det, rec, dict, backend };
    } catch (e) {
      console.warn(`[plamo-ocr] ${backend} failed`, e);
      lastErr = e;
    }
  }
  throw lastErr;
}

function readPixels(image: ImageBitmap): { px: Pixels; scale: number } {
  const scale = Math.min(1, MAX_SIDE / Math.max(image.width, image.height));
  const w = Math.max(1, Math.round(image.width * scale));
  const h = Math.max(1, Math.round(image.height * scale));
  const ctx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#fff"; // transparent areas read as white, not black
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(image, 0, 0, w, h);
  const { data } = ctx.getImageData(0, 0, w, h);
  return { px: { width: w, height: h, data }, scale };
}

async function loadInpaint(): Promise<ort.InferenceSession> {
  const [file] = await getStoredFiles(INPAINT_MODEL);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  const hasGpu = !!gpu && !!(await gpu.requestAdapter().catch(() => null));
  let lastErr: unknown;
  for (const backend of hasGpu ? (["webgpu", "wasm"] as const) : (["wasm"] as const)) {
    try {
      return await ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: "all" });
    } catch (e) {
      console.warn(`[plamo-ocr] inpainting ${backend} failed`, e);
      lastErr = e;
    }
  }
  throw lastErr;
}

/** Inpaint the pending paragraphs of a job. */
async function runInpaint(job: number): Promise<ErasePatch[]> {
  const st = erasing.get(job);
  if (!st?.pending.length) return [];
  inpaintSession ??= loadInpaint().catch((e) => {
    inpaintSession = null;
    throw e;
  });
  const session = await inpaintSession;
  const patches: ErasePatch[] = [];
  for (const { i, rect } of st.pending) {
    // too much text to reconstruct: the overlay keeps its blurred fill there
    if (!(await inpaintFill(ort, session, st.px, st.mask, rect))) continue;
    patches.push({ i, rect: rect.map((v) => v * st.k) as Rect, url: await encodePatch(st.px, rect) });
  }
  st.pending = [];
  return patches;
}

async function run(job: number, image: ImageBitmap, extBase: string, targetLang: string, model: OcrModelId): Promise<OcrResult> {
  const t0 = performance.now();
  if (sessions && sessionsModel !== model) await releaseAll();
  sessionsModel = model;
  sessions ??= load(extBase, model).catch((e) => {
    sessions = null;
    throw e;
  });
  const s = await sessions;
  const width = image.width, height = image.height;
  const { px, scale } = readPixels(image);
  image.close();
  const tLoad = performance.now();
  const boxes = await detect(ort, s.det, px);
  const tDet = performance.now();
  const lines = await recognize(ort, s.rec, s.dict, px, boxes);
  const tRec = performance.now();
  const k = 1 / scale;
  const sc = (b: [number, number, number, number]): [number, number, number, number] => [b[0] * k, b[1] * k, b[2] * k, b[3] * k];
  const styled = lines.map((l) => {
    const style = measure(px, [l.box]);
    const vertical = l.box[3] >= l.box[2] * 1.5 && l.text.length >= 2;
    const ink = inkBand(px, l.box, vertical, style);
    return { ...l, style, ink, span: inkSpan(px, l.box, vertical, ink, style) };
  });
  const groups = group(styled);
  const paragraphs = groups.map((p) => {
    const style = measure(px, p.lines.map((l) => l.box));
    // font size from the ink band of each line (the detector box height depends on its aspect ratio)
    const fontPx = median(p.lines.map((l) => (l.ink ? l.ink[1] - l.ink[0] : p.lineSize * 0.8))) / inkRatio(p.text);
    // rectangle covered by the glyph ink: lines' bands across, spans along
    const across = p.lines.flatMap((l) => l.ink ?? []), along = p.lines.flatMap((l) => l.span ?? []);
    const a0 = Math.min(...across), a1 = Math.max(...across), t0 = Math.min(...along), t1 = Math.max(...along);
    const inkBox: [number, number, number, number] = p.vertical ? [a0, t0, a1 - a0, t1 - t0] : [t0, a0, t1 - t0, a1 - a0];
    const room = freeRoom(px, inkBox, style.bg, Math.max(px.width, px.height), Math.max(2, Math.round(fontPx * 0.12)));
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
  const tLayout = performance.now();
  // erase the text that will be translated: push-pull now for plain/gradient backgrounds, inpainting later for photos
  const mask = new Uint8Array(px.width * px.height);
  const plain: { i: number; rect: Rect; fontPx: number }[] = [];
  const pending: { i: number; rect: Rect }[] = [];
  // text left untranslated is still erased where a translated paragraph's area overlaps it (a "40%" inside a
  // translated headline would otherwise show through the translation)
  const translated = paragraphs.map((p) => !skipParagraph(p, targetLang, height));
  const areas = paragraphs.map((p) => {
    const g = p.fontPx * 0.5;
    return [p.inkBox[0] - g, p.inkBox[1] - g, p.inkBox[2] + 2 * g, p.inkBox[3] + 2 * g];
  });
  const overlaps = (a: number[], b: number[]) => a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3];
  paragraphs.forEach((p, i) => {
    if (!translated[i] && !paragraphs.some((_, j) => translated[j] && j !== i && overlaps(areas[i], areas[j]))) return;
    // back to the pixels the OCR read (the result is in the image's natural size)
    const pp = { inkBox: p.inkBox.map((v) => v / k) as Rect, fontPx: p.fontPx / k, bg: p.bg, fg: p.fg };
    const photo = p.busy && !p.gradient;
    markText(px, mask, pp, photo);
    if (photo) pending.push({ i, rect: patchRect(px, pp) });
    else plain.push({ i, rect: patchRect(px, pp), fontPx: pp.fontPx });
  });
  const patches: ErasePatch[] = [];
  for (const { i, rect, fontPx } of plain) {
    pushPull(px, mask, rect, Math.max(8, Math.round(1.5 * fontPx)));
    patches.push({ i, rect: rect.map((v) => v * k) as Rect, url: await encodePatch(px, rect) });
  }
  if (pending.length) {
    erasing.set(job, { px, mask, k, pending });
    // keep the two most recent images only
    for (const key of [...erasing.keys()].slice(0, -2)) erasing.delete(key);
  }
  const end = performance.now();
  const timing = { load: tLoad - t0, detect: tDet - tLoad, recognize: tRec - tDet, layout: tLayout - tRec, erase: end - tLayout };
  for (const key of Object.keys(timing) as (keyof typeof timing)[]) timing[key] = Math.round(timing[key]);
  return { width, height, paragraphs, patches, pending: pending.map((p) => p.i), backend: s.backend, ms: Math.round(end - t0), timing };
}

self.onmessage = (ev: MessageEvent<OcrWorkerIn>) => {
  const m = ev.data;
  touch();
  if (m.t === "unload") {
    chain = chain.then(releaseAll);
    return;
  }
  if (m.t === "inpaint") {
    chain = chain.then(() =>
      runInpaint(m.job).then(
        (patches) => post({ t: "patches", job: m.job, patches }),
        (e) => {
          console.error("[plamo-ocr] inpainting failed", e);
          post({ t: "patches", job: m.job, patches: [] });
        },
      ),
    );
    return;
  }
  chain = chain.then(() =>
    run(m.job, m.image, m.extBase, m.targetLang, m.model).then(
      (result) => post({ t: "done", job: m.job, result }),
      (e) => {
        console.error("[plamo-ocr] failed", e);
        post({ t: "error", job: m.job, message: String((e as Error)?.message ?? e) });
      },
    ),
  );
};
