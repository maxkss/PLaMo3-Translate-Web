/**
 * OCR worker for image translation. It has its own onnxruntime instance (and GPU device), separate from the
 * translation engine worker, so OCR sessions never interleave with the engine's captured decode graph.
 */

import * as ort from "onnxruntime-web/webgpu";
import { INPAINT_MODEL, OCR_MODELS, type ErasePatch, type OcrModelId, type OcrResult } from "../shared/common.ts";
import { getStoredFiles } from "../engine/storage.ts";
import { describe, planErase } from "./analyse.ts";
import { encodePatch, inpaintFill, pushPull, type Rect } from "./erase.ts";
import { detect, recognize, type Pixels } from "./ocr.ts";

export type OcrWorkerIn =
  | { t: "ocr"; job: number; image: ImageBitmap; extBase: string; targetLang: string; model: OcrModelId }
  /** erase the job's pending paragraphs (on photos) with the inpainting model, installed by now */
  | { t: "inpaint"; job: number }
  | { t: "unload" };
export type OcrWorkerOut =
  | { t: "done"; job: number; result: OcrResult }
  | { t: "patches"; job: number; patches: ErasePatch[] }
  | { t: "error"; job: number; message: string };

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
const erasing = new Map<number, { px: Pixels; mask: Uint8Array; owner: Int16Array; k: number; pending: { i: number; rect: Rect }[] }>();

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
    patches.push({ i, rect: rect.map((v) => v * st.k) as Rect, url: await encodePatch(st.px, rect, st.owner, i) });
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
  const paragraphs = describe(px, lines, k);
  const tLayout = performance.now();
  // erase the text that will be translated: diffusion now wherever the hole is only the glyphs, the inpainting model
  // later for whole boxes on photos
  const { mask, owner, thin: plain, box: pending } = planErase(px, paragraphs, k, targetLang, height);

  const patches: ErasePatch[] = [];
  for (const { i, rect, fontPx } of plain) {
    pushPull(px, mask, rect, Math.max(8, Math.round(1.5 * fontPx)));
    patches.push({ i, rect: rect.map((v) => v * k) as Rect, url: await encodePatch(px, rect, owner, i) });
  }
  if (pending.length) {
    erasing.set(job, { px, mask, owner, k, pending });
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
