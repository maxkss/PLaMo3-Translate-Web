/**
 * Offscreen document: the translation service. It owns the job queue and talks to content scripts /
 * extension pages over runtime ports (ENGINE_PORT); the model itself runs in a dedicated worker
 * (engine-worker.js) so this thread stays responsive.
 */

import type { TranslateInput, TranslateStats } from "./engine/translator.ts";
import type { WorkerIn, WorkerOut } from "./engine/worker.ts";
import type { OcrWorkerIn, OcrWorkerOut } from "./ocr/worker.ts";
import { deleteModel, downloadModel, isInstalled, listInstalled } from "./engine/storage.ts";
import {
  ENGINE_PORT,
  INPAINT_MODEL,
  MODELS,
  OCR_MODELS,
  modelBytes,
  modelSourceUrl,
  type ErasePatch,
  type OcrRequest,
  type OcrResult,
  type StoredModel,
  type ClientToEngine,
  type ModelId,
  type EngineStatus,
  type EngineToClient,
  type Settings,
  type TranslateRequest,
} from "./shared/common.ts";

let status: EngineStatus = { state: "idle" };
/** model the worker has loaded (or is loading) */
let activeModel: ModelId | null = null;
let downloadAbort: AbortController | null = null;
const ports = new Set<chrome.runtime.Port>();

function setStatus(patch: Partial<EngineStatus>): void {
  status = { ...status, ...patch };
  if (status.state !== "downloading" && status.state !== "loading") delete status.progress;
  broadcast({ type: "status", status: { ...status, queue: queue.length } });
}

function broadcast(msg: EngineToClient): void {
  for (const p of ports) safePost(p, msg);
}

function safePost(port: chrome.runtime.Port, msg: EngineToClient): void {
  try {
    port.postMessage(msg);
  } catch {
    ports.delete(port);
  }
}

/** Offscreen documents only get chrome.runtime, so settings are fetched via the service worker. */
function getSettings(): Promise<Settings> {
  return chrome.runtime.sendMessage({ type: "get-settings" });
}

async function selectedModel(): Promise<ModelId> {
  const { model } = await getSettings();
  return model in MODELS ? model : "0.8b";
}

/** What image translation needs besides the translation model: the selected OCR model and the inpainting model. */
async function extraModels(): Promise<StoredModel[]> {
  const { ocrModel } = await getSettings();
  return [OCR_MODELS[ocrModel] ?? OCR_MODELS.mobile, INPAINT_MODEL];
}

/**
 * Translation models that are ready to use. A model only counts as installed together with the image-translation
 * models: everything is downloaded in one go, and nothing is fetched later in the middle of a translation.
 */
async function installedModels(): Promise<ModelId[]> {
  for (const m of await extraModels()) if (!(await isInstalled(m))) return [];
  return listInstalled();
}

/** Recompute idle/missing and the installed list for the selected model (when nothing is in flight). */
async function refreshStatus(message?: string): Promise<void> {
  const [model, installed] = await Promise.all([selectedModel(), installedModels()]);
  const patch: Partial<EngineStatus> = { model, installed };
  if (!loaded && !loading && !downloadAbort) {
    patch.state = installed.includes(model) ? "idle" : "missing";
    patch.message = message;
    patch.backend = undefined;
  }
  setStatus(patch);
}

// ---------------------------------------------------------------- worker proxy

const worker = new Worker(chrome.runtime.getURL("engine-worker.js"), { type: "module" });
let loaded = false;
/** Bumped by every unload so a load that finishes after it is not mistaken for a live model. */
let loadGen = 0;
let loading: { promise: Promise<void>; resolve: () => void; reject: (e: Error) => void } | null = null;
let jobSeq = 0;
/** how many translations the engine decodes together (reported by the worker; 1 = one at a time) */
let engineBatch = 1;
const workerJobs = new Map<number, { resolve: (r: { text: string; stats: TranslateStats }) => void; reject: (e: Error) => void; onDelta?: (t: string) => void }>();

worker.onmessage = (ev: MessageEvent<WorkerOut>) => {
  const m = ev.data;
  switch (m.t) {
    case "loaded":
      if (m.gen !== loadGen) break;
      loaded = true;
      activeModel = m.model;
      engineBatch = m.batch;
      setStatus({ state: "ready", model: m.model, backend: m.backend, batch: m.batch, tokensPerSec: undefined, message: `준비됨 (${MODELS[m.model].label} · ${m.backend}, ${(m.ms / 1000).toFixed(1)}s)` });
      loading?.resolve();
      loading = null;
      void armIdle();
      break;
    case "load-error":
      if (m.gen !== loadGen) break;
      loaded = false;
      setStatus({ state: "error", message: m.message });
      loading?.reject(new Error(m.message));
      loading = null;
      break;
    case "unloaded":
      break;
    case "delta":
      workerJobs.get(m.job)?.onDelta?.(m.text);
      break;
    case "done":
      workerJobs.get(m.job)?.resolve({ text: m.text, stats: m.stats });
      workerJobs.delete(m.job);
      break;
    case "error": {
      const j = workerJobs.get(m.job);
      workerJobs.delete(m.job);
      j?.reject(m.aborted ? new DOMException("aborted", "AbortError") : new Error(m.message));
      break;
    }
  }
};
worker.onerror = (e) => {
  console.error("[plamo] worker error", e);
  setStatus({ state: "error", message: `worker: ${e.message}` });
};

const toWorker = (m: WorkerIn) => worker.postMessage(m);

async function ensureLoaded(): Promise<void> {
  if (loaded) return;
  if (loading) return loading.promise;
  let resolve!: () => void, reject!: (e: Error) => void;
  const promise = new Promise<void>((res, rej) => ((resolve = res), (reject = rej)));
  loading = { promise, resolve, reject };
  const gen = loadGen;
  const { backend, model, batchSize } = await getSettings();
  const info = MODELS[model] ?? MODELS["0.8b"];
  if (!(await installedModels()).includes(info.id)) {
    if (loading?.promise === promise) loading = null;
    setStatus({ state: "missing", model: info.id, message: `${info.label} 모델이 설치되지 않았습니다. 설정에서 내려받아 주세요.` });
    reject(new Error(`${info.label} 모델이 설치되지 않았습니다`));
    return promise;
  }
  if (gen !== loadGen) return promise; // unloaded meanwhile; promise is already rejected
  activeModel = info.id;
  setStatus({ state: "loading", model: info.id, progress: undefined, message: `${info.label} 모델 불러오는 중…` });
  toWorker({ t: "load", gen, model: info.id, backend, extBase: chrome.runtime.getURL(""), batch: batchSize });
  return promise;
}

async function unload(): Promise<void> {
  loaded = false;
  activeModel = null;
  loadGen++;
  loading?.reject(new DOMException("unloaded", "AbortError"));
  loading = null;
  toWorker({ t: "unload" });
  await refreshStatus();
}

function runInWorker(input: TranslateInput, stream: boolean, interactive: boolean, signal: AbortSignal, onDelta?: (t: string) => void) {
  const job = jobSeq++;
  return new Promise<{ text: string; stats: TranslateStats }>((resolve, reject) => {
    workerJobs.set(job, { resolve, reject, onDelta });
    signal.addEventListener("abort", () => toWorker({ t: "abort", job }), { once: true });
    toWorker({ t: "translate", job, input, stream, interactive });
  });
}

// ---------------------------------------------------------------- job queue

/** One requester waiting for a job's result. */
interface Sub {
  id: string;
  port: chrome.runtime.Port;
  stream: boolean;
  priority: number;
}

/**
 * One translation to compute. Identical requests (same model, languages, context and text) that arrive while it is
 * queued or running subscribe to it instead of being computed again.
 */
interface Job {
  key: string;
  input: TranslateInput;
  subs: Sub[];
  abort: AbortController;
  seq: number;
  /** aborted to make room for a more urgent job; goes back into the queue */
  preempted?: boolean;
  /** aborted because nobody waits for it any more; new requests start a job of their own instead of joining it */
  dropped?: boolean;
  /** latest partial text, sent to subscribers that join mid-way */
  partial?: string;
  /** sent to the worker as interactive (decoded ahead of background jobs) */
  interactive?: boolean;
}

const queue: Job[] = [];
/** jobs handed to the worker */
const running = new Set<Job>();
let pumping = false;
const prio = (j: Job) => Math.min(...j.subs.map((s) => s.priority));
/** selection / options-page requests (priority 0); page blocks use 5 */
const isInteractive = (j: Job) => prio(j) <= 0;
const sortQueue = () => queue.sort((a, b) => prio(a) - prio(b) || a.seq - b.seq);
let seq = 0;
const cache = new Map<string, string>();
const cacheKey = (r: TranslateInput) => `${activeModel}\u0001${r.targetLang}\u0001${r.sourceLang ?? ""}\u0001${r.context ?? ""}\u0001${r.text}`;

/**
 * With batching, the worker decodes several background jobs together (one more is sent than fits, so a freed row is
 * refilled without a round trip) and an interactive job decodes alone while they wait, so it starts at once.
 * Without batching only one job runs at a time.
 */
function canStart(j: Job): boolean {
  if (engineBatch <= 1) return running.size === 0;
  if (isInteractive(j)) return true;
  let background = 0;
  for (const r of running) if (!r.interactive) background++;
  return background < engineBatch + 1;
}

function preemptIfNeeded(): void {
  // without batching, an interactive request (e.g. a selection) must not wait for a background paragraph to finish:
  // stop the running job at its next token and put it back in the queue
  if (engineBatch > 1) return;
  const head = queue[0];
  for (const r of running) {
    if (head && !r.preempted && prio(head) < prio(r)) {
      r.preempted = true;
      r.abort.abort();
    }
  }
}

function enqueue(req: TranslateRequest, port: chrome.runtime.Port): void {
  clearIdle();
  const key = cacheKey(req);
  const hit = cache.get(key);
  if (hit !== undefined) {
    safePost(port, { type: "done", id: req.id, text: hit, cached: true });
    return;
  }
  const sub: Sub = { id: req.id, port, stream: !!req.stream, priority: req.priority ?? 10 };
  const same = [...running].find((j) => j.key === key && !j.preempted && !j.dropped) ?? queue.find((j) => j.key === key);
  if (same) {
    same.subs.push(sub);
    if (sub.stream && same.partial) safePost(port, { type: "delta", id: sub.id, text: same.partial });
  } else {
    const { text, targetLang, sourceLang, context } = req;
    queue.push({ key, input: { text, targetLang, sourceLang, context }, subs: [sub], abort: new AbortController(), seq: seq++ });
  }
  sortQueue();
  preemptIfNeeded();
  void pump();
}

function cancel(port: chrome.runtime.Port, ids?: string[], all?: boolean): void {
  const match = (s: Sub) => s.port === port && (all || !!ids?.includes(s.id));
  for (let i = queue.length - 1; i >= 0; i--) {
    queue[i].subs = queue[i].subs.filter((s) => !match(s));
    if (!queue[i].subs.length) queue.splice(i, 1);
  }
  for (const r of running) {
    r.subs = r.subs.filter((s) => !match(s));
    if (!r.subs.length && !r.dropped) {
      r.dropped = true;
      r.abort.abort();
    }
  }
}

async function pump(): Promise<void> {
  if (pumping) return; // one loop at a time (ensureLoaded awaits)
  pumping = true;
  try {
    while (queue.length) {
      try {
        await ensureLoaded();
      } catch (e) {
        for (const j of queue.splice(0)) for (const s of j.subs) safePost(s.port, { type: "error", id: s.id, message: status.message ?? String(e) });
        return;
      }
      if (!queue.length || !canStart(queue[0])) {
        preemptIfNeeded();
        break;
      }
      const job = queue.shift()!;
      job.subs = job.subs.filter((s) => ports.has(s.port));
      if (job.subs.length) void runJob(job);
    }
  } finally {
    pumping = false;
  }
}

async function runJob(job: Job): Promise<void> {
  running.add(job);
  job.interactive = isInteractive(job);
  try {
    const { text, stats } = await runInWorker(job.input, true, job.interactive, job.abort.signal, (partial) => {
      job.partial = partial;
      for (const s of job.subs) if (s.stream) safePost(s.port, { type: "delta", id: s.id, text: partial });
    });
    if (cache.size > 5000) cache.delete(cache.keys().next().value!);
    cache.set(job.key, text);
    if (stats.outputTokens > 4 && stats.decodeMs > 0) {
      status.tokensPerSec = Math.round((stats.outputTokens / stats.decodeMs) * 10000) / 10;
      // broadcast first so clients read this job's speed when "done" arrives
      if (queue.length === 0) setStatus({});
    }
    for (const s of job.subs) safePost(s.port, { type: "done", id: s.id, text });
  } catch (e) {
    const live = job.subs.filter((s) => ports.has(s.port));
    const aborted = (e as Error)?.name === "AbortError";
    if (aborted && live.length) {
      // preempted, or someone still waits for it: restart later from the beginning (its partial output is discarded).
      // An aborted job must not end with subscribers left waiting: they would never hear back.
      queue.push({ ...job, subs: live, abort: new AbortController(), preempted: false, dropped: false, partial: undefined });
      sortQueue();
    } else if (!aborted) {
      for (const s of job.subs) safePost(s.port, { type: "error", id: s.id, message: String((e as Error)?.message ?? e) });
    }
  } finally {
    running.delete(job);
  }
  if (queue.length === 0 || queue.length % 8 === 0) setStatus({});
  void pump();
  void armIdle();
}

// ---------------------------------------------------------------- image OCR

let ocrWorker: Worker | null = null;
let ocrSeq = 0;
const ocrJobs = new Map<number, { resolve: (r: OcrResult) => void; reject: (e: Error) => void }>();
/** inpainting passes waiting for their patches */
const patchJobs = new Map<number, (p: ErasePatch[]) => void>();
/** recent results (with every patch) by image URL and target language, so toggling an image back on is instant */
const ocrCache = new Map<string, OcrResult>();

function getOcrWorker(): Worker {
  if (ocrWorker) return ocrWorker;
  const w = new Worker(chrome.runtime.getURL("ocr-worker.js"), { type: "module" });
  w.onmessage = (ev: MessageEvent<OcrWorkerOut>) => {
    const m = ev.data;
    if (m.t === "patches") {
      patchJobs.get(m.job)?.(m.patches);
      patchJobs.delete(m.job);
      return;
    }
    const j = ocrJobs.get(m.job);
    ocrJobs.delete(m.job);
    if (m.t === "done") j?.resolve(m.result);
    else j?.reject(new Error(m.message));
  };
  w.onerror = (e) => {
    console.error("[plamo] ocr worker error", e);
    for (const [, j] of ocrJobs) j.reject(new Error(`ocr worker: ${e.message}`));
    ocrJobs.clear();
    for (const [, done] of patchJobs) done([]);
    patchJobs.clear();
    ocrWorker?.terminate();
    ocrWorker = null;
  };
  return (ocrWorker = w);
}

async function readImage(req: OcrRequest): Promise<ImageBitmap> {
  const url = req.dataUrl ?? req.src;
  if (!url) throw new Error("FETCH_FAILED: 이미지 주소가 없습니다");
  let blob: Blob;
  try {
    // the extension's host permissions let it read cross-origin images; cookies are sent for images behind a login
    const res = await fetch(url, { credentials: "include" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    blob = await res.blob();
  } catch (e) {
    throw new Error(`FETCH_FAILED: ${(e as Error)?.message ?? e}`);
  }
  try {
    return await createImageBitmap(blob);
  } catch (e) {
    throw new Error(`FETCH_FAILED: 이미지를 읽지 못했습니다 (${blob.type || "알 수 없는 형식"})`);
  }
}

async function handleOcr(req: OcrRequest, port: chrome.runtime.Port): Promise<void> {
  clearIdle();
  ocrActive++;
  try {
    await runOcr(req, port);
  } finally {
    ocrActive--;
    void armIdle();
  }
}

async function runOcr(req: OcrRequest, port: chrome.runtime.Port): Promise<void> {
  const ocrModel = req.ocrModel ?? (await getSettings()).ocrModel ?? "mobile";
  const key = req.src && !req.dataUrl ? `${req.src}\u0001${req.targetLang}\u0001${ocrModel}` : null;
  const hit = key ? ocrCache.get(key) : undefined;
  if (hit) return safePost(port, { type: "ocr-done", id: req.id, result: hit });
  try {
    const ocr = OCR_MODELS[ocrModel];
    if (!(await isInstalled(ocr)) || !(await isInstalled(INPAINT_MODEL))) throw new Error("모델이 설치되지 않았습니다. 설정에서 내려받아 주세요.");
    const image = await readImage(req);
    const job = ocrSeq++;
    const result = await new Promise<OcrResult>((resolve, reject) => {
      ocrJobs.set(job, { resolve, reject });
      getOcrWorker().postMessage({ t: "ocr", job, image, extBase: chrome.runtime.getURL(""), targetLang: req.targetLang, model: ocrModel } satisfies OcrWorkerIn, [image]);
    });
    safePost(port, { type: "ocr-done", id: req.id, result });
    // text on photos: erased by the inpainting model (downloaded the first time it is needed)
    let more: ErasePatch[] = [];
    if (result.pending.length) {
      try {
        more = await new Promise<ErasePatch[]>((resolve) => {
          patchJobs.set(job, resolve);
          getOcrWorker().postMessage({ t: "inpaint", job } satisfies OcrWorkerIn);
        });
      } catch (e) {
        console.warn("[plamo] inpainting unavailable", e);
      }
      safePost(port, { type: "ocr-patches", id: req.id, patches: more });
    }
    if (key) {
      if (ocrCache.size >= 30) ocrCache.delete(ocrCache.keys().next().value!);
      ocrCache.set(key, { ...result, patches: [...result.patches, ...more], pending: [] });
    }
  } catch (e) {
    safePost(port, { type: "error", id: req.id, message: String((e as Error)?.message ?? e) });
  }
}

// ---------------------------------------------------------------- unload when idle

let idleTimer: ReturnType<typeof setTimeout> | null = null;
let ocrActive = 0;
const busy = () => running.size > 0 || queue.length > 0 || ocrActive > 0 || !!loading;

function clearIdle(): void {
  if (idleTimer) clearTimeout(idleTimer);
  idleTimer = null;
}

/** (Re)start the countdown after which an idle engine frees its models (Settings.idleUnloadMin; 0 = never). */
async function armIdle(): Promise<void> {
  clearIdle();
  if (busy() || (!loaded && !ocrWorker)) return;
  const { idleUnloadMin } = await getSettings();
  if (!idleUnloadMin || busy()) return;
  clearIdle();
  idleTimer = setTimeout(() => {
    idleTimer = null;
    if (busy()) return;
    if (loaded) void unload();
    ocrWorker?.terminate();
    ocrWorker = null;
  }, idleUnloadMin * 60_000);
}

// ---------------------------------------------------------------- ports

async function startDownload(id: ModelId): Promise<void> {
  if (downloadAbort) return;
  const info = MODELS[id];
  const { modelUrlOverride } = await getSettings();
  // only the model being replaced has to leave memory
  if (activeModel === id || loading) await unload();
  downloadAbort = new AbortController();
  setStatus({ state: "downloading", model: id, progress: 0, message: `${info.label} 모델 내려받는 중…` });
  try {
    // the translation model, then the image-translation models (OCR, inpainting): one download, one progress bar
    const parts: { m: StoredModel; base: string }[] = [{ m: info, base: modelSourceUrl(info, modelUrlOverride) }, ...(await extraModels()).map((m) => ({ m, base: "" }))];
    const total = parts.reduce((a, p) => a + modelBytes(p.m), 0);
    let before = 0, last = 0;
    for (const { m, base } of parts) {
      await downloadModel(
        m,
        base,
        (loadedBytes) => {
          const now = performance.now();
          if (now - last < 250) return;
          last = now;
          const got = before + loadedBytes;
          setStatus({ progress: got / total, message: `${info.label} 모델 내려받는 중… ${(got / 1e9).toFixed(2)} / ${(total / 1e9).toFixed(2)} GB` });
        },
        downloadAbort.signal,
      );
      before += modelBytes(m);
    }
    downloadAbort = null;
    await refreshStatus(`${info.label} 다운로드 완료`);
    if ((await selectedModel()) === id) ensureLoaded().catch(() => {});
  } catch (e) {
    downloadAbort = null;
    const aborted = (e as Error)?.name === "AbortError";
    setStatus({ state: "error", message: aborted ? "다운로드를 취소했습니다 (다시 누르면 이어받습니다)" : `다운로드 실패: ${(e as Error)?.message ?? e} (다시 누르면 이어받습니다)` });
    void installedModels().then((installed) => setStatus({ installed }));
  } finally {
    downloadAbort = null;
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name !== ENGINE_PORT) return;
  ports.add(port);
  safePost(port, { type: "status", status: { ...status, queue: queue.length } });
  port.onDisconnect.addListener(() => {
    ports.delete(port);
    cancel(port, undefined, true);
  });
  port.onMessage.addListener(async (msg: ClientToEngine) => {
    switch (msg.type) {
      case "translate":
        enqueue(msg, port);
        break;
      case "ocr":
        void handleOcr(msg, port);
        break;
      case "cancel":
        cancel(port, msg.ids, msg.all);
        break;
      case "status":
        safePost(port, { type: "status", status: { ...status, queue: queue.length } });
        break;
      case "load":
        ensureLoaded().catch(() => {});
        break;
      case "unload":
        await unload();
        break;
      case "idle-changed":
        void armIdle();
        break;
      case "delete-model":
        if (activeModel === msg.model || loading) await unload();
        await deleteModel(MODELS[msg.model]);
        await refreshStatus(`${MODELS[msg.model].label} 모델을 삭제했습니다.`);
        break;
      case "download":
        void startDownload(msg.model);
        break;
      case "cancel-download":
        downloadAbort?.abort();
        break;
      case "select-model":
        // settings were already saved by the page; drop the old model so the next job loads the new one
        // (also sent when the OCR model choice changes: the installed list depends on it)
        if (activeModel !== msg.model && (activeModel || loading)) await unload();
        else await refreshStatus();
        break;
    }
  });
});

void refreshStatus();
