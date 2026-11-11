/**
 * Dedicated worker that owns the ONNX Runtime session. Running inference here keeps the offscreen
 * document's main thread (shared with the popup/options pages' renderer) responsive.
 */

import * as ort from "onnxruntime-web/webgpu";
import { Generator } from "./generator.ts";
import { getModelFiles } from "./storage.ts";
import { MODELS, type ModelId } from "../shared/common.ts";
import { PlamoTokenizer } from "./tokenizer.ts";
import { Translator, type TranslateInput, type TranslateStats } from "./translator.ts";

export type WorkerIn =
  | { t: "load"; gen: number; model: ModelId; backend: "auto" | "webgpu" | "wasm"; extBase: string; batch: number }
  | { t: "unload" }
  | { t: "translate"; job: number; input: TranslateInput; stream: boolean; interactive: boolean }
  | { t: "abort"; job: number };

export type WorkerOut =
  /** batch: how many translations the engine decodes together (1 = one at a time) */
  | { t: "loaded"; gen: number; model: ModelId; backend: string; ms: number; batch: number }
  | { t: "load-error"; gen: number; message: string }
  | { t: "unloaded" }
  | { t: "delta"; job: number; text: string }
  | { t: "done"; job: number; text: string; stats: TranslateStats }
  | { t: "error"; job: number; message: string; aborted?: boolean };

const post = (m: WorkerOut) => (self as unknown as Worker).postMessage(m);

let session: ort.InferenceSession | null = null;
let generator: Generator | null = null;
let translator: Translator | null = null;
let currentBackend = "";
let currentModel: ModelId | null = null;
/** Settings.batchSize the model was loaded with */
let currentBatch = 0;
const aborts = new Map<number, AbortController>();
/** load/unload run in order, each after the translations already started have settled */
let chain: Promise<unknown> = Promise.resolve();
/** translations run concurrently (the generator batches them) */
const inflight = new Set<Promise<unknown>>();
/** waits for the translations started before this call (not later ones: they wait for the chain) */
const settled = () => {
  const before = [...inflight];
  return () => Promise.allSettled(before);
};

/**
 * WebGPU: every output stays on the GPU (static KV buffers, logits for the GPU argmax) and decode steps replay a
 * captured graph. `capture: false` is the fallback if capture cannot be set up.
 */
async function createSession(model: File, data: File, backend: "webgpu" | "wasm", capture: boolean): Promise<ort.InferenceSession> {
  const dataUrl = URL.createObjectURL(data);
  try {
    const opts: ort.InferenceSession.SessionOptions = {
      // the EP must be an object: onnxruntime-web only forwards enableGraphCapture to the EP that way
      executionProviders: [backend === "webgpu" ? { name: "webgpu" } : "wasm"],
      graphOptimizationLevel: "all",
      externalData: [{ path: data.name, data: dataUrl }],
    };
    if (backend === "webgpu") {
      opts.preferredOutputLocation = "gpu-buffer";
      opts.enableGraphCapture = capture;
    }
    return await ort.InferenceSession.create(new Uint8Array(await model.arrayBuffer()), opts);
  } finally {
    URL.revokeObjectURL(dataUrl);
  }
}

async function release(): Promise<void> {
  translator = null;
  currentModel = null;
  generator?.dispose();
  generator = null;
  await session?.release().catch(() => {});
  session = null;
}

async function load(gen: number, id: ModelId, pref: "auto" | "webgpu" | "wasm", extBase: string, batch: number): Promise<void> {
  if (translator && currentModel === id && currentBatch === batch) {
    post({ t: "loaded", gen, model: id, backend: currentBackend, ms: 0, batch: generator?.maxBatch ?? 1 });
    return;
  }
  await release();
  const info = MODELS[id];
  ort.env.wasm.wasmPaths = extBase + "ort/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu;
  const hasGpu = !!gpu && !!(await gpu.requestAdapter().catch(() => null));
  const order: ("webgpu" | "wasm")[] = pref === "wasm" || !hasGpu ? ["wasm"] : pref === "webgpu" ? ["webgpu"] : ["webgpu", "wasm"];
  const [asset, files] = await Promise.all([fetch(extBase + "tokenizer/plamo3.json").then((r) => r.json()), getModelFiles(info)]);
  const tok = new PlamoTokenizer(asset);
  let lastErr: unknown;
  // WebGPU with graph capture first, then WebGPU without capture, then wasm
  type Attempt = { b: "webgpu" | "wasm"; capture: boolean };
  const attempts = order.flatMap((b): Attempt[] => (b === "webgpu" ? [{ b, capture: true }, { b, capture: false }] : [{ b, capture: false }]));
  for (const { b: backend, capture } of attempts) {
    try {
      const t0 = performance.now();
      session = await createSession(files.model, files.data, backend, capture);
      generator = new Generator(session, ort, {
        numKvHeads: info.numKvHeads,
        headDim: info.headDim,
        gpu:
          backend === "webgpu"
            ? { device: (await ort.env.webgpu.device) as GPUDevice, capture, maxBatch: Math.max(1, Math.min(16, Math.round(batch) || 1)) }
            : undefined,
      });
      translator = new Translator(tok, generator);
      // warm-up: compiles the GPU pipelines (and captures the decode graph) so the first real request is fast;
      // then the batched shapes (texts of different lengths, so the batch also shrinks through its sizes)
      await translator.translate({ text: "Hello.", targetLang: "ko" });
      if (generator.maxBatch > 1) {
        const texts = ["Thank you.", "Good morning.", "See you.", "Welcome back.", "Read more", "Sign in to continue.", "Latest news", "Contact us"];
        await Promise.all(
          Array.from({ length: generator.maxBatch }, (_, i) => (i < texts.length ? texts[i] : `${texts[i % texts.length]} (${i + 1})`)).map((text) =>
            translator!.translate({ text, targetLang: "ko" }),
          ),
        );
      }
      currentBackend = backend === "webgpu" && !capture ? "webgpu (no capture)" : backend;
      currentModel = id;
      currentBatch = batch;
      post({ t: "loaded", gen, model: id, backend: currentBackend, ms: performance.now() - t0, batch: generator.maxBatch });
      return;
    } catch (e) {
      console.warn(`[plamo] ${backend}${capture ? " + graph capture" : ""} failed`, e);
      lastErr = e;
      await release();
    }
  }
  throw lastErr;
}

self.onmessage = (ev: MessageEvent<WorkerIn>) => {
  const m = ev.data;
  switch (m.t) {
    case "load":
      chain = chain.then(settled()).then(() =>
        load(m.gen, m.model, m.backend, m.extBase, m.batch).catch((e) => post({ t: "load-error", gen: m.gen, message: String((e as Error)?.message ?? e) })),
      );
      break;
    case "unload":
      chain = chain.then(settled()).then(async () => {
        await release();
        post({ t: "unloaded" });
      });
      break;
    case "abort":
      aborts.get(m.job)?.abort();
      break;
    case "translate": {
      const ac = new AbortController();
      aborts.set(m.job, ac);
      const p = chain.then(async () => {
        try {
          if (!translator) throw new Error("model not loaded");
          if (ac.signal.aborted) throw new DOMException("aborted", "AbortError");
          // first visible text goes out at once; later updates at most every 40 ms, with a trailing flush
          let last = 0;
          let sent = "";
          let pending: string | null = null;
          let timer: ReturnType<typeof setTimeout> | null = null;
          const flush = () => {
            timer = null;
            if (pending !== null && pending !== sent) {
              sent = pending;
              last = performance.now();
              post({ t: "delta", job: m.job, text: sent });
            }
            pending = null;
          };
          const { text, stats } = await translator.translate(
            m.input,
            m.stream
              ? (partial) => {
                  if (partial === sent || partial === pending) return;
                  pending = partial;
                  const wait = 40 - (performance.now() - last);
                  if (!sent.trim() || wait <= 0) flush();
                  else timer ??= setTimeout(flush, wait);
                }
              : undefined,
            ac.signal,
            m.interactive,
          );
          if (timer) clearTimeout(timer);
          post({ t: "done", job: m.job, text, stats });
        } catch (e) {
          const aborted = (e as Error)?.name === "AbortError";
          if (!aborted) console.error("[plamo] translate failed", e);
          post({ t: "error", job: m.job, message: String((e as Error)?.message ?? e), aborted });
        } finally {
          aborts.delete(m.job);
        }
      });
      inflight.add(p);
      void p.finally(() => inflight.delete(p));
      break;
    }
  }
};
