/**
 * Greedy decoder for the exported PLaMo 3 graph (tools/convert/build_fused.py).
 *
 * Two paths:
 *  - WebGPU (`opts.gpu`): requires the static-KV graph (inputs seqlens_k / total_sequence_length). The KV cache
 *    lives in fixed-capacity GPU buffers that are both the past inputs and the present outputs, logits stay on the
 *    GPU and the next token is picked by a small WGSL argmax (a few bytes are read back). Decode steps of a single
 *    sequence replay a captured WebGPU graph when the session was created with graph capture; prefills run uncaptured.
 *    KV of the previous prompt stays valid, so a prompt that starts with the same tokens (the fixed template prefix,
 *    or the same context) only prefills the new part.
 *
 *    Concurrent generate() calls are batched (continuous batching, up to `maxBatch` sequences) when the graph has a
 *    symbolic batch dimension: each sequence is prefilled alone in the single-sequence buffers, its KV is copied into
 *    a row of the batch buffers, and all rows then decode together with per-row lengths (seqlens_k). Rows that finish
 *    are refilled from the queue, a few at a time (1, 2, 4…) so the first ones start at once. With up to 3 rows the
 *    tokens are the same as decoding each sequence alone; wider batches (Settings.batchSize) change a word here and there.
 *    `interactive` sequences jump the queue and decode alone while the batch waits.
 *    Decode steps are pipelined: the argmax writes the next input token on the GPU, and the next step is queued
 *    before the previous one's tokens are read back.
 *  - Otherwise (wasm, Node tests, or a dynamic-KV graph): the cache grows by concatenation and is fed back at its
 *    exact length; logits are read on the CPU. Calls run one at a time.
 *
 * All paths produce the same tokens (checked by tools/convert/verify_variants.py and extension/src/dev/bench-ab.ts,
 * bench-engine.ts).
 */

import type { InferenceSession, Tensor, TypedTensor } from "onnxruntime-common";
import { EXACT_BATCH, KV_CAPACITY } from "../shared/common.ts";

export interface OrtLike {
  Tensor: typeof Tensor;
}

export interface GenerateOptions {
  maxNewTokens: number;
  eosIds: number[];
  onToken?: (id: number) => void;
  signal?: AbortSignal;
  /** served before background sequences (and, on WebGPU, decoded alone while batched rows wait) */
  interactive?: boolean;
}

export interface GenerateResult {
  ids: number[];
  promptTokens: number;
  /** prompt tokens whose KV was reused from the previous prompt */
  reusedTokens: number;
  prefillMs: number;
  /** decode time charged to this sequence: a batched step of b rows counts 1/b of its time */
  decodeMs: number;
  stoppedBy: "eos" | "length" | "repetition" | "abort";
}

/** Max total sequence (prompt + output). Must stay within the 2048-token sliding window. */
export const MAX_CONTEXT = 2040;
/**
 * Default batch: the largest that gives exactly the same tokens as decoding alone (measured; larger batches pick
 * other kernels). Wider batches (Settings.batchSize) are much faster and change a word here and there.
 */
export const MAX_BATCH = EXACT_BATCH;
/** Decode steps kept in flight: the next step is queued while the previous step's tokens are read back. */
const PIPELINE_DEPTH = 2;

export interface GpuOptions {
  device: GPUDevice;
  /** the session was created with enableGraphCapture */
  capture: boolean;
  /** KV buffer capacity in tokens (default KV_CAPACITY) */
  capacity?: number;
  /** most sequences decoded together (default MAX_BATCH; 1 disables batching). Needs a graph with a symbolic batch dim. */
  maxBatch?: number;
}

export interface GeneratorOptions {
  numKvHeads?: number;
  headDim?: number;
  kvType?: "float16" | "float32";
  /** dynamic path only: keep present.* on the GPU (preferredOutputLocation gpu-buffer) */
  gpuKv?: boolean;
  gpu?: GpuOptions;
}

type Feeds = Record<string, Tensor>;

// WebGPU flag constants (the TS DOM lib has the interfaces but not these namespaces)
const BUF = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 } as const;
const MAP_READ_MODE = 0x1;
const VOCAB = 107520;

/** Buffers created here and handed to onnxruntime are flagged so its patched glue keeps them registered (build.mjs). */
const KEEP = "__plamoKeep";
const REBIND = "__plamoRebind";

/** One generate() call on the WebGPU path. */
interface Seq {
  prompt: number[];
  maxNew: number;
  eos: Set<number>;
  o: GenerateOptions;
  out: number[];
  /** tokens whose KV is written (prompt, then each fed output token) */
  tokens: number[];
  /** last sampled token, fed in the next step */
  next: number;
  reuse: number;
  prefillMs: number;
  decodeMs: number;
  resolve: (r: GenerateResult) => void;
  reject: (e: unknown) => void;
}

export class Generator {
  private readonly session: InferenceSession;
  private readonly ort: OrtLike;
  private readonly pastNames: string[];
  private readonly presentNames: string[];
  private readonly kvDims: [number, number];
  private readonly kvType: "float16" | "float32";
  private readonly gpuKv: boolean;
  private readonly isStatic: boolean;
  private readonly gpu: GpuState | null;
  private readonly gpuOpts: GpuOptions | null;
  /** most sequences decoded together */
  readonly maxBatch: number;
  private batch: BatchState | null = null;

  // scheduler state (WebGPU path)
  private readonly waiting: Seq[] = [];
  /** the one sequence decoding in the single-sequence buffers (only while no rows are batched) */
  private solo: Seq | null = null;
  /** sequences in the batch buffers; index = row */
  private rows: Seq[] = [];
  private driving = false;
  /** which tensors onnxruntime has bound (a different set must be re-bound before running) */
  private bound = "";

  // dynamic path: one call at a time, interactive first
  private dynBusy = false;
  private readonly dynWait: { interactive: boolean; go: () => void }[] = [];

  constructor(session: InferenceSession, ort: OrtLike, opts: GeneratorOptions = {}) {
    this.session = session;
    this.ort = ort;
    this.pastNames = session.inputNames.filter((n) => n.startsWith("past_key_values."));
    this.presentNames = this.pastNames.map((n) => n.replace("past_key_values.", "present."));
    this.kvDims = [opts.numKvHeads ?? 4, opts.headDim ?? 128];
    this.kvType = opts.kvType ?? "float16";
    this.gpuKv = !!opts.gpuKv || !!opts.gpu;
    this.isStatic = session.inputNames.includes("seqlens_k");
    this.gpuOpts = opts.gpu && this.isStatic ? opts.gpu : null;
    this.gpu = this.gpuOpts ? new GpuState(ort, this.gpuOpts, this.pastNames, this.kvDims) : null;
    const ids = (session as { inputMetadata?: readonly InferenceSession.ValueMetadata[] }).inputMetadata?.find((m) => m.name === "input_ids");
    const symbolicBatch = !!ids && ids.isTensor && typeof ids.shape[0] === "string";
    this.maxBatch = this.gpu && symbolicBatch ? Math.max(1, this.gpuOpts!.maxBatch ?? MAX_BATCH) : 1;
  }

  get usesGpuKv(): boolean {
    return this.gpuKv;
  }

  /** Forget which tokens the KV buffers hold (the next prompt is prefilled in full). */
  resetCache(): void {
    if (this.gpu) this.gpu.cached.length = 0;
  }

  /** Free GPU buffers owned by the generator (call before releasing the session). */
  dispose(): void {
    const err = new DOMException("generator disposed", "AbortError");
    for (const s of this.live()) s.reject(err);
    this.waiting.length = 0;
    this.rows = [];
    this.solo = null;
    this.gpu?.dispose();
    this.batch?.dispose();
    this.batch = null;
  }

  async generate(promptIds: number[], o: GenerateOptions): Promise<GenerateResult> {
    if (promptIds.length >= MAX_CONTEXT) throw new Error(`prompt too long (${promptIds.length} tokens)`);
    const maxNew = Math.max(1, Math.min(o.maxNewTokens, MAX_CONTEXT - promptIds.length));
    if (!this.gpu) return this.generateDynamicLocked(promptIds, maxNew, o);
    return new Promise<GenerateResult>((resolve, reject) => {
      this.waiting.push({
        prompt: promptIds,
        maxNew,
        eos: new Set(o.eosIds),
        o,
        out: [],
        tokens: [],
        next: -1,
        reuse: 0,
        prefillMs: 0,
        decodeMs: 0,
        resolve,
        reject,
      });
      void this.drive();
    });
  }

  // ------------------------------------------------------------------ WebGPU scheduler

  private live(): Seq[] {
    return [...this.waiting, ...(this.solo ? [this.solo] : []), ...this.rows];
  }

  private active(): number {
    return (this.solo ? 1 : 0) + this.rows.length;
  }

  private async drive(): Promise<void> {
    if (this.driving) return;
    this.driving = true;
    try {
      while (this.waiting.length || this.solo || this.rows.length) await this.tick();
    } catch (e) {
      // a GPU/session failure: fail everything in flight and start over with empty caches
      for (const s of this.live()) s.reject(e);
      this.waiting.length = 0;
      this.rows = [];
      this.solo = null;
      this.gpu!.cached.length = 0;
      this.bound = "";
    } finally {
      this.driving = false;
    }
  }

  private finish(s: Seq, stoppedBy: GenerateResult["stoppedBy"]): void {
    s.resolve({ ids: s.out, promptTokens: s.prompt.length, reusedTokens: s.reuse, prefillMs: s.prefillMs, decodeMs: s.decodeMs, stoppedBy });
  }

  /** Take a sampled token. Returns true when the sequence is finished (and resolved). */
  private accept(s: Seq, tok: number): boolean {
    if (s.eos.has(tok)) {
      this.finish(s, "eos");
      return true;
    }
    s.out.push(tok);
    s.o.onToken?.(tok);
    const cut = repetitionCut(s.out);
    if (cut > 0) {
      s.out.length = cut;
      this.finish(s, "repetition");
      return true;
    }
    if (s.out.length >= s.maxNew) {
      this.finish(s, "length");
      return true;
    }
    s.next = tok;
    return false;
  }

  private dropAborted(): void {
    for (let i = this.waiting.length - 1; i >= 0; i--) {
      const s = this.waiting[i];
      if (s.o.signal?.aborted) {
        this.waiting.splice(i, 1);
        this.finish(s, "abort");
      }
    }
    if (this.solo?.o.signal?.aborted) {
      this.finish(this.solo, "abort");
      this.solo = null;
    }
    for (let r = this.rows.length - 1; r >= 0; r--) if (this.rows[r].o.signal?.aborted) this.removeRow(r, "abort");
  }

  private removeRow(r: number, stoppedBy: GenerateResult["stoppedBy"] | null): void {
    const s = this.rows[r];
    if (stoppedBy) this.finish(s, stoppedBy);
    const last = this.rows.length - 1;
    if (r !== last) {
      // keep rows packed at the front: the last row moves into the gap
      this.batch!.moveRow(last, r);
      this.rows[r] = this.rows[last];
    }
    this.rows.pop();
  }

  private ensureBatch(): BatchState {
    this.batch ??= new BatchState(this.ort, this.gpuOpts!, this.pastNames, this.kvDims, this.maxBatch);
    return this.batch;
  }

  /** The solo sequence joins the batch (as row 0) so another sequence can be prefilled. */
  private soloToRow(): void {
    const s = this.solo!;
    this.ensureBatch().copyIn(this.gpu!, this.rows.length);
    this.rows.push(s);
    this.solo = null;
  }

  /** The last batched sequence goes back to the single-sequence buffers (captured decode). */
  private rowToSolo(): void {
    const s = this.rows[0];
    this.batch!.copyOut(0, this.gpu!);
    this.gpu!.cached.length = 0;
    this.gpu!.cached.push(...s.tokens);
    this.rows = [];
    this.solo = s;
  }

  private async run<T>(key: string, fn: (rebind: boolean) => Promise<T>): Promise<T> {
    const rebind = key === "prefill" || key !== this.bound;
    this.bound = key;
    return fn(rebind);
  }

  /** Prefill a prompt in the single-sequence buffers (reusing the KV of a common prefix) and sample its first token. */
  private async prefill(s: Seq): Promise<number> {
    const g = this.gpu!;
    let reuse = 0;
    const lim = Math.min(g.cached.length, s.prompt.length - 1);
    while (reuse < lim && g.cached[reuse] === s.prompt[reuse]) reuse++;
    g.cached.length = reuse;
    s.reuse = reuse;
    const input = s.prompt.slice(reuse);
    const t0 = performance.now();
    const [tok] = await this.run("prefill", (rb) => g.step(this.session, input, reuse, true, rb));
    g.cached.push(...input);
    s.tokens = s.prompt.slice();
    s.prefillMs = performance.now() - t0;
    return tok;
  }

  /**
   * Decode `rows` (the solo sequence in the single-sequence buffers, or the batch rows) until one of them finishes or
   * is aborted, or `more()` says the scheduler has something else to do. Up to PIPELINE_DEPTH steps are in flight:
   * each step's argmax writes the next input token on the GPU, so a step is queued before the previous one's tokens
   * are read back (the readback alone takes a few ms). When a row finishes, the steps already queued are still read
   * (the other rows' tokens are valid; the finished row's extra KV entries are never used). Returns the rows that
   * finished. The rows do not change while steps are in flight.
   */
  private async burst(rows: Seq[], solo: boolean, more: () => boolean): Promise<Set<Seq>> {
    const g = this.gpu!;
    const b = rows.length;
    const key = solo ? "solo" : `batch${b}`;
    const pending: Promise<number[]>[] = [];
    const finished = new Set<Seq>();
    let stop = false;
    let first = true;
    let last = performance.now();
    try {
      for (;;) {
        if (!stop && pending.length < PIPELINE_DEPTH && (first || !more())) {
          // the first step feeds the known tokens; later ones read what the previous step's argmax wrote
          const feed = first ? rows.map((s) => s.next) : null;
          const k = pending.length;
          const { tokens } = await this.run(key, (rb) =>
            solo
              ? g.submit(this.session, feed, rows[0].tokens.length + k, false, rb)
              : this.batch!.submit(this.session, feed, rows.map((s) => s.tokens.length + k), rb),
          );
          pending.push(tokens);
          first = false;
          if (pending.length < PIPELINE_DEPTH) continue;
        }
        if (!pending.length) return finished;
        const toks = await pending.shift()!;
        const now = performance.now();
        const share = (now - last) / b;
        last = now;
        rows.forEach((s, r) => {
          if (finished.has(s)) return;
          s.tokens.push(s.next);
          if (solo) g.cached.push(s.next);
          s.decodeMs += share;
          if (this.accept(s, toks[r])) finished.add(s);
        });
        if (finished.size || rows.some((s) => s.o.signal?.aborted)) stop = true;
      }
    } catch (e) {
      for (const p of pending) p.catch(() => {});
      throw e;
    }
  }

  /** Prefill and decode an interactive sequence to the end, alone (batched rows keep their KV and wait). */
  private async runAlone(s: Seq): Promise<void> {
    if (this.accept(s, await this.prefill(s))) return;
    const finished = await this.burst([s], true, () => false);
    if (!finished.has(s)) this.finish(s, "abort");
  }

  private async tick(): Promise<void> {
    this.dropAborted();

    // 1. interactive requests first, decoded alone
    const ii = this.waiting.findIndex((s) => s.o.interactive);
    if (ii >= 0) {
      const s = this.waiting.splice(ii, 1)[0];
      if (this.solo) {
        if (this.maxBatch > 1) this.soloToRow();
        else {
          // no batching: finish the running sequence first
          this.waiting.unshift(s);
          await this.decode();
          return;
        }
      }
      await this.runAlone(s);
      return;
    }

    // 2. admit queued sequences while there are free rows, in stages (1, then up to as many as are decoding: 1, 2, 4,
    //    8…): each one is prefilled before the next decode step, so the first ones start decoding without waiting for
    //    a whole batch of prefills
    let admit = Math.max(1, this.active());
    while (this.waiting.length && this.active() < this.maxBatch && admit-- > 0) {
      const s = this.waiting.shift()!;
      const alone = this.active() === 0;
      if (!alone && this.solo) this.soloToRow();
      const tok = await this.prefill(s);
      if (this.accept(s, tok)) continue;
      if (alone) this.solo = s;
      else {
        this.ensureBatch().copyIn(this.gpu!, this.rows.length);
        this.rows.push(s);
      }
      if (this.waiting.some((w) => w.o.interactive)) return;
    }
    // the batch shrank to one row and nothing is waiting: continue in the single-sequence buffers
    if (this.rows.length === 1 && !this.waiting.length) this.rowToSolo();

    // 3. decode everything active until a row finishes or the scheduler has a sequence to admit
    await this.decode();
  }

  private async decode(): Promise<void> {
    const more = () => this.waiting.length > 0 && (this.active() < this.maxBatch || this.waiting.some((w) => w.o.interactive));
    if (this.solo) {
      const s = this.solo;
      if ((await this.burst([s], true, more)).has(s)) this.solo = null;
      return;
    }
    if (!this.rows.length) return;
    const finished = await this.burst(this.rows.slice(), false, more);
    for (let r = this.rows.length - 1; r >= 0; r--) if (finished.has(this.rows[r])) this.removeRow(r, null);
  }

  // ------------------------------------------------------------------ dynamic KV (wasm / Node / old graphs)

  private async generateDynamicLocked(promptIds: number[], maxNew: number, o: GenerateOptions): Promise<GenerateResult> {
    if (this.dynBusy) {
      await new Promise<void>((go) => {
        const w = { interactive: !!o.interactive, go };
        const at = o.interactive ? this.dynWait.findIndex((x) => !x.interactive) : -1;
        if (at < 0) this.dynWait.push(w);
        else this.dynWait.splice(at, 0, w);
      });
    }
    this.dynBusy = true;
    try {
      return await this.generateDynamic(promptIds, maxNew, o);
    } finally {
      const nextUp = this.dynWait.shift();
      if (nextUp) nextUp.go();
      else this.dynBusy = false;
    }
  }

  private emptyPast(): Feeds {
    const [h, d] = this.kvDims;
    const data = this.kvType === "float16" ? new Uint16Array(0) : new Float32Array(0);
    const feeds: Feeds = {};
    for (const n of this.pastNames) feeds[n] = new this.ort.Tensor(this.kvType, data, [1, h, 0, d]);
    return feeds;
  }

  private async generateDynamic(promptIds: number[], maxNew: number, o: GenerateOptions): Promise<GenerateResult> {
    const eos = new Set(o.eosIds);
    let past = this.emptyPast();
    let input: number[] = promptIds;
    let pos = 0;
    const out: number[] = [];
    let stoppedBy: GenerateResult["stoppedBy"] = "length";
    const t0 = performance.now();
    let prefillMs = 0;

    try {
      for (let step = 0; step < maxNew; step++) {
        if (o.signal?.aborted) {
          stoppedBy = "abort";
          break;
        }
        const feeds: Feeds = {
          input_ids: new this.ort.Tensor("int64", BigInt64Array.from(input, BigInt), [1, input.length]),
          ...past,
        };
        if (this.isStatic) {
          // the static-KV graph also works with an exact-length past: present = past ++ new
          feeds.seqlens_k = new this.ort.Tensor("int32", Int32Array.from([pos + input.length - 1]), [1]);
          feeds.total_sequence_length = new this.ort.Tensor("int32", Int32Array.from([pos + input.length]), []);
        }
        const res = await this.session.run(feeds);
        // Release the KV tensors we just consumed (GPU buffers must be freed explicitly).
        for (const n of this.pastNames) past[n].dispose();
        past = {};
        this.presentNames.forEach((n, i) => (past[this.pastNames[i]] = res[n]));
        pos += input.length;

        const next = argmax((res.logits as TypedTensor<"float32">).data);
        res.logits.dispose();
        if (step === 0) prefillMs = performance.now() - t0;

        if (eos.has(next)) {
          stoppedBy = "eos";
          break;
        }
        out.push(next);
        o.onToken?.(next);
        const cut = repetitionCut(out);
        if (cut > 0) {
          out.length = cut;
          stoppedBy = "repetition";
          break;
        }
        input = [next];
      }
    } finally {
      for (const n of this.pastNames) past[n]?.dispose();
    }
    return { ids: out, promptTokens: promptIds.length, reusedTokens: 0, prefillMs, decodeMs: performance.now() - t0 - prefillMs, stoppedBy };
  }
}

function keptBuffer(device: GPUDevice, bytes: number, keep = true): GPUBuffer {
  const b = device.createBuffer({
    size: Math.max(16, Math.ceil(bytes / 16) * 16),
    usage: BUF.STORAGE | BUF.COPY_SRC | BUF.COPY_DST,
  });
  if (keep) (b as unknown as Record<string, boolean>)[KEEP] = true;
  return b;
}

/** Persistent GPU buffers for one sequence (static KV path); also the prefill scratch for batched rows. */
class GpuState {
  readonly cached: number[] = [];
  private readonly ort: OrtLike;
  private readonly device: GPUDevice;
  private readonly capture: boolean;
  private readonly kv: Feeds = {};
  private readonly fetches: Feeds = {};
  private readonly ids: Tensor;
  private readonly seqk: Tensor;
  private readonly total: Tensor;
  private readonly owned: GPUBuffer[] = [];
  private readonly argmax: GpuArgmax;

  constructor(ort: OrtLike, o: GpuOptions, pastNames: string[], [h, d]: [number, number]) {
    this.ort = ort;
    this.device = o.device;
    this.capture = o.capture;
    const cap = o.capacity ?? KV_CAPACITY;
    for (const n of pastNames) {
      const t = this.tensor("float16", [1, h, cap, d], h * cap * d * 2);
      this.kv[n] = t;
      this.fetches[n.replace("past_key_values.", "present.")] = t;
    }
    this.ids = this.tensor("int64", [1, 1], 8);
    this.seqk = this.tensor("int32", [1], 4);
    this.total = this.tensor("int32", [], 4);
    const logits = this.tensor("float32", [1, 1, VOCAB], VOCAB * 4);
    this.fetches.logits = logits;
    this.argmax = new GpuArgmax(this.device, logits.gpuBuffer as GPUBuffer, this.ids.gpuBuffer as GPUBuffer, 1);
  }

  /** KV buffer of one past input (layout [1, heads, capacity, headDim]) */
  kvBuffer(name: string): GPUBuffer {
    return this.kv[name].gpuBuffer as GPUBuffer;
  }

  private tensor(type: Tensor.GpuBufferDataTypes, dims: number[], bytes: number): Tensor {
    const b = keptBuffer(this.device, bytes);
    this.owned.push(b);
    return this.ort.Tensor.fromGpuBuffer(b, { dataType: type, dims });
  }

  /** Run the model on `input` (see submit) and wait for the greedy next token. */
  async step(session: InferenceSession, input: number[], pos: number, prefill: boolean, rebind: boolean): Promise<number[]> {
    return (await this.submit(session, input, pos, prefill, rebind)).tokens;
  }

  /**
   * Queue the model on `input` (written at KV positions [pos, pos + input.length)); `tokens` resolves to the greedy
   * next token, which is also written as the next single-token input. `input` null: one token, the one the previous
   * step's argmax wrote. `prefill`: multi-token prompt pass (uncaptured, binds its own input buffer); `rebind`:
   * onnxruntime has other tensors bound (after a prefill or a batched step).
   */
  async submit(session: InferenceSession, input: number[] | null, pos: number, prefill: boolean, rebind: boolean): Promise<{ tokens: Promise<number[]> }> {
    const q = this.device.queue;
    const n = input?.length ?? 1;
    let ids = this.ids;
    let temp: GPUBuffer | null = null;
    if (prefill) {
      temp = keptBuffer(this.device, n * 8, false);
      ids = this.ort.Tensor.fromGpuBuffer(temp, { dataType: "int64", dims: [1, n] });
    }
    if (input) q.writeBuffer(ids.gpuBuffer as GPUBuffer, 0, BigInt64Array.from(input, BigInt));
    q.writeBuffer(this.seqk.gpuBuffer as GPUBuffer, 0, Int32Array.from([pos + n - 1]));
    q.writeBuffer(this.total.gpuBuffer as GPUBuffer, 0, Int32Array.from([pos + n]));
    const g = globalThis as Record<string, unknown>;
    g[REBIND] = prefill || rebind;
    try {
      await session.run(
        { ...this.kv, input_ids: ids, seqlens_k: this.seqk, total_sequence_length: this.total },
        this.fetches,
        this.capture ? ({ extra: { gpu_graph_id: prefill ? "-1" : "0" } } as InferenceSession.RunOptions) : undefined,
      );
    } finally {
      g[REBIND] = false;
      temp?.destroy();
    }
    return { tokens: this.argmax.submit(1) };
  }

  dispose(): void {
    this.cached.length = 0;
    for (const b of this.owned) b.destroy();
    this.owned.length = 0;
    this.argmax.dispose();
  }
}

/**
 * KV buffers for up to `maxB` sequences ([maxB, heads, capacity, headDim] per past input) and the tensors for a
 * batched single-token step over the first b rows (front views of the same buffers).
 */
class BatchState {
  private readonly ort: OrtLike;
  private readonly device: GPUDevice;
  private readonly capture: boolean;
  private readonly names: string[];
  private readonly kv: Record<string, GPUBuffer> = {};
  private readonly rowBytes: number;
  /** a temporary for moving a row inside a buffer (a copy may not have the same source and destination buffer) */
  private readonly tmp: GPUBuffer;
  private readonly ids: GPUBuffer;
  private readonly seqk: GPUBuffer;
  private readonly total: GPUBuffer;
  private readonly logits: GPUBuffer;
  private readonly views = new Map<number, { feeds: Feeds; fetches: Feeds }>();
  private readonly argmax: GpuArgmax;
  private readonly dims: [number, number, number];
  private readonly owned: GPUBuffer[] = [];

  constructor(ort: OrtLike, o: GpuOptions, pastNames: string[], [h, d]: [number, number], maxB: number) {
    this.ort = ort;
    this.device = o.device;
    this.capture = o.capture;
    this.names = pastNames;
    const cap = o.capacity ?? KV_CAPACITY;
    this.dims = [h, cap, d];
    this.rowBytes = h * cap * d * 2;
    const mk = (bytes: number, keep = true) => {
      const b = keptBuffer(this.device, bytes, keep);
      this.owned.push(b);
      return b;
    };
    for (const n of pastNames) this.kv[n] = mk(this.rowBytes * maxB);
    this.tmp = mk(this.rowBytes, false);
    this.ids = mk(8 * maxB);
    this.seqk = mk(4 * maxB);
    this.total = mk(4);
    this.logits = mk(VOCAB * 4 * maxB);
    this.argmax = new GpuArgmax(this.device, this.logits, this.ids, maxB);
  }

  private view(b: number): { feeds: Feeds; fetches: Feeds } {
    let v = this.views.get(b);
    if (!v) {
      const T = this.ort.Tensor;
      const [h, cap, d] = this.dims;
      const feeds: Feeds = {
        input_ids: T.fromGpuBuffer(this.ids, { dataType: "int64", dims: [b, 1] }),
        seqlens_k: T.fromGpuBuffer(this.seqk, { dataType: "int32", dims: [b] }),
        total_sequence_length: T.fromGpuBuffer(this.total, { dataType: "int32", dims: [] }),
      };
      const fetches: Feeds = { logits: T.fromGpuBuffer(this.logits, { dataType: "float32", dims: [b, 1, VOCAB] }) };
      for (const n of this.names) {
        const t = T.fromGpuBuffer(this.kv[n], { dataType: "float16", dims: [b, h, cap, d] });
        feeds[n] = t;
        fetches[n.replace("past_key_values.", "present.")] = t;
      }
      v = { feeds, fetches };
      this.views.set(b, v);
    }
    return v;
  }

  /** Copy the single-sequence KV into row r. */
  copyIn(g: GpuState, r: number): void {
    const enc = this.device.createCommandEncoder();
    for (const n of this.names) enc.copyBufferToBuffer(g.kvBuffer(n), 0, this.kv[n], r * this.rowBytes, this.rowBytes);
    this.device.queue.submit([enc.finish()]);
  }

  /** Copy row r into the single-sequence KV. */
  copyOut(r: number, g: GpuState): void {
    const enc = this.device.createCommandEncoder();
    for (const n of this.names) enc.copyBufferToBuffer(this.kv[n], r * this.rowBytes, g.kvBuffer(n), 0, this.rowBytes);
    this.device.queue.submit([enc.finish()]);
  }

  moveRow(from: number, to: number): void {
    const enc = this.device.createCommandEncoder();
    for (const n of this.names) {
      enc.copyBufferToBuffer(this.kv[n], from * this.rowBytes, this.tmp, 0, this.rowBytes);
      enc.copyBufferToBuffer(this.tmp, 0, this.kv[n], to * this.rowBytes, this.rowBytes);
    }
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * Queue one decode step: row i feeds tokens[i] (null: the token the previous step's argmax wrote) at KV position
   * pos[i]. `tokens` resolves to each row's next token.
   */
  async submit(session: InferenceSession, tokens: number[] | null, pos: number[], rebind: boolean): Promise<{ tokens: Promise<number[]> }> {
    const b = pos.length;
    const q = this.device.queue;
    if (tokens) q.writeBuffer(this.ids, 0, BigInt64Array.from(tokens, BigInt));
    q.writeBuffer(this.seqk, 0, Int32Array.from(pos));
    q.writeBuffer(this.total, 0, Int32Array.from([Math.max(...pos) + 1]));
    const { feeds, fetches } = this.view(b);
    const g = globalThis as Record<string, unknown>;
    g[REBIND] = rebind;
    try {
      await session.run(feeds, fetches, this.capture ? ({ extra: { gpu_graph_id: "-1" } } as InferenceSession.RunOptions) : undefined);
    } finally {
      g[REBIND] = false;
    }
    return { tokens: this.argmax.submit(b) };
  }

  dispose(): void {
    for (const b of this.owned) b.destroy();
    this.owned.length = 0;
    this.argmax.dispose();
  }
}

/**
 * Index of the largest logit of each row (lowest index on ties), computed on the GPU; a few bytes are read back. The
 * index is also written into the model's int64 input_ids buffer (row i at element i), so the next decode step can be
 * queued before it is read back.
 */
class GpuArgmax {
  private readonly device: GPUDevice;
  private readonly pipeline: GPUComputePipeline;
  private readonly bindGroup: GPUBindGroup;
  private readonly out: GPUBuffer;
  /** one per step in flight (a buffer cannot be mapped again before it is unmapped) */
  private readonly staging: GPUBuffer[];
  private slot = 0;
  private readonly bytes: number;

  constructor(device: GPUDevice, logits: GPUBuffer, ids: GPUBuffer, rows: number) {
    this.device = device;
    this.bytes = Math.max(16, Math.ceil((rows * 4) / 16) * 16);
    const module = device.createShaderModule({
      code: /* wgsl */ `
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let base = wg.x * ${VOCAB}u;
  var v = -3.4e38;
  var idx = 0u;
  for (var i = t; i < ${VOCAB}u; i += 256u) { let xv = x[base + i]; if (xv > v) { v = xv; idx = i; } }
  bv[t] = v; bi[t] = idx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let ov = bv[t + s]; let oi = bi[t + s];
      if (ov > bv[t] || (ov == bv[t] && oi < bi[t])) { bv[t] = ov; bi[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    o[wg.x] = bi[0];
    ids[2u * wg.x] = bi[0];
    ids[2u * wg.x + 1u] = 0u;
  }
}`,
    });
    this.pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    this.out = device.createBuffer({ size: this.bytes, usage: BUF.STORAGE | BUF.COPY_SRC });
    this.staging = Array.from({ length: PIPELINE_DEPTH + 1 }, () => device.createBuffer({ size: this.bytes, usage: BUF.MAP_READ | BUF.COPY_DST }));
    this.bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: logits, size: rows * VOCAB * 4 } },
        { binding: 1, resource: { buffer: this.out } },
        { binding: 2, resource: { buffer: ids, size: rows * 8 } },
      ],
    });
  }

  /** Queue the argmax of the first `rows` rows; resolves to their indices once read back. */
  submit(rows: number): Promise<number[]> {
    const staging = this.staging[this.slot];
    this.slot = (this.slot + 1) % this.staging.length;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(rows);
    pass.end();
    enc.copyBufferToBuffer(this.out, 0, staging, 0, this.bytes);
    this.device.queue.submit([enc.finish()]);
    return staging.mapAsync(MAP_READ_MODE).then(() => {
      const v = Array.from(new Uint32Array(staging.getMappedRange(), 0, rows));
      staging.unmap();
      return v;
    });
  }

  dispose(): void {
    this.out.destroy();
    for (const s of this.staging) s.destroy();
  }
}

function argmax(a: Float32Array): number {
  let best = 0;
  let bv = a[0];
  for (let i = 1; i < a.length; i++) {
    if (a[i] > bv) {
      bv = a[i];
      best = i;
    }
  }
  return best;
}

/**
 * Detects degenerate loops (a block of tokens repeated back-to-back). Returns the length to keep
 * (output up to the end of the first occurrence) or 0 when no loop is detected.
 */
export function repetitionCut(ids: number[]): number {
  const n = ids.length;
  for (let p = 1; p <= 64 && p * 3 <= n; p++) {
    const reps = p < 4 ? 12 : p < 16 ? 4 : 3;
    if (p * reps > n) continue;
    let ok = true;
    for (let k = 1; k < reps && ok; k++) {
      for (let j = 0; j < p; j++) {
        if (ids[n - 1 - j] !== ids[n - 1 - j - k * p]) {
          ok = false;
          break;
        }
      }
    }
    if (ok) return n - (reps - 1) * p;
  }
  return 0;
}
