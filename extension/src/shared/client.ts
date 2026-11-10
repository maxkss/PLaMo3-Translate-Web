/** Client side of the engine port, used by content scripts, the popup and the options page. */

import { ENGINE_PORT, MODELS, type ClientToEngine, type EngineStatus, type EngineToClient, type ErasePatch, type OcrRequest, type OcrResult, type TranslateRequest } from "./common.ts";

interface Pending {
  resolve: (text: string) => void;
  reject: (e: Error) => void;
  onDelta?: (text: string) => void;
}

interface PendingOcr {
  resolve: (r: OcrResult) => void;
  reject: (e: Error) => void;
  /** after the result: the remaining patches (inpainting on photos) */
  patches?: (p: ErasePatch[]) => void;
}

/** An OCR result, and the patches of its pending paragraphs (resolved with [] if there are none or they failed). */
export interface OcrOutcome {
  result: OcrResult;
  morePatches: Promise<ErasePatch[]>;
}

export class EngineClient {
  private port: chrome.runtime.Port | null = null;
  private connecting: Promise<chrome.runtime.Port> | null = null;
  private readonly pending = new Map<string, Pending>();
  private readonly pendingOcr = new Map<string, PendingOcr>();
  private readonly statusListeners = new Set<(s: EngineStatus) => void>();
  status: EngineStatus | null = null;
  private counter = 0;

  private async connect(): Promise<chrome.runtime.Port> {
    if (this.port) return this.port;
    if (this.connecting) return this.connecting;
    this.connecting = (async () => {
      const res = await chrome.runtime.sendMessage({ type: "ensure-engine" });
      if (!res?.ok) throw new Error(res?.error ?? "엔진을 시작하지 못했습니다");
      const port = chrome.runtime.connect({ name: ENGINE_PORT });
      port.onMessage.addListener((m: EngineToClient) => this.onMessage(m));
      port.onDisconnect.addListener(() => {
        void chrome.runtime.lastError;
        this.port = null;
        for (const [, p] of [...this.pending, ...this.pendingOcr]) p.reject(new Error("엔진 연결이 끊어졌습니다"));
        for (const [, o] of this.pendingOcr) o.patches?.([]);
        this.pending.clear();
        this.pendingOcr.clear();
      });
      this.port = port;
      return port;
    })().finally(() => {
      this.connecting = null;
    });
    return this.connecting;
  }

  private onMessage(m: EngineToClient): void {
    if (m.type === "status") {
      this.status = m.status;
      for (const l of this.statusListeners) l(m.status);
      return;
    }
    const o = this.pendingOcr.get(m.id);
    if (o) {
      if (m.type === "ocr-patches") {
        this.pendingOcr.delete(m.id);
        o.patches?.(m.patches);
      } else if (m.type === "ocr-done") {
        // stay registered for the patches when some are still being made
        if (!m.result.pending.length) this.pendingOcr.delete(m.id);
        o.resolve(m.result);
      } else if (m.type === "error") {
        this.pendingOcr.delete(m.id);
        o.reject(new Error(m.message));
      }
      return;
    }
    const p = this.pending.get(m.id);
    if (!p || m.type === "ocr-done" || m.type === "ocr-patches") return;
    if (m.type === "delta") p.onDelta?.(m.text);
    else {
      this.pending.delete(m.id);
      if (m.type === "done") p.resolve(m.text);
      else p.reject(new Error(m.message));
    }
  }

  async send(msg: ClientToEngine): Promise<void> {
    (await this.connect()).postMessage(msg);
  }

  onStatus(cb: (s: EngineStatus) => void): () => void {
    this.statusListeners.add(cb);
    if (this.status) cb(this.status);
    return () => this.statusListeners.delete(cb);
  }

  newId(): string {
    return `${Date.now().toString(36)}-${(this.counter++).toString(36)}`;
  }

  async translate(req: Omit<TranslateRequest, "type" | "id"> & { id?: string }, onDelta?: (text: string) => void): Promise<string> {
    const id = req.id ?? this.newId();
    const port = await this.connect();
    return new Promise<string>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onDelta });
      port.postMessage({ ...req, type: "translate", id, stream: !!onDelta } satisfies TranslateRequest);
    });
  }

  /** OCR an image (see OcrRequest). */
  async ocr(req: Omit<OcrRequest, "type" | "id">): Promise<OcrOutcome> {
    const id = this.newId();
    const port = await this.connect();
    let patches!: (p: ErasePatch[]) => void;
    const morePatches = new Promise<ErasePatch[]>((r) => (patches = r));
    const result = await new Promise<OcrResult>((resolve, reject) => {
      this.pendingOcr.set(id, { resolve, reject, patches });
      port.postMessage({ ...req, type: "ocr", id } satisfies OcrRequest);
    });
    if (!result.pending.length) patches([]);
    return { result, morePatches };
  }

  cancel(ids: string[]): void {
    for (const id of ids) {
      const p = this.pending.get(id);
      if (p) {
        this.pending.delete(id);
        p.reject(new DOMException("cancelled", "AbortError"));
      }
    }
    this.port?.postMessage({ type: "cancel", ids } satisfies ClientToEngine);
  }

  cancelAll(): void {
    for (const [, p] of this.pending) p.reject(new DOMException("cancelled", "AbortError"));
    this.pending.clear();
    this.port?.postMessage({ type: "cancel", all: true } satisfies ClientToEngine);
  }
}

export function statusText(s: EngineStatus | null): string {
  if (!s) return "엔진 연결 중…";
  const pct = s.progress != null ? ` ${(s.progress * 100).toFixed(0)}%` : "";
  switch (s.state) {
    case "missing":
      return `${s.model ? `${MODELS[s.model].label} ` : ""}모델이 설치되지 않았습니다`;
    case "downloading":
      return s.message ?? `모델 내려받는 중…${pct}`;
    case "loading":
      return `${s.model ? `${MODELS[s.model].label} ` : ""}모델 불러오는 중…${pct}`;
    case "idle":
      return "대기 중 (첫 번역 시 모델을 불러옵니다)";
    case "ready":
      return `준비됨 · ${s.model ? `${MODELS[s.model].label} · ` : ""}${s.backend}${s.tokensPerSec ? ` · ${s.tokensPerSec} tok/s` : ""}`;
    case "error":
      return `오류: ${s.message ?? ""}`;
  }
}
