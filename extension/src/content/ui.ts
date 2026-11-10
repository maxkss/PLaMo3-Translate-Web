/** Shadow-DOM UI for the content script: selection button, translation panel, page status pill. */

import { statusText } from "../shared/client.ts";
import { LANGUAGES, type EngineStatus } from "../shared/common.ts";
import { icon } from "../shared/icons.ts";
import COMPONENTS_CSS from "../styles/components.css";
import TOKENS_CSS from "../styles/tokens.css";

const ICON = icon("translate");
/** Spinner (md): a 14px ring with a 3px stroke and a 37.5% arc, in a 20px box. */
export const SPINNER = `<svg class="spin" viewBox="0 0 20 20" width="20" height="20" fill="none" stroke-width="3" aria-hidden="true"><circle class="t" cx="10" cy="10" r="5.5"/><circle cx="10" cy="10" r="5.5" stroke="currentColor" stroke-linecap="round" stroke-dasharray="12.96 34.56"/></svg>`;

/** Overlay-only styles; tokens and components come from the shared stylesheets (src/styles). */
const OVERLAY_CSS = `
:host { all: initial; }
* { box-sizing: border-box; }
.root {
  color-scheme: light dark;
  font: 14px/20px var(--font);
  color: var(--text); -webkit-font-smoothing: antialiased;
}
/* selection button */
.fab {
  position: absolute; z-index: 2147483647; width: 32px; height: 32px; padding: 0; border: 0; margin: 0;
  display: grid; place-items: center; cursor: pointer; color: var(--text);
  transition: background-image var(--fast) var(--ease), transform var(--fast) var(--ease);
}
@media (hover: hover) { .fab:hover { background-image: linear-gradient(var(--hover), var(--hover)); } }
.fab:active { background-image: linear-gradient(var(--pressed), var(--pressed)); transform: scale(.98); }
/* selection panel */
.panel { position: absolute; z-index: 2147483647; width: min(440px, calc(100vw - 24px)); display: flex; flex-direction: column; overflow: hidden; }
.head { display: flex; align-items: center; gap: 4px; padding: 8px 8px 0; }
.head .select { max-width: 240px; }
.head .select > select { height: 28px; }
.body { padding: 8px 16px 16px; max-height: 50vh; overflow: auto; white-space: pre-wrap; word-break: keep-all; overflow-wrap: anywhere; font-size: 15px; line-height: 24px; }
.body.error { color: var(--on-error-muted); }
.notice { margin: 0 8px 8px; }
.notice .nicon { display: contents; }
.caret { display: inline-block; width: 2px; height: 1.1em; vertical-align: -3px; background: var(--text); margin-left: 1px; animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: 0; } }
/* page progress (Toast layout: 16px padding, 12px gap, end slot 20px high) */
.pill {
  position: fixed; z-index: 2147483647; right: 16px; bottom: 16px; display: flex; align-items: center; gap: 12px;
  width: max-content; max-width: min(400px, calc(100vw - 32px)); padding: 16px;
  animation: rise var(--fast) var(--ease);
}
@keyframes rise { from { opacity: 0; transform: translateY(8px); } }
.pill .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
.pill .state { display: grid; place-items: center; width: 20px; height: 20px; flex: none; }
.pill .state .i { width: 20px; height: 20px; }
.pill .state[data-tone="warn"] { color: var(--on-warning-muted); }
.pill .state[data-tone="error"] { color: var(--on-error-muted); }
.pill .end { display: flex; align-items: center; gap: 8px; height: 20px; margin-inline-end: -4px; }
.pill .bar { width: 64px; flex: none; }
/* image translation overlay (content/image.ts) */
.img-ov { position: fixed; z-index: 2147483646; overflow: hidden; pointer-events: none; }
.img-stage { position: absolute; left: 0; top: 0; transform-origin: 0 0; }
.img-para {
  position: absolute; pointer-events: auto; cursor: pointer; word-break: keep-all; overflow-wrap: anywhere;
  border-radius: 3px; overflow: visible; transition: opacity .15s;
  font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Hiragino Sans", "Noto Sans JP", sans-serif;
}
.img-para.busy { -webkit-backdrop-filter: blur(6px); backdrop-filter: blur(6px); }
.img-para.peek, .img-ov.peek .img-para { opacity: 0; }
.img-badge {
  position: absolute; right: 8px; top: 8px; display: flex; align-items: center; gap: 4px; pointer-events: auto;
  padding: 4px 4px 4px 12px; font-weight: 500; max-width: calc(100% - 16px); transition: opacity var(--fast) var(--ease);
}
.img-badge .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 4px; }
.img-ov.peek .img-badge .peek { background-color: var(--pressed); color: var(--text); }
.img-badge .spin { display: none; }
.img-badge.busy { padding: 8px 12px 8px 8px; gap: 8px; }
.img-badge.busy .spin { display: block; }
.img-badge.busy .btn { display: none; }
/* when finished the badge is just its buttons, faint until hovered */
.img-badge.done { padding: 4px; }
.img-badge.done .label { display: none; }
.img-badge.done:not(:hover) { opacity: .6; }
.img-badge.done:not(:hover) .btn:not(.peek) { display: none; }
`;

const CSS = TOKENS_CSS + COMPONENTS_CSS + OVERLAY_CSS;

export type PillTone = "busy" | "done" | "warn" | "error";

export interface PanelHandle {
  setText(text: string, streaming: boolean): void;
  setLoading(): void;
  setError(msg: string): void;
  /** Show the engine state when it needs attention (not installed, downloading, loading, error). */
  setStatus(s: EngineStatus | null): void;
  onLangChange(cb: (lang: string) => void): void;
  onOpenOptions(cb: () => void): void;
  close(): void;
}

export class Ui {
  readonly host: HTMLElement;
  private readonly root: HTMLDivElement;
  private fab: HTMLButtonElement | null = null;
  private panel: HTMLDivElement | null = null;
  private pill: HTMLDivElement | null = null;
  private panelClose: (() => void) | null = null;

  constructor() {
    this.host = document.createElement("plamo-translate-ui");
    this.host.style.cssText = "position:absolute;top:0;left:0;width:0;height:0;z-index:2147483647;";
    const shadow = this.host.attachShadow({ mode: "open" });
    const style = document.createElement("style");
    style.textContent = CSS;
    this.root = document.createElement("div");
    this.root.className = "root";
    shadow.append(style, this.root);
    document.documentElement.append(this.host);
  }

  /** Add an element to the shadow root (image overlays). */
  mount(el: HTMLElement): void {
    this.root.append(el);
  }

  /** Hide all UI (e.g. while taking a screenshot of the page). */
  setHidden(hidden: boolean): void {
    this.host.style.visibility = hidden ? "hidden" : "";
  }

  contains(node: EventTarget | null): boolean {
    return node instanceof Node && (node === this.host || this.host.contains(node));
  }

  showFab(x: number, y: number, onClick: () => void): void {
    this.hideFab();
    const b = document.createElement("button");
    b.className = "fab float";
    b.title = "PLaMo로 번역";
    b.innerHTML = ICON;
    b.style.left = `${x}px`;
    b.style.top = `${y}px`;
    b.addEventListener("mousedown", (e) => e.preventDefault()); // keep the selection
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      this.hideFab();
      onClick();
    });
    this.root.append(b);
    this.fab = b;
  }

  hideFab(): void {
    this.fab?.remove();
    this.fab = null;
  }

  get panelOpen(): boolean {
    return !!this.panel;
  }

  /** Open the translation panel anchored to a document-coordinate rect. */
  openPanel(anchor: { left: number; top: number; bottom: number }, lang: string, onClose: () => void): PanelHandle {
    this.closePanel();
    const p = document.createElement("div");
    p.className = "panel float";
    p.innerHTML = `
      <div class="head">
        <span class="select ghost"><select class="lang" title="번역할 언어"></select></span>
        <span class="grow"></span>
        <button class="btn ghost icon sm copy" title="복사">${icon("copy")}</button>
        <button class="btn ghost icon sm x" title="닫기">${icon("x")}</button>
      </div>
      <div class="body"></div>
      <div class="banner notice hidden"><span class="nicon"></span><span class="title msg"></span><button class="btn sm opts hidden">설정</button></div>`;
    const sel = p.querySelector<HTMLSelectElement>(".lang")!;
    for (const [code, label] of LANGUAGES) sel.add(new Option(label, code, false, code === lang));
    const body = p.querySelector<HTMLDivElement>(".body")!;
    const notice = p.querySelector<HTMLDivElement>(".notice")!;
    const msg = notice.querySelector<HTMLSpanElement>(".msg")!;
    const opts = p.querySelector<HTMLButtonElement>(".opts")!;
    const copy = p.querySelector<HTMLButtonElement>(".copy")!;
    let finalText = "";
    copy.addEventListener("click", () => {
      void navigator.clipboard.writeText(finalText).then(() => {
        copy.innerHTML = icon("check");
        setTimeout(() => (copy.innerHTML = icon("copy")), 1200);
      });
    });
    p.querySelector(".x")!.addEventListener("click", () => this.closePanel());
    p.addEventListener("mousedown", (e) => e.stopPropagation());

    this.root.append(p);
    this.panel = p;
    this.panelClose = onClose;

    // position: below the anchor, or above when there is no room
    const vw = document.documentElement.clientWidth;
    const width = p.offsetWidth;
    const left = Math.max(window.scrollX + 12, Math.min(anchor.left, window.scrollX + vw - width - 12));
    p.style.left = `${left}px`;
    const spaceBelow = window.scrollY + window.innerHeight - anchor.bottom;
    p.style.top = `${anchor.bottom + 8}px`;
    if (spaceBelow < 220 && anchor.top - window.scrollY > 240) {
      p.style.top = "";
      p.style.top = `${Math.max(window.scrollY + 8, anchor.top - 8 - Math.max(p.offsetHeight, 160))}px`;
    }

    return {
      setText: (text, streaming) => {
        finalText = text;
        body.classList.remove("error");
        body.textContent = text;
        if (streaming) body.append(Object.assign(document.createElement("span"), { className: "caret" }));
      },
      setLoading: () => {
        body.classList.remove("error");
        body.innerHTML = `<span class="sk"></span><span class="sk"></span><span class="sk"></span>`;
      },
      setError: (m) => {
        body.classList.add("error");
        body.textContent = m;
      },
      setStatus: (st) => {
        const show = !st || !(st.state === "ready" || st.state === "idle");
        notice.classList.toggle("hidden", !show);
        if (!show) return;
        msg.textContent = st?.state === "missing" ? "모델 설치 필요" : statusText(st);
        notice.className = `banner notice${st?.state === "missing" ? " warning" : st?.state === "error" ? " error" : ""}`;
        notice.querySelector(".nicon")!.innerHTML = st?.state === "missing" ? icon("warning", 20) : st?.state === "error" ? icon("error", 20) : SPINNER;
        opts.classList.toggle("hidden", st?.state !== "missing" && st?.state !== "error");
        opts.textContent = st?.state === "missing" ? "설치" : "설정";
      },
      onLangChange: (cb) => sel.addEventListener("change", () => cb(sel.value)),
      onOpenOptions: (cb) => opts.addEventListener("click", cb),
      close: () => this.closePanel(),
    };
  }

  closePanel(): void {
    if (!this.panel) return;
    this.panel.remove();
    this.panel = null;
    const cb = this.panelClose;
    this.panelClose = null;
    cb?.();
  }

  /** The page-progress toast; the returned update sets its label, progress bar and leading icon. */
  showPill(opts: { onRestore: () => void; onClose: () => void }): (label: string, progress: number | null, tone: PillTone) => void {
    this.pill?.remove();
    const pill = document.createElement("div");
    pill.className = "pill float";
    pill.innerHTML = `<span class="state"></span><span class="label"></span><span class="end"><span class="bar"><i></i></span><button class="btn sm restore">원문 보기</button><button class="btn ghost icon sm x" title="숨기기">${icon("x")}</button></span>`;
    pill.querySelector(".restore")!.addEventListener("click", opts.onRestore);
    pill.querySelector(".x")!.addEventListener("click", () => {
      pill.classList.add("hidden");
      opts.onClose();
    });
    this.root.append(pill);
    this.pill = pill;
    const label = pill.querySelector<HTMLSpanElement>(".label")!;
    const bar = pill.querySelector<HTMLElement>(".bar")!;
    const fill = pill.querySelector<HTMLElement>(".bar > i")!;
    const state = pill.querySelector<HTMLElement>(".state")!;
    let shown: PillTone | null = null;
    return (text, progress, tone) => {
      label.textContent = text;
      if (tone !== shown) {
        state.innerHTML = tone === "busy" ? SPINNER : icon(tone === "done" ? "check" : tone === "warn" ? "warning" : "error", 20);
        state.dataset.tone = tone;
        shown = tone;
      }
      bar.classList.toggle("hidden", tone !== "busy" || progress == null);
      if (progress != null) fill.style.width = `${Math.round(progress * 100)}%`;
    };
  }

  hidePill(): void {
    this.pill?.remove();
    this.pill = null;
  }
}
