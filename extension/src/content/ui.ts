/** Shadow-DOM UI for the content script: selection button, translation panel, page status pill. */

import { LANGUAGES } from "../shared/common.ts";

const ICON = `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M12.9 15.1l-2.5-2.5.03-.03A17.5 17.5 0 0 0 14.1 6H17V4h-7V2H8v2H1v2h11.2A15.7 15.7 0 0 1 9 11.4 15.6 15.6 0 0 1 6.7 8H4.7a17.5 17.5 0 0 0 3 4.6l-5.1 5 1.4 1.4L9 14l3.1 3.1.8-2zM18.5 10h-2L12 22h2l1.1-3h4.8l1.1 3h2l-4.5-12zm-2.6 7l1.6-4.3 1.6 4.3h-3.2z"/></svg>`;

const CSS = `
:host { all: initial; }
* { box-sizing: border-box; }
.root {
  --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #e3e3e8; --accent: #3b5bdb; --accent-fg: #fff;
  --shadow: 0 8px 28px rgba(0,0,0,.18), 0 1px 3px rgba(0,0,0,.12);
  font: 14px/1.55 system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Segoe UI", sans-serif;
  color: var(--fg);
}
@media (prefers-color-scheme: dark) {
  .root { --bg: #1f1f23; --fg: #ececf1; --muted: #a1a1aa; --line: #34343b; --accent: #748ffc; --accent-fg: #111; }
}
.fab {
  position: absolute; z-index: 2147483647; width: 30px; height: 30px; border-radius: 9px;
  display: grid; place-items: center; cursor: pointer; border: 1px solid var(--line);
  background: var(--bg); color: var(--accent); box-shadow: var(--shadow); padding: 0;
}
.fab:hover { background: var(--accent); color: var(--accent-fg); }
.panel {
  position: absolute; z-index: 2147483647; width: min(440px, calc(100vw - 24px));
  background: var(--bg); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow);
  display: flex; flex-direction: column; overflow: hidden;
}
.head { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-bottom: 1px solid var(--line); }
.brand { font-weight: 600; font-size: 12px; color: var(--muted); letter-spacing: .02em; flex: 1; }
select { font: inherit; font-size: 12px; color: var(--fg); background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 2px 4px; }
.btn { font: inherit; font-size: 12px; border: 1px solid var(--line); background: var(--bg); color: var(--fg); border-radius: 6px; padding: 3px 8px; cursor: pointer; }
.btn:hover { border-color: var(--accent); color: var(--accent); }
.x { border: 0; background: none; color: var(--muted); cursor: pointer; font-size: 18px; line-height: 1; padding: 0 2px; }
.body { padding: 12px 14px; max-height: 50vh; overflow: auto; white-space: pre-wrap; word-break: break-word; }
.body.pending { color: var(--muted); }
.foot { padding: 6px 12px; border-top: 1px solid var(--line); font-size: 11px; color: var(--muted); display: flex; gap: 8px; align-items: center; }
.foot .grow { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.caret { display: inline-block; width: 7px; height: 1em; vertical-align: -2px; background: var(--accent); opacity: .6; animation: blink 1s steps(2) infinite; margin-left: 1px; }
@keyframes blink { 50% { opacity: 0; } }
.pill {
  position: fixed; z-index: 2147483647; right: 16px; bottom: 16px; display: flex; align-items: center; gap: 8px;
  background: var(--bg); border: 1px solid var(--line); border-radius: 999px; box-shadow: var(--shadow);
  padding: 6px 8px 6px 12px; font-size: 12px; max-width: calc(100vw - 32px);
}
.pill .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar { width: 60px; height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; flex: none; }
.bar > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .3s; }
.hidden { display: none !important; }
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
  position: absolute; right: 6px; top: 6px; display: flex; align-items: center; gap: 6px; pointer-events: auto;
  background: var(--bg); border: 1px solid var(--line); border-radius: 999px; box-shadow: var(--shadow);
  padding: 3px 4px 3px 10px; font-size: 12px; max-width: calc(100% - 12px); opacity: .92;
}
.img-badge .label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.img-badge .btn { padding: 1px 6px; }
.img-ov.peek .img-badge .peek { color: var(--accent); border-color: var(--accent); }
/* when finished the badge shrinks to an icon in the corner; hovering it brings the controls back */
.img-badge .ico { display: none; color: var(--accent); }
.img-badge.done:not(:hover) { padding: 3px; opacity: .75; }
.img-badge.done:not(:hover) > :not(.ico) { display: none; }
.img-badge.done:not(:hover) .ico { display: grid; place-items: center; }
.spin { display: none; width: 10px; height: 10px; border-radius: 50%; border: 2px solid var(--line); border-top-color: var(--accent); animation: spin .8s linear infinite; flex: none; }
.img-badge.busy .spin { display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }
`;

export interface PanelHandle {
  setText(text: string, streaming: boolean): void;
  setPending(msg: string): void;
  setFooter(msg: string): void;
  onLangChange(cb: (lang: string) => void): void;
  onOpenOptions(cb: () => void): void;
  close(): void;
}

export const TRANSLATE_ICON = ICON;

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
    b.className = "fab";
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
    p.className = "panel";
    p.innerHTML = `
      <div class="head">
        <span class="brand">PLaMo 3 Translate</span>
        <select class="lang" title="번역할 언어"></select>
        <button class="btn copy" title="복사">복사</button>
        <button class="x" title="닫기">×</button>
      </div>
      <div class="body pending"></div>
      <div class="foot"><span class="grow"></span><button class="btn opts hidden">모델 설정</button></div>`;
    const sel = p.querySelector<HTMLSelectElement>(".lang")!;
    for (const [code, label] of LANGUAGES) sel.add(new Option(label, code, false, code === lang));
    const body = p.querySelector<HTMLDivElement>(".body")!;
    const foot = p.querySelector<HTMLSpanElement>(".foot .grow")!;
    const opts = p.querySelector<HTMLButtonElement>(".opts")!;
    let finalText = "";
    p.querySelector(".copy")!.addEventListener("click", () => void navigator.clipboard.writeText(finalText));
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
        body.classList.remove("pending");
        body.textContent = text;
        if (streaming) body.append(Object.assign(document.createElement("span"), { className: "caret" }));
      },
      setPending: (msg) => {
        body.classList.add("pending");
        body.textContent = msg;
      },
      setFooter: (msg) => {
        foot.textContent = msg;
        opts.classList.toggle("hidden", !/설치|오류|내려받/.test(msg));
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

  showPill(opts: { onRestore: () => void; onClose: () => void }): (label: string, progress: number | null) => void {
    this.pill?.remove();
    const pill = document.createElement("div");
    pill.className = "pill";
    pill.innerHTML = `<span class="label"></span><span class="bar"><i></i></span><button class="btn restore">원문 보기</button><button class="x" title="숨기기">×</button>`;
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
    return (text, progress) => {
      label.textContent = text;
      bar.classList.toggle("hidden", progress == null);
      if (progress != null) fill.style.width = `${Math.round(progress * 100)}%`;
    };
  }

  hidePill(): void {
    this.pill?.remove();
    this.pill = null;
  }
}
