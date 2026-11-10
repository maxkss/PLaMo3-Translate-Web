/** Content script: drag-to-translate and whole-page translation. */

import { EngineClient, statusText } from "../shared/client.ts";
import { getSettings, setSettings, type Settings } from "../shared/common.ts";
import { ImageTranslator } from "./image.ts";
import { PageTranslator } from "./page.ts";
import { Ui, type PanelHandle } from "./ui.ts";

declare global {
  interface Window {
    __plamoTranslate?: boolean;
  }
}

if (!window.__plamoTranslate) {
  window.__plamoTranslate = true;
  void main();
}

async function main(): Promise<void> {
  let settings: Settings = await getSettings();
  chrome.storage.onChanged.addListener(async () => {
    const prev = settings;
    settings = await getSettings();
    if (page.active && prev.displayMode !== settings.displayMode) page.setMode(settings.displayMode);
    if (page.active && prev.targetLang !== settings.targetLang) startPage();
  });

  const client = new EngineClient();
  let ui: Ui | null = null;
  const getUi = () => (ui ??= new Ui());

  // ------------------------------------------------------------------ selection

  let selReqId: string | null = null;
  let panel: PanelHandle | null = null;
  let unsubscribeStatus: (() => void) | null = null;

  function selectionInfo(): { text: string; rect: DOMRect } | null {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const text = sel.toString().trim();
    if (!text) return null;
    const range = sel.getRangeAt(0);
    const rects = range.getClientRects();
    // last line box of the selection (where the mouse usually ends), falling back to the bounding box
    const rect = rects.length ? rects[rects.length - 1] : range.getBoundingClientRect();
    return { text, rect };
  }

  function translateSelection(text: string, rect: DOMRect | null): void {
    text = text.trim().slice(0, 12000);
    if (!text) return;
    if (selReqId) client.cancel([selReqId]);
    const u = getUi();
    u.hideFab();
    const anchor = rect
      ? { left: rect.left + window.scrollX, top: rect.top + window.scrollY, bottom: rect.bottom + window.scrollY }
      : { left: window.scrollX + window.innerWidth / 2 - 220, top: window.scrollY + 80, bottom: window.scrollY + 80 };
    unsubscribeStatus?.();
    panel = u.openPanel(anchor, settings.targetLang, () => {
      if (selReqId) client.cancel([selReqId]);
      selReqId = null;
      panel = null;
      unsubscribeStatus?.();
      unsubscribeStatus = null;
    });
    const p = panel;
    p.onOpenOptions(() => void chrome.runtime.sendMessage({ type: "open-options" }));
    p.setStatus(client.status);
    unsubscribeStatus = client.onStatus((s) => p.setStatus(s));
    const run = (lang: string) => {
      if (selReqId) client.cancel([selReqId]);
      const id = (selReqId = client.newId());
      p.setLoading();
      client
        .translate({ id, text, targetLang: lang, priority: 0 }, (partial) => {
          if (selReqId === id) p.setText(partial, true);
        })
        .then(
          (out) => {
            if (selReqId !== id) return;
            p.setText(out, false);
            selReqId = null;
          },
          (e: Error) => {
            if (e?.name === "AbortError" || selReqId !== id) return;
            p.setError(`번역하지 못했습니다: ${e.message}`);
            p.setStatus(client.status);
          },
        );
    };
    p.onLangChange((lang) => {
      void setSettings({ targetLang: lang });
      run(lang);
    });
    run(settings.targetLang);
  }

  document.addEventListener(
    "mouseup",
    (e) => {
      if (ui?.contains(e.target)) return;
      if (e.button !== 0) return;
      setTimeout(() => {
        const info = selectionInfo();
        if (!info || info.text.length < 2) return;
        if (settings.selectionAuto) {
          translateSelection(info.text, info.rect);
        } else if (settings.selectionButton) {
          const x = Math.min(info.rect.right + window.scrollX + 4, window.scrollX + document.documentElement.clientWidth - 40);
          const y = info.rect.bottom + window.scrollY + 6;
          getUi().showFab(x, y, () => translateSelection(info.text, info.rect));
        }
      }, 0);
    },
    true,
  );

  document.addEventListener(
    "mousedown",
    (e) => {
      if (ui?.contains(e.target)) return;
      ui?.hideFab();
      if (ui?.panelOpen) ui.closePanel();
    },
    true,
  );

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      ui?.hideFab();
      ui?.closePanel();
    }
  });

  // ------------------------------------------------------------------ page

  const page = new PageTranslator(client);
  let pillUpdate: ReturnType<Ui["showPill"]> | null = null;
  let lastError = "";
  let pageStatusUnsub: (() => void) | null = null;

  function startPage(): void {
    lastError = "";
    pillUpdate = getUi().showPill({
      onRestore: () => stopPage(),
      onClose: () => {},
    });
    page.onProgress = ({ done, total, running }) => {
      if (!pillUpdate) return;
      const s = client.status;
      if (s?.state === "missing") {
        pillUpdate("모델 설치 필요", null, "warn");
      } else if (s?.state === "error") {
        pillUpdate(statusText(s), null, "error");
      } else if (s && (s.state === "loading" || s.state === "downloading")) {
        pillUpdate(statusText(s), s.progress ?? null, "busy");
      } else if (running || done < total) {
        pillUpdate(`번역 중 · ${done}/${total}`, total ? done / total : null, "busy");
      } else {
        pillUpdate(lastError ? `일부 번역 실패: ${lastError}` : "번역 완료", null, lastError ? "warn" : "done");
      }
    };
    page.onError = (msg) => {
      lastError = msg;
    };
    pageStatusUnsub?.();
    pageStatusUnsub = client.onStatus(() => page.active && page.onProgress(page.progress()));
    page.start(settings.targetLang, settings.displayMode);
  }

  function stopPage(): void {
    page.stop();
    pageStatusUnsub?.();
    pageStatusUnsub = null;
    ui?.hidePill();
    pillUpdate = null;
  }

  // ------------------------------------------------------------------ images

  const images = new ImageTranslator(
    client,
    (el) => getUi().mount(el),
    (hidden) => ui?.setHidden(hidden),
  );

  // ------------------------------------------------------------------ messages

  chrome.runtime.onMessage.addListener((msg: { type: string; text?: string; srcUrl?: string }, _sender, sendResponse) => {
    switch (msg.type) {
      case "page-toggle":
        if (page.active) stopPage();
        else startPage();
        sendResponse({ active: page.active });
        break;
      case "page-state":
        sendResponse({ active: page.active, ...page.progress() });
        break;
      case "translate-image":
        images.toggle(msg.srcUrl ?? "", settings.targetLang);
        sendResponse({ ok: true });
        break;
      case "translate-selection": {
        const info = selectionInfo();
        const text = msg.text || info?.text || "";
        if (text) translateSelection(text, info?.rect ?? null);
        sendResponse({ ok: !!text });
        break;
      }
    }
    return false;
  });
}
