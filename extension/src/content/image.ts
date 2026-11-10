/**
 * Image translation: OCR the right-clicked image (offscreen document + OCR worker), translate each paragraph with the
 * engine, and draw the translations over the image — each paragraph is covered with a box in the sampled background
 * colour and the translation is written in the sampled text colour, sized to fit the box.
 *
 * The overlay lives in the content script's Shadow DOM, positioned (fixed) over the image's drawn area and kept in
 * place on scroll/resize. Paragraph boxes use the OCR image's pixel coordinates inside a scaled "stage".
 */

import type { EngineClient } from "../shared/client.ts";
import type { OcrOutcome } from "../shared/client.ts";
import type { ErasePatch, OcrParagraph, OcrResult } from "../shared/common.ts";
import { FitGroup, fontMetrics, ParaView, type FontMetrics } from "./image-view.ts";
import { axes, inferLayout } from "./layout.ts";
import { skipParagraph } from "../shared/skip.ts";
import { TRANSLATE_ICON } from "./ui.ts";

/** The part of the drawn image the OCR result covers, as fractions (a screenshot may only cover the visible part). */
interface Region {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}
const FULL: Region = { x0: 0, y0: 0, x1: 1, y1: 1 };

interface Overlay {
  img: HTMLImageElement;
  src: string;
  root: HTMLDivElement;
  stage: HTMLDivElement;
  badge: HTMLDivElement;
  label: HTMLSpanElement;
  result: OcrResult | null;
  region: Region;
  reqIds: string[];
  dispose(): void;
}


/** Where the image content is drawn inside the element (object-fit / object-position), in viewport coordinates. */
function drawnRect(img: HTMLImageElement): DOMRect | null {
  const r = img.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  const cs = getComputedStyle(img);
  const px = (v: string) => parseFloat(v) || 0;
  const left = r.left + px(cs.borderLeftWidth) + px(cs.paddingLeft);
  const top = r.top + px(cs.borderTopWidth) + px(cs.paddingTop);
  const cw = r.width - px(cs.borderLeftWidth) - px(cs.borderRightWidth) - px(cs.paddingLeft) - px(cs.paddingRight);
  const ch = r.height - px(cs.borderTopWidth) - px(cs.borderBottomWidth) - px(cs.paddingTop) - px(cs.paddingBottom);
  const nw = img.naturalWidth || cw, nh = img.naturalHeight || ch;
  let w = cw, h = ch;
  const fit = cs.objectFit;
  if (fit === "contain" || fit === "scale-down" || fit === "cover" || fit === "none") {
    let s = fit === "cover" ? Math.max(cw / nw, ch / nh) : fit === "none" ? 1 : Math.min(cw / nw, ch / nh);
    if (fit === "scale-down") s = Math.min(1, s);
    w = nw * s;
    h = nh * s;
  }
  const [px0 = "50%", py0 = "50%"] = cs.objectPosition.split(/\s+/);
  const pos = (v: string, free: number) => (v.endsWith("%") ? (free * parseFloat(v)) / 100 : parseFloat(v) || 0);
  return new DOMRect(left + pos(px0, cw - w), top + pos(py0, ch - h), w, h);
}

/** The visible part of the element's content box, in viewport coordinates. */
function clipRect(img: HTMLImageElement): DOMRect {
  const r = img.getBoundingClientRect();
  const x0 = Math.max(0, r.left), y0 = Math.max(0, r.top);
  const x1 = Math.min(window.innerWidth, r.right), y1 = Math.min(window.innerHeight, r.bottom);
  return new DOMRect(x0, y0, Math.max(0, x1 - x0), Math.max(0, y1 - y0));
}

export class ImageTranslator {
  private readonly overlays = new Map<HTMLImageElement, Overlay>();
  private lastTarget: EventTarget | null = null;
  private ticking = false;

  private readonly client: EngineClient;
  private readonly mount: (el: HTMLElement) => void;
  private readonly hideUi: (hidden: boolean) => void;

  constructor(client: EngineClient, mount: (el: HTMLElement) => void, hideUi: (hidden: boolean) => void) {
    this.client = client;
    this.mount = mount;
    this.hideUi = hideUi;
    document.addEventListener("contextmenu", (e) => (this.lastTarget = e.composedPath()[0] ?? e.target), true);
    const schedule = () => {
      if (this.ticking || !this.overlays.size) return;
      this.ticking = true;
      requestAnimationFrame(() => {
        this.ticking = false;
        for (const o of this.overlays.values()) this.place(o);
      });
    };
    addEventListener("scroll", schedule, { capture: true, passive: true });
    addEventListener("resize", schedule, { passive: true });
    // layout changes that fire no event (lazy content, carousels)
    setInterval(schedule, 500);
  }

  /** Find the right-clicked <img> for the context menu's srcUrl. */
  private findImage(srcUrl: string): HTMLImageElement | null {
    const t = this.lastTarget;
    if (t instanceof HTMLImageElement && (!srcUrl || t.currentSrc === srcUrl || t.src === srcUrl)) return t;
    for (const img of document.images) if (img.currentSrc === srcUrl || img.src === srcUrl) return img;
    return t instanceof HTMLImageElement ? t : null;
  }

  /** Translate the image, or remove its overlay if it is already translated. */
  toggle(srcUrl: string, targetLang: string): void {
    const img = this.findImage(srcUrl);
    if (!img) return;
    const existing = this.overlays.get(img);
    if (existing) {
      existing.dispose();
      return;
    }
    void this.translate(img, srcUrl || img.currentSrc || img.src, targetLang);
  }

  clear(): void {
    for (const o of [...this.overlays.values()]) o.dispose();
  }

  private create(img: HTMLImageElement, src: string): Overlay {
    const root = document.createElement("div");
    root.className = "img-ov";
    const stage = document.createElement("div");
    stage.className = "img-stage";
    const badge = document.createElement("div");
    badge.className = "img-badge";
    badge.innerHTML = `<span class="ico">${TRANSLATE_ICON}</span><span class="spin"></span><span class="label"></span><button class="btn peek" title="원문 보기 / 번역 보기">원문</button><button class="x" title="번역 지우기">×</button>`;
    root.append(stage, badge);
    const ro = new ResizeObserver(() => this.place(o));
    const o: Overlay = {
      img,
      src,
      root,
      stage,
      badge,
      label: badge.querySelector(".label")!,
      result: null,
      region: FULL,
      reqIds: [],
      dispose: () => {
        ro.disconnect();
        root.remove();
        this.client.cancel(o.reqIds);
        this.overlays.delete(img);
      },
    };
    badge.querySelector(".x")!.addEventListener("click", () => o.dispose());
    badge.querySelector(".peek")!.addEventListener("click", () => root.classList.toggle("peek"));
    for (const el of [badge, stage]) el.addEventListener("mousedown", (e) => e.stopPropagation());
    this.mount(root);
    this.overlays.set(img, o);
    ro.observe(img);
    this.place(o);
    return o;
  }

  /** `done`: finished without problems, so the badge can shrink out of the way. */
  private status(o: Overlay, text: string, busy: boolean, done = false): void {
    o.label.textContent = text;
    o.badge.classList.toggle("busy", busy);
    o.badge.classList.toggle("done", done);
  }

  private async translate(img: HTMLImageElement, src: string, targetLang: string): Promise<void> {
    const o = this.create(img, src);
    // one label for the whole job (reading, erasing and translating are not told apart)
    this.status(o, "번역 중…", true);
    let outcome: OcrOutcome;
    const t0 = performance.now();
    try {
      try {
        outcome = await this.client.ocr({ src, targetLang });
      } catch (e) {
        if (!/^FETCH_FAILED/.test((e as Error).message)) throw e;
        // the extension could not fetch it (blob: URL, unsupported format, …): read what the page shows instead
        const alt = await this.readFromPage(img, src);
        o.region = alt.region;
        outcome = await this.client.ocr({ dataUrl: alt.dataUrl, targetLang });
      }
    } catch (e) {
      if (this.overlays.get(img) === o) this.status(o, `읽지 못했습니다: ${(e as Error).message.replace(/^FETCH_FAILED: /, "")}`, false);
      return;
    }
    if (this.overlays.get(img) !== o) return;
    o.result = outcome.result;
    this.place(o);
    const tOcr = performance.now() - t0;
    await this.render(o, outcome, targetLang);
    // phase timings, for test/image-load.mjs
    o.root.dataset.timing = JSON.stringify({ ...JSON.parse(o.root.dataset.timing ?? "{}"), ocr: Math.round(tOcr), total: Math.round(performance.now() - t0), worker: outcome.result.timing });
  }

  /** Same-origin / blob: images are read through a canvas; anything else from a screenshot of the visible tab. */
  private async readFromPage(img: HTMLImageElement, src: string): Promise<{ dataUrl: string; region: Region }> {
    try {
      const res = await fetch(src);
      if (res.ok) {
        const blob = await res.blob();
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const fr = new FileReader();
          fr.onload = () => resolve(fr.result as string);
          fr.onerror = () => reject(fr.error);
          fr.readAsDataURL(blob);
        });
        return { dataUrl, region: FULL };
      }
    } catch {
      /* cross-origin without CORS: fall through to the screenshot */
    }
    const drawn = drawnRect(img);
    const clip = clipRect(img);
    if (!drawn) throw new Error("이미지가 화면에 보이지 않습니다");
    const x0 = Math.max(drawn.left, clip.left), y0 = Math.max(drawn.top, clip.top);
    const x1 = Math.min(drawn.right, clip.right), y1 = Math.min(drawn.bottom, clip.bottom);
    if (x1 - x0 < 8 || y1 - y0 < 8) throw new Error("이미지가 화면에 보이지 않습니다");
    console.info("[plamo] reading the image from a screenshot of the tab");
    this.hideUi(true);
    let shot: { dataUrl?: string; error?: string };
    try {
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      shot = await chrome.runtime.sendMessage({ type: "capture-visible" });
    } finally {
      this.hideUi(false);
    }
    if (!shot?.dataUrl) throw new Error(shot?.error ?? "화면을 캡처하지 못했습니다");
    const bmp = await createImageBitmap(await (await fetch(shot.dataUrl)).blob());
    const k = bmp.width / window.innerWidth;
    const c = document.createElement("canvas");
    c.width = Math.round((x1 - x0) * k);
    c.height = Math.round((y1 - y0) * k);
    c.getContext("2d")!.drawImage(bmp, x0 * k, y0 * k, c.width, c.height, 0, 0, c.width, c.height);
    bmp.close();
    const region = { x0: (x0 - drawn.left) / drawn.width, y0: (y0 - drawn.top) / drawn.height, x1: (x1 - drawn.left) / drawn.width, y1: (y1 - drawn.top) / drawn.height };
    return { dataUrl: c.toDataURL("image/png"), region };
  }

  /** Keep the overlay on the image's drawn area (or remove it when the image is gone or replaced). */
  private place(o: Overlay): void {
    const { img } = o;
    if (!img.isConnected || (o.src !== img.currentSrc && o.src !== img.src)) {
      o.dispose();
      return;
    }
    const drawn = drawnRect(img);
    const clip = clipRect(img);
    if (!drawn || !clip.width || !clip.height) {
      o.root.style.display = "none";
      return;
    }
    o.root.style.display = "";
    Object.assign(o.root.style, { left: `${clip.left}px`, top: `${clip.top}px`, width: `${clip.width}px`, height: `${clip.height}px` });
    if (!o.result) return;
    const { region: g, result } = o;
    const sx = ((g.x1 - g.x0) * drawn.width) / result.width;
    const sy = ((g.y1 - g.y0) * drawn.height) / result.height;
    const ox = drawn.left + g.x0 * drawn.width - clip.left;
    const oy = drawn.top + g.y0 * drawn.height - clip.top;
    Object.assign(o.stage.style, { width: `${result.width}px`, height: `${result.height}px`, transform: `translate(${ox}px, ${oy}px) scale(${sx}, ${sy})` });
  }

  /**
   * Translate every paragraph and wait for the erased backgrounds, then lay everything out and show it at once (no
   * text changing while it streams in).
   */
  private async render(o: Overlay, { result, morePatches }: OcrOutcome, targetLang: string): Promise<void> {
    const todo = result.paragraphs.filter((p) => !skipParagraph(p, targetLang, result.height));
    if (!todo.length) {
      this.status(o, result.paragraphs.length ? "번역할 글자가 없습니다" : "글자를 찾지 못했습니다", false);
      return;
    }
    const live = () => this.overlays.get(o.img) === o;
    // with little text the whole image is useful context; with a lot (an app screen) a fixed context helped some labels
    // but hurt others (¥ read as yen, English UI words translated), so it is left out
    const all = result.paragraphs.map((p) => p.text).join("\n");
    const context = todo.length > 1 && all.length <= 400 ? all : null;
    let failed = 0;
    const texts = Promise.all(
      todo.map((p) => {
        const id = this.client.newId();
        o.reqIds.push(id);
        return this.client
          .translate({ id, text: p.text, targetLang, context, priority: 2 })
          .catch((e: Error) => {
            if (e?.name !== "AbortError") failed++;
            return null;
          });
      }),
    );
    const t0 = performance.now();
    let tTexts = 0, tPatches = 0;
    const [translations, later] = await Promise.all([
      texts.finally(() => (tTexts = performance.now() - t0)),
      morePatches.finally(() => (tPatches = performance.now() - t0)),
    ]);
    if (!live()) return;
    const tBuild = performance.now();
    const patches = new Map<number, ErasePatch>([...result.patches, ...later].map((p) => [p.i, p]));

    // structure is inferred over all text, including text left untranslated (it still anchors rows and columns)
    const lays = inferLayout(result.paragraphs);
    const space = freeSpace(result);
    const metrics = new Map<boolean, FontMetrics>();
    const metricsFor = (bold: boolean) => {
      if (!metrics.has(bold)) metrics.set(bold, fontMetrics(targetLang, bold ? "700" : "400"));
      return metrics.get(bold)!;
    };
    const groups = new Map<number, FitGroup>();
    // built hidden (layout still runs, so text can be measured), then shown in one step
    o.stage.style.visibility = "hidden";
    todo.forEach((p, k) => {
      const i = result.paragraphs.indexOf(p);
      const lay = lays[i];
      if (!groups.has(lay.group)) groups.set(lay.group, new FitGroup());
      const view = new ParaView(p, lay, { stage: o.stage, result, targetLang, ...space[i], metrics: metricsFor, patch: patches.get(i) }, groups.get(lay.group)!);
      if (translations[k] != null) view.set(translations[k]);
      else view.fail("번역하지 못했습니다");
    });
    for (const g of groups.values()) g.run();
    o.stage.style.visibility = "";
    o.root.dataset.timing = JSON.stringify({
      paragraphs: todo.length,
      translate: Math.round(tTexts),
      inpaint: result.pending.length ? Math.round(tPatches) : 0,
      render: Math.round(performance.now() - tBuild),
    });
    this.status(o, failed ? `번역 완료 (실패 ${failed}개)` : "번역 완료", false, !failed);
  }
}

/**
 * Free space for each paragraph's translation: along the line on each side — the plain background measured on the
 * image, cut at the midpoint to the next text on the same line (both may grow into the gap, keeping 0.6 em
 * apart) and at the image edge — and below it, up to the next text underneath. Text on a busy background stays put.
 */
function freeSpace(result: OcrResult): { limits: [number, number]; below: number }[] {
  const ps = result.paragraphs;
  const ax = ps.map(axes);
  return ps.map((p, i) => {
    if (p.busy) return { limits: [0, 0], below: 0 };
    const [s, e] = ax[i].along;
    const size = p.vertical ? result.height : result.width;
    const keep = p.fontPx * 0.3;
    let before = Math.min(s, p.vertical ? p.room[2] : p.room[0]);
    let after = Math.min(size - e, p.vertical ? p.room[3] : p.room[1]);
    let below = p.vertical ? p.room[0] : p.room[3];
    const [a0, a1] = ax[i].across;
    ps.forEach((q: OcrParagraph, j) => {
      if (j === i || q.vertical !== p.vertical) return;
      const [b0, b1] = ax[j].across;
      const [qs, qe] = ax[j].along;
      if (Math.min(a1, b1) - Math.max(a0, b0) > 0) {
        if (qe <= s) before = Math.min(before, (s - qe) / 2 - keep);
        else if (qs >= e) after = Math.min(after, (qs - e) / 2 - keep);
      } else if (Math.min(e, qe) - Math.max(s, qs) > 0) {
        // text underneath (columns: to the left) that the paragraph overlaps along its line
        const gap = p.vertical ? a0 - b1 : b0 - a1;
        if (gap >= 0) below = Math.min(below, gap - keep);
      }
    });
    return { limits: [Math.max(0, before), Math.max(0, after)], below: Math.max(0, below) };
  });
}
