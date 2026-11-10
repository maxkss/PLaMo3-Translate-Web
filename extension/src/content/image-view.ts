/**
 * One translated paragraph drawn over the image, and the groups that size paragraphs together.
 *
 * A view has two parts: a background patch that hides the original ink (plus whatever area the translation needs),
 * and the translation text, placed by the paragraph's anchors from layout.ts — its start, centre or end along the
 * line and, for a single line, its ink centre across it — so text that lined up in the original lines up again.
 *
 * Sizing: every view of a fit group reports how far it can go on one line and on two lines (the largest scale ≤ 1 of
 * its style's size that fits, checked in both dimensions and against the space below). The group takes one target
 * scale — the 20th percentile of what its members can reach, kept within [0.85, 1] (or the median, if most members are
 * below 0.85) — so one long translation does not shrink a whole row, nor does a row split into big and small; a member that cannot reach it shrinks on its own. Members of a row or list keep their line count;
 * only a paragraph standing alone may take a second line.
 */

import type { ErasePatch, OcrParagraph, OcrResult, Rgb } from "../shared/common.ts";
import { axes, type ParaLayout } from "./layout.ts";

const rgb = (c: Rgb) => `rgb(${c[0]},${c[1]},${c[2]})`;
/** CSS line height of the overlay text. */
const LH = 1.2;
export const FONT_FAMILY = `system-ui, -apple-system, "Apple SD Gothic Neo", "Noto Sans KR", "Hiragino Sans", "Noto Sans JP", sans-serif`;
/** The group's common scale never goes below this: members that need less are exceptions. */
const FLOOR = 0.85;
/** Smallest scale; beyond it the text overflows its box. */
const MIN_SCALE = 0.45;

/**
 * Vertical metrics of the overlay font for the target language, per 1px of font size: where the ink centre of a
 * typical glyph sits in a line box of height LH, and how tall that ink is (canvas TextMetrics on a sample).
 */
export interface FontMetrics {
  inkCentre: number;
  inkHeight: number;
}
const SAMPLES: Record<string, string> = { ko: "한글가나다", ja: "日本語あア", "ja-easy": "日本語あア", "zh-hans": "中文汉字", "zh-hant": "中文漢字" };

export function fontMetrics(lang: string, weight: string): FontMetrics {
  const ctx = document.createElement("canvas").getContext("2d")!;
  ctx.font = `${weight} 100px ${FONT_FAMILY}`;
  const m = ctx.measureText(SAMPLES[lang] ?? "Hxgd");
  const A = m.fontBoundingBoxAscent, D = m.fontBoundingBoxDescent;
  const baseline = (LH * 100 - (A + D)) / 2 + A;
  const a = m.actualBoundingBoxAscent, d = m.actualBoundingBoxDescent;
  return { inkCentre: (baseline - (a - d) / 2) / 100, inkHeight: (a + d) / 100 };
}

/** Largest scale at which the text fits on one line (or as a block), and on two lines (0: not possible). */
interface Fit {
  one: number;
  two: number;
}

const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))] ?? 1;
};

export class FitGroup {
  readonly views: ParaView[] = [];
  /** Size and place every member (once all translations are in). */
  run(): void {
    const live = this.views.filter((v) => v.text);
    const fits = live.map((v) => v.measure());
    // the 20th percentile, but not below the floor — unless most members are below it, then the group goes down to
    // its median together rather than splitting into big and small
    const caps = fits.map((f) => Math.max(f.one, f.two));
    const target = Math.min(1, Math.max(quantile(caps, 0.2), Math.min(FLOOR, quantile(caps, 0.5))));
    // members of a row or list keep the same line count; only a paragraph standing alone may take a second line
    const alone = this.views.length === 1;
    live.forEach((v, i) => {
      const { one, two } = fits[i];
      if (one >= target || !alone) v.apply(Math.max(Math.min(one, target), MIN_SCALE), false);
      else if (two >= target) v.apply(target, true);
      // a slightly smaller single line reads better than a wrapped label
      else if (one >= 0.8 * target || one >= two) v.apply(Math.max(one, MIN_SCALE), false);
      else v.apply(two, true);
    });
  }
}

export interface ViewContext {
  stage: HTMLElement;
  result: OcrResult;
  targetLang: string;
  /** free space along the line on each side, already limited by neighbouring text: [before, after] */
  limits: [number, number];
  /** free space below the paragraph's ink (left of it for columns), up to the next text */
  below: number;
  metrics: (bold: boolean) => FontMetrics;
  /** the background with the original text erased (else a fill colour stands in for it) */
  patch?: ErasePatch;
}

export class ParaView {
  text = "";
  private readonly el: HTMLDivElement;
  private readonly span: HTMLSpanElement;
  private erased: HTMLImageElement | null = null;
  private shown = false;
  private readonly single: boolean;
  /** writing mode of the translation */
  private readonly vertical: boolean;
  /** font size at scale 1 (image px): the style size, adjusted so the target script's glyphs look as tall */
  private readonly size: number;
  private readonly pad: number;
  private readonly p: OcrParagraph;
  private readonly lay: ParaLayout;
  private readonly ctx: ViewContext;
  private readonly group: FitGroup;

  constructor(p: OcrParagraph, lay: ParaLayout, ctx: ViewContext, group: FitGroup) {
    this.p = p;
    this.lay = lay;
    this.ctx = ctx;
    this.group = group;
    group.views.push(this);
    this.vertical = p.vertical && /^(ja|ja-easy|zh-hans|zh-hant|ko)$/.test(ctx.targetLang);
    this.single = p.lines.length === 1 && p.vertical === this.vertical;
    // the source size was estimated from ink assuming CJK ink spans 0.93 em; match the target script's ink height
    const ratio = Math.min(1.15, Math.max(0.9, 0.93 / ctx.metrics(p.bold).inkHeight));
    this.size = lay.em * ratio;
    this.pad = Math.max(1, lay.em * 0.18);
    this.el = document.createElement("div");
    this.el.className = p.busy && !p.gradient && !ctx.patch ? "img-para busy" : "img-para";
    this.el.title = p.text;
    this.el.dataset.style = `noise=${p.noise} stroke=${p.ink} em=${lay.em.toFixed(1)} align=${lay.align} group=${lay.group} room=${p.room.map(Math.round)} lim=${ctx.limits.map(Math.round)} ink=${p.inkBox.map(Math.round)}`;
    Object.assign(this.el.style, {
      // a smooth gradient is repainted; on a photo, blur what is behind (mostly opaque, so the original does not show
      // through) instead of painting a solid patch
      background: ctx.patch
        ? "transparent"
        : p.gradient
          ? `linear-gradient(to ${p.gradient.to}, ${rgb(p.gradient.from)}, ${rgb(p.gradient.end)})`
          : p.busy
            ? `rgba(${p.bg[0]},${p.bg[1]},${p.bg[2]},0.8)`
            : rgb(p.bg),
      color: rgb(p.fg),
      fontWeight: p.bold ? "700" : "400",
    });
    if (ctx.patch) {
      // the erased background goes under the text; the translation may extend over untouched background around it
      this.erased = document.createElement("img");
      this.erased.src = ctx.patch.url;
      Object.assign(this.erased.style, { position: "absolute", width: `${ctx.patch.rect[2]}px`, height: `${ctx.patch.rect[3]}px`, pointerEvents: "none" });
      this.el.append(this.erased);
    }
    this.span = document.createElement("span");
    Object.assign(this.span.style, { position: "absolute", lineHeight: String(LH), writingMode: this.vertical ? "vertical-rl" : "horizontal-tb" });
    this.el.append(this.span);
    this.el.addEventListener("click", (e) => {
      // peek at the original under this paragraph
      e.stopPropagation();
      e.preventDefault();
      this.el.classList.toggle("peek");
    });
  }

  set(text: string): void {
    // text that comes out unchanged (names, numbers, "DISCOVER") keeps the original pixels
    const norm = (t: string) => t.normalize("NFKC").toLowerCase().replace(/\s+/g, "");
    if (!text.trim() || norm(this.p.text).startsWith(norm(text))) {
      if (this.shown) this.el.remove();
      this.shown = false;
      this.text = "";
    } else {
      this.text = text;
      this.span.textContent = text;
      if (!this.shown) this.ctx.stage.append(this.el);
      this.shown = true;
    }
  }

  fail(msg: string): void {
    this.el.title = `${this.p.text}\n\n번역 실패: ${msg}`;
  }

  /** Along-line interval the text may occupy, from its anchor: [min, max]. */
  private interval(): [number, number] {
    const { along } = axes(this.p);
    const [before, after] = this.ctx.limits;
    const lo = along[0] - before, hi = along[1] + after;
    const a = this.lay.anchor;
    if (this.lay.align === "start") return [a, hi];
    if (this.lay.align === "end") return [lo, a];
    const half = Math.min(a - lo, hi - a);
    return [a - half, a + half];
  }

  /** Largest scales (≤ 1) at which the current text fits on one line / two lines. */
  measure(): Fit {
    const s = this.span.style;
    const search = (fits: (k: number) => boolean) => {
      if (fits(1)) return 1;
      if (!fits(MIN_SCALE)) return 0;
      let lo = MIN_SCALE, hi = 1;
      for (let i = 0; i < 7; i++) {
        const mid = (lo + hi) / 2;
        if (fits(mid)) lo = mid;
        else hi = mid;
      }
      return lo;
    };
    if (this.single && !this.vertical) {
      const [lo, hi] = this.interval();
      const avail = hi - lo;
      s.whiteSpace = "nowrap";
      s.width = "";
      s.fontSize = `${this.size}px`;
      const one = Math.min(1, avail / Math.max(1, this.span.offsetWidth + 1));
      // a button/chip, an item of a list or text on a photo keeps its single line
      if (one >= 1 || this.lay.enclosed || this.lay.stacked || this.p.busy) return { one, two: 0 };
      // a second line grows downwards: it reaches about 1.45 em below the original ink, up to the next text
      s.whiteSpace = "normal";
      s.width = `${avail}px`;
      const two = search((k) => {
        s.fontSize = `${this.size * k}px`;
        return this.span.offsetHeight <= LH * this.size * k * 2 + 1 && this.span.scrollWidth <= avail + 1 && 1.45 * this.size * k <= this.ctx.below;
      });
      return { one, two };
    }
    // a block (several lines, or a column): fill the original area, growing into plain space after it
    const [w, h] = this.blockSize();
    s.whiteSpace = "normal";
    s.width = this.vertical ? "" : `${w}px`;
    s.height = this.vertical ? `${h}px` : "";
    const one = search((k) => {
      s.fontSize = `${this.size * k}px`;
      return this.span.offsetWidth <= w + 1 && this.span.offsetHeight <= h + 1 && this.span.scrollWidth <= w + 1;
    });
    return { one: one || MIN_SCALE, two: 0 };
  }

  /** Width and height a block may use: the ink area plus most of the plain space below (or, for columns, to the left). */
  private blockSize(): [number, number] {
    const [, , iw, ih] = this.p.inkBox;
    const room = this.p.busy ? 0 : this.ctx.below;
    if (this.vertical) return [iw + room * 0.8, ih];
    if (this.p.vertical) {
      // a column translated into a horizontal script: a wider box around the same centre
      const w = Math.max(iw, Math.min(this.ctx.result.width, ih * 0.9));
      return [w, Math.max(this.size * LH * 2, (iw * ih) / w)];
    }
    return [iw, ih + room * 0.8];
  }

  apply(scale: number, twoLine: boolean): void {
    const f = this.size * scale;
    const s = this.span.style;
    s.fontSize = `${f}px`;
    const m = this.ctx.metrics(this.p.bold);
    const [ix, iy, iw, ih] = this.p.inkBox;
    let tx: number, ty: number, tw: number, th: number;
    if (this.single && !this.vertical) {
      s.whiteSpace = twoLine ? "normal" : "nowrap";
      const [lo, hi] = this.interval();
      s.width = twoLine ? `${hi - lo}px` : "";
      s.height = "";
      s.textAlign = this.lay.align === "end" ? "right" : this.lay.align;
      tw = this.span.offsetWidth;
      th = this.span.offsetHeight;
      const a = this.lay.anchor;
      tx = this.lay.align === "start" ? a : this.lay.align === "end" ? a - tw : a - tw / 2;
      // the first line's ink centre on the original line's (row-snapped) ink centre
      ty = this.lay.mid - m.inkCentre * f;
    } else {
      const [w, h] = this.blockSize();
      s.whiteSpace = "normal";
      s.textAlign = this.lay.align === "end" ? "right" : this.lay.align;
      s.width = this.vertical ? "" : `${w}px`;
      s.height = this.vertical ? `${h}px` : "";
      tw = this.span.offsetWidth;
      th = this.span.offsetHeight;
      // the inferred anchor along the line axis (start edge, centre or end edge)
      const place = (extent: number) => (this.lay.align === "start" ? this.lay.anchor : this.lay.align === "end" ? this.lay.anchor - extent : this.lay.anchor - extent / 2);
      if (this.vertical) {
        // columns run right to left from the original right edge
        tx = ix + iw - tw;
        ty = place(th);
      } else if (this.p.vertical) {
        tx = ix + iw / 2 - tw / 2;
        ty = iy + ih / 2 - th / 2;
      } else {
        tx = place(tw);
        // first line's ink top on the original ink top
        ty = iy - (m.inkCentre - m.inkHeight / 2) * f;
      }
    }
    // the box covers the original ink (or the erased patch) and the new text
    const pd = this.pad;
    const pr = this.ctx.patch?.rect;
    let x0 = Math.max(0, Math.min(ix - pd, tx - pd * 0.5)), y0 = Math.max(0, Math.min(iy - pd, ty));
    let x1 = Math.min(this.ctx.result.width, Math.max(ix + iw + pd, tx + tw + pd * 0.5));
    let y1 = Math.min(this.ctx.result.height, Math.max(iy + ih + pd, ty + th));
    if (pr) {
      x0 = Math.min(x0, pr[0]);
      y0 = Math.min(y0, pr[1]);
      x1 = Math.max(x1, pr[0] + pr[2]);
      y1 = Math.max(y1, pr[1] + pr[3]);
      Object.assign(this.erased!.style, { left: `${pr[0] - x0}px`, top: `${pr[1] - y0}px` });
    }
    Object.assign(this.el.style, {
      left: `${x0}px`,
      top: `${y0}px`,
      width: `${x1 - x0}px`,
      height: `${y1 - y0}px`,
      // inside a pill or a rounded chip the patch must not show square corners
      borderRadius: this.lay.enclosed && !pr ? `${(y1 - y0) / 2}px` : "",
    });
    Object.assign(s, { left: `${tx - x0}px`, top: `${ty - y0}px` });
  }
}
