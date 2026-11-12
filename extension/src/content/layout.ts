/**
 * Typographic structure of the OCR'd text, inferred once per image before anything is translated, so translations
 * keep the original's consistency (test/typo-eval.mjs measures it):
 *  - styles: paragraphs of the same orientation, weight and (within 10%) size form one text style; every member gets
 *    the style's size (a weighted median, so detector noise does not make equal text unequal);
 *  - alignment: each paragraph is start-, centre- or end-aligned, decided by which of its edges/centre line up with
 *    other text (its own lines, or other paragraphs on other lines) — then the anchor is snapped to the shared line;
 *  - rows: same-style single lines whose ink centres line up share one centre line;
 *  - fit groups: same-style paragraphs in one row or one aligned stack share one fitted size (shrinking together),
 *    so a long translation does not make one label of a row smaller than its neighbours.
 * Everything is measured on the glyph ink (inkBox), not on the detector boxes, whose padding depends on their shape.
 */

import type { OcrParagraph } from "../shared/common.ts";

export type Align = "start" | "center" | "end";

export interface ParaLayout {
  /** style id and its font size (image px) */
  style: number;
  em: number;
  align: Align;
  /** along the line axis (x, or y for columns): the start edge, centre or end edge of the text, snapped to peers */
  anchor: number;
  /** across the line axis (y, or x for columns): the ink centre of a single line, snapped to its row */
  mid: number;
  /** paragraphs that share one fitted size */
  group: number;
  /** tightly enclosed with even margins (a button, chip or tag): its patch gets round ends, it never wraps */
  enclosed: boolean;
  /** part of a stack (a list): an extra line would break its rhythm */
  stacked: boolean;
}

/** Along-axis [start, end] and across-axis [start, end] of the paragraph's ink. */
export function axes(p: OcrParagraph): { along: [number, number]; across: [number, number] } {
  const [x, y, w, h] = p.inkBox;
  return p.vertical ? { along: [y, y + h], across: [x, x + w] } : { along: [x, x + w], across: [y, y + h] };
}

class Union {
  private readonly parent: number[];
  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
  }
  find(i: number): number {
    return this.parent[i] === i ? i : (this.parent[i] = this.find(this.parent[i]));
  }
  join(a: number, b: number): void {
    this.parent[this.find(a)] = this.find(b);
  }
  groups(): number[][] {
    const m = new Map<number, number[]>();
    this.parent.forEach((_, i) => m.set(this.find(i), [...(m.get(this.find(i)) ?? []), i]));
    return [...m.values()];
  }
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return s.length ? (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2 : 0;
};

function weightedMedian(items: { v: number; w: number }[]): number {
  const s = [...items].sort((a, b) => a.v - b.v);
  const half = s.reduce((a, b) => a + b.w, 0) / 2;
  let acc = 0;
  for (const it of s) if ((acc += it.w) >= half) return it.v;
  return s.at(-1)?.v ?? 0;
}

/** Tolerance for "the same line/edge": two pixels, or 20% of the font size (glyph side bearings differ). */
const tol = (em: number) => Math.max(2, 0.2 * em);

export function inferLayout(ps: OcrParagraph[], size?: { width: number; height: number }): ParaLayout[] {
  const n = ps.length;
  // ---- 1. styles: sizes sorted per (orientation, weight); a style spans at most 12% from its smallest to its largest
  // member, so close sizes cannot chain a heading into body text. Its size is the members' weighted median.
  const styleOf = new Array<number>(n);
  const styles: { members: number[]; size: number }[] = [];
  const weight = (i: number) => Math.min(10, ps[i].text.length); // longer text: a steadier size estimate
  const kinds = new Map<string, number[]>();
  ps.forEach((p, i) => {
    // text on a photo or inside an icon is measured unreliably: it keeps a style of its own
    const key = p.busy ? `busy${i}` : `${p.vertical}/${p.bold}`;
    kinds.set(key, [...(kinds.get(key) ?? []), i]);
  });
  for (const members of kinds.values()) {
    members.sort((a, b) => ps[a].fontPx - ps[b].fontPx);
    let cur: number[] = [];
    const flush = () => {
      if (!cur.length) return;
      const size = weightedMedian(cur.map((j) => ({ v: ps[j].fontPx, w: weight(j) })));
      for (const j of cur) styleOf[j] = styles.length;
      styles.push({ members: cur, size });
      cur = [];
    };
    for (const i of members) {
      if (cur.length && ps[i].fontPx > ps[cur[0]].fontPx * 1.12) flush();
      cur.push(i);
    }
    flush();
  }
  const em = (i: number) => styles[styleOf[i]].size;

  // ---- 2. alignment
  const ax = ps.map(axes);
  const value = (i: number, k: Align) => {
    const [s, e] = ax[i].along;
    return k === "start" ? s : k === "end" ? e : (s + e) / 2;
  };
  const lineValue = (p: OcrParagraph, l: OcrParagraph["lines"][number], k: Align) => {
    const [s, e] = l.span ?? (p.vertical ? [l.box[1], l.box[1] + l.box[3]] : [l.box[0], l.box[0] + l.box[2]]);
    return k === "start" ? s : k === "end" ? e : (s + e) / 2;
  };
  const KINDS: Align[] = ["start", "center", "end"];
  /**
   * Gap between two paragraphs across their lines (vertical distance for horizontal text), or -1 if they share a line.
   * Alignment evidence and anchor snapping only reach paragraphs within LOCAL em of each other, so separate cards,
   * columns or panels do not pull on each other.
   */
  const acrossGap = (i: number, j: number) => {
    const [a0, a1] = ax[i].across, [b0, b1] = ax[j].across;
    return Math.min(a1, b1) - Math.max(a0, b0) > 0 ? -1 : Math.max(b0 - a1, a0 - b1);
  };
  const LOCAL = 4;
  // single lines of the same style on the same centre line
  const mids = ax.map((a) => (a.across[0] + a.across[1]) / 2);
  const sameRow = (i: number, j: number) =>
    styleOf[i] === styleOf[j] && ps[i].lines.length === 1 && ps[j].lines.length === 1 && Math.abs(mids[i] - mids[j]) <= Math.max(2, 0.2 * em(i));
  const rowPeers = ps.map((_, i) => ps.map((_, j) => j).filter((j) => j === i || sameRow(i, j)));
  const align: Align[] = ps.map((p, i) => {
    const t = tol(em(i));
    const support = new Map<Align, number>(KINDS.map((k) => [k, 0]));
    // the paragraph's own lines are the strongest evidence
    if (p.lines.length > 1) {
      // a last line may be short (left/justified text) and a first line indented: starts are compared without the
      // first line, ends without the last; text whose starts line up is read as start (also when justified)
      const spread = (k: Align, ls: typeof p.lines) => {
        const vs = ls.map((l) => lineValue(p, l, k));
        return Math.max(...vs) - Math.min(...vs);
      };
      const many = p.lines.length > 2;
      const w = 2 * (p.lines.length - 1);
      const start = spread("start", many ? p.lines.slice(1) : p.lines) <= t * 1.5;
      const centre = spread("center", p.lines) <= t * 1.5;
      // lines of about one width line up at their starts and at their centres alike: they tell nothing, and what is
      // around the paragraph decides (a start-aligned reading would otherwise win every centred block of even lines)
      if (start && centre) {
        support.set("start", support.get("start")! + 1);
        support.set("center", support.get("center")! + 1);
        // a block sitting on the image's centre line was most likely centred there
        if (size && !p.vertical && Math.abs(value(i, "center") - size.width / 2) <= t) support.set("center", support.get("center")! + 1);
      } else if (start) support.set("start", support.get("start")! + w);
      else if (many && spread("end", p.lines.slice(0, -1)) <= t * 1.5) support.set("end", support.get("end")! + w);
      else if (centre) support.set("center", support.get("center")! + w);
    }
    // other paragraphs on other lines (stacked above/below, not beside) whose edge or centre lines up; one such
    // match can be a coincidence, so a kind needs at least two
    // (a list continues the stack: each step to the next aligned item must be local, and every item must line up
    // with this paragraph itself, so the line cannot drift)
    const stacked = new Map<Align, number>(KINDS.map((k) => [k, 0]));
    for (const k of KINDS) {
      const lined = (j: number) => j !== i && ps[j].vertical === p.vertical && Math.abs(value(i, k) - value(j, k)) <= Math.min(t, tol(em(j)));
      const seen = new Set([i]);
      const queue = [i];
      while (queue.length) {
        const a = queue.pop()!;
        for (let j = 0; j < n; j++) {
          if (seen.has(j) || !lined(j)) continue;
          const gap = acrossGap(a, j);
          if (gap < 0 || gap > LOCAL * Math.max(em(a), em(j))) continue;
          seen.add(j);
          queue.push(j);
        }
      }
      stacked.set(k, seen.size - 1);
    }
    for (const k of KINDS) if (stacked.get(k)! >= 2) support.set(k, support.get(k)! + stacked.get(k)!);
    // repeated items in a row (a grid of icon labels, a tab bar): the edge or centre that is evenly spaced
    const row = rowPeers[i];
    if (row.length >= 3) {
      const evenness = (k: Align) => {
        const vs = row.map((j) => value(j, k)).sort((a, b) => a - b);
        const steps = vs.slice(1).map((v, m) => v - vs[m]);
        return Math.max(...steps) - Math.min(...steps);
      };
      const ev = KINDS.map(evenness);
      const bestEv = Math.min(...ev);
      // one kind clearly more regular than the others; when all are (items of about the same width), evenly spaced
      // items are the cells of a grid or tab bar, whose text is centred
      const regular = KINDS.filter((_, m) => ev[m] <= bestEv + t);
      if (regular.length === 1) support.set(regular[0], support.get(regular[0])! + row.length - 1);
      else if (regular.length === 3 && bestEv <= 2 * t) support.set("center", support.get("center")! + row.length - 1);
    }
    // text tightly enclosed with even margins (a button, a chip, a tag): centred
    const [rl, rr] = p.vertical ? [p.room[2], p.room[3]] : [p.room[0], p.room[1]];
    if (rl < em(i) && rr < em(i) && Math.abs(rl - rr) <= 0.35 * (rl + rr) + 2) support.set("center", support.get("center")! + 3);
    // as much free background on one side as on the other, up to whatever ends it (a panel's edges, the image's): set
    // in the middle of its space. Weak evidence — two aligned neighbours outweigh it.
    // (text that fills its space from edge to edge is not placed by its margins at all)
    else if (!p.busy && Math.min(rl, rr) >= em(i) && rl + rr >= 0.5 * (ax[i].along[1] - ax[i].along[0]) && Math.abs(rl - rr) <= Math.max(t, 0.04 * (rl + rr))) support.set("center", support.get("center")! + 1.5);
    const best = Math.max(...support.values());
    const winners = KINDS.filter((k) => support.get(k) === best);
    if (best > 0 && winners.length === 1) return winners[0];
    // even lines, nothing else to go by: where the block sits in the image
    if (winners.includes("start") && winners.includes("center") && p.lines.length > 1 && size && !p.vertical && Math.abs(value(i, "center") - size.width / 2) <= t) return "center";
    if (best > 0 && winners.includes("start")) return "start";
    if (best > 0 && winners.includes("center")) return "center";
    // no evidence: start (reading direction)
    return "start";
  });
  // snap anchors: nearby paragraphs with the same alignment whose anchors line up share the median anchor; a member
  // that would move more than its tolerance (a chain drifted) keeps its own
  const anchors = new Union(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (ps[i].vertical !== ps[j].vertical || align[i] !== align[j]) continue;
      const gap = acrossGap(i, j);
      if (gap < 0 || gap > LOCAL * Math.max(em(i), em(j))) continue;
      if (Math.abs(value(i, align[i]) - value(j, align[j])) <= Math.min(tol(em(i)), tol(em(j)))) anchors.join(i, j);
    }
  }
  const anchor = new Array<number>(n);
  for (const g of anchors.groups()) {
    const v = median(g.map((i) => value(i, align[i])));
    for (const i of g) anchor[i] = Math.abs(value(i, align[i]) - v) <= tol(em(i)) ? v : value(i, align[i]);
  }

  // ---- 3. rows: same-style single lines with the same centre line
  const rows = new Union(n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (sameRow(i, j)) rows.join(i, j);
  const mid = new Array<number>(n);
  for (const g of rows.groups()) {
    const v = median(g.map((i) => mids[i]));
    for (const i of g) mid[i] = Math.abs(mids[i] - v) <= Math.max(2, 0.2 * em(i)) ? v : mids[i];
  }

  // ---- 4. fit groups: same style and (same row or same aligned stack)
  const fit = new Union(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (styleOf[i] !== styleOf[j]) continue;
      if (rows.find(i) === rows.find(j) || anchors.find(i) === anchors.find(j)) fit.join(i, j);
    }
  }
  const enclosed = ps.map((p, i) => {
    if (p.busy) return false; // its room is measured against a photo or gradient, not a container
    const [rl, rr] = p.vertical ? [p.room[2], p.room[3]] : [p.room[0], p.room[1]];
    return rl < em(i) && rr < em(i) && Math.abs(rl - rr) <= 0.35 * (rl + rr) + 2;
  });
  const stackSize = new Map<number, number>();
  for (let i = 0; i < n; i++) stackSize.set(anchors.find(i), (stackSize.get(anchors.find(i)) ?? 0) + 1);
  return ps.map((_, i) => ({
    style: styleOf[i],
    em: em(i),
    align: align[i],
    anchor: anchor[i],
    mid: mid[i],
    group: fit.find(i),
    enclosed: enclosed[i],
    stacked: stackSize.get(anchors.find(i))! > 1 && !ps.some((_, j) => j !== i && anchors.find(j) === anchors.find(i) && rows.find(j) === rows.find(i)),
  }));
}
