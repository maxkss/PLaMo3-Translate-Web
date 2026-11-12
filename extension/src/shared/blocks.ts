/**
 * Phrases set on several lines of their own — "NO / SMOKING" on a sign, "Smart app / to / lose weight" on a poster, the
 * columns of a speech bubble read as separate paragraphs. Each line is its own paragraph to the OCR (another size,
 * another colour, a gap), and translated alone it comes out word by word ("아니요 / 흡연"). A block is such a stack:
 * it is translated as one text, and the translation is laid back over the block's lines.
 *
 * Shared by the OCR worker (which erases what will be replaced) and the content script (which translates and draws).
 */

import type { OcrParagraph } from "./common.ts";
import { skipParagraph } from "./skip.ts";

type Para = Pick<OcrParagraph, "text" | "lines" | "inkBox" | "vertical" | "fontPx">;

/** Longest paragraph that can be a line of a block, and the most lines and characters of one block. */
const MAX_PART = 40, MAX_PARTS = 6, MAX_CHARS = 140;

const letters = (t: string) => (t.match(/\p{L}/gu) ?? []).length;

export interface Blocks {
  /** paragraphs of each block (two or more), in reading order */
  blocks: number[][];
  /** which paragraphs are translated: those not skipped on their own, and every line of a block */
  translate: boolean[];
}

/**
 * The blocks of an image. Two short paragraphs belong together when one lies directly under the other (columns: beside
 * it) — within 0.7 em across the lines, sharing at least half of the shorter one's extent along them. A stack of more
 * than six is a list or a menu, not a phrase, and stays as it is.
 */
export function textBlocks(ps: Para[], lang: string, imageHeight: number): Blocks {
  const n = ps.length;
  const own = ps.map((p) => !skipParagraph(p, lang, imageHeight));
  // a line of a phrase may be too short to be translated alone ("NO", "to"); it still has to be text of another script
  const part = (i: number) => {
    const p = ps[i];
    if (p.text.length > MAX_PART || p.lines.length > 2 || letters(p.text) < 2) return false;
    return own[i] || !skipParagraph({ ...p, text: p.text + p.text + p.text, lines: p.lines.map((l) => ({ ...l, score: 1 })) }, lang, imageHeight);
  };
  const ok = ps.map((_, i) => part(i));
  const ext = (p: Para) => (p.vertical ? { a0: p.inkBox[1], a1: p.inkBox[1] + p.inkBox[3], c0: p.inkBox[0], c1: p.inkBox[0] + p.inkBox[2] } : { a0: p.inkBox[0], a1: p.inkBox[0] + p.inkBox[2], c0: p.inkBox[1], c1: p.inkBox[1] + p.inkBox[3] });
  const parent = ps.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (!ok[i] || !ok[j] || ps[i].vertical !== ps[j].vertical) continue;
      const a = ext(ps[i]), b = ext(ps[j]);
      const em = Math.max(ps[i].fontPx, ps[j].fontPx);
      const gap = Math.max(a.c0, b.c0) - Math.min(a.c1, b.c1);
      if (gap > 0.7 * em || gap < -0.25 * em) continue;
      const shared = Math.min(a.a1, b.a1) - Math.max(a.a0, b.a0);
      if (shared < 0.5 * Math.min(a.a1 - a.a0, b.a1 - b.a0)) continue;
      parent[find(i)] = find(j);
    }
  }
  const groups = new Map<number, number[]>();
  ps.forEach((_, i) => ok[i] && groups.set(find(i), [...(groups.get(find(i)) ?? []), i]));
  const blocks: number[][] = [];
  const translate = [...own];
  for (const g of groups.values()) {
    if (g.length < 2 || g.length > MAX_PARTS || g.reduce((a, i) => a + ps[i].text.length, 0) > MAX_CHARS) continue;
    // nothing in it would be translated anyway: leave it
    if (!g.some((i) => own[i])) continue;
    // reading order: top to bottom; columns right to left
    g.sort((i, j) => (ps[i].vertical ? ps[j].inkBox[0] - ps[i].inkBox[0] : ps[i].inkBox[1] - ps[j].inkBox[1]));
    blocks.push(g);
    for (const i of g) translate[i] = true;
  }
  return { blocks, translate };
}

const norm = (t: string) => t.normalize("NFKC").toLowerCase().replace(/[\s\p{P}]+/gu, "");

function editDistance(a: string, b: string): number {
  const A = [...a], B = [...b];
  let prev = Array.from({ length: B.length + 1 }, (_, j) => j);
  for (let i = 1; i <= A.length; i++) {
    const cur = [i];
    for (let j = 1; j <= B.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] === B[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[B.length];
}

/**
 * Whether the lines of a block are a list (each translated on its own says the same as all of them translated
 * together) rather than one phrase: `byLine` is the translation of the lines given one per line, `whole` that of the
 * lines joined into one text.
 */
export function isList(byLine: string[], whole: string): boolean {
  const a = norm(byLine.join("")), b = norm(whole);
  return editDistance(a, b) <= 0.3 * Math.max(1, a.length, b.length);
}

/** Rough set width of a text in em: full-width scripts 1, Latin letters and digits about half. */
const widthEm = (t: string) => [...t].reduce((a, c) => a + (/[ᄀ-ᇿ⺀-鿿가-힯豈-﫿＀-￯]/.test(c) ? 1 : c === " " ? 0.3 : 0.55), 0);

/**
 * Lay a translation over the lines of a block: `capacity` is each line's room in em (its width over its font size).
 * The words keep their order and fill the lines in proportion to their room; with fewer words than lines, the lines
 * with the most room take them and the rest stay empty. Text without spaces (Chinese, Japanese) is divided by
 * characters. Returns one string per line.
 */
export function reflow(text: string, capacity: number[]): string[] {
  const n = capacity.length;
  // words, also for a single one; characters only for scripts written without spaces
  const spaced = /\s/.test(text.trim()) || !/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u.test(text);
  const words = spaced ? text.trim().split(/\s+/) : [...text.trim()];
  const join = (ws: string[]) => ws.join(spaced ? " " : "");
  const out: string[][] = capacity.map(() => []);
  if (words.length <= n) {
    // one word per line, on the lines with the most room, in reading order
    const pick = capacity.map((c, i) => [c, i]).sort((a, b) => b[0] - a[0]).slice(0, words.length).map((x) => x[1]).sort((a, b) => a - b);
    pick.forEach((line, k) => out[line].push(words[k]));
    return out.map(join);
  }
  const total = words.reduce((a, w) => a + widthEm(w), 0);
  const room = capacity.reduce((a, b) => a + b, 0) || 1;
  // the share of the text that should have been placed by the end of each line
  let line = 0, placed = 0, quota = (capacity[0] / room) * total;
  words.forEach((w, k) => {
    const wd = widthEm(w);
    // on to the next line when more than half of this word would run past the line's share (and words remain for it)
    while (line < n - 1 && out[line].length && placed + wd / 2 > quota) quota += (capacity[++line] / room) * total;
    // every remaining line gets at least one of the remaining words
    while (line < n - 1 && words.length - k <= n - 1 - line && out[line].length) quota += (capacity[++line] / room) * total;
    out[line].push(w);
    placed += wd;
  });
  return out.map(join);
}
