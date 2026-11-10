/** Which OCR paragraphs image translation leaves alone (shared by the OCR worker, which erases only the others, and
 * the content script, which translates only the others). */

import type { OcrParagraph } from "./common.ts";

/**
 * Paragraphs left alone: no letters, already in the target language's script, or too little to be text rather than
 * an icon read as a letter (a magnifier as "Q", a coin as "币").
 */
export function skipParagraph(p: Pick<OcrParagraph, "text" | "lines" | "inkBox" | "vertical">, lang: string, imageHeight: number): boolean {
  // a line cut off by the image's top or bottom edge cannot be measured (or read) reliably
  if (!p.vertical && (p.inkBox[1] <= 1 || p.inkBox[1] + p.inkBox[3] >= imageHeight - 1)) return true;
  const text = p.text;
  const letters = text.match(/\p{L}/gu) ?? [];
  if (letters.length <= 1) return true;
  const latin = letters.every((c) => /\p{Script=Latin}/u.test(c));
  if (latin && letters.length <= 2) return true;
  if (letters.length <= 3 && Math.min(...p.lines.map((l) => l.score)) < 0.8) return true;
  const share = (re: RegExp) => letters.filter((c) => re.test(c)).length / letters.length;
  if (lang === "ko") return share(/\p{Script=Hangul}/u) > 0.5;
  if (lang === "ja" || lang === "ja-easy") return share(/[\p{Script=Hiragana}\p{Script=Katakana}]/u) > 0.2;
  if (lang === "en") return latin;
  return false;
}
