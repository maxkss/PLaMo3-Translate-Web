import assert from "node:assert/strict";
import { test } from "node:test";
import { markGlyphs, pushPull } from "../src/ocr/erase.ts";
import { glyphGaps, inkBand, type Pixels } from "../src/ocr/ocr.ts";

type Rgb = [number, number, number];

/** A w × h image filled by `at(x, y)`. */
function image(w: number, h: number, at: (x: number, y: number) => Rgb): Pixels {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const [r, g, b] = at(x, y);
      data.set([r, g, b, 255], (y * w + x) * 4);
    }
  }
  return { width: w, height: h, data };
}

/** "Letters": vertical strokes 4 px wide every 12 px between x0 and x1, from y0 to y1. */
const stroke = (x: number, y: number, x0: number, x1: number, y0: number, y1: number) => x >= x0 && x < x1 && y >= y0 && y < y1 && (x - x0) % 12 < 4;
const WHITE: Rgb = [255, 255, 255], BLACK: Rgb = [0, 0, 0];

test("markGlyphs: white letters on a black bar on a white page — the letters, not the page", () => {
  // bar 20..180 × 30..70, letters inside it
  const img = image(200, 100, (x, y) => (stroke(x, y, 30, 170, 38, 62) ? WHITE : x >= 20 && x < 180 && y >= 30 && y < 70 ? BLACK : WHITE));
  const mask = new Uint8Array(200 * 100);
  // a line box that reaches past the bar has the page's white in it, running out to the frame: which white is letters
  // cannot be told from colour alone, and the whole box is erased instead (the known weak case)
  assert.equal(markGlyphs(img, mask, [14, 26, 172, 48], 30, false), null);
  assert.equal(mask.reduce((a, b) => a + b, 0), 0);
  // a line box inside the bar
  assert.deepEqual(markGlyphs(img, mask, [26, 34, 148, 32], 30, false), [255, 255, 255], "and their colour");
  let letters = 0, hit = 0, page = 0, bar = 0;
  for (let y = 0; y < 100; y++) {
    for (let x = 0; x < 200; x++) {
      const m = mask[y * 200 + x];
      if (stroke(x, y, 30, 170, 38, 62)) letters++, (hit += m);
      else if (x < 16 || x >= 184 || y < 26 || y >= 74) page += m;
      else if (x >= 20 && x < 180 && y >= 30 && y < 70) bar += m;
    }
  }
  assert.equal(hit, letters, "every letter pixel is in the mask");
  assert.equal(page, 0, "the page beyond the bar is not");
  assert.ok(bar < 3 * letters, "of the bar only what the mask's growth takes");
});

test("markGlyphs: letters over a background of two colours", () => {
  const img = image(240, 80, (x, y) => (stroke(x, y, 20, 220, 26, 54) ? WHITE : x < 120 ? [30, 60, 120] : [120, 90, 40]));
  const mask = new Uint8Array(240 * 80);
  assert.deepEqual(markGlyphs(img, mask, [14, 20, 212, 40], 34, false), [255, 255, 255]);
  for (let y = 26; y < 54; y++) for (let x = 20; x < 220; x++) if (stroke(x, y, 20, 220, 26, 54)) assert.equal(mask[y * 240 + x], 1);
  assert.equal(mask[40 * 240 + 2], 0);
});

test("markGlyphs: declines when the text's colour also runs through the background among the letters", () => {
  // a white band across the whole image behind the letters: which white is text cannot be told
  const img = image(240, 80, (x, y) => (stroke(x, y, 20, 220, 26, 54) || (y >= 36 && y < 46) ? WHITE : [40, 40, 60]));
  const mask = new Uint8Array(240 * 80);
  assert.equal(markGlyphs(img, mask, [14, 20, 212, 40], 34, false), null);
  assert.equal(mask.reduce((a, b) => a + b, 0), 0, "and marks nothing");
});

test("pushPull: a hole in a gradient is filled with the gradient, without blocks", () => {
  const at = (x: number, y: number): Rgb => [40 + x, 200 - x, 90 + y];
  const img = image(160, 120, at);
  const mask = new Uint8Array(160 * 120);
  for (let y = 40; y < 80; y++) for (let x = 50; x < 110; x++) (mask[y * 160 + x] = 1), img.data.set([255, 0, 255, 255], (y * 160 + x) * 4);
  pushPull(img, mask, [46, 36, 68, 48], 24);
  let worst = 0;
  for (let y = 40; y < 80; y++) {
    for (let x = 50; x < 110; x++) {
      const t = at(x, y);
      for (let k = 0; k < 3; k++) worst = Math.max(worst, Math.abs(img.data[(y * 160 + x) * 4 + k] - t[k]));
    }
  }
  assert.ok(worst <= 12, `largest error ${worst}`);
});

test("glyphGaps: the empty stretches between glyphs", () => {
  const img = image(200, 40, (x, y) => (stroke(x, y, 20, 180, 10, 30) ? BLACK : WHITE));
  const gaps = glyphGaps(img, false, [10, 30], [20, 180], { bg: WHITE, fg: BLACK });
  assert.ok(gaps.length >= 12 && gaps.every((g) => g === 8), JSON.stringify(gaps));
});

test("inkBand: one ascender in a long line counts, the next line's ink does not", () => {
  // body rows 20..32 full of strokes; a single ascender stroke up to row 12; a descender to row 37; the next line
  // starts at row 41 inside the same detector box
  const img = image(600, 60, (x, y) => {
    if (stroke(x, y, 10, 590, 20, 32)) return BLACK;
    if (x >= 300 && x < 303 && y >= 12 && y < 20) return BLACK;
    if (x >= 400 && x < 403 && y >= 32 && y < 37) return BLACK;
    if (stroke(x, y, 10, 590, 41, 60)) return BLACK;
    return WHITE;
  });
  const [a, b] = inkBand(img, [5, 6, 590, 46], false, { bg: WHITE, fg: BLACK });
  assert.equal(a, 12, "from the top of the ascender");
  assert.equal(b, 37, "to the bottom of the descender, not into the next line");
});
