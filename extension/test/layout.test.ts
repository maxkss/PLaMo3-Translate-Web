import assert from "node:assert/strict";
import { test } from "node:test";
import { inferLayout } from "../src/content/layout.ts";
import type { OcrParagraph } from "../src/shared/common.ts";

/** A one-line horizontal paragraph whose ink is (x, y, w, h); font size from the ink height. */
function para(x: number, y: number, w: number, h: number, opts: Partial<OcrParagraph> = {}): OcrParagraph {
  const box: [number, number, number, number] = [x - 4, y - 6, w + 8, h + 12];
  return {
    box,
    lines: [{ box, text: "文字", score: 1, ink: [y, y + h], span: [x, x + w] }],
    text: "文字",
    vertical: false,
    lineSize: h + 12,
    fontPx: h / 0.93,
    bg: [255, 255, 255],
    fg: [0, 0, 0],
    busy: false,
    bold: false,
    // open space, more of it to the right (as much on both sides would read as centred)
    room: [200, 600, 200, 200],
    inkBox: [x, y, w, h],
    ...opts,
  };
}

test("labels centred under a grid of icons: centre alignment, one size, one row line", () => {
  // two rows of five labels of different widths, centred in 150px cells
  const ps: OcrParagraph[] = [];
  const widths = [48, 96, 72, 120, 48];
  for (const y of [100, 220]) widths.forEach((w, c) => ps.push(para(75 + 150 * c - w / 2, y + (c % 2), w, 24 + (c % 3) * 0.5)));
  const lay = inferLayout(ps);
  assert.ok(lay.every((l) => l.align === "center"), JSON.stringify(lay.map((l) => l.align)));
  assert.equal(new Set(lay.map((l) => l.style)).size, 1);
  assert.equal(new Set(lay.slice(0, 5).map((l) => l.mid)).size, 1, "first row shares one centre line");
});

test("a left-aligned stack and right-aligned values in a settings list", () => {
  const ps: OcrParagraph[] = [];
  for (let r = 0; r < 4; r++) {
    ps.push(para(30 + (r % 2), 40 + 80 * r, 50 + 20 * r, 26));
    ps.push(para(690 - (60 + 15 * r) + (r % 2), 42 + 80 * r, 60 + 15 * r, 22, { fg: [150, 150, 150] }));
  }
  const lay = inferLayout(ps);
  for (let r = 0; r < 4; r++) {
    assert.equal(lay[2 * r].align, "start");
    assert.equal(lay[2 * r + 1].align, "end");
  }
  // anchors snapped to one edge
  assert.equal(new Set(lay.filter((_, i) => i % 2 === 0).map((l) => l.anchor)).size, 1);
  assert.equal(new Set(lay.filter((_, i) => i % 2 === 1).map((l) => l.anchor)).size, 1);
});

test("styles do not chain: sizes 20, 21.5, 23, 24.5 do not all become one style", () => {
  const ps = [20, 21.5, 23, 24.5].map((h, i) => para(10, 10 + 60 * i, 100, h * 0.93));
  const lay = inferLayout(ps);
  assert.ok(new Set(lay.map((l) => l.style)).size >= 2);
  for (const l of lay) assert.ok(Math.abs(l.em - ps[lay.indexOf(l)].fontPx) <= 0.1 * l.em);
});

test("busy text keeps its own style and a single coincidence does not decide alignment", () => {
  const ps = [para(100, 10, 80, 24), para(100, 200, 40, 24, { busy: true }), para(300, 400, 60, 24)];
  const lay = inferLayout(ps);
  assert.notEqual(lay[1].style, lay[0].style);
  // paragraph 0 starts where the busy one starts, but one match is not evidence; with open space it stays "start"
  assert.equal(lay[0].align, "start");
});

test("text tightly enclosed with even margins (a button) is centred", () => {
  const ps = [para(100, 10, 80, 24, { room: [10, 11, 6, 6] })];
  assert.equal(inferLayout(ps)[0].align, "center");
});

test("a line with as much free space on its left as on its right is centred; aligned neighbours outweigh that", () => {
  // a headline in the middle of a banner
  assert.equal(inferLayout([para(400, 50, 200, 40, { room: [400, 404, 50, 300] })])[0].align, "center");
  // three lines starting at one x: a left-aligned stack, whatever the room of each
  const ps = [0, 1, 2].map((r) => para(100, 40 + 60 * r, 120 + 40 * r, 24, { room: [100, 100 + 2 * r, 20, 20] }));
  assert.deepEqual(inferLayout(ps).map((l) => l.align), ["start", "start", "start"]);
});

test("a block of even lines on the image's centre line is centred, not start-aligned", () => {
  // four lines of nearly one width: their starts line up as well as their centres do
  const lines = [0, 1, 2, 3].map((r) => ({ box: [300 - r, 100 + 40 * r, 200 + 2 * r, 36] as [number, number, number, number], text: "TEXT", score: 1, ink: [104 + 40 * r, 132 + 40 * r] as [number, number], span: [302 - r, 498 + r] as [number, number] }));
  const p = para(299, 104, 202, 148, { lines, room: [150, 260, 50, 50] });
  assert.equal(inferLayout([p], { width: 800, height: 600 })[0].align, "center");
  assert.equal(inferLayout([p], { width: 1600, height: 600 })[0].align, "start");
});
