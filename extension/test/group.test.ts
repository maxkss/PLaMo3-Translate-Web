/** Group requests in page translation: tag numbers shifted into one request and back (content/page.ts). */
import assert from "node:assert/strict";
import { test } from "node:test";
import { packGroups, shiftTags, unitCost, unshiftTags, type PackNode } from "../src/content/page.ts";

test("shiftTags moves every numbered tag, opening, closing and self-closing", () => {
  assert.equal(shiftTags("<a1>Home</a1> <x2/> <b3>new</b3>", 4), "<a5>Home</a5> <x6/> <b7>new</b7>");
  assert.equal(shiftTags("no tags &lt;here", 3), "no tags &lt;here");
});

test("unshiftTags restores one unit's numbers and drops tags that belong to another unit", () => {
  // units: [a1] (offset 0, 1 slot) and [a1, x2] (offset 1, 2 slots) → in the request: a1 | a2, x3
  assert.equal(unshiftTags("<a2>홈</a2> <x3/>", 1, 2), "<a1>홈</a1> <x2/>");
  // the model moved the first unit's tag into the second line: it is not this unit's, so it goes
  assert.equal(unshiftTags("<a1>뉴스</a1> <a2>홈</a2>", 1, 2), "뉴스 <a1>홈</a1>");
  // tags without numbers are left for the lenient parser
  assert.equal(unshiftTags("<a>홈</a>", 1, 2), "<a>홈</a>");
});

test("a round trip through a group keeps each unit's source", () => {
  const sources = ["<a1>Home</a1>", "<a1>News</a1> <x2/>", "About <b1>us</b1>"];
  const counts = [1, 2, 1];
  let off = 0;
  const offsets = sources.map((_, i) => ((off += i ? counts[i - 1] : 0), off));
  const joined = sources.map((s, i) => shiftTags(s, offsets[i])).join("\n");
  assert.equal(joined, "<a1>Home</a1>\n<a2>News</a2> <x3/>\nAbout <b4>us</b4>");
  const back = joined.split("\n").map((l, i) => unshiftTags(l, offsets[i], counts[i]));
  assert.deepEqual(back, sources);
});

// a tree in short: a number is a unit of that cost, an array an element, { hard: [...] } a landmark
type T = number | T[] | { hard: T[] };
let ids = 0;
const node = (t: T): PackNode<string> =>
  typeof t === "number" ? { item: `u${++ids}`, cost: t } : Array.isArray(t) ? { children: t.map(node) } : { hard: true, children: t.hard.map(node) };
const groups = (t: T, budget = 100, max = 16) => ((ids = 0), packGroups({ hard: true, children: [node(t)] }, budget, max).map((g) => g.join(" ")));

test("a container that fits is one group, with the small containers next to it", () => {
  // <ul><li>×3</ul>
  assert.deepEqual(groups([10, 10, 10]), ["u1 u2 u3"]);
  // heading, paragraph, short list, paragraph
  assert.deepEqual(groups([5, 30, [10, 10, 10], 20]), ["u1 u2 u3 u4 u5 u6"]);
  // cards in column wrappers (div.col > div.card > title + price), rows of a table
  assert.deepEqual(groups([[[10, 5]], [[10, 5]], [[10, 5]]]), ["u1 u2 u3 u4 u5 u6"]);
});

test("a container above the budget is cut at its children, from its start, each fitting child whole", () => {
  // six cards of 40: two fit a group, a card is never cut
  assert.deepEqual(groups([[30, 10], [30, 10], [30, 10], [30, 10], [30, 10]]), ["u1 u2 u3 u4", "u5 u6 u7 u8", "u9 u10"]);
  // a long list: by count
  assert.deepEqual(groups([1, 1, 1, 1, 1], 100, 2), ["u1 u2", "u3 u4", "u5"]);
  // a list that fits stays whole when the text around it does not fit with it: paragraphs | list | paragraph
  assert.deepEqual(groups([40, 40, [20, 20, 20], 50]), ["u1 u2", "u3 u4 u5", "u6"]);
  // and it takes the short text after it along
  assert.deepEqual(groups([40, 40, [20, 20, 20], 40]), ["u1 u2", "u3 u4 u5 u6"]);
});

test("nothing is packed across a container that was cut, a unit above the budget, or a landmark", () => {
  // the tail of a cut list does not join the paragraph after it
  assert.deepEqual(groups([[60, 60, 10], 10]), ["u1", "u2 u3", "u4"]);
  // a long paragraph between two short ones
  assert.deepEqual(groups([10, 500, 10]), ["u1", "u2", "u3"]);
  // a small nav next to a small article
  assert.deepEqual(groups([{ hard: [5, 5] }, [10, 10]]), ["u1 u2", "u3 u4"]);
  assert.deepEqual(groups([10, { hard: [5, 5] }, 10]), ["u1", "u2 u3", "u4"]);
});

test("unitCost follows the tokenizer roughly: Latin is cheap, CJK and tags are not", () => {
  const en = unitCost("The WebGPU API enables web developers to use the underlying system's GPU.");
  const ja = unitCost("WebGPUは、ウェブブラウザ上でGPUを利用した高性能な計算を可能にするAPIである。");
  assert.ok(en > 10 && en < 25, String(en));
  assert.ok(ja > 15 && ja < 35, String(ja));
  assert.ok(Math.abs(unitCost("<a1>Home</a1>") - (1 + 8 + 4 * 0.18)) < 1e-9);
  assert.ok(unitCost("2026-11-10 12:34:56") > unitCost("abcdefghij klmnopqr"));
});
