/** Group requests in page translation: tag numbers shifted into one request and back (content/page.ts). */
import assert from "node:assert/strict";
import { test } from "node:test";
import { shape, shiftTags, unshiftTags } from "../src/content/page.ts";

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

test("shape ignores classes that number or mark one item", () => {
  const el = (tag: string, cls: string) => ({ tagName: tag, classList: cls.split(" ").filter(Boolean) }) as unknown as Element;
  assert.equal(shape(el("TR", "athing submission")), "TR.athing.submission");
  assert.equal(shape(el("LI", "card card-3 is-active")), shape(el("LI", "card card-7")));
  assert.equal(shape(el("DIV", "")), "DIV");
  assert.notEqual(shape(el("DIV", "price")), shape(el("DIV", "title")));
});
