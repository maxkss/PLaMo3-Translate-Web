/** Parsing of translated text with numbered inline tags (content/page.ts), including the slips a small model makes. */
import assert from "node:assert/strict";
import { test } from "node:test";
import { parseTagged } from "../src/content/page.ts";

type Spec = [id: number, tag: string, atom?: boolean, len?: number][];
const tags = (spec: Spec) => new Map(spec.map(([id, tag, atom = false, len = 10]) => [id, { tag, atom, len }]));
/** compact rendering: [id:children] */
const show = (ns: ReturnType<typeof parseTagged>): string =>
  ns.map((n) => (typeof n === "string" ? n : `[${n.id}:${show(n.children)}]`)).join("");
const parse = (s: string, spec: Spec) => show(parseTagged(s, tags(spec)));

test("well-formed tags, reordered, nested, atoms", () => {
  assert.equal(parse("<i1>Octocat</i1>은 <b3>문어 같은</b3> 팔의 <a2>마스코트</a2>입니다.<x4/>", [[1, "i"], [2, "a"], [3, "b"], [4, "x", true]]), "[1:Octocat]은 [3:문어 같은] 팔의 [2:마스코트]입니다.[4:]");
  assert.equal(parse("<a1><i2>GitHub Essentials</i2></a1>. Packt", [[1, "a"], [2, "i"]]), "[1:[2:GitHub Essentials]]. Packt");
});

test("number dropped from a tag", () => {
  assert.equal(parse("<i>Octocat</i1>는 GitHub의 <a2>마스코트</a2>로", [[1, "i"], [2, "a"]]), "[1:Octocat]는 GitHub의 [2:마스코트]로");
});

test("closing tag missing: ends at the next clause boundary", () => {
  assert.equal(parse("2020년 8월 7일 원본에서 <a3> 보관됨. 2015년 7월 30일 조회.", [[3, "a", false, 8]]), "2020년 8월 7일 원본에서 [3:보관됨]. 2015년 7월 30일 조회.");
  assert.equal(parse("<a2>US$75억 달러에 GitHub를 인수하겠다는 의사를 밝혔다.<x3/> 이 거래는", [[2, "a", false, 3], [3, "x", true]]), "[2:US$75억 달러에] GitHub를 인수하겠다는 의사를 밝혔다.[3:] 이 거래는");
});

test("closing tag without '>' and closing tag with the wrong number", () => {
  assert.equal(parse("<a3>2021년 3월 22일 원본 보관</a3. 2019년 조회.", [[3, "a", false, 8]]), "[3:2021년 3월 22일 원본 보관]. 2019년 조회.");
  assert.equal(parse("옥슬리가 <a11>디지털 이미지</a12>를 <a8>클립 아트</a8>로", [[8, "a"], [11, "a", false, 13], [12, "a", false, 14]]), "옥슬리가 [11:디지털 이미지를] [8:클립 아트]로");
});

test("unknown tags are dropped, escaped text comes back, spaces move outside tags", () => {
  assert.equal(parse("a <br> b <b1>x</b1>", [[1, "b"]]), "a  b [1:x]");
  assert.equal(parse("the &lt;div&gt; element", []), "the <div> element");
  assert.equal(parse("GitHub의 <a1> 마스코트 </a1>는", [[1, "a"]]), "GitHub의 [1:마스코트] 는");
});
