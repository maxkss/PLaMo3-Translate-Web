import assert from "node:assert/strict";
import { test } from "node:test";
import { isList, reflow, textBlocks } from "../src/shared/blocks.ts";

/** A one-line paragraph with its ink at (x, y, w, h). */
const para = (text: string, x: number, y: number, w: number, h: number, vertical = false) => ({
  text,
  vertical,
  fontPx: (vertical ? w : h) / 0.74,
  inkBox: [x, y, w, h] as [number, number, number, number],
  lines: [{ box: [x, y, w, h] as [number, number, number, number], text, score: 0.99 }],
});

test("textBlocks: a phrase stacked on lines of different sizes is one block, short lines included", () => {
  const ps = [para("NO", 150, 100, 80, 50), para("SMOKING", 100, 165, 180, 40), para("Open daily from nine to five", 100, 600, 400, 20)];
  const { blocks, translate } = textBlocks(ps, "ko", 1000);
  assert.deepEqual(blocks, [[0, 1]]);
  // "NO" alone is too short to translate; as a line of the phrase it is replaced with it
  assert.deepEqual(translate, [true, true, true]);
});

test("textBlocks: lines side by side, far apart, or in a long stack are not a phrase", () => {
  // two labels in a row
  assert.deepEqual(textBlocks([para("Home", 10, 10, 60, 20), para("Gallery", 120, 10, 80, 20)], "ko", 500).blocks, []);
  // a heading and a line two ems below it
  assert.deepEqual(textBlocks([para("Services", 10, 10, 90, 20), para("Web design", 10, 90, 110, 20)], "ko", 500).blocks, []);
  // a menu of eight
  const menu = Array.from({ length: 8 }, (_, k) => para("Menu item", 10, 10 + 34 * k, 100, 20));
  assert.deepEqual(textBlocks(menu, "ko", 500).blocks, []);
});

test("textBlocks: columns of a speech bubble, read right to left", () => {
  const ps = [para("天文部へ！", 40, 100, 30, 150, true), para("夢いっぱいの", 80, 100, 30, 180, true), para("風船いっぱい", 120, 100, 30, 180, true)];
  assert.deepEqual(textBlocks(ps, "ko", 1000).blocks, [[2, 1, 0]]);
});

test("textBlocks: text already in the target language is left out", () => {
  const ps = [para("금연", 100, 100, 100, 50), para("구역", 100, 160, 100, 50)];
  assert.deepEqual(textBlocks(ps, "ko", 1000), { blocks: [], translate: [false, false] });
});

test("isList: lines that translate alike alone and together are a list", () => {
  assert.equal(isList(["홈", "갤러리", "문의하기"], "홈 갤러리 문의하기"), true);
  assert.equal(isList(["당신의", "미래는", "오늘부터", "시작됩니다"], "당신의 미래는 오늘부터 시작됩니다"), true);
  assert.equal(isList(["아니요", "흡연"], "금연"), false);
  assert.equal(isList(["스마트 앱", "체중 감량", "용"], "체중 감량을 위한 스마트 앱"), false);
});

test("reflow: words fill the lines in proportion to their room, in order", () => {
  // four lines, five words: the widest takes two
  assert.deepEqual(reflow("모든 주문 대금 무료 배송", [4, 9, 5, 6]), ["모든", "주문 대금", "무료", "배송"]);
  // a wide line takes more
  assert.deepEqual(reflow("유럽의 윈드서핑 캠프", [5, 9]), ["유럽의", "윈드서핑 캠프"]);
});

test("reflow: fewer words than lines go to the lines with the most room", () => {
  assert.deepEqual(reflow("금연", [2, 7]), ["", "금연"]);
  assert.deepEqual(reflow("정비 차량 주의", [9, 11, 8]), ["정비", "차량", "주의"]);
});

test("reflow: text without spaces is divided by characters", () => {
  const out = reflow("欢迎来到天文部", [4, 3]);
  assert.equal(out.join(""), "欢迎来到天文部");
  assert.equal(out[0].length, 4);
});
