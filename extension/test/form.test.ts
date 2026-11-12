/** A page's sentence style: which text decides it (content/page.ts isProse) and how it is read (textForm). */
import assert from "node:assert/strict";
import { test } from "node:test";
import { isProse } from "../src/content/page.ts";
import { textForm } from "../src/shared/common.ts";

test("sentences are prose", () => {
  assert.ok(isProse("The city council approved a plan on Tuesday to expand the bus network."));
  assert.ok(isProse("Officials said the first buses would run before the end of the year. Residents welcomed it"));
  assert.ok(isProse("市議会は火曜日、バス路線を北部郊外に広げる計画を承認した。"));
});

test("labels, headings and menus are not", () => {
  assert.ok(!isProse("Home"));
  assert.ok(!isProse("Terms of service and privacy policy"));
  assert.ok(!isProse("New bus network for the northern suburbs approved"));
  assert.ok(!isProse("Version 3.12.1"));
  assert.ok(!isProse("Read more..."));
});

test("Korean: -다 is plain, -습니다 and -요 polite, by most sentences", () => {
  assert.equal(textForm("시의회는 계획을 승인했다. 당국은 연말 전에 운행될 것이라고 밝혔다.", "ko"), "plain");
  assert.equal(textForm("이 도구는 Python 3.10 이상이 필요합니다. 설치 프로그램을 실행하세요.", "ko"), "polite");
  // footnote marks and closing quotes after the ending
  assert.equal(textForm("트램은 도시 철도의 한 형태이다.[2][3]\n그는 “좋은 첫걸음”이라고 말했다.”", "ko"), "plain");
  assert.equal(textForm("하나는 이렇다. 둘은 저렇습니다. 셋도 그렇습니다.", "ko"), "polite");
});

test("Japanese: です・ます is polite, だ・である plain", () => {
  assert.equal(textForm("この道具は便利です。毎日使っています。", "ja"), "polite");
  assert.equal(textForm("市議会は計画を承認した。初便は年内に走る予定である。", "ja"), "plain");
});

test("no sentence endings, or another language: unknown", () => {
  assert.equal(textForm("홈 · 뉴스 · 스포츠", "ko"), null);
  assert.equal(textForm("", "ko"), null);
  assert.equal(textForm("Le conseil a approuvé le plan.", "fr"), null);
});
