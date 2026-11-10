import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { PlamoTokenizer } from "../src/engine/tokenizer.ts";
import { buildPrompt } from "../src/engine/prompt.ts";

const asset = JSON.parse(readFileSync(new URL("../public/tokenizer/plamo3.json", import.meta.url), "utf-8"));
const vectors: { text: string; ids: number[] }[] = JSON.parse(
  readFileSync(new URL("./tokenizer_vectors.json", import.meta.url), "utf-8"),
);

const t0 = performance.now();
const tok = new PlamoTokenizer(asset);
console.log(`tokenizer build: ${(performance.now() - t0).toFixed(0)} ms`);

for (const v of vectors) {
  test(`encode matches reference: ${JSON.stringify(v.text.slice(0, 40))}`, () => {
    assert.deepEqual(tok.encode(v.text), v.ids);
  });
}

test("decode round-trips (incl. byte fallback for Hangul)", () => {
  for (const v of vectors.slice(0, 16)) assert.equal(tok.decode(v.ids), v.text);
});

test("prompt builder matches chat template output", () => {
  const p = buildPrompt({ text: "Page", targetLang: "ko", sourceLang: "en", context: "Document title: Example" });
  assert.equal(p, vectors[vectors.length - 1].text);
  const p2 = buildPrompt({
    text: "The meeting has been moved to next Monday because the client requested more time.",
    targetLang: "ko",
  });
  assert.equal(p2, vectors[16].text);
});

test("encode speed", () => {
  const long = vectors.slice(0, 16).map((v) => v.text).join("\n").repeat(20);
  const t = performance.now();
  const ids = tok.encode(long);
  const ms = performance.now() - t;
  console.log(`${long.length} chars -> ${ids.length} tokens in ${ms.toFixed(1)} ms`);
  assert.ok(ms < 500);
});
