/** How the translator cuts a long request into model calls (engine/translator.ts split()). */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { PlamoTokenizer } from "../src/engine/tokenizer.ts";
import { Translator } from "../src/engine/translator.ts";

const tok = new PlamoTokenizer(JSON.parse(readFileSync(new URL("../public/tokenizer/plamo3.json", import.meta.url), "utf-8")));
const tr = new Translator(tok, {} as never);
const CHUNK_TOKENS = 384;
const line = (i: number) => `Item ${i}: the quick brown fox jumps over the lazy dog near the river bank.`;

test("a request that fits is one call", () => {
  const text = [1, 2, 3].map(line).join("\n");
  assert.deepEqual(tr.split(text).map((p) => p.text), [text]);
});

test("the lines of a longer request are packed into as few calls as fit, in order", () => {
  const lines = Array.from({ length: 60 }, (_, i) => line(i));
  const text = lines.join("\n");
  const total = tok.encode(text).length;
  assert.ok(total > CHUNK_TOKENS * 2, String(total));
  const pieces = tr.split(text);
  // nothing lost, and the separators between chunks are kept as they were
  assert.equal(pieces.map((p) => p.text).join(""), text);
  const chunks = pieces.filter((p) => p.translate);
  for (const c of chunks) assert.ok(tok.encode(c.text).length <= CHUNK_TOKENS, c.text);
  assert.ok(chunks.length <= Math.ceil(total / CHUNK_TOKENS) + 1, `${chunks.length} chunks for ${total} tokens`);
  assert.ok(chunks.every((c) => c.text.split("\n").length > 1));
});

test("blank lines (paragraphs) still separate calls", () => {
  const para = Array.from({ length: 12 }, (_, i) => line(i)).join(" ");
  const pieces = tr.split(`${para}\n\n${para}\n\n${para}`);
  assert.deepEqual(pieces.map((p) => p.translate), [true, false, true, false, true]);
});

test("a numbered tag is not cut apart by a line break inside it", () => {
  const lines = Array.from({ length: 40 }, (_, i) => line(i));
  lines[20] = "<a1>open";
  lines[21] = "close</a1>";
  for (const p of tr.split(lines.join("\n")).filter((p) => p.translate)) {
    assert.equal(p.text.includes("<a1>"), p.text.includes("</a1>"));
  }
});
