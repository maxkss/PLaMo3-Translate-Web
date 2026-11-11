/** The batch size the settings suggest for the GPU and model (shared/common.ts recommendedBatch). */
import assert from "node:assert/strict";
import { test } from "node:test";
import { BATCH_CHOICES, DEFAULT_SETTINGS, EXACT_BATCH, MODELS, kvBytesPerRow, recommendedBatch } from "../src/shared/common.ts";

test("0.8B: 16 on NVIDIA, 4 on other GPUs, 1 without a GPU", () => {
  assert.equal(recommendedBatch("nvidia", MODELS["0.8b"]).size, 16);
  assert.equal(recommendedBatch("NVIDIA Corporation", MODELS["0.8b"]).size, 16);
  assert.equal(recommendedBatch("amd", MODELS["0.8b"]).size, 4);
  assert.equal(recommendedBatch("", MODELS["0.8b"]).size, 4);
  assert.equal(recommendedBatch(null, MODELS["0.8b"]).size, 1);
});

test("2B: the KV memory per row (96 MB) limits the suggestion", () => {
  assert.equal(kvBytesPerRow(MODELS["2b"]), 96 * 2 ** 20);
  // 512 MB fits 5 rows: the largest choice within that is 4
  const r = recommendedBatch("nvidia", MODELS["2b"]);
  assert.equal(r.size, 4);
  assert.match(r.reason, /96 MB/);
  assert.equal(recommendedBatch("intel", MODELS["2b"]).size, 4);
});

test("every suggestion is one of the choices", () => {
  for (const vendor of ["nvidia", "amd", "", null]) for (const m of Object.values(MODELS)) assert.ok(BATCH_CHOICES.includes(recommendedBatch(vendor, m).size));
});

test("the default is 4 and is a choice in the settings", () => {
  assert.equal(DEFAULT_SETTINGS.batchSize, 4);
  assert.ok(DEFAULT_SETTINGS.batchSize > EXACT_BATCH);
  assert.ok(BATCH_CHOICES.includes(DEFAULT_SETTINGS.batchSize));
  assert.ok(BATCH_CHOICES.includes(16) && BATCH_CHOICES.includes(4));
});
