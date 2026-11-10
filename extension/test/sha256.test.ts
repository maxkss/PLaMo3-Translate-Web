import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { test } from "node:test";
import { Sha256 } from "../src/engine/sha256.ts";

test("sha256 matches node:crypto for many lengths and chunkings", () => {
  for (const len of [0, 1, 55, 56, 63, 64, 65, 119, 120, 1000, 100_003]) {
    const data = new Uint8Array(len).map((_, i) => (i * 131 + 7) & 255);
    const want = createHash("sha256").update(data).digest("hex");
    for (const chunk of [1, 7, 64, 1000, len || 1]) {
      const h = new Sha256();
      for (let i = 0; i < len; i += chunk) h.update(data.subarray(i, i + chunk));
      assert.equal(h.hex(), want, `len=${len} chunk=${chunk}`);
    }
  }
});
