/**
 * In-flight merging: identical requests sent at the same time are computed once and all get the result.
 * Prereqs: npm run build; npm run serve-model -- ../models/onnx   Run: node test/dedup.e2e.mjs [size=0.8b]
 */
import { resolve } from "node:path";
import { chromium } from "playwright";
import { profileDir } from "./lib.mjs";

const size = (process.argv[2] ?? "size=0.8b").split("=")[1];
const dist = resolve("dist");
const ctx = await chromium.launchPersistentContext(profileDir("plamo-dd-"), {
  channel: "chromium",
  headless: true,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`, "--enable-unsafe-webgpu"],
});
try {
  const sw = ctx.serviceWorkers()[0] ?? (await ctx.waitForEvent("serviceworker"));
  const opts = await ctx.newPage();
  await opts.goto(`chrome-extension://${new URL(sw.url()).host}/options.html`);
  await opts.evaluate(async (s) => chrome.storage.local.set({ model: s, modelUrlOverride: `http://localhost:8787/plamo-3-translate-${s}/` }), size);
  await opts.reload();
  await opts.waitForFunction(() => !/연결 중/.test(document.getElementById("status").textContent));
  if (/설치되지/.test(await opts.textContent("#status"))) await opts.click("#download");
  await opts.waitForFunction(() => /준비됨/.test(document.getElementById("status").textContent), null, { timeout: 600_000 });
  // drive the engine port directly: same text 4x at once, then one different text
  const out = await opts.evaluate(async () => {
    const port = chrome.runtime.connect({ name: "plamo-engine" });
    const results = new Map();
    const deltas = new Map();
    const wait = new Promise((res) => {
      port.onMessage.addListener((m) => {
        if (m.type === "delta") deltas.set(m.id, (deltas.get(m.id) ?? 0) + 1);
        if (m.type === "done") {
          results.set(m.id, { text: m.text, t: performance.now() });
          if (results.size === 5) res();
        }
      });
    });
    const text = "Duplicate requests should be translated only once, and every caller gets the same result.";
    const t0 = performance.now();
    for (let i = 0; i < 4; i++) port.postMessage({ type: "translate", id: `dup${i}`, text, targetLang: "ko", priority: 5, stream: i === 0 });
    port.postMessage({ type: "translate", id: "other", text: "A different sentence is translated separately.", targetLang: "ko", priority: 5 });
    await wait;
    const dup = [0, 1, 2, 3].map((i) => results.get(`dup${i}`));
    return {
      sameText: dup.every((d) => d.text === dup[0].text),
      dupDoneSpreadMs: Math.round(Math.max(...dup.map((d) => d.t)) - Math.min(...dup.map((d) => d.t))),
      dupMs: Math.round(dup[0].t - t0),
      otherAfterDupMs: Math.round(results.get("other").t - dup[0].t),
      streamedDeltas: deltas.get("dup0") ?? 0,
      text: dup[0].text,
    };
  });
  console.log(JSON.stringify(out, null, 1));
  if (!out.sameText || out.dupDoneSpreadMs > 50) process.exitCode = 1;
} finally {
  await ctx.close();
}
