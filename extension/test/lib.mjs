/**
 * Shared Playwright setup for the browser tests: Chromium with the built extension, the model installed from the
 * local model server (npm run serve-model -- ../models/onnx). Set PROFILE=<dir> to keep the installed model
 * between runs (otherwise a fresh temporary profile copies it again).
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

/**
 * Chromium profile directory for one run. With `reuse` and PROFILE=<dir>, that directory is used and kept (so the
 * installed model survives between runs). Otherwise a fresh temporary profile is created and deleted when the process
 * exits: each one holds a full copy of the model in OPFS (~0.5–1.6 GB), and they used to pile up in $TMPDIR.
 */
export function profileDir(prefix = "plamo-", { reuse = false } = {}) {
  if (reuse && process.env.PROFILE) return resolve(process.env.PROFILE);
  const dir = mkdtempSync(join(tmpdir(), prefix));
  process.once("exit", () => rmSync(dir, { recursive: true, force: true }));
  return dir;
}

export async function launch({ size = "2b", headed = false, viewport = { width: 1280, height: 900 } } = {}) {
  const dist = resolve("dist");
  const profile = profileDir("plamo-", { reuse: true });
  const ctx = await chromium.launchPersistentContext(profile, {
    channel: "chromium",
    headless: !headed,
    viewport,
    args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`, "--enable-unsafe-webgpu"],
  });
  const sw = ctx.serviceWorkers()[0] ?? (await ctx.waitForEvent("serviceworker"));
  const extId = new URL(sw.url()).host;
  const opts = await ctx.newPage();
  await opts.goto(`chrome-extension://${extId}/options.html`);
  await opts.evaluate(
    async (s) => chrome.storage.local.set({ model: s, modelUrlOverride: `http://localhost:8787/plamo-3-translate-${s}/` }),
    size,
  );
  await opts.reload();
  await opts.waitForFunction(() => !/연결 중/.test(document.getElementById("status").textContent));
  if (/설치되지/.test(await opts.textContent("#status"))) await opts.click("#download");
  await opts.waitForFunction(() => /준비됨|대기/.test(document.getElementById("status").textContent), null, { timeout: 900_000, polling: 500 });
  return { ctx, sw, extId, opts };
}

/** Translate texts through the engine port from an extension page (no streaming); resolves to outputs in order. */
export function translateAll(opts, texts, targetLang = "ko", context = null) {
  return opts.evaluate(
    async ([texts, targetLang, context]) => {
      await chrome.runtime.sendMessage({ type: "ensure-engine" });
      const port = chrome.runtime.connect({ name: "plamo-engine" });
      const waits = new Map();
      port.onMessage.addListener((m) => {
        if (m.type === "done" || m.type === "error") waits.get(m.id)?.(m.type === "done" ? m.text : "ERROR " + m.message);
      });
      const out = await Promise.all(
        texts.map(
          (text, i) =>
            new Promise((res) => {
              const id = "probe-" + i + "-" + Math.random();
              waits.set(id, res);
              port.postMessage({ type: "translate", id, text, targetLang, context, priority: 5, stream: false });
            }),
        ),
      );
      port.disconnect();
      return out;
    },
    [texts, targetLang, context],
  );
}

/** Toggle whole-page translation in the tab showing `url`. */
export function togglePage(sw, url) {
  return sw.evaluate(async (u) => {
    const [tab] = await chrome.tabs.query({ url: u.split("#")[0] + "*" });
    return chrome.tabs.sendMessage(tab.id, { type: "page-toggle" });
  }, url);
}

/**
 * Translate `text` through the engine port from an extension page (the options page): returns "<translation> | <seconds>s".
 * Errors come back as "오류: <message>" so callers can log them like a result.
 */
export function translateIn(page, text, targetLang) {
  return page.evaluate(
    async ({ text, targetLang }) => {
      const t0 = performance.now();
      const res = await chrome.runtime.sendMessage({ type: "ensure-engine" });
      if (!res?.ok) return `오류: ${res?.error ?? "engine"}`;
      const port = chrome.runtime.connect({ name: "plamo-engine" });
      const id = `e2e-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const out = await new Promise((resolve) => {
        port.onMessage.addListener((m) => {
          if (m.id !== id) return;
          if (m.type === "done") resolve(m.text);
          else if (m.type === "error") resolve(`오류: ${m.message}`);
        });
        port.postMessage({ type: "translate", id, text, targetLang, priority: 0, stream: false });
      });
      port.disconnect();
      return `${out} | ${((performance.now() - t0) / 1000).toFixed(2)}s`;
    },
    { text, targetLang },
  );
}
