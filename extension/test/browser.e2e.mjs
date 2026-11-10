/**
 * Loads dist/ into Chromium (Playwright), installs the 2B model from Hugging Face,
 * and exercises selection + whole-page translation on test/fixtures/article.html.
 *
 * Prereqs: npm run build (needs network for the first model download)
 * Run: node test/browser.e2e.mjs [--headed]
 * MODEL_BASE=http://localhost:8787/ installs from a local model server (npm run serve-model -- ../models/onnx) instead;
 * SIZE=0.8b uses the 0.8B model.
 */
import { createServer } from "node:http";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";
import { profileDir } from "./lib.mjs";

const headed = process.argv.includes("--headed");
const dist = resolve("dist");
const shots = process.env.SHOTS ?? tmpdir();

// fixture server
const fixture = readFileSync("test/fixtures/article.html");
const srv = createServer((_, res) => res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" }).end(fixture)).listen(8788);

const ctx = await chromium.launchPersistentContext(profileDir("plamo-"), {
  channel: "chromium",
  headless: !headed,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`, "--enable-unsafe-webgpu", "--enable-features=Vulkan"],
  viewport: { width: 1100, height: 900 },
});
const log = (...a) => console.log(`[${(performance.now() / 1000).toFixed(1)}s]`, ...a);

try {
  let sw = ctx.serviceWorkers()[0] ?? (await ctx.waitForEvent("serviceworker"));
  const extId = new URL(sw.url()).host;
  log("extension", extId);

  // ---- options: install model + test translation
  const opts = await ctx.newPage();
  opts.on("console", (m) => m.type() === "error" && log("options console:", m.text()));
  await opts.goto(`chrome-extension://${extId}/options.html`);
  const size = process.env.SIZE ?? "2b";
  await opts.evaluate((m) => chrome.storage.local.set({ model: m }), size);
  if (process.env.MODEL_BASE) {
    await opts.evaluate((u) => chrome.storage.local.set({ modelUrlOverride: u }), `${process.env.MODEL_BASE}plamo-3-translate-${size}/`);
  }
  await opts.reload();
  log("webgpu adapter:", await opts.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter();
    return a ? `${a.info?.vendor} ${a.info?.architecture} f16=${a.features.has("shader-f16")}` : "none";
  }));
  await opts.waitForFunction(() => !/연결 중/.test(document.getElementById("status").textContent));
  log("status:", await opts.textContent("#status"));
  if (/설치되지/.test(await opts.textContent("#status"))) {
    await opts.click("#download");
    await opts.waitForFunction(() => /내려받는/.test(document.getElementById("status").textContent), null, { timeout: 30_000 });
    await opts.waitForFunction(() => !/내려받는/.test(document.getElementById("status").textContent), null, { timeout: 600_000, polling: 1000 });
    log("after download:", await opts.textContent("#status"));
  }
  await opts.click("#load");
  await opts.waitForFunction(() => /준비됨|오류/.test(document.getElementById("status").textContent), null, { timeout: 600_000, polling: 500 });
  log("engine:", await opts.textContent("#status"));

  for (const [text, lang] of [
    ["PLaMo 3 Translate runs entirely in your browser, so the pages you read never leave your device.", "ko"],
    ["会議は来週の月曜日に変更されました。資料は前日までに共有してください。", "ko"],
    ["안녕하세요, 오늘 날씨가 정말 좋네요. 내일 회의에서 뵙겠습니다.", "en"],
  ]) {
    await opts.fill("#src", text);
    await opts.selectOption("#lang", lang);
    await opts.click("#go");
    await opts.waitForFunction(() => document.getElementById("speed").textContent || /오류/.test(document.getElementById("out").textContent), null, { timeout: 120_000 });
    log(`[${lang}]`, await opts.textContent("#out"), "|", await opts.textContent("#speed"));
  }
  await opts.screenshot({ path: join(shots, "options.png"), fullPage: true });

  // ---- selection translation
  const page = await ctx.newPage();
  page.on("console", (m) => m.type() === "error" && log("page console:", m.text()));
  await page.goto("http://localhost:8788/");
  await page.waitForTimeout(500);
  const box = await page.locator("#sel").boundingBox();
  await page.mouse.move(box.x + 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width - 2, box.y + box.height / 2, { steps: 5 });
  await page.mouse.up();
  const fab = page.locator("plamo-translate-ui").locator(".fab");
  await fab.waitFor({ timeout: 5000 });
  await fab.click();
  const body = page.locator("plamo-translate-ui").locator(".panel .body");
  await page.waitForFunction(() => {
    const b = document.querySelector("plamo-translate-ui")?.shadowRoot?.querySelector(".panel .body");
    return b && !b.classList.contains("pending") && !b.querySelector(".caret");
  }, null, { timeout: 60_000 });
  log("selection:", await body.textContent());
  await page.screenshot({ path: join(shots, "selection.png") });
  await page.keyboard.press("Escape");

  // ---- whole page translation (dual mode)
  sw = ctx.serviceWorkers()[0];
  await page.evaluate(() => document.documentElement.setAttribute("data-plamo-debug", "")); // finished blocks get data-plamo-done
  const t0 = performance.now();
  await sw.evaluate(async () => {
    const [tab] = await chrome.tabs.query({ url: "http://localhost:8788/*" });
    await chrome.tabs.sendMessage(tab.id, { type: "page-toggle" });
  });
  // ---- an interactive request while the page is being translated must not wait for the page job
  await page.waitForTimeout(400);
  const tSel = performance.now();
  await opts.fill("#src", "Interactive requests should jump ahead of background page translation.");
  await opts.selectOption("#lang", "ko");
  await opts.evaluate(() => (document.getElementById("speed").textContent = ""));
  await opts.click("#go");
  await opts.waitForFunction(() => document.getElementById("speed").textContent, null, { timeout: 60_000 });
  const pageBusy = await page.evaluate(() => !document.querySelectorAll("li plamo-tr").length || !document.querySelector("#late plamo-tr"));
  log(`interactive during page translation: ${((performance.now() - tSel) / 1000).toFixed(2)}s (page still translating: ${pageBusy}) ->`, await opts.textContent("#out"));
  await page.waitForFunction(() => {
    const done = (el) => !!el && (el.hasAttribute("data-plamo-done") || !!el.querySelector("[data-plamo-done]"));
    return ["p1", "p2", "late"].every((id) => done(document.getElementById(id))) && [...document.querySelectorAll("li")].every(done);
  }, null, { timeout: 180_000, polling: 200 });
  log(`page translated in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
  const result = await page.evaluate(() => ({
    h1: document.querySelector("h1 plamo-tr")?.textContent,
    p1: document.querySelector("#p1 plamo-tr")?.textContent,
    p2: document.querySelector("#p2 plamo-tr")?.textContent,
    li: [...document.querySelectorAll("li plamo-tr")].map((e) => e.textContent),
    late: document.querySelector("#late plamo-tr")?.textContent,
    ko_untouched: !document.querySelector("#ko plamo-tr"),
    code_untouched: !document.querySelector("pre plamo-tr"),
    link_intact: document.querySelector("#p2 a")?.textContent,
  }));
  console.log(JSON.stringify(result, null, 1));
  await page.screenshot({ path: join(shots, "page-dual.png"), fullPage: true });

  // ---- switch to replace mode, then restore
  await opts.evaluate(() => chrome.storage.local.set({ displayMode: "replace" }));
  await page.waitForFunction(() => !document.querySelector("#p1 plamo-tr"), null, { timeout: 10_000 });
  log("replace mode p1:", await page.textContent("#p1"), "| p2:", await page.textContent("#p2"));
  await page.screenshot({ path: join(shots, "page-replace.png"), fullPage: true });
  await sw.evaluate(async () => {
    const [tab] = await chrome.tabs.query({ url: "http://localhost:8788/*" });
    await chrome.tabs.sendMessage(tab.id, { type: "page-toggle" });
  });
  await page.waitForTimeout(300);
  log("restored p1:", await page.textContent("#p1"));
  await opts.evaluate(() => chrome.storage.local.set({ displayMode: "dual" }));
  log("screenshots in", shots);
} finally {
  await ctx.close();
  srv.close();
}
