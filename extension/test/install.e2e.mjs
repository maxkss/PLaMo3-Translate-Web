/**
 * Installs both models from Hugging Face in a fresh Chromium profile and switches between them.
 * Also cancels the first download half-way and checks that the retry resumes instead of restarting.
 *
 * Prereqs: npm run build. Needs network (downloads ~2.1 GB).
 * Run: node test/install.e2e.mjs [--headed]
 */
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";
import { profileDir } from "./lib.mjs";

const headed = process.argv.includes("--headed");
const dist = resolve("dist");
const shots = process.env.SHOTS ?? tmpdir();
const log = (...a) => console.log(`[${(performance.now() / 1000).toFixed(1)}s]`, ...a);

const ctx = await chromium.launchPersistentContext(profileDir("plamo-install-"), {
  channel: "chromium",
  headless: !headed,
  args: [`--disable-extensions-except=${dist}`, `--load-extension=${dist}`, "--enable-unsafe-webgpu"],
  viewport: { width: 1000, height: 1000 },
});

const status = (p) => p.textContent("#status");
const progress = (p) => p.evaluate(() => parseFloat(document.querySelector("#bar > i").style.width) || 0);

async function translate(p, text, lang) {
  await p.fill("#src", text);
  await p.selectOption("#lang", lang);
  await p.evaluate(() => (document.getElementById("speed").textContent = ""));
  await p.click("#go");
  await p.waitForFunction(() => document.getElementById("speed").textContent || /오류/.test(document.getElementById("out").textContent), null, { timeout: 120_000 });
  return `${await p.textContent("#out")} | ${await p.textContent("#speed")}`;
}

async function waitReady(p) {
  await p.waitForFunction(() => /준비됨|오류|실패/.test(document.getElementById("status").textContent), null, { timeout: 1_200_000, polling: 500 });
  const s = await status(p);
  if (!/준비됨/.test(s)) throw new Error(`not ready: ${s}`);
  return s;
}

let failed = false;
const check = (ok, msg) => {
  log(ok ? "PASS" : "FAIL", msg);
  if (!ok) failed = true;
};

try {
  const sw = ctx.serviceWorkers()[0] ?? (await ctx.waitForEvent("serviceworker"));
  const extId = new URL(sw.url()).host;
  const p = await ctx.newPage();
  p.on("console", (m) => m.type() === "error" && log("console:", m.text()));
  await p.goto(`chrome-extension://${extId}/options.html`);
  await p.waitForFunction(() => !/연결 중/.test(document.getElementById("status").textContent));
  log("initial:", await status(p), "|", await p.textContent("#download"));
  check(/2B 모델이 설치되지/.test(await status(p)), "fresh profile: 2B selected and missing");

  // ---- 0.8B: start, cancel half-way, resume
  await p.check('input[name="model"][value="0.8b"]');
  await p.waitForFunction(() => /0\.8B/.test(document.getElementById("download").textContent));
  log("download button:", await p.textContent("#download"), "|", await p.textContent("#source"));
  const tStart = performance.now() / 1000;
  await p.click("#download");
  await p.waitForFunction(() => parseFloat(document.querySelector("#bar > i").style.width) > 40, null, { timeout: 600_000, polling: 200 });
  const atCancel = await progress(p);
  await p.click("#cancelDl");
  await p.waitForFunction(() => /취소/.test(document.getElementById("status").textContent), null, { timeout: 30_000 });
  log("cancelled at", atCancel.toFixed(0) + "%:", await status(p));
  // the retry re-hashes the bytes already on disk (fast, local) instead of downloading them again
  const t0 = performance.now();
  await p.click("#download");
  await p.waitForFunction((x) => parseFloat(document.querySelector("#bar > i").style.width) >= x, atCancel - 1, { timeout: 600_000, polling: 50 });
  const dlTime = (performance.now() - t0) / 1000;
  check(dlTime < 5, `resume: back at ${atCancel.toFixed(0)}% after ${dlTime.toFixed(1)}s (first ${atCancel.toFixed(0)}% took ${(t0 / 1000 - tStart).toFixed(1)}s from the network)`);
  log("0.8B:", await waitReady(p));
  log("[0.8B ko]", await translate(p, "PLaMo 3 Translate runs entirely in your browser, so the pages you read never leave your device.", "ko"));
  log("[0.8B en]", await translate(p, "会議は来週の月曜日に変更されました。資料は前日までに共有してください。", "en"));

  // ---- 2B
  await p.check('input[name="model"][value="2b"]');
  await p.waitForFunction(() => /2B 모델이 설치되지/.test(document.getElementById("status").textContent), null, { timeout: 30_000 });
  check(true, "switching to an uninstalled 2B unloads 0.8B and shows missing");
  await p.click("#download");
  log("2B:", await waitReady(p));
  log("[2B ko]", await translate(p, "PLaMo 3 Translate runs entirely in your browser, so the pages you read never leave your device.", "ko"));

  // ---- back to 0.8B (installed): next job loads it
  const badges = await p.$$eval("[data-badge]", (els) => els.map((e) => e.textContent));
  check(badges.every((b) => b === "설치됨"), `both models marked installed (${badges.join(", ")})`);
  await p.check('input[name="model"][value="0.8b"]');
  log("[0.8B again]", await translate(p, "Switching models should not require downloading again.", "ko"));
  const s = await status(p);
  check(/0\.8B/.test(s), `engine now runs 0.8B (${s})`);
  check(/설치됨/.test(await p.textContent("#download")), "download button shows installed");
  await p.screenshot({ path: join(shots, "options-models.png"), fullPage: true });
  log("screenshot", join(shots, "options-models.png"));
} catch (e) {
  failed = true;
  log("ERROR", e);
} finally {
  await ctx.close();
  process.exit(failed ? 1 : 0);
}
