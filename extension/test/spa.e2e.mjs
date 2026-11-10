/**
 * Whole-page translation on a single-page app: the framework re-renders over translated nodes and navigates
 * client-side. Checks that the page throws no DOM errors, that newly rendered content is translated, and that
 * restoring brings the original text back.
 *
 * Prereqs: npm run build; npm run serve-model -- ../models/onnx
 * Run: PROFILE=<dir> node test/spa.e2e.mjs [url=https://react.dev/learn] [link=/learn/thinking-in-react] [mode=replace]
 */
import { launch, togglePage } from "./lib.mjs";

const kv = Object.fromEntries(process.argv.slice(2).map((a) => a.split("=")));
const url = kv.url ?? "https://react.dev/learn";
const link = kv.link ?? "/learn/thinking-in-react";
const mode = kv.mode ?? "replace";
const log = (...a) => console.log(`[${(performance.now() / 1000).toFixed(1)}s]`, ...a);

const { ctx, sw, opts } = await launch({ size: kv.size ?? "0.8b" });
let failed = false;
const check = (ok, msg) => {
  log(ok ? "ok  " : "FAIL", msg);
  if (!ok) failed = true;
};
try {
  await opts.evaluate((m) => chrome.storage.local.set({ displayMode: m, targetLang: "ko" }), mode);
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
  await page.goto(url, { waitUntil: "load" });
  await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => {});
  await page.waitForTimeout(1500);
  const errorsBefore = errors.length;
  const heading = () => page.evaluate(() => document.querySelector("main h1, article h1, h1")?.textContent?.trim() ?? "");
  const hangulShare = () =>
    page.evaluate(() => {
      const t = [...document.querySelectorAll("main p, article p")].map((p) => p.innerText).join(" ");
      const letters = t.match(/\p{L}/gu)?.length ?? 0;
      return letters ? (t.match(/\p{Script=Hangul}/gu)?.length ?? 0) / letters : 0;
    });
  const h1 = await heading();
  log("page:", h1);

  await togglePage(sw, url);
  await page.waitForTimeout(Number(kv.wait ?? 60) * 1000);
  const share1 = await hangulShare();
  check(share1 > 0.4, `paragraphs translated (hangul share ${share1.toFixed(2)}), h1: ${await heading()}`);

  // client-side navigation: the framework replaces the article under our feet
  await page.click(`a[href="${link}"]`);
  await page.waitForTimeout(Number(kv.wait2 ?? 45) * 1000);
  log("after navigation:", page.url(), "|", await heading());
  const share2 = await hangulShare();
  check(share2 > 0.4, `new page translated (hangul share ${share2.toFixed(2)})`);

  // interaction still works: scroll and go back
  await page.goBack();
  await page.waitForTimeout(Number(kv.wait3 ?? 30) * 1000);
  const share3 = await hangulShare();
  check(share3 > 0.4, `back navigation translated (hangul share ${share3.toFixed(2)}), h1: ${await heading()}`);

  await togglePage(sw, page.url());
  await page.waitForTimeout(1000);
  const share4 = await hangulShare();
  check(share4 < 0.02, `restored (hangul share ${share4.toFixed(3)}), h1: ${await heading()}`);
  check(await heading() === h1, "heading restored exactly");
  const newErrors = errors.slice(errorsBefore).filter((e) => !/favicon|net::ERR|Failed to load resource|Content Security Policy/i.test(e));
  check(newErrors.length === 0, `no page errors (${newErrors.length})`);
  for (const e of newErrors.slice(0, 10)) console.log("   ", e.slice(0, 300));
} finally {
  await ctx.close();
}
process.exit(failed ? 1 : 0);
