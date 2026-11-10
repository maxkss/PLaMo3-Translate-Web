/**
 * Image translation end to end: renders test images (Japanese horizontal / vertical, English), serves them from
 * another origin without CORS headers, then right-click-translates them in a page and checks the overlay.
 *
 * Prereqs: npm run build; npm run serve-model -- ../models/onnx (local PLaMo model; the OCR model comes from HF)
 * Run: node test/image.e2e.mjs [size=0.8b] [--headed]   (PROFILE=<dir> keeps installed models between runs)
 */
import { createServer } from "node:http";
import { mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { launch } from "./lib.mjs";

const size = process.argv.find((a) => /^(0\.8b|2b)$/.test(a)) ?? "0.8b";
const headed = process.argv.includes("--headed");
const shots = process.env.SHOTS ?? join(tmpdir(), "plamo-image-e2e");
mkdirSync(shots, { recursive: true });
const log = (...a) => console.log(`[${(performance.now() / 1000).toFixed(1)}s]`, ...a);

const IMAGES = {
  "ja.png": `<div style="width:640px;padding:28px;background:#fdf6e3;color:#222;font:26px/1.5 'Hiragino Sans','Noto Sans JP',sans-serif">
      <div style="font-size:34px;font-weight:700;color:#c0392b;margin-bottom:16px">期間限定イベント開催中！</div>
      <div>今日はみんなでライブの練習をしましょう。新しい衣装も届きましたよ。</div>
      <div style="margin-top:18px;background:#2c3e50;color:#fff;padding:10px 14px;border-radius:10px">プロデューサーさん、明日も頑張りましょうね！</div>
    </div>`,
  "vertical.png": `<div style="width:360px;height:420px;padding:24px;background:#fff;color:#111;display:flex;justify-content:center;font:28px/1.6 'Hiragino Mincho ProN','Noto Serif JP',serif">
      <div style="writing-mode:vertical-rl;height:360px;border:3px solid #111;border-radius:50%;padding:30px 40px">ここは私に任せて、先に行ってください。</div>
    </div>`,
  "en.png": `<div style="width:560px;padding:30px;background:#0b6e4f;color:#fff;font:24px/1.4 Helvetica,Arial,sans-serif">
      <div style="font-size:36px;font-weight:800">Summer Sale</div>
      <div style="margin-top:10px">Get up to 50% off on all items this weekend only. Free shipping on orders over $30.</div>
    </div>`,
};

const { ctx, sw, extId, opts } = await launch({ size, headed });
const files = {};
try {
  // ---- render the test images
  const gen = await ctx.newPage();
  for (const [name, html] of Object.entries(IMAGES)) {
    await gen.setContent(`<body style="margin:0;display:inline-block">${html}</body>`);
    files[name] = await gen.locator("body > div").screenshot();
    writeFileSync(join(shots, name), files[name]);
  }
  await gen.close();

  // image host without CORS/CORP headers (another origin), and the test page
  // an SVG with text: createImageBitmap cannot decode it in the offscreen document, so this exercises the
  // screenshot fallback (cross-origin without CORS, so the page cannot read it either)
  files["sign.svg"] = Buffer.from(`<svg xmlns="http://www.w3.org/2000/svg" width="420" height="140"><rect width="420" height="140" fill="#ffd400"/><text x="210" y="88" font-size="44" font-family="Hiragino Sans, sans-serif" font-weight="700" text-anchor="middle" fill="#222">立入禁止</text></svg>`);
  const imgSrv = createServer((req, res) => {
    const f = files[req.url.slice(1)];
    if (!f) return res.writeHead(404).end();
    res.writeHead(200, { "Content-Type": req.url.endsWith(".svg") ? "image/svg+xml" : "image/png" }).end(f);
  }).listen(8789);
  const page$ = `<!doctype html><meta charset="utf-8"><title>image test</title>
    <body style="font:16px sans-serif;margin:24px">
      <h1>이미지 번역 테스트</h1>
      <p><img id="ja" src="http://127.0.0.1:8789/ja.png" style="width:480px"></p>
      <p><img id="vertical" src="http://127.0.0.1:8789/vertical.png" style="width:240px;height:240px;object-fit:contain;background:#eee"></p>
      <p><img id="en" src="http://127.0.0.1:8789/en.png"></p>
      <p><img id="blob" style="width:320px"></p>
      <p><img id="svg" src="http://127.0.0.1:8789/sign.svg"></p>
      <script>fetch("/same/ja.png").then(r => r.blob()).then(b => document.getElementById("blob").src = URL.createObjectURL(b));</script>
    </body>`;
  const pageSrv = createServer((req, res) => {
    if (req.url.startsWith("/same/")) {
      const f = files[req.url.slice(6)];
      return f ? res.writeHead(200, { "Content-Type": "image/png" }).end(f) : res.writeHead(404).end();
    }
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" }).end(page$);
  }).listen(8788);

  // ---- OCR directly through the engine port (accuracy check)
  for (const name of Object.keys(IMAGES)) {
    const t0 = performance.now();
    const r = await opts.evaluate(async (src) => {
      await chrome.runtime.sendMessage({ type: "ensure-engine" });
      const port = chrome.runtime.connect({ name: "plamo-engine" });
      const out = await new Promise((res) => {
        port.onMessage.addListener((m) => {
          if (m.id !== "ocr-probe") return;
          if (m.type === "ocr-progress") console.log(m.message);
          if (m.type === "ocr-done") res(m.result);
          if (m.type === "error") res({ error: m.message });
        });
        port.postMessage({ type: "ocr", id: "ocr-probe", src });
      });
      port.disconnect();
      return out;
    }, `http://127.0.0.1:8789/${name}`);
    if (r.error) throw new Error(`${name}: ${r.error}`);
    log(`OCR ${name} (${r.backend}, ${r.ms} ms worker, ${((performance.now() - t0) / 1000).toFixed(1)} s total):`);
    for (const p of r.paragraphs) log(`   [${p.vertical ? "V" : "H"} ${p.box.map(Math.round).join(",")} bg=${p.bg} fg=${p.fg}] ${p.text}`);
    if (process.env.LINES) for (const p of r.paragraphs) for (const l of p.lines) log(`      line ${l.box.map(Math.round).join(",")} ${l.score.toFixed(2)} ${l.text}`);
  }

  // ---- right-click translation in a page
  const page = await ctx.newPage();
  page.on("console", (m) => /plamo|error/i.test(m.text()) && log("page:", m.text()));
  await page.goto("http://localhost:8788/");
  await page.waitForFunction(() => [...document.images].every((i) => i.complete && i.naturalWidth));
  const overlayState = () =>
    page.evaluate(() => {
      const sr = document.querySelector("plamo-translate-ui")?.shadowRoot;
      return [...(sr?.querySelectorAll(".img-ov") ?? [])].map((ov) => ({
        badge: ov.querySelector(".img-badge .label")?.textContent,
        paras: [...ov.querySelectorAll(".img-para")].map((p) => ({ text: p.textContent, src: p.title, font: p.style.fontSize })),
      }));
    });
  const ids = ["ja", "vertical", "en", "blob", "svg"];
  for (const id of ids) {
    const t0 = performance.now();
    await page.locator(`#${id}`).scrollIntoViewIfNeeded();
    await page.locator(`#${id}`).dispatchEvent("contextmenu");
    const srcUrl = await page.$eval(`#${id}`, (i) => i.currentSrc);
    await sw.evaluate(async ([u, src]) => {
      const [tab] = await chrome.tabs.query({ url: u });
      await chrome.tabs.sendMessage(tab.id, { type: "translate-image", srcUrl: src });
    }, ["http://localhost:8788/*", srcUrl]);
    await page.waitForFunction(
      (n) => {
        const ovs = document.querySelector("plamo-translate-ui")?.shadowRoot?.querySelectorAll(".img-ov");
        const b = ovs?.[n]?.querySelector(".img-badge .label")?.textContent ?? "";
        return /완료|없습니다|못했습니다/.test(b);
      },
      ids.indexOf(id),
      { timeout: 300_000, polling: 300 },
    );
    const st = (await overlayState()).at(-1);
    log(`${id}: ${st.badge} (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
    for (const p of st.paras) log(`   ${p.src} → ${p.text}  [${p.font}]`);
    await page.locator(`#${id}`).screenshot({ path: join(shots, `${id}-translated.png`) });
  }
  await page.screenshot({ path: join(shots, "page.png"), fullPage: true });

  // toggling removes the overlay
  await page.locator("#ja").dispatchEvent("contextmenu");
  await sw.evaluate(async ([u, src]) => {
    const [tab] = await chrome.tabs.query({ url: u });
    await chrome.tabs.sendMessage(tab.id, { type: "translate-image", srcUrl: src });
  }, ["http://localhost:8788/*", "http://127.0.0.1:8789/ja.png"]);
  log("overlays after toggle:", (await overlayState()).length);
  log("screenshots in", shots);
  imgSrv.close();
  pageSrv.close();
} finally {
  await ctx.close();
}
