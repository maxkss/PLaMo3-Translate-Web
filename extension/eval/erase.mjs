/**
 * Compares ways of erasing the original text, on designs whose true text-free background is known (the crello set of
 * eval/build.mjs). The extension's own OCR, paragraph and mask code runs in Chromium (WebGPU, like the extension);
 * each variant of eval/harness/variants.ts then fills the same paragraphs and is scored against the clean background.
 *
 * Per paragraph: l1 (mean absolute error over the hole and the true text, 0–255), grad (gradient-magnitude error:
 * blur and invented texture), p95 (leftover strokes, seams), maskRecall (true text pixels inside the mask), time.
 * Paragraphs are bucketed by how textured the true background under the text is (mean gradient magnitude):
 * flat < 1, smooth < 4, textured ≥ 4 — independent of what the pipeline took the background for.
 *
 * Run: node eval/erase.mjs [--variants=current,diffusion,migan] [--limit=N] [--save=N] [--set=crello]
 * Writes eval/results/erase-<time>.json (and erased images of the first --save items to eval/results/erase/).
 */
import { createServer } from "node:http";
import { createReadStream, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";
import { chromium } from "playwright";

const HERE = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => args.find((a) => a.startsWith(`--${k}=`))?.slice(k.length + 3) ?? d;
const set = opt("set", "crello");
const variants = opt("variants", "current,diffusion,migan").split(",");
const limit = Number(opt("limit", 0)), save = Number(opt("save", 0));
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);

// models, pinned as in src/shared/common.ts (plus candidates under test)
const MODELS = {
  "det_mobile.onnx": "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx/resolve/e6f4fa85f00e168c862bc462aebca69eef9b3d3d/inference.onnx",
  "rec_mobile.onnx": "https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx/resolve/ed152b8b495f84de93cda5709d768548a9127622/inference.onnx",
  "migan.onnx": "https://huggingface.co/FreeHugsForRobots/ps-inpaint-migan/resolve/42eef60d02d0736f0edf7574ebbeba6802d9f533/migan-512-generator.onnx",
};
const modelDir = join(HERE, "data", "models");
mkdirSync(modelDir, { recursive: true });
for (const [name, url] of Object.entries(MODELS)) {
  if (existsSync(join(modelDir, name))) continue;
  log("downloading", name);
  writeFileSync(join(modelDir, name), Buffer.from(await (await fetch(url)).arrayBuffer()));
}

const bundle = await esbuild.build({ entryPoints: [join(HERE, "harness", "erase.ts")], bundle: true, format: "esm", write: false, target: "chrome121", logLevel: "warning" });
const ROOTS = { "/data/": join(HERE, opt("data", "data")), "/models/": modelDir, "/ort/": join(HERE, "..", "node_modules", "onnxruntime-web", "dist"), "/ocr/": join(HERE, "..", "public", "ocr") };
const TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json", ".mjs": "text/javascript", ".wasm": "application/wasm" };
const srv = createServer((req, res) => {
  // cross-origin isolated, so the wasm backend gets its threads
  const headers = { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "require-corp" };
  const url = decodeURIComponent(req.url.split("?")[0]);
  if (url === "/") return res.writeHead(200, { ...headers, "Content-Type": "text/html" }).end(`<!doctype html><script type="module" src="/harness.js"></script>`);
  if (url === "/harness.js") return res.writeHead(200, { ...headers, "Content-Type": "text/javascript" }).end(bundle.outputFiles[0].text);
  const root = Object.keys(ROOTS).find((r) => url.startsWith(r));
  const file = root && join(ROOTS[root], normalize(url.slice(root.length)));
  if (!file || !file.startsWith(ROOTS[root]) || !existsSync(file) || !statSync(file).isFile()) return res.writeHead(404, headers).end();
  res.writeHead(200, { ...headers, "Content-Type": TYPES[extname(file)] ?? "application/octet-stream" });
  createReadStream(file).pipe(res);
}).listen(8790);

let items = readFileSync(join(HERE, opt("data", "data"), set, "manifest.jsonl"), "utf8").trim().split("\n").map((l) => JSON.parse(l)).filter((i) => i.clean || opt("ocr"));
if (limit) items = items.slice(0, limit);
const browser = await chromium.launch({ ...(process.env.CHROME ? { executablePath: process.env.CHROME } : { channel: "chromium" }), args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
const page = await browser.newPage();
page.on("console", (m) => m.type() === "error" && !/W:onnxruntime/.test(m.text()) && log("page:", m.text().slice(0, 300)));
const report = { date: new Date().toISOString(), set, variants, items: [] };
try {
  await page.goto("http://localhost:8790/");
  await page.waitForFunction(() => typeof evalInit === "function");
  log("harness:", await page.evaluate(() => evalInit()));
  // --ocr=<image id>: print that image's paragraphs and stop
  if (opt("ocr")) {
    const it = items.find((i) => i.id.includes(opt("ocr")));
    for (const p of await page.evaluate((u) => evalOcr(u), `/data/${set}/${it.image}`)) console.log(JSON.stringify({ text: p.text, fontPx: +p.fontPx.toFixed(1), bold: p.bold, busy: p.busy, vertical: p.vertical, inkBox: p.inkBox.map(Math.round), lines: p.lines.map((l) => ({ t: l.text, box: l.box.map(Math.round), ink: l.ink?.map(Math.round), span: l.span?.map(Math.round) })) }));
    if (opt("lines")) for (const l of await page.evaluate((u) => evalLines(u), `/data/${set}/${it.image}`)) console.log("line", JSON.stringify({ t: l.text.slice(0, 24), box: l.box, fg: l.fg, bg: l.bg, noise: Math.round(l.noise) }));
    for (const s of it.styles ?? []) console.log("truth", JSON.stringify({ text: s.text, fontSize: s.fontSize, box: s.box, align: s.align, font: s.font }));
    items = [];
  }
  const shots = join(HERE, "results", "erase");
  if (save) mkdirSync(shots, { recursive: true });
  for (const [k, item] of items.entries()) {
    const r = await page.evaluate((req) => evalErase(req), { image: `/data/${set}/${item.image}`, clean: `/data/${set}/${item.clean}`, mask: `/data/${set}/${item.mask}`, variants, save: k < save });
    for (const [name, v] of Object.entries(r.variants)) {
      if (v.png) writeFileSync(join(shots, `${item.id}.${name}.png`), Buffer.from(v.png, "base64"));
      delete v.png;
    }
    report.items.push({ id: item.id, format: item.format, ...r });
    if ((k + 1) % 10 === 0) log(`${k + 1}/${items.length}`);
  }
} finally {
  await browser.close();
  srv.close();
}

// ---- summary
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const med = (xs) => [...xs].sort((a, b) => a - b)[xs.length >> 1] ?? NaN;
const bucket = (t) => (t < 1 ? "flat" : t < 4 ? "smooth" : "textured");
// the background's texture is a property of the paragraph, measured once (on the first variant) so every variant is
// bucketed alike even though their masks differ
const texture = new Map();
for (const it of report.items) for (const p of it.variants[variants[0]].paras) texture.set(`${it.id}/${p.i}`, p.texture);
const table = {};
for (const name of variants) {
  // text the OCR found that is not a text element of the design (lettering inside a photo) has no clean background to
  // compare with: its "mask" is many times the true text under it
  const rows = report.items.flatMap((it) => it.variants[name].paras.filter((p) => p.maskExcess < 12).map((p) => ({ ...p, b: bucket(texture.get(`${it.id}/${p.i}`) ?? p.texture) })));
  table[name] = {};
  for (const b of ["flat", "smooth", "textured", "all"]) {
    const rs = rows.filter((r) => b === "all" || r.b === b);
    const ok = rs.filter((r) => r.filled);
    table[name][b] = { n: rs.length, unfilled: rs.length - ok.length, l1: +mean(ok.map((r) => r.l1)).toFixed(2), grad: +mean(ok.map((r) => r.grad)).toFixed(2), p95: +mean(ok.map((r) => r.p95)).toFixed(1), maskRecall: +mean(rs.map((r) => r.maskRecall)).toFixed(3), excess: +med(rs.map((r) => r.maskExcess)).toFixed(1), ms: +mean(ok.map((r) => r.ms)).toFixed(1), msMed: +med(ok.map((r) => r.ms)).toFixed(1) };
  }
}
report.summary = table;
mkdirSync(join(HERE, "results"), { recursive: true });
const out = join(HERE, "results", `erase-${report.date.replace(/[:.]/g, "-")}.json`);
writeFileSync(out, JSON.stringify(report));
for (const b of ["flat", "smooth", "textured", "all"]) {
  console.log(`\n${b}`);
  console.table(Object.fromEntries(variants.map((v) => [v, table[v][b]])));
}
log("report:", out);
