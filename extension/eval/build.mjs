/**
 * Builds the image-translation evaluation set from public datasets into eval/data/ (not committed: most sources forbid
 * redistribution — rebuild it with this script). Every set gets images/ and a manifest.jsonl in one schema:
 *
 *   { id, set, image, width, height, lang, license, source,
 *     regions: [{ box: [x, y, w, h] | null, text, level, vertical?, refs?: { en?, zh?, … } }],
 *     styles?: [{ box, text, fontSize, fontWeight, color, align, key }] }   // websight only: rendered text runs
 *
 * Sets (sizes are defaults, --n=<count> overrides all):
 *   cagui       Chinese app screens (CAGUI grounding/OCR, CC BY-NC 4.0): one ground-truth text element per screen
 *   hiertext    text-rich photos and documents (HierText validation, CC BY-SA 4.0; images Open Images CC BY 2.0):
 *               paragraphs with their lines (text, vertical flag)
 *   openmantra  Japanese manga pages (OpenMantra, CC BY-NC 4.0): speech-bubble boxes, Japanese text and
 *               professional English / Chinese translations
 *   vistra      English text in photos (Vistra, CC BY-SA 4.0): transcript groups (no boxes) with translations into
 *               de/es/ru/zh
 *   websight    synthetic English web pages (WebSight v0.1 HTML, CC BY 4.0), rendered here at 1024 px: every text
 *               run with its box and computed style (size, weight, colour, alignment)
 *
 * Run: node eval/build.mjs [--sets=cagui,hiertext,…] [--n=50]
 */
import { mkdirSync, writeFileSync, existsSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "data");
const args = process.argv.slice(2);
const only = args.find((a) => a.startsWith("--sets="))?.slice(7).split(",");
const nArg = Number(args.find((a) => a.startsWith("--n="))?.slice(4) ?? 0);
const DEFAULT_N = { cagui: 400, hiertext: 80, openmantra: 60, vistra: 100, websight: 40 };
const n = (set) => nArg || DEFAULT_N[set];
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);

async function get(url, as = "buffer") {
  for (let attempt = 0; ; attempt++) {
    try {
      const res = await fetch(url, { headers: { "User-Agent": "plamo-translate-eval" } });
      if (!res.ok) throw new Error(`${res.status} ${url}`);
      return as === "json" ? res.json() : as === "text" ? res.text() : Buffer.from(await res.arrayBuffer());
    } catch (e) {
      if (attempt >= 2 || /\b404\b/.test(String(e))) throw e;
      await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
    }
  }
}

/** Run `fn` over items, `k` at a time. */
async function pool(items, k, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: k }, async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i).catch((e) => (log("skip:", String(e.message ?? e).slice(0, 120)), null));
    }
  }));
  return out.filter(Boolean);
}

/** Files of a remote .tar.gz, streamed: yields { name, data } and stops downloading when the caller stops. */
async function* tarEntries(url) {
  const res = await fetch(url);
  const gunzip = (await import("node:zlib")).createGunzip();
  const { Readable } = await import("node:stream");
  const body = Readable.fromWeb(res.body);
  body.pipe(gunzip);
  let buf = Buffer.alloc(0);
  try {
    for await (const chunk of gunzip) {
      buf = Buffer.concat([buf, chunk]);
      while (buf.length >= 512) {
        const header = buf.subarray(0, 512);
        if (header.every((b) => b === 0)) return;
        const name = header.subarray(0, 100).toString().replace(/\0.*$/s, "");
        const size = parseInt(header.subarray(124, 136).toString().replace(/\0.*$/s, "").trim() || "0", 8);
        const total = 512 + Math.ceil(size / 512) * 512;
        if (buf.length < total) break;
        const type = String.fromCharCode(header[156]);
        if (type === "0" || type === "\0") yield { name, data: Buffer.from(buf.subarray(512, 512 + size)) };
        buf = buf.subarray(total);
      }
    }
  } finally {
    body.destroy();
    gunzip.destroy();
  }
}

/** Pixel size of a JPEG/PNG/WebP buffer (headers only). */
function imageSize(b) {
  if (b[0] === 0x89 && b[1] === 0x50) return [b.readUInt32BE(16), b.readUInt32BE(20)];
  if (b[0] === 0xff && b[1] === 0xd8) {
    let o = 2;
    while (o < b.length) {
      const marker = b[o + 1], len = b.readUInt16BE(o + 2);
      if (marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker)) return [b.readUInt16BE(o + 7), b.readUInt16BE(o + 5)];
      o += 2 + len;
    }
  }
  throw new Error("unknown image format");
}

function writeSet(set, rows) {
  writeFileSync(join(ROOT, set, "manifest.jsonl"), rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
  log(`${set}: ${rows.length} images, ${rows.reduce((a, r) => a + r.regions.length, 0)} regions`);
}

function prepare(set) {
  mkdirSync(join(ROOT, set, "images"), { recursive: true });
}

// ---------------------------------------------------------------- sets

const SETS = {
  async cagui() {
    const set = "cagui";
    prepare(set);
    const base = "https://huggingface.co/datasets/openbmb/CAGUI/resolve/main/CAGUI_grounding";
    const lines = (await get(`${base}/code/ocr.jsonl`, "text")).trim().split("\n").map((l) => JSON.parse(l));
    const rows = await pool(lines.slice(0, n(set)), 8, async (r) => {
      const name = r.image.split("/").pop();
      const buf = await get(`${base}/images/ocr/${name}`);
      writeFileSync(join(ROOT, set, "images", name), buf);
      const [width, height] = imageSize(buf);
      const [x1, y1, x2, y2] = r.abs_position.replace(/[<>]/g, "").split(",").map(Number);
      return {
        id: `cagui-${name.replace(/\.\w+$/, "")}`, set, image: `images/${name}`, width, height, lang: "zh",
        license: "CC BY-NC 4.0", source: "openbmb/CAGUI (CAGUI_grounding/ocr)",
        // the ground truth is an element's label, which is often not visible text (an avatar's or a back button's
        // accessibility name): only label-shaped boxes with a non-generic label count as likely visible
        regions: [{ box: [x1, y1, x2 - x1, y2 - y1], text: r.text, level: "element", visible: x2 - x1 >= 1.2 * (y2 - y1) && !/头像|图标|按钮|返回|图片|更多|搜索|关闭/.test(r.text) }],
      };
    });
    writeSet(set, rows);
  },

  async hiertext() {
    const set = "hiertext";
    prepare(set);
    const gt = JSON.parse(gunzipSync(await get("https://github.com/google-research-datasets/hiertext/raw/main/gt/validation.jsonl.gz")).toString());
    const bbox = (v) => {
      const xs = v.map((p) => p[0]), ys = v.map((p) => p[1]);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
    };
    // images with a moderate amount of legible text (not empty, not a dense page of prose)
    const picks = gt.annotations.filter((a) => {
      const paras = a.paragraphs.filter((p) => p.legible && p.lines.some((l) => l.text?.trim()));
      return paras.length >= 3 && paras.length <= 40;
    });
    // the images: HierText's own archive of Open Images (577 MB), streamed and read only until enough are found
    const want = new Map(picks.map((a) => [a.image_id, a]));
    const rows = [];
    for await (const { name, data } of tarEntries("https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz")) {
      const id = name.split("/").pop().replace(/\.jpg$/, "");
      const a = want.get(id);
      if (!a) continue;
      const [width, height] = imageSize(data);
      const sx = width / a.image_width, sy = height / a.image_height;
      const scale = (b) => [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy].map((v) => Math.round(v));
      writeFileSync(join(ROOT, set, "images", `${id}.jpg`), data);
      const regions = a.paragraphs
        .filter((p) => p.legible && p.lines.some((l) => l.text?.trim()))
        .map((p) => ({
          box: scale(bbox(p.vertices)),
          text: p.lines.map((l) => l.text).join("\n"),
          level: "paragraph",
          lines: p.lines.filter((l) => l.legible && l.text?.trim()).map((l) => ({ box: scale(bbox(l.vertices)), text: l.text, vertical: !!l.vertical })),
        }));
      rows.push({ id: `hiertext-${id}`, set, image: `images/${id}.jpg`, width, height, lang: "mixed", license: "annotations CC BY-SA 4.0; image Open Images (CC BY 2.0)", source: "google-research-datasets/hiertext (validation)", regions });
      if (rows.length >= n(set)) break;
    }
    writeSet(set, rows.slice(0, n(set)));
  },

  async openmantra() {
    const set = "openmantra";
    prepare(set);
    const base = "https://raw.githubusercontent.com/mantra-inc/open-mantra-dataset/main";
    const books = await get(`${base}/annotation.json`, "json");
    // pages spread evenly over the five series
    const pages = books.flatMap((b) => b.pages.map((p) => ({ book: b.book_title, ...p })));
    const per = Math.ceil(n(set) / books.length);
    const picks = books.flatMap((b) => {
      const ps = pages.filter((p) => p.book === b.book_title && p.text.length);
      const step = Math.max(1, Math.floor(ps.length / per));
      return ps.filter((_, i) => i % step === 0).slice(0, per);
    });
    const rows = await pool(picks.slice(0, n(set)), 6, async (p) => {
      const path = p.image_paths.ja;
      const name = `${p.book}-${String(p.page_index).padStart(3, "0")}.jpg`;
      const buf = await get(`${base}/${path}`);
      writeFileSync(join(ROOT, set, "images", name), buf);
      const [width, height] = imageSize(buf);
      return {
        id: `openmantra-${name.replace(/\.jpg$/, "")}`, set, image: `images/${name}`, width, height, lang: "ja",
        license: "CC BY-NC 4.0", source: "mantra-inc/open-mantra-dataset",
        regions: p.text.map((t) => ({ box: [t.x, t.y, t.w, t.h], text: t.text_ja, level: "bubble", refs: { en: t.text_en, zh: t.text_zh } })),
      };
    });
    writeSet(set, rows);
  },

  async vistra() {
    const set = "vistra";
    prepare(set);
    const base = "https://raw.githubusercontent.com/esalesky/vistra-benchmark/main";
    const list = await get("https://api.github.com/repos/esalesky/vistra-benchmark/contents/annotations", "json");
    const step = Math.max(1, Math.floor(list.length / n(set)));
    const picks = list.filter((_, i) => i % step === 0).slice(0, n(set));
    const rows = await pool(picks, 8, async (f) => {
      const a = await get(`${base}/annotations/${f.name}`, "json");
      const buf = await get(`${base}/images/${a.image_file}`);
      writeFileSync(join(ROOT, set, "images", a.image_file), buf);
      const [width, height] = imageSize(buf);
      return {
        id: `vistra-${a.image_file.replace(/\.\w+$/, "")}`, set, image: `images/${a.image_file}`, width, height, lang: "en",
        license: "CC BY-SA 4.0", source: "esalesky/vistra-benchmark",
        regions: a.transcript.map((text, k) => ({ box: null, text, level: "group", refs: Object.fromEntries(Object.entries(a.translation).map(([l, t]) => [l, t[k]])) })),
      };
    });
    writeSet(set, rows);
  },

  async websight() {
    const set = "websight";
    prepare(set);
    const browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1024, height: 900 } });
    const rows = [];
    for (let offset = 0; rows.length < n(set); offset += 20) {
      const data = await get(`https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/WebSight&config=v0.1&split=train&offset=${offset}&length=20`, "json");
      for (const { row_idx, row } of data.rows) {
        if (rows.length >= n(set)) break;
        try {
          await page.setContent(row.text, { waitUntil: "networkidle", timeout: 30_000 });
          // the first 2000 px of the page
          const height = Math.min(2000, await page.evaluate(() => document.documentElement.scrollHeight));
          await page.setViewportSize({ width: 1024, height });
          const styles = await page.evaluate(() => {
            const out = [];
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            for (let t = walker.nextNode(); t; t = walker.nextNode()) {
              const text = t.textContent.replace(/\s+/g, " ").trim();
              if (!text || !/\p{L}/u.test(text)) continue;
              const el = t.parentElement;
              const cs = getComputedStyle(el);
              if (cs.visibility === "hidden" || cs.display === "none" || parseFloat(cs.opacity) === 0) continue;
              const range = document.createRange();
              range.selectNodeContents(t);
              const r = range.getBoundingClientRect();
              if (r.width < 2 || r.height < 2 || r.bottom > innerHeight || r.right > innerWidth) continue;
              const fontSize = parseFloat(cs.fontSize), fontWeight = parseInt(cs.fontWeight);
              // visual alignment inside the nearest block container (CSS text-align says little for inline or
              // flex-centred elements)
              let block = el;
              while (block.parentElement && getComputedStyle(block).display.startsWith("inline")) block = block.parentElement;
              const bs = getComputedStyle(block), br = block.getBoundingClientRect();
              const l = br.left + parseFloat(bs.paddingLeft), rr = br.right - parseFloat(bs.paddingRight);
              const gapL = r.left - l, gapR = rr - r.right;
              const align = Math.abs(gapL - gapR) <= 3 && gapL > 3 ? "center" : gapL <= 3 ? "left" : gapR <= 3 ? "right" : cs.textAlign;
              out.push({
                box: [r.left, r.top, r.width, r.height].map(Math.round), text, fontSize, fontWeight, color: cs.color, align,
                lines: range.getClientRects().length,
                key: `${fontSize}/${fontWeight}/${cs.color}`,
              });
            }
            return out;
          });
          if (styles.length < 4) continue;
          const name = `ws${row_idx}.png`;
          await page.screenshot({ path: join(ROOT, set, "images", name) });
          rows.push({
            id: `websight-${row_idx}`, set, image: `images/${name}`, width: 1024, height, lang: "en", license: "CC BY 4.0",
            source: "HuggingFaceM4/WebSight v0.1 (rendered at 1024 px)",
            regions: styles.map((s) => ({ box: s.box, text: s.text, level: "run" })),
            styles,
          });
        } catch (e) {
          log("skip websight", row_idx, String(e.message ?? e).slice(0, 80));
        }
      }
    }
    await browser.close();
    writeSet(set, rows);
  },
};

mkdirSync(ROOT, { recursive: true });
for (const set of only ?? Object.keys(SETS)) {
  log("building", set);
  await SETS[set]();
}
writeFileSync(join(ROOT, "BUILT.txt"), `built ${new Date().toISOString()} by eval/build.mjs${existsSync(join(ROOT, "BUILT.txt")) ? "" : ""}\n`);
