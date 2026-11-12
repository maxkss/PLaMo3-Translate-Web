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

const args = process.argv.slice(2);
// --holdout: other images of the same sources, into eval/holdout/ — never looked at while changing the code, to tell
// a rule that generalises from one fitted to the images it was tuned on
const HOLD = args.includes("--holdout");
const ROOT = join(dirname(fileURLToPath(import.meta.url)), HOLD ? "holdout" : "data");
const only = args.find((a) => a.startsWith("--sets="))?.slice(7).split(",");
const nArg = Number(args.find((a) => a.startsWith("--n="))?.slice(4) ?? 0);
const DEFAULT_N = { cagui: 400, hiertext: 80, openmantra: 60, vistra: 100, websight: 40, crello: 80 };
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
    const rows = await pool(lines.slice(HOLD ? 400 : 0, (HOLD ? 400 : 0) + n(set)), 8, async (r) => {
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
    let skipped = 0;
    for await (const { name, data } of tarEntries("https://open-images-dataset.s3.amazonaws.com/ocr/validation.tgz")) {
      const id = name.split("/").pop().replace(/\.jpg$/, "");
      const a = want.get(id);
      if (!a) continue;
      // the hold-out set starts after the 80 images of the main one
      if (HOLD && ++skipped <= 80) continue;
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
      return ps.filter((_, i) => i % step === (HOLD ? step >> 1 : 0)).slice(0, per);
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
    const picks = list.filter((_, i) => i % step === (HOLD ? step >> 1 : 0)).slice(0, n(set));
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
    for (let offset = HOLD ? 3000 : 0; rows.length < n(set); offset += 20) {
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

  /**
   * Designed graphics — posters, flyers, social-media posts, banners — from Crello's layered templates. Every element
   * comes as its own image, so each template is composited here twice: with its text (images/) and without it
   * (clean/, the exact background an eraser should restore). masks/ holds the text's own pixels (its alpha), and
   * every text element carries its font size, alignment, colour and line count.
   */
  async crello() {
    const set = "crello";
    prepare(set);
    for (const d of ["clean", "masks"]) mkdirSync(join(ROOT, set, d), { recursive: true });
    const api = "https://datasets-server.huggingface.co";
    const info = await get(`${api}/info?dataset=cyberagent/crello`, "json");
    const f = info.dataset_info.default.features;
    const FORMATS = f.format.names, FONTS = f.font[0].names, ALIGN = f.text_align[0].names, TYPES = f.type[0].names;
    const split = HOLD ? "validation" : "test";
    const total = info.dataset_info.default.splits[split].num_examples;
    const browser = await chromium.launch();
    const page = await browser.newPage();
    const rows = [];
    const perFormat = new Map();
    // pages of 10 templates spread over the split; at most a fifth of the set from one format
    const PAGE = 10, cap = Math.max(2, Math.ceil(n(set) / 5));
    const offsets = Array.from({ length: Math.floor(total / PAGE) }, (_, k) => k * PAGE);
    for (let k = 0, step = 7; rows.length < n(set) && k < offsets.length; k++) {
      const offset = offsets[(k * step) % offsets.length];
      let data;
      try {
        data = await get(`${api}/rows?dataset=cyberagent/crello&config=default&split=${split}&offset=${offset}&length=${PAGE}`, "json");
      } catch (e) {
        log("skip crello page", offset, String(e.message ?? e).slice(0, 80));
        continue;
      }
      for (const { row } of data.rows) {
        if (rows.length >= n(set)) break;
        const format = FORMATS[row.format];
        const texts = row.type.map((t, i) => (TYPES[t] === "TextElement" && row.text[i]?.trim() ? i : -1)).filter((i) => i >= 0);
        // axis-aligned text only (the OCR has no rotated boxes), and enough of it to say something about layout
        if (texts.length < 2 || texts.some((i) => Math.abs(row.angle[i]) > 0.5) || (perFormat.get(format) ?? 0) >= cap) continue;
        if (Math.min(row.canvas_width, row.canvas_height) < 300) continue;
        try {
          const els = await pool(row.type.map((_, i) => i), 6, async (i) => {
            const src = row.image[i]?.src;
            if (!src) return null;
            return { i, url: `data:image/png;base64,${(await get(src)).toString("base64")}` };
          });
          if (els.length !== row.type.length) continue; // an element failed to download: the composite would be wrong
          els.sort((a, b) => a.i - b.i);
          const scale = Math.min(1, 1600 / Math.max(row.canvas_width, row.canvas_height));
          const out = await page.evaluate(async ({ els, row, texts, scale }) => {
            const W = Math.round(row.canvas_width * scale), H = Math.round(row.canvas_height * scale);
            const mk = () => {
              const c = document.createElement("canvas");
              c.width = W;
              c.height = H;
              return c;
            };
            const full = mk(), clean = mk(), mask = mk();
            const draw = (c, bmp, i, opaque) => {
              const g = c.getContext("2d");
              g.save();
              g.globalAlpha = opaque ? 1 : (row.opacity[i] ?? 1);
              const x = row.left[i] * scale, y = row.top[i] * scale, w = row.width[i] * scale, h = row.height[i] * scale;
              g.translate(x + w / 2, y + h / 2);
              g.rotate(row.angle[i] ?? 0);
              g.drawImage(bmp, -w / 2, -h / 2, w, h);
              g.restore();
            };
            for (const c of [full, clean]) {
              const g = c.getContext("2d");
              g.fillStyle = "#fff";
              g.fillRect(0, 0, W, H);
            }
            const inks = {};
            for (const { i, url } of els) {
              const bmp = await createImageBitmap(await (await fetch(url)).blob());
              const isText = texts.includes(i);
              draw(full, bmp, i);
              if (!isText) {
                draw(clean, bmp, i);
                // an element drawn over text hides it: take it out of the mask too
                const g = mask.getContext("2d");
                g.save();
                g.globalCompositeOperation = "destination-out";
                draw(mask, bmp, i);
                g.restore();
                continue;
              }
              draw(mask, bmp, i);
              // the text's own ink: bounding box of its alpha, and of each row of ink (lines)
              const one = mk();
              draw(one, bmp, i, true);
              const d = one.getContext("2d").getImageData(0, 0, W, H).data;
              let x0 = W, y0 = H, x1 = -1, y1 = -1;
              for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
                if (d[(y * W + x) * 4 + 3] < 48) continue;
                if (x < x0) x0 = x;
                if (x > x1) x1 = x;
                if (y < y0) y0 = y;
                if (y > y1) y1 = y;
              }
              if (x1 >= 0) inks[i] = [x0, y0, x1 - x0 + 1, y1 - y0 + 1];
            }
            const png = (c) => c.toDataURL("image/png").split(",")[1];
            return { W, H, full: png(full), clean: png(clean), mask: png(mask), inks };
          }, { els, row, texts, scale });
          const regions = [], styles = [];
          for (const i of texts) {
            const box = out.inks[i];
            if (!box || box[2] < 4 || box[3] < 4) continue;
            const text = row.text[i].replace(/\s*\n\s*/g, "\n").trim();
            const nLines = row.text_line[i].length ? Math.max(...row.text_line[i]) + 1 : text.split("\n").length;
            const fontSize = +(row.font_size[i] * scale).toFixed(2);
            const bold = row.font_bold[i].filter(Boolean).length > row.font_bold[i].length / 2;
            const color = row.text_color[i][0] ?? "";
            const s = { box, text: row.capitalize[i] ? text.toUpperCase() : text, fontSize, fontWeight: bold ? 700 : 400, color, align: ALIGN[row.text_align[i]] || "left", lines: nLines || 1, font: FONTS[row.font[i]], key: `${fontSize}/${bold}/${color}` };
            styles.push(s);
            regions.push({ box, text: s.text, level: "element" });
          }
          if (regions.length < 2) continue;
          const name = `${row.id}.png`;
          writeFileSync(join(ROOT, set, "images", name), Buffer.from(out.full, "base64"));
          writeFileSync(join(ROOT, set, "clean", name), Buffer.from(out.clean, "base64"));
          writeFileSync(join(ROOT, set, "masks", name), Buffer.from(out.mask, "base64"));
          perFormat.set(format, (perFormat.get(format) ?? 0) + 1);
          rows.push({ id: `crello-${row.id}`, set, image: `images/${name}`, clean: `clean/${name}`, mask: `masks/${name}`, width: out.W, height: out.H, lang: "mixed", format, license: "CDLA-Permissive-2.0", source: `cyberagent/crello (${split})`, regions, styles });
        } catch (e) {
          log("skip crello", row.id, String(e.message ?? e).slice(0, 100));
        }
      }
    }
    await browser.close();
    log("crello formats:", JSON.stringify(Object.fromEntries(perFormat)));
    writeSet(set, rows);
  },
};

mkdirSync(ROOT, { recursive: true });
for (const set of only ?? Object.keys(SETS)) {
  log("building", set);
  await SETS[set]();
}
writeFileSync(join(ROOT, "BUILT.txt"), `built ${new Date().toISOString()} by eval/build.mjs${existsSync(join(ROOT, "BUILT.txt")) ? "" : ""}\n`);
