/**
 * Evaluates image translation on the set built by eval/build.mjs. Writes eval/results/<time>-<size>.json and prints
 * a summary per set.
 *
 * Metrics (boxes in image pixels; a ground-truth region is "found" when OCR paragraphs cover ≥ 50% of its box):
 *   recall       share of ground-truth regions found                                      cagui hiertext openmantra websight
 *                (cagui: only labels that are likely visible text — see eval/build.mjs)
 *   precision    share of OCR paragraphs lying ≥ 50% inside some ground-truth region      hiertext openmantra websight
 *   split        ground-truth paragraphs/bubbles covered by more than one OCR paragraph   hiertext openmantra
 *   merge        OCR paragraphs spanning more than one ground-truth paragraph/bubble      hiertext openmantra
 *   cer          character error rate of the text read for found regions (NFKC, no spaces, case-folded)
 *   textRecall   vistra (no boxes): 1 − best CER of each transcript group against the OCR text
 *   chrF         translation of the ground-truth source text against the reference       openmantra ja→en, ja→zh
 *                (character n-grams 1–6, β = 2, as sacreBLEU's chrF)                      vistra en→zh
 *   typography   websight, through the real page overlay (→ ko): size spread within each source style, anchor
 *                error by the source's text-align (em), collisions between translated texts
 *
 * Run: node eval/run.mjs [size=0.8b|2b] [--ocr=mobile|server] [--sets=…] [--limit=N] [--no-translate] [--no-typography]
 *      (PROFILE=<dir> with the models installed; the local model server as in test/lib.mjs)
 */
import { createServer } from "node:http";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { launch } from "../test/lib.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
// --data=holdout evaluates the hold-out set (node eval/build.mjs --holdout)
const DATA = join(HERE, process.argv.find((a) => a.startsWith("--data="))?.slice(7) ?? "data");
const args = process.argv.slice(2);
const size = args.find((a) => /^(0\.8b|2b)$/.test(a)) ?? "0.8b";
const sets = args.find((a) => a.startsWith("--sets="))?.slice(7).split(",") ?? ["cagui", "hiertext", "openmantra", "vistra", "websight", "crello"];
const limit = Number(args.find((a) => a.startsWith("--limit="))?.slice(8) ?? 0);
const doTranslate = !args.includes("--no-translate");
const doTypography = !args.includes("--no-typography");
const ocrModel = args.find((a) => a.startsWith("--ocr="))?.slice(6) ?? "mobile";
// screenshots of the first N translated images of each typography set, to eval/results/shots-<tag>/
// --tag=<name> labels the run (report file name, screenshot folder)
const tag = args.find((a) => a.startsWith("--tag="))?.slice(6) ?? "run";
// --overlay=N: translate N images of each set without style ground truth through the page overlay as well
const overlay = Number(args.find((a) => a.startsWith("--overlay="))?.slice(10) ?? 0);
const shots = Number(args.find((a) => a.startsWith("--shots="))?.slice(8) ?? 0);
const log = (...a) => console.log(`[${new Date().toISOString().slice(11, 19)}]`, ...a);

// ---------------------------------------------------------------- metrics

const norm = (t) => t.normalize("NFKC").toLowerCase().replace(/\s+/g, "");
function levenshtein(a, b) {
  const A = [...a], B = [...b];
  let prev = Array.from({ length: B.length + 1 }, (_, j) => j);
  for (let i = 1; i <= A.length; i++) {
    const cur = [i];
    for (let j = 1; j <= B.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] === B[j - 1] ? 0 : 1));
    prev = cur;
  }
  return prev[B.length];
}
/**
 * Edit distance of `a` against the best-matching stretch of `b` (free start and end in b): an OCR paragraph may hold
 * more text than the ground-truth region, which is not an error in reading that region.
 */
function substringDistance(a, b) {
  const A = [...a], B = [...b];
  let prev = new Array(B.length + 1).fill(0);
  for (let i = 1; i <= A.length; i++) {
    const cur = [i];
    for (let j = 1; j <= B.length; j++) cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] === B[j - 1] ? 0 : 1));
    prev = cur;
  }
  return Math.min(...prev);
}
const cer = (gt, hyp) => (norm(gt).length ? Math.min(1, substringDistance(norm(gt), norm(hyp)) / [...norm(gt)].length) : 0);

/** chrF (sacreBLEU default: character n-grams 1–6, β = 2, whitespace removed), corpus-level over segment pairs. */
function chrF(pairs) {
  const N = 6, beta = 2;
  const stats = Array.from({ length: N }, () => ({ match: 0, hyp: 0, ref: 0 }));
  const grams = (s, n) => {
    const m = new Map();
    const c = [...s.replace(/\s+/g, "")];
    for (let i = 0; i + n <= c.length; i++) {
      const g = c.slice(i, i + n).join("");
      m.set(g, (m.get(g) ?? 0) + 1);
    }
    return m;
  };
  for (const [hyp, ref] of pairs) {
    for (let n = 1; n <= N; n++) {
      const h = grams(hyp, n), r = grams(ref, n);
      let match = 0;
      for (const [g, c] of h) match += Math.min(c, r.get(g) ?? 0);
      stats[n - 1].match += match;
      stats[n - 1].hyp += [...h.values()].reduce((a, b) => a + b, 0);
      stats[n - 1].ref += [...r.values()].reduce((a, b) => a + b, 0);
    }
  }
  let p = 0, r = 0, k = 0;
  for (const s of stats) {
    if (!s.hyp || !s.ref) continue;
    p += s.match / s.hyp;
    r += s.match / s.ref;
    k++;
  }
  if (!k) return 0;
  p /= k;
  r /= k;
  return p + r ? +((100 * (1 + beta ** 2) * p * r) / (beta ** 2 * p + r)).toFixed(1) : 0;
}

const area = (b) => Math.max(0, b[2]) * Math.max(0, b[3]);
const inter = (a, b) => Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
const mean = (xs) => (xs.length ? +(xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(3) : null);
const median = (xs) => (xs.length ? +[...xs].sort((a, b) => a - b)[xs.length >> 1].toFixed(3) : null);
const share = (xs, f) => (xs.length ? +(xs.filter(f).length / xs.length).toFixed(3) : null);

/** Detection and recognition of one image against its boxed ground truth. */
function scoreBoxes(item, ocr) {
  const gts = item.regions.filter((g) => g.box && g.visible !== false);
  const paras = ocr.paragraphs;
  const out = { gt: gts.length, found: 0, cer: [], split: 0, merge: 0, ocr: paras.length, precise: 0 };
  for (const g of gts) {
    // OCR paragraphs touching the region, in reading order
    const hits = paras.filter((p) => inter(p.box, g.box) > 0.25 * Math.min(area(p.box), area(g.box)));
    const covered = hits.reduce((a, p) => a + inter(p.box, g.box), 0) / Math.max(1, area(g.box));
    if (Math.min(1, covered) < 0.5) continue;
    out.found++;
    out.cer.push(cer(g.text, hits.map((p) => p.text).join("")));
    if (hits.length > 1) out.split++;
  }
  for (const p of paras) {
    const touching = gts.filter((g) => inter(p.box, g.box) > 0.25 * Math.min(area(p.box), area(g.box)));
    if (touching.length > 1) out.merge++;
    if (gts.some((g) => inter(p.box, g.box) >= 0.5 * area(p.box))) out.precise++;
  }
  return out;
}

// ---------------------------------------------------------------- engine access (from the extension page)

const ocrOn = (ext, src) =>
  ext.evaluate(async (src) => {
    await chrome.runtime.sendMessage({ type: "ensure-engine" });
    const port = (window.__port ??= chrome.runtime.connect({ name: "plamo-engine" }));
    const id = "ocr-" + Math.random();
    return new Promise((done) => {
      const on = (m) => {
        if (m.id !== id) return;
        if (m.type === "ocr-done" || m.type === "error") {
          port.onMessage.removeListener(on);
          done(m.type === "ocr-done" ? { paragraphs: m.result.paragraphs.map((p) => ({ box: p.box, text: p.text, fontPx: p.fontPx, bold: p.bold, lines: p.lines.length, inkBox: p.inkBox })), ms: m.result.ms } : { error: m.message });
        }
      };
      port.onMessage.addListener(on);
      port.postMessage({ type: "ocr", id, src, targetLang: "ko", ocrModel: window.__ocrModel });
    });
  }, src);

const translateOn = (ext, texts, targetLang) =>
  ext.evaluate(async ([texts, targetLang]) => {
    const port = window.__port;
    const out = new Array(texts.length).fill("");
    const waits = new Map();
    const on = (m) => (m.type === "done" || m.type === "error") && waits.get(m.id)?.(m.type === "done" ? m.text : "");
    port.onMessage.addListener(on);
    await Promise.all(
      texts.map((text, i) => new Promise((r) => {
        const id = "t-" + i + "-" + Math.random();
        waits.set(id, (t) => ((out[i] = t), r()));
        port.postMessage({ type: "translate", id, text, targetLang, priority: 5, stream: false });
      })),
    );
    port.onMessage.removeListener(on);
    return out;
  }, [texts, targetLang]);

// ---------------------------------------------------------------- run

const TYPES = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp" };
const srv = createServer((req, res) => {
  const path = join(DATA, decodeURIComponent(req.url.split("?")[0]));
  if (!path.startsWith(DATA) || !existsSync(path)) return res.writeHead(404).end();
  res.writeHead(200, { "Content-Type": TYPES[extname(path).toLowerCase()] ?? "application/octet-stream" }).end(readFileSync(path));
}).listen(8789);
const pageSrv = createServer((req, res) => {
  const img = new URL(req.url, "http://x").searchParams.get("img");
  res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" }).end(`<!doctype html><meta charset="utf-8"><body style="margin:0"><img id="im" src="http://127.0.0.1:8789/${img}" style="display:block;max-width:1060px;max-height:2000px">`);
}).listen(8788);

const { ctx, sw, opts: ext } = await launch({ size, viewport: { width: 1100, height: 2100 } });
const report = { size, ocrModel, date: new Date().toISOString(), sets: {} };
try {
  await sw.evaluate((m) => chrome.storage.local.set({ targetLang: "ko", ocrModel: m }), ocrModel);
  await ext.evaluate((m) => (window.__ocrModel = m), ocrModel);
  for (const set of sets) {
    const file = join(DATA, set, "manifest.jsonl");
    if (!existsSync(file)) {
      log(`${set}: not built (node eval/build.mjs --sets=${set})`);
      continue;
    }
    let items = readFileSync(file, "utf8").trim().split("\n").map((l) => JSON.parse(l));
    if (limit) items = items.slice(0, limit);
    log(`${set}: ${items.length} images`);
    const per = [];
    for (const item of items) {
      const ocr = await ocrOn(ext, `http://127.0.0.1:8789/${set}/${item.image}`);
      if (ocr.error) {
        per.push({ id: item.id, error: ocr.error });
        continue;
      }
      const r = { id: item.id, ocrMs: ocr.ms };
      if (set === "vistra") {
        const all = ocr.paragraphs.map((p) => p.text);
        // best match of each transcript group: a paragraph or the whole OCR text
        r.textRecall = item.regions.map((g) => 1 - Math.min(...[all.join(""), ...all].map((t) => cer(g.text, t))));
      } else {
        r.boxes = scoreBoxes(item, ocr);
      }
      // style read-back (sets with rendered styles): a source run covered by exactly one OCR paragraph that covers
      // nothing else — the estimated font size over the true one, and whether bold was seen
      if (item.styles) {
        r.font = [];
        for (const s of item.styles) {
          const hits = ocr.paragraphs.filter((p) => inter(p.box, s.box) > 0.5 * Math.min(area(p.box), area(s.box)));
          if (hits.length !== 1 || item.styles.filter((t) => inter(hits[0].box, t.box) > 0.3 * area(t.box)).length !== 1) continue;
          r.font.push({ ratio: hits[0].fontPx / s.fontSize, bold: hits[0].bold === s.fontWeight >= 600, size: s.fontSize, text: s.text.slice(0, 30), font: s.font });
        }
      }
      per.push(r);
    }
    const ok = per.filter((r) => !r.error);
    const sum = (k) => ok.reduce((a, r) => a + (r.boxes?.[k] ?? 0), 0);
    const summary = { images: items.length, errors: per.length - ok.length, ocrMs: mean(ok.map((r) => r.ocrMs)) };
    if (set === "vistra") summary.textRecall = mean(ok.flatMap((r) => r.textRecall));
    else {
      summary.recall = +(sum("found") / Math.max(1, sum("gt"))).toFixed(3);
      summary.cer = mean(ok.flatMap((r) => r.boxes.cer));
      if (set !== "cagui") {
        summary.precision = +(sum("precise") / Math.max(1, sum("ocr"))).toFixed(3);
        summary.split = +(sum("split") / Math.max(1, sum("found"))).toFixed(3);
        summary.merge = +(sum("merge") / Math.max(1, sum("ocr"))).toFixed(3);
      }
    }

    if (items[0]?.styles) {
      const f = ok.flatMap((r) => r.font ?? []);
      summary.font = { n: f.length, ratio: median(f.map((x) => x.ratio)), within10: share(f, (x) => x.ratio >= 0.9 && x.ratio <= 1.1), within20: share(f, (x) => x.ratio >= 0.8 && x.ratio <= 1.2), bold: share(f, (x) => x.bold) };
    }

    // translation quality of the ground-truth text against professional references
    if (doTranslate && (set === "openmantra" || set === "vistra")) {
      const src = items.flatMap((i) => i.regions);
      // OpenMantra's Chinese references are Traditional, Vistra's Simplified
      const targets = set === "openmantra" ? ["en", "zh-hant"] : ["zh-hans"];
      summary.chrF = {};
      for (const lang of targets) {
        const refKey = lang.startsWith("zh") ? "zh" : lang;
        const pairs = src.filter((g) => g.refs?.[refKey]);
        const t0 = performance.now();
        const hyp = await translateOn(ext, pairs.map((g) => g.text), lang);
        summary.chrF[`${set === "openmantra" ? "ja" : "en"}→${refKey}`] = chrF(hyp.map((h, k) => [h, pairs[k].refs[refKey]]));
        summary[`translateSeconds_${refKey}`] = +((performance.now() - t0) / 1000).toFixed(1);
        report.samples ??= {};
        report.samples[`${set}-${refKey}`] = pairs.slice(0, 8).map((g, k) => ({ src: g.text, hyp: hyp[k], ref: g.refs[refKey] }));
      }
    }

    // typography through the real page overlay
    // (sets without style ground truth: --overlay=N translates N images spread over the set, for the screenshots and
    // the measures that need no ground truth)
    if (doTypography && (items[0]?.styles || overlay)) {
      const page = await ctx.newPage();
      const typo = [];
      // --e2e: sets with English references are translated into English, and what the overlay shows is scored
      const e2e = args.includes("--e2e") && items[0]?.regions[0]?.refs?.en;
      if (e2e) await sw.evaluate(() => chrome.storage.local.set({ targetLang: "en" }));
      const every = items[0]?.styles ? 1 : Math.max(1, Math.floor(items.length / overlay));
      for (const item of items.filter((_, k) => k % every === 0).slice(0, items[0]?.styles ? items.length : overlay)) {
        await page.goto(`http://localhost:8788/?img=${set}/${item.image}`);
        await page.waitForFunction(() => document.images[0]?.complete && document.images[0].naturalWidth);
        await page.locator("#im").dispatchEvent("contextmenu");
        await sw.evaluate(async (s) => {
          const [tab] = await chrome.tabs.query({ url: "http://localhost:8788/*" });
          await chrome.tabs.sendMessage(tab.id, { type: "translate-image", srcUrl: s });
        }, await page.$eval("#im", (i) => i.currentSrc));
        const finished = await page.waitForFunction(() => document.querySelector("plamo-translate-ui")?.shadowRoot?.querySelector(".img-ov")?.dataset.timing?.includes("total") || /없습니다|못했습니다/.test(document.querySelector("plamo-translate-ui")?.shadowRoot?.querySelector(".img-badge .label")?.textContent ?? ""), null, { timeout: 90_000, polling: 200 }).then(() => true, () => false);
        // an overlay that never finishes is a failure of the build under test, not of the run
        if (!finished) {
          log("overlay did not finish:", item.id);
          (summary.overlayFailed ??= []).push(item.id);
          continue;
        }
        const out = await page.evaluate(() => {
          const im = document.getElementById("im");
          const ir = im.getBoundingClientRect();
          // everything in the image's own pixels (the page may show it smaller)
          const k = im.naturalWidth / ir.width;
          return [...document.querySelector("plamo-translate-ui").shadowRoot.querySelectorAll(".img-para")].map((p) => {
            const span = p.querySelector("span");
            const range = document.createRange();
            range.selectNodeContents(span);
            const r = range.getBoundingClientRect();
            const lines = new Set([...range.getClientRects()].map((c) => Math.round(c.top))).size;
            return { box: [(r.left - ir.left) * k, (r.top - ir.top) * k, r.width * k, r.height * k], font: parseFloat(getComputedStyle(span).fontSize), lines, text: span.textContent, src: p.title.split("\n")[0], align: /align=(\w+)/.exec(p.dataset.style ?? "")?.[1], em: parseFloat(/em=([\d.]+)/.exec(p.dataset.style ?? "")?.[1] ?? "0"), ink: /ink=([\d.,-]+)/.exec(p.dataset.style ?? "")?.[1].split(",").map(Number) };
          });
        });
        if (typo.length < shots) {
          mkdirSync(join(HERE, "results", `shots-${tag}`), { recursive: true });
          await page.locator("#im").screenshot({ path: join(HERE, "results", `shots-${tag}`, `${item.id}.png`) });
        }
        // match each source run to the translated paragraph covering most of it
        const matched = (item.styles ?? []).map((s) => {
          let best = null, bv = 0;
          for (const o of out) {
            const v = inter(s.box, o.box) / Math.max(1, area(s.box));
            if (v > bv) (bv = v), (best = o);
          }
          return bv > 0.2 ? { s, o: best } : null;
        }).filter(Boolean);
        const byKey = {};
        for (const m of matched) (byKey[m.s.key] ??= []).push(m);
        const spreads = Object.values(byKey).filter((ms) => ms.length > 1).map((ms) => {
          const f = ms.map((m) => m.o.font);
          return (Math.max(...f) - Math.min(...f)) / f.sort((a, b) => a - b)[f.length >> 1];
        });
        const anchor = matched.filter((m) => m.s.lines === 1).map((m) => {
          const at = (b, a) => (a === "center" ? b[0] + b[2] / 2 : a === "right" || a === "end" ? b[0] + b[2] : b[0]);
          return Math.abs(at(m.o.box, m.s.align) - at(m.s.box, m.s.align)) / m.s.fontSize;
        });
        let collisions = 0;
        const clashes = [];
        for (let a = 0; a < out.length; a++) {
          for (let b = a + 1; b < out.length; b++) {
            if (inter(out[a].box, out[b].box) <= 0.02 * Math.min(area(out[a].box), area(out[b].box))) continue;
            collisions++;
            clashes.push([out[a].text.slice(0, 20), out[b].text.slice(0, 20)]);
          }
        }
        // one translated paragraph per source run (a paragraph covering several runs says nothing about any one of them)
        const single = matched.filter((m) => matched.filter((x) => x.o === m.o).length === 1);
        // size against the source's: the translation's font size over the original's
        const sizeRatio = single.map((m) => m.o.font / m.s.fontSize);
        // the translation stays inside the image, and on as many lines as the original
        const outside = out.filter((o) => o.box[0] < -1 || o.box[1] < -1 || o.box[0] + o.box[2] > item.width + 1 || o.box[1] + o.box[3] > item.height + 1).length;
        const sameLines = single.filter((m) => m.o.lines === m.s.lines).length;
        // its area against the original's: how much more (or less) of the design the text now takes
        const areaRatio = single.map((m) => area(m.o.box) / Math.max(1, area(m.s.box)));
        typo.push({
          id: item.id, matched: matched.length, runs: (item.styles ?? []).length, paragraphs: out.length, shrunk: out.filter((o) => o.em && o.font / o.em < 0.6).length, texts: out.map((o) => ({ ink: o.ink, text: o.text })), regions: item.regions, words: out.reduce((a, o) => a + o.text.length, 0), sizeSpread: mean(spreads), anchorErr: mean(anchor),
          anchorP95: anchor.sort((a, b) => a - b)[Math.floor(anchor.length * 0.95)] ?? null, collisions, sizeRatio, areaRatio, outside, sameLines, single: single.length,
          detail: single.map((m) => ({ src: m.s.text.slice(0, 40), out: m.o.text.slice(0, 40), size: +(m.o.font / m.s.fontSize).toFixed(2), align: [m.s.align, m.o.align], lines: [m.s.lines, m.o.lines] })),
          // pairs of colliding translations, for looking into them
          clashes,
        });
        await page.locator("#im").dispatchEvent("contextmenu");
      }
      await page.close();
      summary.typography = {
        coverage: +(typo.reduce((a, t) => a + t.matched, 0) / Math.max(1, typo.reduce((a, t) => a + t.runs, 0))).toFixed(3),
        sizeSpread: mean(typo.map((t) => t.sizeSpread).filter((x) => x != null)),
        anchorErrEm: mean(typo.map((t) => t.anchorErr).filter((x) => x != null)),
        anchorP95Em: mean(typo.map((t) => t.anchorP95).filter((x) => x != null)),
        collisions: typo.reduce((a, t) => a + t.collisions, 0),
        // size fidelity: median translated/original font size, the share within ±15% of the original and below 70%
        sizeRatio: median(typo.flatMap((t) => t.sizeRatio)),
        sizeWithin15: share(typo.flatMap((t) => t.sizeRatio), (r) => r >= 0.85 && r <= 1.15),
        sizeBelow70: share(typo.flatMap((t) => t.sizeRatio), (r) => r < 0.7),
        // the same size to the eye: Hangul is as tall as a Latin capital at 0.87× the font size (image-view.ts), so a
        // Latin source run translated into Korean is on target at 0.87; the share within ±15% of that
        sizeOnTarget: share(typo.flatMap((t) => t.detail), (d) => {
          const target = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u.test(d.src) ? 1 : 0.87;
          return d.size >= 0.85 * target && d.size <= 1.15 * target;
        }),
        areaRatio: median(typo.flatMap((t) => t.areaRatio)),
        sameLines: +(typo.reduce((a, t) => a + t.sameLines, 0) / Math.max(1, typo.reduce((a, t) => a + t.single, 0))).toFixed(3),
        // alignment read against the source's (start = left, end = right), over single runs
        alignAcc: share(typo.flatMap((t) => t.detail), (d) => ({ start: "left", center: "center", end: "right" })[d.align[1]] === ({ justify: "left", start: "left", end: "right" }[d.align[0]] ?? d.align[0])),
        outside: typo.reduce((a, t) => a + t.outside, 0),
        paragraphs: typo.reduce((a, t) => a + t.paragraphs, 0),
      };
      // what needs no ground truth: translations running into each other, out of the image, or shrunk below 60% of
      // the size read from the original
      summary.overlay = { images: typo.length, paragraphs: typo.reduce((a, t) => a + t.paragraphs, 0), collisions: summary.typography.collisions, outside: summary.typography.outside, shrunk: typo.reduce((a, t) => a + t.shrunk, 0) };
      if (!items[0]?.styles) delete summary.typography;
      if (e2e) {
        await sw.evaluate(() => chrome.storage.local.set({ targetLang: "ko" }));
        // each reference bubble against the translations shown over it, in reading order (columns: right to left)
        const pairs = [];
        let shown = 0;
        for (const t of typo) {
          for (const g of t.regions.filter((g) => g.box && g.refs?.en)) {
            const hits = t.texts.filter((o) => o.ink && inter(o.ink, g.box) > 0.5 * area(o.ink)).sort((a, b) => b.ink[0] - a.ink[0]);
            if (hits.length) shown++;
            pairs.push([hits.map((o) => o.text).join(" "), g.refs.en]);
          }
        }
        summary.e2e = { bubbles: pairs.length, shown, chrF: chrF(pairs), chrFShown: chrF(pairs.filter((p) => p[0])) };
      }
      for (const t of typo) delete t.regions;
      (report.typography ??= {})[set] = typo;
    }
    report.sets[set] = { summary, items: per };
    log(set, JSON.stringify(summary));
  }
  mkdirSync(join(HERE, "results"), { recursive: true });
  const out = join(HERE, "results", `${report.date.replace(/[:.]/g, "-")}-${size}-ocr-${ocrModel}-${tag}.json`);
  writeFileSync(out, JSON.stringify(report, null, 1));
  log("report:", out);
} finally {
  srv.close();
  pageSrv.close();
  await ctx.close();
}
