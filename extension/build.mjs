// Bundles the extension into dist/. `node build.mjs --watch` rebuilds on change.
import * as esbuild from "esbuild";
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";

const watch = process.argv.includes("--watch");
const out = "dist";
rmSync(out, { recursive: true, force: true });
mkdirSync(`${out}/ort`, { recursive: true });

// static files
cpSync("src/manifest.json", `${out}/manifest.json`);
cpSync("src/offscreen.html", `${out}/offscreen.html`);
for (const f of ["popup.html", "options.html", "pages.css"]) cpSync(`src/pages/${f}`, `${out}/${f}`);
cpSync("public", out, { recursive: true });
// onnxruntime-web WebGPU build (asyncify) runtime files
for (const f of ["ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.mjs"]) {
  cpSync(`node_modules/onnxruntime-web/dist/${f}`, `${out}/ort/${f}`);
}

/**
 * Two small patches to onnxruntime-web 1.31's JS glue so WebGPU graph capture can be used for decode while
 * prefill runs in the same session (see src/engine/generator.ts):
 *  1. buffers flagged `__plamoKeep` are not unregistered after each run (ORT would release their handles,
 *     and the captured graph would replay with dangling buffers);
 *  2. when `globalThis.__plamoRebind` is set, inputs/outputs are re-bound even though a capture session
 *     normally binds them only on its first run.
 * Without the flags, behaviour is unchanged.
 */
const ortPatch = {
  name: "ort-graph-capture-patch",
  setup(build) {
    build.onLoad({ filter: /onnxruntime-web[\\/]dist[\\/]ort\.webgpu\.bundle\.min\.mjs$/ }, async (args) => {
      let src = readFileSync(args.path, "utf8");
      const unreg = /(\w+)\.webgpuUnregisterBuffer\((\w+)\[2\]\.gpuBuffer\)/g;
      const bind = /if\((\w+)&&!(\w+)\)\{let\{handle:(\w+),outputPreferredLocations/;
      const nUnreg = src.match(unreg)?.length ?? 0;
      if (nUnreg !== 2 || !bind.test(src)) throw new Error(`ort patch: patterns not found (unregister x${nUnreg}, bind ${bind.test(src)})`);
      src = src.replace(unreg, "($2[2].gpuBuffer.__plamoKeep||$1.webgpuUnregisterBuffer($2[2].gpuBuffer))");
      src = src.replace(bind, "if($1&&(!$2||globalThis.__plamoRebind)){let{handle:$3,outputPreferredLocations");
      return { contents: src, loader: "js" };
    });
  },
};

const common = { bundle: true, target: "chrome121", sourcemap: "linked", logLevel: "info", legalComments: "none", plugins: [ortPatch] };
const builds = [
  { entryPoints: { background: "src/background.ts" }, format: "esm" },
  { entryPoints: { offscreen: "src/offscreen.ts", "engine-worker": "src/engine/worker.ts", "ocr-worker": "src/ocr/worker.ts" }, format: "esm" },
  { entryPoints: { popup: "src/pages/popup.ts", options: "src/pages/options.ts" }, format: "esm" },
  // content scripts are classic scripts
  { entryPoints: { content: "src/content/content.ts" }, format: "iife" },
];
// dev-only latency benchmark pages (BENCH=1 npm run build); src/dev is local-only and not in the repository
if (process.env.BENCH && existsSync("src/dev")) {
  for (const rt of ["asyncify", "jspi"]) {
    builds.push({ entryPoints: { [`bench-${rt}`]: `src/dev/bench-${rt}.ts` }, format: "esm" });
    writeFileSync(`${out}/bench-${rt}.html`, `<!doctype html><meta charset="utf-8"><title>bench</title><script type="module" src="bench-${rt}.js"></script>`);
  }
  builds.push({ entryPoints: { "bench-variants": "src/dev/bench-variants.ts", "bench-tiny": "src/dev/bench-tiny.ts", "bench-ab": "src/dev/bench-ab.ts", "bench-batch": "src/dev/bench-batch.ts", "bench-engine": "src/dev/bench-engine.ts" }, format: "esm" });
  writeFileSync(`${out}/bench-engine.html`, `<!doctype html><meta charset="utf-8"><script type="module" src="bench-engine.js"></script>`);
  writeFileSync(`${out}/bench-batch.html`, `<!doctype html><meta charset="utf-8"><script type="module" src="bench-batch.js"></script>`);
  writeFileSync(`${out}/bench-ab.html`, `<!doctype html><meta charset="utf-8"><script type="module" src="bench-ab.js"></script>`);
  writeFileSync(`${out}/bench-tiny.html`, `<!doctype html><meta charset="utf-8"><script type="module" src="bench-tiny.js"></script>`);
  writeFileSync(`${out}/bench-variants.html`, `<!doctype html><meta charset="utf-8"><title>bench</title><script type="module" src="bench-variants.js"></script>`);
  for (const f of ["ort-wasm-simd-threaded.jspi.wasm", "ort-wasm-simd-threaded.jspi.mjs"]) cpSync(`node_modules/onnxruntime-web/dist/${f}`, `${out}/ort/${f}`);
}
for (const b of builds) Object.assign(b, { ...common, ...b, outdir: out });

if (watch) {
  for (const b of builds) await (await esbuild.context(b)).watch();
} else {
  await Promise.all(builds.map((b) => esbuild.build(b)));
}
