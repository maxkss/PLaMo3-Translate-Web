/**
 * End-to-end check of tokenizer + generator + translator against the exported ONNX model,
 * using onnxruntime-web (wasm backend) in Node. Run: node test/translate.e2e.ts [path/to/model_q4f16.onnx]
 */
import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { Generator } from "../src/engine/generator.ts";
import { PlamoTokenizer } from "../src/engine/tokenizer.ts";
import { Translator } from "../src/engine/translator.ts";

const modelPath = process.argv[2] ?? new URL("../../models/onnx/plamo-3-translate-2b/onnx/model_q4f16.onnx", import.meta.url).pathname;
const tok = new PlamoTokenizer(JSON.parse(readFileSync(new URL("../public/tokenizer/plamo3.json", import.meta.url), "utf-8")));
ort.env.wasm.numThreads = 8;
const session = await ort.InferenceSession.create(readFileSync(modelPath), {
  graphOptimizationLevel: "all",
  externalData: [{ path: "model_q4f16.onnx_data", data: readFileSync(modelPath + "_data") }],
});
const tr = new Translator(tok, new Generator(session as any, ort as any));

const cases: [string, string][] = [
  ["The meeting has been moved to next Monday because the client requested more time.", "ko"],
  ["会議は来週の月曜日に変更されました。資料は前日までに共有してください。", "ko"],
  ["안녕하세요, 오늘 날씨가 정말 좋네요.", "en"],
  ["  Click the button below to continue.  ", "ko"],
  ["2024-01-01 — 12:30", "ko"],
];
for (const [text, tgt] of cases) {
  let deltas = 0;
  const t = performance.now();
  const { text: out, stats } = await tr.translate({ text, targetLang: tgt }, () => deltas++);
  const ms = performance.now() - t;
  console.log(`[${tgt}] ${JSON.stringify(out)}  (${stats.outputTokens} tok, ${deltas} deltas, ${(stats.outputTokens / (stats.decodeMs / 1000 || 1)).toFixed(1)} tok/s, ${ms.toFixed(0)} ms)`);
}
const long = Array.from({ length: 12 }, (_, i) => `Paragraph ${i + 1}: Machine translation running locally in the browser keeps your data private and works offline once the model has been downloaded.`).join("\n\n");
console.log("split pieces:", tr.split(long).length);
const { text: longOut } = await tr.translate({ text: long, targetLang: "ko" });
console.log(longOut.split("\n\n").slice(0, 3).join("\n"), "\n...", longOut.split("\n\n").length, "paragraphs");
