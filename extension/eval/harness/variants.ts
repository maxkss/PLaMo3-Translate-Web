/**
 * The erasers compared by eval/erase.mjs. A variant is a mask plan (default: the extension's own planErase) and a fill
 * for one paragraph's patch.
 */
import type { InferenceSession, Tensor } from "onnxruntime-common";
import { planErase, type ErasePlan } from "../../src/ocr/analyse.ts";
import { inpaintFill, pushPull, type Rect } from "../../src/ocr/erase.ts";
import type { Pixels } from "../../src/ocr/ocr.ts";
import type { OcrParagraph } from "../../src/shared/common.ts";

export interface FillContext {
  ort: { Tensor: typeof Tensor };
  px: Pixels;
  mask: Uint8Array;
  rect: Rect;
  fontPx: number;
  kind: "plain" | "gradient" | "photo";
  /** a photo paragraph masked by its glyphs (else by its whole box) */
  glyphs?: boolean;
  paragraph: OcrParagraph;
  inpaint(file: string, backend: "webgpu" | "wasm"): Promise<InferenceSession>;
}

export interface Variant {
  plan?(px: Pixels, paragraphs: OcrParagraph[], truth: Uint8Array): ErasePlan;
  fill(c: FillContext): Promise<boolean>;
}

const diffuse = async (c: FillContext) => (pushPull(c.px, c.mask, c.rect, Math.max(8, Math.round(1.5 * c.fontPx))), true);
const migan = (backend: "webgpu" | "wasm") => async (c: FillContext) => inpaintFill(c.ort, await c.inpaint("migan.onnx", backend), c.px, c.mask, c.rect);

/** The extension's plan with the mask replaced by the true text pixels grown by `r`: what a perfect mask would give. */
const oraclePlan = (r: number) => (px: Pixels, paragraphs: OcrParagraph[], truth: Uint8Array): ErasePlan => {
  const plan = planErase(px, paragraphs, 1, "ko", px.height, { glyphs: false });
  const W = px.width, H = px.height;
  const mask = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (truth[y * W + x] !== 2) continue;
    for (let yy = Math.max(0, y - r); yy <= Math.min(H - 1, y + r); yy++) mask.fill(1, yy * W + Math.max(0, x - r), yy * W + Math.min(W - 1, x + r) + 1);
  }
  return { ...plan, mask };
};

const boxPlan = (px: Pixels, paragraphs: OcrParagraph[]) => planErase(px, paragraphs, 1, "ko", px.height, { glyphs: false });
const glyphPlan = (px: Pixels, paragraphs: OcrParagraph[]) => planErase(px, paragraphs, 1, "ko", px.height, { glyphs: true });

export const VARIANTS: Record<string, Variant> = {
  /** 0.3.0: the whole box on photos; diffusion on plain and gradient backgrounds, MI-GAN on photos */
  current: { plan: boxPlan, fill: (c) => (c.kind === "photo" ? migan("webgpu")(c) : diffuse(c)) },
  diffusion: { plan: boxPlan, fill: diffuse },
  migan: { plan: boxPlan, fill: migan("webgpu") },
  "migan-wasm": { plan: boxPlan, fill: migan("wasm") },
  /** glyph masks on photos where the text colour separates */
  "glyph-diffusion": { plan: glyphPlan, fill: diffuse },
  "glyph-migan": { plan: glyphPlan, fill: (c) => (c.kind === "photo" ? migan("webgpu")(c) : diffuse(c)) },
  /** diffusion wherever the glyphs were found, MI-GAN only for whole boxes */
  "glyph-hybrid": { plan: glyphPlan, fill: async (c) => (c.kind === "photo" && !c.glyphs ? (await migan("webgpu")(c)) || diffuse(c) : diffuse(c)) },
  "oracle-diffusion": { plan: oraclePlan(2), fill: diffuse },
  "oracle-migan": { plan: oraclePlan(2), fill: migan("webgpu") },
};
