/** Shared constants, settings and message shapes. */

export const LANGUAGES: [code: string, label: string][] = [
  ["ko", "한국어"],
  ["en", "English"],
  ["ja", "日本語"],
  ["ja-easy", "やさしい日本語"],
  ["zh-hans", "简体中文"],
  ["zh-hant", "繁體中文"],
  ["ar", "العربية"],
  ["az", "Azərbaycan"],
  ["be", "Беларуская"],
  ["bg", "Български"],
  ["bn", "বাংলা"],
  ["ca", "Català"],
  ["ceb", "Cebuano"],
  ["de", "Deutsch"],
  ["es", "Español"],
  ["et", "Eesti"],
  ["eu", "Euskara"],
  ["fa", "فارسی"],
  ["fr", "Français"],
  ["gu", "ગુજરાતી"],
  ["he", "עברית"],
  ["hi", "हिन्दी"],
  ["hr", "Hrvatski"],
  ["hu", "Magyar"],
  ["hy", "Հայերեն"],
  ["id", "Bahasa Indonesia"],
  ["it", "Italiano"],
  ["jv", "Basa Jawa"],
  ["ka", "ქართული"],
  ["kk", "Қазақ"],
  ["km", "ខ្មែរ"],
  ["lt", "Lietuvių"],
  ["lv", "Latviešu"],
  ["ml", "മലയാളം"],
  ["mr", "मराठी"],
  ["ms", "Bahasa Melayu"],
  ["my", "မြန်မာ"],
  ["ne", "नेपाली"],
  ["nl", "Nederlands"],
  ["pa", "ਪੰਜਾਬੀ"],
  ["pl", "Polski"],
  ["ru", "Русский"],
  ["si", "සිංහල"],
  ["sk", "Slovenčina"],
  ["sl", "Slovenščina"],
  ["sr", "Српски"],
  ["sw", "Kiswahili"],
  ["ta", "தமிழ்"],
  ["te", "తెలుగు"],
  ["th", "ไทย"],
  ["tl", "Tagalog"],
  ["ur", "اردو"],
  ["uz", "Oʻzbek"],
  ["vi", "Tiếng Việt"],
];

export type DisplayMode = "dual" | "replace";
export type Backend = "auto" | "webgpu" | "wasm";

export interface Settings {
  /** Language for page and image translation (the popup's choice). */
  targetLang: string;
  /** Language last chosen in the selection panel; empty = follow targetLang. Kept apart so changing it there does
   *  not re-translate the page. */
  selectionLang: string;
  displayMode: DisplayMode;
  /** Show the floating translate button when text is selected. */
  selectionButton: boolean;
  /** Translate immediately on selection instead of showing the button first. */
  selectionAuto: boolean;
  /** Keep translating after the tab navigates: a tab whose page was being translated starts again on the next page. */
  keepOnNavigate: boolean;
  /** Which model size to run. */
  model: ModelId;
  /** OCR recognition model for image translation: mobile (small, the default) or server (more accurate, 5× larger) */
  ocrModel: OcrModelId;
  /** Optional base URL that serves onnx/model_q4f16.onnx(+_data); empty = the pinned Hugging Face repo. */
  modelUrlOverride: string;
  backend: Backend;
  /** Unload the models after this many minutes without work (0 = keep them loaded). */
  idleUnloadMin: number;
  /** Translations decoded together on the GPU (one of BATCH_CHOICES; recommendedBatch suggests one). */
  batchSize: number;
}

/** Choices for Settings.batchSize. */
export const BATCH_CHOICES = [1, 2, 3, 4, 6, 8, 12, 16];
/**
 * The largest batch that gives exactly the same tokens as translating one at a time (measured on 0.8B; wider batches
 * pick other GPU kernels and change a word here and there: 12 of 16 paragraphs identical at 8 and 16).
 */
export const EXACT_BATCH = 3;
/** Capacity of the static KV buffers in tokens (engine/generator.ts; >= its MAX_CONTEXT). */
export const KV_CAPACITY = 2048;
/** GPU memory the batch KV buffers should stay within (all rows together), for the recommendation. */
const BATCH_KV_BUDGET = 512 * 1024 * 1024;

/** KV buffers of one sequence: key and value per layer, [heads, KV_CAPACITY, headDim] fp16. */
export const kvBytesPerRow = (m: ModelInfo) => m.numLayers * 2 * m.numKvHeads * KV_CAPACITY * m.headDim * 2;

/**
 * The batch size to suggest, and why. Without a GPU translations run one at a time. NVIDIA GPUs: 16 (measured on an
 * RTX 3070 with 0.8B: a 44-paragraph page 3× faster than with 3); other GPUs: 4, as 16 was not measured on them. Either
 * way no more rows than fit BATCH_KV_BUDGET of KV buffers (0.8B: 16 MB per row, 2B: 96 MB).
 * `vendor`: the WebGPU adapter's vendor, null without WebGPU.
 */
export function recommendedBatch(vendor: string | null, m: ModelInfo): { size: number; reason: string } {
  if (vendor === null) return { size: 1, reason: "GPU를 쓸 수 없어 한 번에 하나씩 번역합니다" };
  const nvidia = /nvidia/i.test(vendor);
  const fit = Math.floor(BATCH_KV_BUDGET / kvBytesPerRow(m));
  const want = nvidia ? 16 : 4;
  // the largest of the choices that fits
  const size = Math.max(...BATCH_CHOICES.filter((n) => n === 1 || n <= Math.min(want, fit)));
  const gpu = nvidia ? "NVIDIA GPU" : "NVIDIA 외 GPU";
  if (fit < want) return { size, reason: `${gpu} · ${m.label}는 문단당 KV 메모리가 ${Math.round(kvBytesPerRow(m) / 2 ** 20)} MB라 ${size}개까지` };
  return { size, reason: nvidia ? `${gpu} · ${m.label}` : `${gpu}에서는 16개를 측정하지 않았습니다` };
}

/** Choices for Settings.idleUnloadMin. */
export const IDLE_UNLOAD_CHOICES: [minutes: number, label: string][] = [
  [0, "내리지 않음"],
  [1, "1분 후"],
  [5, "5분 후"],
  [10, "10분 후"],
  [30, "30분 후"],
  [60, "1시간 후"],
];

export const DEFAULT_SETTINGS: Settings = {
  targetLang: "ko",
  selectionLang: "",
  displayMode: "replace",
  selectionButton: true,
  selectionAuto: false,
  keepOnNavigate: false,
  // the light model by default (fast and small enough for phones); 2B is the more accurate choice in the settings
  model: "0.8b",
  ocrModel: "mobile",
  modelUrlOverride: "",
  backend: "auto",
  idleUnloadMin: 5,
  // one more than EXACT_BATCH (a word may differ now and then); the settings suggest a width for the GPU (recommendedBatch)
  batchSize: 4,
};

export async function getSettings(): Promise<Settings> {
  const s = await chrome.storage.local.get(DEFAULT_SETTINGS as unknown as Record<string, unknown>);
  return { ...DEFAULT_SETTINGS, ...(s as Partial<Settings>) };
}

export function setSettings(patch: Partial<Settings>): Promise<void> {
  return chrome.storage.local.set(patch);
}

export type ModelId = "0.8b" | "2b";

export interface ModelFile {
  path: string;
  size: number;
  sha256: string;
  /** absolute download URL (default: `path` relative to the model's base URL) */
  url?: string;
  /** file name in OPFS (default: the last segment of `path`) */
  name?: string;
}

/** Anything stored in OPFS as a set of checksummed files (engine/storage.ts). */
export interface StoredModel {
  label: string;
  /** OPFS directory name */
  dir: string;
  files: readonly ModelFile[];
}

export interface ModelInfo {
  id: ModelId;
  label: string;
  /** OPFS directory name */
  dir: string;
  /** Hugging Face repo and the commit the files are pinned to */
  repo: string;
  revision: string;
  base: string;
  numLayers: number;
  numKvHeads: number;
  headDim: number;
  files: [model: ModelFile, data: ModelFile];
}

/**
 * Converted models published on Hugging Face (tools/publish). Downloads are pinned to a commit and
 * checked against these sizes and SHA-256 hashes, so a changed or tampered file is rejected.
 */
export const MODELS: Record<ModelId, ModelInfo> = {
  "0.8b": {
    id: "0.8b",
    label: "0.8B",
    dir: "plamo-3-translate-0.8b-q4f16",
    repo: "chiro0318/plamo-3-translate-0.8b-onnx-webgpu",
    revision: "8bfacd0680cc464ebba538454352d6d554095f4c",
    base: "pfnet/plamo-3-translate-0.8b",
    numLayers: 16,
    numKvHeads: 1,
    headDim: 128,
    files: [
      { path: "onnx/model_q4f16.onnx", size: 82976, sha256: "f8e6339b965b16f718a8a0f5180228ec701598074efe698f523ab3707f88a535" },
      { path: "onnx/model_q4f16.onnx_data", size: 503057408, sha256: "5d81a1d1b8a43152460a4ee4a33e343d3af6ee4ee5218ed0dddedf03ae6e20fd" },
    ],
  },
  "2b": {
    id: "2b",
    label: "2B",
    dir: "plamo-3-translate-2b-q4f16",
    repo: "chiro0318/plamo-3-translate-2b-onnx-webgpu",
    revision: "598dca5f070b5f57cb3c28432941685d3ea3879d",
    base: "pfnet/plamo-3-translate-2b",
    numLayers: 24,
    numKvHeads: 4,
    headDim: 128,
    files: [
      { path: "onnx/model_q4f16.onnx", size: 124079, sha256: "2cfcee9356d5a1985abe2d9146b538b97e15325f84e744f7b28fdc4e5c26b616" },
      { path: "onnx/model_q4f16.onnx_data", size: 1621660672, sha256: "7c7b992c7b8791afe64fd38a5b4e8fe823e6fe0b1eabdb92218aa4c9a4933c22" },
    ],
  },
};

/**
 * OCR for image translation: PP-OCRv5 (PaddlePaddle, Apache-2.0) text detection + recognition, official ONNX exports
 * pinned to a commit. The recognizer reads Chinese, Japanese and English (and Latin script); both recognizers share
 * the character list bundled as public/ocr/ppocrv5_dict.json (the models' inference.yml).
 *  - mobile (default): mobile detector + mobile recognizer, 21 MB
 *  - server: mobile detector + server recognizer, 89 MB (more accurate on small or decorated text)
 */
export type OcrModelId = "mobile" | "server";
const ppocr = (repo: string, rev: string, name: string, size: number, sha256: string): ModelFile => ({
  path: "inference.onnx",
  url: `https://huggingface.co/PaddlePaddle/${repo}/resolve/${rev}/inference.onnx`,
  name,
  size,
  sha256,
});
const PPOCR_DET = ppocr("PP-OCRv5_mobile_det_onnx", "e6f4fa85f00e168c862bc462aebca69eef9b3d3d", "det_mobile.onnx", 4826518, "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d");
export const OCR_MODELS: Record<OcrModelId, StoredModel> = {
  mobile: {
    label: "OCR (PP-OCRv5 mobile)",
    dir: "ppocrv5-mobile",
    files: [PPOCR_DET, ppocr("PP-OCRv5_mobile_rec_onnx", "ed152b8b495f84de93cda5709d768548a9127622", "rec_mobile.onnx", 16534782, "da72dc72ca4dc220df0dfde68c1dedc31c58d3e76a25871122e5056227d50092")],
  },
  server: {
    label: "OCR (PP-OCRv5 server)",
    dir: "ppocrv5",
    files: [PPOCR_DET, ppocr("PP-OCRv5_server_rec_onnx", "b70df217f4fd99d14f970bad092cebe7d74cc4d1", "rec_server.onnx", 84503027, "d9dc333c9c7b042c6dffb8e33d72b6f65c9c1d463d0a3c2f78174fea55e94752")],
  },
};

/**
 * Inpainting for erasing text on photos and busy backgrounds: MI-GAN (Picsart AI Research, ICCV 2023; code and
 * weights MIT), a GAN built for phones — the 512² generator alone (its ONNX pipeline's pre/post-processing is done
 * here), 28 MB, ~4× faster than LaMa on the same crops and much lighter in memory. Pinned to a commit.
 */
export const INPAINT_MODEL: StoredModel = {
  label: "배경 복원 (MI-GAN)",
  dir: "migan-512",
  files: [
    {
      path: "migan-512-generator.onnx",
      url: "https://huggingface.co/FreeHugsForRobots/ps-inpaint-migan/resolve/42eef60d02d0736f0edf7574ebbeba6802d9f533/migan-512-generator.onnx",
      name: "migan.onnx",
      size: 28098112,
      sha256: "40b78086443a1104b6aa47b19f91fd8679857ae1583e3730d72d6298f1a232f6",
    },
  ],
};

export const modelBytes = (m: StoredModel) => m.files.reduce((a, f) => a + f.size, 0);

/** Base URL the model files are fetched from: the user's override, or the pinned Hugging Face commit. */
export function modelSourceUrl(m: ModelInfo, override?: string): string {
  const o = override?.trim();
  if (o) return o.endsWith("/") ? o : o + "/";
  return `https://huggingface.co/${m.repo}/resolve/${m.revision}/`;
}

export type EngineState = "missing" | "downloading" | "idle" | "loading" | "ready" | "error";

export interface EngineStatus {
  state: EngineState;
  /** 0..1 for downloading/loading */
  progress?: number;
  message?: string;
  backend?: string;
  /** selected model and which models are installed */
  model?: ModelId;
  installed?: ModelId[];
  /** decode speed of the last job */
  tokensPerSec?: number;
  queue?: number;
  /** how many translations the loaded engine decodes together (clients keep about that many requests in flight) */
  batch?: number;
}

/** Port name used by content scripts / extension pages to talk to the offscreen engine. */
export const ENGINE_PORT = "plamo-engine";

export interface TranslateRequest {
  type: "translate";
  id: string;
  text: string;
  targetLang: string;
  sourceLang?: string | null;
  context?: string | null;
  /**
   * Sentence style of the translation (FORM_LANGS only): unset, the model picks one per request, and the paragraphs of
   * a page come out in either. A page decides it once (content/page.ts).
   */
  form?: TextForm | null;
  /** lower = sooner */
  priority?: number;
  stream?: boolean;
}

/** Korean -다 / -습니다·-요, Japanese だ・である / です・ます. */
export type TextForm = "plain" | "polite";
/** Target languages with a TextForm. */
export const FORM_LANGS = new Set(["ko", "ja"]);

const FORM_ENDINGS: Record<string, { polite: RegExp; plain: RegExp }> = {
  ko: { polite: /(니다|니까|요)$/u, plain: /[다라자냐]$/u },
  ja: { polite: /(です|ます|でした|ました|ません|ましょう|でしょう|ください)$/u, plain: /(だ|である|た|ない|る|う|い|か)$/u },
};

/** The style most sentences of `text` (a translation into `lang`) end in; null when none of them tells. */
export function textForm(text: string, lang: string): TextForm | null {
  const end = FORM_ENDINGS[lang];
  if (!end) return null;
  let polite = 0;
  let plain = 0;
  for (const s of text.split(/(?<=[.!?。！？])\s*|\n+/u)) {
    // without the closing punctuation, quotes, brackets and footnote marks ([2])
    const t = s.replace(/(\[\d+\]|[\s.!?。！？"'”’」』)\]])+$/u, "");
    if (end.polite.test(t)) polite++;
    else if (end.plain.test(t)) plain++;
  }
  return polite + plain === 0 ? null : polite > plain ? "polite" : "plain";
}

/** Image translation: OCR of one image. Coordinates are in the image's natural pixels. */
export interface OcrRequest {
  type: "ocr";
  id: string;
  /** image URL (fetched by the extension, so cross-origin images work) */
  src?: string;
  /** already-read image (data: URL), e.g. a cropped screenshot when `src` cannot be fetched */
  dataUrl?: string;
  /** the translation target: paragraphs that need no translation are not erased */
  targetLang: string;
  /** which OCR model to use (default: the settings') */
  ocrModel?: OcrModelId;
}

/** The image with a paragraph's original text erased, for `rect` (x, y, w, h in the image's natural pixels). */
export interface ErasePatch {
  /** index into OcrResult.paragraphs */
  i: number;
  rect: [number, number, number, number];
  /** PNG data: URL */
  url: string;
}

export type Rgb = [number, number, number];

export interface OcrLine {
  /** x, y, width, height */
  box: [number, number, number, number];
  text: string;
  score: number;
  /** extent of the glyph ink across the line (y range, or x range for a column) */
  ink?: [number, number];
  /** extent of the glyph ink along the line (x range, or y range for a column) */
  span?: [number, number];
}

export interface OcrParagraph {
  box: [number, number, number, number];
  lines: OcrLine[];
  /** lines joined in reading order */
  text: string;
  /** text runs top-to-bottom (columns right to left) */
  vertical: boolean;
  /** median line height (horizontal) or column width (vertical) of the detector boxes */
  lineSize: number;
  /** estimated font size (em) of the original text, from the ink of its glyphs */
  fontPx: number;
  bg: Rgb;
  fg: Rgb;
  /** the background is a photo or gradient: a solid box would stand out */
  busy: boolean;
  /** a busy background that is a smooth gradient: paint it from these colours (CSS gradient), from → to */
  gradient?: { to: "right" | "bottom"; from: Rgb; end: Rgb };
  /** the text looks bold */
  bold: boolean;
  /** rectangle covered by the glyph ink (x, y, w, h) */
  inkBox: [number, number, number, number];
  /** plain background around the ink rectangle — left, right, up, down (image pixels): room for a longer translation */
  room: [number, number, number, number];
  /** raw measurements, for tuning */
  noise?: number;
  ink?: number;
}

export interface OcrResult {
  width: number;
  height: number;
  paragraphs: OcrParagraph[];
  /** erased backgrounds of the paragraphs on plain or gradient backgrounds */
  patches: ErasePatch[];
  /** paragraphs on photos still being erased by the inpainting model (an "ocr-patches" message follows) */
  pending: number[];
  backend: string;
  ms: number;
  /** worker phases (ms): model load (first image only), detection, recognition, grouping and measuring, erasing */
  timing?: { load: number; detect: number; recognize: number; layout: number; erase: number };
}

export type ClientToEngine =
  | TranslateRequest
  | OcrRequest
  | { type: "cancel"; ids?: string[]; all?: boolean }
  | { type: "status" }
  | { type: "download"; model: ModelId }
  | { type: "cancel-download" }
  | { type: "select-model"; model: ModelId }
  | { type: "load" }
  | { type: "unload" }
  | { type: "idle-changed" }
  | { type: "delete-model"; model: ModelId };

export type EngineToClient =
  | { type: "delta"; id: string; text: string }
  | { type: "done"; id: string; text: string; cached?: boolean }
  | { type: "error"; id: string; message: string }
  | { type: "status"; status: EngineStatus }
  | { type: "ocr-done"; id: string; result: OcrResult }
  /** the remaining erased patches (inpainting model); empty if it could not run */
  | { type: "ocr-patches"; id: string; patches: ErasePatch[] };

/** One-shot messages handled by the background service worker. */
export type BackgroundMessage =
  | { type: "ensure-engine" }
  | { type: "page-toggle"; tabId?: number }
  | { type: "open-options" }
  | { type: "get-settings" }
  /** the sender tab's page translation started or stopped (remembered for Settings.keepOnNavigate) */
  | { type: "page-active"; active: boolean }
  /** whether the sender tab's page was being translated before it navigated (answers a boolean) */
  | { type: "page-was-active" }
  /** screenshot of the sender's visible tab (fallback for images the extension cannot fetch) */
  | { type: "capture-visible" };
