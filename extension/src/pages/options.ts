import { EngineClient, statusText } from "../shared/client.ts";
import {
  BATCH_CHOICES,
  EXACT_BATCH,
  IDLE_UNLOAD_CHOICES,
  INPAINT_MODEL,
  MODELS,
  OCR_MODELS,
  getSettings,
  modelBytes,
  recommendedBatch,
  setSettings,
  type Backend,
  type EngineStatus,
  type ModelId,
  type Settings,
} from "../shared/common.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const radios = (name: string) => [...document.querySelectorAll<HTMLInputElement>(`input[name="${name}"]`)];
const gb = (bytes: number) => `${(bytes / 1e9).toFixed(2)} GB`;
const settings = await getSettings();
const client = new EngineClient();

let state: EngineStatus["state"] | undefined;
let installed: ModelId[] = [];
let selected: ModelId = settings.model;
let ocrModel: Settings["ocrModel"] = settings.ocrModel;
let backend: Backend = settings.backend;
// the GPU's vendor decides the suggested batch size (null: no WebGPU here)
const gpuAdapter = await (navigator as Navigator & { gpu?: { requestAdapter(): Promise<{ info?: { vendor?: string } } | null> } }).gpu
  ?.requestAdapter()
  .catch(() => null);
const gpuVendor = gpuAdapter ? (gpuAdapter.info?.vendor ?? "") : null;

function render(): void {
  const info = MODELS[selected];
  const ocr = OCR_MODELS[ocrModel] ?? OCR_MODELS.mobile;
  const has = installed.includes(selected);
  const busy = state === "downloading";
  for (const r of radios("model")) r.checked = r.value === selected;
  for (const r of radios("ocrModel")) r.checked = r.value === ocrModel;
  $("modelInfo").textContent = `${gb(modelBytes(info))}${has ? " · 설치됨" : ""}`;
  $("ocrInfo").textContent = `${Math.round(modelBytes(ocr) / 1e6)} MB`;
  // everything the extension needs comes in this one download: the translation model plus the image-translation models
  $("installInfo").textContent = has ? "" : gb(modelBytes(info) + modelBytes(ocr) + modelBytes(INPAINT_MODEL));
  const dl = $<HTMLButtonElement>("download");
  dl.disabled = busy || has;
  dl.textContent = has ? `${info.label} 설치됨` : `${info.label} 내려받기`;
  dl.classList.toggle("primary", !has);
  $("cancelDl").classList.toggle("hidden", !busy);
  $("deleteInfo").textContent = has ? `${info.label} · ${gb(modelBytes(info))}` : "";
  $<HTMLButtonElement>("delete").disabled = !has || busy;
  $<HTMLButtonElement>("load").disabled = !has || state === "ready" || state === "loading";
  $<HTMLButtonElement>("unload").disabled = state !== "ready" && state !== "loading";
  $<HTMLAnchorElement>("baseLink").href = `https://huggingface.co/${info.base}`;
  $<HTMLAnchorElement>("repoLink").href = `https://huggingface.co/${info.repo}`;
  renderBatch();
}

// ---- batch size: the user's choice; the suggestion depends on the GPU and the model
const batch = $<HTMLSelectElement>("batchSize");
for (const n of BATCH_CHOICES) batch.add(new Option(String(n), String(n), false, n === settings.batchSize));
function renderBatch(): void {
  const cpu = backend === "wasm" || gpuVendor === null;
  const rec = recommendedBatch(cpu ? null : gpuVendor, MODELS[selected]);
  for (const o of batch.options) o.textContent = Number(o.value) === rec.size && !cpu ? `${o.value} (권장)` : o.value;
  batch.disabled = cpu;
  $("batchInfo").textContent = cpu
    ? "CPU에서는 한 번에 하나씩 번역합니다"
    : `권장 ${rec.size}개 (${rec.reason}). ${EXACT_BATCH}개를 넘으면 빨라지지만 표현이 조금 달라질 수 있고 GPU 메모리를 더 씁니다`;
}
batch.addEventListener("change", async () => {
  await setSettings({ batchSize: Number(batch.value) });
  // the engine takes it when it loads the model
  if (state === "ready" || state === "loading") {
    await client.send({ type: "unload" });
    await client.send({ type: "load" });
  }
});

// ---- model and OCR choice (both are part of the install: changing either may need another download)
for (const r of radios("model")) {
  r.addEventListener("change", async () => {
    if (!r.checked) return;
    selected = r.value as ModelId;
    await setSettings({ model: selected });
    await client.send({ type: "select-model", model: selected });
    render();
  });
}
for (const r of radios("ocrModel")) {
  r.addEventListener("change", async () => {
    if (!r.checked) return;
    ocrModel = r.value as Settings["ocrModel"];
    await setSettings({ ocrModel });
    await client.send({ type: "select-model", model: selected });
    render();
  });
}

// ---- status
client.onStatus((s) => {
  state = s.state;
  if (s.installed) installed = s.installed;
  $("status").textContent = statusText(s);
  $("dot").className = `dot ${s.state}`;
  const bar = $("bar");
  bar.classList.toggle("hidden", s.progress == null);
  (bar.firstElementChild as HTMLElement).style.width = `${(s.progress ?? 0) * 100}%`;
  render();
});
void client.send({ type: "status" });

// ---- install
$("download").addEventListener("click", () => void client.send({ type: "download", model: selected }));
$("cancelDl").addEventListener("click", () => void client.send({ type: "cancel-download" }));

// ---- backend (reloads a running model)
for (const r of radios("backend")) {
  r.checked = r.value === settings.backend;
  r.addEventListener("change", async () => {
    if (!r.checked) return;
    backend = r.value as Backend;
    await setSettings({ backend });
    renderBatch();
    if (state === "ready") {
      await client.send({ type: "unload" });
      await client.send({ type: "load" });
    }
  });
}

// ---- memory
const idle = $<HTMLSelectElement>("idleUnload");
for (const [min, label] of IDLE_UNLOAD_CHOICES) idle.add(new Option(label, String(min), false, min === settings.idleUnloadMin));
idle.addEventListener("change", async () => {
  await setSettings({ idleUnloadMin: Number(idle.value) });
  await client.send({ type: "idle-changed" });
});
$("load").addEventListener("click", () => void client.send({ type: "load" }));
$("unload").addEventListener("click", () => void client.send({ type: "unload" }));

// ---- storage
$("delete").addEventListener("click", () => {
  const info = MODELS[selected];
  if (confirm(`내려받은 ${info.label} 모델(약 ${(modelBytes(info) / 1e9).toFixed(1)}GB)을 삭제할까요?`)) {
    void client.send({ type: "delete-model", model: selected });
  }
});
render();
