import { importModelFiles } from "../engine/storage.ts";
import { EngineClient, statusText } from "../shared/client.ts";
import { LANGUAGES, MODELS, getSettings, modelBytes, setSettings, type Backend, type EngineStatus, type ModelId, type Settings } from "../shared/common.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const settings = await getSettings();
const client = new EngineClient();

// ---- model choice
let state: EngineStatus["state"] | undefined;
let installed: ModelId[] = [];
let selected: ModelId = settings.model;
const radios = [...document.querySelectorAll<HTMLInputElement>('input[name="model"]')];

function renderModel(): void {
  const info = MODELS[selected];
  for (const r of radios) r.checked = r.value === selected;
  for (const el of document.querySelectorAll<HTMLElement>(".model")) {
    el.querySelector<HTMLElement>("[data-badge]")!.textContent = installed.includes(el.dataset.id as ModelId) ? "설치됨" : "";
  }
  const has = installed.includes(selected);
  const busy = state === "downloading";
  const dl = $<HTMLButtonElement>("download");
  dl.disabled = busy || has;
  dl.textContent = has ? `${info.label} 설치됨` : `${info.label} 내려받기 (${(modelBytes(info) / 1e9).toFixed(2)} GB)`;
  $("cancelDl").classList.toggle("hidden", !busy);
  $<HTMLButtonElement>("delete").disabled = !has || busy;
  $<HTMLButtonElement>("load").disabled = !has;
  $("source").textContent = url.value.trim() ? `출처: ${url.value.trim()}` : `출처: Hugging Face (${info.repo})`;
  const base = $<HTMLAnchorElement>("baseLink");
  base.href = `https://huggingface.co/${info.base}`;
  base.textContent = info.base;
  const repo = $<HTMLAnchorElement>("repoLink");
  repo.href = `https://huggingface.co/${info.repo}`;
  repo.textContent = info.repo;
}

for (const r of radios) {
  r.addEventListener("change", async () => {
    if (!r.checked) return;
    selected = r.value as ModelId;
    await setSettings({ model: selected });
    await client.send({ type: "select-model", model: selected });
    renderModel();
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
  renderModel();
});
void client.send({ type: "status" });

// ---- download
const url = $<HTMLInputElement>("url");
url.value = settings.modelUrlOverride;
url.addEventListener("change", () => {
  void setSettings({ modelUrlOverride: url.value.trim() });
  renderModel();
});
$("download").addEventListener("click", () => void client.send({ type: "download", model: selected }));
$("cancelDl").addEventListener("click", () => void client.send({ type: "cancel-download" }));

// ---- import from folder
const folder = $<HTMLInputElement>("folder");
$("pick").addEventListener("click", () => folder.click());
folder.addEventListener("change", async () => {
  const files = [...(folder.files ?? [])];
  folder.value = "";
  const msg = $("importMsg");
  try {
    await client.send({ type: "unload" });
    await importModelFiles(MODELS[selected], files, (loaded, total) => {
      msg.textContent = `복사·확인 중… ${((loaded / total) * 100).toFixed(0)}%`;
    });
    msg.textContent = "가져오기 완료";
    await client.send({ type: "select-model", model: selected });
    await client.send({ type: "load" });
  } catch (e) {
    msg.textContent = `실패: ${(e as Error).message}`;
  }
});

// ---- image OCR model (downloaded the first time an image is translated with it)
const ocrModel = $<HTMLSelectElement>("ocrModel");
ocrModel.value = settings.ocrModel;
ocrModel.addEventListener("change", () => void setSettings({ ocrModel: ocrModel.value as Settings["ocrModel"] }));

// ---- backend / lifecycle
const backend = $<HTMLSelectElement>("backend");
backend.value = settings.backend;
backend.addEventListener("change", async () => {
  await setSettings({ backend: backend.value as Backend });
  if (state === "ready") {
    await client.send({ type: "unload" });
    await client.send({ type: "load" });
  }
});
$("load").addEventListener("click", () => void client.send({ type: "load" }));
$("unload").addEventListener("click", () => void client.send({ type: "unload" }));
$("delete").addEventListener("click", () => {
  const info = MODELS[selected];
  if (confirm(`내려받은 ${info.label} 모델(약 ${(modelBytes(info) / 1e9).toFixed(1)}GB)을 삭제할까요?`)) {
    void client.send({ type: "delete-model", model: selected });
  }
});
renderModel();

// ---- test translation
const lang = $<HTMLSelectElement>("lang");
for (const [code, label] of LANGUAGES) lang.add(new Option(`${label} (${code})`, code, false, code === settings.targetLang));
$("go").addEventListener("click", async () => {
  const out = $("out");
  const speed = $("speed");
  out.classList.add("muted");
  out.textContent = "번역 중…";
  speed.textContent = "";
  const t0 = performance.now();
  let first = 0;
  try {
    const text = await client.translate({ text: $<HTMLTextAreaElement>("src").value, targetLang: lang.value, priority: 0 }, (partial) => {
      first ||= performance.now();
      out.classList.remove("muted");
      out.textContent = partial;
    });
    out.classList.remove("muted");
    out.textContent = text;
    const total = (performance.now() - t0) / 1000;
    speed.textContent = `${total.toFixed(2)}s${first ? ` · 첫 글자 ${((first - t0) / 1000).toFixed(2)}s` : ""}${client.status?.tokensPerSec ? ` · ${client.status.tokensPerSec} tok/s` : ""}`;
  } catch (e) {
    out.textContent = `오류: ${(e as Error).message}`;
  }
});
