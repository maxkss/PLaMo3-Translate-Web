import { EngineClient, statusText } from "../shared/client.ts";
import { LANGUAGES, getSettings, setSettings, type DisplayMode } from "../shared/common.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const settings = await getSettings();
const lang = $<HTMLSelectElement>("lang");
for (const [code, label] of LANGUAGES) lang.add(new Option(`${label} (${code})`, code, false, code === settings.targetLang));
lang.addEventListener("change", () => void setSettings({ targetLang: lang.value }));

for (const r of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
  r.checked = r.value === settings.displayMode;
  r.addEventListener("change", () => r.checked && void setSettings({ displayMode: r.value as DisplayMode }));
}

const selButton = $<HTMLInputElement>("selButton");
const selAuto = $<HTMLInputElement>("selAuto");
selButton.checked = settings.selectionButton;
selAuto.checked = settings.selectionAuto;
selButton.addEventListener("change", () => void setSettings({ selectionButton: selButton.checked }));
selAuto.addEventListener("change", () => void setSettings({ selectionAuto: selAuto.checked }));

$("options").addEventListener("click", (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});

// ---- page toggle
const pageBtn = $<HTMLButtonElement>("page");
const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
async function refreshPageState(): Promise<void> {
  if (!tab?.id) return;
  try {
    const s = await chrome.tabs.sendMessage(tab.id, { type: "page-state" });
    pageBtn.textContent = s?.active ? "원문 보기" : "이 페이지 번역";
  } catch {
    pageBtn.textContent = "이 페이지 번역";
    if (!/^https?:|^file:/.test(tab.url ?? "")) {
      pageBtn.disabled = true;
      pageBtn.title = "이 페이지에서는 번역할 수 없습니다";
    }
  }
}
pageBtn.addEventListener("click", async () => {
  if (!tab?.id) return;
  pageBtn.disabled = true;
  await chrome.runtime.sendMessage({ type: "page-toggle", tabId: tab.id });
  window.close();
});
void refreshPageState();

// ---- engine status
const client = new EngineClient();
client.onStatus((s) => {
  $("status").textContent = statusText(s);
  $("dot").className = `dot ${s.state}`;
  const bar = $("bar");
  bar.classList.toggle("hidden", s.progress == null);
  (bar.firstElementChild as HTMLElement).style.width = `${(s.progress ?? 0) * 100}%`;
});
void client.send({ type: "status" }).catch((e) => ($("status").textContent = String(e)));
