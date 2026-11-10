import { EngineClient, statusText } from "../shared/client.ts";
import { LANGUAGES, getSettings, setSettings, type DisplayMode } from "../shared/common.ts";
import { hydrateIcons, icon } from "../shared/icons.ts";

hydrateIcons();

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const settings = await getSettings();
const lang = $<HTMLSelectElement>("lang");
for (const [code, label] of LANGUAGES) lang.add(new Option(label, code, false, code === settings.targetLang));
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

const openOptions = () => void chrome.runtime.openOptionsPage();
$("options").addEventListener("click", openOptions);
$("noticeAction").addEventListener("click", openOptions);

// ---- page toggle
const pageBtn = $<HTMLButtonElement>("page");
const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
async function refreshPageState(): Promise<void> {
  if (!tab?.id) return;
  try {
    const s = await chrome.tabs.sendMessage(tab.id, { type: "page-state" });
    pageBtn.textContent = s?.active ? "원문 보기" : "이 페이지 번역";
    pageBtn.classList.toggle("primary", !s?.active);
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
  // a missing model or an error replaces the status line with a banner that leads to the settings page
  const problem = s.state === "missing" || s.state === "error";
  const kind = s.state === "missing" ? "warning" : "error";
  $("notice").className = `banner ${kind}${problem ? "" : " hidden"}`;
  $("noticeIcon").innerHTML = icon(kind, 20);
  $("noticeText").textContent = s.state === "missing" ? "모델 설치 필요" : statusText(s);
  $("noticeAction").textContent = s.state === "missing" ? "설치" : "설정";
  $("statusBox").classList.toggle("hidden", problem);
  $("status").textContent = statusText(s);
  $("dot").className = `dot ${s.state}`;
  const bar = $("bar");
  bar.classList.toggle("hidden", s.progress == null);
  (bar.firstElementChild as HTMLElement).style.width = `${(s.progress ?? 0) * 100}%`;
});
void client.send({ type: "status" }).catch((e) => ($("status").textContent = String(e)));
