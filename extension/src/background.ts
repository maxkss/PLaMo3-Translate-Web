/**
 * Service worker: owns the offscreen engine document, context menus and keyboard commands.
 * Translation traffic goes directly from content scripts / pages to the offscreen document
 * over runtime ports (ENGINE_PORT); this worker only makes sure that document exists.
 */

import { getSettings, type BackgroundMessage } from "./shared/common.ts";

const OFFSCREEN_URL = "offscreen.html";
let creating: Promise<void> | null = null;

async function ensureEngine(): Promise<void> {
  const url = chrome.runtime.getURL(OFFSCREEN_URL);
  const contexts = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
    documentUrls: [url],
  });
  if (contexts.length) return;
  if (!creating) {
    creating = chrome.offscreen
      .createDocument({
        url: OFFSCREEN_URL,
        reasons: [chrome.offscreen.Reason.WORKERS],
        justification: "Run the on-device PLaMo 3 Translate model with WebGPU/WebAssembly.",
      })
      .finally(() => {
        creating = null;
      });
  }
  await creating;
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({ id: "translate-selection", title: "PLaMo로 선택 영역 번역", contexts: ["selection"] });
    chrome.contextMenus.create({ id: "translate-page", title: "PLaMo로 이 페이지 번역 / 원문 보기", contexts: ["page"] });
    chrome.contextMenus.create({ id: "translate-image", title: "PLaMo로 이미지 번역 / 원문 보기", contexts: ["image"] });
  });
});

async function sendToTab(tabId: number, msg: unknown): Promise<void> {
  await ensureEngine();
  try {
    await chrome.tabs.sendMessage(tabId, msg);
  } catch {
    // content script not present (e.g. tab opened before install): inject and retry
    await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
    await chrome.tabs.sendMessage(tabId, msg);
  }
}

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab?.id) return;
  if (info.menuItemId === "translate-selection") {
    void sendToTab(tab.id, { type: "translate-selection", text: info.selectionText ?? "" });
  } else if (info.menuItemId === "translate-image") {
    void sendToTab(tab.id, { type: "translate-image", srcUrl: info.srcUrl ?? "" });
  } else if (info.menuItemId === "translate-page") {
    void sendToTab(tab.id, { type: "page-toggle" });
  }
});

chrome.commands.onCommand.addListener(async (command, tab) => {
  const tabId = tab?.id ?? (await chrome.tabs.query({ active: true, currentWindow: true }))[0]?.id;
  if (!tabId) return;
  if (command === "translate-page") void sendToTab(tabId, { type: "page-toggle" });
  if (command === "translate-selection") void sendToTab(tabId, { type: "translate-selection" });
});

// ---- tabs whose page is being translated (for Settings.keepOnNavigate)
// Kept in session storage so the set survives this worker being suspended; it is cleared with the browser session.

const ACTIVE_TABS = "pageActiveTabs";
let tabsLock: Promise<unknown> = Promise.resolve();

async function activeTabs(): Promise<number[]> {
  const got = await chrome.storage.session.get(ACTIVE_TABS);
  return (got[ACTIVE_TABS] as number[] | undefined) ?? [];
}

/** Adds or removes a tab; updates run one after another so concurrent changes are not lost. */
function setTabActive(tabId: number, active: boolean): Promise<unknown> {
  tabsLock = tabsLock.then(async () => {
    const tabs = await activeTabs();
    const has = tabs.includes(tabId);
    if (has === active) return;
    await chrome.storage.session.set({ [ACTIVE_TABS]: active ? [...tabs, tabId] : tabs.filter((t) => t !== tabId) });
  }).catch(() => {});
  return tabsLock;
}

chrome.tabs.onRemoved.addListener((tabId) => void setTabActive(tabId, false));
// a link opened in a new tab from a translated page carries the translation along. Not tabs.onCreated: Chrome also
// sets openerTabId on tabs opened from the browser's UI (Ctrl+T, the + button, bookmarks), which are unrelated pages.
chrome.webNavigation.onCreatedNavigationTarget.addListener(({ sourceTabId, tabId }) => {
  tabsLock = tabsLock.then(async () => {
    const tabs = await activeTabs();
    if (tabs.includes(sourceTabId) && !tabs.includes(tabId)) await chrome.storage.session.set({ [ACTIVE_TABS]: [...tabs, tabId] });
  }).catch(() => {});
});

chrome.runtime.onMessage.addListener((msg: BackgroundMessage, sender, sendResponse) => {
  switch (msg.type) {
    case "ensure-engine":
      ensureEngine().then(
        () => sendResponse({ ok: true }),
        (e) => sendResponse({ ok: false, error: String(e) }),
      );
      return true;
    case "page-toggle": {
      const tabId = msg.tabId ?? sender.tab?.id;
      if (tabId) sendToTab(tabId, { type: "page-toggle" }).then(() => sendResponse({ ok: true }), (e) => sendResponse({ ok: false, error: String(e) }));
      return true;
    }
    case "page-active":
      if (sender.tab?.id != null) void setTabActive(sender.tab.id, msg.active);
      break;
    case "page-was-active": {
      const tabId = sender.tab?.id;
      if (tabId == null || sender.frameId) return false;
      tabsLock.then(activeTabs).then((tabs) => sendResponse(tabs.includes(tabId)), () => sendResponse(false));
      return true;
    }
    case "get-settings":
      getSettings().then(sendResponse);
      return true;
    case "capture-visible": {
      const windowId = sender.tab?.windowId;
      if (windowId == null) return false;
      chrome.tabs.captureVisibleTab(windowId, { format: "png" }).then(
        (dataUrl) => sendResponse({ dataUrl }),
        (e) => sendResponse({ error: String(e) }),
      );
      return true;
    }
    case "open-options":
      chrome.runtime.openOptionsPage();
      break;
  }
  return false;
});
