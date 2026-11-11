/**
 * Whole-page translation.
 *
 * The page is cut into units: a unit is a run of inline content (text and inline elements) inside one block element,
 * so the model sees whole paragraphs. Inline markup is sent along as numbered tags the model carries through
 * translation (`<a1>link</a1>`, `<b2>bold</b2>`, `<x3/>` for things like citation marks and images), and the
 * translation is put back into the *same* DOM nodes: text goes into the existing text nodes, links and formatting
 * elements are reused (reordered when the target language orders them differently). Nothing of the page's own is
 * detached, so frameworks that hold on to their nodes keep working, and restoring is exact.
 *
 * Hidden inline content (display:none, screen-reader-only text) is not sent to the model and left alone, so it can
 * neither swallow the translation nor make hidden text show up. Each unit is shown only once its translation is
 * complete (no token-by-token streaming). Every unit on the page is translated, on-screen first, then by distance from
 * the viewport; dynamic content is picked up through a MutationObserver.
 */

import type { DisplayMode } from "../shared/common.ts";
import type { EngineClient } from "../shared/client.ts";

/** subtrees never looked into */
const SKIP_TAGS = new Set([
  "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "TEXTAREA", "INPUT", "SELECT", "OPTION", "PRE", "SVG", "MATH", "CANVAS",
  "VIDEO", "AUDIO", "IFRAME", "OBJECT", "PLAMO-TR", "PLAMO-TRANSLATE-UI",
]);
/** inline content kept verbatim (shown to the model for context, never rewritten) */
const VERBATIM_TAGS = new Set(["CODE", "KBD", "SAMP", "VAR", "TT"]);
/** inline elements without translatable text of their own: kept as a placeholder */
const ATOM_TAGS = new Set([
  "IMG", "BR", "WBR", "SVG", "MATH", "INPUT", "SELECT", "TEXTAREA", "CANVAS", "VIDEO", "AUDIO", "IFRAME", "OBJECT",
  "PICTURE", "METER", "PROGRESS", "OUTPUT",
]);
/** never part of the visible text */
const INVISIBLE_TAGS = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "LINK", "META", "PLAMO-TR", "PLAMO-TRANSLATE-UI", "RP"]);
const INLINE_TAGS = new Set([
  "A", "ABBR", "B", "BDI", "BDO", "BR", "CITE", "CODE", "DATA", "DFN", "EM", "FONT", "I", "IMG", "INS", "DEL", "KBD",
  "LABEL", "MARK", "Q", "RUBY", "RT", "RP", "S", "SAMP", "SMALL", "SPAN", "STRIKE", "STRONG", "SUB", "SUP", "TIME",
  "TT", "U", "VAR", "WBR", "BIG",
]);
/** tag name shown to the model per element (it knows what <a>, <b>, <i>… mean) */
const TAG_NAME: Record<string, string> = {
  A: "a", BUTTON: "a", B: "b", STRONG: "b", I: "i", EM: "i", CITE: "i", DFN: "i", U: "u", INS: "u", S: "s",
  DEL: "s", STRIKE: "s", SUP: "sup", SUB: "sub", MARK: "mark", SMALL: "small", Q: "q", ABBR: "abbr", CODE: "code",
  KBD: "code", SAMP: "code", VAR: "code", TT: "code",
};
/** computed properties that make an inline element look different from its parent */
const LOOK = [
  "fontWeight", "fontStyle", "fontFamily", "fontSize", "color", "backgroundColor", "textDecorationLine",
  "verticalAlign", "textTransform", "letterSpacing", "borderTopWidth", "borderBottomWidth", "paddingLeft",
  "paddingRight", "marginLeft", "marginRight", "fontVariantCaps",
] as const;

/** units requested at once (at least; more when the engine decodes more together, see EngineStatus.batch) */
const MAX_INFLIGHT = 4;
/**
 * Units are translated in groups: one request, one line per unit (a unit's text has no line breaks). The model keeps
 * the line structure (314/314 items on real pages, test/group-probe.mjs), each line is put back into its own unit, and
 * the units give each other context: a footer's "Terms" is 약관, not 용어, and the items of a list share their terms.
 *
 * A group is a part of the DOM tree, so what belongs together on the page is translated together (packGroups):
 * - the largest subtree that fits the budget is one group: a list, a table row, a card, a heading with its paragraph
 *   (wrappers with a single child do not matter, any element counts);
 * - small neighbouring subtrees are packed into one group, each of them whole;
 * - a container larger than the budget is cut at the boundaries of its children, from its start;
 * - a group never crosses a landmark or widget (HARD), nor a unit that is too long to share a request.
 * Groups depend on the page only (not on what is on screen or already translated): the reading order decides which
 * group is sent next, and a group is sent with all its pending units.
 *
 * The budget is in source tokens, estimated (unitCost): one group is meant to be one model call (CHUNK_TOKENS in
 * engine/translator.ts, which packs the lines of a request that is longer after all into as few calls as fit).
 */
const GROUP_TOKENS = 340;
/**
 * The budget when the engine decodes many requests together (EngineStatus.batch >= 8): a group's translation takes as
 * long as its longest request, so smaller groups fill the batch and finish sooner (a 44-paragraph article: 22.9 s →
 * 12.6 s, first paragraph 9 s → 4.8 s). A group still holds a few neighbouring paragraphs; alone (80 tokens), they
 * lose each other's context (the register changed between paragraphs).
 */
const WIDE_GROUP_TOKENS = 170;
const GROUP_SIZE = 16;
/** test hook: <html data-plamo-group-tokens="N"> overrides the budget (for comparisons) */
const groupTokens = (engineBatch: number) => {
  const v = Number(document.documentElement.dataset.plamoGroupTokens);
  return Number.isFinite(v) && v > 0 ? v : engineBatch >= 8 ? WIDE_GROUP_TOKENS : GROUP_TOKENS;
};
/** landmarks and widgets: a group never crosses their boundary, however small they are */
const HARD = [
  "header", "nav", "main", "aside", "footer", "form", "dialog",
  ...["banner", "navigation", "main", "complementary", "contentinfo", "search", "form", "dialog", "alertdialog",
    "menu", "menubar", "tablist", "toolbar", "tree", "listbox"].map((r) => `[role=${r}]`),
].join(", ");

/**
 * Source tokens of a unit, estimated from its characters (the tokenizer is not in the page). Measured with the PLaMo
 * tokenizer: English text ~0.15–0.2 tokens a letter, digits ~1 and punctuation ~0.5, kana ~0.5, kanji ~0.9, Hangul
 * more than 1 (bytes), a numbered tag ~4. An estimate that is too low costs a second model call for the group, one
 * that is too high a smaller group.
 */
export function unitCost(source: string): number {
  let n = 1; // its line break
  const text = source.replace(/<\/?[a-z]+\d+\s*\/?>/gi, () => ((n += 4), ""));
  for (const ch of text) {
    const c = ch.codePointAt(0)!;
    if (c < 0x80) n += c >= 0x30 && c <= 0x39 ? 1 : c === 0x20 || (c | 0x20) >= 0x61 && (c | 0x20) <= 0x7a ? 0.18 : 0.5;
    else if (c >= 0x3040 && c <= 0x30ff) n += 0.5; // kana
    else if (c >= 0xac00 && c <= 0xd7af) n += 1.3; // Hangul
    else if (c >= 0x2e80 && c <= 0x9fff) n += 0.9; // CJK ideographs and punctuation
    else n += 0.5;
  }
  return n;
}

/** A node of the tree that groups are cut from: a unit (`item`) or an element with the nodes inside it. */
export interface PackNode<T> {
  item?: T;
  cost?: number;
  /** nothing inside is grouped with anything outside */
  hard?: boolean;
  children?: PackNode<T>[];
}

/**
 * Cut the tree into groups, in document order: a subtree that fits (`budget` in cost, `max` items) stays whole and is
 * packed with the fitting subtrees next to it; one that does not fit is cut at its children; an item above the budget
 * is a group of its own. Nothing is packed across a subtree that was cut, or across a `hard` node.
 */
export function packGroups<T>(root: PackNode<T>, budget: number, max: number): T[][] {
  // the items of `n` for its parent to pack with its neighbours, or the groups `n` was cut into
  const walk = (n: PackNode<T>): { items: T[]; cost: number; groups?: undefined } | { groups: T[][] } => {
    if (n.item !== undefined) return { items: [n.item], cost: n.cost ?? 0 };
    const groups: T[][] = [];
    let buf: T[] = [], cost = 0, cut = false;
    const flush = () => {
      if (buf.length) groups.push(buf);
      buf = [];
      cost = 0;
      cut = true;
    };
    for (const c of n.children ?? []) {
      const sub = walk(c);
      if (sub.groups) {
        flush();
        groups.push(...sub.groups);
        continue;
      }
      if (buf.length + sub.items.length > max || cost + sub.cost > budget) flush();
      buf.push(...sub.items);
      cost += sub.cost;
      if (cost > budget) flush(); // a single item above the budget
    }
    if (!cut && !n.hard) return { items: buf, cost };
    flush();
    return { groups };
  };
  const all = walk(root);
  return all.groups ?? (all.items.length ? [all.items] : []);
}
/** verbatim content longer than this is sent as a placeholder only */
const VERBATIM_MAX = 80;
/** test hook: <html data-plamo-debug> */
const DEBUG = () => document.documentElement.hasAttribute("data-plamo-debug");
const HAS_LETTER = /\p{L}/u;
const MAIN_SELECTOR = "main, article, [role=main], #content, #main, #main-content, .post, .entry-content, .article";

// ------------------------------------------------------------------ unit model

type Item =
  | { k: "text"; node: Text }
  /** an inline element that looks like plain text (a bare span): its text joins the stream, the element is emptied */
  | { k: "flat"; el: Element; texts: Text[] }
  /** an inline element sent as a numbered tag; `level` is its content (null when verbatim) */
  | { k: "slot"; el: Element; id: number; tag: string; level: Level | null }
  /** sent as <xN/>, moved as a whole */
  | { k: "atom"; node: Element; id: number }
  /** not sent (hidden content, comments); stays in the run */
  | { k: "keep"; node: Node };

interface Level {
  parent: Node;
  items: Item[];
}

interface Unit {
  /** block element (or link, for inline units) the unit belongs to: used for ranking and dual-mode placement */
  outer: HTMLElement;
  /** inline wrappers unwrapped between `outer` and the level's parent, outermost first */
  chain: HTMLElement[];
  level: Level;
  slots: Map<number, Item & { k: "slot" | "atom" }>;
  /** all text nodes of the unit (to recognise them in mutations and rescans) */
  texts: Text[];
  /** source with numbered tags, and its plain text */
  source: string;
  text: string;
  /** a single link/button inside a menu-like run: dual mode renders inline */
  inline: boolean;
  /** only hidden text (icon-button labels): translated in place in every display mode, never shown */
  hidden: boolean;
  /** its original is fading out; the translation is put in when that ends */
  revealing?: boolean;
  main: boolean;
  state: "pending" | "running" | "done" | "error" | "skipped";
  order: number;
  translation?: string;
  reqId?: string;
  /** source tokens, estimated (unitCost) */
  cost: number;
  /** the units it is translated with (packGroups), itself included; set by regroup() */
  group?: Unit[];
  /** its group's output did not split back into lines: the half of that group it is sent with once more */
  retry?: Unit[];
  /** the retry did not split back into lines either: translate it alone */
  solo?: boolean;
  /** the group request it is part of, while running */
  batch?: Batch;
  /** replace mode: text nodes we rewrote (original and what we wrote) and nodes we created */
  writes: Map<Text, { orig: string; wrote: string }>;
  created: Node[];
  /** dual mode: the inserted translation */
  trEl?: HTMLElement;
}

/** A request translating several units at once. */
interface Batch {
  id: string;
  /** units still waiting for it (a unit dropped meanwhile leaves) */
  members: Set<Unit>;
}

/** Shift the numbered tags of a source (`<a1>`, `</a1>`, `<x2/>`) by `by`. */
export const shiftTags = (s: string, by: number) => s.replace(/<(\/?)([a-z]+)(\d+)(\s*\/?)>/gi, (_m, c, t, n, e) => `<${c}${t}${Number(n) + by}${e}>`);

/** A group's output line back to one unit's tag numbers: tags of other units (outside 1..count after the shift) go. */
export function unshiftTags(s: string, by: number, count: number): string {
  return s.replace(/<(\/?)([a-z]+)(\d+)(\s*\/?)>/gi, (m, c, t, n, e) => {
    const id = Number(n) - by;
    return id >= 1 && id <= count ? `<${c}${t}${id}${e}>` : "";
  });
}

/** Parsed translation: strings and tagged spans (by slot id). */
type PNode = string | { id: number; children: PNode[] };

const escapeText = (s: string) => s.replace(/<(?=[\p{L}\d/!?])/gu, "&lt;");
const unescapeText = (s: string) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">");

/** Where a tag the model opened but never closed most likely ended: a clause boundary. */
const CLAUSE_END = /[.,;:!?。、，．！？（(\[「『"“”]/u;

/**
 * Parse model output with numbered tags. Lenient, because small models slip: a tag without its number is matched by
 * name, `</a3` without `>` still closes, unknown or stray tags are dropped, a closing tag closes everything opened
 * inside it, and a tag never closed ends at the next clause boundary or tag (not longer than about its source text).
 * Whitespace just inside a tag is moved outside it.
 */
export function parseTagged(out: string, tags: Map<number, { tag: string; atom: boolean; len?: number }>): PNode[] {
  type Open = { id: number; children: PNode[] };
  const root: Open = { id: 0, children: [] };
  const stack: Open[] = [root];
  const opened = new Set<number>();
  /** elements closed only implicitly, with the element they are in */
  const unclosed: { node: Open; parent: Open }[] = [];
  const push = (s: string) => {
    if (!s) return;
    const c = stack[stack.length - 1].children;
    if (typeof c[c.length - 1] === "string") c[c.length - 1] += unescapeText(s);
    else c.push(unescapeText(s));
  };
  const re = /<(\/?)([a-z]+)(\d*)\s*(\/?)(?:>|(?<=\d)(?=[^\w>]|$))/gi;
  let last = 0;
  for (const m of out.matchAll(re)) {
    const [full, close, rawTag, num, self] = m;
    const tag = rawTag.toLowerCase();
    let id = num ? Number(num) : 0;
    if (!id) {
      // number dropped: the innermost open tag (closing) or the first unused slot (opening) with this name
      if (close) id = [...stack].reverse().find((o) => o.id && tags.get(o.id)?.tag === tag)?.id ?? 0;
      else for (const [i, t] of tags) if (t.tag === tag && !opened.has(i)) { id = i; break; }
    }
    const t = tags.get(id);
    push(out.slice(last, m.index));
    last = m.index! + full.length;
    if (!t) continue;
    if (t.atom) {
      if (!close) stack[stack.length - 1].children.push({ id, children: [] });
      continue;
    }
    if (close) {
      const at = stack.findLastIndex((o) => o.id === id);
      if (at > 0) {
        for (let i = stack.length - 1; i > at; i--) unclosed.push({ node: stack[i], parent: stack[i - 1] });
        stack.length = at;
      }
      continue;
    }
    opened.add(id);
    const node: Open = { id, children: [] };
    stack[stack.length - 1].children.push(node);
    if (!self) stack.push(node);
  }
  push(out.slice(last));
  for (let i = stack.length - 1; i > 0; i--) unclosed.push({ node: stack[i], parent: stack[i - 1] });
  // inner ones first, so an outer element sees its final content
  for (const { node, parent } of unclosed) {
    const max = Math.max(8, Math.round((tags.get(node.id)?.len ?? 20) * 1.5) + 4);
    const inside: PNode[] = [];
    const after: PNode[] = [];
    let len = 0, done = false;
    for (const c of node.children) {
      if (done) after.push(c);
      else if (typeof c === "string") {
        let cut = -1, started = len > 0;
        for (let i = 0; i < c.length; i++) {
          const ch = c[i];
          if (!started) {
            if (/[\p{L}\p{N}]/u.test(ch)) started = true;
            continue;
          }
          if (CLAUSE_END.test(ch) || (len + i >= max && /\s/.test(ch))) {
            cut = i;
            break;
          }
        }
        if (cut >= 0) {
          inside.push(c.slice(0, cut));
          after.push(c.slice(cut));
          done = true;
        } else {
          inside.push(c);
          len += c.length;
        }
      } else if (len > 0 && plain(inside).trim()) {
        after.push(c);
        done = true;
      } else inside.push(c);
    }
    node.children = inside;
    const at = parent.children.indexOf(node);
    parent.children.splice(at + 1, 0, ...after);
  }
  // " <a1> word</a1>" → " <a1>word</a1>": underline and link without the space
  const tidy = (ns: PNode[]): PNode[] => {
    const res: PNode[] = [];
    const add = (s: string) => {
      if (!s) return;
      if (typeof res[res.length - 1] === "string") res[res.length - 1] += s;
      else res.push(s);
    };
    for (const n of ns) {
      if (typeof n === "string") {
        add(n);
        continue;
      }
      n.children = tidy(n.children);
      let pre = "", post = "";
      const first = n.children[0];
      if (typeof first === "string") {
        pre = /^\s*/.exec(first)![0];
        n.children[0] = first.slice(pre.length);
      }
      const end = n.children.length - 1, lastC = n.children[end];
      if (typeof lastC === "string") {
        post = /\s*$/.exec(lastC)![0];
        n.children[end] = lastC.slice(0, lastC.length - post.length);
      }
      if (pre && !/\s$/.test(String(res[res.length - 1] ?? " "))) add(" ");
      res.push(n);
      if (post) add(" ");
    }
    return res;
  };
  return tidy(root.children);
}

const plain = (ns: PNode[]): string => ns.map((n) => (typeof n === "string" ? n : plain(n.children))).join("");

/** True when the text is (mostly) already in the target language, judged by script. */
export function alreadyInTarget(text: string, target: string, pageLang: string): boolean {
  let letters = 0, hangul = 0, kana = 0, han = 0, latin = 0, cyr = 0;
  for (const ch of text) {
    if (!/\p{L}/u.test(ch)) continue;
    letters++;
    if (/\p{Script=Hangul}/u.test(ch)) hangul++;
    else if (/[\p{Script=Hiragana}\p{Script=Katakana}]/u.test(ch)) kana++;
    else if (/\p{Script=Han}/u.test(ch)) han++;
    else if (/\p{Script=Latin}/u.test(ch)) latin++;
    else if (/\p{Script=Cyrillic}/u.test(ch)) cyr++;
  }
  if (!letters) return true;
  const r = (n: number) => n / letters;
  if (target === "ko") return r(hangul) > 0.5;
  if (target === "ja" || target === "ja-easy") return kana > 0 && r(kana + han) > 0.6;
  if (target.startsWith("zh")) return kana === 0 && r(han) > 0.6;
  const page = pageLang.toLowerCase().split("-")[0];
  if (page && page === target) return r(latin + cyr) > 0.6;
  return false;
}

function isSkippedTree(el: Element | null): boolean {
  for (let e = el; e; e = e.parentElement) {
    if (SKIP_TAGS.has(e.tagName) || noTranslate(e)) return true;
  }
  return false;
}

/**
 * Form fields and editable regions: never part of a unit, so they are not counted, moved, copied (dual mode) or
 * re-translated while the user types in them.
 */
function isEditField(el: Element): boolean {
  const t = el.tagName;
  if (t === "TEXTAREA" || t === "SELECT") return true;
  if (t === "INPUT") return (el as HTMLInputElement).type !== "hidden";
  return !!(el as HTMLElement).isContentEditable;
}

const EDIT_FIELDS = 'input:not([type="hidden"]), textarea, select, [contenteditable]:not([contenteditable="false"])';

/** True for a node inside a text field or editable region (typing there changes the DOM). */
function inEditField(n: Node): boolean {
  const el = n.nodeType === Node.ELEMENT_NODE ? (n as Element) : n.parentElement;
  return !!el && (!!(el as HTMLElement).isContentEditable || !!el.closest("textarea, select"));
}

const noTranslate = (e: Element) =>
  e.getAttribute("translate") === "no" || e.classList.contains("notranslate") || (e as HTMLElement).isContentEditable;

export interface PageProgress {
  done: number;
  total: number;
  running: number;
}

export class PageTranslator {
  active = false;
  private readonly client: EngineClient;
  private units: Unit[] = [];
  private byText = new WeakMap<Text, Unit>();
  /** element whose children a unit's run (or a tagged element's content) consists of → that unit */
  private byParent = new WeakMap<Node, Unit>();
  /** units were added or dropped since the groups were made */
  private groupsStale = true;
  /** token budget the current groups were made with */
  private groupBudget = 0;
  /** EngineStatus.batch last seen (0: not known yet) */
  private engineBatch = 0;
  /** the engine's status was asked for before the first request (or that failed) */
  private statusAsked = false;
  /** nodes we created (text nodes in replace mode); never scanned */
  private ours = new WeakSet<Node>();
  private blockCache = new WeakMap<Element, boolean>();
  private static readonly CONTROL = "button, a, label, summary, select, [role=button], [role=tab], [role=menuitem], [role=link], [role=option]";
  /** set while collecting a run that has only hidden text (skip links, icon-button labels): translate it as is */
  private showHidden = false;
  private blockInside = new WeakMap<Element, boolean>();
  private mo: MutationObserver | null = null;
  private readonly onScroll = () => {
    // re-rank what to send next after the reader moves
    if (this.scrollTimer) return;
    this.scrollTimer = window.setTimeout(() => {
      this.scrollTimer = 0;
      this.pump();
    }, 150);
  };
  private scrollTimer = 0;
  private inflight = 0;
  private order = 0;
  private dirtyRoots = new Set<Node>();
  private rescanTimer = 0;
  /** finished units waiting for the next frame (applied together) */
  private toPaint = new Set<Unit>();
  private paintQueued = false;
  /** running transition animations (cancelled on stop) */
  private anims = new Set<Animation>();
  private target = "ko";
  private mode: DisplayMode = "replace";
  onProgress: (p: PageProgress) => void = () => {};
  onError: (msg: string) => void = () => {};

  constructor(client: EngineClient) {
    this.client = client;
    // the engine's batch decides how many requests to keep in flight and how large groups are: re-plan when it changes
    // (it becomes known once the model is loaded)
    client.onStatus((s) => {
      if ((s.batch ?? 0) === this.engineBatch) return;
      this.engineBatch = s.batch ?? 0;
      if (this.active) this.pump();
    });
  }

  start(target: string, mode: DisplayMode): void {
    if (this.active) this.stop();
    this.active = true;
    this.target = target;
    this.mode = mode;
    window.addEventListener("scroll", this.onScroll, { passive: true, capture: true });
    window.addEventListener("resize", this.onScroll, { passive: true });
    this.mo = new MutationObserver((records) => this.onMutations(records));
    this.mo.observe(document.body, { childList: true, subtree: true, characterData: true });
    this.scan(document.body);
    this.report();
    this.pump();
  }

  stop(): void {
    this.active = false;
    window.removeEventListener("scroll", this.onScroll, { capture: true });
    window.removeEventListener("resize", this.onScroll);
    clearTimeout(this.rescanTimer);
    clearTimeout(this.scrollTimer);
    this.scrollTimer = 0;
    this.client.cancel(this.units.filter((u) => u.state === "running" && u.reqId).map((u) => u.reqId!));
    for (const a of this.anims) a.cancel();
    this.anims.clear();
    this.mutate(() => {
      for (const u of this.units) this.restore(u);
    });
    this.mo?.disconnect();
    this.mo = null;
    this.units = [];
    this.byText = new WeakMap();
    this.byParent = new WeakMap();
    this.toPaint.clear();
    this.dirtyRoots.clear();
    this.inflight = 0;
  }

  /** Re-render finished units after a display-mode change. */
  setMode(mode: DisplayMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.mutate(() => {
      for (const u of this.units) {
        if (u.state !== "done" || u.revealing) continue; // a unit mid-transition renders in the new mode when it lands
        this.restore(u);
        this.render(u);
      }
    });
  }

  progress(): PageProgress {
    let done = 0, running = 0, total = 0;
    for (const u of this.units) {
      if (u.state === "skipped") continue;
      total++;
      if (u.state === "done" || u.state === "error") done++;
      if (u.state === "running") running++;
    }
    return { done, total, running };
  }

  private report(): void {
    this.onProgress(this.progress());
  }

  // ------------------------------------------------------------ layout helpers

  private isBlock(el: Element): boolean {
    let v = this.blockCache.get(el);
    if (v === undefined) {
      if (INLINE_TAGS.has(el.tagName)) v = false;
      else {
        // display:none hides the real display: a hidden element with text that is not inline by nature (a collapsed
        // menu item) counts as a block; an empty one (a <link> or popup shell inside a paragraph) does not
        const d = getComputedStyle(el).display;
        v = d === "none"
          ? !INVISIBLE_TAGS.has(el.tagName) && HAS_LETTER.test(el.textContent ?? "")
          : !(d.startsWith("inline") || d === "contents" || d === "ruby");
      }
      this.blockCache.set(el, v);
    }
    return v;
  }

  /** an inline element with block-level descendants (e.g. a card link wrapping divs) is not a single run */
  private hasBlockInside(el: Element): boolean {
    let v = this.blockInside.get(el);
    if (v === undefined) {
      v = false;
      for (const d of el.querySelectorAll("*")) {
        // a hidden popup (display:none div) inside an inline widget does not break the line
        if (!SKIP_TAGS.has(d.tagName) && this.isBlock(d) && getComputedStyle(d).display !== "none") {
          v = true;
          break;
        }
      }
      this.blockInside.set(el, v);
    }
    return v;
  }

  private blockOf(node: Node): HTMLElement | null {
    for (let el = node.parentElement; el; el = el.parentElement) {
      if (el === document.body || this.isBlock(el)) return el as HTMLElement;
    }
    return null;
  }

  /** Hidden relative to its parent: display:none, visibility:hidden, or screen-reader-only clipping. */
  private isHidden(el: Element): boolean {
    if (this.showHidden) return false;
    const cs = getComputedStyle(el);
    if (cs.display === "none") return true;
    if (cs.visibility === "hidden" || cs.visibility === "collapse") {
      const p = el.parentElement;
      return !p || getComputedStyle(p).visibility === "visible";
    }
    if (cs.position === "absolute" || cs.position === "fixed") {
      if ((cs.clip && cs.clip !== "auto") || (cs.clipPath && cs.clipPath !== "none")) return true;
      if (parseFloat(cs.width) <= 1 && parseFloat(cs.height) <= 1 && cs.overflow !== "visible") return true;
    }
    return false;
  }

  /** Looks exactly like its parent's text (a bare span), so its text can simply join the parent's. */
  private looksPlain(el: Element): boolean {
    if (!(el.tagName === "SPAN" || el.tagName === "FONT" || el.tagName === "BDI" || el.tagName === "DATA" || el.tagName === "TIME")) return false;
    if (el.hasAttribute("title") || el.hasAttribute("onclick") || el.hasAttribute("role")) return false;
    const p = el.parentElement;
    if (!p) return false;
    const a = getComputedStyle(el), b = getComputedStyle(p);
    if (a.display !== "inline") return false;
    for (const k of LOOK) {
      if (k === "backgroundColor" ? a[k] !== "rgba(0, 0, 0, 0)" : a[k] !== b[k]) return false;
    }
    return true;
  }

  // ------------------------------------------------------------ collection

  private scan(root: Node): void {
    if (!this.active) return;
    // layout may have changed since the last scan (widgets initialised, menus opened)
    this.blockCache = new WeakMap();
    this.blockInside = new WeakMap();
    if (root.nodeType === Node.TEXT_NODE) root = root.parentNode ?? root;
    if (root.nodeType !== Node.ELEMENT_NODE || isSkippedTree(root as Element) || this.isOurs(root)) return;
    const blocks = new Set<HTMLElement>();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode: (n) => {
        if (n.nodeType === Node.ELEMENT_NODE) {
          const el = n as Element;
          // a shadow host shows its shadow tree, not its light text (<relative-time>): not ours to translate
          if (SKIP_TAGS.has(el.tagName) || noTranslate(el) || el.shadowRoot) return NodeFilter.FILTER_REJECT;
          return NodeFilter.FILTER_SKIP;
        }
        const t = n as Text;
        return t.data.trim() && !this.byText.has(t) && !this.ours.has(t) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
      },
    });
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      const b = this.blockOf(n);
      if (b) blocks.add(b);
    }
    for (const b of blocks) {
      const runs: { parent: HTMLElement; nodes: Node[] }[] = [];
      this.runsOf(b, runs);
      for (const r of runs) this.addRun(b, r.parent, r.nodes);
    }
    this.report();
  }

  /** Split a container's children into maximal runs of inline content (blocks and block-wrapping inlines cut them). */
  private runsOf(container: HTMLElement, out: { parent: HTMLElement; nodes: Node[] }[]): void {
    let run: Node[] = [];
    const flush = () => {
      if (run.length) out.push({ parent: container, nodes: run });
      run = [];
    };
    for (const c of container.childNodes) {
      if (c.nodeType === Node.ELEMENT_NODE) {
        const el = c as HTMLElement;
        if (el.tagName === "PLAMO-TR" || el.tagName === "PLAMO-TRANSLATE-UI") continue;
        // a field cuts the run and stays out of it: the text on either side is translated around it
        if (isEditField(el)) {
          flush();
          continue;
        }
        if (!SKIP_TAGS.has(el.tagName) && !INVISIBLE_TAGS.has(el.tagName)) {
          if (this.isBlock(el)) {
            flush();
            continue;
          }
          // a field inside an inline element (<label>Search <input></label>): the element's text is split around it
          if (!noTranslate(el) && el.querySelector(EDIT_FIELDS)) {
            flush();
            this.runsOf(el, out);
            continue;
          }
          if (!noTranslate(el) && this.hasBlockInside(el)) {
            flush();
            this.runsOf(el, out);
            continue;
          }
        }
      }
      run.push(c);
    }
    flush();
  }

  /** Significant nodes of a run: text with content, and elements that are not hidden. */
  private significant(nodes: Node[]): Node[] {
    return nodes.filter((n) =>
      n.nodeType === Node.TEXT_NODE
        ? !!(n as Text).data.trim()
        : n.nodeType === Node.ELEMENT_NODE && !INVISIBLE_TAGS.has((n as Element).tagName) && !this.isHidden(n as Element),
    );
  }

  private addRun(block: HTMLElement, parent: HTMLElement, nodes: Node[]): void {
    // a page change inside a run we already translated: start that run over
    for (const n of nodes) {
      for (const t of textsIn(n)) {
        const old = this.byText.get(t);
        if (old) this.drop(old);
      }
    }
    const letters = (ns: Node[]) => ns.some((n) => HAS_LETTER.test(n.textContent ?? ""));
    this.showHidden = !letters(this.significant(nodes)) && letters(nodes.filter((n) => !(n instanceof Element && INVISIBLE_TAGS.has(n.tagName))));
    try {
      this.addRunInner(block, parent, nodes);
    } finally {
      this.showHidden = false;
    }
  }

  /**
   * What of a run is sent: its significant nodes without placeholders at either end (a radio button before its
   * label, a citation mark after a sentence) — those stay where they are.
   */
  private core(nodes: Node[]): Node[] {
    const sig = this.significant(nodes);
    const edge = (n: Node) => {
      if (n.nodeType !== Node.ELEMENT_NODE) return false;
      const k = this.classify(n as Element);
      return k === "atom" || k === "keep";
    };
    let a = 0, b = sig.length;
    while (a < b && edge(sig[a])) a++;
    while (b > a && edge(sig[b - 1])) b--;
    return sig.slice(a, b);
  }

  /** Descend while a run is a single inline element (<p><span>…</span></p>, <h2><span>…</span></h2>, <li><a>…</a></li>). */
  private unwrap(parent: HTMLElement, nodes: Node[], chain: HTMLElement[]): { parent: HTMLElement; nodes: Node[]; chain: HTMLElement[] } {
    for (;;) {
      const core = this.core(nodes);
      if (core.length !== 1 || core[0].nodeType !== Node.ELEMENT_NODE) break;
      const el = core[0] as HTMLElement;
      const kind = this.classify(el);
      if (kind !== "slot" && kind !== "flat") break;
      chain = [...chain, el];
      parent = el;
      nodes = [...el.childNodes].filter((n) => !this.isOurs(n));
    }
    return { parent, nodes, chain };
  }

  private addRunInner(block: HTMLElement, parent: HTMLElement, nodes: Node[]): void {
    let chain: HTMLElement[];
    ({ parent, nodes, chain } = this.unwrap(parent, nodes, []));
    // menu-like runs (only links/buttons with separators between them): each link is its own unit
    const sig = this.significant(nodes);
    if (sig.length >= 2) {
      const links = sig.filter((n) => n.nodeType === Node.ELEMENT_NODE && /^(A|BUTTON)$/.test((n as Element).tagName));
      const others = sig.filter((n) => !links.includes(n));
      if (links.length >= 2 && others.every((n) => !HAS_LETTER.test(n.textContent ?? ""))) {
        for (const l of links) this.makeUnit(l as HTMLElement, [], l as HTMLElement, [...l.childNodes].filter((n) => !this.isOurs(n)), true);
        return;
      }
    }
    this.makeUnit(block, chain, parent, nodes, false);
  }

  private makeUnit(outer: HTMLElement, chain: HTMLElement[], parent: HTMLElement, nodes: Node[], inline: boolean): void {
    // unwrap inside a menu link too (<a><span>Home</span></a>)
    if (inline) ({ parent, nodes, chain } = this.unwrap(parent, nodes, chain));
    const u: Unit = {
      outer,
      chain,
      level: { parent, items: [] },
      slots: new Map(),
      texts: [],
      source: "",
      text: "",
      inline,
      hidden: this.showHidden,
      main: !!outer.closest(MAIN_SELECTOR),
      state: "pending",
      order: this.order++,
      cost: 0,
      writes: new Map(),
      created: [],
    };
    let n = 0;
    // placeholders at the ends of the run are not sent and stay where they are
    const core = new Set(this.core(nodes));
    const sig = this.significant(nodes);
    const first = sig.findIndex((x) => core.has(x)), last = sig.findLastIndex((x) => core.has(x));
    const edges = new Set(sig.filter((_, i) => first < 0 || i < first || i > last));
    u.level = this.buildLevel(parent, nodes, u, () => ++n, edges);
    u.source = this.serialize(u.level, true).replace(/\s+/g, " ").trim();
    u.text = this.serialize(u.level, false).replace(/\s+/g, " ").trim();
    if (!HAS_LETTER.test(u.text) && !this.showHidden && nodes.some((x) => HAS_LETTER.test(x.textContent ?? ""))) {
      // only hidden text (an icon button's label): translate that
      this.showHidden = true;
      try {
        this.makeUnit(outer, chain, parent, nodes, inline);
      } finally {
        this.showHidden = false;
      }
      return;
    }
    for (const t of u.texts) if (t.data.trim()) this.byText.set(t, u);
    for (const p of levelParents(u.level)) this.byParent.set(p, u);
    if (!u.texts.some((t) => t.data.trim())) return; // nothing of ours to track (e.g. only an image)
    if (u.text.length < 2 || !HAS_LETTER.test(u.text) || alreadyInTarget(u.text, this.target, document.documentElement.lang)) u.state = "skipped";
    u.cost = unitCost(u.source);
    this.units.push(u);
    this.groupsStale = true;
    this.mark(u);
  }

  /** test hook: unit state on the element whose content it translates (with <html data-plamo-debug>) */
  private mark(u: Unit): void {
    if (DEBUG() && u.level.parent.nodeType === Node.ELEMENT_NODE) {
      const el = u.level.parent as HTMLElement;
      el.dataset.plamoState = u.state;
      if (u.hidden) el.dataset.plamoHidden = "";
      if (u.batch) el.dataset.plamoBatch = u.batch.id;
      else if (u.state === "running") delete el.dataset.plamoBatch; // sent alone (after its group failed)
      if (u.retry) el.dataset.plamoRetry = "";
      if (u.solo) el.dataset.plamoSolo = "";
      el.dataset.plamoSrc = u.source.slice(0, 600);
      if (u.translation) el.dataset.plamoTr = u.translation.slice(0, 600);
    }
  }

  private classify(el: Element): "keep" | "atom" | "verbatim" | "flat" | "slot" {
    const tag = el.tagName;
    if (INVISIBLE_TAGS.has(tag)) return "keep";
    if (this.isHidden(el)) return "keep";
    if (ATOM_TAGS.has(tag) || el.shadowRoot) return "atom";
    const text = el.textContent ?? "";
    if (VERBATIM_TAGS.has(tag) || noTranslate(el) || SKIP_TAGS.has(tag)) return text.trim() && text.length <= VERBATIM_MAX ? "verbatim" : "atom";
    if (!HAS_LETTER.test(text)) {
      // punctuation, numbers, icons: plain when it looks like text, otherwise a placeholder ([12] citation marks)
      return this.looksPlain(el) && this.allPlain(el) ? "flat" : text.trim() || el.children.length ? "atom" : "keep";
    }
    // footnote marks such as [a] or [note 3]
    if ((tag === "SUP" || tag === "SUB") && /^\s*\[[^\]]{1,12}\]\s*$/.test(text)) return "atom";
    // an element with an id is likely updated by the page's scripts (a vote count): keep its text in it
    if (!el.id && this.looksPlain(el) && this.allPlain(el)) return "flat";
    return "slot";
  }

  /** all descendants are text or plain-looking spans */
  private allPlain(el: Element): boolean {
    for (const c of el.children) if (!(this.looksPlain(c) && this.allPlain(c))) return false;
    return true;
  }

  private buildLevel(parent: Node, nodes: Node[], u: Unit, nextId: () => number, edges?: Set<Node>): Level {
    const items: Item[] = [];
    for (const n of nodes) {
      if (edges?.has(n)) {
        items.push({ k: "keep", node: n });
        continue;
      }
      if (n.nodeType === Node.TEXT_NODE) {
        items.push({ k: "text", node: n as Text });
        u.texts.push(n as Text);
        continue;
      }
      if (n.nodeType !== Node.ELEMENT_NODE) {
        items.push({ k: "keep", node: n });
        continue;
      }
      const el = n as HTMLElement;
      switch (this.classify(el)) {
        case "keep":
          items.push({ k: "keep", node: el });
          break;
        case "atom": {
          const it = { k: "atom" as const, node: el, id: nextId() };
          items.push(it);
          u.slots.set(it.id, it);
          break;
        }
        case "verbatim": {
          const it = { k: "slot" as const, el, id: nextId(), tag: TAG_NAME[el.tagName] ?? "code", level: null };
          items.push(it);
          u.slots.set(it.id, it);
          break;
        }
        case "flat": {
          const texts = textsIn(el);
          items.push({ k: "flat", el, texts });
          u.texts.push(...texts);
          break;
        }
        case "slot": {
          const id = nextId();
          const it = { k: "slot" as const, el, id, tag: TAG_NAME[el.tagName] ?? "span", level: null as Level | null };
          u.slots.set(id, it);
          it.level = this.buildLevel(el, [...el.childNodes].filter((c) => !this.isOurs(c)), u, nextId);
          items.push(it);
          break;
        }
      }
    }
    return { parent, items };
  }

  /** The level as the model sees it (`tags`) or as plain text. */
  private serialize(level: Level, tags: boolean): string {
    let s = "";
    let prevEl: Element | null = null;
    for (const it of level.items) {
      const el = it.k === "slot" || it.k === "flat" ? it.el : it.k === "atom" ? it.node : null;
      // separately boxed neighbours (inline-block chips, label + value) read as separate words
      if (el && prevEl && !/\s$/.test(s) && (getComputedStyle(el).display !== "inline" || getComputedStyle(prevEl).display !== "inline")) s += " ";
      if (it.k !== "keep") prevEl = el;
      switch (it.k) {
        case "text":
          s += tags ? escapeText(it.node.data) : it.node.data;
          break;
        case "flat":
          s += it.texts.map((t) => (tags ? escapeText(t.data) : t.data)).join("");
          break;
        case "atom":
          if (tags) s += `<x${it.id}/>`;
          break;
        case "slot": {
          const inner = it.level ? this.serialize(it.level, tags) : tags ? escapeText(it.el.textContent ?? "") : it.el.textContent ?? "";
          s += tags ? `<${it.tag}${it.id}>${inner.replace(/\s+/g, " ")}</${it.tag}${it.id}>` : inner;
          break;
        }
      }
    }
    return s;
  }

  /** Forget a unit (restoring what we changed), e.g. because the page changed it. */
  private drop(u: Unit): void {
    if (u.reqId && u.state === "running") {
      // a group request goes on for its other units; it is cancelled with its last one
      const b = u.batch;
      b?.members.delete(u);
      if (!b || !b.members.size) {
        this.client.cancel([u.reqId]);
        this.inflight--;
      }
    }
    this.mutate(() => this.restore(u));
    for (const t of u.texts) if (this.byText.get(t) === u) this.byText.delete(t);
    for (const p of levelParents(u.level)) if (this.byParent.get(p) === u) this.byParent.delete(p);
    this.toPaint.delete(u);
    const i = this.units.indexOf(u);
    if (i >= 0) this.units.splice(i, 1);
    this.groupsStale = true;
  }

  // ------------------------------------------------------------ scheduling

  /**
   * Reading priority over all pending units: on screen first (top-to-bottom, central column before side
   * columns, main content first), then off-screen units by distance from the viewport (below before
   * above), then units that are not rendered at all (hidden menus, collapsed sections).
   */
  /** Requests kept in flight: enough to fill the engine's batch (it decodes that many together). */
  private maxInflight(): number {
    return Math.max(MAX_INFLIGHT, this.engineBatch);
  }

  private pump(): void {
    if (!this.active) return;
    // know the engine's batch before the first request, so the first groups are sized for it
    if (!this.client.status && !this.statusAsked) {
      this.statusAsked = true;
      void this.client
        .ready()
        .catch(() => {}) // the requests report the failure
        .finally(() => this.pump());
      return;
    }
    const max = this.maxInflight();
    if (this.inflight >= max) return;
    // the engine's batch became known (or changed): group the pending units for it
    if (groupTokens(this.engineBatch) !== this.groupBudget) this.groupsStale = true;
    const vw = document.documentElement.clientWidth;
    const vh = window.innerHeight;
    const ranked = this.units
      .filter((u) => u.state === "pending")
      .map((u) => {
        const r = (u.chain[0] ?? u.outer).getBoundingClientRect();
        const rendered = r.width > 0 || r.height > 0;
        const onScreen = rendered && r.bottom > 0 && r.top < vh;
        const side = Math.abs((r.left + r.right) / 2 - vw / 2) * 2;
        const score = !rendered
          ? 1e9 + u.order
          : onScreen
            ? Math.max(0, r.top) + side - (u.main ? 300 : 0)
            : 100_000 + (r.top >= vh ? r.top - vh : 1.5 * -r.bottom) + side;
        return { u, score };
      })
      .sort((x, y) => x.score - y.score);
    for (const { u } of ranked) {
      if (this.inflight >= max) break;
      if (u.state !== "pending") continue; // taken by a group already
      const group = this.groupFor(u);
      if (group.length > 1) this.runGroup(group);
      else this.run(u);
    }
  }

  /**
   * Make the groups (packGroups) from all units that are translated, whatever their state, so that a group does not
   * depend on what happens to be pending: the tree of the elements around the units, in document order.
   */
  private regroup(): void {
    this.groupsStale = false;
    this.groupBudget = groupTokens(this.engineBatch);
    const debug = DEBUG();
    // hidden labels (icon buttons) are grouped among themselves: they are not part of the visible text around them
    for (const hidden of [false, true]) {
      const anchor = (u: Unit): Node => u.texts.find((t) => t.isConnected) ?? u.outer;
      const units = this.units
        .filter((u) => u.state !== "skipped" && u.hidden === hidden)
        .map((u) => ({ u, at: anchor(u) }))
        .sort((a, b) => (a.at === b.at ? 0 : a.at.compareDocumentPosition(b.at) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
      const root: PackNode<Unit> = { hard: true, children: [] };
      const nodes = new Map<Element, PackNode<Unit>>([[document.body, root]]);
      const nodeOf = (el: Element | null): PackNode<Unit> => {
        if (!el) return root;
        let n = nodes.get(el);
        if (!n) {
          nodes.set(el, (n = { hard: el.matches(HARD), children: [] }));
          nodeOf(el.parentElement).children!.push(n);
        }
        return n;
      };
      for (const { u } of units) nodeOf(u.outer).children!.push({ item: u, cost: u.cost });
      for (const group of packGroups(root, this.groupBudget, GROUP_SIZE)) {
        for (const u of group) {
          u.group = group;
          if (debug && u.level.parent.nodeType === Node.ELEMENT_NODE) (u.level.parent as HTMLElement).dataset.plamoGroup = String(group[0].order);
        }
      }
    }
  }

  /**
   * The pending units to translate with `u`: those of its group — after a group's lines did not come back one per
   * unit, those of the half it is tried again with, and after that, itself.
   */
  private groupFor(u: Unit): Unit[] {
    // <html data-plamo-nogroup>: one request per unit (for comparisons)
    if (u.solo || document.documentElement.hasAttribute("data-plamo-nogroup")) return [u];
    if (this.groupsStale) this.regroup();
    const live = new Set(this.units);
    return (u.retry ?? u.group ?? [u]).filter((v) => v === u || (v.state === "pending" && !v.solo && !!v.retry === !!u.retry && live.has(v)));
  }

  /** Translate several units in one request, one line each; when the lines do not come back one per unit, each is
   * queued again on its own. */
  private runGroup(units: Unit[]): void {
    const id = this.client.newId();
    const batch: Batch = { id, members: new Set(units) };
    // each unit's numbered tags shifted so they are unique within the request
    const offsets: number[] = [];
    let next = 0;
    const lines = units.map((u) => {
      offsets.push(next);
      const line = shiftTags(u.source, next);
      next += u.slots.size;
      return line;
    });
    for (const u of units) {
      u.state = "running";
      u.reqId = id;
      u.batch = batch;
      this.mark(u);
    }
    this.inflight++;
    const live = (u: Unit) => this.active && this.units.includes(u) && batch.members.has(u);
    this.client
      .translate({ id, text: lines.join("\n"), targetLang: this.target, context: document.title || null, priority: 5 })
      .then(
        (tr) => {
          const out = tr.split(/\n+/).map((l) => l.trim()).filter(Boolean);
          const rest = units.filter(live);
          for (const u of rest) u.batch = undefined;
          if (out.length !== units.length) {
            // the model merged or split lines: no reliable way to tell which part is whose. Once more in two halves
            // (the lines that went wrong are in one of them), then each unit alone.
            const again = rest.length >= 4 && !rest.some((u) => u.retry);
            const half = Math.ceil(rest.length / 2);
            const halves = [rest.slice(0, half), rest.slice(half)];
            rest.forEach((u, i) => {
              u.state = "pending";
              if (again) u.retry = halves[i < half ? 0 : 1];
              else u.solo = true;
              if (DEBUG() && u.level.parent.nodeType === Node.ELEMENT_NODE) {
                const el = u.level.parent as HTMLElement;
                el.dataset.plamoGroupIn = lines.join("\n").slice(0, 3000);
                el.dataset.plamoGroupOut = tr.slice(0, 3000);
              }
              this.mark(u);
            });
            return;
          }
          units.forEach((u, i) => {
            if (!rest.includes(u)) return;
            u.state = "done";
            u.translation = unshiftTags(out[i], offsets[i], u.slots.size);
            this.mark(u);
            this.toPaint.add(u);
          });
          this.schedulePaint();
        },
        (e: Error) => {
          for (const u of units) {
            if (!live(u)) continue;
            u.batch = undefined;
            if (e?.name === "AbortError") continue;
            u.state = "error";
          }
          if (e?.name !== "AbortError" && units.some((u) => this.units.includes(u))) this.onError(e.message);
        },
      )
      .finally(() => {
        if (batch.members.size || !this.active) this.inflight = Math.max(0, this.inflight - 1);
        this.report();
        this.pump();
      });
  }

  private run(u: Unit): void {
    u.state = "running";
    this.mark(u);
    this.inflight++;
    const id = (u.reqId = this.client.newId());
    const live = () => this.active && this.units.includes(u);
    // no streaming: a unit appears once its translation is complete
    this.client
      .translate({ id, text: u.source, targetLang: this.target, context: document.title || null, priority: 5 })
      .then(
        (tr) => {
          if (!live()) return;
          u.state = "done";
          u.translation = tr;
          this.mark(u);
          this.toPaint.add(u);
          this.schedulePaint();
        },
        (e: Error) => {
          if (e?.name === "AbortError" || !live()) return;
          u.state = "error";
          this.onError(e.message);
        },
      )
      .finally(() => {
        if (live() || !this.active) this.inflight = Math.max(0, this.inflight - 1);
        this.report();
        this.pump();
      });
  }

  /** Finished units are applied together on the next frame. */
  private schedulePaint(): void {
    if (this.paintQueued) return;
    this.paintQueued = true;
    requestAnimationFrame(() => {
      this.paintQueued = false;
      const batch = [...this.toPaint].filter((u) => this.active && this.units.includes(u) && u.state === "done");
      this.toPaint.clear();
      const motion = !matchMedia("(prefers-reduced-motion: reduce)").matches;
      const now: Unit[] = [];
      const fadeOut: { u: Unit; el: Element; top: number }[] = [];
      for (const u of batch) {
        const box = motion && this.mode === "replace" && !u.hidden && this.changes(u) ? this.onScreen(u.level.parent) : null;
        if (box) fadeOut.push({ u, el: box.el, top: box.top });
        else now.push(u);
      }
      const dualIn: { el: Element; top: number }[] = [];
      this.mutate(() => {
        for (const u of now) {
          if (this.render(u) && motion && u.trEl) {
            const box = this.onScreen(u.trEl);
            if (box) dualIn.push(box);
          }
          this.markDone(u);
        }
      });
      // what finished in the same frame changes top to bottom, a beat apart
      const order = [...fadeOut, ...dualIn].sort((a, b) => a.top - b.top);
      const delayOf = (el: Element) => Math.min(order.findIndex((o) => o.el === el) * 35, 210);
      for (const { el } of dualIn) this.growIn(el as HTMLElement, delayOf(el));
      for (const { u, el } of fadeOut) this.crossfade(u, el, delayOf(el));
    });
  }

  private markDone(u: Unit): void {
    if (DEBUG()) u.outer.dataset.plamoDone = "1"; // lets test/page-bench.mjs tell finished units
  }

  /** The element (if it is on screen) with its top, for animating. */
  private onScreen(n: Node): { el: Element; top: number } | null {
    if (n.nodeType !== Node.ELEMENT_NODE) return null;
    const el = n as Element;
    if (el === document.body || el === document.documentElement || !el.isConnected) return null;
    const r = el.getBoundingClientRect();
    if (r.bottom < 0 || r.top > window.innerHeight || (!r.width && !r.height)) return null;
    return { el, top: r.top };
  }

  /**
   * Replace mode: the original text dims and softens, the translation is swapped in while it is faint, then it comes
   * back into focus — so the change reads as one smooth transition rather than a flash. Web Animations only: nothing
   * is written into the page's style attributes.
   */
  private crossfade(u: Unit, el: Element, delay: number): void {
    const SOFT = { opacity: 0.35, filter: "blur(1.5px)" };
    u.revealing = true;
    // one-sided keyframes: from / back to the element's own opacity and filter, whatever the page set them to
    const out = el.animate([SOFT], {
      duration: 190,
      delay,
      // slows down into the faint point and starts slowly out of it: a valley, not a "V" that reads as a blink
      easing: "cubic-bezier(.45,0,.55,1)",
      fill: "forwards",
    });
    this.anims.add(out);
    const swap = () => {
      this.anims.delete(out);
      u.revealing = false;
      if (!this.active || !this.units.includes(u) || u.state !== "done") {
        out.cancel();
        return;
      }
      this.mutate(() => this.render(u));
      this.markDone(u);
      // start the way back before dropping the held frame, so no frame shows the text at full strength in between
      const back = el.animate([{ ...SOFT, offset: 0 }], { duration: 340, easing: "cubic-bezier(.4,0,.2,1)" });
      this.anims.add(back);
      back.finished.then(() => this.anims.delete(back), () => this.anims.delete(back));
      out.cancel();
    };
    out.finished.then(swap, swap);
  }

  /** Dual mode: the new translation line opens up (no jump of the content below) and fades in out of a soft blur. */
  private growIn(el: HTMLElement, delay: number): void {
    const inline = getComputedStyle(el).display.startsWith("inline");
    const frames: Keyframe[] = inline
      ? [{ opacity: 0, filter: "blur(2px)", offset: 0 }]
      : [
          { opacity: 0, filter: "blur(2px)", height: "0px", marginTop: "0px", overflow: "hidden" },
          { height: `${el.getBoundingClientRect().height}px`, marginTop: getComputedStyle(el).marginTop, overflow: "hidden" },
        ];
    const a = el.animate(frames, { duration: inline ? 320 : 360, delay, easing: "cubic-bezier(.2,.7,.3,1)", fill: "backwards" });
    this.anims.add(a);
    a.finished.then(() => this.anims.delete(a), () => this.anims.delete(a));
  }

  // ------------------------------------------------------------ rendering

  /** Run DOM changes of ours without seeing them as page changes. */
  private mutate(fn: () => void): void {
    if (this.mo) {
      const pending = this.mo.takeRecords();
      if (pending.length) this.onMutations(pending);
    }
    try {
      fn();
    } finally {
      this.mo?.takeRecords();
    }
  }

  /** The parsed translation, or null when it would change nothing (proper nouns, numbers…). */
  private parsed(u: Unit): PNode[] | null {
    const tr = u.translation ?? "";
    const tags = new Map<number, { tag: string; atom: boolean; len: number }>();
    for (const [id, s] of u.slots) {
      tags.set(id, { tag: s.k === "atom" ? "x" : s.tag, atom: s.k === "atom", len: s.k === "slot" ? (s.el.textContent ?? "").trim().length : 0 });
    }
    const tree = parseTagged(tr, tags);
    const norm = (t: string) => t.replace(/\s+/g, " ").trim().toLowerCase();
    return norm(plain(tree)) === norm(u.text) || !plain(tree).trim() ? null : tree;
  }

  private changes(u: Unit): boolean {
    return !!this.parsed(u);
  }

  /** Show a unit's translation; false when there was nothing to change. */
  private render(u: Unit): boolean {
    const tree = this.parsed(u);
    if (!tree) return false;
    if (this.mode === "replace" || u.hidden) this.renderInPlace(u, u.level, tree, new Set());
    else this.renderDual(u, tree);
    return true;
  }

  private write(u: Unit, t: Text, s: string): void {
    const w = u.writes.get(t);
    if (w) w.wrote = s;
    else u.writes.set(t, { orig: t.data, wrote: s });
    t.data = s;
  }

  /** Put the translation of one level into its existing nodes; returns nothing, rearranges `level.parent`'s run. */
  private renderInPlace(u: Unit, level: Level, children: PNode[], used: Set<number>): void {
    const pool = level.items.filter((i): i is Item & { k: "text" } => i.k === "text").map((i) => i.node);
    let next = 0;
    const out: Node[] = [];
    // the run's outer whitespace separates it from what surrounds it
    const lead = level === u.level && /^\s/.test(pool[0]?.data ?? "");
    const trail = level === u.level && /\s$/.test(pool[pool.length - 1]?.data ?? "");
    const putText = (s: string) => {
      if (!s) return;
      if (out.length === headLen && lead && !/^\s/.test(s)) s = " " + s;
      const prev = out[out.length - 1];
      if (prev?.nodeType === Node.TEXT_NODE && u.writes.has(prev as Text)) {
        this.write(u, prev as Text, (prev as Text).data + s);
        return;
      }
      let t: Text;
      if (next < pool.length) t = pool[next++];
      else {
        t = document.createTextNode("");
        this.ours.add(t);
        u.created.push(t);
      }
      this.write(u, t, s);
      out.push(t);
    };
    const mine = new Set(level.items);
    // content the translation does not cover keeps its place at the start/end of the run (e.g. a radio button)
    let h = 0;
    while (h < level.items.length && level.items[h].k === "keep") h++;
    const head = level.items.slice(0, h);
    let t = level.items.length;
    while (t > h && level.items[t - 1].k === "keep") t--;
    const tail = level.items.slice(t);
    for (const it of head) out.push((it as { node: Node }).node);
    const headLen = out.length;
    for (const c of children) {
      if (typeof c === "string") {
        putText(c);
        continue;
      }
      const s = u.slots.get(c.id);
      // a tag used twice, or moved out of the element it belongs in: keep its text only
      if (!s || used.has(c.id) || !mine.has(s)) {
        putText(plain(c.children));
        continue;
      }
      used.add(c.id);
      if (s.k === "atom") out.push(s.node);
      else {
        if (s.level) this.renderInPlace(u, s.level, c.children, used);
        out.push(s.el);
      }
    }
    // whatever the translation did not use: emptied (text) or kept at the end (placeholders, hidden content)
    for (; next < pool.length; next++) {
      this.write(u, pool[next], "");
      out.push(pool[next]);
    }
    for (const it of level.items) {
      if (it.k === "flat") {
        for (const t of it.texts) this.write(u, t, "");
        out.push(it.el);
      } else if (it.k === "slot" && !used.has(it.id)) {
        if (it.level) this.blank(u, it.level);
        out.push(it.el);
      } else if (it.k === "atom" && !used.has(it.id)) out.push(it.node);
      else if (it.k === "keep" && !head.includes(it) && !tail.includes(it)) out.push(it.node);
    }
    for (const it of tail) out.push((it as { node: Node }).node);
    if (trail) {
      const lastText = [...out].reverse().find((n): n is Text => n.nodeType === Node.TEXT_NODE && !!(n as Text).data);
      if (lastText && !/\s$/.test(lastText.data)) this.write(u, lastText, lastText.data + " ");
    }
    place(level.parent, out);
  }

  private blank(u: Unit, level: Level): void {
    for (const it of level.items) {
      if (it.k === "text") this.write(u, it.node, "");
      else if (it.k === "flat") for (const t of it.texts) this.write(u, t, "");
      else if (it.k === "slot" && it.level) this.blank(u, it.level);
    }
  }

  private renderDual(u: Unit, tree: PNode[]): void {
    const anchor = u.inline ? null : (u.chain[0] ?? lastNode(u.level));
    // a block would split the line when the run is followed by more inline content ("[show]") or sits in an inline box
    const inline = u.inline || (!!anchor && this.midLine(anchor));
    const tr = document.createElement("plamo-tr");
    tr.setAttribute("translate", "no");
    tr.style.cssText = inline
      ? "display:inline;margin-left:.35em;opacity:.88;font-style:normal;"
      : "display:block;margin:.25em 0 0;opacity:.88;font-style:normal;";
    let inner: Node = tr;
    // inline units sit inside their link (and inherit its look); block units repeat the unwrapped wrappers
    for (const el of u.inline ? [] : u.chain) {
      const c = cleanClone(el, false);
      inner.appendChild(c);
      inner = c;
    }
    const used = new Set<number>();
    const build = (into: Node, ns: PNode[]) => {
      for (const n of ns) {
        if (typeof n === "string") {
          into.appendChild(document.createTextNode(n));
          continue;
        }
        const s = u.slots.get(n.id);
        if (!s || used.has(n.id)) {
          into.appendChild(document.createTextNode(plain(n.children)));
          continue;
        }
        used.add(n.id);
        if (s.k === "atom") into.appendChild(cleanClone(s.node, true));
        else if (!s.level) into.appendChild(cleanClone(s.el, true));
        else {
          const c = cleanClone(s.el, false);
          build(c, n.children);
          into.appendChild(c);
        }
      }
    };
    build(inner, tree);
    if (u.inline) u.level.parent.appendChild(tr);
    else if (anchor?.parentNode) anchor.parentNode.insertBefore(tr, anchor.nextSibling); // before any block that follows
    else u.outer.append(tr);
    u.trEl = tr;
  }

  /** More visible inline content follows `n` on its line, or `n` sits inside an inline-level box. */
  private midLine(n: Node): boolean {
    const p = n.parentElement;
    if (p?.closest(PageTranslator.CONTROL)) return true; // inside a button or link: a block would break the control
    if (p && p !== document.body && getComputedStyle(p).display.startsWith("inline")) return true;
    for (let x = n.nextSibling; x; x = x.nextSibling) {
      if (x.nodeType === Node.TEXT_NODE) {
        if ((x as Text).data.trim()) return true;
        continue;
      }
      if (x.nodeType !== Node.ELEMENT_NODE) continue;
      const el = x as Element;
      if (INVISIBLE_TAGS.has(el.tagName)) continue;
      if (this.isBlock(el)) return false;
      if (!this.isHidden(el) && (el.textContent?.trim() || ATOM_TAGS.has(el.tagName))) return true;
    }
    return false;
  }

  private restore(u: Unit): void {
    if (u.trEl) {
      u.trEl.remove();
      u.trEl = undefined;
    }
    if (!u.writes.size && !u.created.length) return;
    for (const [t, w] of u.writes) if (t.data === w.wrote) t.data = w.orig; // unless the page rewrote it since
    u.writes.clear();
    for (const n of u.created) (n as ChildNode).remove();
    u.created = [];
    const reorder = (level: Level) => {
      const nodes: Node[] = [];
      for (const it of level.items) {
        const n = it.k === "slot" || it.k === "flat" ? it.el : it.node;
        if (n.parentNode === level.parent) nodes.push(n);
        if (it.k === "slot" && it.level) reorder(it.level);
      }
      place(level.parent, nodes);
    };
    reorder(u.level);
  }

  // ------------------------------------------------------------ dynamic pages

  private onMutations(records: MutationRecord[]): void {
    if (!this.active) return;
    for (const r of records) {
      // typing in a field or editable region is the user's, not the page's
      if (inEditField(r.target)) continue;
      if (r.type === "characterData") {
        const t = r.target as Text;
        if (this.ours.has(t)) continue;
        const u = this.byText.get(t);
        if (u) {
          if (u.writes.get(t)?.wrote === t.data) continue;
          this.dirtyRoots.add(u.outer);
        } else if (!this.isOurs(t)) this.dirtyRoots.add(t);
        continue;
      }
      for (const n of r.addedNodes) {
        if (this.isOurs(n)) continue;
        // inserted into a unit's run: that unit has to be redone
        const u = this.unitAround(r.target);
        this.dirtyRoots.add(u ? u.outer : n);
      }
      for (const n of r.removedNodes) {
        if (n.nodeType !== Node.TEXT_NODE || this.ours.has(n)) continue;
        const u = this.byText.get(n as Text);
        if (u) this.dirtyRoots.add(u.outer);
      }
    }
    if (this.dirtyRoots.size) {
      clearTimeout(this.rescanTimer);
      this.rescanTimer = window.setTimeout(() => this.flushDirty(), 400);
    }
  }

  /** the unit whose run contains `node` (a text node or an element inside the run) */
  private unitAround(node: Node): Unit | undefined {
    for (let n: Node | null = node; n && n !== document.body; n = n.parentNode) {
      const u = this.byParent.get(n);
      if (u) return this.units.includes(u) ? u : undefined;
      if (n !== node && n.nodeType === Node.ELEMENT_NODE && this.isBlock(n as Element)) return undefined; // left the run
    }
    return undefined;
  }

  private isOurs(n: Node): boolean {
    if (this.ours.has(n)) return true;
    const el = n.nodeType === Node.ELEMENT_NODE ? (n as Element) : n.parentElement;
    return !!el?.closest("plamo-tr, plamo-translate-ui");
  }

  private flushDirty(): void {
    if (!this.active) return;
    const roots = [...this.dirtyRoots].filter((n) => n.isConnected);
    this.dirtyRoots.clear();
    for (const root of roots) {
      // redo every unit inside the changed element
      const el = root.nodeType === Node.ELEMENT_NODE ? (root as Element) : root.parentElement;
      if (el) {
        for (const u of [...this.units]) {
          if (el === u.outer || el.contains(u.outer) || (u.outer.contains(el) && u.texts.some((t) => el.contains(t)))) this.drop(u);
        }
      }
      this.scan(root);
    }
    // drop units whose element left the document
    for (const u of [...this.units]) if (!u.outer.isConnected) this.drop(u);
    this.pump();
  }
}

// ------------------------------------------------------------------ DOM helpers

function textsIn(n: Node): Text[] {
  if (n.nodeType === Node.TEXT_NODE) return [n as Text];
  const out: Text[] = [];
  const w = document.createTreeWalker(n, NodeFilter.SHOW_TEXT);
  for (let t = w.nextNode(); t; t = w.nextNode()) out.push(t as Text);
  return out;
}

function levelParents(level: Level, out: Node[] = []): Node[] {
  out.push(level.parent);
  for (const it of level.items) if (it.k === "slot" && it.level) levelParents(it.level, out);
  return out;
}

function lastNode(level: Level): Node | undefined {
  for (let i = level.items.length - 1; i >= 0; i--) {
    const it = level.items[i];
    const n = it.k === "slot" || it.k === "flat" ? it.el : it.node;
    if (n.parentNode === level.parent) return n;
  }
  return undefined;
}

/** Make `nodes` consecutive children of `parent` in this order, where the first of them currently is. */
function place(parent: Node, nodes: Node[]): void {
  if (!nodes.length) return;
  const set = new Set(nodes);
  let first: Node | null = null;
  for (const c of parent.childNodes) {
    if (set.has(c)) {
      first = c;
      break;
    }
  }
  let prev: Node | null = first ? first.previousSibling : parent.lastChild;
  for (const n of nodes) {
    const want = prev ? prev.nextSibling : parent.firstChild;
    if (want !== n) parent.insertBefore(n, want);
    prev = n;
  }
}

/** Copy for the dual-mode translation, without ids (they must stay unique). */
function cleanClone<T extends Node>(n: T, deep: boolean): T {
  const c = n.cloneNode(deep) as T;
  if (c.nodeType === Node.ELEMENT_NODE) {
    const e = c as unknown as Element;
    e.removeAttribute("id");
    for (const d of e.querySelectorAll("[id]")) d.removeAttribute("id");
  }
  return c;
}
