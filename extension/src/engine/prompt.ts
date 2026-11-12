/** PFML3 prompt construction, mirroring the model's chat_template.jinja. */
import type { TextForm } from "../shared/common.ts";

export interface PromptOptions {
  text: string;
  targetLang: string;
  sourceLang?: string | null;
  /** Reference context (e.g. page title) that helps interpretation; not translated. */
  context?: string | null;
  /** Previous source/translation pairs. */
  history?: { source: string; target: string }[];
  glossary?: { source: string; target: string }[];
  /** plain | polite (TextForm): Japanese and Korean only (FORM_EXAMPLES). */
  form?: TextForm | null;
}

/**
 * Translations in each style, shown before the text (as history) when a form is asked for. The model knows the form
 * key from Japanese; for Korean it is weak on its own: -다 in 6 of 12 sentences of a blog post, with these examples 10
 * of 12 (a news text 12 of 12 either way, 7 of 12 with neither).
 */
const FORM_EXAMPLES: Record<string, Record<TextForm, { source: string; target: string }[]>> = {
  ko: {
    plain: [
      { source: "The museum opened in 1998 and attracts about two million visitors a year.", target: "이 박물관은 1998년에 문을 열었으며, 연간 약 200만 명의 방문객이 찾는다." },
      { source: "Most of the collection was donated by local families.", target: "소장품 대부분은 지역 가문들이 기증한 것이다." },
    ],
    polite: [
      { source: "The museum opened in 1998 and attracts about two million visitors a year.", target: "이 박물관은 1998년에 문을 열었으며, 연간 약 200만 명의 방문객이 찾습니다." },
      { source: "Most of the collection was donated by local families.", target: "소장품 대부분은 지역 가문들이 기증한 것입니다." },
    ],
  },
  ja: { plain: [], polite: [] },
};

const LANG_RE = /^[a-z]{2,3}(-[a-z]{4})?$/;

/** Neutralise reserved PFML markers inside user-provided text (the template rejects them). */
export function sanitize(text: string): string {
  return text.replaceAll("<|plamo:", "<|plamo​:");
}

function lang(code: string): string {
  const c = code === "ja(easy)" ? "ja-easy" : code;
  if (!LANG_RE.test(c)) throw new Error(`invalid language code: ${code}`);
  return c;
}

export function buildPrompt(o: PromptOptions): string {
  const tgt = lang(o.targetLang);
  const src = o.sourceLang ? lang(o.sourceLang) : null;
  let p = "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation";
  if (o.context) p += "<|plamo:tag|>context<|plamo:msg|>" + sanitize(o.context);
  const forms = FORM_EXAMPLES[tgt];
  const form = o.form && forms ? o.form : null;
  for (const h of o.history ?? (form ? forms[form] : [])) {
    p += "<|plamo:tag|>input" + (src ? "<|plamo:key|>lang<|plamo:val|>" + src : "");
    p += "<|plamo:msg|>" + sanitize(h.source);
    p += "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + tgt + "<|plamo:msg|>" + sanitize(h.target);
  }
  if (o.glossary?.length) {
    p += "<|plamo:tag|>glossary";
    for (const g of o.glossary) p += "<|plamo:index|>" + sanitize(g.source) + "<|plamo:item|>" + sanitize(g.target);
  }
  p += "<|plamo:tag|>input" + (src ? "<|plamo:key|>lang<|plamo:val|>" + src : "");
  p += "<|plamo:msg|>" + sanitize(o.text);
  p += "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>" + tgt;
  if (form) p += "<|plamo:key|>form<|plamo:val|>" + form;
  p += "<|plamo:msg|>";
  return p;
}
