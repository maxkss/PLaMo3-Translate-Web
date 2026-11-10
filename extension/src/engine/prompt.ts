/** PFML3 prompt construction, mirroring the model's chat_template.jinja. */

export interface PromptOptions {
  text: string;
  targetLang: string;
  sourceLang?: string | null;
  /** Reference context (e.g. page title) that helps interpretation; not translated. */
  context?: string | null;
  /** Previous source/translation pairs. */
  history?: { source: string; target: string }[];
  glossary?: { source: string; target: string }[];
  /** plain | polite, only valid with targetLang === "ja". */
  form?: "plain" | "polite" | null;
}

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
  for (const h of o.history ?? []) {
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
  if (o.form && tgt === "ja") p += "<|plamo:key|>form<|plamo:val|>" + o.form;
  p += "<|plamo:msg|>";
  return p;
}
