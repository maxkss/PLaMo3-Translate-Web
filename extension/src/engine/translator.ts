import { Generator, MAX_CONTEXT } from "./generator.ts";
import { buildPrompt } from "./prompt.ts";
import { PlamoTokenizer, StreamDecoder } from "./tokenizer.ts";

export interface TranslateInput {
  text: string;
  targetLang: string;
  sourceLang?: string | null;
  context?: string | null;
}

export interface TranslateStats {
  outputTokens: number;
  decodeMs: number;
  prefillMs: number;
}

/** Source tokens per model call. Long inputs are split at paragraph/sentence boundaries. */
const CHUNK_TOKENS = 384;

const HAS_LETTER = /\p{L}/u;

/** Numbered paired tags opened but not closed in `s`. */
function openTags(s: string): string[] {
  const stack: string[] = [];
  for (const m of s.matchAll(/<(\/?)([a-z]+\d+)>/g)) {
    if (!m[1]) stack.push(m[2]);
    else {
      const i = stack.lastIndexOf(m[2]);
      if (i >= 0) stack.length = i;
    }
  }
  return stack;
}

export class Translator {
  readonly tok: PlamoTokenizer;
  private readonly gen: Generator;
  private readonly eosIds: number[];
  private readonly cache = new Map<string, string>();
  private readonly cacheLimit = 4000;

  constructor(tok: PlamoTokenizer, gen: Generator) {
    this.tok = tok;
    this.gen = gen;
    // generation_config.json: eos_token_id = [1 (<|plamo:bos|>), 16 (<|plamo:tag|>)]
    this.eosIds = [tok.tokenId("<|plamo:bos|>"), tok.tokenId("<|plamo:tag|>")];
  }

  cacheKey(i: TranslateInput): string {
    return `${i.targetLang}\u0001${i.sourceLang ?? ""}\u0001${i.context ?? ""}\u0001${i.text}`;
  }

  cached(i: TranslateInput): string | undefined {
    return this.cache.get(this.cacheKey(i));
  }

  /**
   * Translate one text. Long texts are split into chunks that are generated concurrently (the engine batches them);
   * `onDelta` gets the text so far in reading order (finished chunks plus the partial output of the first unfinished one).
   * `interactive` requests are served before background ones.
   */
  async translate(
    input: TranslateInput,
    onDelta?: (textSoFar: string) => void,
    signal?: AbortSignal,
    interactive = false,
  ): Promise<{ text: string; stats: TranslateStats }> {
    const stats: TranslateStats = { outputTokens: 0, decodeMs: 0, prefillMs: 0 };
    const key = this.cacheKey(input);
    const hit = this.cache.get(key);
    if (hit !== undefined) return { text: hit, stats };

    // Keep surrounding whitespace untouched; only translate the core.
    const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(input.text)!;
    const [, lead, core, trail] = m;
    if (!core || !HAS_LETTER.test(core)) return { text: input.text, stats };

    const pieces = this.split(core);
    const final: (string | null)[] = pieces.map((p) => (p.translate ? null : p.text));
    const partial: string[] = pieces.map(() => "");
    const show = () => {
      let s = "";
      for (let i = 0; i < pieces.length; i++) {
        if (final[i] === null) {
          s += partial[i];
          break;
        }
        s += final[i];
      }
      onDelta?.(lead + s);
    };
    const firstOpen = () => final.indexOf(null);
    if (signal?.aborted) throw new DOMException("aborted", "AbortError");
    await Promise.all(
      pieces.map(async (piece, i) => {
        if (!piece.translate) return;
        final[i] = await this.translateChunk(
          { ...input, text: piece.text },
          stats,
          (s) => {
            partial[i] = s;
            if (firstOpen() === i) show();
          },
          signal,
          piece.tokens,
          interactive,
        );
        show();
      }),
    );
    const result = lead + final.join("") + trail;
    if (!signal?.aborted) {
      if (this.cache.size >= this.cacheLimit) this.cache.delete(this.cache.keys().next().value!);
      this.cache.set(key, result);
    }
    return { text: result, stats };
  }

  private async translateChunk(
    input: TranslateInput,
    stats: TranslateStats,
    onDelta: (s: string) => void,
    signal?: AbortSignal,
    knownTokens?: number,
    interactive = false,
  ): Promise<string> {
    const m = /^(\s*)([\s\S]*?)(\s*)$/.exec(input.text)!;
    const [, lead, core, trail] = m;
    if (!core || !HAS_LETTER.test(core)) return input.text;

    const prompt = buildPrompt({ ...input, text: core });
    const ids = this.tok.encode(prompt);
    // source length bounds the output budget; reuse the count from split() when the piece had no outer whitespace
    const srcTokens = knownTokens !== undefined && core === input.text ? knownTokens : this.tok.encode(core).length;
    const dec = new StreamDecoder(this.tok);
    let text = "";
    const res = await this.gen.generate(ids, {
      maxNewTokens: Math.min(MAX_CONTEXT - ids.length, Math.max(48, srcTokens * 3 + 16)),
      eosIds: this.eosIds,
      signal,
      interactive,
      onToken: (id) => {
        const d = dec.push(id);
        if (!d) return; // incomplete UTF-8 sequence: nothing new to show yet
        text += d;
        onDelta(lead + text);
      },
    });
    if (res.stoppedBy === "abort") throw new DOMException("aborted", "AbortError");
    // Re-decode the final ids (the repetition guard may have trimmed the tail).
    text = this.tok.decode(res.ids);
    stats.outputTokens += res.ids.length;
    stats.decodeMs += res.decodeMs;
    stats.prefillMs += res.prefillMs;
    return lead + text.trim() + trail;
  }

  /**
   * Split into model-sized chunks, keeping separators (newlines) verbatim. Numbered inline tags (`<a1>…</a1>`, see
   * content/page.ts) are never cut apart: chunks are merged until every tag opened in a chunk is closed in it.
   */
  split(text: string): { text: string; translate: boolean; tokens?: number }[] {
    const whole = this.tok.encode(text).length;
    if (whole <= CHUNK_TOKENS) return [{ text, translate: true, tokens: whole }];
    const pieces = this.splitPlain(text);
    const out: { text: string; translate: boolean; tokens?: number }[] = [];
    let open: string[] = [];
    for (const p of pieces) {
      const prev = out[out.length - 1];
      if (open.length && prev) {
        prev.text += p.text;
        prev.translate ||= p.translate;
      } else out.push({ ...p });
      open = openTags(out[out.length - 1].text);
    }
    return out;
  }

  private splitPlain(text: string): { text: string; translate: boolean; tokens?: number }[] {
    const out: { text: string; translate: boolean; tokens?: number }[] = [];
    // paragraphs first
    for (const part of text.split(/(\n+)/)) {
      if (!part) continue;
      if (/^\n+$/.test(part)) {
        out.push({ text: part, translate: false });
        continue;
      }
      if (this.tok.encode(part).length <= CHUNK_TOKENS) {
        out.push({ text: part, translate: true });
        continue;
      }
      // then sentences, greedily packed
      const sentences = part.match(/[^.!?。！？]+(?:[.!?。！？]+["'”’」』)\]]*\s*|$)/gu) ?? [part];
      let buf = "";
      for (const s of sentences) {
        if (buf && this.tok.encode(buf + s).length > CHUNK_TOKENS) {
          out.push({ text: buf, translate: true });
          buf = "";
        }
        buf += s;
        // a single giant "sentence": hard-split by characters
        while (this.tok.encode(buf).length > CHUNK_TOKENS * 1.5) {
          const cut = Math.floor(buf.length / 2);
          out.push({ text: buf.slice(0, cut), translate: true });
          buf = buf.slice(cut);
        }
      }
      if (buf) out.push({ text: buf, translate: true });
    }
    return out;
  }
}
