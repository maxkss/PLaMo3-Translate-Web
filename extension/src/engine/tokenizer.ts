/**
 * Exact port of pfnet's Plamo3Tokenizer (tokenization_plamo.py): a Unigram/Viterbi tokenizer with
 * byte fallback, special-token boundaries and forced splits around runs of spaces / repeated chars.
 *
 * The asset (public/tokenizer/plamo3.json) is produced by tools/convert/build_tokenizer.py and holds
 * scores already rounded the same way as the reference (round-half-even of score * 1e4).
 */

export interface TokenizerAsset {
  tokens: string[];
  scores: number[];
  byte: number[];
  break_around_consecutive_spaces_threshold: number | null;
  break_around_repeated_chars_threshold: number | null;
}

const BOUNDARY_CHAR = "";
const BOUNDARY_CP = 0xee00;
const BOUNDARY_TOKEN_ID = 10_000_000;
const UNKNOWN_SCORE = -10_000_000;
const NO_TOKEN = -1;

export class PlamoTokenizer {
  readonly tokens: string[];
  readonly vocab: Map<string, number>;
  private readonly isByte: Uint8Array;
  private readonly byteIds = new Int32Array(256);
  private readonly byteValue: Int16Array;
  /** Trie: edge key = node * 0x110000 + codepoint -> child node. */
  private readonly edges = new Map<number, number>();
  private nodeToken: Int32Array;
  private nodeScore: Int32Array;
  private nodeCount = 1;
  private readonly spTokenRe = /(<\|plamo:[^|\s]{0,64}\|>)/gu;
  private readonly matcherRe: RegExp | null;
  private readonly encoder = new TextEncoder();

  constructor(asset: TokenizerAsset) {
    this.tokens = asset.tokens;
    this.vocab = new Map(asset.tokens.map((t, i) => [t, i]));
    const n = asset.tokens.length;
    this.isByte = Uint8Array.from(asset.byte);
    this.byteValue = new Int16Array(n).fill(-1);
    // Rough upper bound on trie nodes; grown on demand.
    this.nodeToken = new Int32Array(1 << 19).fill(NO_TOKEN);
    this.nodeScore = new Int32Array(1 << 19);

    for (let id = 0; id < n; id++) {
      const tok = asset.tokens[id];
      if (asset.byte[id]) {
        const b = parseInt(tok.slice(3, 5), 16);
        this.byteIds[b] = id;
        this.byteValue[id] = b;
        continue;
      }
      this.insert(tok, id, asset.scores[id]);
    }
    this.insert(BOUNDARY_CHAR, BOUNDARY_TOKEN_ID, 0);

    const patterns: string[] = [];
    const rep = asset.break_around_repeated_chars_threshold;
    const sp = asset.break_around_consecutive_spaces_threshold;
    // Python's `.` matches anything but "\n"; `u` flag makes it code-point based like Python str.
    if (rep != null) patterns.push(`([^\\n])\\2{${rep - 1},}`);
    if (sp != null) patterns.push(` {${sp},}`);
    this.matcherRe = patterns.length ? new RegExp(`(${patterns.join("|")})`, "gu") : null;
  }

  private insert(token: string, id: number, score: number): void {
    let node = 0;
    for (const ch of token) {
      const key = node * 0x110000 + ch.codePointAt(0)!;
      let next = this.edges.get(key);
      if (next === undefined) {
        next = this.nodeCount++;
        if (next >= this.nodeToken.length) this.grow();
        this.edges.set(key, next);
      }
      node = next;
    }
    this.nodeToken[node] = id;
    this.nodeScore[node] = score;
  }

  private grow(): void {
    const tok = new Int32Array(this.nodeToken.length * 2).fill(NO_TOKEN);
    tok.set(this.nodeToken);
    const sc = new Int32Array(this.nodeScore.length * 2);
    sc.set(this.nodeScore);
    this.nodeToken = tok;
    this.nodeScore = sc;
  }

  get bosId(): number {
    return this.vocab.get("<|plamo:bos|>")!;
  }

  tokenId(token: string): number {
    const id = this.vocab.get(token);
    if (id === undefined) throw new Error(`unknown token ${token}`);
    return id;
  }

  encode(text: string): number[] {
    let data = text.replace(this.spTokenRe, `${BOUNDARY_CHAR}$1${BOUNDARY_CHAR}`);
    if (this.matcherRe) data = data.replace(this.matcherRe, `${BOUNDARY_CHAR}$1${BOUNDARY_CHAR}`);
    const cps: number[] = [];
    for (const ch of data) cps.push(ch.codePointAt(0)!);
    const n = cps.length;

    const cost = new Float64Array(n + 1).fill(Infinity);
    cost[n] = 0;
    const pathLen = new Int32Array(n + 1);
    const pathId = new Int32Array(n + 1);

    for (let i = n - 1; i >= 0; i--) {
      let best = Infinity;
      let bestLen = 0;
      let bestId = NO_TOKEN;
      let node = 0;
      // Increasing length with `<=` == reference's decreasing length with strict `<` (longest wins ties).
      for (let j = i; j < n; j++) {
        const next = this.edges.get(node * 0x110000 + cps[j]);
        if (next === undefined) break;
        node = next;
        const id = this.nodeToken[node];
        if (id !== NO_TOKEN) {
          const s = cost[j + 1] - this.nodeScore[node];
          if (s <= best) {
            best = s;
            bestLen = j + 1 - i;
            bestId = id;
          }
        }
      }
      // Unknown-character sentinel (byte fallback), evaluated last with strict `<`.
      const s = cost[i + 1] - UNKNOWN_SCORE;
      if (s < best) {
        best = s;
        bestLen = 1;
        bestId = NO_TOKEN;
      }
      cost[i] = best;
      pathLen[i] = bestLen;
      pathId[i] = bestId;
    }

    const out: number[] = [];
    let pos = 0;
    while (pos < n) {
      const id = pathId[pos];
      if (id >= 0) {
        if (id !== BOUNDARY_TOKEN_ID) out.push(id);
      } else {
        const cp = cps[pos];
        const ch = cp === BOUNDARY_CP ? BOUNDARY_CHAR : String.fromCodePoint(cp);
        for (const b of this.encoder.encode(ch)) out.push(this.byteIds[b]);
      }
      pos += pathLen[pos];
    }
    return out;
  }

  /** Bytes of a token (byte tokens map to a single raw byte). */
  tokenBytes(id: number): Uint8Array {
    const b = this.byteValue[id];
    if (b >= 0) return Uint8Array.of(b);
    return this.encoder.encode(this.tokens[id]);
  }

  decode(ids: number[]): string {
    const dec = new StreamDecoder(this);
    let s = "";
    for (const id of ids) s += dec.push(id);
    return s + dec.flush();
  }
}

/** Incremental detokenizer: handles UTF-8 sequences split across byte-fallback tokens. */
export class StreamDecoder {
  private readonly dec = new TextDecoder("utf-8");
  private readonly tok: PlamoTokenizer;
  constructor(tok: PlamoTokenizer) {
    this.tok = tok;
  }
  push(id: number): string {
    return this.dec.decode(this.tok.tokenBytes(id), { stream: true });
  }
  flush(): string {
    return this.dec.decode();
  }
}
