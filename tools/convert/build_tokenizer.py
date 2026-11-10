"""Build the compact tokenizer asset used by the extension + reference test vectors.

Scores are pre-rounded exactly like tokenization_plamo.py (Python's round-half-even on score*1e4),
so the JS Viterbi port reproduces the reference token ids bit-for-bit.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../../models/src/plamo-3-translate-2b")
ap.add_argument("--out", default="../../extension/public/tokenizer/plamo3.json")
ap.add_argument("--vectors", default="../../extension/test/tokenizer_vectors.json")
args = ap.parse_args()

src = Path(args.src).resolve()
rows = [json.loads(line) for line in (src / "tokenizer.jsonl").open(encoding="utf-8")]
cfg = json.loads((src / "tokenizer_config.json").read_text())

asset = {
    "tokens": [r[0] for r in rows],
    "scores": [round(float(r[1]) * 1e4) for r in rows],
    "byte": [1 if r[2] == "BYTE" else 0 for r in rows],
    "break_around_consecutive_spaces_threshold": cfg.get("break_around_consecutive_spaces_threshold"),
    "break_around_repeated_chars_threshold": cfg.get("break_around_repeated_chars_threshold"),
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(asset, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("wrote", out, out.stat().st_size / 1e6, "MB")

tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
texts = [
    "Hello, world!",
    "The meeting has been moved to next Monday because the client requested more time.",
    "会議は来週の月曜日に変更されました。資料は前日までに共有してください。",
    "안녕하세요, 오늘 날씨가 정말 좋네요. 녕늘씨뷁",
    "简体中文和繁體中文混合文本。",
    "Emoji 😀👍🏽 and symbols ∑∫√ — “quotes” ‘single’…",
    "spaces    and\ttabs\n\nnew lines\r\nCRLF",
    "repeated !!!!!!!! aaaa bbbbbbb ———— ....",
    "  leading and trailing  ",
    "Code: `const x = a?.b ?? 0;` <div class=\"x\">html</div> https://example.com/path?q=1&r=2",
    "Ünïcödé àccents ñ ß Ελληνικά русский текст العربية हिन्दी ไทย",
    "　全角スペース　と　ｶﾀｶﾅ ﾊﾝｶｸ",
    "mixed<|plamo:tag|>control<|plamo:msg|>tokens<|plamo:bos|>",
    "<|plamo:begin_x and <|plamo: broken",
    "\r\r\r\r\n\n\n\n",
    "𠮷野家 𩸽 surrogate pairs",
]
for text, tgt in [(texts[1], "ko"), (texts[2], "ko"), (texts[3], "en")]:
    texts.append(
        tok.apply_chat_template([{"role": "user", "content": text}], target_lang=tgt, add_generation_prompt=True, tokenize=False)
    )
texts.append(
    tok.apply_chat_template(
        [{"role": "user", "content": "Page"}],
        target_lang="ko",
        source_lang="en",
        context="Document title: Example",
        add_generation_prompt=True,
        tokenize=False,
    )
)

vectors = [{"text": t, "ids": tok(t, add_special_tokens=False).input_ids} for t in texts]
vp = Path(args.vectors)
vp.parent.mkdir(parents=True, exist_ok=True)
vp.write_text(json.dumps(vectors, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", vp, len(vectors), "vectors")
print(vectors[-1])
