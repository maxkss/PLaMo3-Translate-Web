"""Compare the ONNX-friendly re-implementation against pfnet's original modeling code (fp32, CPU)."""

import gc
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from plamo3_onnx import Plamo3ForOnnx

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../../models/src/plamo-3-translate-2b").resolve()
MAX_NEW = 48
EOS = {1, 16}

CASES = [
    ("The meeting has been moved to next Monday because the client requested more time.", "ko"),
    ("会議は来週の月曜日に変更されました。資料は前日までに共有してください。", "ko"),
    ("안녕하세요, 오늘 날씨가 정말 좋네요.", "en"),
]

tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
prompts = []
for text, tgt in CASES:
    p = tok.apply_chat_template([{"role": "user", "content": text}], target_lang=tgt, add_generation_prompt=True, tokenize=False)
    ids = tok(p, add_special_tokens=False, return_tensors="pt").input_ids
    prompts.append(ids)
    print(repr(p[:120]), ids.shape)

# ---- reference ----
ref = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True, dtype=torch.float32).eval()
ref_out = []
with torch.no_grad():
    for ids in prompts:
        logits = ref(ids).logits[0, -1]
        gen = ref.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, eos_token_id=list(EOS), pad_token_id=3)
        ref_out.append((logits, gen[0, ids.shape[1]:].tolist()))
del ref
gc.collect()

# ---- re-implementation ----
model = Plamo3ForOnnx.from_pretrained(SRC, dtype=torch.float32)
cfg = model.cfg
nkv, hd = cfg["num_key_value_heads"], cfg["head_dim"]


def greedy(ids):
    past = [torch.zeros(1, nkv, 0, hd) for _ in range(2 * cfg["num_hidden_layers"])]
    out, first = [], None
    cur = ids
    for _ in range(MAX_NEW):
        logits, *past = model(cur, *past)
        if first is None:
            first = logits[0, -1]
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt in EOS:
            break
        cur = torch.tensor([[nxt]])
    return first, out


ok = True
with torch.no_grad():
    for (ref_logits, ref_gen), ids in zip(ref_out, prompts):
        logits, gen = greedy(ids)
        diff = (logits - ref_logits).abs().max().item()
        same = gen == ref_gen
        ok &= same and diff < 1e-2
        print(f"max|dlogit|={diff:.2e}  same_tokens={same}")
        print("  ref:", tok.decode(ref_gen))
        print("  new:", tok.decode(gen))

# activation range check for fp16 safety
with torch.no_grad():
    stats = []
    x = model.embed_tokens(prompts[0])
    past = [torch.zeros(1, nkv, 0, hd) for _ in range(2 * cfg["num_hidden_layers"])]
    for i, layer in enumerate(model.layers):
        x, _, _ = layer(x, past[2 * i], past[2 * i + 1])
        stats.append(round(x.abs().max().item(), 1))
    print("residual max|x| per layer:", stats)

print("OK" if ok else "MISMATCH")
