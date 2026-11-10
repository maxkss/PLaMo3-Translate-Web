"""Greedy-decode sample prompts with graph variants on onnxruntime CPU and compare token ids.

    uv run python verify_variants.py <model-name> <variant.onnx> [<variant.onnx> ...]

The first file is the reference. Variants may output `logits` or `next_token`, and may use a static KV cache
(inputs seqlens_k / total_sequence_length, fixed-capacity past/present).
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

NAME = sys.argv[1]
FILES = sys.argv[2:]
SRC = Path(f"../../models/src/{NAME}").resolve()
DIR = Path(f"../../models/onnx/{NAME}/onnx").resolve()
EOS = {1, 16}
MAXLEN = 256
CASES = [
    ("The meeting has been moved to next Monday because the client requested more time.", "ko"),
    ("会議は来週の月曜日に変更されました。資料は前日までに共有してください。", "ko"),
    ("안녕하세요, 오늘 날씨가 정말 좋네요.", "en"),
    ("WebGPU lets web pages run machine learning models directly on the user's graphics card.", "ja"),
    ("Le comité a examiné la proposition en détail et a conclu que le calendrier était trop ambitieux.", "ko"),
]
tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)


def run(path: str, ids: list[int]) -> list[int]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(DIR / path), so, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    past = [i for i in sess.get_inputs() if i.name.startswith("past_key_values")]
    static = "seqlens_k" in names
    h, d = past[0].shape[1], past[0].shape[3]
    feeds = {i.name: np.zeros((1, h, MAXLEN if static else 0, d), np.float16) for i in past}
    outs = [o.name for o in sess.get_outputs()]
    cur, plen, gen = ids, 0, []
    for _ in range(96):
        feeds["input_ids"] = np.array([cur], np.int64)
        if static:
            feeds["seqlens_k"] = np.array([plen + len(cur) - 1], np.int32)
            feeds["total_sequence_length"] = np.array(plen + len(cur), np.int32)
        res = dict(zip(outs, sess.run(None, feeds)))
        for i in past:
            feeds[i.name] = res[i.name.replace("past_key_values", "present")]
        plen += len(cur)
        nxt = int(res["next_token"].reshape(-1)[0]) if "next_token" in res else int(res["logits"][0, -1].argmax())
        if nxt in EOS:
            break
        gen.append(nxt)
        cur = [nxt]
    return gen


ok = True
for text, tgt in CASES:
    p = tok.apply_chat_template([{"role": "user", "content": text}], target_lang=tgt, add_generation_prompt=True, tokenize=False)
    ids = tok(p, add_special_tokens=False).input_ids
    ref = run(FILES[0], ids)
    print(f"[{tgt}] {tok.decode(ref)}")
    for f in FILES[1:]:
        got = run(f, ids)
        same = got == ref
        ok &= same
        print(f"   {'same' if same else 'DIFF'} {f}" + ("" if same else f": {tok.decode(got)}"))
print("ALL SAME" if ok else "SOME DIFFER")
