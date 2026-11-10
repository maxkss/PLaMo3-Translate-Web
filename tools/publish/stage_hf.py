"""Stage a converted model as a Hugging Face repo folder (hard links, no copies).

    python3 stage_hf.py <size> <repo_id> <out_dir>

Layout of the staged repo:
    README.md                    model card (license metadata, "Built with PLaMo")
    LICENSE, LICENSE.ja          PLaMo Community License (copied from PFN's repo; required for redistribution)
    config.json                  original PFN config (reference)
    manifest.json                file list with size + sha256 (the extension verifies downloads against it)
    onnx/model_q4f16.onnx(+_data)
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "0.8b": ("pfnet/plamo-3-translate-0.8b", "573eb1be9fd8c44402650810c680104e5aec81e8"),
    "2b": ("pfnet/plamo-3-translate-2b", "78355e2f59605d1c2c7c46e915333ffded24c8ff"),
}

size, repo_id, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
name = f"plamo-3-translate-{size}"
base_repo, base_rev = SOURCES[size]
src = ROOT / "models/src" / name
onnx_dir = ROOT / "models/onnx" / name / "onnx"

if out.exists():
    shutil.rmtree(out)
(out / "onnx").mkdir(parents=True)


def link(a: Path, b: Path) -> None:
    try:
        os.link(a, b)
    except OSError:
        shutil.copy2(a, b)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


for f in ("model_q4f16.onnx", "model_q4f16.onnx_data"):
    link(onnx_dir / f, out / "onnx" / f)
shutil.copy2(src / "config.json", out / "config.json")
shutil.copy2(ROOT / "models/license/LICENSE-en", out / "LICENSE")
shutil.copy2(ROOT / "models/license/LICENSE-ja", out / "LICENSE.ja")

files = []
for rel in ("onnx/model_q4f16.onnx", "onnx/model_q4f16.onnx_data"):
    p = out / rel
    files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256(p)})
    print(f"{rel}: {files[-1]['size']} {files[-1]['sha256']}")

cfg = json.loads((src / "config.json").read_text())
manifest = {
    "format": "plamo3-onnx-webgpu/3",
    "kv_cache": "static (seqlens_k / total_sequence_length inputs; past and present may share one buffer)",
    "batch": "symbolic batch dimension; batch > 1 for single-token decode steps (per-row lengths in seqlens_k)",
    "model": name,
    "base_model": base_repo,
    "base_revision": base_rev,
    "dtype": "q4f16",
    "num_hidden_layers": cfg["num_hidden_layers"],
    "num_key_value_heads": cfg["num_key_value_heads"],
    "head_dim": cfg["head_dim"],
    "files": files,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

total_gb = sum(f["size"] for f in files) / 1e9
card = f"""---
license: other
license_name: plamo-community-license
license_link: LICENSE
base_model: {base_repo}
base_model_relation: quantized
pipeline_tag: translation
library_name: onnxruntime
tags:
  - plamo
  - onnx
  - webgpu
  - onnxruntime-web
  - translation
language:
  - ja
  - en
  - ko
  - zh
  - multilingual
---

# PLaMo 3 Translate {size.upper()} — ONNX (q4f16, WebGPU)

**Built with PLaMo.**

This is a 4-bit ONNX conversion of [{base_repo}](https://huggingface.co/{base_repo}) (revision `{base_rev[:12]}`),
made for running in the browser with [onnxruntime-web](https://onnxruntime.ai/) on WebGPU.
It is used by the PLaMo Translate (On-device) Chrome extension.

This is an unofficial conversion. It is not made or endorsed by Preferred Networks.

## Files

| file | size |
|---|---|
| `onnx/model_q4f16.onnx` + `onnx/model_q4f16.onnx_data` | {total_gb:.2f} GB |
| `manifest.json` | sizes and SHA-256 of the files above |

## Conversion

- The graph is built directly from the safetensors weights with onnxruntime fused ops:
  `GroupQueryAttention` (RoPE, KV cache, sliding window as in the original), `SimplifiedLayerNormalization`.
  The RMSNorm offsets of the original model are folded into the norm weights.
- Linear layers are quantized to 4-bit (`MatMulNBits`, block 32, symmetric). The token embedding uses `GatherBlockQuantized`.
  Activations and the KV cache are fp16.
- After the last layer's attention only the last position is computed (the KV cache is already written), which
  shortens the prompt pass.
- The batch dimension is symbolic, so several sequences can decode in one run.
- The reimplementation matches the original PyTorch model (max |Δlogit| ≈ 3e-5 in fp32, identical greedy output).
- The 4-bit model has only been spot-checked on sample sentences. Its quality has not been benchmarked.

## Inputs / outputs

- `input_ids`: int64 `[batch, seq]`
- `seqlens_k`: int32 `[batch]` = past + seq − 1 for each row; `total_sequence_length`: int32 scalar = the largest past + seq
- `past_key_values.{{i}}.key|value`: fp16 `[batch, {cfg['num_key_value_heads']}, kv, {cfg['head_dim']}]` for `i` in 0..{cfg['num_hidden_layers'] - 1}
- `logits`: float32 `[batch, 1, {cfg['vocab_size']}]` (last position only)
- `present.{{i}}.key|value`: the updated KV cache

The KV cache can be used two ways:
- **Static (WebGPU):** allocate `[batch, kv_heads, capacity, head_dim]` buffers once and pass the same buffer as
  `past_key_values.i.*` and `present.i.*` (onnxruntime detects the shared buffer and writes new entries in place).
  The decode step then has fixed shapes, so it works with WebGPU graph capture.
- **Growing (CPU):** pass a past of exactly `past` positions; `present` comes back with `past + seq` positions.

Batching: several sequences can decode together (one token per row, each row with its own length in `seqlens_k`).
A prompt with an existing past is processed with batch 1 (onnxruntime's GroupQueryAttention does not accept
multi-token input with a past for batch > 1); its KV can then be copied into a row of the batch buffers.
With up to 3 rows the greedy tokens on WebGPU were identical to decoding each sequence alone in our tests; larger
batches use other kernels and can change near-tied choices.

The context limit is 2048 tokens (the model's sliding window). Longer sequences need extra masking that is not implemented here.

The prompt format and tokenizer are the same as the original model (see `chat_template.jinja` in the base repo).
Requires onnxruntime(-web) 1.22 or newer with the WebGPU or CPU execution provider.

## License

The [PLaMo Community License](LICENSE) ([日本語](LICENSE.ja)) applies to this model, the same as the original.
Personal and non-commercial use is free. Commercial use requires prior registration with Preferred Networks,
and is limited to users whose annual revenue does not exceed 1 billion yen. See the license text for details.
"""
(out / "README.md").write_text(card)
print("staged", out)
