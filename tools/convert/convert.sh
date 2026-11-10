#!/usr/bin/env bash
# Download pfnet/plamo-3-translate-{0.8b,2b}, build the fused ONNX graph, quantize to q4f16 and verify.
# Usage: convert.sh [0.8b|2b]   (default 2b)
# Output: models/onnx/plamo-3-translate-<size>/onnx/model_q4f16.onnx(+_data)  (0.8b ~0.5 GB, 2b ~1.6 GB)
set -euo pipefail
cd "$(dirname "$0")"

SIZE=${1:-2b}
NAME=plamo-3-translate-$SIZE
case $SIZE in
  0.8b) REV=573eb1be9fd8c44402650810c680104e5aec81e8; WEIGHTS="model.safetensors" ;;
  2b)   REV=78355e2f59605d1c2c7c46e915333ffded24c8ff; WEIGHTS="model.safetensors.index.json model-00001-of-00002.safetensors model-00002-of-00002.safetensors" ;;
  *) echo "unknown size: $SIZE" >&2; exit 1 ;;
esac

SRC=../../models/src/$NAME
OUT=../../models/onnx/$NAME/onnx
mkdir -p "$SRC"
for f in config.json generation_config.json modeling_plamo.py tokenization_plamo.py tokenizer.jsonl \
         tokenizer_config.json special_tokens_map.json chat_template.jinja $WEIGHTS; do
  echo "download $f"
  curl -fsSL --retry 5 -C - -o "$SRC/$f" "https://huggingface.co/pfnet/$NAME/resolve/$REV/$f"
done

uv sync
uv run python verify_torch.py "$SRC"                                            # reimplementation == original
uv run python build_fused.py --src "$SRC" --out "$OUT/model_fp16.onnx" --static-kv --early-slice --batch  # fused fp16 graph
uv run python quantize.py --quantize-embedding --src "$OUT/model_fp16.onnx" --out "$OUT/model_q4f16.onnx"
uv run python build_tokenizer.py --src "$SRC"                                   # same tokenizer for 0.8b and 2b
uv run python verify_variants.py "$NAME" model_q4f16.onnx                        # sample translations (CPU)
