"""Export PLaMo 3 Translate to a single fp16 ONNX graph with KV cache IO."""

import argparse
import shutil
import tempfile
from pathlib import Path

import onnx
import torch

from plamo3_onnx import Plamo3ForOnnx

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../../models/src/plamo-3-translate-2b")
ap.add_argument("--out", default="../../models/onnx/plamo-3-translate-2b/onnx/model_traced_fp16.onnx")
ap.add_argument("--opset", type=int, default=21)
args = ap.parse_args()

src = Path(args.src).resolve()
out = Path(args.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)

model = Plamo3ForOnnx.from_pretrained(src, dtype=torch.float16)
cfg = model.cfg
L, nkv, hd = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["head_dim"]

input_names = ["input_ids"]
output_names = ["logits"]
dynamic_axes = {"input_ids": {1: "sequence_length"}}
past = []
for i in range(L):
    for kind in ("key", "value"):
        input_names.append(f"past_key_values.{i}.{kind}")
        output_names.append(f"present.{i}.{kind}")
        dynamic_axes[f"past_key_values.{i}.{kind}"] = {2: "past_sequence_length"}
        dynamic_axes[f"present.{i}.{kind}"] = {2: "total_sequence_length"}
        past.append(torch.zeros(1, nkv, 3, hd, dtype=torch.float16))

ids = torch.tensor([[1, 16, 100, 200]], dtype=torch.long)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp) / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            (ids, *past),
            str(tmp_path),
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    del model
    m = onnx.load(str(tmp_path), load_external_data=True)
    data_name = out.name + "_data"
    if (out.parent / data_name).exists():
        (out.parent / data_name).unlink()
    onnx.save_model(m, str(out), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)

print("saved", out, sum(f.stat().st_size for f in out.parent.glob(out.name + "*")) / 1e9, "GB")
