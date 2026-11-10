"""Weight-only 4-bit quantization (MatMulNBits / GatherBlockQuantized) of the fp16 export -> q4f16."""

import argparse
import logging
from pathlib import Path

import onnx
from onnxruntime.quantization import matmul_nbits_quantizer as mnq

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../../models/onnx/plamo-3-translate-2b/onnx/model_fp16.onnx")
ap.add_argument("--out", default="../../models/onnx/plamo-3-translate-2b/onnx/model_q4f16.onnx")
ap.add_argument("--block-size", type=int, default=32)
ap.add_argument("--quantize-embedding", action="store_true", help="also quantize the token embedding (Gather)")
ap.add_argument("--lm-head-block-size", type=int, default=0,
                help="quantize the lm_head MatMul (node logits16_node) with this block size instead")
args = ap.parse_args()

logging.basicConfig(level=logging.WARNING)

model = onnx.load(args.src, load_external_data=True)
op_types = ("MatMul", "Gather") if args.quantize_embedding else ("MatMul",)
quant_axes = (("MatMul", 0), ("Gather", 1)) if args.quantize_embedding else (("MatMul", 0),)
cfg = mnq.DefaultWeightOnlyQuantConfig(
    block_size=args.block_size,
    is_symmetric=True,
    accuracy_level=4,
    op_types_to_quantize=op_types,
    quant_axes=quant_axes,
)
LM_HEAD = "logits16_node"
q = mnq.MatMulNBitsQuantizer(
    model,
    block_size=args.block_size,
    is_symmetric=True,
    accuracy_level=4,
    op_types_to_quantize=op_types,
    quant_axes=quant_axes,
    algo_config=cfg,
    nodes_to_exclude=[LM_HEAD] if args.lm_head_block_size else None,
)
q.process()
if args.lm_head_block_size:
    # second pass: only the lm_head, with its own block size (it always runs with M=1 when the graph slices early)
    cfg2 = mnq.DefaultWeightOnlyQuantConfig(block_size=args.lm_head_block_size, is_symmetric=True, accuracy_level=4, op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),))
    q = mnq.MatMulNBitsQuantizer(q.model.model, block_size=args.lm_head_block_size, is_symmetric=True, accuracy_level=4,
                                 op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),), algo_config=cfg2, nodes_to_include=[LM_HEAD])
    q.process()

out = Path(args.out).resolve()
data_name = out.name + "_data"
if (out.parent / data_name).exists():
    (out.parent / data_name).unlink()
onnx.save_model(q.model.model, str(out), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)

ops: dict[str, int] = {}
for n in q.model.model.graph.node:
    ops[n.op_type] = ops.get(n.op_type, 0) + 1
print("ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
print("saved", out, sum(f.stat().st_size for f in out.parent.glob(out.name + "*")) / 1e9, "GB")
