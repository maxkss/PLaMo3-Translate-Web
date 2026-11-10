"""Build an optimized PLaMo 3 ONNX graph directly from the safetensors weights.

Uses onnxruntime fused ops so WebGPU runs ~10x fewer kernels per token than the traced export:
  * SimplifiedLayerNormalization (RMSNorm, offsets folded into the weights)
  * com.microsoft.GroupQueryAttention with do_rotary=1 (KV cache concat + RoPE + GQA + causal mask)
    - YaRN "concentration" is folded into the cos/sin caches (rotation is linear in cos/sin)
  * MatMul weights are later quantized to MatMulNBits by quantize.py

IO is identical to export_onnx.py so the extension's generator works with either graph.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors import safe_open

from plamo3_onnx import NORM_OFFSETS, is_full_attn, rope_inv_freq

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="../../models/src/plamo-3-translate-2b")
ap.add_argument("--out", default="../../models/onnx/plamo-3-translate-2b/onnx/model_fp16.onnx")
ap.add_argument("--max-pos", type=int, default=4096, help="rows of the RoPE cos/sin caches")
ap.add_argument("--static-kv", action="store_true",
                help="fixed-capacity KV cache shared between past/present (seqlens_k and total_sequence_length become inputs)")
ap.add_argument("--argmax", action="store_true", help="also output next_token = argmax(logits) computed on the GPU")
ap.add_argument("--fuse", action="store_true", help="SkipSimplifiedLayerNormalization for residual+norm, QuickGelu for SiLU")
ap.add_argument("--split-proj", action="store_true",
                help="separate q/k/v and gate/up MatMuls so onnxruntime's WebGPU MatMulNBitsQkv/MatMulNBitsMlp fusions apply")
ap.add_argument("--batch", action="store_true", help="symbolic batch dimension (several sequences per run)")
ap.add_argument("--logits-fp16", action="store_true", help="output fp16 logits (no Cast); the caller does argmax on the GPU")
ap.add_argument("--early-slice", action="store_true",
                help="after the last layer's attention keep only the last position (everything after it is per-token)")
args = ap.parse_args()

src = Path(args.src).resolve()
cfg = json.loads((src / "config.json").read_text())
H, L = cfg["hidden_size"], cfg["num_hidden_layers"]
NQ, NKV, HD = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
I = cfg["intermediate_size"]
V = cfg["vocab_size"]
EPS = cfg["rms_norm_eps"]

if (src / "model.safetensors.index.json").exists():
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
else:  # single-file checkpoint (e.g. 0.8B)
    with safe_open(src / "model.safetensors", framework="pt") as fh:
        index = {k: "model.safetensors" for k in fh.keys()}
handles = {f: safe_open(src / f, framework="pt") for f in set(index.values())}


def w(name: str) -> torch.Tensor:
    return handles[index[name]].get_tensor(name).float()


inits: list[onnx.TensorProto] = []
nodes: list[onnx.NodeProto] = []


def init(name: str, arr: np.ndarray) -> str:
    inits.append(numpy_helper.from_array(arr, name))
    return name


def node(op: str, inputs: list[str], outputs: list[str], domain: str = "", **attrs) -> None:
    nodes.append(helper.make_node(op, inputs, outputs, name=outputs[0] + "_node", domain=domain, **attrs))


def f16(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float16).numpy()


def norm_weight(name: str, key: str) -> np.ndarray:
    return f16(w(name) + NORM_OFFSETS[key])


def rms(x: str, weight: str, out: str) -> None:
    node("SimplifiedLayerNormalization", [x, weight], [out], axis=-1, epsilon=EPS, stash_type=1)


def linear(x: str, weight_name: str, out: str, rows: slice | None = None, suffix: str = "") -> None:
    # nn.Linear(x) = x @ W^T  (optionally only output features `rows` of W)
    wt = w(weight_name)
    if rows is not None:
        wt = wt[rows]
    wname = init(weight_name + suffix, f16(wt.T.contiguous()))
    node("MatMul", [x, wname], [out])


def ints(name: str, vals: list[int]) -> str:
    return init(name, np.array(vals, dtype=np.int64))


# ---- RoPE caches (fp16, concentration folded in)
caches = {}
for kind in ("sliding_attention", "full_attention"):
    inv_freq, conc = rope_inv_freq(HD, cfg["rope_parameters"][kind])
    t = torch.arange(args.max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [max_pos, HD/2]
    caches[kind] = (
        init(f"cos_{kind}", f16(freqs.cos() * conc)),
        init(f"sin_{kind}", f16(freqs.sin() * conc)),
    )

# ---- shape bookkeeping: seqlens_k = past + S - 1, total_sequence_length = past + S
if args.static_kv:
    pass  # provided by the caller as graph inputs
else:
  node("Shape", ["input_ids"], ["ids_shape"])
  node("Gather", ["ids_shape", ints("one_i", [1])], ["seq_len"], axis=0)  # [1]
  node("Shape", ["past_key_values.0.key"], ["past_shape"])
  node("Gather", ["past_shape", ints("two_i", [2])], ["past_len"], axis=0)  # [1]
  node("Add", ["past_len", "seq_len"], ["total_len"])
  node("Sub", ["total_len", ints("one_i2", [1])], ["seqlens_k_i64"])
  node("Cast", ["seqlens_k_i64"], ["seqlens_k"], to=TensorProto.INT32)  # [1]
  node("Squeeze", ["total_len"], ["total_len_scalar"])
  node("Cast", ["total_len_scalar"], ["total_sequence_length"], to=TensorProto.INT32)  # []

# ---- embeddings
init("embed_tokens", f16(w("model.embed_tokens.weight")))
node("Gather", ["embed_tokens", "input_ids"], ["h0"], axis=0)

shape_q = ints("shape_q", [0, 0, NQ, HD])
shape_k = ints("shape_k", [0, 0, NKV, HD])
shape_q_flat = ints("shape_q_flat", [0, 0, NQ * HD])
shape_k_flat = ints("shape_k_flat", [0, 0, NKV * HD])
split_qkv = ints("split_qkv", [NQ * HD, NKV * HD, NKV * HD])
split_gu = ints("split_gu", [I, I])

def add_norm(a: str, b: str, weight: str, out_norm: str, out_sum: str) -> None:
    """out_sum = a + b (the residual stream), out_norm = RMSNorm(out_sum) * weight."""
    if args.fuse:
        node("SkipSimplifiedLayerNormalization", [a, b, weight], [out_norm, "", "", out_sum], domain="com.microsoft", epsilon=EPS)
    else:
        node("Add", [a, b], [out_sum])
        rms(out_sum, weight, out_norm)


def last_pos(t: str, out: str) -> None:
    node("Slice", [t, ints(out + "_s", [-1]), ints(out + "_e", [2**62]), ints(out + "_a", [1])], [out])


carry = None  # (x1, mlp_out) of the previous layer; their sum is this layer's residual input
for l in range(L):
    p = f"model.layers.layers.{l}"
    kind = "full_attention" if is_full_attn(cfg["sliding_window_pattern"], l) else "sliding_attention"
    cos, sin = caches[kind]
    n = lambda s: f"l{l}_{s}"  # noqa: E731

    pre_mixer = init(n("pre_mixer_w"), norm_weight(f"{p}.pre_mixer_norm.weight", "pre_mixer_norm"))
    if carry is None:
        x = "h0"
        rms(x, pre_mixer, n("h"))
    else:
        x = n("x0")
        add_norm(carry[0], carry[1], pre_mixer, n("h"), x)
    if args.split_proj:
        qw = f"{p}.mixer.qkv_proj.weight"
        linear(n("h"), qw, n("q"), slice(0, NQ * HD), ".q")
        linear(n("h"), qw, n("k"), slice(NQ * HD, (NQ + NKV) * HD), ".k")
        linear(n("h"), qw, n("v"), slice((NQ + NKV) * HD, (NQ + 2 * NKV) * HD), ".v")
    else:
        linear(n("h"), f"{p}.mixer.qkv_proj.weight", n("qkv"))
        node("Split", [n("qkv"), split_qkv], [n("q"), n("k"), n("v")], axis=-1)
    node("Reshape", [n("q"), shape_q], [n("q4")])
    rms(n("q4"), init(n("q_norm_w"), norm_weight(f"{p}.mixer.q_norm.weight", "q_norm")), n("qn4"))
    node("Reshape", [n("qn4"), shape_q_flat], [n("qn")])
    node("Reshape", [n("k"), shape_k], [n("k4")])
    rms(n("k4"), init(n("k_norm_w"), norm_weight(f"{p}.mixer.k_norm.weight", "k_norm")), n("kn4"))
    node("Reshape", [n("kn4"), shape_k_flat], [n("kn")])
    node(
        "GroupQueryAttention",
        [n("qn"), n("kn"), n("v"), f"past_key_values.{l}.key", f"past_key_values.{l}.value", "seqlens_k", "total_sequence_length", cos, sin],
        [n("attn"), f"present.{l}.key", f"present.{l}.value"],
        domain="com.microsoft",
        num_heads=NQ,
        kv_num_heads=NKV,
        do_rotary=1,
        rotary_interleaved=0,
        scale=1.0 / math.sqrt(HD),
    )
    if args.early_slice and l == L - 1:
        # the KV cache is already written; o_proj, norms and MLP are per-token, and only the last token is used
        last_pos(n("attn"), n("attn_last"))
        last_pos(x, n("x0_last"))
        attn_in, x = n("attn_last"), n("x0_last")
    else:
        attn_in = n("attn")
    linear(attn_in, f"{p}.mixer.o_proj.weight", n("o"))
    rms(n("o"), init(n("post_mixer_w"), norm_weight(f"{p}.post_mixer_norm.weight", "post_mixer_norm")), n("on"))
    add_norm(x, n("on"), init(n("pre_mlp_w"), norm_weight(f"{p}.pre_mlp_norm.weight", "pre_mlp_norm")), n("m_in"), n("x1"))

    if args.split_proj:
        gw = f"{p}.mlp.gate_up_proj.weight"
        linear(n("m_in"), gw, n("g"), slice(0, I), ".gate")
        linear(n("m_in"), gw, n("u"), slice(I, 2 * I), ".up")
    else:
        linear(n("m_in"), f"{p}.mlp.gate_up_proj.weight", n("gu"))
        node("Split", [n("gu"), split_gu], [n("g"), n("u")], axis=-1)
    if args.fuse:
        node("QuickGelu", [n("g")], [n("silu")], domain="com.microsoft", alpha=1.0)  # x * sigmoid(1.0 * x) == SiLU
    else:
        node("Sigmoid", [n("g")], [n("gs")])
        node("Mul", [n("g"), n("gs")], [n("silu")])
    node("Mul", [n("silu"), n("u")], [n("act")])
    linear(n("act"), f"{p}.mlp.down_proj.weight", n("m"))
    rms(n("m"), init(n("post_mlp_w"), norm_weight(f"{p}.post_mlp_norm.weight", "post_mlp_norm")), n("mn"))
    carry = (n("x1"), n("mn"))

# ---- last position -> logits
last = lambda t, o: node("Slice", [t, ints(o + "_s", [-1]), ints(o + "_e", [2**62]), ints(o + "_a", [1])], [o])  # noqa: E731
final_norm = init("final_norm_w", f16(w("model.norm.weight") + NORM_OFFSETS["model.norm"]))
if args.early_slice:  # already a single position
    if args.fuse:
        node("SkipSimplifiedLayerNormalization", [carry[0], carry[1], final_norm], ["x_norm"], domain="com.microsoft", epsilon=EPS)
    else:
        node("Add", [carry[0], carry[1]], ["x_final"])
        rms("x_final", final_norm, "x_norm")
elif args.fuse:
    last(carry[0], "x1_last")
    last(carry[1], "mn_last")
    node("SkipSimplifiedLayerNormalization", ["x1_last", "mn_last", final_norm], ["x_norm"], domain="com.microsoft", epsilon=EPS)
else:
    node("Add", [carry[0], carry[1]], ["x_final"])
    last("x_final", "x_last")
    rms("x_last", final_norm, "x_norm")
lm = w("lm_head.weight")[:V]
init("lm_head_w", f16(lm.T.contiguous()))
node("MatMul", ["x_norm", "lm_head_w"], ["logits16"])
if args.argmax:
    node("ArgMax", ["logits16"], ["next_token"], axis=-1, keepdims=0)  # [1, 1] int64
elif args.logits_fp16:
    node("Identity", ["logits16"], ["logits"])
else:
    node("Cast", ["logits16"], ["logits"], to=TensorProto.FLOAT)

# ---- graph IO
Bdim = "batch" if args.batch else 1
inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [Bdim, "sequence_length"])]
if args.static_kv:
    inputs.append(helper.make_tensor_value_info("seqlens_k", TensorProto.INT32, [Bdim]))
    inputs.append(helper.make_tensor_value_info("total_sequence_length", TensorProto.INT32, []))
if args.argmax:
    outputs = [helper.make_tensor_value_info("next_token", TensorProto.INT64, [Bdim, 1])]
else:
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT16 if args.logits_fp16 else TensorProto.FLOAT, [Bdim, 1, V])]
past_dim, present_dim = ("max_sequence_length", "max_sequence_length") if args.static_kv else ("past_sequence_length", "total_sequence_length")
for l in range(L):
    for kind in ("key", "value"):
        inputs.append(helper.make_tensor_value_info(f"past_key_values.{l}.{kind}", TensorProto.FLOAT16, [Bdim, NKV, past_dim, HD]))
        outputs.append(helper.make_tensor_value_info(f"present.{l}.{kind}", TensorProto.FLOAT16, [Bdim, NKV, present_dim, HD]))

graph = helper.make_graph(nodes, "plamo3_translate", inputs, outputs, initializer=inits)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)])
model.ir_version = 10

out = Path(args.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
data_name = out.name + "_data"
(out.parent / data_name).unlink(missing_ok=True)
onnx.save_model(model, str(out), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
ops: dict[str, int] = {}
for nd in nodes:
    ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
print("nodes:", len(nodes), ops)
print("saved", out)
