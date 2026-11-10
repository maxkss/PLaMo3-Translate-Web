"""ONNX-friendly re-implementation of PLaMo 3 (Plamo3ForCausalLM) for inference.

Mathematically equivalent to pfnet's modeling_plamo.py for batch size 1, no padding,
and total sequence length <= window_size (2048), which is the regime the browser
extension runs in. Differences that do not change results in that regime:

* RMSNorm offsets are folded into the weights.
* RoPE is applied to keys once when they enter the cache (positions are absolute
  0..T-1, identical to the original "rotate the whole cache every step" path).
* Sliding-window masking is omitted (all keys are inside the 2048-token window).

Graph IO (one graph handles both prefill and decode):
  inputs : input_ids int64 [1, S]
           past_key_values.{i}.key / .value  [1, n_kv, P, head_dim]
  outputs: logits float32 [1, 1, vocab]   (last position only)
           present.{i}.key / .value [1, n_kv, P+S, head_dim]
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


def is_full_attn(pattern: int, layer_idx: int) -> bool:
    return (layer_idx + 1) % pattern == 0


def rope_inv_freq(dim: int, params: dict) -> tuple[torch.Tensor, float]:
    base = params["rope_theta"]
    freq = base ** (torch.arange(0, dim, 2).float() / dim)
    if params.get("rope_type", "default") == "default":
        return 1.0 / freq, 1.0
    assert params["rope_type"] == "yarn" and not params["truncate"]
    factor = params["factor"]
    d_h = dim // 2
    orig = params["original_max_position_embeddings"]
    low = d_h * math.log(orig / (params["beta_fast"] * 2 * math.pi)) / math.log(base)
    high = d_h * math.log(orig / (params["beta_slow"] * 2 * math.pi)) / math.log(base)
    interpolation = 1.0 / (factor * freq)
    extrapolation = 1.0 / freq
    ramp = (torch.arange(d_h).to(torch.float32) - low) / (high - low)
    mask = 1 - torch.clamp(ramp, 0, 1)
    inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
    return inv_freq, 0.1 * math.log(factor) + 1.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # offset already folded in
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: dict, layer_idx: int) -> None:
        super().__init__()
        self.nq = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.group = self.nq // self.nkv
        hidden = cfg["hidden_size"]
        self.qkv_proj = nn.Linear(hidden, (self.nq + 2 * self.nkv) * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nq * self.hd, hidden, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.hd, cfg["rms_norm_eps"])
        kind = "full_attention" if is_full_attn(cfg["sliding_window_pattern"], layer_idx) else "sliding_attention"
        inv_freq, conc = rope_inv_freq(self.hd, cfg["rope_parameters"][kind])
        # plain attribute (not a buffer) so Module.to(fp16) keeps it in fp32
        self.inv_freq = inv_freq
        self.concentration = conc

    def rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        freqs = pos.float()[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None] * self.concentration
        sin = emb.sin()[None, None] * self.concentration
        xf = x.float()
        return (xf * cos + rotate_half(xf) * sin).to(x.dtype)

    def forward(self, x, past_k, past_v):
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [self.nq * self.hd, self.nkv * self.hd, self.nkv * self.hd], dim=-1)
        q = q.view(b, s, self.nq, self.hd).transpose(1, 2)
        k = k.view(b, s, self.nkv, self.hd).transpose(1, 2)
        v = v.view(b, s, self.nkv, self.hd).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        p = past_k.shape[2]
        pos = torch.arange(s, device=x.device) + p
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)
        present_k, present_v = k, v
        t = k.shape[2]

        # GQA: query head h uses kv head h // group (repeat_interleave semantics).
        k = k[:, :, None].expand(b, self.nkv, self.group, t, self.hd).reshape(b, self.nq, t, self.hd)
        v = v[:, :, None].expand(b, self.nkv, self.group, t, self.hd).reshape(b, self.nq, t, self.hd)

        scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (1.0 / math.sqrt(self.hd))
        key_pos = torch.arange(t, device=x.device)
        allowed = key_pos[None, :] <= pos[:, None]
        scores = torch.where(allowed[None, None], scores, torch.tensor(float("-inf")))
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(b, s, self.nq * self.hd)
        return self.o_proj(out), present_k, present_v


class MLP(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.inter = cfg["intermediate_size"]
        self.gate_up_proj = nn.Linear(cfg["hidden_size"], self.inter * 2, bias=False)
        self.down_proj = nn.Linear(self.inter, cfg["hidden_size"], bias=False)

    def forward(self, x):
        h = self.gate_up_proj(x)
        g, u = torch.split(h, [self.inter, self.inter], dim=-1)
        return self.down_proj(torch.nn.functional.silu(g) * u)


class Layer(nn.Module):
    def __init__(self, cfg: dict, idx: int) -> None:
        super().__init__()
        eps = cfg["rms_norm_eps"]
        hidden = cfg["hidden_size"]
        self.mixer = Attention(cfg, idx)
        self.mlp = MLP(cfg)
        self.pre_mixer_norm = RMSNorm(hidden, eps)
        self.post_mixer_norm = RMSNorm(hidden, eps)
        self.pre_mlp_norm = RMSNorm(hidden, eps)
        self.post_mlp_norm = RMSNorm(hidden, eps)

    def forward(self, x, pk, pv):
        a, k, v = self.mixer(self.pre_mixer_norm(x), pk, pv)
        x = x + self.post_mixer_norm(a)
        x = x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))
        return x, k, v


# Offsets of each RMSNorm in the original implementation (weight_eff = offset + weight).
NORM_OFFSETS = {
    "q_norm": 1.0,
    "k_norm": 1.0,
    "pre_mixer_norm": 1.0,
    "post_mixer_norm": 1.0 / 5,
    "pre_mlp_norm": 1.0,
    "post_mlp_norm": 1.0 / (5**1.5),
    "model.norm": 1.0,
}


class Plamo3ForOnnx(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        assert not cfg.get("scale_embedding", False)
        assert not cfg.get("tie_word_embeddings", False)
        self.cfg = cfg
        self.n_layers = cfg["num_hidden_layers"]
        self.vocab = cfg["vocab_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(cfg["hidden_size"], ((self.vocab + 15) // 16) * 16, bias=False)

    def forward(self, input_ids, *past):
        x = self.embed_tokens(input_ids)
        presents = []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, past[2 * i], past[2 * i + 1])
            presents += [k, v]
        x = self.norm(x[:, -1:, :])
        logits = self.lm_head(x)[..., : self.vocab].float()
        return (logits, *presents)

    @classmethod
    def from_pretrained(cls, path: str | Path, dtype: torch.dtype = torch.float16) -> "Plamo3ForOnnx":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        model = cls(cfg)
        if (path / "model.safetensors.index.json").exists():
            index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        else:  # single-file checkpoint
            with safe_open(path / "model.safetensors", framework="pt") as fh:
                index = {k: "model.safetensors" for k in fh.keys()}
        state: dict[str, torch.Tensor] = {}
        by_file: dict[str, list[str]] = {}
        for name, fname in index.items():
            by_file.setdefault(fname, []).append(name)
        for fname, names in by_file.items():
            with safe_open(path / fname, framework="pt") as f:
                for name in names:
                    t = f.get_tensor(name).float()
                    new = name.replace("model.layers.layers.", "layers.").replace("model.embed_tokens", "embed_tokens")
                    new = new.replace("model.norm", "norm")
                    if new == "norm.weight":
                        t = t + NORM_OFFSETS["model.norm"]
                    else:
                        for key, off in NORM_OFFSETS.items():
                            if new.endswith(f".{key}.weight"):
                                t = t + off
                                break
                    state[new] = t
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert not missing, missing
        return model.to(dtype).eval()
