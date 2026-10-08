"""Llama-style decoder-only transformer, the reference for the engine's second architecture.

Keeps GPT-2's plan and changes four parts: RMSNorm instead of LayerNorm, rotary position
embeddings (RoPE) instead of a learned position table, a SwiGLU feed forward instead of GELU, and
grouped-query attention, where several query heads share one key and value head. LLaMA (Touvron
et al., 2023a) brought the first three, and Llama 2's larger models added the fourth (Touvron et
al., 2023b). There are no biases. The output stays tied to the token embedding, as in our GPT-2,
so the two models have about the same size. Like gpt.py, every step is written out so it can be
recorded and compared with the TypeScript engine.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig
from .gpt import Trace


class RMSNorm(nn.Module):
    """Divides each vector by its root mean square, then scales it. No mean, no shift."""

    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        out = x * torch.rsqrt(mean_square + self.eps) * self.weight
        if trace is not None:
            trace.update({"meanSquare": mean_square.squeeze(-1), "out": out})
        return out


def rope_angles(seq: int, d_head: int, base: float) -> torch.Tensor:
    """[seq, d_head / 2]. Pair i of the vector at position p turns by p · base^(-2i / d_head)."""
    freqs = base ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    return torch.arange(seq, dtype=torch.float32)[:, None] * freqs[None, :]


def rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Turns each pair of numbers (x[2i], x[2i+1]) by its angle. x is [..., seq, d_head]."""
    even, odd = x[..., 0::2], x[..., 1::2]
    cos, sin = angles.cos(), angles.sin()
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.n_kv, self.d_head = cfg.n_heads, cfg.n_kv_heads, cfg.d_head
        self.rope_base = cfg.rope_base
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.d_head, cfg.d_model, bias=False)
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        split = lambda t, n: t.view(batch, seq, n, self.d_head).transpose(1, 2)  # noqa: E731
        q_raw = split(self.wq(x), self.n_heads)
        k_raw = split(self.wk(x), self.n_kv)
        v = split(self.wv(x), self.n_kv)
        angles = rope_angles(seq, self.d_head, self.rope_base).to(x.device)
        q, k = rope(q_raw, angles), rope(k_raw, angles)
        # Query head h reads key and value head h // group.
        group = self.n_heads // self.n_kv
        k_per_q = k.repeat_interleave(group, dim=1)
        v_per_q = v.repeat_interleave(group, dim=1)
        scores = q @ k_per_q.transpose(-2, -1)
        scaled_masked = (scores * (1.0 / math.sqrt(self.d_head))).masked_fill(
            ~self.mask[:seq, :seq], float("-inf")
        )
        weights = F.softmax(scaled_masked, dim=-1)
        out = weights @ v_per_q
        concat = out.transpose(1, 2).contiguous().view(batch, seq, self.n_heads * self.d_head)
        y = self.wo(concat)
        if trace is not None:
            trace["heads"] = [
                {
                    "qBeforeRope": q_raw[:, h],
                    "kBeforeRope": k_raw[:, h // group],
                    "q": q[:, h],
                    "k": k_per_q[:, h],
                    "v": v_per_q[:, h],
                    "scores": scores[:, h],
                    "scaledMasked": scaled_masked[:, h],
                    "weights": weights[:, h],
                    "out": out[:, h],
                }
                for h in range(self.n_heads)
            ]
            trace["attnConcat"] = concat
            trace["attnOut"] = y
        return y


class SwiGLU(nn.Module):
    """Two projections of the input. One goes through SiLU and opens or closes the other."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_mlp, bias=False)
        self.w_down = nn.Linear(cfg.d_mlp, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        hidden = F.silu(gate) * up
        y = self.w_down(hidden)
        if trace is not None:
            trace.update({"mlpGate": gate, "mlpUp": up, "mlpAct": hidden, "mlpOut": y})
        return y


class LlamaBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm_2 = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        n1: Trace | None = {} if trace is not None else None
        n2: Trace | None = {} if trace is not None else None
        resid = x + self.attn(self.norm_1(x, n1), trace)
        out = resid + self.mlp(self.norm_2(resid, n2), trace)
        if trace is not None:
            trace.update({"input": x, "ln1": n1, "residAfterAttn": resid, "ln2": n2, "output": out})
        return out


class Llama(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.h = nn.ModuleList([LlamaBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model, cfg.layer_norm_eps)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # The same scaling GPT-2 uses for the projections that write into the residual stream.
        for name, param in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def forward(self, idx: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        seq = idx.shape[1]
        if seq > self.cfg.context_length:
            raise ValueError(
                f"sequence length {seq} exceeds context length {self.cfg.context_length}"
            )
        x = self.wte(idx)
        tok = x
        layer_traces: list[Trace] = []
        for block in self.h:
            layer_trace: Trace | None = {} if trace is not None else None
            x = block(x, layer_trace)
            if layer_trace is not None:
                layer_traces.append(layer_trace)
        final: Trace | None = {} if trace is not None else None
        normed = self.norm_f(x, final)
        logits = normed @ self.wte.weight.T
        if trace is not None:
            trace.update(
                {
                    "tokenEmbeddings": tok,
                    "embeddings": tok,
                    "layers": layer_traces,
                    "lnFinal": final,
                    "logits": logits,
                }
            )
        return logits


def build_model(cfg: ModelConfig) -> nn.Module:
    """The network for a config, GPT-2 or Llama."""
    from .gpt import GPT

    return Llama(cfg) if cfg.arch == "llama" else GPT(cfg)
