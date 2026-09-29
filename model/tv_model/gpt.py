"""GPT-2 style decoder-only transformer, the numerical reference for the web engine.

Follows GPT-2 (Radford et al., 2019): pre-LayerNorm blocks, learned absolute position
embeddings, GELU with the tanh approximation, and an output projection tied to the token
embedding. Attention is written out step by step instead of using a fused kernel, so every
intermediate value can be recorded and compared with the TypeScript engine.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .config import ModelConfig

Trace = dict[str, Any]


def _norm_trace(x: torch.Tensor, out: torch.Tensor) -> Trace:
    return {"mean": x.mean(dim=-1), "variance": x.var(dim=-1, unbiased=False), "out": out}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        mask = torch.tril(torch.ones(cfg.context_length, cfg.context_length, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        batch, seq, d_model = x.shape
        q, k, v = self.c_attn(x).split(d_model, dim=2)
        # [batch, seq, d_model] -> [batch, heads, seq, d_head]
        q = q.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.d_head).transpose(1, 2)
        scores = q @ k.transpose(-2, -1)
        scaled_masked = (scores * (1.0 / math.sqrt(self.d_head))).masked_fill(
            ~self.mask[:seq, :seq], float("-inf")
        )
        weights = F.softmax(scaled_masked, dim=-1)
        out = weights @ v
        concat = out.transpose(1, 2).contiguous().view(batch, seq, d_model)
        y = self.c_proj(concat)
        if trace is not None:
            trace["heads"] = [
                {
                    "q": q[:, h],
                    "k": k[:, h],
                    "v": v[:, h],
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


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        hidden = self.c_fc(x)
        activated = self.gelu(hidden)
        y = self.c_proj(activated)
        if trace is not None:
            trace["mlpHidden"] = hidden
            trace["mlpAct"] = activated
            trace["mlpOut"] = y
        return y


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        ln1 = self.ln_1(x)
        resid = x + self.attn(ln1, trace)
        ln2 = self.ln_2(resid)
        out = resid + self.mlp(ln2, trace)
        if trace is not None:
            trace["input"] = x
            trace["ln1"] = _norm_trace(x, ln1)
            trace["residAfterAttn"] = resid
            trace["ln2"] = _norm_trace(resid, ln2)
            trace["output"] = out
        return out


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.context_length, cfg.d_model)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.apply(self._init_weights)
        # GPT-2 scales residual projections by 1/sqrt(2 * n_layers) at initialization.
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, trace: Trace | None = None) -> torch.Tensor:
        seq = idx.shape[1]
        if seq > self.cfg.context_length:
            raise ValueError(
                f"sequence length {seq} exceeds context length {self.cfg.context_length}"
            )
        tok = self.wte(idx)
        pos = self.wpe(torch.arange(seq, device=idx.device)).unsqueeze(0).expand_as(tok)
        x = tok + pos
        layer_traces: list[Trace] = []
        for block in self.h:
            layer_trace: Trace | None = {} if trace is not None else None
            x = block(x, layer_trace)
            if layer_trace is not None:
                layer_traces.append(layer_trace)
        ln_f = self.ln_f(x)
        logits = ln_f @ self.wte.weight.T
        if trace is not None:
            trace["tokenEmbeddings"] = tok
            trace["positionEmbeddings"] = pos
            trace["embeddings"] = tok + pos
            trace["layers"] = layer_traces
            trace["lnFinal"] = _norm_trace(x, ln_f)
            trace["logits"] = logits
        return logits
