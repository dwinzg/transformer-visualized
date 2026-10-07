"""Flatten a model trace into tensors named by their path in the engine's Trace type."""

from __future__ import annotations

from typing import Any

import torch

from .config import ModelConfig
from .gpt import Trace

_TOP_LEVEL = [
    "tokenEmbeddings",
    "positionEmbeddings",
    "embeddings",
    "lnFinal.mean",
    "lnFinal.variance",
    "lnFinal.out",
    "logits",
]
_PER_LAYER = [
    "input",
    "ln1.mean",
    "ln1.variance",
    "ln1.out",
    "attnConcat",
    "attnOut",
    "residAfterAttn",
    "ln2.mean",
    "ln2.variance",
    "ln2.out",
    "mlpHidden",
    "mlpAct",
    "mlpOut",
    "output",
]
_PER_HEAD = ["q", "k", "v", "scores", "scaledMasked", "weights", "out"]


# A Llama-style model has no position table, an RMSNorm that keeps only the mean square, a gated
# feed forward, and queries and keys before and after their rotation.
_LLAMA_TOP_LEVEL = ["tokenEmbeddings", "embeddings", "lnFinal.meanSquare", "lnFinal.out", "logits"]
_LLAMA_PER_LAYER = [
    "input",
    "ln1.meanSquare",
    "ln1.out",
    "attnConcat",
    "attnOut",
    "residAfterAttn",
    "ln2.meanSquare",
    "ln2.out",
    "mlpGate",
    "mlpUp",
    "mlpAct",
    "mlpOut",
    "output",
]
_LLAMA_PER_HEAD = ["qBeforeRope", "kBeforeRope", *_PER_HEAD]


def expected_trace_names(cfg: ModelConfig) -> set[str]:
    llama = cfg.arch == "llama"
    names = set(_LLAMA_TOP_LEVEL if llama else _TOP_LEVEL)
    for layer in range(cfg.n_layers):
        names.update(
            f"layers.{layer}.{name}" for name in (_LLAMA_PER_LAYER if llama else _PER_LAYER)
        )
        for head in range(cfg.n_heads):
            per_head = _LLAMA_PER_HEAD if llama else _PER_HEAD
            names.update(f"layers.{layer}.heads.{head}.{name}" for name in per_head)
    return names


def flatten_trace(trace: Trace) -> dict[str, torch.Tensor]:
    """Map a batch-of-one trace to float32 tensors keyed by path, e.g. `layers.0.heads.1.weights`.

    Names are engine Trace paths; the leading batch dimension is dropped from every tensor.
    """
    flat: dict[str, torch.Tensor] = {}

    def visit(path: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if value.shape[0] != 1:
                raise ValueError(f"{path}: expected a batch of one, got shape {tuple(value.shape)}")
            # clone(): traced tensors can share storage (a block's output is the next block's
            # input, heads are views of one buffer), and safetensors refuses shared memory.
            flat[path] = value[0].detach().to(torch.float32).contiguous().cpu().clone()
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{path}.{key}" if path else key, child)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(f"{path}.{index}", child)
        else:
            raise TypeError(f"{path}: unsupported trace value of type {type(value).__name__}")

    visit("", trace)
    return flat
