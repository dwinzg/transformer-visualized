"""Save and load models as safetensors files that the web engine can read."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .config import config_from_engine_json
from .gpt import GPT

FORMAT_ID = "transformer-visualized/gpt2/1"


def save_model(model: GPT, path: Path, extra_metadata: Mapping[str, str] | None = None) -> None:
    metadata = {"format": FORMAT_ID, "config": json.dumps(model.cfg.to_engine_json())}
    if extra_metadata:
        reserved = sorted(set(extra_metadata) & set(metadata))
        if reserved:
            raise ValueError(f"extra metadata may not override {reserved}")
        metadata.update(extra_metadata)
    tensors = {
        name: tensor.detach().to(torch.float32).contiguous().cpu()
        for name, tensor in model.state_dict().items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def load_model(path: Path) -> GPT:
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        if metadata.get("format") != FORMAT_ID:
            raise ValueError(
                f"{path}: expected format {FORMAT_ID!r}, got {metadata.get('format')!r}"
            )
        cfg = config_from_engine_json(json.loads(metadata["config"]))
        state = {name: f.get_tensor(name) for name in f.keys()}
    model = GPT(cfg)
    model.load_state_dict(state, strict=True)
    return model.eval()
