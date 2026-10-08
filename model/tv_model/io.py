"""Save and load models as safetensors files that the web engine can read."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from .config import config_from_engine_json
from .llama import build_model

FORMAT_ID = "transformer-visualized/gpt2/1"
# Llama-style files have their own format, so an older engine refuses them instead of misreading.
LLAMA_FORMAT_ID = "transformer-visualized/llama/1"


def format_for(cfg) -> str:
    return LLAMA_FORMAT_ID if cfg.arch == "llama" else FORMAT_ID


def _sort_metadata_header(path: Path) -> None:
    """Rewrite a safetensors file's header so its `__metadata__` keys are sorted.

    safetensors 0.8.0 serializes `__metadata__` from a Rust HashMap, whose iteration order is
    randomized per process (stable within one process, different across separate runs). That
    turns every regeneration into a spurious diff. Sorting the keys and re-encoding the header
    compactly reproduces exactly the same bytes every tensor entry already had -- only the
    key/value pairs inside `__metadata__` are permuted, so the encoded length before padding is
    unchanged; the trailing space padding that keeps the tensor data 8-byte aligned is
    recomputed to fill the same header length, so the 8-byte length prefix and every data
    offset stay correct.
    """
    data = bytearray(path.read_bytes())
    header_len = int.from_bytes(data[:8], "little")
    header = json.loads(bytes(data[8 : 8 + header_len]))
    if "__metadata__" not in header:
        return
    header["__metadata__"] = dict(sorted(header["__metadata__"].items()))
    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(new_header) > header_len:
        raise AssertionError(
            f"{path}: sorting metadata keys grew the header from {header_len} to "
            f"{len(new_header)} bytes"
        )
    new_header += b" " * (header_len - len(new_header))
    data[8 : 8 + header_len] = new_header
    path.write_bytes(bytes(data))


def save_model(
    model: nn.Module, path: Path, extra_metadata: Mapping[str, str] | None = None
) -> None:
    metadata = {"format": format_for(model.cfg), "config": json.dumps(model.cfg.to_engine_json())}
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
    _sort_metadata_header(path)


def load_model(path: Path) -> nn.Module:
    """A GPT or a Llama, by the file's arch. Both take token ids and an optional trace."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
        if metadata.get("format") not in (FORMAT_ID, LLAMA_FORMAT_ID):
            raise ValueError(f"{path}: unknown format {metadata.get('format')!r}")
        cfg = config_from_engine_json(json.loads(metadata["config"]))
        if metadata.get("format") != format_for(cfg):
            raise ValueError(
                f"{path}: expected format {format_for(cfg)!r}, got {metadata.get('format')!r}"
            )
        state = {name: f.get_tensor(name) for name in f.keys()}
    model = build_model(cfg)
    model.load_state_dict(state, strict=True)
    return model.eval()
