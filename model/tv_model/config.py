"""Model shapes shared by training, export and fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    def to_engine_json(self) -> dict[str, int | float]:
        """The config as the TypeScript engine reads it (camelCase keys)."""
        return {
            "vocabSize": self.vocab_size,
            "contextLength": self.context_length,
            "dModel": self.d_model,
            "nLayers": self.n_layers,
            "nHeads": self.n_heads,
            "dHead": self.d_head,
            "dMlp": self.d_mlp,
            "layerNormEps": self.layer_norm_eps,
        }


def config_from_engine_json(data: Mapping[str, int | float]) -> ModelConfig:
    """Inverse of `ModelConfig.to_engine_json`. Rejects a `dHead` that disagrees with the shapes."""
    cfg = ModelConfig(
        vocab_size=int(data["vocabSize"]),
        context_length=int(data["contextLength"]),
        d_model=int(data["dModel"]),
        n_layers=int(data["nLayers"]),
        n_heads=int(data["nHeads"]),
        d_mlp=int(data["dMlp"]),
        layer_norm_eps=float(data["layerNormEps"]),
    )
    if int(data["dHead"]) != cfg.d_head:
        raise ValueError(f"dHead {data['dHead']} does not match dModel / nHeads = {cfg.d_head}")
    return cfg


TINY = ModelConfig(
    vocab_size=4096, context_length=128, d_model=128, n_layers=4, n_heads=4, d_mlp=512
)
MICRO = ModelConfig(vocab_size=64, context_length=16, d_model=16, n_layers=2, n_heads=2, d_mlp=64)
PRESETS: dict[str, ModelConfig] = {"tiny": TINY, "micro": MICRO}
