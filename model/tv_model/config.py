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
    # "gpt2", or "llama" for RMSNorm, rotary positions, SwiGLU and grouped-query attention.
    arch: str = "gpt2"
    # Llama only. Key and value heads, each shared by n_heads / n_kv_heads query heads.
    n_kv_heads: int = 0
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} is not divisible by n_heads {self.n_heads}")
        if self.arch not in ("gpt2", "llama"):
            raise ValueError(f"unknown arch {self.arch!r}")
        if self.arch == "llama" and (self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads):
            raise ValueError(
                f"n_heads {self.n_heads} is not a multiple of n_kv_heads {self.n_kv_heads}"
            )
        if self.arch == "llama" and not self.rope_base > 1:
            raise ValueError(f"rope_base must be above 1, got {self.rope_base}")
        if self.arch == "gpt2" and self.n_kv_heads != 0:
            raise ValueError("n_kv_heads is only for the llama arch")
        if self.arch == "llama" and self.d_head % 2:
            raise ValueError(f"rotary positions need an even d_head, got {self.d_head}")

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
            # GPT-2 files keep exactly the keys they always had.
            **(
                {"arch": "llama", "nKvHeads": self.n_kv_heads, "ropeBase": self.rope_base}
                if self.arch == "llama"
                else {}
            ),
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
        arch=str(data.get("arch", "gpt2")),
        n_kv_heads=int(data.get("nKvHeads", 0)),
        rope_base=float(data.get("ropeBase", 10_000.0)),
    )
    if int(data["dHead"]) != cfg.d_head:
        raise ValueError(f"dHead {data['dHead']} does not match dModel / nHeads = {cfg.d_head}")
    return cfg


TINY = ModelConfig(
    vocab_size=4096, context_length=128, d_model=128, n_layers=4, n_heads=4, d_mlp=512
)
MICRO = ModelConfig(vocab_size=64, context_length=16, d_model=16, n_layers=2, n_heads=2, d_mlp=64)
# The same budget as TINY. A SwiGLU feed forward has three matrices instead of two, so 344 hidden
# units cost about what GPT-2's 512 do.
LLAMA_TINY = ModelConfig(
    vocab_size=4096,
    context_length=128,
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_mlp=344,
    arch="llama",
    n_kv_heads=2,
)
LLAMA_MICRO = ModelConfig(
    vocab_size=64,
    context_length=16,
    d_model=16,
    n_layers=2,
    n_heads=4,
    d_mlp=48,
    arch="llama",
    n_kv_heads=2,
)
PRESETS: dict[str, ModelConfig] = {
    "tiny": TINY,
    "micro": MICRO,
    "llama-tiny": LLAMA_TINY,
    "llama-micro": LLAMA_MICRO,
}
