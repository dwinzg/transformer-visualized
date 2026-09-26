"""Golden fixtures for the TypeScript engine's parity tests.

Run `python -m tv_model.fixtures micro --out ../packages/engine/test/fixtures/micro`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import MICRO, ModelConfig
from .gpt import GPT, Trace
from .io import save_model
from .trace import flatten_trace

MICRO_SEED = 0
MICRO_CASES: dict[str, list[int]] = {
    "single": [7],
    "five": [1, 2, 3, 4, 5],
    "full": [(i * 37 + 11) % MICRO.vocab_size for i in range(MICRO.context_length)],
}


def make_fixture_model(cfg: ModelConfig, seed: int) -> GPT:
    """Random weights large enough to exercise softmax, GELU and LayerNorm.

    The default initialization (std 0.02) keeps every activation tiny, which would let
    an engine bug in a nonlinearity pass the parity tests unnoticed.
    """
    torch.manual_seed(seed)
    model = GPT(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
            elif name.endswith(".bias"):
                param.copy_(0.1 * torch.randn_like(param))
            else:
                param.copy_(0.3 * torch.randn_like(param))
    return model.eval()


def _write_trace(model: GPT, token_ids: list[int], path: Path) -> None:
    trace: Trace = {}
    with torch.no_grad():
        model(torch.tensor([token_ids], dtype=torch.long), trace)
    save_file(flatten_trace(trace), str(path))


def write_micro_fixtures(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = make_fixture_model(MICRO, MICRO_SEED)
    save_model(
        model, out_dir / "model.safetensors", {"purpose": "Engine test fixture with random weights"}
    )
    cases = []
    for name, token_ids in MICRO_CASES.items():
        file = f"trace-{name}.safetensors"
        _write_trace(model, token_ids, out_dir / file)
        cases.append({"name": name, "tokenIds": token_ids, "file": file})
    index = {"model": "model.safetensors", "cases": cases}
    (out_dir / "cases.json").write_text(json.dumps(index, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write golden fixtures for the engine tests.")
    parser.add_argument("preset", choices=["micro"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    write_micro_fixtures(args.out)


if __name__ == "__main__":
    main()
