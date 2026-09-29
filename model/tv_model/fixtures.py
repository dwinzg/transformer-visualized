"""Golden fixtures for the TypeScript engine's parity tests.

Run `python -m tv_model.fixtures micro --out ../packages/engine/test/fixtures/micro`.
"""

from __future__ import annotations

import argparse
import json
import os
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
    an engine bug in a nonlinearity pass the parity tests unnoticed. Drawn from a local
    generator, not the global RNG: `GPT(cfg)`'s own default init already consumes an
    architecture-dependent number of draws from the global stream, so seeding it globally would
    make the fixture depend on that unrelated detail and would also reseed the caller's RNG.
    """
    model = GPT(cfg)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            noise = torch.randn(param.shape, generator=generator)
            if name.endswith(("ln_1.weight", "ln_2.weight", "ln_f.weight")):
                param.copy_(1.0 + 0.1 * noise)
            elif name.endswith(".bias"):
                param.copy_(0.1 * noise)
            else:
                param.copy_(0.3 * noise)
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


TINY_PROMPTS: dict[str, str] = {
    "story": "Once upon a time, there was a little",
    "cat": "The cat sat on the",
    "because": "Lily was sad because",
}


def write_model_fixtures(
    model_path: Path, tokenizer_path: Path, prompts: dict[str, str], out_dir: Path
) -> None:
    """Golden traces of a shipped model for fixed prompts."""
    from .io import load_model
    from .tokenizer import load_tokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path)
    tokenizer = load_tokenizer(tokenizer_path)
    cases = []
    for name, text in prompts.items():
        token_ids = tokenizer.encode(text).ids
        file = f"trace-{name}.safetensors"
        _write_trace(model, token_ids, out_dir / file)
        cases.append({"name": name, "text": text, "tokenIds": token_ids, "file": file})
    model_rel = Path(os.path.relpath(model_path.resolve(), out_dir.resolve())).as_posix()
    index = {"model": model_rel, "cases": cases}
    (out_dir / "cases.json").write_text(json.dumps(index, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write golden fixtures for the engine tests.")
    parser.add_argument("preset", choices=["micro", "tiny"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args(argv)
    if args.preset == "micro":
        write_micro_fixtures(args.out)
    else:
        if args.model is None or args.tokenizer is None:
            parser.error("the tiny preset needs --model and --tokenizer")
        write_model_fixtures(args.model, args.tokenizer, TINY_PROMPTS, args.out)


if __name__ == "__main__":
    main()
