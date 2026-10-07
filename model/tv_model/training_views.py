"""Turn the snapshots of a training run into the small JSON the training playground shows.

python -m tv_model.training_views --snapshots runs/snapshots/snapshots \
    --tokenizer ../models/tiny/tokenizer.json --out ../models/tiny/training.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .gpt import GPT
from .io import load_model
from .sample import generate_text
from .tokenizer import load_tokenizer

# The same sentence the chapters follow, a story start, a fact from earlier in the text, and a
# made-up word to copy. Each shows a different skill appearing during training.
NEXT = "Lily wanted to play with her"
STORY = "Once upon a time"
RECALL = ("Sara had a cat named Fluffy. Every morning, Sara gave milk to", " Fluffy")
COPY = ('Ben said, "Zog." Mia said, "Zog." Tim said, "', "Zog")

STORY_TOKENS = 30
# Random tokens repeated twice. A head that looks from each token of the copy to the token right
# after its first appearance is an induction head (Olsson et al. 2022).
INDUCTION_HALF = 40
INDUCTION_SEEDS = 8


@torch.no_grad()
def top_next(model: GPT, tokenizer, text: str, count: int = 5) -> list[dict]:
    ids = tokenizer.encode(text).ids
    probs = torch.softmax(model(torch.tensor([ids]))[0, -1], dim=-1)
    values, indices = probs.topk(count)
    return [
        {"id": int(i), "text": tokenizer.decode([int(i)]), "p": round(float(v), 4)}
        for v, i in zip(values, indices, strict=True)
    ]


@torch.no_grad()
def chance_of(model: GPT, tokenizer, prompt: str, answer: str) -> float:
    """The chance the model writes exactly `answer` next, one token after another."""
    prompt_ids = tokenizer.encode(prompt).ids
    full = tokenizer.encode(prompt + answer).ids
    if full[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{answer!r} changes how {prompt!r} is split into tokens")
    answer_ids = full[len(prompt_ids) :]
    ids = prompt_ids + answer_ids
    logp = torch.log_softmax(model(torch.tensor([ids[:-1]]))[0], dim=-1)
    start = len(prompt_ids) - 1
    total = sum(float(logp[start + k, t]) for k, t in enumerate(answer_ids))
    return round(float(np.exp(total)), 4)


@torch.no_grad()
def induction_scores(model: GPT) -> list[list[float]]:
    """Per layer and head, the mean weight from a repeated token to the one after its first copy."""
    half = INDUCTION_HALF
    totals = np.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for seed in range(INDUCTION_SEEDS):
        rng = np.random.default_rng(seed)
        # Skip the 256 single bytes, so the tokens are ordinary pieces of words.
        first = rng.integers(256, model.cfg.vocab_size, size=half)
        ids = torch.from_numpy(np.concatenate([first, first])[None])
        trace: dict = {}
        model(ids, trace)
        rows = torch.arange(half + 1, 2 * half)
        for layer, layer_trace in enumerate(trace["layers"]):
            for head, head_trace in enumerate(layer_trace["heads"]):
                weights = head_trace["weights"][0]
                totals[layer, head] += float(weights[rows, rows - half + 1].mean())
    return [[round(float(s), 4) for s in row] for row in totals / INDUCTION_SEEDS]


def induction_chance() -> float:
    """The score of a head that spreads its weight evenly over every token it can see."""
    rows = range(INDUCTION_HALF + 1, 2 * INDUCTION_HALF)
    return round(float(np.mean([1 / (i + 1) for i in rows])), 4)


def printable(text: str) -> str:
    """An untrained model writes stray bytes. Control characters and broken bytes become spaces."""
    return "".join(c if c.isprintable() or c == "\n" else " " for c in text).replace("\ufffd", " ")


def snapshot_view(path: Path, tokenizer) -> dict:
    model = load_model(path)
    with safe_open(str(path), framework="pt") as f:
        meta = f.metadata()
    step = int(meta["step"])
    return {
        "step": step,
        "valLoss": round(float(meta["val_loss"]), 4),
        "next": top_next(model, tokenizer, NEXT),
        "story": printable(
            generate_text(model, tokenizer, STORY, STORY_TOKENS, 0.8, 40, 0)[len(STORY) :]
        ),
        "recall": chance_of(model, tokenizer, *RECALL),
        "copy": chance_of(model, tokenizer, *COPY),
        "induction": induction_scores(model),
    }


def build(snapshots: Path, tokenizer_path: Path, tokens_per_step: int = 64 * 128) -> dict:
    tokenizer = load_tokenizer(tokenizer_path)
    views = [snapshot_view(p, tokenizer) for p in sorted(snapshots.glob("step-*.safetensors"))]
    return {
        "prompts": {
            "next": NEXT,
            "story": STORY,
            "recall": {"text": RECALL[0], "answer": RECALL[1]},
            "copy": {"text": COPY[0], "answer": COPY[1]},
        },
        "tokensPerStep": tokens_per_step,
        "inductionChance": induction_chance(),
        "snapshots": views,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # The tiny preset trains on batches of 64 sequences of 128 tokens.
    parser.add_argument("--tokens-per-step", type=int, default=64 * 128)
    args = parser.parse_args(argv)
    data = build(args.snapshots, args.tokenizer, args.tokens_per_step)
    args.out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {len(data['snapshots'])} snapshots to {args.out}")


if __name__ == "__main__":
    main()
