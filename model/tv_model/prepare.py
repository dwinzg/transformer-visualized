"""Run the data pipeline: download, clean, train the tokenizer, write fixtures and token files.

python -m tv_model.prepare --data-dir data \
    --tokenizer-out ../models/tiny/tokenizer.json \
    --cases-out ../packages/engine/test/fixtures/tokenizer/cases.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .data import REVISION, clean_stories, clean_story, download, iter_stories
from .encode import encode_to_file
from .tokenizer import (
    load_tokenizer,
    save_tokenizer,
    synthetic_samples,
    tokenizer_cases,
    train_tokenizer,
)


def _count(path: Path) -> tuple[int, int]:
    kept = dropped = 0
    for story in iter_stories(path):
        if clean_story(story) is None:
            dropped += 1
        else:
            kept += 1
    return kept, dropped


def prepare(
    data_dir: Path,
    tokenizer_out: Path,
    cases_out: Path,
    vocab_size: int = 4096,
    tokenizer_stories: int = 300_000,
    synthetic_cases: int = 250,
    seed: int = 0,
) -> dict:
    raw_dir = data_dir / "raw"
    train_path = download("train", raw_dir)
    valid_path = download("valid", raw_dir)

    sample = itertools.islice(clean_stories(iter_stories(train_path)), tokenizer_stories)
    save_tokenizer(train_tokenizer(sample, vocab_size), tokenizer_out)
    # Reload from the exported JSON so every later step uses exactly what the site ships.
    tokenizer = load_tokenizer(tokenizer_out)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"expected {vocab_size} tokens, got {tokenizer.get_vocab_size()}")

    # Seeded strings of vocabulary tokens, so the committed cases hold no dataset text.
    samples = synthetic_samples(tokenizer, synthetic_cases, seed)
    cases_out.parent.mkdir(parents=True, exist_ok=True)
    cases = tokenizer_cases(tokenizer, samples)
    cases_out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    stats: dict = {"vocab_size": vocab_size, "revision": REVISION, "tokenizer_cases": len(cases)}
    for split, path in (("train", train_path), ("valid", valid_path)):
        kept, dropped = _count(path)
        tokens = encode_to_file(
            tokenizer, clean_stories(iter_stories(path)), data_dir / f"{split}.bin"
        )
        stats[split] = {"kept": kept, "dropped": dropped, "tokens": tokens}
    (data_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare TinyStories for training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--tokenizer-out", type=Path, required=True)
    parser.add_argument("--cases-out", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-stories", type=int, default=300_000)
    args = parser.parse_args(argv)
    stats = prepare(
        args.data_dir, args.tokenizer_out, args.cases_out, args.vocab_size, args.tokenizer_stories
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
