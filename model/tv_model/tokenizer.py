"""Byte-level BPE tokenizer trained on TinyStories and exported for the web engine."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from .data import EOT

FORMAT = "gpt2-byte-bpe"

# Inputs that exercise the pre-tokenizer and byte fallback. The engine's tokenizer must
# reproduce the ids for every one of them. Non-ASCII and invisible characters are escaped.
EDGE_CASES = [
    "",
    " ",
    "Hello, world!",
    "Once upon a time, there was a little girl named Lily.",
    "  two leading spaces",
    "trailing space ",
    "multiple   inner   spaces",
    "line one\nline two\n\nline four",
    "three\n\n\nnewlines",
    "windows\r\nline end",
    "tab\tseparated",
    "don't won't I'm we're they've he'll she'd",
    "it\u2019s a curly apostrophe",
    "DON'T SHOUT",
    "numbers 7 42 1234 3.14 1,000",
    'punctuation?! (brackets) [square] {curly} "quotes"',
    "symbols @#$%^&*_+=~`|\\/<>",
    "caf\u00e9 na\u00efve fa\u00e7ade",
    "cafe\u0301 with a combining accent",
    "no\u00a0break\u00a0space",
    "\u4f60\u597d\u4e16\u754c",
    "emoji \U0001f600\U0001f680",
    "zwj \U0001f469\u200d\U0001f4bb sequence",
    "mixed \u00e9\U0001f600 text",
    EOT,
    f"The end.{EOT}Once upon a time",
    "unusual words like photosynthesis and xylophone",
    "Lily's friend Tom's dog",
]


def train_tokenizer(stories: Iterable[str], vocab_size: int) -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer(add_prefix_space=False)
    tokenizer.train_from_iterator(
        stories,
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOT],
        show_progress=False,
    )
    return tokenizer


def to_engine_json(tokenizer: ByteLevelBPETokenizer) -> dict:
    """Vocabulary, merges in rank order, and special tokens, as the TypeScript engine reads them."""
    data = json.loads(tokenizer.to_str())
    merges = [m.split(" ") if isinstance(m, str) else list(m) for m in data["model"]["merges"]]
    special = {t["content"]: t["id"] for t in data["added_tokens"] if t["special"]}
    return {
        "type": FORMAT,
        "vocab": data["model"]["vocab"],
        "merges": merges,
        "specialTokens": special,
    }


def save_tokenizer(tokenizer: ByteLevelBPETokenizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(to_engine_json(tokenizer), ensure_ascii=False, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def load_tokenizer(path: Path) -> ByteLevelBPETokenizer:
    """Rebuild a tokenizer from the engine JSON, so Python encodes exactly what the site will."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != FORMAT:
        raise ValueError(f"{path}: expected tokenizer type {FORMAT!r}, got {data.get('type')!r}")
    tokenizer = ByteLevelBPETokenizer(
        vocab=data["vocab"],
        merges=[tuple(pair) for pair in data["merges"]],
        add_prefix_space=False,
    )
    tokenizer.add_special_tokens(list(data["specialTokens"]))
    return tokenizer


def tokenizer_cases(tokenizer: ByteLevelBPETokenizer, samples: Iterable[str]) -> list[dict]:
    """Edge cases first, then samples, without duplicates, each with its expected ids."""
    texts = list(dict.fromkeys([*EDGE_CASES, *samples]))
    return [{"text": text, "ids": tokenizer.encode(text).ids} for text in texts]


def synthetic_samples(tokenizer: ByteLevelBPETokenizer, count: int, seed: int) -> list[str]:
    """Seeded strings of 3 to 20 decoded vocabulary tokens, so the test cases hold no dataset text.

    Special tokens are skipped, and so are tokens that hold only part of a UTF-8 character,
    since they decode to U+FFFD.
    """
    special = set(to_engine_json(tokenizer)["specialTokens"].values())
    pieces: list[str] = []
    for token_id in range(tokenizer.get_vocab_size()):
        text = tokenizer.decode([token_id])
        if token_id not in special and "\ufffd" not in text:
            pieces.append(text)
    rng = random.Random(seed)
    return ["".join(rng.choices(pieces, k=rng.randint(3, 20))) for _ in range(count)]
