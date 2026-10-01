"""Write tokenized stories as little-endian uint16 files for training."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .data import EOT


def _write_batch(tokenizer, batch: list[str], eot_id: int, f: BinaryIO) -> int:
    ids: list[int] = []
    for encoding in tokenizer.encode_batch(batch):
        ids.extend(encoding.ids)
        ids.append(eot_id)
    np.asarray(ids, dtype="<u2").tofile(f)
    return len(ids)


def encode_to_file(tokenizer, stories: Iterable[str], path: Path, batch_size: int = 2048) -> int:
    """Encode stories, each followed by the end-of-text id. Returns the number of tokens."""
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("uint16 token files need a vocabulary of at most 65536 tokens")
    eot_id = tokenizer.token_to_id(EOT)
    if eot_id is None:
        raise ValueError(f"the tokenizer has no {EOT} token")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("wb") as f:
        batch: list[str] = []
        for story in stories:
            batch.append(story)
            if len(batch) == batch_size:
                count += _write_batch(tokenizer, batch, eot_id, f)
                batch = []
        if batch:
            count += _write_batch(tokenizer, batch, eot_id, f)
    return count


def read_tokens(path: Path) -> np.memmap:
    return np.memmap(path, dtype="<u2", mode="r")
