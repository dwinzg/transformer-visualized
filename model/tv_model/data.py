"""Download and clean TinyStories (Eldan and Li, 2023), licensed CDLA-Sharing-1.0."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

REPO_ID = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "valid": "TinyStoriesV2-GPT4-valid.txt"}
EOT = "<|endoftext|>"

# Curly quotes, en and em dashes, the ellipsis and the no-break space, with their ASCII
# equivalents. Stories that still contain non-ASCII characters after this mapping are dropped,
# which keeps a small vocabulary for plain English.
_ASCII_EQUIVALENTS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def normalize(text: str) -> str:
    return text.translate(_ASCII_EQUIVALENTS).strip()


def iter_stories(path: Path) -> Iterator[str]:
    """Stream raw stories from a TinyStories V2 file. A line `<|endoftext|>` ends each story."""
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                story = "".join(lines).strip()
                if story:
                    yield story
                lines = []
            else:
                lines.append(line)
    story = "".join(lines).strip()
    if story:
        yield story


def clean_story(story: str) -> str | None:
    """The normalized story, or None when it is empty or still contains non-ASCII characters."""
    cleaned = normalize(story)
    return cleaned if cleaned and cleaned.isascii() else None


def clean_stories(stories: Iterable[str]) -> Iterator[str]:
    """Normalize typography and keep only stories that are plain ASCII afterwards."""
    for story in stories:
        cleaned = clean_story(story)
        if cleaned is not None:
            yield cleaned


def download(split: str, dest: Path) -> Path:
    """Download one split at the pinned revision into `dest` and return the local path."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=FILES[split],
            revision=REVISION,
            local_dir=dest,
        )
    )
