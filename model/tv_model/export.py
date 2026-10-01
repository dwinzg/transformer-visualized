"""Export the best checkpoint of a training run as the shipped model file.

python -m tv_model.export --run runs/tiny --out ../models/tiny
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import REVISION
from .io import save_model
from .train import load_checkpoint

DATASET = (
    "TinyStories V2 GPT-4 split (Eldan and Li 2023), roneneldan/TinyStories revision "
    f"{REVISION}, license CDLA-Sharing-1.0"
)


def export_run(run_dir: Path, out_dir: Path) -> dict:
    model, info = load_checkpoint(run_dir / "ckpt.pt")
    summary = json.loads((run_dir / "summary.json").read_text())
    training = {**info, "tokens_seen": summary["tokens_seen"], "steps": summary["steps"]}
    metadata = {"training": json.dumps(training), "dataset": DATASET}
    save_model(model, out_dir / "model.safetensors", metadata)
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export a training run for the site.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export_run(args.run, args.out), indent=2))


if __name__ == "__main__":
    main()
