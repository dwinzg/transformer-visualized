"""Train the GPT-2 style model on tokenized TinyStories.

python -m tv_model.train --preset tiny --data-dir data --out-dir runs/tiny
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import PRESETS, config_from_engine_json
from .encode import read_tokens
from .gpt import GPT


@dataclass
class TrainConfig:
    preset: str = "tiny"
    data_dir: str = "data"
    out_dir: str = "runs/tiny"
    batch_size: int = 64
    max_steps: int = 25_000
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup_steps: int = 1_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 500
    eval_batches: int = 40
    seed: int = 1337
    device: str = "auto"


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to lr, then cosine decay to min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    context: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = rng.integers(0, len(tokens) - context - 1, size=batch_size)
    x = np.stack([tokens[s : s + context] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1 : s + 1 + context] for s in starts]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def configure_optimizer(model: GPT, cfg: TrainConfig) -> torch.optim.AdamW:
    """Weight decay on matrices only, not on biases, LayerNorm parameters or other vectors."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    plain = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def _loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def _estimate_loss(
    model: GPT, tokens: np.ndarray, cfg: TrainConfig, context: int, device: torch.device
) -> float:
    model.eval()
    rng = np.random.default_rng(cfg.seed + 1)  # the same batches at every evaluation
    losses = [
        _loss(model, *get_batch(tokens, cfg.batch_size, context, rng, device)).item()
        for _ in range(cfg.eval_batches)
    ]
    model.train()
    return float(np.mean(losses))


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    model_cfg = PRESETS[cfg.preset]
    device = pick_device(cfg.device)
    data_dir, out_dir = Path(cfg.data_dir), Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_tokens = read_tokens(data_dir / "train.bin")
    valid_tokens = read_tokens(data_dir / "valid.bin")
    model = GPT(model_cfg).to(device)
    optimizer = configure_optimizer(model, cfg)
    rng = np.random.default_rng(cfg.seed)
    best_val = float("inf")
    start = time.time()
    with (out_dir / "log.jsonl").open("w") as log:

        def write(record: dict) -> None:
            log.write(json.dumps(record) + "\n")
            log.flush()

        for step in range(cfg.max_steps + 1):
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                val_loss = _estimate_loss(
                    model, valid_tokens, cfg, model_cfg.context_length, device
                )
                write(
                    {"step": step, "val_loss": val_loss, "elapsed_s": round(time.time() - start, 1)}
                )
                print(f"step {step} val_loss {val_loss:.4f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    checkpoint = {
                        "model": state,
                        "config": model_cfg.to_engine_json(),
                        "step": step,
                        "val_loss": val_loss,
                    }
                    torch.save(checkpoint, out_dir / "ckpt.pt")
            if step == cfg.max_steps:
                break
            lr = lr_at(step, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_tokens, cfg.batch_size, model_cfg.context_length, rng, device)
            loss = _loss(model, x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if step % 100 == 0:
                write({"step": step, "train_loss": loss.item(), "lr": lr})
    summary = {
        "best_val_loss": best_val,
        "steps": cfg.max_steps,
        "tokens_seen": cfg.max_steps * cfg.batch_size * model_cfg.context_length,
        "elapsed_s": round(time.time() - start, 1),
        "device": str(device),
        "train_config": asdict(cfg),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def load_checkpoint(path: Path) -> tuple[GPT, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(config_from_engine_json(checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.eval(), {"step": checkpoint["step"], "val_loss": checkpoint["val_loss"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the model on tokenized TinyStories.")
    for field in fields(TrainConfig):
        flag = f"--{field.name.replace('_', '-')}"
        parser.add_argument(flag, type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    print(json.dumps(train(TrainConfig(**vars(args))), indent=2))


if __name__ == "__main__":
    main()
