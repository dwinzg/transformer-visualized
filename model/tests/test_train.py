import json
import math

import numpy as np
import torch

from tv_model.config import MICRO
from tv_model.gpt import GPT
from tv_model.train import (
    TrainConfig,
    configure_optimizer,
    get_batch,
    load_checkpoint,
    lr_at,
    train,
)


def test_learning_rate_warms_up_then_decays_to_the_minimum():
    cfg = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=110)
    assert lr_at(0, cfg) == 1e-4
    assert lr_at(9, cfg) == 1e-3
    assert math.isclose(lr_at(10, cfg), 1e-3)
    assert math.isclose(lr_at(110, cfg), 1e-4)
    after_warmup = [lr_at(step, cfg) for step in range(10, 111)]
    assert all(a >= b for a, b in zip(after_warmup, after_warmup[1:], strict=False))


def test_batches_pair_each_token_with_the_next():
    tokens = np.arange(100, dtype=np.uint16)
    x, y = get_batch(tokens, 4, 8, np.random.default_rng(0), torch.device("cpu"))
    assert x.shape == y.shape == (4, 8)
    assert x.dtype == torch.long
    torch.testing.assert_close(y, x + 1)


def test_weight_decay_applies_only_to_matrices():
    optimizer = configure_optimizer(GPT(MICRO), TrainConfig(weight_decay=0.1))
    decayed, plain = optimizer.param_groups
    assert decayed["weight_decay"] == 0.1 and all(p.dim() >= 2 for p in decayed["params"])
    assert plain["weight_decay"] == 0.0 and all(p.dim() < 2 for p in plain["params"])


def test_training_learns_a_predictable_pattern(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pattern = np.tile(np.arange(64, dtype="<u2"), 400)  # the next token is always this one + 1
    pattern.tofile(data / "train.bin")
    pattern.tofile(data / "valid.bin")
    cfg = TrainConfig(
        preset="micro",
        data_dir=str(data),
        out_dir=str(tmp_path / "run"),
        batch_size=16,
        max_steps=300,
        lr=3e-3,
        min_lr=3e-4,
        warmup_steps=20,
        eval_every=100,
        eval_batches=4,
        device="cpu",
    )
    summary = train(cfg)
    records = [
        json.loads(line) for line in (tmp_path / "run" / "log.jsonl").read_text().splitlines()
    ]
    val = [r["val_loss"] for r in records if "val_loss" in r]
    assert val[0] > 0.9 * math.log(64)  # starts near chance
    assert summary["best_val_loss"] < 0.5 * val[0]
    model, info = load_checkpoint(tmp_path / "run" / "ckpt.pt")
    assert model.cfg == MICRO and not model.training
    assert info["val_loss"] == summary["best_val_loss"]
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["steps"] == 300
