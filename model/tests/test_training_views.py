from pathlib import Path

import pytest

from tv_model.io import load_model, save_model
from tv_model.training_views import build

TINY = Path(__file__).resolve().parents[2] / "models" / "tiny"


def test_a_snapshot_becomes_the_numbers_the_playground_shows(tmp_path):
    save_model(
        load_model(TINY / "model.safetensors"),
        tmp_path / "step-25000.safetensors",
        {"step": "25000", "val_loss": "1.85"},
    )
    data = build(tmp_path, TINY / "tokenizer.json")
    [snap] = data["snapshots"]
    assert snap["step"] == 25000 and snap["valLoss"] == 1.85
    # The shipped model's best guess after the chapters' sentence, and its sample story.
    assert snap["next"][0]["text"] == " ball"
    ps = [g["p"] for g in snap["next"]]
    assert ps == sorted(ps, reverse=True)
    assert snap["story"].startswith(", in a small town, there was a girl named Sue.")
    assert 0.8 < snap["recall"] < 0.9
    assert 0.1 < snap["copy"] < 0.2
    assert data["inductionChance"] == pytest.approx(0.017, abs=1e-4)
    assert len(snap["induction"]) == 4 and all(len(row) == 4 for row in snap["induction"])
