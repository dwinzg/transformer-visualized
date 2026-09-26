import json

import torch
from safetensors import safe_open

from tv_model.config import MICRO
from tv_model.fixtures import MICRO_CASES, write_micro_fixtures
from tv_model.io import load_model
from tv_model.trace import expected_trace_names


def read_tensors(path):
    with safe_open(str(path), framework="pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}


def test_cases_are_the_documented_sequences():
    assert MICRO_CASES["single"] == [7]
    assert MICRO_CASES["five"] == [1, 2, 3, 4, 5]
    assert len(MICRO_CASES["full"]) == MICRO.context_length
    assert all(0 <= t < MICRO.vocab_size for ids in MICRO_CASES.values() for t in ids)


def test_writes_the_documented_layout(tmp_path):
    write_micro_fixtures(tmp_path)
    index = json.loads((tmp_path / "cases.json").read_text())
    assert index["model"] == "model.safetensors"
    assert [c["name"] for c in index["cases"]] == ["single", "five", "full"]
    for case in index["cases"]:
        assert case["tokenIds"] == MICRO_CASES[case["name"]]
        assert case["file"] == f"trace-{case['name']}.safetensors"
        tensors = read_tensors(tmp_path / case["file"])
        assert set(tensors) == expected_trace_names(MICRO)
        seq = len(case["tokenIds"])
        assert tensors["layers.0.heads.0.weights"].shape == (seq, seq)
        assert tensors["logits"].shape == (seq, MICRO.vocab_size)


def test_saved_traces_match_the_saved_model(tmp_path):
    write_micro_fixtures(tmp_path)
    model = load_model(tmp_path / "model.safetensors")
    tensors = read_tensors(tmp_path / "trace-full.safetensors")
    with torch.no_grad():
        logits = model(torch.tensor([MICRO_CASES["full"]]))
    torch.testing.assert_close(tensors["logits"], logits[0], rtol=0, atol=0)


def test_masked_scores_are_negative_infinity_above_the_diagonal(tmp_path):
    write_micro_fixtures(tmp_path)
    scaled = read_tensors(tmp_path / "trace-five.safetensors")["layers.1.heads.1.scaledMasked"]
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.all(torch.isneginf(scaled[future]))
    assert torch.all(torch.isfinite(scaled[~future]))


def test_fixture_weights_exercise_the_nonlinearities(tmp_path):
    write_micro_fixtures(tmp_path)
    tensors = read_tensors(tmp_path / "trace-full.safetensors")
    last_row = tensors["layers.0.heads.0.weights"][-1]
    assert last_row.max() > 2.0 / MICRO.context_length  # attention is far from uniform
    assert tensors["layers.0.mlpHidden"].abs().max() > 1.0  # GELU leaves its linear region


def test_generation_is_deterministic(tmp_path):
    write_micro_fixtures(tmp_path / "a")
    write_micro_fixtures(tmp_path / "b")
    for name in ["model.safetensors", "trace-full.safetensors"]:
        a, b = read_tensors(tmp_path / "a" / name), read_tensors(tmp_path / "b" / name)
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
