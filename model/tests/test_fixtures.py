import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open

from tv_model.config import MICRO
from tv_model.fixtures import MICRO_CASES, MICRO_SEED, make_fixture_model, write_micro_fixtures
from tv_model.gpt import GPT
from tv_model.io import load_model
from tv_model.trace import expected_trace_names

# The engine's committed golden fixtures, addressed relative to this file rather than the
# working directory so the test runs the same way regardless of where pytest is invoked from.
COMMITTED_FIXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "packages" / "engine" / "test" / "fixtures" / "micro"
)


def read_tensors(path):
    with safe_open(str(path), framework="pt") as f:
        return {name: f.get_tensor(name) for name in f.keys()}


def _lookup_trace_path(trace, path):
    """Walk a nested Trace dict/list by its dotted engine path, e.g. `layers.0.heads.1.q`."""
    node = trace
    for part in path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


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


def test_generated_files_are_byte_identical_across_processes(tmp_path):
    """safetensors' `__metadata__` key order is randomized per process (stable within one
    process), so a same-process double write can't see it: this spawns two real processes.
    """
    model_dir = Path(__file__).resolve().parents[1]
    dest_a, dest_b = tmp_path / "a", tmp_path / "b"
    for dest in (dest_a, dest_b):
        subprocess.run(
            [sys.executable, "-m", "tv_model.fixtures", "micro", "--out", str(dest)],
            cwd=model_dir,
            check=True,
        )
    names = {p.name for p in dest_a.iterdir()}
    assert names == {p.name for p in dest_b.iterdir()}
    for name in names:
        assert (dest_a / name).read_bytes() == (dest_b / name).read_bytes(), name


def test_committed_fixtures_match_a_fresh_trace_of_the_committed_model():
    """Every other fixture test writes to `tmp_path`; nothing else reads the committed
    `packages/engine/test/fixtures/micro/` files the engine actually tests against. This loads
    them, reruns each case, and checks every committed tensor against the value found by
    walking a fresh trace dict at the same path -- never through `flatten_trace`, so a bug that
    renamed or reordered paths there could not hide from both the writer and this check.
    """
    index = json.loads((COMMITTED_FIXTURES_DIR / "cases.json").read_text())
    model = load_model(COMMITTED_FIXTURES_DIR / index["model"])
    names = expected_trace_names(model.cfg)
    for case in index["cases"]:
        trace = {}
        with torch.no_grad():
            model(torch.tensor([case["tokenIds"]]), trace)
        committed = read_tensors(COMMITTED_FIXTURES_DIR / case["file"])
        assert set(committed) == names
        for path, tensor in committed.items():
            expected = _lookup_trace_path(trace, path)[0]
            nonfinite = ~torch.isfinite(expected)
            assert torch.equal(tensor[nonfinite], expected[nonfinite]), path
            torch.testing.assert_close(
                tensor[~nonfinite], expected[~nonfinite], rtol=1e-5, atol=1e-6, msg=path
            )


def test_make_fixture_model_advances_rather_than_resets_the_callers_global_rng():
    """`GPT(cfg)`'s own default init unavoidably draws from the global RNG; that's ordinary.
    The bug was that `make_fixture_model` used to also call `torch.manual_seed(seed)`, which
    forces the caller's subsequent draws to a fixed sequence no matter what its RNG state was
    beforehand. After the fix, two different starting states must still lead to different
    draws afterward -- the function may advance the caller's RNG, but must not reset it.
    """
    torch.manual_seed(1)
    make_fixture_model(MICRO, MICRO_SEED)
    after_seed_1 = torch.randn(5)

    torch.manual_seed(2)
    make_fixture_model(MICRO, MICRO_SEED)
    after_seed_2 = torch.randn(5)

    assert not torch.equal(after_seed_1, after_seed_2)


def test_model_fixtures_record_text_ids_and_traces(tmp_path):
    from tv_model.fixtures import write_model_fixtures
    from tv_model.io import save_model
    from tv_model.tokenizer import save_tokenizer, train_tokenizer

    tokenizer = train_tokenizer(["Lily ran home.", "Tom ate a red apple."] * 30, vocab_size=280)
    save_tokenizer(tokenizer, tmp_path / "tokenizer.json")
    cfg = dataclasses.replace(MICRO, vocab_size=tokenizer.get_vocab_size())
    torch.manual_seed(0)
    save_model(GPT(cfg), tmp_path / "model.safetensors")
    write_model_fixtures(
        tmp_path / "model.safetensors",
        tmp_path / "tokenizer.json",
        {"a": "Lily ran"},
        tmp_path / "out",
    )
    index = json.loads((tmp_path / "out" / "cases.json").read_text())
    assert index["model"] == "../model.safetensors"
    case = index["cases"][0]
    assert case["text"] == "Lily ran"
    assert case["tokenIds"] == tokenizer.encode("Lily ran").ids
    tensors = read_tensors(tmp_path / "out" / case["file"])
    assert set(tensors) == expected_trace_names(cfg)
