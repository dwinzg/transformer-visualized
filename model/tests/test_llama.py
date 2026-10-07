import math

import pytest
import torch
import torch.nn.functional as F

from tv_model.config import LLAMA_MICRO, LLAMA_TINY, TINY, ModelConfig, config_from_engine_json
from tv_model.io import load_model, save_model
from tv_model.llama import Llama, RMSNorm, build_model, rope, rope_angles


def test_rmsnorm_divides_by_the_root_mean_square():
    norm = RMSNorm(4, eps=0.0)
    x = torch.tensor([[1.0, -1.0, 3.0, -3.0]])
    rms = math.sqrt((1 + 1 + 9 + 9) / 4)
    assert torch.allclose(norm(x), x / rms)


def test_rope_turns_pairs_and_keeps_their_length():
    x = torch.randn(1, 1, 5, 8)
    turned = rope(x, rope_angles(5, 8, 10_000.0))
    # Position 0 is not turned at all.
    assert torch.allclose(turned[..., 0, :], x[..., 0, :])
    pairs = lambda t: t.view(*t.shape[:-1], -1, 2).norm(dim=-1)  # noqa: E731
    assert torch.allclose(pairs(turned), pairs(x), atol=1e-6)


def test_rope_scores_depend_only_on_the_distance_between_tokens():
    q, k = torch.randn(8), torch.randn(8)
    angles = rope_angles(10, 8, 10_000.0)
    at = lambda i, j: float(rope(q, angles[i]) @ rope(k, angles[j]))  # noqa: E731
    assert at(5, 2) == pytest.approx(at(7, 4), abs=1e-5)


def test_grouped_heads_share_keys_and_values():
    torch.manual_seed(0)
    model = Llama(LLAMA_MICRO).eval()
    trace: dict = {}
    model(torch.tensor([[1, 2, 3, 4]]), trace)
    heads = trace["layers"][0]["heads"]
    # Four query heads, two key and value heads: heads 0 and 1 share, 2 and 3 share.
    assert torch.equal(heads[0]["k"], heads[1]["k"]) and torch.equal(heads[2]["v"], heads[3]["v"])
    assert not torch.equal(heads[0]["k"], heads[2]["k"])
    assert not torch.equal(heads[0]["q"], heads[1]["q"])


def test_the_trace_matches_the_logits():
    torch.manual_seed(0)
    model = Llama(LLAMA_MICRO).eval()
    trace: dict = {}
    logits = model(torch.tensor([[5, 9, 2]]), trace)
    assert torch.equal(trace["logits"], logits)
    weights = trace["layers"][1]["heads"][2]["weights"][0]
    assert torch.allclose(weights.sum(-1), torch.ones(3))
    assert float(weights[0, 1]) == 0.0  # the causal mask
    hidden = F.silu(trace["layers"][0]["mlpGate"]) * trace["layers"][0]["mlpUp"]
    assert torch.allclose(hidden, trace["layers"][0]["mlpAct"])


def test_the_tiny_llama_is_about_the_size_of_the_tiny_gpt2():
    def count(cfg: ModelConfig) -> int:
        return sum(p.numel() for p in build_model(cfg).parameters())

    assert abs(count(LLAMA_TINY) - count(TINY)) / count(TINY) < 0.1


def test_a_llama_file_round_trips_with_its_own_format(tmp_path):
    torch.manual_seed(0)
    model = Llama(LLAMA_MICRO).eval()
    path = tmp_path / "m.safetensors"
    save_model(model, path)
    loaded = load_model(path)
    assert isinstance(loaded, Llama) and loaded.cfg == LLAMA_MICRO
    ids = torch.tensor([[1, 2, 3]])
    assert torch.equal(loaded(ids), model(ids))
    assert config_from_engine_json(LLAMA_MICRO.to_engine_json()) == LLAMA_MICRO


def test_gpt2_configs_keep_their_old_keys():
    assert "arch" not in TINY.to_engine_json()


def test_bad_llama_configs_are_refused():
    with pytest.raises(ValueError):
        ModelConfig(64, 16, 16, 2, 4, 48, arch="llama", n_kv_heads=3)


def test_a_llama_trace_has_exactly_the_expected_names():
    from tv_model.trace import expected_trace_names, flatten_trace

    torch.manual_seed(0)
    trace: dict = {}
    Llama(LLAMA_MICRO).eval()(torch.tensor([[1, 2, 3]]), trace)
    assert set(flatten_trace(trace)) == expected_trace_names(LLAMA_MICRO)
