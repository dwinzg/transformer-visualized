import pytest

from tv_model.config import MICRO, PRESETS, TINY, ModelConfig, config_from_engine_json


def test_presets_have_the_documented_shapes():
    assert (TINY.vocab_size, TINY.context_length, TINY.d_model) == (4096, 128, 128)
    assert (TINY.n_layers, TINY.n_heads, TINY.d_mlp) == (4, 4, 512)
    assert (MICRO.vocab_size, MICRO.context_length, MICRO.d_model) == (64, 16, 16)
    assert (MICRO.n_layers, MICRO.n_heads, MICRO.d_mlp) == (2, 2, 64)
    assert PRESETS == {"tiny": TINY, "micro": MICRO}


def test_d_head_divides_d_model_across_heads():
    assert TINY.d_head == 32
    assert MICRO.d_head == 8


def test_rejects_d_model_not_divisible_by_heads():
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=8, context_length=4, d_model=10, n_layers=1, n_heads=3, d_mlp=8)


def test_engine_json_uses_engine_keys():
    assert MICRO.to_engine_json() == {
        "vocabSize": 64,
        "contextLength": 16,
        "dModel": 16,
        "nLayers": 2,
        "nHeads": 2,
        "dHead": 8,
        "dMlp": 64,
        "layerNormEps": 1e-5,
    }


def test_engine_json_round_trips():
    assert config_from_engine_json(TINY.to_engine_json()) == TINY


def test_engine_json_with_inconsistent_d_head_is_rejected():
    bad = {**MICRO.to_engine_json(), "dHead": 4}
    with pytest.raises(ValueError):
        config_from_engine_json(bad)
