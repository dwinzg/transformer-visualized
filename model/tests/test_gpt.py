import math

import pytest
import torch

from tv_model.config import MICRO, TINY
from tv_model.gpt import GPT

PER_LAYER_NAMES = [
    "ln_1.weight",
    "ln_1.bias",
    "attn.c_attn.weight",
    "attn.c_attn.bias",
    "attn.c_proj.weight",
    "attn.c_proj.bias",
    "ln_2.weight",
    "ln_2.bias",
    "mlp.c_fc.weight",
    "mlp.c_fc.bias",
    "mlp.c_proj.weight",
    "mlp.c_proj.bias",
]


def traced(model, token_ids):
    trace = {}
    with torch.no_grad():
        logits = model(torch.tensor([token_ids]), trace)
    return logits, trace


@pytest.fixture
def micro():
    torch.manual_seed(0)
    return GPT(MICRO).eval()


def test_state_dict_names_match_the_engine_contract():
    expected = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
    for i in range(MICRO.n_layers):
        expected.update(f"h.{i}.{name}" for name in PER_LAYER_NAMES)
    assert set(GPT(MICRO).state_dict()) == expected


def test_state_dict_shapes_use_linear_layout():
    sd = GPT(MICRO).state_dict()
    d, m = MICRO.d_model, MICRO.d_mlp
    assert sd["wte.weight"].shape == (MICRO.vocab_size, d)
    assert sd["wpe.weight"].shape == (MICRO.context_length, d)
    assert sd["h.0.attn.c_attn.weight"].shape == (3 * d, d)
    assert sd["h.0.attn.c_attn.bias"].shape == (3 * d,)
    assert sd["h.0.attn.c_proj.weight"].shape == (d, d)
    assert sd["h.0.mlp.c_fc.weight"].shape == (m, d)
    assert sd["h.0.mlp.c_proj.weight"].shape == (d, m)
    assert sd["ln_f.weight"].shape == (d,)


def test_tiny_parameter_count():
    assert sum(p.numel() for p in GPT(TINY).parameters()) == 1_334_016


def test_logits_have_one_row_per_position(micro):
    assert micro(torch.tensor([[1, 2, 3]])).shape == (1, 3, MICRO.vocab_size)


def test_rejects_sequences_longer_than_the_context(micro):
    with pytest.raises(ValueError):
        micro(torch.zeros((1, MICRO.context_length + 1), dtype=torch.long))


def test_attention_is_causal(micro):
    with torch.no_grad():
        a = micro(torch.tensor([[1, 2, 3, 4]]))
        b = micro(torch.tensor([[1, 2, 3, 9]]))
    torch.testing.assert_close(a[:, :3], b[:, :3])
    assert not torch.allclose(a[:, 3], b[:, 3])


def test_output_head_is_tied_to_the_token_embedding(micro):
    logits, trace = traced(micro, [4, 5])
    torch.testing.assert_close(logits, trace["lnFinal"]["out"] @ micro.wte.weight.T)


def test_gelu_uses_the_tanh_approximation(micro):
    assert micro.h[0].mlp.gelu.approximate == "tanh"


def test_trace_records_every_intermediate(micro):
    logits, trace = traced(micro, [3, 1, 4, 1, 5])
    assert set(trace) == {
        "tokenEmbeddings",
        "positionEmbeddings",
        "embeddings",
        "layers",
        "lnFinal",
        "logits",
    }
    assert len(trace["layers"]) == MICRO.n_layers
    layer = trace["layers"][0]
    assert set(layer) == {
        "input",
        "ln1",
        "heads",
        "attnConcat",
        "attnOut",
        "residAfterAttn",
        "ln2",
        "mlpHidden",
        "mlpAct",
        "mlpOut",
        "output",
    }
    assert len(layer["heads"]) == MICRO.n_heads
    assert set(layer["heads"][0]) == {"q", "k", "v", "scores", "scaledMasked", "weights", "out"}
    assert set(layer["ln1"]) == {"mean", "variance", "out"}
    torch.testing.assert_close(trace["logits"], logits)


def test_trace_values_obey_the_block_equations(micro):
    _, trace = traced(micro, [3, 1, 4, 1, 5])
    first, second = trace["layers"]
    torch.testing.assert_close(
        trace["embeddings"], trace["tokenEmbeddings"] + trace["positionEmbeddings"]
    )
    torch.testing.assert_close(first["input"], trace["embeddings"])
    torch.testing.assert_close(first["residAfterAttn"], first["input"] + first["attnOut"])
    torch.testing.assert_close(first["output"], first["residAfterAttn"] + first["mlpOut"])
    torch.testing.assert_close(second["input"], first["output"])
    torch.testing.assert_close(
        first["mlpAct"], torch.nn.functional.gelu(first["mlpHidden"], approximate="tanh")
    )
    heads_out = torch.cat([h["out"] for h in first["heads"]], dim=-1)
    torch.testing.assert_close(first["attnConcat"], heads_out)


def test_layer_norm_trace_matches_its_definition(micro):
    _, trace = traced(micro, [3, 1, 4, 1, 5])
    layer = trace["layers"][0]
    torch.testing.assert_close(layer["ln1"]["mean"], layer["input"].mean(-1))
    torch.testing.assert_close(layer["ln1"]["variance"], layer["input"].var(-1, unbiased=False))


def test_attention_trace_is_a_scaled_masked_softmax(micro):
    _, trace = traced(micro, [3, 1, 4, 1, 5])
    head = trace["layers"][0]["heads"][1]
    scores, scaled, weights = head["scores"][0], head["scaledMasked"][0], head["weights"][0]
    future = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)
    assert torch.all(torch.isneginf(scaled[future]))
    torch.testing.assert_close(scaled[~future], scores[~future] / math.sqrt(MICRO.d_head))
    assert torch.all(weights[future] == 0)
    torch.testing.assert_close(weights.sum(-1), torch.ones(5))
    torch.testing.assert_close(head["out"][0], weights @ head["v"][0])
