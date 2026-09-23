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


def reference_forward(model, idx):
    """Every traced value recomputed in float64 from the weights, independent of gpt.py."""
    cfg = model.cfg
    sd = {k: v.detach().to(torch.float64) for k, v in model.state_dict().items()}
    batch, seq = idx.shape
    d, n_heads, dh = cfg.d_model, cfg.n_heads, cfg.d_head

    def layer_norm(x, prefix):
        mean = x.mean(-1, keepdim=True)
        var = ((x - mean) ** 2).mean(-1, keepdim=True)
        out = (x - mean) / torch.sqrt(var + cfg.layer_norm_eps)
        out = out * sd[f"{prefix}.weight"] + sd[f"{prefix}.bias"]
        return {"mean": mean.squeeze(-1), "variance": var.squeeze(-1), "out": out}

    def gelu_tanh(x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    tok = sd["wte.weight"][idx]
    pos = sd["wpe.weight"][:seq].unsqueeze(0).expand(batch, seq, d)
    x = tok + pos
    expected = {"tokenEmbeddings": tok, "positionEmbeddings": pos, "embeddings": x, "layers": []}
    future = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
    for i in range(cfg.n_layers):
        p = f"h.{i}"
        ln1 = layer_norm(x, f"{p}.ln_1")
        qkv = ln1["out"] @ sd[f"{p}.attn.c_attn.weight"].T + sd[f"{p}.attn.c_attn.bias"]
        q_all, k_all, v_all = qkv[..., :d], qkv[..., d : 2 * d], qkv[..., 2 * d :]
        heads = []
        for h in range(n_heads):
            cols = slice(h * dh, (h + 1) * dh)
            q, k, v = q_all[..., cols], k_all[..., cols], v_all[..., cols]
            scores = q @ k.transpose(-1, -2)
            scaled = (scores / math.sqrt(dh)).masked_fill(future, float("-inf"))
            weights = torch.softmax(scaled, dim=-1)
            heads.append(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "scores": scores,
                    "scaledMasked": scaled,
                    "weights": weights,
                    "out": weights @ v,
                }
            )
        concat = torch.cat([head["out"] for head in heads], dim=-1)
        attn_out = concat @ sd[f"{p}.attn.c_proj.weight"].T + sd[f"{p}.attn.c_proj.bias"]
        resid = x + attn_out
        ln2 = layer_norm(resid, f"{p}.ln_2")
        hidden = ln2["out"] @ sd[f"{p}.mlp.c_fc.weight"].T + sd[f"{p}.mlp.c_fc.bias"]
        act = gelu_tanh(hidden)
        mlp_out = act @ sd[f"{p}.mlp.c_proj.weight"].T + sd[f"{p}.mlp.c_proj.bias"]
        out = resid + mlp_out
        expected["layers"].append(
            {
                "input": x,
                "ln1": ln1,
                "heads": heads,
                "attnConcat": concat,
                "attnOut": attn_out,
                "residAfterAttn": resid,
                "ln2": ln2,
                "mlpHidden": hidden,
                "mlpAct": act,
                "mlpOut": mlp_out,
                "output": out,
            }
        )
        x = out
    ln_f = layer_norm(x, "ln_f")
    expected["lnFinal"] = ln_f
    expected["logits"] = ln_f["out"] @ sd["wte.weight"].T
    return expected


def assert_trace_close(actual, expected, path="trace"):
    if isinstance(expected, torch.Tensor):
        assert actual.shape == expected.shape, path
        torch.testing.assert_close(
            actual.to(torch.float64), expected, rtol=1e-4, atol=1e-4, msg=path
        )
    elif isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            assert_trace_close(actual[key], expected[key], f"{path}.{key}")
    else:
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_trace_close(a, e, f"{path}.{i}")


@pytest.mark.parametrize("seed", [0, 1])
def test_trace_matches_an_independent_recomputation(seed):
    torch.manual_seed(seed)
    model = GPT(MICRO)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(0.5 * torch.randn_like(param))
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, MICRO.vocab_size, (2, 7), generator=generator)
    trace = {}
    with torch.no_grad():
        model(idx, trace)
    assert_trace_close(trace, reference_forward(model, idx))


def test_mlp_applies_tanh_gelu_between_its_layers():
    torch.manual_seed(0)
    mlp = GPT(MICRO).h[0].mlp
    with torch.no_grad():
        mlp.c_fc.weight.copy_(3 * torch.randn_like(mlp.c_fc.weight))
    trace = {}
    with torch.no_grad():
        mlp(torch.randn(1, 3, MICRO.d_model), trace)
    h = trace["mlpHidden"]
    tanh_formula = 0.5 * h * (1 + torch.tanh(math.sqrt(2 / math.pi) * (h + 0.044715 * h**3)))
    torch.testing.assert_close(trace["mlpAct"], tanh_formula, rtol=1e-5, atol=1e-6)
    exact = torch.nn.functional.gelu(h)
    assert (trace["mlpAct"] - exact).abs().max() > 1e-5  # distinguishable from exact GELU
