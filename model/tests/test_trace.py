import pytest
import torch

from tv_model.config import MICRO
from tv_model.gpt import GPT
from tv_model.trace import expected_trace_names, flatten_trace


def micro_trace(token_ids):
    torch.manual_seed(0)
    model = GPT(MICRO).eval()
    trace = {}
    with torch.no_grad():
        model(torch.tensor([token_ids]), trace)
    return trace


def test_expected_names_cover_every_layer_and_head():
    names = expected_trace_names(MICRO)
    assert len(names) == 7 + MICRO.n_layers * (14 + MICRO.n_heads * 7)
    assert "tokenEmbeddings" in names
    assert "lnFinal.variance" in names
    assert "layers.1.heads.1.scaledMasked" in names
    assert "layers.0.mlpAct" in names


def test_flatten_uses_engine_paths_and_drops_the_batch_dimension():
    flat = flatten_trace(micro_trace([3, 1, 4]))
    assert set(flat) == expected_trace_names(MICRO)
    assert flat["tokenEmbeddings"].shape == (3, MICRO.d_model)
    assert flat["layers.0.ln1.mean"].shape == (3,)
    assert flat["layers.1.heads.0.weights"].shape == (3, 3)
    assert flat["layers.0.heads.1.q"].shape == (3, MICRO.d_head)
    assert flat["layers.0.mlpHidden"].shape == (3, MICRO.d_mlp)
    assert flat["logits"].shape == (3, MICRO.vocab_size)
    assert all(t.dtype == torch.float32 and t.is_contiguous() for t in flat.values())


def test_flattened_tensors_do_not_share_memory():
    flat = flatten_trace(micro_trace([3, 1, 4]))
    pointers = [t.untyped_storage().data_ptr() for t in flat.values()]
    assert len(set(pointers)) == len(pointers)


def test_flatten_rejects_a_batch_larger_than_one():
    trace = {"logits": torch.zeros(2, 3, 4)}
    with pytest.raises(ValueError):
        flatten_trace(trace)
