import json

import pytest
import torch
from safetensors import safe_open

from tv_model.config import MICRO
from tv_model.gpt import GPT
from tv_model.io import FORMAT_ID, load_model, save_model


def test_save_then_load_reproduces_every_weight(tmp_path):
    torch.manual_seed(0)
    model = GPT(MICRO).eval()
    path = tmp_path / "model.safetensors"
    save_model(model, path)
    loaded = load_model(path)
    assert loaded.cfg == MICRO
    assert not loaded.training
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], tensor, rtol=0, atol=0)


def test_metadata_holds_format_and_engine_config(tmp_path):
    path = tmp_path / "model.safetensors"
    save_model(GPT(MICRO), path, {"purpose": "test"})
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata()
        assert all(f.get_tensor(name).dtype == torch.float32 for name in f.keys())
    assert metadata["format"] == FORMAT_ID
    assert json.loads(metadata["config"]) == MICRO.to_engine_json()
    assert metadata["purpose"] == "test"


def test_extra_metadata_cannot_override_reserved_keys(tmp_path):
    with pytest.raises(ValueError):
        save_model(GPT(MICRO), tmp_path / "model.safetensors", {"format": "other"})


def test_load_rejects_an_unknown_format(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "other.safetensors"
    save_file({"x": torch.zeros(1)}, str(path), metadata={"format": "something-else"})
    with pytest.raises(ValueError):
        load_model(path)
