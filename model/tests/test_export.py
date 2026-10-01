import json

import torch

from tv_model.config import MICRO
from tv_model.export import export_run
from tv_model.gpt import GPT
from tv_model.io import load_model


def test_export_writes_weights_with_training_metadata(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    torch.manual_seed(0)
    model = GPT(MICRO)
    checkpoint = {
        "model": {k: v.detach() for k, v in model.state_dict().items()},
        "config": MICRO.to_engine_json(),
        "step": 7,
        "val_loss": 1.5,
    }
    torch.save(checkpoint, run / "ckpt.pt")
    (run / "summary.json").write_text(json.dumps({"tokens_seen": 1000, "steps": 10}))
    metadata = export_run(run, tmp_path / "out")
    exported = load_model(tmp_path / "out" / "model.safetensors")
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        torch.testing.assert_close(exported(ids), model.eval()(ids), rtol=0, atol=0)
    training = json.loads(metadata["training"])
    assert training == {"step": 7, "val_loss": 1.5, "tokens_seen": 1000, "steps": 10}
    assert "TinyStories" in metadata["dataset"]
