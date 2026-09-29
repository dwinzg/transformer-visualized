import dataclasses
from types import SimpleNamespace

import torch

from tv_model.config import MICRO
from tv_model.gpt import GPT
from tv_model.io import load_model, save_model
from tv_model.sample import generate_text, main
from tv_model.tokenizer import save_tokenizer, train_tokenizer

STORIES = ["Lily ran home.", "Tom ate a red apple.", "The dog was happy."] * 20


def test_generation_is_reproducible_and_continues_the_prompt():
    tokenizer = train_tokenizer(STORIES, vocab_size=280)
    cfg = dataclasses.replace(MICRO, vocab_size=tokenizer.get_vocab_size())
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    a = generate_text(model, tokenizer, "Lily", max_new_tokens=5, temperature=1.0, top_k=10, seed=3)
    b = generate_text(model, tokenizer, "Lily", max_new_tokens=5, temperature=1.0, top_k=10, seed=3)
    assert a == b
    assert a.startswith("Lily")
    assert len(a) > len("Lily")


def test_generation_stops_at_the_end_of_text_token_and_excludes_it():
    eot_id = 4

    class StubTokenizer:
        def token_to_id(self, token):
            return eot_id

        def encode(self, text):
            return SimpleNamespace(ids=[0, 1])

        def decode(self, ids, skip_special_tokens=False):
            return ",".join(str(i) for i in ids)

    class StubModel:
        def __init__(self):
            self.cfg = SimpleNamespace(context_length=16)

        def __call__(self, ids):
            # Always makes the end-of-text id the argmax, so greedy decoding (top_k=1)
            # samples it deterministically on the very first generated step.
            logits = torch.zeros(1, ids.shape[1], 8)
            logits[..., eot_id] = 100.0
            return logits

    text = generate_text(
        StubModel(), StubTokenizer(), "x", max_new_tokens=5, temperature=1.0, top_k=1, seed=0
    )
    assert text == "0,1"
    assert str(eot_id) not in text.split(",")


def test_command_line_generates_from_shipped_weights(tmp_path, capsys):
    tokenizer = train_tokenizer(STORIES, vocab_size=280)
    tokenizer_path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, tokenizer_path)
    cfg = dataclasses.replace(MICRO, vocab_size=tokenizer.get_vocab_size())
    torch.manual_seed(0)
    weights_path = tmp_path / "model.safetensors"
    save_model(GPT(cfg).eval(), weights_path)

    main(
        [
            "--weights",
            str(weights_path),
            "--tokenizer",
            str(tokenizer_path),
            "--prompt",
            "Lily",
            "--tokens",
            "5",
            "--temperature",
            "1.0",
            "--top-k",
            "10",
            "--seed",
            "3",
        ]
    )
    printed = capsys.readouterr().out.strip()
    expected = generate_text(
        load_model(weights_path),
        tokenizer,
        "Lily",
        max_new_tokens=5,
        temperature=1.0,
        top_k=10,
        seed=3,
    )
    assert printed == expected
    assert printed.startswith("Lily")
