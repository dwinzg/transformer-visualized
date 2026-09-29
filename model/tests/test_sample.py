import dataclasses

import torch

from tv_model.config import MICRO
from tv_model.gpt import GPT
from tv_model.sample import generate_text
from tv_model.tokenizer import train_tokenizer

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
