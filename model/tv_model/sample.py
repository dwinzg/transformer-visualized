"""Generate text from a trained model, for sanity checks and the model card."""

from __future__ import annotations

import torch

from .gpt import GPT


@torch.no_grad()
def generate_text(
    model: GPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> str:
    generator = torch.Generator().manual_seed(seed)
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        if len(ids) >= model.cfg.context_length:
            break
        logits = model(torch.tensor([ids]))[0, -1] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        ids.append(int(torch.multinomial(probs, 1, generator=generator)))
    return tokenizer.decode(ids, skip_special_tokens=False)
