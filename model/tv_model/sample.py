"""Generate text from a trained model, for sanity checks and the model card.

python -m tv_model.sample --weights ../models/tiny/model.safetensors \
    --tokenizer ../models/tiny/tokenizer.json --prompt "Once upon a time" \
    --tokens 80 --temperature 0.8 --top-k 40 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import EOT
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
    eot_id = tokenizer.token_to_id(EOT)
    ids = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        if len(ids) >= model.cfg.context_length:
            break
        logits = model(torch.tensor([ids]))[0, -1] / temperature
        if top_k is not None:
            cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
            logits = logits.masked_fill(logits < cutoff, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        next_id = int(torch.multinomial(probs, 1, generator=generator))
        if next_id == eot_id:
            break
        ids.append(next_id)
    return tokenizer.decode(ids, skip_special_tokens=False)


def main(argv: list[str] | None = None) -> None:
    from .io import load_model
    from .tokenizer import load_tokenizer
    from .train import load_checkpoint

    parser = argparse.ArgumentParser(description="Generate text from a trained model.")
    weights_group = parser.add_mutually_exclusive_group(required=True)
    weights_group.add_argument("--checkpoint", type=Path, help="a training checkpoint (ckpt.pt)")
    weights_group.add_argument("--weights", type=Path, help="a shipped model.safetensors file")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    model = (
        load_model(args.weights)
        if args.weights is not None
        else load_checkpoint(args.checkpoint)[0]
    )
    tokenizer = load_tokenizer(args.tokenizer)
    text = generate_text(
        model, tokenizer, args.prompt, args.tokens, args.temperature, args.top_k, args.seed
    )
    print(text)


if __name__ == "__main__":
    main()
