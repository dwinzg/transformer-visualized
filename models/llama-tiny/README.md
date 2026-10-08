# Tiny Llama-style model

A second small model, trained the same way as [`../tiny`](../tiny/README.md), with the four changes most language models since LLaMA have made. It lets the site compare the two designs part by part.

## What changes from the GPT-2 style model

| Part | GPT-2 style (`tiny`) | Llama style (`llama-tiny`) |
| --- | --- | --- |
| Normalization | LayerNorm, with mean, scale and shift | RMSNorm, scale only (Zhang and Sennrich 2019) |
| Position | A learned table of 128 position embeddings | Rotary positions inside attention, base 10,000 (Su et al. 2024) |
| Feed forward | 512 hidden units with GELU | SwiGLU with 344 hidden units, so it costs about the same (Shazeer 2020) |
| Attention | 4 heads, each with its own keys and values | 4 query heads sharing 2 key and value heads (Ainslie et al. 2023) |
| Biases | In every linear layer and norm | None |

Both models tie the output to the token embedding table. Real LLaMA models do not tie it. We kept it so the two tiny models are about the same size.

## Shape

| Setting | Value |
| --- | --- |
| Vocabulary | 4,096 byte-level BPE tokens, the same tokenizer as `tiny` |
| Context | 128 tokens |
| Width | 128 numbers per token |
| Layers | 4 |
| Attention heads | 4 query heads of 32 numbers, 2 key and value heads |
| Feed-forward width | 344 |
| Parameters | 1,250,432 |

## Training

The same data, tokenizer, schedule and seed as `tiny`. 25,000 steps of 64 sequences of 128 tokens (204,800,000 tokens seen), AdamW, learning rate 2e-3 with warmup and cosine decay, on an Apple M4, in about 70 minutes.

Validation loss 1.8281, against 1.8494 for `tiny`, on the same 40 fixed batches. With 6% fewer parameters it predicts slightly better. One run of each is not enough to say how much of that is the design and how much is chance.

The tokenizer file is not copied here. Use `../tiny/tokenizer.json`.

## Samples

Temperature 0.8, top-k 40. Prompts "Once upon a time", "Lily wanted to" and "The little dog", 80 tokens each, with seeds 0, 1 and 2.

```text
Once upon a time, in a small town, there was a little boy named Tim. Tim had a favorite toy, a small piece of paper. He loved to draw and play all day. One day, Tim saw a big ball of paper on the paper. It was very pretty.
Tim wanted to show his mom. He went to her and showed her the paper. "Look, Mom! I made a
```

```text
Lily wanted to make a cake with butter and put them in a oven. She thought of a plan. She decided to use the jar of toast for the cake. She mixed it up and poured it on the table. The cake tasted very good. It tasted funny. Lily loved the cake and did not make any noise.
She put the cake in the oven and waited for the cake to bake. As soon
```

```text
The little dog saw his friend, a little boy named Tim. Tim was anxious because the big cat was in his way. He did not like the dark and dark clouds.
The dark cloud saw Tim and the big cat. The big bird said, "I will help you. We can fly together to the dark cloud." Tim was happy and they started to fly together. They climbed higher and higher.
```

## Reproduce

From `model/`, after preparing the data as in the model README:

```sh
.venv/bin/python -m tv_model.train --preset llama-tiny --data-dir data --out-dir runs/llama-tiny
.venv/bin/python -m tv_model.export --run runs/llama-tiny --out ../models/llama-tiny
.venv/bin/python -m tv_model.fixtures llama-tiny --model ../models/llama-tiny/model.safetensors --tokenizer ../models/tiny/tokenizer.json --out ../packages/engine/test/fixtures/llama-tiny
```

Retraining on MPS is not bit-reproducible, so a new run gives a comparable model, not the same file.

## License

MIT, like the code. The training data is credited in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
