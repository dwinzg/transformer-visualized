# Tiny TinyStories model

A small GPT-2 style language model that runs in the browser on this site. It is small enough that every number inside it can be shown and inspected.

## Shape

| Setting | Value |
| --- | --- |
| Vocabulary | 4,096 byte-level BPE tokens |
| Context | 128 tokens |
| Width | 128 numbers per token |
| Layers | 4 |
| Attention heads | 4 per layer, 32 numbers each |
| Feed-forward width | 512 |
| Parameters | 1,334,016 |

## Training

- Data. TinyStories V2, the GPT-4 generated split (Eldan and Li 2023), revision `f54c09f`. Curly quotes and dashes were mapped to ASCII, and stories with other non-ASCII characters were dropped (291 of 2,717,495 training stories).
- Tokens. 552,202,462 training tokens, 5,574,483 validation tokens.
- Run. 25,000 steps of 64 sequences of 128 tokens (204,800,000 tokens seen), AdamW, learning rate 2e-3 with warmup and cosine decay, on an Apple M4.
- Result. Validation loss 1.8494 (cross-entropy per token).

## Samples

Temperature 0.8, top-k 40.

```text
Once upon a time, in a small town, there was a girl named Sue. Sue liked to play with her ball. One sunny day, she went outside to play with her ball.
Sue saw a big tree and wanted to play with it. She thought about how she could bring the ball to a fun part of the slide. Sue said, "Mom, can I bring the ball?" Her mom smiled and
```

```text
Lily wanted to make the cake. She found some paper and a pen. She thought of a plan. She made a big cake. She said, "You can make a cake. You can make something with the matches. I can make a cake for you."
Mom and Dad were very happy. They said, "Thank you, Mom and Dad. We love you, Lily. I love you."
```

```text
The little dog saw his friend, a cat named Kitty, playing with a ball. Kitty wanted to play with Bounce, so she said, "Kitty, do you want to play with me?"
Bounce said, "Yes! I want to join you, Kitty. But I can play too fast." So, Kitty and Bounce played and had a lot of fun. They became good
```

## Limitations

- It only knows the simple world of children's stories. It does not know facts about the real world.
- It often loses the thread of a story after a few sentences.
- It sees at most 128 tokens at a time.

## License and data

The weights are released under the MIT License, like the code. TinyStories is licensed under the Community Data License Agreement Sharing 1.0. Section 3.5 of that license states that it "imposes no obligations or restrictions on Your Use or Publication of Results", and the trained weights are such a result. We credit the dataset authors all the same. See `THIRD_PARTY_NOTICES.md`.

## Reproduce

See `model/README.md`.
