# Transformer Visualized

An interactive, visual guide to how large language models work. Type a sentence and follow it through a real transformer, step by step, until it predicts the next word. Every number on screen traces back to the math that produced it.

Built for learners without a technical background, and quick to review for those with one.

> **Status.** Early development. The sections below describe the planned first release.

## What it covers

- **Tokens and embeddings.** How text becomes numbers.
- **Self-attention.** Queries, keys and values, from a simple story up to the full formula.
- **The transformer block.** The residual stream, layer normalization and the feed-forward network.
- **Prediction and generation.** Probabilities, sampling, temperature and the KV cache.
- **Training.** How a model learns its numbers.

## How it works

- **Three zoom levels.** A toy example small enough to compute by hand, a tiny real model where every value is visible, and a GPT-2 class model for realistic behavior.
- **Depth on demand.** Each section reads as a story, as worked numbers, as the formula, or as code.
- **Grounded in the literature.** Claims link to the exact section, equation or figure of the paper they come from. Simplifications are labeled.
- **Runs in the browser.** No server, no account and no tracking. Installable and usable offline.

## Contributing

Contributions are welcome, especially from people learning the material for the first time.

- **Report an error.** Open an issue naming the section and what looks wrong. Mistakes in math or explanations are the highest priority.
- **Flag something confusing.** Point to the step where you got lost.
- **Suggest improvements.** Better examples, visuals, exercises or references.
- **Contribute code or content.** See [CONTRIBUTING.md](CONTRIBUTING.md) for the workflow, writing standards and citation rules.

## License

- Code is released under the [MIT License](LICENSE).
- Written lessons and figures are released under [CC BY 4.0](LICENSE-CONTENT).
- Third-party material keeps its original license.
