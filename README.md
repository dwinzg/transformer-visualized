# Transformer Visualized

An interactive, visual guide to how transformer language models work. Type a sentence and follow it through a real transformer, step by step, until it predicts what comes next. Every number on screen traces back to the math that produced it.

Built for learners without a technical background, and quick to review for those with one.

> **Status.** Early development. The sections below describe the planned first release.

## Screenshots

![The home page, with two ways to start: Learn from scratch or Quick review](docs/images/home-desktop-light.png)

![A chapter page in the Story view, with the depth dial set to Story](docs/images/chapter-story-desktop-light.png)

![The same chapter in dark mode, switched to the Formula view](docs/images/chapter-formula-desktop-dark.png)

<img src="docs/images/chapter-story-phone-light.png" alt="A chapter page on a phone" width="300">

## What it covers

Four chapters take you from text to the model's prediction.

- **Tokens.** How text is split into tokens, small pieces from a fixed vocabulary, each with an ID number.
- **Embeddings and position.** How each token becomes a list of numbers, called an embedding, and how information about its position is added.
- **Attention.** Queries, keys and values, from a simple story up to the full formula.
- **Prediction.** How the model gives every token in its vocabulary a probability of coming next, and how one is picked. Temperature, top-k and top-p, sampling versus greedy choice, and the generation loop. It also briefly covers what happens between attention and the output (the MLP, the residual stream and stacked layers) and the KV cache.

**Later.** A full chapter on the transformer block, how training works, and a larger GPT-2 class model.

## How it works

- **Playground.** Type a short sentence and follow it through tokens, embeddings, every attention head and MLP of every layer, and the output probabilities. Then sample the next token. You can inspect every number and the arithmetic that produced it.
- **Depth on demand.** Each chapter step reads as a story, as worked numbers, as the formula, or as code.
- **Two scales.** A toy example small enough to compute by hand, and a tiny real GPT-2 style model where every value is visible. The tiny model is trained on short, simple stories.
- **Quick review.** A page of formula cards and a glossary.
- **Grounded in the literature.** A references page lists every source with a pinned link, so each link always opens the same version. Simplifications are labeled. Links to exact sections and equations will follow once the chapters are written.
- **Runs in the browser.** No server, no account and no tracking. Installable and usable offline.

## How to run

You need [Node.js](https://nodejs.org/) 22.13 or newer (24 is recommended) and Git.

1. Get the code and install everything.

   ```sh
   git clone https://github.com/dwinzg/transformer-visualized.git
   cd transformer-visualized
   npm install
   ```

2. Start the site.

   ```sh
   npm run dev --workspace @transformer-visualized/site
   ```

   Then open http://localhost:4321/transformer-visualized/ in your browser. The page reloads as you edit.

3. Check your changes before opening a pull request.

   ```sh
   npm run check
   ```

   This runs the format check, lint, type checks and unit tests. To run the browser tests too, install the test browsers once with `npx playwright install` and then run `npm run e2e`.

To build the site the way it is published, run `npm run build`. The finished pages land in `site/dist/`.

The model in this project is trained with Python. You only need that to retrain it. The steps are in [model/README.md](model/README.md).

## Contributing

Contributions are welcome, especially from people learning the material for the first time. See [CONTRIBUTING.md](CONTRIBUTING.md) for ways to contribute, the workflow, writing standards and citation rules.

## License

Copyright (c) 2026 dwinzg

- Code is released under the [MIT License](LICENSE).
- Written lessons, explanations and figures are released under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0), unless otherwise noted. The full text is in [LICENSE-CONTENT](LICENSE-CONTENT).
- Code samples inside the lessons are also available under the MIT License.
- Third-party material keeps its original license.
- The trained model in `models/` is released under the MIT License, like the code.
- The dataset it was trained on is credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
