# Third-party notices

## TinyStories

The model in `models/tiny/` was trained on TinyStories, created by Ronen Eldan and Yuanzhi Li.

- Paper. Eldan, R. and Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.
- Dataset. https://huggingface.co/datasets/roneneldan/TinyStories (revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64).
- License. Community Data License Agreement, Sharing, Version 1.0 (https://cdla.dev/sharing-1-0/).

This repository does not redistribute the dataset. The tokenizer and the model weights were computed from it.

## Software in the published site

The built site includes code and files from these packages. Each is under the MIT License, printed once below.

| Package                                                  | What the site uses                                            | Copyright                                                   |
| -------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| [KaTeX](https://github.com/KaTeX/KaTeX)                  | Its stylesheet and fonts, for formulas rendered at build time | Copyright (c) 2013-2020 Khan Academy and other contributors |
| [React](https://github.com/facebook/react) and React DOM | The interactive figures                                       | Copyright (c) Meta Platforms, Inc. and affiliates           |
| [Workbox](https://github.com/GoogleChrome/workbox)       | The service worker that keeps the site for offline use        | Copyright 2018 Google LLC                                   |

### MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## The transformer diagram

The map at `/learn/architecture/` is our own drawing of the layout of Figure 1 in Vaswani et al. (2017), "Attention Is All You Need", arXiv:1706.03762. No part of the original image is copied. The page credits the paper and links to it on the references page.
