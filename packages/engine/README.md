# @transformer-visualized/engine

A small TypeScript engine that runs GPT-2 style models and records every intermediate value, so a visualization can show where each number comes from. It has no runtime dependencies and runs in browsers and Node.

## Usage

```ts
import { forward, loadModel, nextTokenDistribution, rowView } from '@transformer-visualized/engine';

const response = await fetch('models/tiny/model.safetensors');
const model = loadModel(await response.arrayBuffer());

const trace = forward(model, [12, 845, 9]);
const next = nextTokenDistribution(rowView(trace.logits, 2), { temperature: 0.8, topK: 20 });
```

## What it provides

- `loadModel` reads a safetensors file written by `model/tv_model/io.py` and checks every weight name and shape.
- `loadLlama` reads, and `forwardLlama` runs, the Llama-style model in `models/llama-tiny`, with RMSNorm, rotary positions, a SwiGLU feed forward and grouped-query attention. Its `LlamaTrace` has the same names where the parts are the same, plus the query and key before rotation and the feed forward's gate and up projections.
- `forward` runs the model and returns a `Trace` with the embeddings, every head's queries, keys, values, scores and weights, the feed-forward activations, the residual stream after each step, and the logits.
- `nextTokenDistribution`, `sample` and `generate` turn logits into next tokens with temperature, top-k and top-p. `generate` reuses earlier keys and values (a KV cache).
- The `explain*` functions return the terms behind a single value, such as one attention weight or one probability.

## Correctness

The tests compare every traced value with the PyTorch reference model in `model/` on fixed inputs. Values are stored as float32 and agree within 1e-4 + 1e-3 × |expected|.

```sh
npm test --workspace @transformer-visualized/engine
```
