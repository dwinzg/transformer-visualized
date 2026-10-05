import { forward, normalizeText, rowView, type Tokenizer } from '@transformer-visualized/engine';
import { round } from './embeddings';
import { displayToken, type Model } from './model-loader';
import { attentionView, embeddingView, scoresView } from './model-views';

/** A grid for every token would be huge, so attention shows the last few. */
export const GRID_TOKENS = 24;

/** Everything the playground shows for one text, as plain data a worker can send. */
export function runPlayground(model: Model, tokenizer: Tokenizer, text: string) {
  const all = tokenizer.encode(normalizeText(text));
  const limit = model.config.contextLength;
  // The model reads at most contextLength tokens, so a long text keeps its end.
  const ids = all.slice(-limit);
  const tokens = ids.map((n) => displayToken(tokenizer, n));
  if (ids.length === 0) return { text, tokens, cut: 0, limit, views: null };
  const trace = forward(model, ids);
  const attention = attentionView(tokens, trace);
  const from = Math.max(tokens.length - GRID_TOKENS, 0);
  return {
    text,
    tokens,
    cut: all.length - ids.length,
    limit,
    views: {
      embeddings: embeddingView(tokens, trace),
      attention: {
        from,
        tokens: attention.tokens.slice(from),
        weights: attention.weights.map((heads) =>
          heads.map((grid) => grid.slice(from).map((row) => row.slice(from))),
        ),
      },
      scores: scoresView(tokens, trace, (n) => displayToken(tokenizer, n), model.config.vocabSize),
      activations: trace.layers.map((layer) =>
        tokens.map((_, i) => Array.from(rowView(layer.mlpAct, i), (v) => round(v, 2))),
      ),
    },
  };
}

export type PlaygroundRun = ReturnType<typeof runPlayground>;

/** Messages from the playground worker. */
export type WorkerMessage =
  | { type: 'ready' }
  | { type: 'failed' }
  | { type: 'run-failed'; seq: number }
  | { type: 'run'; seq: number; run: PlaygroundRun };
