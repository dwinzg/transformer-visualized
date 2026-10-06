import {
  explainAttentionWeight,
  forward,
  normalizeText,
  rowView,
  type Tokenizer,
  type Trace,
} from '@transformer-visualized/engine';
import { round } from './embeddings';
import { displayToken, type Model } from './model-loader';
import { attentionView, embeddingView, scoresView } from './model-views';

/** A grid for every token would be huge, so attention shows the last few. */
export const GRID_TOKENS = 24;

/** One step of the residual stream for one token: what was added, and the length after. */
export interface ResidualStep {
  label: string;
  added: number;
  length: number;
}

const norm = (row: ArrayLike<number>) => Math.sqrt(Array.from(row).reduce((s, v) => s + v * v, 0));

/** Each token's residual stream, from its embedding through every attention and feed-forward step. */
function residualView(trace: Trace, count: number): ResidualStep[][] {
  return Array.from({ length: count }, (_, i) => {
    const start = rowView(trace.embeddings, i);
    const steps: ResidualStep[] = [
      { label: 'Start', added: round(norm(start), 2), length: round(norm(start), 2) },
    ];
    trace.layers.forEach((layer, n) => {
      steps.push({
        label: `Block ${n + 1} attention`,
        added: round(norm(rowView(layer.attnOut, i)), 2),
        length: round(norm(rowView(layer.residAfterAttn, i)), 2),
      });
      steps.push({
        label: `Block ${n + 1} feed forward`,
        added: round(norm(rowView(layer.mlpOut, i)), 2),
        length: round(norm(rowView(layer.output, i)), 2),
      });
    });
    return steps;
  });
}

/** Everything the playground shows for one text, as plain data a worker can send. */
export function runPlayground(
  model: Model,
  tokenizer: Tokenizer,
  text: string,
  /** Gets the full trace, for a caller that answers questions about it later. */
  keep?: (trace: Trace) => void,
) {
  const all = tokenizer.encode(normalizeText(text));
  const limit = model.config.contextLength;
  // The model reads at most contextLength tokens, so a long text keeps its end.
  const ids = all.slice(-limit);
  const tokens = ids.map((n) => displayToken(tokenizer, n));
  if (ids.length === 0) return { text, tokens, cut: 0, limit, views: null };
  const trace = forward(model, ids);
  keep?.(trace);
  const attention = attentionView(tokens, trace, true);
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
        // Rows only hold the tokens they can see, so the same cut keeps them lined up.
        scores: attention.scores!.map((heads) =>
          heads.map((grid) => grid.slice(from).map((row) => row.slice(from))),
        ),
      },
      scores: scoresView(tokens, trace, (n) => displayToken(tokenizer, n), model.config.vocabSize),
      residual: residualView(trace, tokens.length),
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
  | { type: 'run'; seq: number; run: PlaygroundRun }
  | { type: 'explained'; seq: number; ask: Ask; math: AttentionMath | null };

/** A question about one attention weight: in this layer and head, row looks at column. */
export interface Ask {
  layer: number;
  head: number;
  row: number;
  column: number;
}

/** How one attention weight was worked out, as plain numbers. */
export interface AttentionMath {
  query: number[];
  key: number[];
  value: number[];
  products: number[];
  dot: number;
  scale: number;
  score: number;
  /** exp(score - largest score in the row) for this token, and summed over the row. */
  exp: number;
  expSum: number;
  weight: number;
}

const r4 = (v: number) => round(v, 4);

/** Explains one attention weight with the engine, rounded to 4 decimals for the page. */
export function attentionMath(trace: Trace, ask: Ask): AttentionMath {
  const e = explainAttentionWeight(trace, ask.layer, ask.head, ask.row, ask.column);
  const head = trace.layers[ask.layer].heads[ask.head];
  return {
    query: Array.from(e.query, r4),
    key: Array.from(e.key, r4),
    value: Array.from(rowView(head.v, ask.column), r4),
    products: Array.from(e.dot.products, r4),
    dot: r4(e.dot.sum),
    scale: r4(e.scale),
    score: r4(e.scaledScores[ask.column]),
    exp: r4(e.exps[ask.column]),
    expSum: r4(e.expSum),
    weight: r4(e.weight),
  };
}
