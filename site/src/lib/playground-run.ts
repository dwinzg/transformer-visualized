import {
  explainAttentionWeight,
  explainLayerNorm,
  explainRmsNorm,
  forward,
  forwardLlama,
  normalizeText,
  rowView,
  type AttentionTraceLike,
  type LlamaTrace,
  type Tokenizer,
  type Trace,
} from '@transformer-visualized/engine';
import { round } from './embeddings';
import { displayToken, type Arch, type LlamaModel, type Model } from './model-loader';
import { attentionView, embeddingView, scoresView } from './model-views';

/** A grid for every token would be huge, so attention shows the last few. */
export const GRID_TOKENS = 24;

/** One step of the residual stream for one token: what was added, and the length after. */
export interface ResidualStep {
  label: string;
  added: number;
  length: number;
}

const r4 = (v: number) => round(v, 4);
const norm = (row: ArrayLike<number>) => Math.sqrt(Array.from(row).reduce((s, v) => s + v * v, 0));

/** Each token's residual stream, from its embedding through every attention and feed-forward step. */
function residualView(trace: Trace | LlamaTrace, count: number): ResidualStep[][] {
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

/** How many of a token's numbers the norm view walks through. */
const NORM_SHOWN = 4;

/** How the final norm rescales one token's numbers before they become chances. */
export interface NormStep {
  /** GPT-2's LayerNorm takes the average away and shifts. Llama's RMSNorm does neither. */
  kind: 'layer' | 'rms';
  /** The average, taken away by LayerNorm. 0 for RMSNorm, which takes nothing away. */
  mean: number;
  /** What every number is divided by: √(variance + eps), or for RMSNorm √(mean square + eps). */
  spread: number;
  /** The length of all the numbers before and after. */
  before: number;
  after: number;
  /** The first few numbers, each step by step. */
  values: { input: number; normalized: number; gamma: number; beta: number; output: number }[];
}

function normView(model: Model, trace: Trace, count: number): NormStep[] {
  const input = trace.layers.at(-1)!.output;
  const eps = model.config.layerNormEps;
  return Array.from({ length: count }, (_, i) => {
    const values = Array.from({ length: NORM_SHOWN }, (_, col) =>
      explainLayerNorm(input, trace.lnFinal, model.lnFinal, eps, i, col),
    );
    return {
      kind: 'layer' as const,
      mean: r4(values[0].mean),
      spread: r4(Math.sqrt(values[0].variance + eps)),
      before: round(norm(rowView(input, i)), 2),
      after: round(norm(rowView(trace.lnFinal.out, i)), 2),
      values: values.map((v) => ({
        input: r4(v.input),
        normalized: r4(v.normalized),
        gamma: r4(v.gamma),
        beta: r4(v.beta),
        output: r4(v.output),
      })),
    };
  });
}

function rmsNormView(model: LlamaModel, trace: LlamaTrace, count: number): NormStep[] {
  const input = trace.layers.at(-1)!.output;
  const eps = model.config.layerNormEps;
  return Array.from({ length: count }, (_, i) => {
    const values = Array.from({ length: NORM_SHOWN }, (_, col) =>
      explainRmsNorm(input, trace.lnFinal.meanSquare, model.normFinal, eps, i, col),
    );
    return {
      kind: 'rms' as const,
      mean: 0,
      spread: r4(Math.sqrt(values[0].meanSquare + eps)),
      before: round(norm(rowView(input, i)), 2),
      after: round(norm(rowView(trace.lnFinal.out, i)), 2),
      values: values.map((v) => ({
        input: r4(v.input),
        normalized: r4(v.normalized),
        gamma: r4(v.gamma),
        beta: 0,
        output: r4(v.output),
      })),
    };
  });
}

/** Everything the playground shows for one text, as plain data a worker can send. */
export function runPlayground(
  model: Model | LlamaModel,
  tokenizer: Tokenizer,
  text: string,
  /** Gets the full trace, for a caller that answers questions about it later. */
  keep?: (trace: AttentionTraceLike) => void,
) {
  const arch: Arch = 'arch' in model.config ? 'llama' : 'gpt2';
  const all = tokenizer.encode(normalizeText(text));
  const limit = model.config.contextLength;
  // The model reads at most contextLength tokens, so a long text keeps its end.
  const ids = all.slice(-limit);
  const tokens = ids.map((n) => displayToken(tokenizer, n));
  if (ids.length === 0) return { arch, text, tokens, cut: 0, limit, views: null };
  const llama = 'normFinal' in model ? model : null;
  const gpt2 = 'normFinal' in model ? null : model;
  const trace = llama ? forwardLlama(llama, ids) : forward(gpt2!, ids);
  keep?.(trace);
  const attention = attentionView(tokens, trace, true);
  const from = Math.max(tokens.length - GRID_TOKENS, 0);
  return {
    arch,
    text,
    tokens,
    cut: all.length - ids.length,
    limit,
    views: {
      // Llama adds no position numbers here, so its strip shows the token rows alone.
      embeddings: embeddingView(tokens, {
        tokenEmbeddings: trace.tokenEmbeddings,
        positionEmbeddings: 'positionEmbeddings' in trace ? trace.positionEmbeddings : null,
      }),
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
      norm: llama
        ? rmsNormView(llama, trace as LlamaTrace, tokens.length)
        : normView(gpt2!, trace as Trace, tokens.length),
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

/** Explains one attention weight with the engine, rounded to 4 decimals for the page. */
export function attentionMath(trace: AttentionTraceLike, ask: Ask): AttentionMath {
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
