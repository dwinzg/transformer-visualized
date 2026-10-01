import type { ModelConfig } from './model';
import type { NormTrace } from './ops';
import type { Matrix } from './tensor';

/** Everything one attention head computed. T is the number of tokens. */
export interface HeadTrace {
  /** Queries [T, dHead]. */
  readonly q: Matrix;
  /** Keys [T, dHead]. */
  readonly k: Matrix;
  /** Values [T, dHead]. */
  readonly v: Matrix;
  /** q·kᵀ before scaling and masking [T, T]. */
  readonly scores: Matrix;
  /** scores / √dHead, with -Infinity where a token would see a later token [T, T]. */
  readonly scaledMasked: Matrix;
  /** Row-wise softmax of scaledMasked [T, T]. Each row sums to 1. */
  readonly weights: Matrix;
  /** weights · v [T, dHead]. */
  readonly out: Matrix;
}

/** Everything one transformer block computed. */
export interface LayerTrace {
  /** Residual stream entering the block [T, dModel]. */
  readonly input: Matrix;
  readonly ln1: NormTrace;
  readonly heads: readonly HeadTrace[];
  /** Head outputs side by side [T, dModel]. */
  readonly attnConcat: Matrix;
  /** Attention output after the output projection [T, dModel]. */
  readonly attnOut: Matrix;
  /** input + attnOut. */
  readonly residAfterAttn: Matrix;
  readonly ln2: NormTrace;
  /** Feed-forward layer before GELU [T, dMlp]. */
  readonly mlpHidden: Matrix;
  /** Feed-forward layer after GELU [T, dMlp]. */
  readonly mlpAct: Matrix;
  /** Feed-forward output [T, dModel]. */
  readonly mlpOut: Matrix;
  /** residAfterAttn + mlpOut, the next block's input. */
  readonly output: Matrix;
}

/** Every intermediate value of one forward pass. */
export interface Trace {
  readonly config: ModelConfig;
  readonly tokenIds: readonly number[];
  /** Rows of the token embedding table [T, dModel]. */
  readonly tokenEmbeddings: Matrix;
  /** Rows of the position embedding table [T, dModel]. */
  readonly positionEmbeddings: Matrix;
  /** tokenEmbeddings + positionEmbeddings. */
  readonly embeddings: Matrix;
  readonly layers: readonly LayerTrace[];
  readonly lnFinal: NormTrace;
  /** A score for every vocabulary token at every position [T, vocab]. */
  readonly logits: Matrix;
}

export interface FlatTensor {
  readonly shape: readonly number[];
  readonly data: Float32Array;
}

/**
 * Lists every traced value under its path, for example `layers.0.heads.1.weights`.
 * The paths match the tensor names in the Python reference fixtures.
 */
export function flattenTrace(trace: Trace): Map<string, FlatTensor> {
  const flat = new Map<string, FlatTensor>();
  const matrix = (path: string, m: Matrix): void => {
    flat.set(path, { shape: [m.rows, m.cols], data: m.data });
  };
  const norm = (path: string, n: NormTrace): void => {
    flat.set(`${path}.mean`, { shape: [n.mean.length], data: n.mean });
    flat.set(`${path}.variance`, { shape: [n.variance.length], data: n.variance });
    matrix(`${path}.out`, n.out);
  };
  matrix('tokenEmbeddings', trace.tokenEmbeddings);
  matrix('positionEmbeddings', trace.positionEmbeddings);
  matrix('embeddings', trace.embeddings);
  trace.layers.forEach((layer, l) => {
    const p = `layers.${l}`;
    matrix(`${p}.input`, layer.input);
    norm(`${p}.ln1`, layer.ln1);
    layer.heads.forEach((head, h) => {
      const hp = `${p}.heads.${h}`;
      matrix(`${hp}.q`, head.q);
      matrix(`${hp}.k`, head.k);
      matrix(`${hp}.v`, head.v);
      matrix(`${hp}.scores`, head.scores);
      matrix(`${hp}.scaledMasked`, head.scaledMasked);
      matrix(`${hp}.weights`, head.weights);
      matrix(`${hp}.out`, head.out);
    });
    matrix(`${p}.attnConcat`, layer.attnConcat);
    matrix(`${p}.attnOut`, layer.attnOut);
    matrix(`${p}.residAfterAttn`, layer.residAfterAttn);
    norm(`${p}.ln2`, layer.ln2);
    matrix(`${p}.mlpHidden`, layer.mlpHidden);
    matrix(`${p}.mlpAct`, layer.mlpAct);
    matrix(`${p}.mlpOut`, layer.mlpOut);
    matrix(`${p}.output`, layer.output);
  });
  norm('lnFinal', trace.lnFinal);
  matrix('logits', trace.logits);
  return flat;
}
