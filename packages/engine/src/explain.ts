import type { LayerNormWeights } from './model';
import type { NormTrace } from './ops';
import { assertTemperature } from './sampling';
import { rowView, valueAt, type Matrix } from './tensor';
import type { Trace } from './trace';

export interface DotProductExplanation {
  /** a[i] × b[i] for every i, kept at full precision so the terms add up to `sum` exactly. */
  readonly products: Float64Array;
  /** The sum of the products. */
  readonly sum: number;
}

/** A dot product written out term by term. */
export function explainDot(a: ArrayLike<number>, b: ArrayLike<number>): DotProductExplanation {
  if (a.length !== b.length) {
    throw new RangeError(`explainDot: lengths ${a.length} and ${b.length} differ`);
  }
  const products = new Float64Array(a.length);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const product = a[i] * b[i];
    products[i] = product;
    sum += product;
  }
  return { products, sum };
}

export interface AttentionWeightExplanation {
  /** Query vector of the token that is looking [dHead]. */
  readonly query: Float32Array;
  /** Key vector of the token being looked at [dHead]. */
  readonly key: Float32Array;
  /** query · key, the raw match score. */
  readonly dot: DotProductExplanation;
  /** 1 / √dHead. */
  readonly scale: number;
  /** True when the key token comes after the query token, so it is hidden. */
  readonly masked: boolean;
  /** The query token's full row of scaled scores, -Infinity where hidden. */
  readonly scaledScores: Float32Array;
  /** The largest scaled score in the row, subtracted before exponentiating. */
  readonly rowMax: number;
  /**
   * exp(score - rowMax) for every key in the row, 0 where hidden. Kept at full precision so
   * the terms add up to `expSum` exactly.
   */
  readonly exps: Float64Array;
  readonly expSum: number;
  /** exps[key] / expSum, the attention weight. */
  readonly weight: number;
}

/** Explains how token `queryIndex` came to give weight to token `keyIndex` in one head. */
export function explainAttentionWeight(
  trace: Trace,
  layer: number,
  head: number,
  queryIndex: number,
  keyIndex: number,
): AttentionWeightExplanation {
  const headTrace = trace.layers.at(layer)?.heads.at(head);
  if (!Number.isInteger(layer) || !Number.isInteger(head) || layer < 0 || head < 0 || !headTrace) {
    throw new RangeError(`explainAttentionWeight: no head ${head} in layer ${layer}`);
  }
  const length = trace.tokenIds.length;
  for (const index of [queryIndex, keyIndex]) {
    if (!Number.isInteger(index) || index < 0 || index >= length) {
      throw new RangeError(`explainAttentionWeight: token ${index} is outside 0..${length - 1}`);
    }
  }
  const query = rowView(headTrace.q, queryIndex).slice();
  const key = rowView(headTrace.k, keyIndex).slice();
  const scaledScores = rowView(headTrace.scaledMasked, queryIndex).slice();
  let rowMax = -Infinity;
  for (const score of scaledScores) if (score > rowMax) rowMax = score;
  const exps = new Float64Array(length);
  let expSum = 0;
  for (let j = 0; j < length; j++) {
    const e = Math.exp(scaledScores[j] - rowMax);
    exps[j] = e;
    expSum += e;
  }
  return {
    query,
    key,
    dot: explainDot(query, key),
    scale: 1 / Math.sqrt(trace.config.dHead),
    masked: keyIndex > queryIndex,
    scaledScores,
    rowMax,
    exps,
    expSum,
    weight: exps[keyIndex] / expSum,
  };
}

export interface ProbabilityExplanation {
  readonly logit: number;
  readonly temperature: number;
  /** logit / temperature. */
  readonly scaledLogit: number;
  /** The largest scaled logit, subtracted before exponentiating. */
  readonly maxScaledLogit: number;
  /** exp(scaledLogit - maxScaledLogit). */
  readonly exp: number;
  /** The same exponential summed over the whole vocabulary. */
  readonly expSum: number;
  /** exp / expSum. */
  readonly probability: number;
}

/**
 * Explains one next-token probability from the logits. This is the temperature softmax only,
 * before any top-k or top-p filtering, so it can differ from a filtered `nextTokenDistribution`.
 */
export function explainProbability(
  logits: ArrayLike<number>,
  tokenId: number,
  temperature: number,
): ProbabilityExplanation {
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= logits.length) {
    throw new RangeError(`explainProbability: token ${tokenId} is outside the vocabulary`);
  }
  assertTemperature(temperature);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / temperature);
  let expSum = 0;
  for (let i = 0; i < logits.length; i++) expSum += Math.exp(logits[i] / temperature - max);
  const scaledLogit = logits[tokenId] / temperature;
  const exp = Math.exp(scaledLogit - max);
  return {
    logit: logits[tokenId],
    temperature,
    scaledLogit,
    maxScaledLogit: max,
    exp,
    expSum,
    probability: exp / expSum,
  };
}

export interface LayerNormExplanation {
  readonly input: number;
  readonly mean: number;
  readonly variance: number;
  readonly eps: number;
  /** (input - mean) / √(variance + eps). */
  readonly normalized: number;
  /** Learned scale for this column. */
  readonly gamma: number;
  /** Learned shift for this column. */
  readonly beta: number;
  /** normalized × gamma + beta. */
  readonly output: number;
}

/** Explains one LayerNorm output value from its row statistics and learned scale and shift. */
export function explainLayerNorm(
  input: Matrix,
  norm: NormTrace,
  weights: LayerNormWeights,
  eps: number,
  row: number,
  col: number,
): LayerNormExplanation {
  if (norm.mean.length !== input.rows || norm.variance.length !== input.rows) {
    throw new RangeError(
      `explainLayerNorm: norm has ${norm.mean.length} rows but the input has ${input.rows}`,
    );
  }
  if (weights.weight.length !== input.cols || weights.bias.length !== input.cols) {
    throw new RangeError(
      `explainLayerNorm: weights have ${weights.weight.length} columns but the input has ${input.cols}`,
    );
  }
  const value = valueAt(input, row, col);
  const mean = norm.mean[row];
  const variance = norm.variance[row];
  const normalized = (value - mean) / Math.sqrt(variance + eps);
  const gamma = weights.weight[col];
  const beta = weights.bias[col];
  return {
    input: value,
    mean,
    variance,
    eps,
    normalized,
    gamma,
    beta,
    output: normalized * gamma + beta,
  };
}
