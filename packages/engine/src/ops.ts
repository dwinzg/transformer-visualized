import { createMatrix, type Matrix } from './tensor';

/** Per-row statistics and output of a LayerNorm. */
export interface NormTrace {
  readonly mean: Float32Array;
  readonly variance: Float32Array;
  readonly out: Matrix;
}

/**
 * LayerNorm over each row, (x - mean) / sqrt(variance + eps) * gamma + beta.
 * The variance is the population (biased) variance, as in PyTorch.
 */
export function layerNorm(
  x: Matrix,
  gamma: Float32Array,
  beta: Float32Array,
  eps: number,
): NormTrace {
  if (gamma.length !== x.cols || beta.length !== x.cols) {
    throw new RangeError(
      `layerNorm: expected ${x.cols} scale and shift values, got ${gamma.length} and ${beta.length}`,
    );
  }
  const n = x.cols;
  const mean = new Float32Array(x.rows);
  const variance = new Float32Array(x.rows);
  const out = createMatrix(x.rows, n);
  for (let r = 0; r < x.rows; r++) {
    const offset = r * n;
    let sum = 0;
    for (let c = 0; c < n; c++) sum += x.data[offset + c];
    const mu = sum / n;
    let squares = 0;
    for (let c = 0; c < n; c++) {
      const centered = x.data[offset + c] - mu;
      squares += centered * centered;
    }
    const v = squares / n;
    mean[r] = mu;
    variance[r] = v;
    const invStd = 1 / Math.sqrt(v + eps);
    for (let c = 0; c < n; c++) {
      out.data[offset + c] = (x.data[offset + c] - mu) * invStd * gamma[c] + beta[c];
    }
  }
  return { mean, variance, out };
}

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);

/** GELU with the tanh approximation used by GPT-2. */
export function geluTanh(x: number): number {
  return 0.5 * x * (1 + Math.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)));
}

/**
 * Scales raw attention scores and hides the future. Entry (i, j) becomes scores[i][j] * scale
 * when j <= i and -Infinity when j > i, so a token never attends to tokens after it.
 */
export function causalScaled(scores: Matrix, scale: number): Matrix {
  if (scores.rows !== scores.cols) {
    throw new RangeError(`causalScaled: scores must be square, got ${scores.rows}x${scores.cols}`);
  }
  const n = scores.cols;
  const out = createMatrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      out.data[i * n + j] = j > i ? -Infinity : scores.data[i * n + j] * scale;
    }
  }
  return out;
}

/**
 * Softmax of a list of scores. The maximum is subtracted first so large scores cannot overflow.
 * -Infinity entries get probability exactly 0.
 */
export function softmax(values: ArrayLike<number>): Float32Array {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
  if (max === -Infinity) throw new RangeError('softmax: needs at least one finite value');
  const exps = new Float64Array(values.length);
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    exps[i] = Math.exp(values[i] - max);
    sum += exps[i];
  }
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = exps[i] / sum;
  return out;
}

/** Applies `softmax` to every row. */
export function softmaxRows(m: Matrix): Matrix {
  const out = createMatrix(m.rows, m.cols);
  for (let r = 0; r < m.rows; r++) {
    out.data.set(softmax(m.data.subarray(r * m.cols, (r + 1) * m.cols)), r * m.cols);
  }
  return out;
}
