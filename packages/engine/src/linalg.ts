import { createMatrix, type Matrix } from './tensor';

/**
 * y = x·Wᵀ + b for every row of x. W is stored as [out, in] (the PyTorch Linear layout), so
 * each output value is the dot product of two contiguous rows.
 */
export function linear(x: Matrix, weight: Matrix, bias: Float32Array | null = null): Matrix {
  if (x.cols !== weight.cols) {
    throw new RangeError(
      `linear: input has ${x.cols} columns but the weight expects ${weight.cols}`,
    );
  }
  if (bias !== null && bias.length !== weight.rows) {
    throw new RangeError(
      `linear: bias has ${bias.length} values but the weight has ${weight.rows} rows`,
    );
  }
  const n = x.cols;
  const out = createMatrix(x.rows, weight.rows);
  for (let i = 0; i < x.rows; i++) {
    const xOffset = i * n;
    const outOffset = i * weight.rows;
    for (let j = 0; j < weight.rows; j++) {
      const wOffset = j * n;
      let sum = 0;
      for (let k = 0; k < n; k++) sum += x.data[xOffset + k] * weight.data[wOffset + k];
      out.data[outOffset + j] = bias === null ? sum : sum + bias[j];
    }
  }
  return out;
}

/** Standard matrix product, [m, k] · [k, n] -> [m, n]. */
export function matmul(a: Matrix, b: Matrix): Matrix {
  if (a.cols !== b.rows) {
    throw new RangeError(`matmul: ${a.rows}x${a.cols} cannot multiply ${b.rows}x${b.cols}`);
  }
  const out = createMatrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) sum += a.data[i * a.cols + k] * b.data[k * b.cols + j];
      out.data[i * b.cols + j] = sum;
    }
  }
  return out;
}

/** Element-wise sum of two matrices with the same shape. */
export function add(a: Matrix, b: Matrix): Matrix {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new RangeError(`add: shapes ${a.rows}x${a.cols} and ${b.rows}x${b.cols} differ`);
  }
  const out = createMatrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) out.data[i] = a.data[i] + b.data[i];
  return out;
}

/** Copies columns [start, end) of every row into a new matrix. */
export function sliceColumns(m: Matrix, start: number, end: number): Matrix {
  const isInteger = Number.isInteger(start) && Number.isInteger(end);
  if (isInteger && start > end) {
    throw new RangeError(`sliceColumns: start ${start} is greater than end ${end}`);
  }
  const valid = isInteger && start >= 0 && end <= m.cols;
  if (!valid) {
    throw new RangeError(`sliceColumns: [${start}, ${end}) is out of range for ${m.cols} columns`);
  }
  const width = end - start;
  const out = createMatrix(m.rows, width);
  for (let r = 0; r < m.rows; r++) {
    out.data.set(m.data.subarray(r * m.cols + start, r * m.cols + end), r * width);
  }
  return out;
}

/** Joins matrices side by side. Every part must have the same number of rows. */
export function concatColumns(parts: readonly Matrix[]): Matrix {
  if (parts.length === 0) throw new RangeError('concatColumns: nothing to join');
  const rows = parts[0].rows;
  if (parts.some((part) => part.rows !== rows)) {
    throw new RangeError('concatColumns: parts have different row counts');
  }
  const cols = parts.reduce((total, part) => total + part.cols, 0);
  const out = createMatrix(rows, cols);
  for (let r = 0; r < rows; r++) {
    let offset = r * cols;
    for (const part of parts) {
      out.data.set(part.data.subarray(r * part.cols, (r + 1) * part.cols), offset);
      offset += part.cols;
    }
  }
  return out;
}

/** Copies the listed rows into a new matrix, for example token ids from an embedding table. */
export function gatherRows(m: Matrix, indices: readonly number[]): Matrix {
  // Validate every index before allocating the output, so a bad index never pays for a
  // (potentially large) allocation it is about to throw away.
  for (const index of indices) {
    if (!Number.isInteger(index) || index < 0 || index >= m.rows) {
      throw new RangeError(`gatherRows: row ${index} is out of range for ${m.rows} rows`);
    }
  }
  const out = createMatrix(indices.length, m.cols);
  indices.forEach((index, i) => {
    out.data.set(m.data.subarray(index * m.cols, (index + 1) * m.cols), i * m.cols);
  });
  return out;
}

/** Applies `fn` to every value. */
export function mapValues(m: Matrix, fn: (value: number) => number): Matrix {
  const out = createMatrix(m.rows, m.cols);
  for (let i = 0; i < m.data.length; i++) out.data[i] = fn(m.data[i]);
  return out;
}
