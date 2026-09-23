/**
 * A row-major grid of float32 values. `data.length === rows * cols`.
 * Build matrices with `createMatrix` or `matrixFromRows`, which check that invariant.
 */
export interface Matrix {
  readonly rows: number;
  readonly cols: number;
  readonly data: Float32Array;
}

/** Creates a matrix. It is zero-filled unless `data` is given, in which case `data` is used as is. */
export function createMatrix(rows: number, cols: number, data?: Float32Array): Matrix {
  if (!Number.isInteger(rows) || !Number.isInteger(cols) || rows < 0 || cols < 0) {
    throw new RangeError(`Invalid matrix shape ${rows}x${cols}`);
  }
  if (data !== undefined && data.length !== rows * cols) {
    throw new RangeError(
      `Expected ${rows * cols} values for a ${rows}x${cols} matrix, got ${data.length}`,
    );
  }
  return { rows, cols, data: data ?? new Float32Array(rows * cols) };
}

/** Builds a matrix from nested arrays. Every row must have the same length. */
export function matrixFromRows(values: readonly (readonly number[])[]): Matrix {
  const rows = values.length;
  const cols = rows === 0 ? 0 : values[0].length;
  const data = new Float32Array(rows * cols);
  values.forEach((row, r) => {
    if (row.length !== cols) {
      throw new RangeError(`Row ${r} has ${row.length} values, expected ${cols}`);
    }
    data.set(row, r * cols);
  });
  return createMatrix(rows, cols, data);
}

/** Returns row `r` as a view that shares memory with the matrix. */
export function rowView(m: Matrix, r: number): Float32Array {
  if (!Number.isInteger(r) || r < 0 || r >= m.rows) {
    throw new RangeError(`Row ${r} is out of range for a matrix with ${m.rows} rows`);
  }
  return m.data.subarray(r * m.cols, (r + 1) * m.cols);
}

/** Reads the value at row `r`, column `c`. */
export function valueAt(m: Matrix, r: number, c: number): number {
  if (
    !Number.isInteger(r) ||
    r < 0 ||
    r >= m.rows ||
    !Number.isInteger(c) ||
    c < 0 ||
    c >= m.cols
  ) {
    throw new RangeError(`Index (${r}, ${c}) is out of range for a ${m.rows}x${m.cols} matrix`);
  }
  return m.data[r * m.cols + c];
}
