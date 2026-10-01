import { describe, expect, it } from 'vitest';
import {
  add,
  concatColumns,
  gatherRows,
  linear,
  mapValues,
  matmul,
  sliceColumns,
} from '../src/linalg';
import { matrixFromRows } from '../src/tensor';

const values = (m: { data: Float32Array }) => Array.from(m.data);

describe('linear', () => {
  it('computes x·Wᵀ + b with W stored as [out, in]', () => {
    const x = matrixFromRows([
      [1, 2],
      [3, 4],
    ]);
    const w = matrixFromRows([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const y = linear(x, w, new Float32Array([0.5, -1, 0]));
    expect([y.rows, y.cols]).toEqual([2, 3]);
    expect(values(y)).toEqual([1.5, 1, 3, 3.5, 3, 7]);
  });

  it('adds no bias when none is given', () => {
    expect(values(linear(matrixFromRows([[2, 3]]), matrixFromRows([[4, 5]])))).toEqual([23]);
  });

  it('rejects mismatched shapes', () => {
    expect(() => linear(matrixFromRows([[1, 2]]), matrixFromRows([[1, 2, 3]]))).toThrow(RangeError);
    expect(() =>
      linear(matrixFromRows([[1, 2]]), matrixFromRows([[1, 2]]), new Float32Array(2)),
    ).toThrow(RangeError);
  });
});

describe('matmul', () => {
  it('multiplies [m, k] by [k, n]', () => {
    const a = matrixFromRows([
      [1, 2],
      [3, 4],
    ]);
    const b = matrixFromRows([
      [5, 6],
      [7, 8],
    ]);
    expect(values(matmul(a, b))).toEqual([19, 22, 43, 50]);
  });

  it('rejects incompatible shapes', () => {
    expect(() => matmul(matrixFromRows([[1, 2]]), matrixFromRows([[1, 2]]))).toThrow(RangeError);
  });
});

describe('add', () => {
  it('adds element by element', () => {
    expect(values(add(matrixFromRows([[1, 2]]), matrixFromRows([[3, 4]])))).toEqual([4, 6]);
  });

  it('rejects different shapes', () => {
    expect(() => add(matrixFromRows([[1, 2]]), matrixFromRows([[1], [2]]))).toThrow(RangeError);
  });
});

describe('sliceColumns', () => {
  const m = matrixFromRows([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ]);

  it('copies a column range from every row', () => {
    const s = sliceColumns(m, 1, 3);
    expect([s.rows, s.cols]).toEqual([2, 2]);
    expect(values(s)).toEqual([2, 3, 6, 7]);
  });

  it('allows an empty range', () => {
    expect(sliceColumns(m, 2, 2).cols).toBe(0);
  });

  it('rejects ranges outside the matrix', () => {
    expect(() => sliceColumns(m, -1, 2)).toThrow(RangeError);
    expect(() => sliceColumns(m, 2, 5)).toThrow(RangeError);
    expect(() => sliceColumns(m, 3, 2)).toThrow(RangeError);
  });

  it('gives a precise message when start is greater than end', () => {
    expect(() => sliceColumns(m, 3, 2)).toThrow(/start 3 is greater than end 2/);
  });
});

describe('concatColumns', () => {
  it('joins matrices side by side', () => {
    const joined = concatColumns([
      matrixFromRows([[1], [2]]),
      matrixFromRows([
        [3, 4],
        [5, 6],
      ]),
    ]);
    expect([joined.rows, joined.cols]).toEqual([2, 3]);
    expect(values(joined)).toEqual([1, 3, 4, 2, 5, 6]);
  });

  it('rejects parts with different row counts and an empty list', () => {
    expect(() => concatColumns([matrixFromRows([[1]]), matrixFromRows([[1], [2]])])).toThrow(
      RangeError,
    );
    expect(() => concatColumns([])).toThrow(RangeError);
  });
});

describe('gatherRows', () => {
  const table = matrixFromRows([
    [0, 0],
    [1, 1],
    [2, 2],
  ]);

  it('copies the listed rows in order, repeats allowed', () => {
    expect(values(gatherRows(table, [2, 0, 2]))).toEqual([2, 2, 0, 0, 2, 2]);
  });

  it('rejects indices outside the table', () => {
    expect(() => gatherRows(table, [3])).toThrow(RangeError);
    expect(() => gatherRows(table, [-1])).toThrow(RangeError);
    expect(() => gatherRows(table, [0.5])).toThrow(RangeError);
  });
});

describe('mapValues', () => {
  it('applies a function to every value', () => {
    expect(values(mapValues(matrixFromRows([[1, -2]]), (v) => v * 2))).toEqual([2, -4]);
  });
});
