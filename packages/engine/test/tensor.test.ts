import { describe, expect, it } from 'vitest';
import { createMatrix, matrixFromRows, rowView, valueAt } from '../src/tensor';

describe('createMatrix', () => {
  it('zero-fills a new matrix of the requested shape', () => {
    const m = createMatrix(2, 3);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(Array.from(m.data)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  it('wraps existing data without copying', () => {
    const data = new Float32Array([1, 2, 3, 4]);
    expect(createMatrix(2, 2, data).data).toBe(data);
  });

  it('rejects data whose length does not match the shape', () => {
    expect(() => createMatrix(2, 2, new Float32Array(3))).toThrow(RangeError);
  });

  it('rejects negative or fractional dimensions', () => {
    expect(() => createMatrix(-1, 2)).toThrow(RangeError);
    expect(() => createMatrix(1.5, 2)).toThrow(RangeError);
  });

  it('rejects fractional or NaN column counts', () => {
    expect(() => createMatrix(2, 1.5)).toThrow(RangeError);
    expect(() => createMatrix(2, Number.NaN)).toThrow(RangeError);
  });
});

describe('matrixFromRows', () => {
  it('lays values out row-major', () => {
    const m = matrixFromRows([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(Array.from(m.data)).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it('returns a 0x0 matrix for no rows', () => {
    const m = matrixFromRows([]);
    expect([m.rows, m.cols, m.data.length]).toEqual([0, 0, 0]);
  });

  it('rejects ragged rows', () => {
    expect(() => matrixFromRows([[1, 2], [3]])).toThrow(RangeError);
  });

  it('stores values rounded to float32', () => {
    expect(matrixFromRows([[0.1]]).data[0]).toBe(Math.fround(0.1));
  });
});

describe('rowView', () => {
  it('returns a view that shares memory with the matrix', () => {
    const m = matrixFromRows([
      [1, 2],
      [3, 4],
    ]);
    const r = rowView(m, 1);
    expect(Array.from(r)).toEqual([3, 4]);
    r[0] = 9;
    expect(valueAt(m, 1, 0)).toBe(9);
  });

  it('throws for out-of-range rows', () => {
    expect(() => rowView(matrixFromRows([[1]]), 1)).toThrow(RangeError);
    expect(() => rowView(matrixFromRows([[1]]), -1)).toThrow(RangeError);
  });

  it('rejects fractional rows', () => {
    expect(() => rowView(matrixFromRows([[1], [2]]), 0.5)).toThrow(RangeError);
  });
});

describe('valueAt', () => {
  it('reads by row and column', () => {
    const m = matrixFromRows([
      [1, 2],
      [3, 4],
    ]);
    expect(valueAt(m, 0, 1)).toBe(2);
    expect(valueAt(m, 1, 1)).toBe(4);
  });

  it('throws for out-of-range indices', () => {
    const m = matrixFromRows([[1, 2]]);
    expect(() => valueAt(m, 0, 2)).toThrow(RangeError);
    expect(() => valueAt(m, 1, 0)).toThrow(RangeError);
  });

  it('rejects negative and fractional columns', () => {
    const m = matrixFromRows([
      [1, 2],
      [3, 4],
    ]);
    expect(() => valueAt(m, 1, -1)).toThrow(RangeError);
    expect(() => valueAt(m, 0, 0.5)).toThrow(RangeError);
  });
});
