import { describe, expect, it } from 'vitest';
import { causalScaled, geluTanh, layerNorm, softmax, softmaxRows } from '../src/ops';
import { matrixFromRows } from '../src/tensor';
import { expectAllClose } from './helpers/close';

const exact = { atol: 1e-6, rtol: 0 };
const ones = (n: number) => new Float32Array(n).fill(1);
const zeros = (n: number) => new Float32Array(n);

describe('layerNorm', () => {
  it('normalizes each row to zero mean and unit variance before scale and shift', () => {
    const { mean, variance, out } = layerNorm(matrixFromRows([[1, 2, 3, 4]]), ones(4), zeros(4), 0);
    expectAllClose('mean', mean, [2.5], exact);
    expectAllClose('variance', variance, [1.25], exact);
    expectAllClose('out', out.data, [-1.3416408, -0.4472136, 0.4472136, 1.3416408], exact);
  });

  it('applies the learned scale and shift per column', () => {
    const gamma = new Float32Array([2, 1, 1, 1]);
    const beta = new Float32Array([0, 0, 0, 1]);
    const { out } = layerNorm(matrixFromRows([[1, 2, 3, 4]]), gamma, beta, 0);
    expectAllClose('out', out.data, [-2.6832816, -0.4472136, 0.4472136, 2.3416408], exact);
  });

  it('adds eps to the variance inside the square root', () => {
    const { out } = layerNorm(matrixFromRows([[1, -1]]), ones(2), zeros(2), 3);
    expectAllClose('out', out.data, [0.5, -0.5], exact);
  });

  it('treats rows independently', () => {
    const { mean, variance } = layerNorm(
      matrixFromRows([
        [1, 2],
        [10, 30],
      ]),
      ones(2),
      zeros(2),
      0,
    );
    expectAllClose('mean', mean, [1.5, 20], exact);
    expectAllClose('variance', variance, [0.25, 100], exact);
  });

  it('rejects scale or shift of the wrong length', () => {
    expect(() => layerNorm(matrixFromRows([[1, 2]]), ones(3), zeros(2), 0)).toThrow(RangeError);
  });
});

describe('geluTanh', () => {
  it('matches the tanh approximation used by GPT-2', () => {
    expectAllClose(
      'gelu',
      [1, -1, 0.5, 3].map(geluTanh),
      [0.841192, -0.158808, 0.345714, 2.996363],
      { atol: 2e-6, rtol: 0 },
    );
  });

  it('is zero at zero and close to the identity for large inputs', () => {
    expect(geluTanh(0)).toBe(0);
    expect(geluTanh(10)).toBeCloseTo(10, 6);
    expect(Math.abs(geluTanh(-10))).toBeLessThan(1e-5);
  });
});

describe('causalScaled', () => {
  it('scales visible scores and hides future positions', () => {
    const m = causalScaled(
      matrixFromRows([
        [1, 2],
        [3, 4],
      ]),
      0.5,
    );
    expect(Array.from(m.data)).toEqual([0.5, -Infinity, 1.5, 2]);
  });

  it('requires a square matrix', () => {
    expect(() => causalScaled(matrixFromRows([[1, 2]]), 1)).toThrow(RangeError);
  });
});

describe('softmax', () => {
  it('turns scores into probabilities that sum to 1', () => {
    expectAllClose('p', softmax([1, 2, 3]), [0.0900306, 0.2447285, 0.665241], exact);
    expectAllClose('p', softmax([0, 0]), [0.5, 0.5], exact);
  });

  it('gives -Infinity entries exactly zero probability', () => {
    expect(Array.from(softmax([-Infinity, 0]))).toEqual([0, 1]);
  });

  it('stays finite for very large scores', () => {
    expectAllClose('p', softmax([1000, 1000]), [0.5, 0.5], exact);
  });

  it('rejects input with no finite value', () => {
    expect(() => softmax([-Infinity, -Infinity])).toThrow(RangeError);
    expect(() => softmax([])).toThrow(RangeError);
  });
});

describe('softmaxRows', () => {
  it('applies softmax to each row separately', () => {
    const m = softmaxRows(
      matrixFromRows([
        [0, 0],
        [-Infinity, 5],
      ]),
    );
    expect(Array.from(m.data)).toEqual([0.5, 0.5, 0, 1]);
  });
});
