import { causalScaled, matmul, matrixFromRows, softmaxRows } from '@transformer-visualized/engine';
import { describe, expect, it } from 'vitest';
import { TOY_K, TOY_Q, TOY_V, toyAttention } from './toy-attention';

const rows = (m: { rows: number; cols: number; data: Float32Array }) =>
  Array.from({ length: m.rows }, (_, r) => Array.from(m.data.slice(r * m.cols, (r + 1) * m.cols)));

describe('toyAttention', () => {
  const toy = toyAttention();
  it('matches the engine, step by step', () => {
    const q = matrixFromRows(TOY_Q);
    const kT = matrixFromRows(TOY_K[0].map((_, c) => TOY_K.map((row) => row[c])));
    const scores = matmul(q, kT);
    const weights = softmaxRows(causalScaled(scores, 1 / Math.sqrt(2)));
    const output = matmul(weights, matrixFromRows(TOY_V));
    rows(scores).forEach((row, i) =>
      row.forEach((s, j) => expect(toy.scores[i][j]).toBeCloseTo(s, 5)),
    );
    rows(weights).forEach((row, i) =>
      row.forEach((w, j) => expect(toy.weights[i][j]).toBeCloseTo(w, 5)),
    );
    rows(output).forEach((row, i) =>
      row.forEach((o, j) => expect(toy.output[i][j]).toBeCloseTo(o, 5)),
    );
  });
  it('hides the future and makes each row of weights add up to 1', () => {
    expect(toy.masked[0][1]).toBeNull();
    expect(toy.masked[1][2]).toBeNull();
    for (const row of toy.weights) expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 10);
    expect(toy.weights[0][1]).toBe(0);
  });
  it('gives the numbers the chapter quotes', () => {
    expect(toy.scores[2]).toEqual([4, 0, 2]);
    expect(toy.weights[2].map((w) => Number(w.toFixed(2)))).toEqual([0.77, 0.05, 0.19]);
  });
});
