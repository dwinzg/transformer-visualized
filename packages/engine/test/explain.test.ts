import { describe, expect, it } from 'vitest';
import {
  explainAttentionWeight,
  explainDot,
  explainLayerNorm,
  explainProbability,
} from '../src/explain';
import { forward } from '../src/forward';
import { loadModel } from '../src/model';
import { probabilities } from '../src/sampling';
import { rowView, valueAt } from '../src/tensor';
import { expectAllClose } from './helpers/close';
import { readFixture } from './helpers/fixtures';

const model = loadModel(readFixture('micro/model.safetensors'));
const trace = forward(model, [3, 1, 4, 1, 5]);
const tight = { atol: 1e-5, rtol: 1e-5 };

describe('explainDot', () => {
  it('lists each product and their sum', () => {
    const { products, sum } = explainDot([1, 2, 3], [4, -5, 0.5]);
    expect(Array.from(products)).toEqual([4, -10, 1.5]);
    expect(sum).toBe(-4.5);
  });

  it('rejects vectors of different lengths', () => {
    expect(() => explainDot([1], [1, 2])).toThrow(RangeError);
  });

  it('adds the displayed products up to exactly the displayed sum', () => {
    // Values chosen so float32 rounding of each product would not add up to sum exactly.
    const a = [0.1, 0.2, 0.3, 0.4, 0.123456789, 0.987654321, 0.333333, 0.5];
    const b = [1.1, 1.2, 1.3, 1.4, 1.523456789, 1.187654321, 1.633333, 1.7];
    const { products, sum } = explainDot(a, b);
    expect(products).toBeInstanceOf(Float64Array);
    let total = 0;
    for (const product of products) total += product;
    expect(total).toBe(sum);
  });
});

describe('explainAttentionWeight', () => {
  it('rebuilds a weight from the query, the key and the softmax of its row', () => {
    const head = trace.layers[1].heads[0];
    const e = explainAttentionWeight(trace, 1, 0, 3, 1);
    expect(e.masked).toBe(false);
    expect(e.scale).toBeCloseTo(1 / Math.sqrt(8), 12);
    expectAllClose('query', e.query, rowView(head.q, 3), { atol: 0, rtol: 0 });
    expectAllClose('key', e.key, rowView(head.k, 1), { atol: 0, rtol: 0 });
    expectAllClose('dot', [e.dot.sum], [valueAt(head.scores, 3, 1)], tight);
    expectAllClose('scaled', e.scaledScores, rowView(head.scaledMasked, 3), { atol: 0, rtol: 0 });
    expectAllClose('weight', [e.weight], [valueAt(head.weights, 3, 1)], tight);
    expectAllClose(
      'row',
      e.exps.map((x) => x / e.expSum),
      rowView(head.weights, 3),
      tight,
    );
  });

  it('marks future keys as masked with zero weight', () => {
    const e = explainAttentionWeight(trace, 0, 1, 1, 3);
    expect(e.masked).toBe(true);
    expect(e.weight).toBe(0);
    expect(e.scaledScores[3]).toBe(-Infinity);
  });

  it('adds the displayed exps up to exactly the displayed expSum', () => {
    const e = explainAttentionWeight(trace, 1, 0, 3, 1);
    expect(e.exps).toBeInstanceOf(Float64Array);
    let total = 0;
    for (const exp of e.exps) total += exp;
    expect(total).toBe(e.expSum);
  });

  it('rejects layers, heads or tokens that do not exist', () => {
    expect(() => explainAttentionWeight(trace, 2, 0, 0, 0)).toThrow(RangeError);
    expect(() => explainAttentionWeight(trace, 0, 2, 0, 0)).toThrow(RangeError);
    expect(() => explainAttentionWeight(trace, 0, 0, 5, 0)).toThrow(RangeError);
    expect(() => explainAttentionWeight(trace, 0, 0, 0, -1)).toThrow(RangeError);
  });
});

describe('explainProbability', () => {
  it('rebuilds a probability from its logit', () => {
    const logits = rowView(trace.logits, 4);
    for (const temperature of [0.5, 1, 2]) {
      const e = explainProbability(logits, 17, temperature);
      expect(e.logit).toBe(logits[17]);
      expect(e.scaledLogit).toBeCloseTo(logits[17] / temperature, 12);
      expectAllClose('p', [e.probability], [probabilities(logits, temperature)[17]], tight);
      expectAllClose('p', [e.exp / e.expSum], [e.probability], tight);
    }
  });

  it('rejects unknown tokens and bad temperatures', () => {
    const logits = rowView(trace.logits, 0);
    expect(() => explainProbability(logits, 64, 1)).toThrow(RangeError);
    expect(() => explainProbability(logits, 0, 0)).toThrow(RangeError);
  });
});

describe('explainLayerNorm', () => {
  const layer = trace.layers[0];
  const norm = layer.ln1;
  const weights = model.blocks[0].ln1;

  it('rebuilds one LayerNorm output from its row statistics', () => {
    const e = explainLayerNorm(layer.input, norm, weights, 1e-5, 2, 7);
    expect(e.input).toBe(valueAt(layer.input, 2, 7));
    expect(e.mean).toBe(norm.mean[2]);
    expect(e.variance).toBe(norm.variance[2]);
    expect(e.gamma).toBe(weights.weight[7]);
    expectAllClose('out', [e.output], [valueAt(layer.ln1.out, 2, 7)], tight);
  });

  it('rejects a norm trace whose row count does not match the input', () => {
    const shortNorm = { ...norm, mean: norm.mean.slice(0, 1) };
    expect(() => explainLayerNorm(layer.input, shortNorm, weights, 1e-5, 2, 7)).toThrow(RangeError);
  });

  it('rejects weights whose column count does not match the input', () => {
    const shortWeights = { weight: weights.weight.slice(0, 1), bias: weights.bias };
    expect(() => explainLayerNorm(layer.input, norm, shortWeights, 1e-5, 2, 7)).toThrow(RangeError);
  });

  it('rejects a row or column outside the input', () => {
    expect(() => explainLayerNorm(layer.input, norm, weights, 1e-5, -1, 7)).toThrow(RangeError);
    expect(() => explainLayerNorm(layer.input, norm, weights, 1e-5, 2, 999)).toThrow(RangeError);
  });
});
