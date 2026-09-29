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
  it('rebuilds one LayerNorm output from its row statistics', () => {
    const layer = trace.layers[0];
    const e = explainLayerNorm(layer.input, layer.ln1, model.blocks[0].ln1, 1e-5, 2, 7);
    expect(e.input).toBe(valueAt(layer.input, 2, 7));
    expect(e.mean).toBe(layer.ln1.mean[2]);
    expect(e.variance).toBe(layer.ln1.variance[2]);
    expect(e.gamma).toBe(model.blocks[0].ln1.weight[7]);
    expectAllClose('out', [e.output], [valueAt(layer.ln1.out, 2, 7)], tight);
  });
});
