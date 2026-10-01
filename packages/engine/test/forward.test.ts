import { describe, expect, it } from 'vitest';
import { forward } from '../src/forward';
import { loadModel } from '../src/model';
import { rowView } from '../src/tensor';
import { flattenTrace } from '../src/trace';
import { expectAllClose } from './helpers/close';
import { readFixture } from './helpers/fixtures';

const model = loadModel(readFixture('micro/model.safetensors'));

describe('forward', () => {
  it('records every intermediate with the expected shapes', () => {
    const trace = forward(model, [3, 1, 4]);
    expect(trace.tokenIds).toEqual([3, 1, 4]);
    expect([trace.embeddings.rows, trace.embeddings.cols]).toEqual([3, 16]);
    expect(trace.layers).toHaveLength(2);
    expect(trace.layers[0].heads).toHaveLength(2);
    const head = trace.layers[1].heads[0];
    expect([head.q.rows, head.q.cols]).toEqual([3, 8]);
    expect([head.weights.rows, head.weights.cols]).toEqual([3, 3]);
    expect([trace.layers[0].mlpHidden.rows, trace.layers[0].mlpHidden.cols]).toEqual([3, 64]);
    expect([trace.logits.rows, trace.logits.cols]).toEqual([3, 64]);
    expect(flattenTrace(trace).size).toBe(7 + 2 * (14 + 2 * 7));
  });

  it('adds token and position embeddings', () => {
    const trace = forward(model, [5, 5]);
    const sum = trace.tokenEmbeddings.data.map((v, i) => v + trace.positionEmbeddings.data[i]);
    expectAllClose('embeddings', trace.embeddings.data, sum, { atol: 0, rtol: 0 });
    // The same token at two positions gets different inputs.
    expect(Array.from(rowView(trace.embeddings, 0))).not.toEqual(
      Array.from(rowView(trace.embeddings, 1)),
    );
  });

  it('never lets a token see later tokens', () => {
    const short = forward(model, [1, 2]);
    const long = forward(model, [1, 2, 3]);
    expect(Array.from(rowView(long.logits, 1))).toEqual(Array.from(rowView(short.logits, 1)));
    const weights = long.layers[0].heads[1].weights;
    expect(weights.data[0 * 3 + 1]).toBe(0);
    expect(weights.data[0 * 3 + 2]).toBe(0);
    expect(weights.data[1 * 3 + 2]).toBe(0);
  });

  it('rejects empty, too long or out-of-vocabulary input', () => {
    expect(() => forward(model, [])).toThrow(RangeError);
    expect(() => forward(model, new Array<number>(17).fill(0))).toThrow(RangeError);
    expect(() => forward(model, [64])).toThrow(RangeError);
    expect(() => forward(model, [-1])).toThrow(RangeError);
    expect(() => forward(model, [1.5])).toThrow(RangeError);
  });
});
