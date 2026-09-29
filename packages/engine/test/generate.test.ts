import { describe, expect, it } from 'vitest';
import { forward } from '../src/forward';
import { createDecoder, generate } from '../src/generate';
import { loadModel } from '../src/model';
import { argmax } from '../src/sampling';
import { rowView } from '../src/tensor';
import { expectAllClose } from './helpers/close';
import { readFixture } from './helpers/fixtures';

const model = loadModel(readFixture('micro/model.safetensors'));
const lastLogits = (ids: number[]) => {
  const trace = forward(model, ids);
  return rowView(trace.logits, ids.length - 1);
};

function greedyByForward(prompt: number[], count: number): number[] {
  const ids = [...prompt];
  for (let i = 0; i < count; i++) ids.push(argmax(lastLogits(ids)));
  return ids;
}

describe('createDecoder', () => {
  it('starts with the logits of the last prompt position', () => {
    const decoder = createDecoder(model, [1, 2, 3]);
    expect(decoder.length).toBe(3);
    expectAllClose('logits', decoder.logits, lastLogits([1, 2, 3]));
  });

  it('matches a full forward pass after every cached step', () => {
    const ids = [1, 2, 3];
    const decoder = createDecoder(model, ids);
    for (const next of [9, 40, 7, 63, 0, 22]) {
      ids.push(next);
      const logits = decoder.step(next);
      expect(decoder.length).toBe(ids.length);
      expectAllClose(`logits after ${ids.length} tokens`, logits, lastLogits(ids));
      expect(decoder.logits).toBe(logits);
    }
  });

  it('refuses to step past the context length', () => {
    const decoder = createDecoder(model, new Array<number>(15).fill(1));
    decoder.step(2);
    expect(decoder.length).toBe(16);
    expect(() => decoder.step(3)).toThrow(RangeError);
  });

  it('rejects out-of-vocabulary tokens', () => {
    const decoder = createDecoder(model, [1]);
    expect(() => decoder.step(64)).toThrow(RangeError);
    expect(() => decoder.step(-1)).toThrow(RangeError);
  });
});

describe('generate', () => {
  it('greedy decoding matches repeated full forward passes', () => {
    const result = generate(model, [1, 2, 3], {
      maxNewTokens: 5,
      temperature: 1,
      seed: 0,
      greedy: true,
    });
    expect(result.tokenIds).toEqual(greedyByForward([1, 2, 3], 5));
    expect(result.steps).toHaveLength(5);
    expect(result.stopReason).toBe('maxNewTokens');
  });

  it('records the distribution and probability of every chosen token', () => {
    const result = generate(model, [4], { maxNewTokens: 3, temperature: 0.8, topK: 10, seed: 3 });
    expect(result.steps).toHaveLength(3);
    for (const step of result.steps) {
      const total = step.distribution.reduce((a, b) => a + b, 0);
      expect(total).toBeCloseTo(1, 5);
      expect(step.probability).toBe(step.distribution[step.tokenId]);
      expect(step.distribution.filter((p) => p > 0)).toHaveLength(10);
    }
  });

  it('is reproducible for a seed and varies across seeds', () => {
    const run = (seed: number) =>
      generate(model, [5, 6], { maxNewTokens: 8, temperature: 1.5, seed }).tokenIds;
    expect(run(11)).toEqual(run(11));
    const others = [12, 13, 14, 15, 16].map(run);
    expect(others.some((ids) => ids.join() !== run(11).join())).toBe(true);
  });

  it('stops after the stop token', () => {
    const first = greedyByForward([1, 2, 3], 1)[3];
    const result = generate(model, [1, 2, 3], {
      maxNewTokens: 5,
      temperature: 1,
      seed: 0,
      greedy: true,
      stopTokenId: first,
    });
    expect(result.tokenIds).toEqual([1, 2, 3, first]);
    expect(result.stopReason).toBe('stopToken');
  });

  it('stops when the context is full and returns a sequence forward can still run', () => {
    const prompt = new Array<number>(14).fill(1);
    const result = generate(model, prompt, { maxNewTokens: 10, temperature: 1, seed: 0 });
    expect(result.tokenIds).toHaveLength(16);
    expect(result.steps).toHaveLength(2);
    expect(result.stopReason).toBe('contextFull');
    expect(() => forward(model, [...result.tokenIds])).not.toThrow();
  });

  it('returns the prompt unchanged when asked for no tokens', () => {
    const result = generate(model, [1, 2], { maxNewTokens: 0, temperature: 1, seed: 0 });
    expect(result.tokenIds).toEqual([1, 2]);
    expect(result.steps).toHaveLength(0);
    expect(result.stopReason).toBe('maxNewTokens');
  });

  it('rejects a negative or fractional token budget', () => {
    expect(() => generate(model, [1], { maxNewTokens: -1, temperature: 1, seed: 0 })).toThrow(
      RangeError,
    );
    expect(() => generate(model, [1], { maxNewTokens: 1.5, temperature: 1, seed: 0 })).toThrow(
      RangeError,
    );
  });
});
