import { describe, expect, it } from 'vitest';
import { softmax } from '../src/ops';
import {
  argmax,
  createRng,
  nextTokenDistribution,
  probabilities,
  sample,
  topKFilter,
  topPFilter,
} from '../src/sampling';
import { expectAllClose } from './helpers/close';

const exact = { atol: 1e-6, rtol: 0 };
const probs = new Float32Array([0.1, 0.5, 0.2, 0.2]);

describe('probabilities', () => {
  it('equals softmax at temperature 1', () => {
    expectAllClose('p', probabilities([1, 2, 3], 1), softmax([1, 2, 3]), exact);
  });

  it('divides logits by the temperature', () => {
    expectAllClose('p', probabilities([1, 2], 0.5), [0.1192029, 0.8807971], exact);
  });

  it('rejects temperatures that are not positive and finite', () => {
    for (const t of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => probabilities([1, 2], t)).toThrow(RangeError);
    }
  });
});

describe('topKFilter', () => {
  it('keeps the k most likely tokens, breaking ties toward the lower id', () => {
    expectAllClose('p', topKFilter(probs, 2), [0, 0.5 / 0.7, 0.2 / 0.7, 0], exact);
  });

  it('keeps a single token when k is 1 and everything when k is large', () => {
    expect(Array.from(topKFilter(probs, 1))).toEqual([0, 1, 0, 0]);
    expectAllClose('p', topKFilter(probs, 10), probs, exact);
  });

  it('rejects k below 1 or not an integer', () => {
    expect(() => topKFilter(probs, 0)).toThrow(RangeError);
    expect(() => topKFilter(probs, 1.5)).toThrow(RangeError);
  });
});

describe('topPFilter', () => {
  it('keeps the most likely tokens until their total reaches p', () => {
    expectAllClose('p', topPFilter(probs, 0.6), [0, 0.5 / 0.7, 0.2 / 0.7, 0], exact);
  });

  it('always keeps at least the top token', () => {
    expect(Array.from(topPFilter(probs, 0.5))).toEqual([0, 1, 0, 0]);
    expect(Array.from(topPFilter(probs, 0.01))).toEqual([0, 1, 0, 0]);
  });

  it('keeps everything at p = 1', () => {
    expectAllClose('p', topPFilter(probs, 1), probs, exact);
  });

  it('rejects p outside (0, 1]', () => {
    expect(() => topPFilter(probs, 0)).toThrow(RangeError);
    expect(() => topPFilter(probs, 1.5)).toThrow(RangeError);
  });
});

describe('nextTokenDistribution', () => {
  it('applies temperature, then top-k, then top-p', () => {
    // Probabilities 0.5, 0.3 and 0.2. Top-k 2 leaves 0.625 and 0.375, so top-p 0.6 keeps one
    // token. Applying top-p first, or skipping top-k, would keep two.
    const logits = [Math.log(0.5), Math.log(0.3), Math.log(0.2)];
    const probs = nextTokenDistribution(logits, { temperature: 1, topK: 2, topP: 0.6 });
    expect(Array.from(probs)).toEqual([1, 0, 0]);
  });

  it('skips filters that are not set', () => {
    expectAllClose(
      'p',
      nextTokenDistribution([0, 1], { temperature: 1 }),
      probabilities([0, 1], 1),
      exact,
    );
  });
});

describe('argmax', () => {
  it('returns the index of the largest value, the first one on ties', () => {
    expect(argmax([1, 3, 2])).toBe(1);
    expect(argmax([5, 5, 1])).toBe(0);
  });

  it('rejects an empty list', () => {
    expect(() => argmax([])).toThrow(RangeError);
  });
});

describe('sample', () => {
  const dist = new Float32Array([0.2, 0.3, 0.5]);

  it('maps random numbers to tokens by cumulative probability', () => {
    expect(sample(dist, () => 0)).toBe(0);
    expect(sample(dist, () => 0.25)).toBe(1);
    expect(sample(dist, () => 0.99)).toBe(2);
  });

  it('never picks a zero-probability token', () => {
    expect(sample(new Float32Array([0, 1, 0]), () => 0)).toBe(1);
    expect(sample(new Float32Array([0, 1, 0]), () => 0.999999)).toBe(1);
  });

  it('draws tokens at their probabilities', () => {
    const random = createRng(7);
    const counts = [0, 0, 0];
    const draws = 20000;
    for (let i = 0; i < draws; i++) counts[sample(dist, random)] += 1;
    expectAllClose(
      'frequencies',
      counts.map((c) => c / draws),
      [0.2, 0.3, 0.5],
      { atol: 0.02, rtol: 0 },
    );
  });

  it('rejects distributions without positive mass', () => {
    expect(() => sample(new Float32Array([0, 0]), () => 0.5)).toThrow(RangeError);
  });
});

describe('createRng', () => {
  it('repeats the same sequence for the same seed', () => {
    const a = createRng(42);
    const b = createRng(42);
    const first = Array.from({ length: 5 }, a);
    expect(Array.from({ length: 5 }, b)).toEqual(first);
    expect(Array.from({ length: 5 }, createRng(43))).not.toEqual(first);
  });

  it('returns numbers in [0, 1) with mean near 0.5', () => {
    const random = createRng(1);
    let sum = 0;
    for (let i = 0; i < 10000; i++) {
      const x = random();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      sum += x;
    }
    expect(sum / 10000).toBeGreaterThan(0.48);
    expect(sum / 10000).toBeLessThan(0.52);
  });

  it('rejects seeds that are not safe integers', () => {
    expect(() => createRng(Number.NaN)).toThrow(RangeError);
    expect(() => createRng(1.5)).toThrow(RangeError);
    expect(() => createRng(2 ** 53)).toThrow(RangeError);
  });

  it('wraps a seed modulo 2^32, as documented', () => {
    const first = Array.from({ length: 5 }, createRng(5));
    expect(Array.from({ length: 5 }, createRng(5 + 2 ** 32))).toEqual(first);
  });
});
