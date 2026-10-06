import { describe, expect, it } from 'vitest';
import { nextTokenDistribution, softmax } from '@transformer-visualized/engine';
import { nextScores, tiedScores } from './demo-data';
import { chances, draw, explainChance, TOP_K, TOP_P, type Keep } from './prediction';

const SENTENCE = 'Lily wanted to play with her';
const data = nextScores(SENTENCE);
// Every score as one flat list, best first, so the engine can check the figure's math.
const logits = [
  ...data.top.map((t) => t.logit),
  ...data.rest.flatMap(([logit, n]) => Array<number>(n).fill(logit)),
];

describe('nextScores', () => {
  it('has the best 50 scores and groups the other 4,046', () => {
    expect(data.top).toHaveLength(50);
    expect(logits).toHaveLength(4096);
    expect(logits).toEqual([...logits].sort((a, b) => b - a));
    expect(data.top.slice(0, 4).map((t) => t.token.text)).toEqual([
      ' ball',
      ' friends',
      ' toy',
      ' doll',
    ]);
  });
});

describe('chances', () => {
  const options: Record<Exclude<Keep, 'greedy'>, object> = {
    all: {},
    'top-k': { topK: TOP_K },
    'top-p': { topP: TOP_P },
  };
  for (const temperature of [0.1, 0.5, 1, 2]) {
    for (const keep of Object.keys(options) as Exclude<Keep, 'greedy'>[]) {
      it(`matches the engine at temperature ${temperature}, keeping ${keep}`, () => {
        const want = nextTokenDistribution(logits, { temperature, ...options[keep] });
        const got = chances(data, temperature, keep);
        got.top.forEach((p, i) => expect(p).toBeCloseTo(want[i], 5));
        const rest = Array.from(want.slice(50));
        expect(got.other).toBeCloseTo(
          rest.reduce((s, x) => s + x, 0),
          5,
        );
        // In 'all', every token stays in, even ones whose chance is too small for a float.
        expect(got.otherCount).toBe(keep === 'all' ? 4046 : rest.filter((x) => x > 0).length);
      });
    }
  }
  it('gives the numbers the chapter quotes', () => {
    const at1 = chances(data, 1, 'all');
    expect(at1.top[0].toFixed(3)).toBe('0.115');
    expect(at1.top[1].toFixed(3)).toBe('0.102');
    expect(chances(data, 0.1, 'all').top[0].toFixed(2)).toBe('0.72');
    expect(chances(data, 0.1, 'all').top[1].toFixed(2)).toBe('0.22');
    expect(at1.top[2].toFixed(3)).toBe('0.085');
    expect(at1.top[7].toFixed(3)).toBe('0.033');
    expect(at1.top.reduce((s, p) => s + p, 0).toFixed(2)).toBe('0.80');
    expect(chances(data, 0.5, 'all').top[0].toFixed(2)).toBe('0.27');
    expect(chances(data, 0.5, 'top-p').top.filter((p) => p > 0)).toHaveLength(7);
    const at2 = chances(data, 2, 'all');
    expect(at2.top[0].toFixed(3)).toBe('0.015');
    expect(at2.top.reduce((s, p) => s + p, 0).toFixed(2)).toBe('0.22');
    expect(data.top[7].token.text).toBe(' mom');
    // The real model keeps 200. Grouping the other scores to 2 decimals can move that by one or two.
    expect(Math.abs(chances(data, 1, 'top-p').otherCount + 50 - 200)).toBeLessThanOrEqual(2);
  });
  it('greedy keeps only the top token', () => {
    const { top, other } = chances(data, 1, 'greedy');
    expect(top[0]).toBe(1);
    expect(top.slice(1).every((p) => p === 0)).toBe(true);
    expect(other).toBe(0);
  });
  it('softmax of the tied scores matches', () => {
    expect(softmax([2 / 0.5, 0])[0]).toBeCloseTo(0.982, 3);
  });
});

describe('draw', () => {
  const c = { top: [0.5, 0, 0.25], other: 0.25, otherCount: 3 };
  it('walks the chances in order and skips tokens that are out', () => {
    expect(draw(c, 0)).toBe(0);
    expect(draw(c, 0.6)).toBe(2);
    expect(draw(c, 0.9)).toBe(-1);
    expect(draw({ top: [0.5, 0.5], other: 0, otherCount: 0 }, 0.999999)).toBe(1);
    expect(draw({ top: [0.5, 0.5, 0], other: 0, otherCount: 0 }, 0.9999999)).toBe(1);
    expect(draw({ top: [0, 0], other: 1, otherCount: 9 }, 0.5)).toBe(-1);
  });
});

describe('tiedScores', () => {
  it('scores each top token with a dot product that equals the model score', () => {
    const tied = tiedScores(SENTENCE);
    expect(tied.hidden).toHaveLength(128);
    expect(tied.rows.map((r) => r.token.text)).toEqual(
      data.top.slice(0, 5).map((t) => t.token.text),
    );
    tied.rows.forEach((r, i) => expect(r.score).toBeCloseTo(data.top[i].logit, 2));
    expect(tied.rows[0].score).toBe(2.6);
    expect(tied.rows[1].score).toBe(2.48);
    expect(tied.hidden.slice(0, 3).map((v) => v.toFixed(2))).toEqual(['0.82', '2.46', '-3.15']);
    expect(tied.last.text).toBe(' her');
  });
});

describe('explainChance', () => {
  const data = nextScores('Lily wanted to play with her');

  it.each([0.5, 1, 2])('gives the same chance as the bars at temperature %s', (t) => {
    const all = chances(data, t, 'all');
    for (const i of [0, 1, 4]) expect(explainChance(data, i, t).chance).toBeCloseTo(all.top[i], 10);
  });

  it('counts every token in the vocabulary, and gives the top one e to the 0', () => {
    const math = explainChance(data, 0, 1);
    expect(math.count).toBe(4096);
    expect(math.exp).toBe(1);
    expect(math.scaled - math.max).toBe(0);
  });
});
