import { describe, expect, it } from 'vitest';
import { displayTokens, GUESS_DEPTH, GUESS_WIDTH, guessTree } from './demo-data';
import type { Guess } from './guess-tree';

describe('displayTokens', () => {
  it('splits text into tokens whose texts join back to the cleaned text', () => {
    const tokens = displayTokens('Transformer Visualized');
    expect(tokens.length).toBeGreaterThan(1);
    expect(tokens.map((t) => t.text).join('')).toBe('Transformer Visualized');
  });

  it('cleans phone punctuation first', () => {
    expect(
      displayTokens('it\u2019s')
        .map((t) => t.text)
        .join(''),
    ).toBe("it's");
  });
});

describe('guessTree', () => {
  const tree = guessTree('Once upon a time, there was a');

  it('stores the top guesses a fixed number of levels deep', () => {
    let level: Guess[] = tree.guesses;
    for (let depth = 0; depth < GUESS_DEPTH; depth++) {
      expect(level).toHaveLength(GUESS_WIDTH);
      level = level[0].next;
    }
    expect(level).toEqual([]);
  });

  it('orders guesses from most to least likely, with real probabilities', () => {
    const ps = tree.guesses.map((g) => g.p);
    expect([...ps].sort((a, b) => b - a)).toEqual(ps);
    expect(ps.every((p) => p > 0 && p <= 1)).toBe(true);
    expect(ps.reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(1 + 1e-6);
  });

  it('is deterministic', () => {
    expect(guessTree('Once upon a time, there was a')).toEqual(tree);
  });
});
