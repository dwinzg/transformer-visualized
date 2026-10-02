import { describe, expect, it } from 'vitest';
import { GUESS_DEPTH, GUESS_WIDTH, guessDisplayToken, guessTree } from './demo-data';
import type { Guess } from './guess-tree';

const PROMPTS = [
  'Once upon a time, there was a',
  'The cat sat on the',
  'Lily wanted to play with her',
];

function allTokens(guesses: readonly Guess[]): Guess[] {
  return guesses.flatMap((g) => [g, ...allTokens(g.next)]);
}

describe('guessTree prompt cleaning', () => {
  it('cleans phone punctuation first', () => {
    expect(
      guessTree('it\u2019s')
        .tokens.map((t) => t.text)
        .join(''),
    ).toBe("it's");
  });

  it('refuses a prompt with characters the model was never trained on', () => {
    expect(() => guessTree('caf\u00e9')).toThrow(/\u00e9/);
  });
});

describe('guessDisplayToken', () => {
  it('labels a lone byte that is not valid text on its own as <0xNN>, not U+FFFD', () => {
    let found: ReturnType<typeof guessDisplayToken> | undefined;
    for (let id = 0; id < 4096 && found === undefined; id++) {
      const candidate = guessDisplayToken(id);
      if (/^<0x[0-9A-F]{2}>$/.test(candidate.text)) found = candidate;
    }
    expect(found).toBeDefined();
    expect(found?.text).not.toContain('\ufffd');
  });
});

describe('guessTree', () => {
  const tree = guessTree('Once upon a time, there was a');

  it('stores the top guesses a fixed number of levels deep', () => {
    // A special guess (the end of the story) has no children, so this walk follows the
    // first non-special guess at each level rather than always level[0].
    let level: Guess[] = tree.guesses;
    for (let depth = 0; depth < GUESS_DEPTH; depth++) {
      expect(level).toHaveLength(GUESS_WIDTH);
      const nonSpecial = level.find((g) => !g.token.special);
      if (nonSpecial === undefined) return;
      level = nonSpecial.next;
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

  it('never shows the raw special-token text and ends a special guess immediately', () => {
    for (const prompt of PROMPTS) {
      const guesses = allTokens(guessTree(prompt).guesses);
      for (const g of guesses) {
        expect(g.token.text).not.toContain('<|');
        if (g.token.special) expect(g.next).toEqual([]);
      }
    }
  });
});
