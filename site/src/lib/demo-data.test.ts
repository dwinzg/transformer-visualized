import { describe, expect, it } from 'vitest';
import { GUESS_DEPTH, GUESS_WIDTH, guessDisplayToken, guessTree, tokenize } from './demo-data';
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

// The Tokens chapter quotes these numbers, so a retrained tokenizer has to update the text too.
describe('tokenize, as the Tokens chapter quotes it', () => {
  const pieces = (text: string) => tokenize(text).map((t) => `${t.text}:${t.id}`);
  it('keeps every word of the shared sentence whole', () => {
    expect(pieces('Lily wanted to play with her')).toEqual([
      'Lily:665',
      ' wanted:408',
      ' to:266',
      ' play:324',
      ' with:329',
      ' her:336',
    ]);
  });
  it('splits rare words and tells spaces and capitals apart', () => {
    expect(tokenize('Tokenization is fun!').map((t) => t.text)).toEqual([
      'To',
      'ken',
      'iz',
      'ation',
      ' is',
      ' fun',
      '!',
    ]);
    expect(pieces('Lily was happy. Happy Lily!')).toEqual([
      'Lily:665',
      ' was:283',
      ' happy:376',
      '.:14',
      ' H:308',
      'appy:3189',
      ' Lily:405',
      '!:1',
    ]);
    expect(tokenize('The little dog ran to the park.')).toHaveLength(8);
    expect(tokenize(' played')).toHaveLength(1);
    expect(tokenize(' plays').map((t) => t.text)).toEqual([' play', 's']);
    expect(tokenize('123')).toHaveLength(3);
  });
  it('refuses text the model was never trained on', () => {
    expect(() => tokenize('café')).toThrow(/never trained on/);
  });
});
