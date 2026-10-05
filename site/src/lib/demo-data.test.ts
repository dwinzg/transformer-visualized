import { describe, expect, it } from 'vitest';
import {
  embeddedTokens,
  GUESS_DEPTH,
  GUESS_WIDTH,
  guessDisplayToken,
  guessTree,
  nearestTokens,
  NEIGHBOR_COUNT,
  positionSimilarity,
  tokenize,
} from './demo-data';
import { dot } from './embeddings';
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

// The Embeddings chapter quotes these numbers, so a retrained model has to update the text too.
describe('embeddings, as the Embeddings chapter quotes them', () => {
  const sentence = embeddedTokens('Lily wanted to play with her');
  it('gives each token a token row, a position row and their sum, 128 numbers each', () => {
    expect(sentence).toHaveLength(6);
    for (const t of sentence) {
      expect(t.tokenRow).toHaveLength(128);
      expect(t.positionRow).toHaveLength(128);
      t.sum.forEach((v, i) => expect(v).toBeCloseTo(t.tokenRow[i] + t.positionRow[i], 2));
    }
    expect(sentence[0].tokenRow.slice(0, 5)).toEqual([0.008, 0.006, -0.046, 0.001, -0.209]);
  });
  it('lists the nearest tokens, most similar first', () => {
    const near = nearestTokens([' girl', ' happy', ' three', ' Lily']);
    for (const { neighbors } of near) {
      expect(neighbors).toHaveLength(NEIGHBOR_COUNT);
      const scores = neighbors.map((n) => n.score);
      expect(scores).toEqual([...scores].sort((a, b) => b - a));
    }
    const score = (word: number, text: string) =>
      near[word].neighbors.find((n) => n.token.text === text)?.score;
    expect(score(0, ' boy')).toBe(0.74);
    expect(near[1].neighbors[0].token.text).toBe(' glad');
    expect(score(1, ' glad')).toBe(0.61);
    expect(score(1, ' sad')).toBe(0.49);
    expect(near[1].neighbors.some((n) => n.token.text === ' tree')).toBe(false);
    expect(score(2, ' 3')).toBe(0.88);
    expect(score(3, 'Lily')).toBe(0.76);
  });
  it('refuses a word that is more than one token', () => {
    expect(() => nearestTokens(['Tokenization'])).toThrow(/not one/);
  });
  it('makes nearby places alike and far places not', () => {
    expect(positionSimilarity(1, 2)).toBe(0.98);
    expect(positionSimilarity(1, 50)).toBe(-0.22);
  });
  it('matches the worked dot products', () => {
    expect(dot([1, 2, 3], [4, 0, -1])).toBe(1);
    expect(dot([2, -1, 3], [1, 4, 2])).toBe(4);
  });
});
