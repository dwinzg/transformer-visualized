import { describe, expect, it } from 'vitest';
import {
  chosenTokens,
  describeGuesses,
  graft,
  guessesAt,
  percent,
  sentenceDone,
  type GuessTree,
} from './guess-tree';

const leaf = (id: number, text: string, p: number) => ({ token: { id, text }, p, next: [] });
const tree: GuessTree = {
  prompt: 'a',
  tokens: [{ id: 1, text: 'a' }],
  guesses: [
    { token: { id: 2, text: ' b' }, p: 0.6, next: [leaf(4, ' d', 0.9)] },
    { token: { id: 3, text: ' c' }, p: 0.3, next: [leaf(5, ' e', 0.5)] },
  ],
};

describe('sentence ends and grafts', () => {
  it('ends a sentence at a full stop, question or exclamation mark, or the end of the story', () => {
    const t = (text: string, special?: boolean) => [{ id: 0, text, special }];
    expect(sentenceDone(t(' Tim.'))).toBe(true);
    expect(sentenceDone(t('?"'))).toBe(true);
    expect(sentenceDone(t('[end of story]', true))).toBe(true);
    expect(sentenceDone(t(' Tim'))).toBe(false);
    expect(sentenceDone(t(','))).toBe(false);
    expect(sentenceDone([])).toBe(false);
  });

  it('adds guesses after a path without changing the original tree', () => {
    const grown = graft(tree, [0, 0], [leaf(9, ' f', 0.4)]);
    expect(guessesAt(grown, [0, 0]).map((g) => g.token.text)).toEqual([' f']);
    expect(guessesAt(tree, [0, 0])).toEqual([]);
    expect(guessesAt(grown, [1])).toBe(guessesAt(tree, [1]));
  });
});

describe('guess tree helpers', () => {
  it('finds the guesses after a path of choices', () => {
    expect(guessesAt(tree, []).map((g) => g.token.text)).toEqual([' b', ' c']);
    expect(guessesAt(tree, [1]).map((g) => g.token.text)).toEqual([' e']);
  });

  it('returns no guesses past the end of the tree or for a bad index', () => {
    expect(guessesAt(tree, [0, 0])).toEqual([]);
    expect(guessesAt(tree, [7])).toEqual([]);
  });

  it('lists the prompt tokens followed by the chosen ones', () => {
    expect(chosenTokens(tree, [1, 0]).map((t) => t.text)).toEqual(['a', ' c', ' e']);
  });

  it('describes guesses in plain words', () => {
    expect(describeGuesses(guessesAt(tree, []))).toBe('b 60 percent, c 30 percent');
  });

  it('formats percentages', () => {
    expect(percent(0.344)).toBe('34%');
    expect(percent(0.004)).toBe('<1%');
  });

  it('reads a special guess naturally, without its brackets', () => {
    const special = {
      token: { id: 6, text: '[end of story]', special: true },
      p: 0.05,
      next: [],
    };
    expect(describeGuesses([special])).toBe('end of story 5 percent');
  });
});
