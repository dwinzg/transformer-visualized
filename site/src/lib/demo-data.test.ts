import { describe, expect, it } from 'vitest';
import {
  compareGuesses,
  contextWindow,
  attentionWeights,
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
import { softmax } from '@transformer-visualized/engine';
import { dot } from './embeddings';
import { chosenTokens, sentenceDone, type Guess } from './guess-tree';

const PROMPTS = [
  'Once upon a time, there was a',
  'The cat sat on the',
  'Lily wanted to play with her',
];

function allTokens(guesses: readonly Guess[]): Guess[] {
  return guesses.flatMap((g) => [g, ...allTokens(g.next)]);
}

describe('guessTree with finish', () => {
  it('follows the first guesses to the end of a sentence, and stops there', () => {
    for (const prompt of PROMPTS) {
      const tree = guessTree(prompt, { finish: true });
      const path: number[] = [];
      for (let level = tree.guesses; level.length > 0; level = level[0].next) path.push(0);
      const tokens = chosenTokens(tree, path);
      expect(sentenceDone(tokens), prompt).toBe(true);
    }
  });

  it('leaves the plain tree for chapter figures as it was', () => {
    const tree = guessTree(PROMPTS[1]);
    const path: number[] = [];
    for (let level = tree.guesses; level.length > 0; level = level[0].next) path.push(0);
    expect(path.length).toBeLessThanOrEqual(GUESS_DEPTH);
  });
});

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
    for (const { word, neighbors } of near) {
      for (const { token } of neighbors) {
        expect(token.id).not.toBe(word.id);
        expect(token.text).not.toMatch(/^<0x|<\|endoftext\|>/);
      }
    }
  });
  it('refuses a word that is more than one token', () => {
    expect(() => nearestTokens(['Tokenization'])).toThrow(/not one/);
  });
  it('makes nearby places alike and far places not', () => {
    // Rows count from 0 here, and the chapter counts places from 1, so row 1 is place 2.
    expect(positionSimilarity(1, 2)).toBe(0.98);
    expect(positionSimilarity(1, 50)).toBe(-0.22);
  });
  it('matches the worked dot products', () => {
    expect(dot([1, 2, 3], [4, 0, -1])).toBe(1);
    expect(dot([2, -1, 3], [1, 4, 2])).toBe(4);
  });
});

// The Attention chapter quotes these numbers, so a retrained model has to update the text too.
describe('attention weights, as the Attention chapter quotes them', () => {
  const { tokens, weights } = attentionWeights('Lily wanted to play with her');
  it('has 4 layers of 4 heads, each a 6 by 6 grid whose rows add up to 1 with no look ahead', () => {
    expect(tokens).toHaveLength(6);
    expect(weights).toHaveLength(4);
    for (const layer of weights) {
      expect(layer).toHaveLength(4);
      for (const head of layer)
        head.forEach((row, i) => {
          expect(Math.abs(row.reduce((a, b) => a + b) - 1)).toBeLessThan(0.03);
          row.forEach((w, j) => j > i && expect(w).toBe(0));
        });
    }
  });
  it('has the patterns the chapter points at', () => {
    const [her, lily] = [5, 0];
    expect(weights[2][2][her][lily]).toBe(0.66);
    for (const row of [1, 2, 3, 4, 5]) {
      const r = weights[2][2][row];
      expect(r.indexOf(Math.max(...r)), `row ${row}`).toBe(lily);
    }
    const previous = [1, 2, 3, 4, 5].filter((row) => {
      const r = weights[1][0][row];
      return r.indexOf(Math.max(...r)) === row - 1;
    });
    expect(previous.length).toBeGreaterThanOrEqual(3);
    // The same head puts the most on Ben in this sentence, so it is not working out who "her" is.
    const ben = attentionWeights('Ben wanted to play with her').weights[2][2][5];
    expect(ben.indexOf(Math.max(...ben))).toBe(0);
    expect(ben[0]).toBe(0.74);
  });
  it('matches the softmax example', () => {
    expect(Array.from(softmax([1, 0, 0]), (p) => Number(p.toFixed(3)))).toEqual([
      0.576, 0.212, 0.212,
    ]);
    expect(Math.exp(2) / (Math.exp(2) + 2)).toBeCloseTo(0.79, 2);
  });
});

describe('the applied chapters', () => {
  const NOTE = 'Sara had a cat named Fluffy.';
  const FILLER =
    ' The sun was bright and the birds sang in the trees. Sara played in the garden with her ball.';
  const QUESTION = ' Every morning, Sara gave milk to';

  it('uses a note in the prompt, and loses it once it falls out of the window', () => {
    const { window, steps } = contextWindow(NOTE, FILLER, QUESTION, 8);
    expect(window).toBe(128);
    const top = steps.map((s) => s.guesses[0].token.text);
    // Matches PyTorch: 0.832 with no filler, still high, then fading, then gone past 128 tokens.
    expect(steps[0].total).toBe(14);
    expect(steps[0].guesses[0].p).toBeCloseTo(0.832, 3);
    expect(top.slice(0, 6)).toEqual(Array(6).fill(' Fluffy'));
    expect(steps[5].total).toBe(119);
    expect(steps[6].total).toBe(140);
    expect(steps[6].seen).toBe(128);
    expect(top[6]).toBe(' her');
    expect(steps[6].guesses.some((g) => g.token.text === ' Fluffy')).toBe(false);
    // The numbers the chapter quotes.
    expect(Math.round(steps[5].guesses[0].p * 100)).toBe(43);
    expect(steps[6].guesses.slice(0, 2).map((g) => [g.token.text, Math.round(g.p * 100)])).toEqual([
      [' her', 44],
      [' the', 43],
    ]);
    // Whenever any token is cut off, the whole note is, so "inside" is never half true.
    for (const s of steps) {
      const cut = s.total - s.seen;
      expect(cut === 0 || cut >= s.note.to).toBe(true);
    }
  });

  // Every number the applied chapters quote. If the model is retrained, update the chapters too.
  it.each([
    ['The capital of France is', [' a', 12], [' not', 11]],
    [
      'Lily has 2 apples. Tom gives her 1 more. Now Lily has',
      [' a', 17],
      [' two', 14],
      [' enough', 7],
      [' three', 7],
    ],
    [
      'Lily has 2 apples. Tom gives her 1 more. 2 plus 1 is 3. Now Lily has',
      [' two', 14],
      [' a', 14],
      [' enough', 7],
      [' three', 7],
    ],
    [
      'The note said that the key was under the rug. Tom looked for the key under the',
      [' rug', 46],
      [' bed', 21],
    ],
    ['Tom looked for the key under the', [' tree', 30], [' bed', 25]],
    ['Ben said, "Zog." Mia said, "Zog." Tim said, "', ['Z', 20], ['I', 9]],
    ['Tim said, "', ['I', 20], ['No', 12]],
    ['Write a poem about a cat.', [' It', 13], [' The', 11], [' I', 9]],
  ] as const)('gives the quoted guesses after %j', (text, ...top) => {
    const [{ guesses }] = compareGuesses([text]);
    top.forEach(([token, percent], i) => {
      expect(guesses[i].token.text).toBe(token);
      expect(Math.round(guesses[i].p * 100)).toBe(percent);
    });
  });

  it.each([
    ['Every morning, Sara gave milk to', ' Fluffy'],
    ['Tom looked for the key under the', ' rug'],
    ['Tim said, "', 'Z'],
  ])('leaves the answer out of the top five after %j', (text, missing) => {
    const [{ guesses }] = compareGuesses([text]);
    expect(guesses.map((g) => g.token.text)).not.toContain(missing);
  });

  it('compares the guesses with and without the note', () => {
    const [withNote, without] = compareGuesses([NOTE + QUESTION, QUESTION.trim()]);
    expect(withNote.guesses[0].token.text).toBe(' Fluffy');
    expect(without.guesses.map((g) => g.token.text).slice(0, 2)).toEqual([' her', ' the']);
    expect(without.guesses[0].p).toBeCloseTo(0.39, 2);
  });
});
