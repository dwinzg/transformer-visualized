/** A piece of text the model reads or writes, with its id in the model's vocabulary. */
export interface DisplayToken {
  id: number;
  text: string;
}

/** One possible next token, its probability and the guesses that follow it. */
export interface Guess {
  token: DisplayToken;
  p: number;
  next: Guess[];
}

/** The model's top guesses for a prompt, a few tokens deep, computed at build time. */
export interface GuessTree {
  prompt: string;
  tokens: DisplayToken[];
  guesses: Guess[];
}

export function guessesAt(tree: GuessTree, path: readonly number[]): Guess[] {
  let level = tree.guesses;
  for (const index of path) {
    const guess = level[index];
    if (guess === undefined) return [];
    level = guess.next;
  }
  return level;
}

export function chosenTokens(tree: GuessTree, path: readonly number[]): DisplayToken[] {
  const tokens = [...tree.tokens];
  let level = tree.guesses;
  for (const index of path) {
    const guess = level[index];
    if (guess === undefined) break;
    tokens.push(guess.token);
    level = guess.next;
  }
  return tokens;
}

export function percent(p: number): string {
  return p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`;
}

export function describeGuesses(guesses: readonly Guess[], count = 3): string {
  return guesses
    .slice(0, count)
    .map((g) => `${g.token.text.trim()} ${percent(g.p).replace('%', ' percent')}`)
    .join(', ');
}
