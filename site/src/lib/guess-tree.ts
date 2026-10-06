/** A piece of text the model reads or writes, with its id in the model's vocabulary. */
export interface DisplayToken {
  id: number;
  text: string;
  /** True for a non-text token, such as the end-of-story marker, shown in square brackets. */
  special?: boolean;
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

/** The most tokens the home demo adds before it stops, in case a sentence never ends. */
export const MAX_NEW_TOKENS = 30;

/** True once the last token ends a sentence, or the story is over. */
export function sentenceDone(tokens: readonly DisplayToken[]): boolean {
  const last = tokens.at(-1);
  return !!last && (last.special === true || /[.!?]["']?$/.test(last.text.trim()));
}

/** A copy of the tree with new guesses after a path of choices. The tree itself is unchanged. */
export function graft(tree: GuessTree, path: readonly number[], guesses: Guess[]): GuessTree {
  const along = (level: Guess[], rest: readonly number[]): Guess[] => {
    if (rest.length === 0) return guesses;
    return level.map((g, i) => (i === rest[0] ? { ...g, next: along(g.next, rest.slice(1)) } : g));
  };
  return { ...tree, guesses: along(tree.guesses, path) };
}

export function percent(p: number): string {
  return p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`;
}

/** A token's text for speaking aloud: special tokens lose their brackets, others just trim. */
export function spoken(token: DisplayToken): string {
  return token.special ? token.text.replace(/^\[|\]$/g, '') : token.text.trim();
}

export function describeGuesses(guesses: readonly Guess[], count = 3): string {
  return guesses
    .slice(0, count)
    .map((g) => `${spoken(g.token)} ${percent(g.p).replace('%', ' percent')}`)
    .join(', ');
}
