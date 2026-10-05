import type { DisplayToken } from './guess-tree';

/** One token of a sentence with its three lists of numbers, rounded for display. */
export interface EmbeddedToken {
  token: DisplayToken;
  /** The token's row of the embedding table. */
  tokenRow: number[];
  /** The row for the token's position in the sentence. */
  positionRow: number[];
  /** What the first layer reads, the two rows added together. */
  sum: number[];
}

export interface Neighbor {
  token: DisplayToken;
  /** Cosine similarity, from -1 to 1, rounded to 2 decimals. */
  score: number;
}

export interface NearestTokens {
  word: DisplayToken;
  neighbors: Neighbor[];
}

export const round = (value: number, places: number) => {
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
};

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

export function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
}
