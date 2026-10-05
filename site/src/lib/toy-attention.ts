/**
 * A hand-sized attention example for the Attention chapter: three tokens with two numbers each.
 * The numbers are made up to show the idea. A unit test checks the math against the engine.
 */
export const TOY_TOKENS = ['Lily', 'ran', 'she'] as const;

/** Each token's question card (query), name tag (key) and backpack (value). */
export const TOY_Q = [
  [1, 0],
  [0, 1],
  [2, 0],
];
export const TOY_K = [
  [2, 0],
  [0, 1],
  [1, 1],
];
export const TOY_V = [
  [1, 0],
  [0, 1],
  [0.5, 0.5],
];

export const TOY_STAGES = ['scores', 'scaled', 'masked', 'weights', 'output'] as const;
export type ToyStage = (typeof TOY_STAGES)[number];

export interface ToyAttention {
  scores: number[][];
  scaled: number[][];
  /** null where a token would look at a later token. */
  masked: (number | null)[][];
  weights: number[][];
  output: number[][];
}

const dot = (a: number[], b: number[]) => a.reduce((sum, x, i) => sum + x * b[i], 0);

export function toyAttention(q = TOY_Q, k = TOY_K, v = TOY_V): ToyAttention {
  const scale = 1 / Math.sqrt(q[0].length);
  const scores = q.map((qi) => k.map((kj) => dot(qi, kj)));
  const scaled = scores.map((row) => row.map((s) => s * scale));
  const masked = scaled.map((row, i) => row.map((s, j) => (j > i ? null : s)));
  const weights = masked.map((row) => {
    const seen = row.filter((s): s is number => s !== null);
    const max = Math.max(...seen);
    const exps = row.map((s) => (s === null ? 0 : Math.exp(s - max)));
    const total = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / total);
  });
  const output = weights.map((row) =>
    v[0].map((_, c) => row.reduce((sum, w, j) => sum + w * v[j][c], 0)),
  );
  return { scores, scaled, masked, weights, output };
}
