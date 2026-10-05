import type { DisplayToken } from './guess-tree';

/** The model's scores for the next token, worked out at build time. */
export interface NextScores {
  /** The sentence so far. */
  tokens: DisplayToken[];
  /** The best scores (logits), highest first. */
  top: { token: DisplayToken; logit: number }[];
  /** Every other token's score as [score, how many tokens have it], highest first. */
  rest: [logit: number, count: number][];
}

export type Keep = 'all' | 'greedy' | 'top-k' | 'top-p';
export const TOP_K = 5;
export const TOP_P = 0.9;

export interface Chances {
  /** The chance of each top token after temperature and the keep rule. 0 means it is out. */
  top: number[];
  /** The chance that the pick is one of the other tokens, and how many of them stay in. */
  other: number;
  otherCount: number;
}

/**
 * softmax(logits / temperature), then the keep rule, the same steps as the engine's
 * nextTokenDistribution. The other tokens are grouped by score, so top-p can still count them.
 */
export function chances({ top, rest }: NextScores, temperature: number, keep: Keep): Chances {
  const max = top[0].logit;
  const weight = (logit: number) => Math.exp((logit - max) / temperature);
  const topW = top.map((t) => weight(t.logit));
  const restW = rest.map(([logit]) => weight(logit));
  const total = topW.reduce((s, w) => s + w, 0) + rest.reduce((s, [, n], i) => s + n * restW[i], 0);
  const p = topW.map((w) => w / total);

  if (keep === 'all') {
    const otherCount = rest.reduce((s, [, n]) => s + n, 0);
    return { top: p, other: Math.max(0, 1 - p.reduce((s, x) => s + x, 0)), otherCount };
  }
  if (keep === 'greedy')
    return { top: p.map((_, i) => (i === 0 ? 1 : 0)), other: 0, otherCount: 0 };
  if (keep === 'top-k') {
    const kept = p.map((x, i) => (i < TOP_K ? x : 0));
    const sum = kept.reduce((s, x) => s + x, 0);
    return { top: kept.map((x) => x / sum), other: 0, otherCount: 0 };
  }

  // Top-p keeps the likeliest tokens until their total reaches p, always at least one.
  const kept: number[] = [];
  let sum = 0;
  for (const x of p) {
    kept.push(sum < TOP_P ? x : 0);
    if (sum < TOP_P) sum += x;
  }
  let other = 0;
  let otherCount = 0;
  for (let i = 0; i < rest.length && sum < TOP_P; i++) {
    const each = restW[i] / total;
    const n = Math.min(rest[i][1], Math.ceil((TOP_P - sum) / each));
    otherCount += n;
    other += n * each;
    sum += n * each;
  }
  return { top: kept.map((x) => x / sum), other: other / sum, otherCount };
}

/** Draws one pick: the index of a top token, or -1 for one of the other tokens. */
export function draw({ top, other }: Chances, random: number): number {
  let target = random * (top.reduce((s, x) => s + x, 0) + other);
  for (let i = 0; i < top.length; i++) {
    if (top[i] <= 0) continue;
    if (target < top[i]) return i;
    target -= top[i];
  }
  return other > 0 ? -1 : top.findLastIndex((x) => x > 0);
}
