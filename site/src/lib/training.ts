/** One saved moment of the training run, as written by model/tv_model/training_views.py. */
export interface Snapshot {
  step: number;
  valLoss: number;
  next: { id: number; text: string; p: number }[];
  story: string;
  recall: number;
  copy: number;
  /** [layer][head], how strongly each head looks from a repeated token to what followed it before. */
  induction: number[][];
}

export interface TrainingData {
  prompts: {
    next: string;
    story: string;
    recall: { text: string; answer: string };
    copy: { text: string; answer: string };
  };
  tokensPerStep: number;
  snapshots: Snapshot[];
}

/** The loss of guessing every one of the vocabulary's tokens with the same chance. */
export const chanceLoss = (vocabSize: number) => Math.log(vocabSize);

/** "8.2 million", "205 million", "12,800" and so on, for the tokens read so far. */
export function tokenCount(n: number): string {
  if (n >= 1e6) {
    const m = n / 1e6;
    return `${m >= 100 ? Math.round(m) : Number(m.toPrecision(2))} million`;
  }
  return n.toLocaleString('en-US');
}

/**
 * Where a step sits on a chart whose x axis is spaced by the log of the step, from 0 to 1. Early
 * training changes fastest, so a log axis gives the first few hundred steps room to show.
 */
export function logX(step: number, last: number): number {
  return Math.log10(step + 1) / Math.log10(last + 1);
}

/** The head with the highest induction score, counting layers and heads from 0. */
export function strongestHead(scores: number[][]): { layer: number; head: number; score: number } {
  let best = { layer: 0, head: 0, score: -Infinity };
  scores.forEach((row, layer) =>
    row.forEach((score, head) => {
      if (score > best.score) best = { layer, head, score };
    }),
  );
  return best;
}
