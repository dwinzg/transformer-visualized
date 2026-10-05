import { rowView, type Trace } from '@transformer-visualized/engine';
import { round, type EmbeddedToken } from './embeddings';
import type { DisplayToken } from './guess-tree';
import type { NextScores } from './prediction';

/**
 * The plain JSON each figure takes, cut from one forward pass. Used at build time for the chapters
 * and in the browser by the playground, so both show the same numbers.
 */

/** Each token with its token row, position row and their sum, rounded to 3 decimals. */
export function embeddingView(tokens: DisplayToken[], trace: Trace): EmbeddedToken[] {
  return tokens.map((token, i) => {
    const tokenRow = Array.from(rowView(trace.tokenEmbeddings, i));
    const positionRow = Array.from(rowView(trace.positionEmbeddings, i));
    return {
      token,
      tokenRow: tokenRow.map((v) => round(v, 3)),
      positionRow: positionRow.map((v) => round(v, 3)),
      sum: tokenRow.map((v, k) => round(v + positionRow[k], 3)),
    };
  });
}

/**
 * Attention weights as [layer][head][row][column]. Rounded once, to the 2 decimals the figure
 * shows, so the page and the chapter text always agree.
 */
export interface AttentionData {
  tokens: DisplayToken[];
  weights: number[][][][];
}

export function attentionView(tokens: DisplayToken[], trace: Trace): AttentionData {
  return {
    tokens,
    weights: trace.layers.map((layer) =>
      layer.heads.map((head) =>
        tokens.map((_, row) => Array.from(rowView(head.weights, row), (w) => round(w, 2))),
      ),
    ),
  };
}

const TOP_SCORES = 50;

/**
 * The scores for the token after the text. The best 50 keep 3 decimals, which matters at low
 * temperature. The rest are grouped by score at 2 decimals, which is enough for their small share.
 */
export function scoresView(
  tokens: DisplayToken[],
  trace: Trace,
  display: (id: number) => DisplayToken,
): NextScores {
  const logits = rowView(trace.logits, tokens.length - 1);
  const ranked = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a] || a - b);
  const groups = new Map<number, number>();
  for (const id of ranked.slice(TOP_SCORES)) {
    const logit = round(logits[id], 2);
    groups.set(logit, (groups.get(logit) ?? 0) + 1);
  }
  return {
    tokens,
    top: ranked
      .slice(0, TOP_SCORES)
      .map((id) => ({ token: display(id), logit: round(logits[id], 3) })),
    rest: [...groups].sort((a, b) => b[0] - a[0]),
  };
}
