import { forward, probabilities, rowView } from '@transformer-visualized/engine';
import type { Guess } from './guess-tree';
import { displayToken, loadTinyModel, loadTokenizer } from './model-loader';

const WIDTH = 5;

/** The model's top guesses after some tokens, worked out in the browser. */
export async function liveGuesses(ids: number[]): Promise<Guess[]> {
  const [model, tokenizer] = await Promise.all([loadTinyModel(), loadTokenizer()]);
  const trace = forward(model, ids);
  const probs = probabilities(rowView(trace.logits, ids.length - 1), 1);
  return Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a] || a - b)
    .slice(0, WIDTH)
    .map((id) => ({ token: displayToken(tokenizer, id), p: probs[id], next: [] }));
}
