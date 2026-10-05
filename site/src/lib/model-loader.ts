import { loadModel, normalizeText, Tokenizer, type Model } from '@transformer-visualized/engine';
import modelUrl from '../../../models/tiny/model.safetensors?url';
import tokenizerUrl from '../../../models/tiny/tokenizer.json?url';
import type { DisplayToken } from './guess-tree';

/**
 * Browser only. One download of each file for the whole page, started by the first figure that
 * needs it. A failed download is forgotten, so the next call tries again.
 */
function once<T>(load: () => Promise<T>): () => Promise<T> {
  let loading: Promise<T> | undefined;
  return () =>
    (loading ??= load().catch((error: unknown) => {
      loading = undefined;
      throw error;
    }));
}

async function fetchOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response;
}

export const loadTokenizer = once(async () =>
  Tokenizer.fromJSON(await (await fetchOk(tokenizerUrl)).json()),
);

export const loadTinyModel = once(async () =>
  loadModel(await (await fetchOk(modelUrl)).arrayBuffer()),
);

export type { Model };

const END_OF_STORY = '<|endoftext|>';

/** One token as the figures show it. The end-of-story marker gets a readable name. */
export function displayToken(tokenizer: Tokenizer, id: number): DisplayToken {
  return id === tokenizer.specialTokenId(END_OF_STORY)
    ? { id, text: '[end of story]', special: true }
    : { id, text: tokenizer.tokenText(id) };
}

/** A text's tokens, cleaned the same way as the model's training text. */
export function toTokens(tokenizer: Tokenizer, text: string): DisplayToken[] {
  return tokenizer.encode(normalizeText(text)).map((id) => displayToken(tokenizer, id));
}
