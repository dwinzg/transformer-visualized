import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  forward,
  loadModel,
  normalizeText,
  probabilities,
  rowView,
  Tokenizer,
  unsupportedCharacters,
  type Model,
} from '@transformer-visualized/engine';
import { cosine, round, type EmbeddedToken, type NearestTokens } from './embeddings';
import type { DisplayToken, Guess, GuessTree } from './guess-tree';

/**
 * Build-time only. Runs the real tiny model in Node so pages can show its numbers without
 * downloading it. Every figure and the home demo get their data from here.
 */

export const GUESS_WIDTH = 5;
export const GUESS_DEPTH = 3;

// Astro, Vitest and Playwright's build all run with the site workspace as the working directory.
const MODEL_DIR = resolve(process.cwd(), '../models/tiny');

function modelFile(name: string): Uint8Array {
  const path = resolve(MODEL_DIR, name);
  if (!existsSync(path)) {
    throw new Error(`demo-data: ${path} is missing. The site needs the tiny model to build.`);
  }
  return readFileSync(path);
}

let tokenizer: Tokenizer | undefined;
let model: Model | undefined;

function getTokenizer(): Tokenizer {
  tokenizer ??= Tokenizer.fromJSON(
    JSON.parse(new TextDecoder().decode(modelFile('tokenizer.json'))),
  );
  return tokenizer;
}

function getModel(): Model {
  if (model === undefined) {
    const bytes = modelFile('model.safetensors');
    const buffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(buffer).set(bytes);
    model = loadModel(buffer);
  }
  return model;
}

function assertSupported(text: string): void {
  const bad = unsupportedCharacters(text);
  if (bad.length > 0) {
    throw new Error(
      `demo-data: the model was never trained on ${bad.join(' ')} in "${text}". Every site prompt must be text the tiny model can read.`,
    );
  }
}

// Prompt tokens are always plain ASCII, checked by assertSupported before encoding, so decode
// and tokenText agree. Use decode: it joins whole words back together the way the page shows them.
function toDisplay(ids: readonly number[]): DisplayToken[] {
  const tok = getTokenizer();
  return ids.map((id) => ({ id, text: tok.decode([id]) }));
}

const END_OF_STORY = '<|endoftext|>';

/**
 * Builds the DisplayToken for one of the model's own guesses, which (unlike a prompt token)
 * can in principle be any byte token or the end-of-story marker. Exported so tests can check
 * it directly without forcing the model to predict a particular id.
 */
export function guessDisplayToken(id: number): DisplayToken {
  const tok = getTokenizer();
  if (tok.specialTokenId(END_OF_STORY) === id) {
    return { id, text: '[end of story]', special: true };
  }
  // tokenText falls back to a <0xNN> label when the token is part of a multi-byte character,
  // instead of decode's U+FFFD replacement character.
  return { id, text: tok.tokenText(id) };
}

function topGuesses(ids: readonly number[], depth: number): Guess[] {
  if (depth === 0) return [];
  const trace = forward(getModel(), ids);
  const probs = probabilities(rowView(trace.logits, trace.logits.rows - 1), 1);
  const ranked = Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a] || a - b)
    .slice(0, GUESS_WIDTH);
  return ranked.map((id) => {
    const token = guessDisplayToken(id);
    // Nothing is generated after the story ends, so a special guess has no children.
    return { token, p: probs[id], next: token.special ? [] : topGuesses([...ids, id], depth - 1) };
  });
}

const cache = new Map<string, GuessTree>();

export function guessTree(prompt: string): GuessTree {
  const cached = cache.get(prompt);
  if (cached) return cached;
  assertSupported(prompt);
  const ids = getTokenizer().encode(normalizeText(prompt));
  const tree = { prompt, tokens: toDisplay(ids), guesses: topGuesses(ids, GUESS_DEPTH) };
  cache.set(prompt, tree);
  return tree;
}

/** The tokens of a text as the live tokenizer shows them, for its first paint before it loads. */
export function tokenize(text: string): DisplayToken[] {
  assertSupported(text);
  const tok = getTokenizer();
  return tok.encode(normalizeText(text)).map((id) => ({ id, text: tok.tokenText(id) }));
}

/** Each token of a sentence with its token row, position row and their sum, for the number strip. */
export function embeddedTokens(text: string): EmbeddedToken[] {
  assertSupported(text);
  const tok = getTokenizer();
  const { wte, wpe } = getModel();
  return tok.encode(normalizeText(text)).map((id, position) => {
    const token = Array.from(rowView(wte, id));
    const place = Array.from(rowView(wpe, position));
    return {
      token: { id, text: tok.decode([id]) },
      tokenRow: token.map((v) => round(v, 3)),
      positionRow: place.map((v) => round(v, 3)),
      sum: token.map((v, i) => round(v + place[i], 3)),
    };
  });
}

export const NEIGHBOR_COUNT = 5;

/**
 * For each word, the tokens whose embedding rows point the most the same way (cosine similarity).
 * Byte tokens and the end-of-story marker are left out, since they are not words.
 */
export function nearestTokens(words: readonly string[]): NearestTokens[] {
  const tok = getTokenizer();
  const { wte } = getModel();
  const isWord = (id: number) =>
    !tok.tokenText(id).startsWith('<0x') && id !== tok.specialTokenId(END_OF_STORY);
  return words.map((word) => {
    assertSupported(word);
    const ids = tok.encode(normalizeText(word));
    if (ids.length !== 1) throw new Error(`demo-data: "${word}" is ${ids.length} tokens, not one`);
    const [id] = ids;
    const row = rowView(wte, id);
    const scored: { id: number; score: number }[] = [];
    for (let other = 0; other < wte.rows; other++) {
      if (other === id || !isWord(other)) continue;
      scored.push({ id: other, score: cosine(row, rowView(wte, other)) });
    }
    scored.sort((a, b) => b.score - a.score || a.id - b.id);
    return {
      word: { id, text: tok.decode([id]) },
      neighbors: scored.slice(0, NEIGHBOR_COUNT).map(({ id: n, score }) => ({
        token: { id: n, text: tok.decode([n]) },
        score: round(score, 2),
      })),
    };
  });
}

/** Cosine similarity of two position rows, rounded to 2 decimals. */
export function positionSimilarity(a: number, b: number): number {
  const { wpe } = getModel();
  return round(cosine(rowView(wpe, a), rowView(wpe, b)), 2);
}
