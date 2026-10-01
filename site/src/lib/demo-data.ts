import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  forward,
  loadModel,
  normalizeText,
  probabilities,
  rowView,
  Tokenizer,
  type Model,
} from '@transformer-visualized/engine';
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

function toDisplay(ids: readonly number[]): DisplayToken[] {
  const tok = getTokenizer();
  return ids.map((id) => ({ id, text: tok.decode([id]) }));
}

export function displayTokens(text: string): DisplayToken[] {
  return toDisplay(getTokenizer().encode(normalizeText(text)));
}

function topGuesses(ids: readonly number[], depth: number): Guess[] {
  if (depth === 0) return [];
  const trace = forward(getModel(), ids);
  const probs = probabilities(rowView(trace.logits, trace.logits.rows - 1), 1);
  const ranked = Array.from(probs.keys())
    .sort((a, b) => probs[b] - probs[a] || a - b)
    .slice(0, GUESS_WIDTH);
  return ranked.map((id) => ({
    token: toDisplay([id])[0],
    p: probs[id],
    next: topGuesses([...ids, id], depth - 1),
  }));
}

const cache = new Map<string, GuessTree>();

export function guessTree(prompt: string): GuessTree {
  const cached = cache.get(prompt);
  if (cached) return cached;
  const ids = getTokenizer().encode(normalizeText(prompt));
  const tree = { prompt, tokens: toDisplay(ids), guesses: topGuesses(ids, GUESS_DEPTH) };
  cache.set(prompt, tree);
  return tree;
}
