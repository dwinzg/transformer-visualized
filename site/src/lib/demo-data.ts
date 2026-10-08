import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  forward,
  forwardLlama,
  loadLlama,
  loadModel,
  normalizeText,
  probabilities,
  rowView,
  Tokenizer,
  unsupportedCharacters,
  type LlamaModel,
  type LlamaTrace,
  type Model,
  type Trace,
} from '@transformer-visualized/engine';
import { cosine, dot, round, type EmbeddedToken, type NearestTokens } from './embeddings';
import { attentionView, embeddingView, scoresView, type AttentionData } from './model-views';
import type { NextScores } from './prediction';
import {
  MAX_NEW_TOKENS,
  sentenceDone,
  type DisplayToken,
  type Guess,
  type GuessTree,
} from './guess-tree';

/**
 * Build-time only. Runs the real tiny model in Node so pages can show its numbers without
 * downloading it. Every figure and the home demo get their data from here.
 */

export const GUESS_WIDTH = 5;
export const GUESS_DEPTH = 3;

// Astro, Vitest and Playwright's build all run with the site workspace as the working directory.
const MODEL_DIR = resolve(process.cwd(), '../models/tiny');

function modelFile(name: string, dir = MODEL_DIR): Uint8Array {
  const path = resolve(dir, name);
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

/**
 * The model's top guesses, GUESS_DEPTH tokens deep. With finish, the path that always takes the
 * first guess goes on until the sentence ends, so the home demo's autoplay completes a sentence
 * without downloading the model.
 */
export function guessTree(prompt: string, { finish = false } = {}): GuessTree {
  const key = `${finish}:${prompt}`;
  const cached = cache.get(key);
  if (cached) return cached;
  assertSupported(prompt);
  const ids = getTokenizer().encode(normalizeText(prompt));
  const tree = { prompt, tokens: toDisplay(ids), guesses: topGuesses(ids, GUESS_DEPTH) };
  if (finish) {
    const chosen = [...ids];
    const tokens = toDisplay(ids);
    let level = tree.guesses;
    for (let n = 0; n < MAX_NEW_TOKENS && level.length > 0; n++) {
      const first = level[0];
      chosen.push(first.token.id);
      tokens.push(first.token);
      if (sentenceDone(tokens)) {
        first.next = [];
        break;
      }
      if (first.next.length === 0) first.next = topGuesses(chosen, 1);
      level = first.next;
    }
  }
  cache.set(key, tree);
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
  const ids = getTokenizer().encode(normalizeText(text));
  return embeddingView(toDisplay(ids), forward(getModel(), ids));
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

/** Attention weights for a sentence, as [layer][head][row][column]. */
export function attentionWeights(text: string): AttentionData {
  assertSupported(text);
  const ids = getTokenizer().encode(normalizeText(text));
  return attentionView(toDisplay(ids), forward(getModel(), ids));
}

let llama: LlamaModel | undefined;

/** The Llama-style model in models/llama-tiny. It shares the tiny model's tokenizer. */
function getLlama(): LlamaModel {
  if (llama === undefined) {
    const bytes = modelFile('model.safetensors', resolve(process.cwd(), '../models/llama-tiny'));
    const buffer = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(buffer).set(bytes);
    llama = loadLlama(buffer);
  }
  return llama;
}

/** Every value from one forward pass of the Llama-style model over a text. */
export function llamaTraceOf(text: string): LlamaTrace {
  assertSupported(text);
  return forwardLlama(getLlama(), getTokenizer().encode(normalizeText(text)));
}

/** Every value from one forward pass over a text. */
export function traceOf(text: string): Trace {
  assertSupported(text);
  return forward(getModel(), getTokenizer().encode(normalizeText(text)));
}

/** The model's scores for the token after a text. */
export function nextScores(text: string): NextScores {
  assertSupported(text);
  const ids = getTokenizer().encode(normalizeText(text));
  return scoresView(toDisplay(ids), forward(getModel(), ids), guessDisplayToken);
}

export interface TiedScores {
  /** The last token of the text. */
  last: DisplayToken;
  /** The last token's final list of numbers, after the last norm. */
  hidden: number[];
  /** The likeliest next tokens, each with its embedding row and its dot product with hidden. */
  rows: { token: DisplayToken; row: number[]; score: number }[];
}

/** The top next tokens' scores worked out by hand, as the dot product with each embedding row. */
export function tiedScores(text: string, count = 5): TiedScores {
  assertSupported(text);
  const ids = getTokenizer().encode(normalizeText(text));
  const { wte } = getModel();
  const trace = forward(getModel(), ids);
  const hidden = rowView(trace.lnFinal.out, ids.length - 1);
  const logits = rowView(trace.logits, ids.length - 1);
  const ranked = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a] || a - b);
  return {
    last: toDisplay(ids).at(-1)!,
    hidden: Array.from(hidden, (v) => round(v, 3)),
    rows: ranked.slice(0, count).map((id) => {
      const row = rowView(wte, id);
      return {
        token: guessDisplayToken(id),
        row: Array.from(row, (v) => round(v, 3)),
        score: round(dot(hidden, row), 2),
      };
    }),
  };
}

/** The model's five likeliest next tokens after each text, side by side. */
export function compareGuesses(
  texts: readonly string[],
): { tokens: DisplayToken[]; guesses: Guess[] }[] {
  return texts.map((text) => {
    assertSupported(text);
    const ids = getTokenizer().encode(normalizeText(text));
    return { tokens: toDisplay(ids), guesses: topGuesses(ids, 1) };
  });
}

export interface WindowedGuesses {
  /** Filler sentences between the note and the question. */
  gap: number;
  /** Tokens in the whole text. */
  total: number;
  /** Tokens the model reads, the last contextLength of them. */
  seen: number;
  /** The note's tokens, counted from the start of the text. */
  note: { from: number; to: number };
  guesses: Guess[];
}

/**
 * A note, then more and more filler, then a question. The model only ever reads its last
 * contextLength tokens, so once the note falls out of that window it cannot use it.
 */
export function contextWindow(note: string, filler: string, question: string, most: number) {
  const window = getModel().config.contextLength;
  const encode = (text: string) => {
    assertSupported(text);
    return getTokenizer().encode(normalizeText(text));
  };
  const noteIds = encode(note);
  const steps: WindowedGuesses[] = [];
  for (let gap = 0; gap <= most; gap++) {
    const ids = encode(note + filler.repeat(gap) + question);
    // The note encodes the same at the start of any text, so its tokens are the first ones.
    if (noteIds.some((id, i) => ids[i] !== id))
      throw new Error('demo-data: the note must come first');
    steps.push({
      gap,
      total: ids.length,
      seen: Math.min(ids.length, window),
      note: { from: 0, to: noteIds.length },
      guesses: topGuesses(ids.slice(-window), 1),
    });
  }
  return { window, steps };
}
