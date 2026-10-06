import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ENCODE_CACHE_LIMIT, Tokenizer } from '../src/tokenizer';
import { readFixtureJson } from './helpers/fixtures';

const tokenizerPath = fileURLToPath(
  new URL('../../../models/tiny/tokenizer.json', import.meta.url),
);
const tokenizer = Tokenizer.fromJSON(JSON.parse(readFileSync(tokenizerPath, 'utf8')));
const cases = readFixtureJson<{ text: string; ids: number[] }[]>('tokenizer/cases.json');

describe('the tiny tokenizer matches the Python tokenizer', () => {
  it('has the full vocabulary', () => {
    expect(tokenizer.vocabSize).toBe(4096);
    expect(cases.length).toBeGreaterThanOrEqual(200);
  });

  it.each(cases.map((c, i) => ({ ...c, i })))('case $i', ({ text, ids }) => {
    expect(tokenizer.encode(text)).toEqual(ids);
    expect(tokenizer.decode(ids)).toBe(text);
  });
});

// Encoded by Python's tokenizers library. Unusual whitespace such as U+FEFF and U+0085 next to
// punctuation, plus typed special tokens, which both sides read as the special token.
const whitespace = readFixtureJson<{ text: string; ids: number[] }[]>('tokenizer/whitespace.json');

describe('the encode cache', () => {
  it('stays under its limit, and encodes the same after clearing', () => {
    const fresh = Tokenizer.fromJSON(JSON.parse(readFileSync(tokenizerPath, 'utf8')));
    const before = fresh.encode('Lily wanted to play with her ball');
    for (let i = 0; i < ENCODE_CACHE_LIMIT + 50; i++) fresh.encode(` w${i}`);
    expect(fresh['cache'].size).toBeLessThanOrEqual(ENCODE_CACHE_LIMIT);
    expect(fresh.encode('Lily wanted to play with her ball')).toEqual(before);
  });
});

describe('unusual whitespace and typed special tokens split the same way as in Python', () => {
  it.each(whitespace.map((c, i) => ({ ...c, i })))('case $i', ({ text, ids }) => {
    expect(tokenizer.encode(text)).toEqual(ids);
    expect(tokenizer.decode(ids)).toBe(text);
  });
});
