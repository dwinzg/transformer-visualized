import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { Tokenizer } from '../src/tokenizer';
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
