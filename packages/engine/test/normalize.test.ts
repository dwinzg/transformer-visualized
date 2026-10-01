import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { ASCII_EQUIVALENTS, normalizeText, unsupportedCharacters } from '../src/normalize';

describe('normalizeText', () => {
  it('maps phone punctuation to the ASCII the model trained on', () => {
    expect(normalizeText('\u201cHi,\u201d she said \u2014 it\u2019s late\u2026')).toBe(
      '"Hi," she said - it\'s late...',
    );
    expect(normalizeText('a\u00a0b \u2013 \u2018c\u2019')).toBe("a b - 'c'");
  });

  it('leaves plain ASCII and edge whitespace alone', () => {
    expect(normalizeText(' Once upon a time ')).toBe(' Once upon a time ');
  });

  it('uses the same table as the Python training pipeline', () => {
    const source = readFileSync(
      fileURLToPath(new URL('../../../model/tv_model/data.py', import.meta.url)),
      'utf8',
    );
    const block = /_ASCII_EQUIVALENTS = str\.maketrans\(\s*\{([\s\S]*?)\}\s*\)/.exec(source);
    if (block === null) throw new Error('_ASCII_EQUIVALENTS not found in data.py');
    const python: Record<string, string> = {};
    for (const [, code, value] of block[1].matchAll(/"\\u([0-9a-f]{4})": ("[^"]*"|'[^']*')/g)) {
      python[String.fromCharCode(parseInt(code, 16))] = value.slice(1, -1);
    }
    // Count entry lines independently of the key regex above, so an entry written another way
    // (a different quote style, uppercase hex, a key that is not a "\uXXXX" escape) still makes
    // the parsed key count visibly wrong instead of silently being skipped.
    const entryLines = block[1]
      .split('\n')
      .filter((line) => line.includes(':') && line.trim() !== '');
    expect(Object.keys(python)).toHaveLength(entryLines.length);
    expect(ASCII_EQUIVALENTS).toEqual(python);
  });
});

describe('unsupportedCharacters', () => {
  it('lists what is still not ASCII after cleaning, once each, as whole characters', () => {
    expect(unsupportedCharacters('caf\u00e9 \u2019 \u00e9 \u{1F600}')).toEqual([
      '\u00e9',
      '\u{1F600}',
    ]);
    expect(unsupportedCharacters('It\u2019s fine.')).toEqual([]);
  });
});
