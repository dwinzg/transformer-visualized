import { describe, expect, it } from 'vitest';
import { bytesToUnicode, Tokenizer, type TokenizerJson } from '../src/tokenizer';

const table = bytesToUnicode();
const b = (text: string) =>
  Array.from(new TextEncoder().encode(text), (byte) => table[byte]).join('');

/** A tiny vocabulary: every byte, three merges, and one special token. */
function tinyJson(): TokenizerJson {
  const vocab: Record<string, number> = { '<|endoftext|>': 0 };
  table.forEach((char, i) => {
    vocab[char] = i + 1;
  });
  const merges: [string, string][] = [
    [b('h'), b('i')],
    [b(' '), b('hi')],
    [b('a'), b('a')],
  ];
  merges.forEach(([left, right], i) => {
    vocab[left + right] = 257 + i;
  });
  return { type: 'gpt2-byte-bpe', vocab, merges, specialTokens: { '<|endoftext|>': 0 } };
}

describe('bytesToUnicode', () => {
  it('maps 256 bytes to 256 distinct single characters', () => {
    expect(table).toHaveLength(256);
    expect(new Set(table).size).toBe(256);
    expect(table.every((char) => char.length === 1)).toBe(true);
  });

  it('keeps printable ASCII as itself and maps space to Ġ', () => {
    expect(table['A'.charCodeAt(0)]).toBe('A');
    expect(table['~'.charCodeAt(0)]).toBe('~');
    expect(table[32]).toBe('Ġ');
    expect(table[10]).toBe('Ċ');
  });
});

describe('Tokenizer', () => {
  const tokenizer = Tokenizer.fromJSON(tinyJson());

  it('applies merges by rank within each pre-token', () => {
    expect(tokenizer.encode('hi')).toEqual([257]);
    expect(tokenizer.encode(' hi')).toEqual([258]);
    expect(tokenizer.encode('hi hi')).toEqual([257, 258]);
  });

  it('merges repeated pairs left to right', () => {
    // "aaa" -> "aa" + "a", because the leftmost pair merges first
    expect(tokenizer.encode('aaa')).toEqual([259, 'a'.charCodeAt(0) + 1]);
  });

  it('matches special tokens in the raw text', () => {
    expect(tokenizer.encode('hi<|endoftext|>hi')).toEqual([257, 0, 257]);
    expect(tokenizer.specialTokenId('<|endoftext|>')).toBe(0);
  });

  it('falls back to single bytes, including for multi-byte characters', () => {
    const ids = tokenizer.encode('é');
    expect(ids).toEqual([0xc3 + 1, 0xa9 + 1]);
    expect(tokenizer.decode(ids)).toBe('é');
  });

  it('decodes back to the exact text', () => {
    for (const text of [
      'hi hi',
      '  spaced  out ',
      'line\nbreak',
      'emoji 😀',
      'hi<|endoftext|>hi',
    ]) {
      expect(tokenizer.decode(tokenizer.encode(text))).toBe(text);
    }
  });

  it('shows the pieces of a split character as byte labels', () => {
    expect(tokenizer.tokenText(0xc3 + 1)).toBe('<0xC3>');
    expect(tokenizer.tokenText(258)).toBe(' hi');
    expect(tokenizer.tokenText(0)).toBe('<|endoftext|>');
    expect(Array.from(tokenizer.tokenBytes(258))).toEqual([32, 104, 105]);
  });

  it('decodes invalid UTF-8 with the replacement character', () => {
    expect(tokenizer.decode([0xc3 + 1])).toBe('�');
  });

  it('rejects unknown ids and malformed JSON', () => {
    expect(() => tokenizer.decode([9999])).toThrow(RangeError);
    expect(() => Tokenizer.fromJSON({ type: 'other' })).toThrow(/gpt2-byte-bpe/);
    const missingByte = tinyJson();
    delete missingByte.vocab[b('z')];
    expect(() => Tokenizer.fromJSON(missingByte)).toThrow(/byte/);
    const badMerge = tinyJson();
    badMerge.merges.push([b('x'), b('y')]);
    expect(() => Tokenizer.fromJSON(badMerge)).toThrow(/merge/);
  });

  it('reports the vocabulary size', () => {
    expect(tokenizer.vocabSize).toBe(260);
  });
});
