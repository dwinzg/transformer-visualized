/** Tokenizer JSON written by model/tv_model/tokenizer.py. */
export interface TokenizerJson {
  type: 'gpt2-byte-bpe';
  vocab: Record<string, number>;
  merges: [string, string][];
  specialTokens: Record<string, number>;
}

// GPT-2's pre-tokenization pattern: contractions, letters, numbers, other symbols, whitespace.
const PRETOKENIZE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

let byteTable: readonly string[] | undefined;

/**
 * GPT-2's map from each byte to a printable character, so every byte sequence can be written as
 * text. Printable Latin-1 bytes map to themselves; the rest map to characters from U+0100 up.
 */
export function bytesToUnicode(): readonly string[] {
  if (byteTable) return byteTable;
  const table = new Array<string>(256);
  const printable = (byte: number) =>
    (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
  let next = 0;
  for (let byte = 0; byte < 256; byte++) {
    table[byte] = String.fromCharCode(printable(byte) ? byte : 256 + next++);
  }
  byteTable = Object.freeze(table);
  return byteTable;
}

function isTokenizerJson(value: unknown): value is TokenizerJson {
  if (typeof value !== 'object' || value === null) return false;
  const json = value as Record<string, unknown>;
  return (
    json.type === 'gpt2-byte-bpe' &&
    typeof json.vocab === 'object' &&
    json.vocab !== null &&
    Array.isArray(json.merges) &&
    typeof json.specialTokens === 'object' &&
    json.specialTokens !== null
  );
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export class Tokenizer {
  readonly vocabSize: number;
  private readonly vocab: Map<string, number>;
  private readonly tokens: string[];
  private readonly ranks: Map<string, number>;
  private readonly special: Map<string, number>;
  private readonly specialIds: Map<number, string>;
  private readonly specialPattern: RegExp | null;
  private readonly byteToChar: readonly string[];
  private readonly charToByte: Map<string, number>;
  private readonly cache = new Map<string, number[]>();
  private readonly decoder = new TextDecoder('utf-8', { fatal: false });
  private readonly strictDecoder = new TextDecoder('utf-8', { fatal: true });

  private constructor(json: TokenizerJson) {
    this.byteToChar = bytesToUnicode();
    this.charToByte = new Map(this.byteToChar.map((char, byte) => [char, byte]));
    this.vocab = new Map(Object.entries(json.vocab));
    // Byte-level BPE can encode any text only if every single byte is a token.
    for (const char of this.byteToChar) {
      if (!this.vocab.has(char))
        throw new Error(`Tokenizer vocab is missing the byte token "${char}"`);
    }
    this.vocabSize = this.vocab.size;
    this.tokens = new Array<string>(this.vocabSize);
    for (const [token, id] of this.vocab) {
      if (
        !Number.isInteger(id) ||
        id < 0 ||
        id >= this.vocabSize ||
        this.tokens[id] !== undefined
      ) {
        throw new Error(`Tokenizer vocab has an invalid or duplicate id ${id} for "${token}"`);
      }
      this.tokens[id] = token;
    }
    this.ranks = new Map();
    json.merges.forEach((pair, rank) => {
      if (!Array.isArray(pair) || pair.length !== 2) {
        throw new Error(`Tokenizer merge ${rank} is not a [string, string] pair`);
      }
      const [left, right] = pair;
      if (typeof left !== 'string' || typeof right !== 'string') {
        throw new Error(`Tokenizer merge ${rank} is not a [string, string] pair`);
      }
      if (!this.vocab.has(left + right)) {
        throw new Error(`Tokenizer merge ${rank} ("${left}" + "${right}") has no vocab entry`);
      }
      this.ranks.set(`${left} ${right}`, rank);
    });
    this.special = new Map(Object.entries(json.specialTokens));
    for (const [text, id] of this.special) {
      if (!Number.isInteger(id) || id < 0 || id >= this.vocabSize) {
        throw new Error(`Special token "${text}" has an id ${id} outside the vocabulary`);
      }
      const vocabId = this.vocab.get(text);
      if (vocabId !== id) {
        const found = vocabId === undefined ? 'no vocab entry' : `vocab id ${vocabId}`;
        throw new Error(`Special token "${text}" has id ${id} but the vocab has ${found}`);
      }
    }
    this.specialIds = new Map([...this.special].map(([text, id]) => [id, text]));
    this.specialPattern =
      this.special.size === 0
        ? null
        : new RegExp(
            `(${[...this.special.keys()]
              .sort((a, b) => b.length - a.length)
              .map(escapeRegExp)
              .join('|')})`,
            'g',
          );
  }

  static fromJSON(json: unknown): Tokenizer {
    if (!isTokenizerJson(json)) {
      throw new Error(
        'Tokenizer JSON must have type "gpt2-byte-bpe", vocab, merges and specialTokens',
      );
    }
    return new Tokenizer(json);
  }

  specialTokenId(text: string): number | undefined {
    return this.special.get(text);
  }

  encode(text: string): number[] {
    const ids: number[] = [];
    const parts = this.specialPattern ? text.split(this.specialPattern) : [text];
    for (const part of parts) {
      if (part === '') continue;
      const specialId = this.special.get(part);
      if (specialId !== undefined) {
        ids.push(specialId);
        continue;
      }
      for (const match of part.matchAll(PRETOKENIZE)) ids.push(...this.encodePiece(match[0]));
    }
    return ids;
  }

  decode(ids: readonly number[]): string {
    const bytes: number[] = [];
    for (const id of ids) bytes.push(...this.tokenBytes(id));
    return this.decoder.decode(new Uint8Array(bytes));
  }

  tokenBytes(id: number): Uint8Array {
    if (!Number.isInteger(id) || id < 0 || id >= this.vocabSize) {
      throw new RangeError(`Token id ${id} is outside the vocabulary of ${this.vocabSize}`);
    }
    const special = this.specialIds.get(id);
    if (special !== undefined) return new TextEncoder().encode(special);
    return Uint8Array.from(this.tokens[id], (char) => {
      const byte = this.charToByte.get(char);
      if (byte === undefined)
        throw new Error(`Token ${id} contains a character outside the byte map`);
      return byte;
    });
  }

  tokenText(id: number): string {
    const bytes = this.tokenBytes(id);
    try {
      return this.strictDecoder.decode(bytes);
    } catch {
      return Array.from(
        bytes,
        (byte) => `<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`,
      ).join('');
    }
  }

  private encodePiece(piece: string): number[] {
    const cached = this.cache.get(piece);
    if (cached) return cached;
    const mapped = Array.from(new TextEncoder().encode(piece), (byte) => this.byteToChar[byte]);
    const ids = this.bpe(mapped).map((symbol) => {
      const id = this.vocab.get(symbol);
      if (id === undefined) throw new Error(`BPE produced "${symbol}", which is not in the vocab`);
      return id;
    });
    this.cache.set(piece, ids);
    return ids;
  }

  /** Repeatedly merges the adjacent pair with the lowest rank, all its occurrences left to right. */
  private bpe(symbols: string[]): string[] {
    let word = symbols;
    while (word.length > 1) {
      let bestRank = Infinity;
      let bestPair: [string, string] | null = null;
      for (let i = 0; i < word.length - 1; i++) {
        const rank = this.ranks.get(`${word[i]} ${word[i + 1]}`);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = [word[i], word[i + 1]];
        }
      }
      if (bestPair === null) break;
      const [left, right] = bestPair;
      const merged: string[] = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === left && word[i + 1] === right) {
          merged.push(left + right);
          i++;
        } else {
          merged.push(word[i]);
        }
      }
      word = merged;
    }
    return word;
  }
}
