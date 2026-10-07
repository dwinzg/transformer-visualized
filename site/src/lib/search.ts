import type { SearchEntry } from './search-index';

export interface Prepared {
  entry: SearchEntry;
  titleWords: string[];
  textWords: string[];
}

/** A piece of a snippet, marked when it matches a word of the query. */
export interface Piece {
  text: string;
  mark: boolean;
}

export interface Hit {
  entry: SearchEntry;
  score: number;
  snippet: Piece[];
}

// Letters with accents match plain ones, so "naive" finds "naïve". Each character stays one
// character, so a position in the folded text is the same position in the original.
const fold = (s: string) =>
  Array.from(s, (c) => {
    const plain = c
      .toLowerCase()
      .normalize('NFKD')
      .replace(/[\u0300-\u036f]/g, '');
    return plain.length === c.length ? plain : c;
  }).join('');

// A light stem, so "tokens" finds "token" and the other way around.
const stem = (w: string) =>
  w.length > 3 && w.endsWith('s') && !w.endsWith('ss') ? w.slice(0, -1) : w;

export const words = (s: string) =>
  fold(s)
    .split(/[^\p{L}\p{N}]+/u)
    .filter(Boolean)
    .map(stem);

export const prepare = (entries: SearchEntry[]): Prepared[] =>
  entries.map((entry) => ({
    entry,
    titleWords: words(entry.title),
    textWords: words(`${entry.context ?? ''} ${entry.text}`),
  }));

// Glossary terms and answers are the most direct hit for the same words.
const KIND_BONUS: Partial<Record<SearchEntry['kind'], number>> = {
  Glossary: 3,
  FAQ: 2,
  Chapter: 1,
};

/** Every query word must start a word of the entry. Words in the title count most. */
export function search(index: Prepared[], query: string, limit = 30): Hit[] {
  const terms = [...new Set(words(query))];
  if (terms.length === 0) return [];
  const hits: Hit[] = [];
  for (const item of index) {
    let score = 0;
    let all = true;
    for (const term of terms) {
      const inTitle = item.titleWords.filter((w) => w.startsWith(term)).length;
      const inText = item.textWords.filter((w) => w.startsWith(term)).length;
      if (inTitle + inText === 0) {
        all = false;
        break;
      }
      // Repeats help a little, but one strong page should not drown out the rest.
      score += inTitle * 10 + Math.min(inText, 10);
      if (item.titleWords.includes(term)) score += 5;
    }
    if (!all) continue;
    score += KIND_BONUS[item.entry.kind] ?? 0;
    hits.push({ entry: item.entry, score, snippet: snippet(item.entry.text, terms) });
  }
  return hits.sort((a, b) => b.score - a.score).slice(0, limit);
}

/** About 160 characters of the text around the first match, with each match marked. */
export function snippet(text: string, terms: string[], width = 160): Piece[] {
  const folded = fold(text);
  const at = (term: string) => {
    const found = new RegExp(`(^|[^\\p{L}\\p{N}])${escape(term)}`, 'u').exec(folded);
    return found ? found.index + found[1].length : -1;
  };
  const first = Math.min(...terms.map(at).filter((i) => i >= 0), Infinity);
  let start = first === Infinity ? 0 : Math.max(0, first - width / 3);
  let end = Math.min(text.length, start + width);
  // Cut at spaces, so no word is broken in half.
  if (start > 0) start = text.indexOf(' ', start) + 1;
  if (end < text.length) end = Math.max(text.lastIndexOf(' ', end), start);
  const piece = text.slice(start, end);

  const pattern = new RegExp(
    `(^|[^\\p{L}\\p{N}])((?:${terms.map(escape).join('|')})[\\p{L}\\p{N}]*)`,
    'gu',
  );
  const pieces: Piece[] = [];
  const foldedPiece = fold(piece);
  let last = 0;
  for (const m of foldedPiece.matchAll(pattern)) {
    const from = m.index + m[1].length;
    const to = from + m[2].length;
    if (from > last) pieces.push({ text: piece.slice(last, from), mark: false });
    pieces.push({ text: piece.slice(from, to), mark: true });
    last = to;
  }
  if (last < piece.length) pieces.push({ text: piece.slice(last), mark: false });
  if (start > 0) pieces.unshift({ text: '… ', mark: false });
  if (end < text.length) pieces.push({ text: ' …', mark: false });
  return pieces;
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
