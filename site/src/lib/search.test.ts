import { describe, expect, it } from 'vitest';
import { prepare, search, snippet, words } from './search';
import type { SearchEntry } from './search-index';

const entry = (kind: SearchEntry['kind'], title: string, text: string): SearchEntry => ({
  kind,
  title,
  href: title.toLowerCase(),
  text,
});
const index = prepare([
  entry('Step', 'Softmax turns scores into weights', 'Softmax turns a row of scores into weights.'),
  entry('Glossary', 'Softmax', 'Turns scores into chances that add up to 1.'),
  entry('Step', 'Text becomes tokens', 'A model reads tokens, pieces of text.'),
  entry('Chapter', 'Attention', 'Each token looks at the tokens before it. Naïve readers welcome.'),
]);

describe('search', () => {
  it('needs every word of the query, as the start of a word', () => {
    expect(search(index, 'soft weights').map((h) => h.entry.title)).toEqual([
      'Softmax turns scores into weights',
    ]);
    expect(search(index, 'softmax banana')).toEqual([]);
  });

  it('ranks a glossary term with the exact title above a longer step', () => {
    expect(search(index, 'softmax')[0].entry.kind).toBe('Glossary');
  });

  it('treats plurals, capitals and accents as the same word', () => {
    expect(search(index, 'TOKEN').length).toBe(2);
    expect(search(index, 'naive').map((h) => h.entry.title)).toEqual(['Attention']);
  });

  it('finds nothing for an empty or blank query', () => {
    expect(search(index, '')).toEqual([]);
    expect(search(index, '  ,. ')).toEqual([]);
  });
});

describe('snippet', () => {
  it('marks each match and keeps the text around it', () => {
    const pieces = snippet('A model reads tokens, pieces of text.', words('token'));
    expect(pieces.filter((p) => p.mark).map((p) => p.text)).toEqual(['tokens']);
    expect(pieces.map((p) => p.text).join('')).toBe('A model reads tokens, pieces of text.');
  });

  it('cuts long text at spaces around the first match', () => {
    const text = `${'word '.repeat(60)}target ${'more '.repeat(60)}`.trim();
    const joined = snippet(text, ['target'])
      .map((p) => p.text)
      .join('');
    expect(joined).toMatch(/^… word/);
    expect(joined).toMatch(/more …$/);
    expect(joined.length).toBeLessThan(180);
  });
});
