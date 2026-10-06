import { describe, expect, it } from 'vitest';
import { citationLabel, formatAuthors } from './citations';

const a = { family: 'Vaswani', given: 'Ashish' };
const b = { family: 'Shazeer', given: 'Noam' };
const c = { family: 'Parmar', given: 'Niki' };

describe('citationLabel', () => {
  it('names one or two authors and shortens three or more', () => {
    expect(citationLabel({ authors: [a], year: 2017 })).toBe('Vaswani, 2017');
    expect(citationLabel({ authors: [a, b], year: 2017 })).toBe('Vaswani and Shazeer, 2017');
    expect(citationLabel({ authors: [a, b, c], year: 2017 })).toBe('Vaswani et al., 2017');
  });

  it('uses organization names as they are', () => {
    expect(citationLabel({ authors: [{ literal: '3Blue1Brown' }], year: 2024 })).toBe(
      '3Blue1Brown, 2024',
    );
  });
});

describe('formatAuthors', () => {
  it('lists family names with initials', () => {
    expect(formatAuthors([a])).toBe('Vaswani, A.');
    expect(formatAuthors([a, b])).toBe('Vaswani, A. and Shazeer, N.');
    expect(formatAuthors([a, b, c])).toBe('Vaswani, A., Shazeer, N., and Parmar, N.');
  });

  it('names the first six of a long list and counts the rest', () => {
    const many = Array.from({ length: 31 }, (_, i) => ({ family: `F${i}`, given: 'G' }));
    expect(formatAuthors(many)).toBe(
      'F0, G., F1, G., F2, G., F3, G., F4, G., F5, G., and 25 others',
    );
    // Seven is short enough to list in full, rather than say "and 1 others".
    expect(formatAuthors(many.slice(0, 7))).toMatch(/, and F6, G\.$/);
  });

  it('shortens hyphenated and multi-part given names to initials', () => {
    expect(formatAuthors([{ family: 'Kim', given: 'Grace C.' }])).toBe('Kim, G. C.');
    expect(formatAuthors([{ family: 'Chau', given: 'Duen-Horng' }])).toBe('Chau, D.-H.');
  });
});
