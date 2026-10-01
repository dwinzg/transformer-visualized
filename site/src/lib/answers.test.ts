import { describe, expect, it } from 'vitest';
import { isCorrect, parseNumber } from './answers';

describe('parseNumber', () => {
  it('reads decimals, percentages and simple fractions', () => {
    expect(parseNumber('0.05')).toBe(0.05);
    expect(parseNumber('.05')).toBe(0.05);
    expect(parseNumber(' 5% ')).toBeCloseTo(0.05, 12);
    expect(parseNumber('1/20')).toBe(0.05);
    expect(parseNumber('-2.5')).toBe(-2.5);
  });

  it('accepts a comma as the decimal mark', () => {
    expect(parseNumber('0,05')).toBe(0.05);
  });

  it('returns null for anything else', () => {
    for (const text of ['', 'abc', '1/0', '1.2.3', '5%%']) expect(parseNumber(text)).toBeNull();
  });
});

describe('isCorrect', () => {
  it('accepts answers within the tolerance', () => {
    expect(isCorrect(0.0501, 0.05, 0.001)).toBe(true);
    expect(isCorrect(0.06, 0.05, 0.001)).toBe(false);
  });
});
