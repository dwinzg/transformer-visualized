import { describe, expect, it } from 'vitest';
import { shownToken, spokenToken } from './token-text';

describe('token text', () => {
  it('draws and speaks spaces, new lines and tabs', () => {
    expect(shownToken({ text: ' Lily' })).toBe('·Lily');
    expect(spokenToken({ text: ' Lily' })).toBe('space Lily');
    expect(shownToken({ text: '\n' })).toBe('↵');
  });

  it('names a byte piece', () => {
    expect(spokenToken({ text: '<0xC3>' })).toBe('byte C3');
  });

  it('leaves a special token as its name, with no space marks', () => {
    const end = { text: '[end of story]', special: true };
    expect(shownToken(end)).toBe('[end of story]');
    expect(spokenToken(end)).toBe('end of story');
  });
});
