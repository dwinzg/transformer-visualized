import { describe, expect, it } from 'vitest';
import { stepId } from './stepId';

describe('stepId', () => {
  it('slugifies a title into a step id', () => {
    expect(stepId('It guesses the next piece of text', new Set())).toBe(
      'step-it-guesses-the-next-piece-of-text',
    );
  });

  it('deduplicates ids already taken on the page', () => {
    const taken = new Set(['step-it-guesses']);
    expect(stepId('It guesses', taken)).toBe('step-it-guesses-2');
    taken.add('step-it-guesses-2');
    expect(stepId('It guesses', taken)).toBe('step-it-guesses-3');
  });

  it('falls back to a numbered id when the title has no ASCII letters or digits', () => {
    expect(stepId('日本語', new Set())).toBe('step-1');
  });

  it('deduplicates fallback ids too', () => {
    const taken = new Set(['step-1']);
    expect(stepId('日本語', taken)).toBe('step-2');
  });
});
