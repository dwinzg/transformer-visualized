import { describe, expect, it } from 'vitest';
import { joinBase } from './paths';

describe('joinBase', () => {
  it('joins a base and a relative path with exactly one slash', () => {
    expect(joinBase('/transformer-visualized', 'learn/')).toBe('/transformer-visualized/learn/');
    expect(joinBase('/transformer-visualized/', 'learn/')).toBe('/transformer-visualized/learn/');
    expect(joinBase('/transformer-visualized/', '/learn/')).toBe('/transformer-visualized/learn/');
  });

  it('keeps anchors and query strings', () => {
    expect(joinBase('/tv/', 'references/#vaswani2017')).toBe('/tv/references/#vaswani2017');
    expect(joinBase('/tv/', 'learn/?depth=formula')).toBe('/tv/learn/?depth=formula');
  });

  it('returns the base itself for an empty path', () => {
    expect(joinBase('/tv', '')).toBe('/tv/');
  });
});
