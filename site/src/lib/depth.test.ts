import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  chooseDepth,
  closestAvailable,
  DEPTH_EVENT,
  DEPTH_STORAGE_KEY,
  DEPTHS,
  isDepth,
  readStoredDepth,
} from './depth';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('isDepth', () => {
  it('accepts only the four levels', () => {
    for (const depth of DEPTHS) expect(isDepth(depth)).toBe(true);
    expect(isDepth('expert')).toBe(false);
    expect(isDepth(undefined)).toBe(false);
  });
});

describe('closestAvailable', () => {
  it('keeps the preferred level when it exists', () => {
    expect(closestAvailable('formula', ['story', 'numbers', 'formula'])).toBe('formula');
  });

  it('falls back to the nearest simpler level, then the nearest deeper one', () => {
    expect(closestAvailable('code', ['story', 'numbers', 'formula'])).toBe('formula');
    expect(closestAvailable('formula', ['story', 'code'])).toBe('story');
    expect(closestAvailable('story', ['formula', 'code'])).toBe('formula');
  });
});

describe('readStoredDepth', () => {
  it('reads a stored level', () => {
    expect(readStoredDepth({ getItem: () => 'numbers' })).toBe('numbers');
  });

  it('defaults to story for missing, invalid or unreadable storage', () => {
    expect(readStoredDepth(undefined)).toBe('story');
    expect(readStoredDepth({ getItem: () => 'nonsense' })).toBe('story');
    expect(
      readStoredDepth({
        getItem: () => {
          throw new Error('denied');
        },
      }),
    ).toBe('story');
  });
});

describe('chooseDepth', () => {
  it('stores the level and announces it on the document', () => {
    const setItem = vi.fn();
    const dispatchEvent = vi.fn();
    vi.stubGlobal('localStorage', { setItem });
    vi.stubGlobal('document', { dispatchEvent });
    chooseDepth('code');
    expect(setItem).toHaveBeenCalledWith(DEPTH_STORAGE_KEY, 'code');
    const event = dispatchEvent.mock.calls[0][0] as CustomEvent<string>;
    expect(event.type).toBe(DEPTH_EVENT);
    expect(event.detail).toBe('code');
  });

  it('still announces the level when storage fails', () => {
    const dispatchEvent = vi.fn();
    vi.stubGlobal('localStorage', {
      setItem: () => {
        throw new Error('full');
      },
    });
    vi.stubGlobal('document', { dispatchEvent });
    chooseDepth('numbers');
    expect(dispatchEvent).toHaveBeenCalledOnce();
  });
});
