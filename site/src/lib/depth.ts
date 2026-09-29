/** Explanation levels, from the simplest to the most technical. */
export const DEPTHS = ['story', 'numbers', 'formula', 'code'] as const;
export type Depth = (typeof DEPTHS)[number];

export const DEPTH_LABELS: Record<Depth, string> = {
  story: 'Story',
  numbers: 'Numbers',
  formula: 'Formula',
  code: 'Code',
};

export const DEPTH_STORAGE_KEY = 'tv-depth';
/** Fired on `document` with the new level as `detail` whenever the reader changes level. */
export const DEPTH_EVENT = 'tv:depthchange';

export function isDepth(value: unknown): value is Depth {
  return typeof value === 'string' && (DEPTHS as readonly string[]).includes(value);
}

/**
 * The level to show when a step does not offer the preferred one. Prefers the nearest simpler
 * level, so a reader who chose "code" sees the formula rather than the story.
 */
export function closestAvailable(preferred: Depth, available: readonly Depth[]): Depth {
  if (available.includes(preferred)) return preferred;
  const index = DEPTHS.indexOf(preferred);
  for (let i = index - 1; i >= 0; i--) if (available.includes(DEPTHS[i])) return DEPTHS[i];
  for (let i = index + 1; i < DEPTHS.length; i++)
    if (available.includes(DEPTHS[i])) return DEPTHS[i];
  return preferred;
}

export function readStoredDepth(storage: Pick<Storage, 'getItem'> | undefined): Depth {
  try {
    const value = storage?.getItem(DEPTH_STORAGE_KEY);
    return isDepth(value) ? value : 'story';
  } catch {
    return 'story';
  }
}

/** `localStorage`, or undefined where the browser blocks it (some private modes do). */
export function browserStorage(): Storage | undefined {
  try {
    return localStorage;
  } catch {
    return undefined;
  }
}

/** Remembers the reader's level and tells every step and dial on the page. */
export function chooseDepth(depth: Depth): void {
  try {
    browserStorage()?.setItem(DEPTH_STORAGE_KEY, depth);
  } catch {
    // Storage can be full or blocked. The choice then lasts for this page only.
  }
  document.dispatchEvent(new CustomEvent<Depth>(DEPTH_EVENT, { detail: depth }));
}
