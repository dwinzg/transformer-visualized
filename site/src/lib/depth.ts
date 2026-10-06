/** Explanation levels, from the simplest to the most technical. */
export const DEPTHS = ['story', 'numbers', 'formula', 'code'] as const;
export type Depth = (typeof DEPTHS)[number];

export const DEPTH_LABELS: Record<Depth, string> = {
  story: 'Story',
  numbers: 'Numbers',
  formula: 'Formula',
  code: 'Code',
};

/** What each level adds, said where a reader picks one. */
export const DEPTH_NOTES: Record<Depth, string> = {
  story: 'Plain words and pictures. No math needed.',
  numbers: 'Adds the real numbers from our model, worked out step by step.',
  formula: 'Adds the math, with every symbol explained.',
  code: 'Adds the code that runs each step, line by line.',
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

/** Drops a `depth` query parameter from the address bar without reloading the page. */
function clearDepthQueryParam(): void {
  const url = new URL(location.href);
  if (!url.searchParams.has('depth')) return;
  url.searchParams.delete('depth');
  history.replaceState(history.state, '', url);
}

/** Remembers the reader's level and tells every step and dial on the page. */
export function chooseDepth(depth: Depth): void {
  try {
    browserStorage()?.setItem(DEPTH_STORAGE_KEY, depth);
  } catch {
    // Storage can be full or blocked. The choice then lasts for this page only.
  }
  markPageDepth(depth);
  document.dispatchEvent(new CustomEvent<Depth>(DEPTH_EVENT, { detail: depth }));
  clearDepthQueryParam();
}

/**
 * Applies a `?depth=` link (for example the "Quick review" door) to the current page view only.
 * Unlike chooseDepth, it does not save the level, so it never overrides a level the reader picked
 * earlier or will pick later. It also drops the parameter from the address bar so reloading the
 * page does not reapply it.
 */
export function applyDepthFromQuery(): void {
  const requested = new URL(location.href).searchParams.get('depth');
  if (!isDepth(requested)) return;
  markPageDepth(requested);
  document.dispatchEvent(new CustomEvent<Depth>(DEPTH_EVENT, { detail: requested }));
  clearDepthQueryParam();
}

/**
 * Records the level shown on this page, so islands that hydrate later (client:visible) start at
 * it instead of missing the event that a `?depth=` link fired at load.
 */
function markPageDepth(depth: Depth): void {
  document.documentElement.dataset.depth = depth;
}

/** The level this page shows right now: a `?depth=` link or a pick on this page, else the saved one. */
export function pageDepth(): Depth {
  const marked = document.documentElement.dataset.depth;
  return isDepth(marked) ? marked : readStoredDepth(browserStorage());
}
