import { expect } from 'vitest';

export interface Tolerance {
  atol: number;
  rtol: number;
}

/** The tolerance for engine values compared with the PyTorch reference. */
export const PARITY_TOLERANCE: Tolerance = { atol: 1e-4, rtol: 1e-3 };

/**
 * Asserts |actual - expected| <= atol + rtol * |expected| for every value.
 * Infinite expected values must match exactly. Fails on the first mismatch, naming its index.
 */
export function expectAllClose(
  label: string,
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: Tolerance = PARITY_TOLERANCE,
): void {
  expect(actual.length, `${label}: length`).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];
    const ok = Number.isFinite(e)
      ? Math.abs(a - e) <= tolerance.atol + tolerance.rtol * Math.abs(e)
      : a === e;
    if (!ok) {
      expect.fail(
        `${label}: value ${i} is ${a}, expected ${e} (atol ${tolerance.atol}, rtol ${tolerance.rtol})`,
      );
    }
  }
}
