import { describe, expect, it } from 'vitest';
import { flipDelta } from './motion';

describe('flipDelta', () => {
  it('measures how far an element moved and how much it grew', () => {
    expect(
      flipDelta(
        { left: 10, top: 20, width: 50, height: 10 },
        { left: 110, top: 0, width: 100, height: 20 },
      ),
    ).toEqual({ x: -100, y: 20, scaleX: 0.5, scaleY: 0.5 });
  });

  it('treats a zero-size target as no scaling', () => {
    expect(
      flipDelta({ left: 0, top: 0, width: 5, height: 5 }, { left: 0, top: 0, width: 0, height: 0 }),
    ).toEqual({ x: 0, y: 0, scaleX: 1, scaleY: 1 });
  });
});
