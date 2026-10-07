import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { chanceLoss, logX, strongestHead, tokenCount, type TrainingData } from './training';

describe('training helpers', () => {
  it('gives the loss of a blind guess among 4,096 tokens', () => {
    expect(chanceLoss(4096)).toBeCloseTo(8.318, 3);
  });

  it('writes token counts in words a reader can take in', () => {
    expect(tokenCount(0)).toBe('0');
    expect(tokenCount(204_800)).toBe('204,800');
    expect(tokenCount(8_192_000)).toBe('8.2 million');
    expect(tokenCount(204_800_000)).toBe('205 million');
  });

  it('spaces steps by their log, from 0 at the start to 1 at the end', () => {
    expect(logX(0, 25_000)).toBe(0);
    expect(logX(25_000, 25_000)).toBe(1);
    expect(logX(250, 25_000)).toBeGreaterThan(0.5);
  });

  it('finds the strongest head', () => {
    expect(
      strongestHead([
        [0.1, 0.2],
        [0.5, 0.3],
      ]),
    ).toEqual({ layer: 1, head: 0, score: 0.5 });
  });
});

describe('the saved training run', () => {
  const data = JSON.parse(
    readFileSync(join(import.meta.dirname, '../../../models/tiny/training.json'), 'utf8'),
  ) as TrainingData;

  it('has snapshots in step order, from the untrained model to the end', () => {
    const steps = data.snapshots.map((s) => s.step);
    expect(steps[0]).toBe(0);
    expect(steps).toEqual([...steps].sort((a, b) => a - b));
    expect(steps.at(-1)).toBe(25_000);
  });

  it('starts near a blind guess and ends near the shipped model', () => {
    expect(data.snapshots[0].valLoss).toBeCloseTo(chanceLoss(4096), 1);
    expect(data.snapshots.at(-1)!.valLoss).toBeCloseTo(1.85, 2);
  });

  it('keeps every chance between 0 and 1, with a 4 by 4 table of heads', () => {
    for (const s of data.snapshots) {
      for (const p of [s.recall, s.copy, ...s.next.map((g) => g.p)]) {
        expect(p).toBeGreaterThanOrEqual(0);
        expect(p).toBeLessThanOrEqual(1);
      }
      expect(s.induction.map((row) => row.length)).toEqual([4, 4, 4, 4]);
    }
  });
});
