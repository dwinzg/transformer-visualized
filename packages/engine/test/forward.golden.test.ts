import { describe, expect, it } from 'vitest';
import { forward } from '../src/forward';
import { loadModel } from '../src/model';
import { parseSafetensors } from '../src/safetensors';
import { flattenTrace } from '../src/trace';
import { expectAllClose } from './helpers/close';
import { readFixture, readFixtureJson } from './helpers/fixtures';

interface FixtureIndex {
  model: string;
  cases: { name: string; tokenIds: number[]; file: string }[];
}

const index = readFixtureJson<FixtureIndex>('micro/cases.json');
const model = loadModel(readFixture(`micro/${index.model}`));

describe('forward matches the PyTorch reference on the micro fixtures', () => {
  it.each(index.cases)('$name', ({ tokenIds, file }) => {
    const expected = parseSafetensors(readFixture(`micro/${file}`)).tensors;
    const actual = flattenTrace(forward(model, tokenIds));
    expect([...actual.keys()].sort()).toEqual([...expected.keys()].sort());
    for (const [name, want] of expected) {
      const got = actual.get(name);
      if (got === undefined) throw new Error(`missing ${name}`);
      expect(got.shape, `${name} shape`).toEqual(want.shape);
      expectAllClose(name, got.data, want.data);
    }
  });
});
