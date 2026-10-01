import { describe, expect, it } from 'vitest';
import { forward } from '../src/forward';
import { loadModel } from '../src/model';
import { parseSafetensors } from '../src/safetensors';
import { flattenTrace } from '../src/trace';
import { Tokenizer } from '../src/tokenizer';
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

interface TinyIndex {
  model: string;
  cases: { name: string; text: string; tokenIds: number[]; file: string }[];
}

const tinyIndex = readFixtureJson<TinyIndex>('tiny/cases.json');
const tinyModel = loadModel(readFixture(`tiny/${tinyIndex.model}`));
const tinyTokenizer = Tokenizer.fromJSON(
  JSON.parse(new TextDecoder().decode(readFixture('../../../../models/tiny/tokenizer.json'))),
);

describe('forward matches the PyTorch reference on the trained tiny model', () => {
  it.each(tinyIndex.cases)('$name', ({ text, tokenIds, file }) => {
    expect(tinyTokenizer.encode(text)).toEqual(tokenIds);
    const expected = parseSafetensors(readFixture(`tiny/${file}`)).tensors;
    const actual = flattenTrace(forward(tinyModel, tokenIds));
    expect([...actual.keys()].sort()).toEqual([...expected.keys()].sort());
    for (const [name, want] of expected) {
      const got = actual.get(name);
      if (got === undefined) throw new Error(`missing ${name}`);
      expect(got.shape, `${name} shape`).toEqual(want.shape);
      expectAllClose(name, got.data, want.data);
    }
  });
});
