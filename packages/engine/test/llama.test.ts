import { describe, expect, it } from 'vitest';
import {
  flattenLlamaTrace,
  forwardLlama,
  loadLlama,
  parseLlamaConfig,
  rmsNorm,
  rope,
  silu,
} from '../src/llama';
import { loadModel } from '../src/model';
import { parseSafetensors } from '../src/safetensors';
import { matrixFromRows, rowView } from '../src/tensor';
import { expectAllClose } from './helpers/close';
import { readFixture, readFixtureJson } from './helpers/fixtures';

interface FixtureIndex {
  model: string;
  cases: { name: string; tokenIds: number[]; file: string }[];
}

const index = readFixtureJson<FixtureIndex>('llama-micro/cases.json');
const model = loadLlama(readFixture(`llama-micro/${index.model}`));

describe('the Llama-style forward pass matches the PyTorch reference', () => {
  it.each(index.cases)('$name', ({ tokenIds, file }) => {
    const expected = parseSafetensors(readFixture(`llama-micro/${file}`)).tensors;
    const actual = flattenLlamaTrace(forwardLlama(model, tokenIds));
    expect([...actual.keys()].sort()).toEqual([...expected.keys()].sort());
    for (const [name, want] of expected) {
      const got = actual.get(name)!;
      expect(got.shape, `${name} shape`).toEqual(want.shape);
      expectAllClose(name, got.data, want.data);
    }
  });
});

describe('Llama-style parts', () => {
  it('RMSNorm divides by the root mean square and scales', () => {
    const { meanSquare, out } = rmsNorm(
      matrixFromRows([[1, -1, 3, -3]]),
      new Float32Array([1, 1, 1, 2]),
      0,
    );
    expect(meanSquare[0]).toBe(5);
    expect([...rowView(out, 0)].map((v) => +v.toFixed(4))).toEqual([
      0.4472, -0.4472, 1.3416, -2.6833,
    ]);
  });

  it('RoPE leaves position 0 alone and keeps each pair the same length', () => {
    const x = matrixFromRows([
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ]);
    const turned = rope(x, 10_000);
    expect([...rowView(turned, 0)]).toEqual([1, 2, 3, 4]);
    const len = (a: number, b: number) => Math.hypot(a, b);
    const row = rowView(turned, 1);
    expect(len(row[0], row[1])).toBeCloseTo(len(1, 2), 5);
    expect(len(row[2], row[3])).toBeCloseTo(len(3, 4), 5);
    expect(row[0]).not.toBeCloseTo(1, 3);
  });

  it('SiLU is x times its sigmoid', () => {
    expect(silu(0)).toBe(0);
    expect(silu(2)).toBeCloseTo(2 / (1 + Math.exp(-2)), 12);
  });

  it('query heads in a group share their key and value', () => {
    const trace = forwardLlama(model, [1, 2, 3]);
    const [h0, h1, h2] = trace.layers[0].heads;
    expect(h0.k).toBe(h1.k);
    expect(h0.v).toBe(h1.v);
    expect(h2.k).not.toBe(h0.k);
  });

  it('refuses bad configs and the other format', () => {
    const good = {
      vocabSize: 64,
      contextLength: 16,
      dModel: 16,
      nLayers: 2,
      nHeads: 4,
      dHead: 4,
      dMlp: 48,
      layerNormEps: 1e-5,
      arch: 'llama',
      nKvHeads: 2,
      ropeBase: 10000,
    };
    expect(parseLlamaConfig(JSON.stringify(good)).nKvHeads).toBe(2);
    expect(() => parseLlamaConfig(JSON.stringify({ ...good, nKvHeads: 3 }))).toThrow(/multiple/);
    expect(() => parseLlamaConfig(JSON.stringify({ ...good, arch: 'gpt2' }))).toThrow(/arch/);
    expect(() => loadModel(readFixture(`llama-micro/${index.model}`))).toThrow(/format/);
    expect(() => loadLlama(readFixture('micro/model.safetensors'))).toThrow(/format/);
  });
});
