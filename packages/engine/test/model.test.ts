import { describe, expect, it } from 'vitest';
import {
  loadModel,
  MODEL_FORMAT,
  modelFromTensors,
  parseConfig,
  type ModelConfig,
} from '../src/model';
import { parseSafetensors, type Tensor } from '../src/safetensors';
import { readFixture } from './helpers/fixtures';
import { encodeSafetensors } from './helpers/safetensors';

const MICRO: ModelConfig = {
  vocabSize: 64,
  contextLength: 16,
  dModel: 16,
  nLayers: 2,
  nHeads: 2,
  dHead: 8,
  dMlp: 64,
  layerNormEps: 1e-5,
};

/** Every tensor a model with `config` needs, zero-filled. */
function syntheticTensors(config: ModelConfig): Map<string, Tensor> {
  const { vocabSize: v, contextLength: c, dModel: d, dMlp: m } = config;
  const shapes: [string, number[]][] = [
    ['wte.weight', [v, d]],
    ['wpe.weight', [c, d]],
    ['ln_f.weight', [d]],
    ['ln_f.bias', [d]],
  ];
  for (let i = 0; i < config.nLayers; i++) {
    shapes.push(
      [`h.${i}.ln_1.weight`, [d]],
      [`h.${i}.ln_1.bias`, [d]],
      [`h.${i}.attn.c_attn.weight`, [3 * d, d]],
      [`h.${i}.attn.c_attn.bias`, [3 * d]],
      [`h.${i}.attn.c_proj.weight`, [d, d]],
      [`h.${i}.attn.c_proj.bias`, [d]],
      [`h.${i}.ln_2.weight`, [d]],
      [`h.${i}.ln_2.bias`, [d]],
      [`h.${i}.mlp.c_fc.weight`, [m, d]],
      [`h.${i}.mlp.c_fc.bias`, [m]],
      [`h.${i}.mlp.c_proj.weight`, [d, m]],
      [`h.${i}.mlp.c_proj.bias`, [d]],
    );
  }
  return new Map(
    shapes.map(([name, shape]) => [
      name,
      { shape, data: new Float32Array(shape.reduce((a, b) => a * b, 1)) },
    ]),
  );
}

describe('parseConfig', () => {
  it('reads every field', () => {
    expect(parseConfig(JSON.stringify(MICRO))).toEqual(MICRO);
  });

  it('rejects missing, non-integer or non-positive fields', () => {
    // JSON.stringify drops undefined properties, so dMlp is missing from the text.
    expect(() => parseConfig(JSON.stringify({ ...MICRO, dMlp: undefined }))).toThrow(/dMlp/);
    expect(() => parseConfig(JSON.stringify({ ...MICRO, nLayers: 1.5 }))).toThrow(/nLayers/);
    expect(() => parseConfig(JSON.stringify({ ...MICRO, layerNormEps: 0 }))).toThrow(
      /layerNormEps/,
    );
  });

  it('rejects heads that do not divide the model width', () => {
    expect(() => parseConfig(JSON.stringify({ ...MICRO, dHead: 4 }))).toThrow(/dModel/);
  });

  it('rejects text that is not JSON', () => {
    expect(() => parseConfig('nope')).toThrow(/not valid JSON/);
  });
});

describe('modelFromTensors', () => {
  it('builds a model with the configured shapes', () => {
    const model = modelFromTensors(MICRO, syntheticTensors(MICRO));
    expect([model.wte.rows, model.wte.cols]).toEqual([64, 16]);
    expect([model.wpe.rows, model.wpe.cols]).toEqual([16, 16]);
    expect(model.blocks).toHaveLength(2);
    const block = model.blocks[1];
    expect([block.attnQkv.weight.rows, block.attnQkv.weight.cols]).toEqual([48, 16]);
    expect([block.mlpFc.weight.rows, block.mlpFc.weight.cols]).toEqual([64, 16]);
    expect([block.mlpProj.weight.rows, block.mlpProj.weight.cols]).toEqual([16, 64]);
    expect(block.mlpFc.bias).toHaveLength(64);
    expect(model.lnFinal.weight).toHaveLength(16);
  });

  it('names a missing tensor', () => {
    const tensors = syntheticTensors(MICRO);
    tensors.delete('h.1.mlp.c_fc.bias');
    expect(() => modelFromTensors(MICRO, tensors)).toThrow(/h\.1\.mlp\.c_fc\.bias/);
  });

  it('names a tensor with the wrong shape', () => {
    const tensors = syntheticTensors(MICRO);
    tensors.set('wpe.weight', { shape: [16, 15], data: new Float32Array(240) });
    expect(() => modelFromTensors(MICRO, tensors)).toThrow(/wpe\.weight/);
  });

  it('rejects unexpected tensors', () => {
    const tensors = syntheticTensors(MICRO);
    tensors.set('lm_head.weight', { shape: [1], data: new Float32Array(1) });
    expect(() => modelFromTensors(MICRO, tensors)).toThrow(/lm_head\.weight/);
  });

  it('rejects a tensor with the wrong rank', () => {
    const tensors = syntheticTensors(MICRO);
    // wpe.weight should be rank 2, [contextLength, dModel].
    tensors.set('wpe.weight', { shape: [16 * 16], data: new Float32Array(256) });
    expect(() => modelFromTensors(MICRO, tensors)).toThrow(/wpe\.weight/);
  });
});

describe('loadModel', () => {
  it('loads the micro fixture written by the Python exporter', () => {
    const buffer = readFixture('micro/model.safetensors');
    const model = loadModel(buffer);
    expect(model.config).toEqual(MICRO);
    const raw = parseSafetensors(buffer).tensors;
    expect(Array.from(model.wte.data.subarray(0, 4))).toEqual(
      Array.from(raw.get('wte.weight')?.data.subarray(0, 4) ?? []),
    );
    expect(Array.from(model.blocks[1].attnProj.bias)).toEqual(
      Array.from(raw.get('h.1.attn.c_proj.bias')?.data ?? []),
    );
  });

  it('rejects files in another format', () => {
    const buffer = encodeSafetensors([], { format: 'other', config: JSON.stringify(MICRO) });
    expect(() => loadModel(buffer)).toThrow(new RegExp(MODEL_FORMAT));
  });

  it('rejects files without a config', () => {
    const buffer = encodeSafetensors([], { format: MODEL_FORMAT });
    expect(() => loadModel(buffer)).toThrow(/config/);
  });
});
