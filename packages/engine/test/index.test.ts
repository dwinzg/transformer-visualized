import { describe, expect, it } from 'vitest';
import * as engine from '../src/index';

const functions = [
  'createMatrix',
  'matrixFromRows',
  'rowView',
  'valueAt',
  'linear',
  'matmul',
  'add',
  'sliceColumns',
  'concatColumns',
  'gatherRows',
  'mapValues',
  'layerNorm',
  'geluTanh',
  'causalScaled',
  'softmax',
  'softmaxRows',
  'parseSafetensors',
  'parseConfig',
  'modelFromTensors',
  'loadModel',
  'forward',
  'flattenTrace',
  'probabilities',
  'topKFilter',
  'topPFilter',
  'nextTokenDistribution',
  'argmax',
  'sample',
  'createRng',
  'createDecoder',
  'generate',
  'explainDot',
  'explainAttentionWeight',
  'explainProbability',
  'explainLayerNorm',
  'bytesToUnicode',
] as const;

describe('public API', () => {
  it.each(functions)('exports %s', (name) => {
    expect(typeof engine[name]).toBe('function');
  });

  it('exports the model format id', () => {
    expect(engine.MODEL_FORMAT).toBe('transformer-visualized/gpt2/1');
  });

  it('exports the Tokenizer class', () => {
    expect(typeof engine.Tokenizer.fromJSON).toBe('function');
  });
});
