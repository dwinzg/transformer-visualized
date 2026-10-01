import { parseSafetensors, type Tensor } from './safetensors';
import { createMatrix, type Matrix } from './tensor';

/** Value of the `format` metadata key in model files written by `model/tv_model/io.py`. */
export const MODEL_FORMAT = 'transformer-visualized/gpt2/1';

/** Shapes of a GPT-2 style model. Keys match the `config` metadata in model files. */
export interface ModelConfig {
  readonly vocabSize: number;
  readonly contextLength: number;
  readonly dModel: number;
  readonly nLayers: number;
  readonly nHeads: number;
  readonly dHead: number;
  readonly dMlp: number;
  readonly layerNormEps: number;
}

export interface LayerNormWeights {
  readonly weight: Float32Array;
  readonly bias: Float32Array;
}

/** A Linear layer. `weight` is [out, in], as in PyTorch. */
export interface LinearWeights {
  readonly weight: Matrix;
  readonly bias: Float32Array;
}

export interface BlockWeights {
  readonly ln1: LayerNormWeights;
  /** [3·dModel, dModel]. Output columns are all queries, then all keys, then all values. */
  readonly attnQkv: LinearWeights;
  readonly attnProj: LinearWeights;
  readonly ln2: LayerNormWeights;
  readonly mlpFc: LinearWeights;
  readonly mlpProj: LinearWeights;
}

export interface Model {
  readonly config: ModelConfig;
  /** Token embeddings [vocab, dModel]. Also used as the output projection (weight tying). */
  readonly wte: Matrix;
  /** Position embeddings [context, dModel]. */
  readonly wpe: Matrix;
  readonly blocks: readonly BlockWeights[];
  readonly lnFinal: LayerNormWeights;
}

const INTEGER_KEYS = ['vocabSize', 'contextLength', 'dModel', 'nLayers', 'nHeads', 'dHead', 'dMlp'];

/** Reads and validates the config JSON stored in a model file. */
export function parseConfig(json: string): ModelConfig {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch (cause) {
    throw new Error('Model config is not valid JSON', { cause });
  }
  if (typeof raw !== 'object' || raw === null) throw new Error('Model config must be an object');
  const record = raw as Record<string, unknown>;
  for (const key of INTEGER_KEYS) {
    const value = record[key];
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
      throw new Error(`Model config field "${key}" must be a positive integer`);
    }
  }
  const eps = record.layerNormEps;
  if (typeof eps !== 'number' || !Number.isFinite(eps) || eps <= 0) {
    throw new Error('Model config field "layerNormEps" must be a positive number');
  }
  const config = record as unknown as ModelConfig;
  if (config.nHeads * config.dHead !== config.dModel) {
    throw new Error(
      `Model config has nHeads × dHead = ${config.nHeads * config.dHead}, expected dModel ${config.dModel}`,
    );
  }
  return {
    vocabSize: config.vocabSize,
    contextLength: config.contextLength,
    dModel: config.dModel,
    nLayers: config.nLayers,
    nHeads: config.nHeads,
    dHead: config.dHead,
    dMlp: config.dMlp,
    layerNormEps: config.layerNormEps,
  };
}

/** Builds a model from named tensors. Checks every expected name and shape and rejects extras. */
export function modelFromTensors(config: ModelConfig, tensors: ReadonlyMap<string, Tensor>): Model {
  const { vocabSize, contextLength, dModel: d, dMlp, nLayers } = config;
  const used = new Set<string>();
  const take = (name: string, shape: readonly number[]): Float32Array => {
    const tensor = tensors.get(name);
    if (tensor === undefined) throw new Error(`Model weights are missing "${name}"`);
    const matches =
      tensor.shape.length === shape.length && tensor.shape.every((size, i) => size === shape[i]);
    if (!matches) {
      throw new Error(
        `Weight "${name}" has shape [${tensor.shape.join(', ')}], expected [${shape.join(', ')}]`,
      );
    }
    used.add(name);
    return tensor.data;
  };
  const matrix = (name: string, rows: number, cols: number): Matrix =>
    createMatrix(rows, cols, take(name, [rows, cols]));
  const linear = (name: string, out: number, inp: number): LinearWeights => ({
    weight: matrix(`${name}.weight`, out, inp),
    bias: take(`${name}.bias`, [out]),
  });
  const norm = (name: string): LayerNormWeights => ({
    weight: take(`${name}.weight`, [d]),
    bias: take(`${name}.bias`, [d]),
  });

  const blocks = Array.from({ length: nLayers }, (_, i): BlockWeights => ({
    ln1: norm(`h.${i}.ln_1`),
    attnQkv: linear(`h.${i}.attn.c_attn`, 3 * d, d),
    attnProj: linear(`h.${i}.attn.c_proj`, d, d),
    ln2: norm(`h.${i}.ln_2`),
    mlpFc: linear(`h.${i}.mlp.c_fc`, dMlp, d),
    mlpProj: linear(`h.${i}.mlp.c_proj`, d, dMlp),
  }));
  const model: Model = {
    config,
    wte: matrix('wte.weight', vocabSize, d),
    wpe: matrix('wpe.weight', contextLength, d),
    blocks,
    lnFinal: norm('ln_f'),
  };
  const unexpected = [...tensors.keys()].filter((name) => !used.has(name));
  if (unexpected.length > 0) {
    throw new Error(`Model weights have unexpected tensors: ${unexpected.join(', ')}`);
  }
  return model;
}

/**
 * Loads a model file written by the Python exporter. The returned weights are views into
 * `buffer`, so it must not be transferred (for example to a Worker) or mutated afterward.
 */
export function loadModel(buffer: ArrayBuffer): Model {
  const { tensors, metadata } = parseSafetensors(buffer);
  if (metadata.format !== MODEL_FORMAT) {
    throw new Error(`Unsupported model format "${metadata.format}", expected "${MODEL_FORMAT}"`);
  }
  if (metadata.config === undefined) throw new Error('Model file has no config metadata');
  return modelFromTensors(parseConfig(metadata.config), tensors);
}
