import { validateTokenIds } from './forward';
import { add, concatColumns, gatherRows, linear, matmul, sliceColumns } from './linalg';
import { parseConfig, type ModelConfig } from './model';
import { causalScaled, softmaxRows } from './ops';
import { parseSafetensors, type Tensor } from './safetensors';
import { createMatrix, type Matrix } from './tensor';
import type { FlatTensor, HeadTrace } from './trace';

/**
 * A Llama-style model: GPT-2's plan with RMSNorm, rotary positions (RoPE), a SwiGLU feed
 * forward and grouped-query attention, and no biases. The reference is model/tv_model/llama.py.
 */

/** Value of the `format` metadata key in Llama-style model files. */
export const LLAMA_FORMAT = 'transformer-visualized/llama/1';

export interface LlamaConfig extends ModelConfig {
  readonly arch: 'llama';
  /** Key and value heads. Each is shared by nHeads / nKvHeads query heads. */
  readonly nKvHeads: number;
  /** RoPE turns pair i of a vector at position p by p · ropeBase^(-2i / dHead). */
  readonly ropeBase: number;
}

export interface LlamaBlockWeights {
  readonly norm1: Float32Array;
  /** [nHeads·dHead, dModel] */
  readonly wq: Matrix;
  /** [nKvHeads·dHead, dModel] */
  readonly wk: Matrix;
  readonly wv: Matrix;
  /** [dModel, nHeads·dHead] */
  readonly wo: Matrix;
  readonly norm2: Float32Array;
  /** [dMlp, dModel] */
  readonly wGate: Matrix;
  readonly wUp: Matrix;
  /** [dModel, dMlp] */
  readonly wDown: Matrix;
}

export interface LlamaModel {
  readonly config: LlamaConfig;
  /** Token embeddings [vocab, dModel], also the output projection. There is no position table. */
  readonly wte: Matrix;
  readonly blocks: readonly LlamaBlockWeights[];
  readonly normFinal: Float32Array;
}

/** Per-row mean of the squares, and the output, of an RMSNorm. */
export interface RmsNormTrace {
  readonly meanSquare: Float32Array;
  readonly out: Matrix;
}

export interface LlamaHeadTrace extends HeadTrace {
  /** The query and key before they are turned by position. `q` and `k` are after. */
  readonly qBeforeRope: Matrix;
  readonly kBeforeRope: Matrix;
}

export interface LlamaLayerTrace {
  readonly input: Matrix;
  readonly ln1: RmsNormTrace;
  readonly heads: readonly LlamaHeadTrace[];
  readonly attnConcat: Matrix;
  readonly attnOut: Matrix;
  readonly residAfterAttn: Matrix;
  readonly ln2: RmsNormTrace;
  /** The two projections of the feed forward, and SiLU(gate) × up. */
  readonly mlpGate: Matrix;
  readonly mlpUp: Matrix;
  readonly mlpAct: Matrix;
  readonly mlpOut: Matrix;
  readonly output: Matrix;
}

export interface LlamaTrace {
  readonly config: LlamaConfig;
  readonly tokenIds: readonly number[];
  readonly tokenEmbeddings: Matrix;
  /** The same as tokenEmbeddings. Position enters inside attention instead. */
  readonly embeddings: Matrix;
  readonly layers: readonly LlamaLayerTrace[];
  readonly lnFinal: RmsNormTrace;
  readonly logits: Matrix;
}

/** RMSNorm over each row, x / sqrt(mean(x²) + eps) · gamma. */
export function rmsNorm(x: Matrix, gamma: Float32Array, eps: number): RmsNormTrace {
  if (gamma.length !== x.cols) {
    throw new RangeError(`rmsNorm: expected ${x.cols} scale values, got ${gamma.length}`);
  }
  const meanSquare = new Float32Array(x.rows);
  const out = createMatrix(x.rows, x.cols);
  for (let r = 0; r < x.rows; r++) {
    const offset = r * x.cols;
    let sum = 0;
    for (let c = 0; c < x.cols; c++) sum += x.data[offset + c] ** 2;
    meanSquare[r] = sum / x.cols;
    const inv = 1 / Math.sqrt(meanSquare[r] + eps);
    for (let c = 0; c < x.cols; c++) out.data[offset + c] = x.data[offset + c] * inv * gamma[c];
  }
  return { meanSquare, out };
}

/**
 * Turns each pair of columns (2i, 2i+1) of row p by the angle p · base^(-2i / cols). Row p is the
 * token at position p. A turn keeps each pair's length, and the dot product of two turned
 * vectors depends only on how far apart their positions are.
 */
export function rope(x: Matrix, base: number): Matrix {
  if (x.cols % 2 !== 0)
    throw new RangeError(`rope: needs an even number of columns, got ${x.cols}`);
  const out = createMatrix(x.rows, x.cols);
  for (let p = 0; p < x.rows; p++) {
    for (let i = 0; i < x.cols; i += 2) {
      // Math.fround matches the float32 angles of the PyTorch reference.
      const angle = Math.fround(p * Math.fround(base ** (-i / x.cols)));
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const a = x.data[p * x.cols + i];
      const b = x.data[p * x.cols + i + 1];
      out.data[p * x.cols + i] = a * cos - b * sin;
      out.data[p * x.cols + i + 1] = a * sin + b * cos;
    }
  }
  return out;
}

/** SiLU, also called swish, x · sigmoid(x). */
export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

/** Runs the Llama-style model on a token sequence and records every intermediate value. */
export function forwardLlama(model: LlamaModel, tokenIds: readonly number[]): LlamaTrace {
  const { config } = model;
  validateTokenIds(config, tokenIds);
  const tokenEmbeddings = gatherRows(model.wte, tokenIds);
  const layers: LlamaLayerTrace[] = [];
  let residual = tokenEmbeddings;
  for (const block of model.blocks) {
    const layer = runLlamaBlock(block, residual, config);
    layers.push(layer);
    residual = layer.output;
  }
  const lnFinal = rmsNorm(residual, model.normFinal, config.layerNormEps);
  return {
    config,
    tokenIds: [...tokenIds],
    tokenEmbeddings,
    embeddings: tokenEmbeddings,
    layers,
    lnFinal,
    logits: linear(lnFinal.out, model.wte),
  };
}

function runLlamaBlock(block: LlamaBlockWeights, input: Matrix, config: LlamaConfig) {
  const { dHead, nHeads, nKvHeads, ropeBase, layerNormEps } = config;
  const ln1 = rmsNorm(input, block.norm1, layerNormEps);
  const qAll = linear(ln1.out, block.wq);
  const kAll = linear(ln1.out, block.wk);
  const vAll = linear(ln1.out, block.wv);
  const group = nHeads / nKvHeads;
  const kv = Array.from({ length: nKvHeads }, (_, g) => {
    const kBeforeRope = sliceColumns(kAll, g * dHead, (g + 1) * dHead);
    return {
      kBeforeRope,
      k: rope(kBeforeRope, ropeBase),
      v: sliceColumns(vAll, g * dHead, (g + 1) * dHead),
    };
  });
  const scale = 1 / Math.sqrt(dHead);
  const heads: LlamaHeadTrace[] = [];
  for (let h = 0; h < nHeads; h++) {
    const qBeforeRope = sliceColumns(qAll, h * dHead, (h + 1) * dHead);
    const q = rope(qBeforeRope, ropeBase);
    // Query head h reads the key and value of head h / group, shared with its neighbors.
    const { kBeforeRope, k, v } = kv[Math.floor(h / group)];
    const scores = linear(q, k);
    const scaledMasked = causalScaled(scores, scale);
    const weights = softmaxRows(scaledMasked);
    heads.push({
      qBeforeRope,
      kBeforeRope,
      q,
      k,
      v,
      scores,
      scaledMasked,
      weights,
      out: matmul(weights, v),
    });
  }
  const attnConcat = concatColumns(heads.map((head) => head.out));
  const attnOut = linear(attnConcat, block.wo);
  const residAfterAttn = add(input, attnOut);
  const ln2 = rmsNorm(residAfterAttn, block.norm2, layerNormEps);
  const mlpGate = linear(ln2.out, block.wGate);
  const mlpUp = linear(ln2.out, block.wUp);
  const mlpAct = createMatrix(mlpGate.rows, mlpGate.cols);
  for (let i = 0; i < mlpAct.data.length; i++)
    mlpAct.data[i] = silu(mlpGate.data[i]) * mlpUp.data[i];
  const mlpOut = linear(mlpAct, block.wDown);
  return {
    input,
    ln1,
    heads,
    attnConcat,
    attnOut,
    residAfterAttn,
    ln2,
    mlpGate,
    mlpUp,
    mlpAct,
    mlpOut,
    output: add(residAfterAttn, mlpOut),
  } satisfies LlamaLayerTrace;
}

/** Reads the config of a Llama-style file: the GPT-2 fields plus arch, nKvHeads and ropeBase. */
export function parseLlamaConfig(json: string): LlamaConfig {
  const base = parseConfig(json);
  const raw = JSON.parse(json) as Record<string, unknown>;
  const { arch, nKvHeads, ropeBase } = raw;
  if (arch !== 'llama') throw new Error(`Model config arch must be "llama", got ${String(arch)}`);
  if (typeof nKvHeads !== 'number' || !Number.isInteger(nKvHeads) || nKvHeads <= 0) {
    throw new Error('Model config field "nKvHeads" must be a positive integer');
  }
  if (base.nHeads % nKvHeads !== 0) {
    throw new Error(`Model config nHeads ${base.nHeads} is not a multiple of nKvHeads ${nKvHeads}`);
  }
  if (typeof ropeBase !== 'number' || !Number.isFinite(ropeBase) || ropeBase <= 1) {
    throw new Error('Model config field "ropeBase" must be a number above 1');
  }
  if (base.dHead % 2 !== 0)
    throw new Error(`Rotary positions need an even dHead, got ${base.dHead}`);
  return { ...base, arch, nKvHeads, ropeBase };
}

/** Builds a Llama-style model from named tensors, checking every name and shape. */
export function llamaFromTensors(
  config: LlamaConfig,
  tensors: ReadonlyMap<string, Tensor>,
): LlamaModel {
  const { vocabSize, dModel: d, dMlp, nLayers, nHeads, nKvHeads, dHead } = config;
  const used = new Set<string>();
  const take = (name: string, shape: readonly number[]): Float32Array => {
    const tensor = tensors.get(name);
    if (tensor === undefined) throw new Error(`Model weights are missing "${name}"`);
    if (tensor.shape.length !== shape.length || tensor.shape.some((s, i) => s !== shape[i])) {
      throw new Error(
        `Weight "${name}" has shape [${tensor.shape.join(', ')}], expected [${shape.join(', ')}]`,
      );
    }
    used.add(name);
    return tensor.data;
  };
  const matrix = (name: string, rows: number, cols: number) =>
    createMatrix(rows, cols, take(name, [rows, cols]));
  const blocks = Array.from({ length: nLayers }, (_, i): LlamaBlockWeights => ({
    norm1: take(`h.${i}.norm_1.weight`, [d]),
    wq: matrix(`h.${i}.attn.wq.weight`, nHeads * dHead, d),
    wk: matrix(`h.${i}.attn.wk.weight`, nKvHeads * dHead, d),
    wv: matrix(`h.${i}.attn.wv.weight`, nKvHeads * dHead, d),
    wo: matrix(`h.${i}.attn.wo.weight`, d, nHeads * dHead),
    norm2: take(`h.${i}.norm_2.weight`, [d]),
    wGate: matrix(`h.${i}.mlp.w_gate.weight`, dMlp, d),
    wUp: matrix(`h.${i}.mlp.w_up.weight`, dMlp, d),
    wDown: matrix(`h.${i}.mlp.w_down.weight`, d, dMlp),
  }));
  const model: LlamaModel = {
    config,
    wte: matrix('wte.weight', vocabSize, d),
    blocks,
    normFinal: take('norm_f.weight', [d]),
  };
  const unexpected = [...tensors.keys()].filter((name) => !used.has(name));
  if (unexpected.length > 0) {
    throw new Error(`Model weights have unexpected tensors: ${unexpected.join(', ')}`);
  }
  return model;
}

/** Loads a Llama-style model file. The weights are views into `buffer`. */
export function loadLlama(buffer: ArrayBuffer): LlamaModel {
  const { tensors, metadata } = parseSafetensors(buffer);
  if (metadata.format !== LLAMA_FORMAT) {
    throw new Error(`Unsupported model format "${metadata.format}", expected "${LLAMA_FORMAT}"`);
  }
  if (metadata.config === undefined) throw new Error('Model file has no config metadata');
  return llamaFromTensors(parseLlamaConfig(metadata.config), tensors);
}

/** Lists every traced value under its path, matching the Python reference fixtures. */
export function flattenLlamaTrace(trace: LlamaTrace): Map<string, FlatTensor> {
  const flat = new Map<string, FlatTensor>();
  const matrix = (path: string, m: Matrix) =>
    flat.set(path, { shape: [m.rows, m.cols], data: m.data });
  const norm = (path: string, n: RmsNormTrace) => {
    flat.set(`${path}.meanSquare`, { shape: [n.meanSquare.length], data: n.meanSquare });
    matrix(`${path}.out`, n.out);
  };
  matrix('tokenEmbeddings', trace.tokenEmbeddings);
  matrix('embeddings', trace.embeddings);
  trace.layers.forEach((layer, l) => {
    const p = `layers.${l}`;
    matrix(`${p}.input`, layer.input);
    norm(`${p}.ln1`, layer.ln1);
    layer.heads.forEach((head, h) => {
      for (const key of [
        'qBeforeRope',
        'kBeforeRope',
        'q',
        'k',
        'v',
        'scores',
        'scaledMasked',
        'weights',
        'out',
      ] as const) {
        matrix(`${p}.heads.${h}.${key}`, head[key]);
      }
    });
    for (const key of [
      'attnConcat',
      'attnOut',
      'residAfterAttn',
      'mlpGate',
      'mlpUp',
      'mlpAct',
      'mlpOut',
      'output',
    ] as const) {
      matrix(`${p}.${key}`, layer[key]);
    }
    norm(`${p}.ln2`, layer.ln2);
  });
  norm('lnFinal', trace.lnFinal);
  matrix('logits', trace.logits);
  return flat;
}
