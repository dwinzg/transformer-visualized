import { add, concatColumns, gatherRows, linear, mapValues, matmul, sliceColumns } from './linalg';
import type { BlockWeights, Model, ModelConfig } from './model';
import { causalScaled, geluTanh, layerNorm, softmaxRows } from './ops';
import type { Matrix } from './tensor';
import type { HeadTrace, LayerTrace, Trace } from './trace';

/** Runs the model on a token sequence and records every intermediate value. */
export function forward(model: Model, tokenIds: readonly number[]): Trace {
  const { config } = model;
  validateTokenIds(config, tokenIds);
  const tokenEmbeddings = gatherRows(model.wte, tokenIds);
  const positionEmbeddings = gatherRows(
    model.wpe,
    tokenIds.map((_, position) => position),
  );
  const embeddings = add(tokenEmbeddings, positionEmbeddings);
  const layers: LayerTrace[] = [];
  let residual = embeddings;
  for (const block of model.blocks) {
    const layer = runBlock(block, residual, config);
    layers.push(layer);
    residual = layer.output;
  }
  const lnFinal = layerNorm(
    residual,
    model.lnFinal.weight,
    model.lnFinal.bias,
    config.layerNormEps,
  );
  // The output projection reuses the token embedding table (weight tying).
  const logits = linear(lnFinal.out, model.wte);
  return {
    config,
    tokenIds: [...tokenIds],
    tokenEmbeddings,
    positionEmbeddings,
    embeddings,
    layers,
    lnFinal,
    logits,
  };
}

/** Throws unless the sequence is non-empty, fits the context and uses known token ids. */
export function validateTokenIds(config: ModelConfig, tokenIds: readonly number[]): void {
  if (tokenIds.length === 0) throw new RangeError('forward: needs at least one token');
  if (tokenIds.length > config.contextLength) {
    throw new RangeError(
      `forward: ${tokenIds.length} tokens exceed the context length of ${config.contextLength}`,
    );
  }
  for (const id of tokenIds) {
    if (!Number.isInteger(id) || id < 0 || id >= config.vocabSize) {
      throw new RangeError(
        `forward: token id ${id} is outside the vocabulary of ${config.vocabSize}`,
      );
    }
  }
}

function runBlock(block: BlockWeights, input: Matrix, config: ModelConfig): LayerTrace {
  const { dModel, dHead, nHeads, layerNormEps } = config;
  const ln1 = layerNorm(input, block.ln1.weight, block.ln1.bias, layerNormEps);
  const qkv = linear(ln1.out, block.attnQkv.weight, block.attnQkv.bias);
  const scale = 1 / Math.sqrt(dHead);
  const heads: HeadTrace[] = [];
  for (let h = 0; h < nHeads; h++) {
    const start = h * dHead;
    const q = sliceColumns(qkv, start, start + dHead);
    const k = sliceColumns(qkv, dModel + start, dModel + start + dHead);
    const v = sliceColumns(qkv, 2 * dModel + start, 2 * dModel + start + dHead);
    const scores = linear(q, k); // q·kᵀ
    const scaledMasked = causalScaled(scores, scale);
    const weights = softmaxRows(scaledMasked);
    heads.push({ q, k, v, scores, scaledMasked, weights, out: matmul(weights, v) });
  }
  const attnConcat = concatColumns(heads.map((head) => head.out));
  const attnOut = linear(attnConcat, block.attnProj.weight, block.attnProj.bias);
  const residAfterAttn = add(input, attnOut);
  return {
    input,
    ln1,
    heads,
    attnConcat,
    attnOut,
    residAfterAttn,
    ...runMlp(block, residAfterAttn, layerNormEps),
  };
}

/** LayerNorm, MLP and the residual add that close a block. The KV-cache decoder reuses it. */
export function runMlp(
  block: BlockWeights,
  residAfterAttn: Matrix,
  layerNormEps: number,
): Pick<LayerTrace, 'ln2' | 'mlpHidden' | 'mlpAct' | 'mlpOut' | 'output'> {
  const ln2 = layerNorm(residAfterAttn, block.ln2.weight, block.ln2.bias, layerNormEps);
  const mlpHidden = linear(ln2.out, block.mlpFc.weight, block.mlpFc.bias);
  const mlpAct = mapValues(mlpHidden, geluTanh);
  const mlpOut = linear(mlpAct, block.mlpProj.weight, block.mlpProj.bias);
  return { ln2, mlpHidden, mlpAct, mlpOut, output: add(residAfterAttn, mlpOut) };
}
