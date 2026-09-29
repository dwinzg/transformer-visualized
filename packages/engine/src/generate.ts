import { forward, runMlp, validateTokenIds } from './forward';
import { add, concatColumns, gatherRows, linear } from './linalg';
import type { Model } from './model';
import { layerNorm, softmax } from './ops';
import { argmax, createRng, nextTokenDistribution, sample, type SamplingOptions } from './sampling';
import { createMatrix, rowView, type Matrix } from './tensor';

/** Processes a sequence one token at a time, reusing earlier keys and values (the KV cache). */
export interface Decoder {
  /** Tokens processed so far. */
  readonly length: number;
  /** Logits for the token that comes next. */
  readonly logits: Float32Array;
  /** Processes one more token and returns the logits for the token after it. */
  step(tokenId: number): Float32Array;
}

interface HeadCache {
  readonly keys: Float32Array; // [contextLength, dHead]
  readonly values: Float32Array; // [contextLength, dHead]
}

/** Runs the prompt once, keeps its keys and values, and returns a decoder for the rest. */
export function createDecoder(model: Model, prompt: readonly number[]): Decoder {
  const { config } = model;
  const trace = forward(model, prompt);
  const cache: HeadCache[][] = trace.layers.map((layer) =>
    layer.heads.map((head) => {
      const keys = new Float32Array(config.contextLength * config.dHead);
      const values = new Float32Array(config.contextLength * config.dHead);
      keys.set(head.k.data);
      values.set(head.v.data);
      return { keys, values };
    }),
  );
  let length = prompt.length;
  let logits: Float32Array = rowView(trace.logits, prompt.length - 1).slice();
  return {
    get length() {
      return length;
    },
    get logits() {
      return logits;
    },
    step(tokenId: number): Float32Array {
      if (length >= config.contextLength) {
        throw new RangeError(`decoder: the context of ${config.contextLength} tokens is full`);
      }
      validateTokenIds(config, [tokenId]);
      logits = decodeOne(model, cache, tokenId, length);
      length += 1;
      return logits;
    },
  };
}

/**
 * One transformer pass for a single new token at `position`. Mirrors `forward`, including where
 * values are rounded to float32, so cached decoding reproduces a full forward pass.
 */
function decodeOne(
  model: Model,
  cache: HeadCache[][],
  tokenId: number,
  position: number,
): Float32Array {
  const { dModel, dHead, nHeads, layerNormEps } = model.config;
  const scale = 1 / Math.sqrt(dHead);
  const count = position + 1;
  let x = add(gatherRows(model.wte, [tokenId]), gatherRows(model.wpe, [position]));
  model.blocks.forEach((block, l) => {
    const ln1 = layerNorm(x, block.ln1.weight, block.ln1.bias, layerNormEps);
    const qkv = linear(ln1.out, block.attnQkv.weight, block.attnQkv.bias);
    const headOutputs: Matrix[] = [];
    for (let h = 0; h < nHeads; h++) {
      const start = h * dHead;
      const { keys, values } = cache[l][h];
      keys.set(qkv.data.subarray(dModel + start, dModel + start + dHead), position * dHead);
      values.set(
        qkv.data.subarray(2 * dModel + start, 2 * dModel + start + dHead),
        position * dHead,
      );
      const q = qkv.data.subarray(start, start + dHead);
      const scaled = new Float32Array(count);
      for (let j = 0; j < count; j++) {
        let dot = 0;
        for (let c = 0; c < dHead; c++) dot += q[c] * keys[j * dHead + c];
        scaled[j] = Math.fround(dot) * scale;
      }
      const weights = softmax(scaled);
      const out = createMatrix(1, dHead);
      for (let c = 0; c < dHead; c++) {
        let sum = 0;
        for (let j = 0; j < count; j++) sum += weights[j] * values[j * dHead + c];
        out.data[c] = sum;
      }
      headOutputs.push(out);
    }
    const attnOut = linear(concatColumns(headOutputs), block.attnProj.weight, block.attnProj.bias);
    x = runMlp(block, add(x, attnOut), layerNormEps).output;
  });
  const lnFinal = layerNorm(x, model.lnFinal.weight, model.lnFinal.bias, layerNormEps);
  return linear(lnFinal.out, model.wte).data;
}

export interface GenerateOptions extends SamplingOptions {
  readonly maxNewTokens: number;
  /** Seed for the random number generator, so a run can be repeated exactly. */
  readonly seed: number;
  /** Always take the most likely token instead of sampling. */
  readonly greedy?: boolean;
  /** Stop right after generating this token, for example the end-of-text token. */
  readonly stopTokenId?: number;
}

export interface GenerateStep {
  readonly tokenId: number;
  /** Probability of the chosen token in `distribution`. */
  readonly probability: number;
  /** The distribution after temperature, top-k and top-p. */
  readonly distribution: Float32Array;
}

export type StopReason = 'maxNewTokens' | 'stopToken' | 'contextFull';

export interface GenerateResult {
  /** The prompt followed by the generated tokens. Never longer than the context length. */
  readonly tokenIds: readonly number[];
  readonly steps: readonly GenerateStep[];
  readonly stopReason: StopReason;
}

/** Generates up to `maxNewTokens` tokens after the prompt. */
export function generate(
  model: Model,
  prompt: readonly number[],
  options: GenerateOptions,
): GenerateResult {
  const { maxNewTokens } = options;
  if (!Number.isInteger(maxNewTokens) || maxNewTokens < 0) {
    throw new RangeError(`maxNewTokens must be a non-negative integer, got ${maxNewTokens}`);
  }
  const decoder = createDecoder(model, prompt);
  const random = createRng(options.seed);
  const tokenIds = [...prompt];
  const steps: GenerateStep[] = [];
  let logits = decoder.logits;
  for (;;) {
    if (steps.length === maxNewTokens) return { tokenIds, steps, stopReason: 'maxNewTokens' };
    if (tokenIds.length >= model.config.contextLength) {
      return { tokenIds, steps, stopReason: 'contextFull' };
    }
    const distribution = nextTokenDistribution(logits, options);
    const tokenId = options.greedy === true ? argmax(logits) : sample(distribution, random);
    steps.push({ tokenId, probability: distribution[tokenId], distribution });
    tokenIds.push(tokenId);
    if (tokenId === options.stopTokenId) return { tokenIds, steps, stopReason: 'stopToken' };
    if (steps.length < maxNewTokens && tokenIds.length < model.config.contextLength) {
      logits = decoder.step(tokenId);
    }
  }
}
