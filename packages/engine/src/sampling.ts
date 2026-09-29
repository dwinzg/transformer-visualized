import { softmax } from './ops';

export interface SamplingOptions {
  /** Logits are divided by this before softmax. Below 1 sharpens, above 1 flattens. */
  readonly temperature: number;
  /** Keep only the k most likely tokens. */
  readonly topK?: number;
  /** Keep the fewest most-likely tokens whose probabilities add up to at least p. */
  readonly topP?: number;
}

/** Throws unless the temperature is a positive, finite number. */
export function assertTemperature(temperature: number): void {
  if (!(temperature > 0) || !Number.isFinite(temperature)) {
    throw new RangeError(`temperature must be a positive number, got ${temperature}`);
  }
}

/** softmax(logits / temperature). */
export function probabilities(logits: ArrayLike<number>, temperature: number): Float32Array {
  assertTemperature(temperature);
  const scaled = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
  return softmax(scaled);
}

/** Keeps the k most likely tokens (ties go to the lower id) and renormalizes. */
export function topKFilter(probs: Float32Array, k: number): Float32Array {
  if (!Number.isInteger(k) || k < 1)
    throw new RangeError(`topK must be a positive integer, got ${k}`);
  return renormalize(probs, rankByProbability(probs).slice(0, k));
}

/** Keeps the most likely tokens until their total reaches p (always at least one), renormalized. */
export function topPFilter(probs: Float32Array, p: number): Float32Array {
  if (!(p > 0 && p <= 1)) throw new RangeError(`topP must be in (0, 1], got ${p}`);
  const keep: number[] = [];
  let total = 0;
  for (const id of rankByProbability(probs)) {
    keep.push(id);
    total += probs[id];
    if (total >= p) break;
  }
  return renormalize(probs, keep);
}

/** The distribution the next token is drawn from. Temperature, then top-k, then top-p. */
export function nextTokenDistribution(
  logits: ArrayLike<number>,
  options: SamplingOptions,
): Float32Array {
  let probs = probabilities(logits, options.temperature);
  if (options.topK !== undefined) probs = topKFilter(probs, options.topK);
  if (options.topP !== undefined) probs = topPFilter(probs, options.topP);
  return probs;
}

/** Index of the largest value, the first one on ties. */
export function argmax(values: ArrayLike<number>): number {
  if (values.length === 0) throw new RangeError('argmax: no values');
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/** Draws an index with probability proportional to `probs`, using `random()` in [0, 1). */
export function sample(probs: ArrayLike<number>, random: () => number): number {
  let total = 0;
  for (let i = 0; i < probs.length; i++) total += probs[i];
  if (!(total > 0)) throw new RangeError('sample: probabilities must have a positive sum');
  const target = random() * total;
  let cumulative = 0;
  let last = -1;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] <= 0) continue;
    cumulative += probs[i];
    last = i;
    if (target < cumulative) return i;
  }
  // Rounding can leave the target a hair above the final cumulative sum.
  return last;
}

/** A small seedable random number generator (mulberry32). Returns numbers in [0, 1). */
export function createRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rankByProbability(probs: Float32Array): number[] {
  return Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a] || a - b);
}

function renormalize(probs: Float32Array, keep: readonly number[]): Float32Array {
  const out = new Float32Array(probs.length);
  let total = 0;
  for (const id of keep) total += probs[id];
  for (const id of keep) out[id] = probs[id] / total;
  return out;
}
