import { describe, expect, it } from 'vitest';
import { llamaTraceOf, traceOf } from './demo-data';
import { BLOCKS, TOUR } from './architecture-layout';
import { LLAMA_PYTORCH, LLAMA_SECOND_NORM, PYTORCH, SECOND_NORM } from './pytorch';

// These are exactly what the snippets printed when run in PyTorch 2.14 on models/tiny, after SETUP and in
// GPT-2 tour order. If the model or a snippet changes, run them again and update this list.
const PRINTED = {
  input: 'tensor([665, 408, 266, 324, 329, 336])',
  embedding: 'tensor([-0.0367,  0.0018, -0.0707,  0.0400])',
  position: 'tensor([-0.2318, -0.0100, -0.0560,  0.0446])',
  'add-norm': 'tensor([-1.7094, -0.0382, -0.1882,  0.3310])',
  'masked-attn': 'tensor([0.1737, 0.1825, 0.1723, 0.1594, 0.1660, 0.1461])',
  ffn: 'tensor([ 2.7906, -0.2493, -0.4626,  0.2078])',
  stack: 'tensor([ 0.1517,  0.9026, -0.8536,  2.3556])',
  'final-norm': 'tensor([ 0.8167,  2.4633, -3.1536,  8.8186])',
  linear: 'tensor([ -4.9049,  -4.8017,  -5.7243, -10.8453])',
  softmax: 'tensor([476, 411, 446])',
  output: 'tensor(476)',
} as const;

describe('PyTorch snippets', () => {
  const trace = traceOf('Lily wanted to play with her');

  it.each(Object.entries(PRINTED))('%s prints what PyTorch printed', (id, printed) => {
    expect(PYTORCH[id as keyof typeof PRINTED]!.prints!(trace)).toBe(printed);
  });

  it('prints ln_2 at the second norm, after attention changed x', () => {
    expect(SECOND_NORM.prints!(trace)).toBe('tensor([-0.0422, -0.5109, -0.2534,  0.5091])');
  });

  it('has numbers to print at every stop of the GPT-2 tour', () => {
    for (const id of TOUR.gpt2) {
      const part = BLOCKS.find((b) => b.id === id)!.part;
      expect(PYTORCH[part]?.prints, part).toBeTypeOf('function');
    }
  });

  it("matches the paper's sine waves at position 1", () => {
    expect(PYTORCH.position!.paper!.prints).toBe('tensor([0.8415, 0.5403, 0.7617, 0.6479])');
  });

  it('gives every snippet a shape', () => {
    for (const snippet of Object.values(PYTORCH)) expect(snippet.shape).not.toBe('');
  });
});

// What the Llama snippets printed in PyTorch 2.14 on models/llama-tiny, after LLAMA_SETUP and in
// Llama tour order. If the model or a snippet changes, run them again and update this list.
const LLAMA_PRINTED = {
  input: 'tensor([665, 408, 266, 324, 329, 336])',
  embedding: 'tensor([0.0278, 0.0120, 0.0096, 0.0103])',
  'add-norm': 'tensor([0.1853, 0.1066, 0.0680, 0.0590])',
  'masked-attn': 'tensor([0.0291, 0.1399, 0.0707, 0.2925, 0.1634, 0.3044])',
  ffn: 'tensor([ 0.2635, -0.2112, -0.2429, -0.2152])',
  stack: 'tensor([ 0.3568, -0.2615,  0.3003, -0.1520])',
  'final-norm': 'tensor([ 1.4549, -1.3055,  1.3374, -0.7040])',
  linear: 'tensor([-6.5304, -3.3971, -7.6321, -9.7709])',
  softmax: 'tensor([615, 446, 544])',
  output: 'tensor(615)',
} as const;

describe('Llama PyTorch snippets', () => {
  const trace = llamaTraceOf('Lily wanted to play with her');

  it.each(Object.entries(LLAMA_PRINTED))('%s prints what PyTorch printed', (id, printed) => {
    expect(LLAMA_PYTORCH[id as keyof typeof LLAMA_PRINTED]!.prints!(trace)).toBe(printed);
  });

  it('prints norm_2 at the second norm', () => {
    expect(LLAMA_SECOND_NORM.prints!(trace)).toBe('tensor([0.4416, 0.2726, 0.1475, 0.2157])');
  });

  it('has numbers to print at every stop of the Llama tour', () => {
    for (const id of TOUR.llama) {
      const part = BLOCKS.find((b) => b.id === id)!.part;
      expect(LLAMA_PYTORCH[part]?.prints, part).toBeTypeOf('function');
    }
  });
});
