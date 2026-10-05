import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadModel, Tokenizer } from '@transformer-visualized/engine';
import { GRID_TOKENS, runPlayground } from './playground-run';

const dir = resolve(process.cwd(), '../models/tiny');
const bytes = readFileSync(resolve(dir, 'model.safetensors'));
const model = loadModel(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const tokenizer = Tokenizer.fromJSON(
  JSON.parse(readFileSync(resolve(dir, 'tokenizer.json'), 'utf8')),
);

describe('runPlayground', () => {
  it('gives the starting sentence its tokens and real chances', () => {
    const run = runPlayground(model, tokenizer, 'Lily wanted to play with her');
    expect(run.tokens).toHaveLength(6);
    expect(run.views!.scores.top[0].token.text).toBe(' ball');
    expect(run.views!.attention.from).toBe(0);
  });

  it('gives the scores that softmax turns into the weights', () => {
    const { attention } = runPlayground(model, tokenizer, 'Lily wanted to play with her').views!;
    // Layer 3, head 3, the row for "her", which sees all 6 tokens.
    const scores = attention.scores![2][2][5];
    expect(scores).toHaveLength(6);
    expect(attention.scores![2][2][0]).toHaveLength(1);
    const exp = scores.map((s) => Math.exp(s));
    const total = exp.reduce((a, b) => a + b, 0);
    exp.forEach((e, j) => expect(e / total).toBeCloseTo(attention.weights[2][2][5][j], 1));
    expect(attention.weights[2][2][5][0]).toBe(0.66);
  });

  it('has no views for empty text', () => {
    expect(runPlayground(model, tokenizer, '').views).toBeNull();
  });

  it('keeps the end of a long text and cuts the grid to the last tokens', () => {
    const run = runPlayground(model, tokenizer, 'Lily ran to the park. '.repeat(40));
    expect(run.tokens).toHaveLength(model.config.contextLength);
    expect(run.cut).toBeGreaterThan(0);
    const { attention } = run.views!;
    expect(attention.tokens).toHaveLength(GRID_TOKENS);
    expect(attention.weights[0][0]).toHaveLength(GRID_TOKENS);
    expect(attention.weights[0][0][0]).toHaveLength(GRID_TOKENS);
    // Each kept row still holds only the tokens it can see.
    expect(attention.scores![0][0][0]).toHaveLength(1);
    expect(attention.scores![0][0][GRID_TOKENS - 1]).toHaveLength(GRID_TOKENS);
  });

  it('is plain data, so a worker can send it', () => {
    const run = runPlayground(model, tokenizer, 'The cat sat');
    expect(structuredClone(run)).toEqual(run);
  });
});
