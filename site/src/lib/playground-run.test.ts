import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadModel, Tokenizer } from '@transformer-visualized/engine';
import { forward, normalizeText } from '@transformer-visualized/engine';
import { attentionMath, GRID_TOKENS, runPlayground } from './playground-run';

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

  it("follows each token's residual stream through every step", () => {
    const { residual } = runPlayground(model, tokenizer, 'Lily wanted to play with her').views!;
    expect(residual).toHaveLength(6);
    const steps = residual[5];
    expect(steps.map((s) => s.label)).toEqual([
      'Start',
      'Block 1 attention',
      'Block 1 feed forward',
      'Block 2 attention',
      'Block 2 feed forward',
      'Block 3 attention',
      'Block 3 feed forward',
      'Block 4 attention',
      'Block 4 feed forward',
    ]);
    for (const step of steps) {
      expect(step.added).toBeGreaterThan(0);
      expect(step.length).toBeGreaterThan(0);
    }
    // Adding a vector changes the length by at most the length of what was added.
    for (let k = 1; k < steps.length; k++) {
      expect(Math.abs(steps[k].length - steps[k - 1].length)).toBeLessThanOrEqual(
        steps[k].added + 0.02,
      );
    }
  });

  it('explains a weight from the query and key, and agrees with the grid', () => {
    const text = 'Lily wanted to play with her';
    const trace = forward(model, tokenizer.encode(normalizeText(text)));
    const math = attentionMath(trace, { layer: 2, head: 2, row: 5, column: 0 });
    expect(math.query).toHaveLength(32);
    expect(math.key).toHaveLength(32);
    expect(math.value).toHaveLength(32);
    // Each product is rounded to 4 decimals, so 32 of them can drift by up to 0.0016.
    expect(Math.abs(math.products.reduce((a, b) => a + b, 0) - math.dot)).toBeLessThan(0.002);
    // The scale 1 / √32 is rounded too.
    expect(Math.abs(math.dot * math.scale - math.score)).toBeLessThan(0.002);
    expect(math.exp / math.expSum).toBeCloseTo(math.weight, 3);
    expect(math.weight.toFixed(2)).toBe('0.66');
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

  it('rescales each token in the final norm, matching PyTorch for "her"', () => {
    const { norm } = runPlayground(model, tokenizer, 'Lily wanted to play with her').views!;
    expect(norm).toHaveLength(6);
    const her = norm[5];
    // The same numbers the Final norm code on the map page prints.
    expect(her.values.map((v) => v.output)).toEqual([
      expect.closeTo(0.8167, 3),
      expect.closeTo(2.4633, 3),
      expect.closeTo(-3.1536, 3),
      expect.closeTo(8.8186, 3),
    ]);
    for (const v of her.values) {
      expect(v.normalized).toBeCloseTo((v.input - her.mean) / her.spread, 3);
      expect(v.output).toBeCloseTo(v.normalized * v.gamma + v.beta, 3);
    }
  });
});
