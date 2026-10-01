import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, test } from 'vitest';
import { forward } from '../src/forward';
import { generate } from '../src/generate';
import { loadModel } from '../src/model';

const file = fileURLToPath(new URL('../../../models/tiny/model.safetensors', import.meta.url));
const model = loadModel(new Uint8Array(readFileSync(file)).buffer);
const prompt = Array.from({ length: 32 }, (_, i) => (i * 97 + 13) % model.config.vocabSize);

describe('tiny model', () => {
  test('forward, 32 tokens', async ({ bench }) => {
    await bench('forward, 32 tokens', () => {
      forward(model, prompt);
    }).run();
  });

  test('generate 20 tokens after 12', async ({ bench }) => {
    await bench('generate 20 tokens after 12', () => {
      generate(model, prompt.slice(0, 12), { maxNewTokens: 20, temperature: 1, seed: 0 });
    }).run();
  });
});
