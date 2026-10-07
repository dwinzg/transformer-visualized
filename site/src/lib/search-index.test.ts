import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { plainText, stepsOf } from './search-index';

describe('plainText', () => {
  it('keeps the words and drops code, math, tags, expressions and Markdown marks', () => {
    const mdx = [
      'A model reads <Term id="token">tokens</Term> (<Ref id="radford2019" />).',
      '<Tex display tex="x^2" />',
      'For our sentence <Tex tex="n = 6" />.',
      "<PredictReveal options={['a', 'b']} answer={0}>",
      'It picks **one** of [the pieces](https://example.com) &quot;fast&quot;.',
      '```ts',
      'const secret = 1;',
      '```',
      '| Token | Id |',
      '| ----- | -- |',
    ].join('\n');
    expect(plainText(mdx)).toBe(
      'A model reads tokens. For our sentence n = 6. It picks one of the pieces "fast". Token Id',
    );
  });
});

describe('stepsOf', () => {
  it('gives each step the id its page gives it, and the text up to the next step', () => {
    const steps = stepsOf(
      'Intro\n<Step title="Same title">One</Step>\n<Step title="Same title" map>Two</Step>',
    );
    expect(steps).toEqual([
      { id: 'step-same-title', title: 'Same title', text: 'One' },
      { id: 'step-same-title-2', title: 'Same title', text: 'Two' },
    ]);
  });

  it('finds steps written in any attribute order, and none inside code', () => {
    const body =
      '<Step map="input" title="First">A</Step>\n```mdx\n<Step title="Fake">\n```\n<Step title="Second">B</Step>';
    expect(stepsOf(body).map((s) => s.title)).toEqual(['First', 'Second']);
  });

  it('finds every step of every chapter', () => {
    const dir = join(import.meta.dirname, '../content/chapters');
    for (const file of readdirSync(dir).filter((f) => f.endsWith('.mdx'))) {
      const body = readFileSync(join(dir, file), 'utf8');
      expect(stepsOf(body).length, file).toBe(body.match(/<Step\b/g)?.length ?? 0);
    }
  });

  it('finds every step of a real chapter, matching the step tags', () => {
    const body = readFileSync(join(import.meta.dirname, '../content/chapters/tokens.mdx'), 'utf8');
    const steps = stepsOf(body);
    expect(steps.map((s) => s.id)).toEqual([
      'step-text-becomes-tokens',
      'step-common-words-stay-whole-rare-words-split',
      'step-spaces-and-capitals-count',
      'step-how-the-list-was-made',
    ]);
    expect(steps[0].text).toContain('A model cannot read letters.');
    expect(steps.every((s) => !/[<>{}]/.test(s.text))).toBe(true);
  });
});
