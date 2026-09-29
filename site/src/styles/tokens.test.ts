import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { contrastRatio, deltaE2000, deltaE2000Lab } from './contrast';

const css = readFileSync(new URL('./tokens.css', import.meta.url), 'utf8');

/** Custom properties declared in the block whose selector starts with `selector`. */
function block(selector: string): Record<string, string> {
  const start = css.indexOf(selector);
  if (start === -1) throw new Error(`No block for ${selector}`);
  const open = css.indexOf('{', start);
  const close = css.indexOf('}', open);
  const values: Record<string, string> = {};
  for (const [, name, value] of css.slice(open + 1, close).matchAll(/(--[\w-]+):\s*([^;]+);/g)) {
    values[name] = value.trim();
  }
  return values;
}

const themes = {
  light: block(":root,\n:root[data-theme='light']"),
  dark: block(":root[data-theme='dark']"),
};

const backgrounds = ['--color-bg', '--color-surface', '--color-surface-2'];

const textTokens = [
  '--color-text',
  '--color-text-secondary',
  '--color-accent',
  '--color-success-text',
  '--color-danger-text',
  '--concept-query-text',
  '--concept-key-text',
  '--concept-value-text',
  '--concept-residual-text',
  '--concept-output-text',
  '--concept-position-text',
  '--code-comment',
  '--code-keyword',
  '--code-string',
  '--code-function',
  '--code-constant',
];

const concepts = ['query', 'key', 'value', 'residual', 'output', 'position'];

describe('contrastRatio', () => {
  it('matches known WCAG values', () => {
    expect(contrastRatio('#000000', '#ffffff')).toBeCloseTo(21, 5);
    expect(contrastRatio('#ffffff', '#ffffff')).toBeCloseTo(1, 5);
    expect(contrastRatio('#777777', '#ffffff')).toBeCloseTo(4.48, 2);
  });
});

describe('deltaE2000', () => {
  it('matches the CIEDE2000 test data of Sharma, Wu and Dalal (2005)', () => {
    expect(deltaE2000Lab([50, 2.6772, -79.7751], [50, 0, -82.7485])).toBeCloseTo(2.0425, 4);
    expect(deltaE2000Lab([50, 2.5, 0], [73, 25, -18])).toBeCloseTo(27.1492, 4);
    expect(deltaE2000Lab([50, 2.49, -0.001], [50, -2.49, 0.0009])).toBeCloseTo(7.1792, 4);
  });

  it('is zero for the same color', () => {
    expect(deltaE2000('#0072b2', '#0072b2')).toBe(0);
  });
});

describe.each(Object.entries(themes))('%s theme', (_name, tokens) => {
  it.each(textTokens)('%s is readable on the page and on surfaces', (token) => {
    for (const background of backgrounds) {
      const ratio = contrastRatio(tokens[token], tokens[background]);
      expect(ratio, `${token} on ${background}`).toBeGreaterThanOrEqual(4.5);
    }
  });

  it('keeps button text readable on the accent color', () => {
    expect(
      contrastRatio(tokens['--color-accent-contrast'], tokens['--color-accent']),
    ).toBeGreaterThanOrEqual(4.5);
  });

  it('keeps control borders visible on the page and on surfaces', () => {
    for (const background of backgrounds) {
      const ratio = contrastRatio(tokens['--color-control-border'], tokens[background]);
      expect(ratio, `--color-control-border on ${background}`).toBeGreaterThanOrEqual(3);
    }
  });

  it('keeps the focus ring visible on the page and on surfaces', () => {
    for (const background of backgrounds) {
      const ratio = contrastRatio(tokens['--color-focus'], tokens[background]);
      expect(ratio, `--color-focus on ${background}`).toBeGreaterThanOrEqual(3);
    }
  });

  it.each([
    ['base', ''],
    ['text', '-text'],
  ])('keeps the %s concept colors apart', (_kind, suffix) => {
    for (const [i, first] of concepts.entries()) {
      for (const second of concepts.slice(i + 1)) {
        const difference = deltaE2000(
          tokens[`--concept-${first}${suffix}`],
          tokens[`--concept-${second}${suffix}`],
        );
        expect(difference, `${first} and ${second}${suffix}`).toBeGreaterThanOrEqual(10);
      }
    }
  });
});

describe('dark theme', () => {
  it('declares the same values for the media query and the explicit dark theme', () => {
    const media = block(":root:not([data-theme='light'])");
    expect(media).toEqual(themes.dark);
  });
});

describe('depth dial selected segment', () => {
  const depthDialSource = readFileSync(
    new URL('../components/DepthDial.astro', import.meta.url),
    'utf8',
  );
  // Passes either way: a strong enough contrast against the track, or a non-color cue such as a
  // border on the selected segment (checked once, since the same rule applies in both themes).
  const hasNonColorCue = /input:checked \+ span\s*{[^}]*\bborder:/s.test(depthDialSource);

  it.each(Object.entries(themes))(
    'is distinguishable from the track in the %s theme',
    (_name, tokens) => {
      const ratio = contrastRatio(tokens['--color-segment-selected'], tokens['--color-surface-2']);
      expect(ratio >= 1.5 || hasNonColorCue).toBe(true);
    },
  );
});
