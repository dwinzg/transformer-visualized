// Takes the README screenshots. Not run in CI.
//
//   npm run build -w site && npm run preview -w site -- --port 4321
//   node site/scripts/capture.mjs <output folder>
//
// Captures the home page, the introduction at Story and at Formula, and the glossary, in light and
// dark, at desktop and phone sizes.
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { chromium } from '@playwright/test';

const out = process.argv[2];
if (!out) {
  console.error('Usage: node site/scripts/capture.mjs <output folder>');
  process.exit(1);
}
mkdirSync(out, { recursive: true });

const BASE = 'http://localhost:4321/transformer-visualized/';
const PAGES = [
  { name: 'home', path: '' },
  { name: 'chapter-story', path: 'learn/introduction/?depth=story' },
  { name: 'chapter-formula', path: 'learn/introduction/?depth=formula' },
  { name: 'glossary', path: 'glossary/' },
];
const SIZES = [
  { name: 'desktop', viewport: { width: 1280, height: 800 } },
  { name: 'phone', viewport: { width: 390, height: 844 } },
];

const browser = await chromium.launch();
for (const size of SIZES) {
  for (const colorScheme of ['light', 'dark']) {
    const page = await browser.newPage({
      viewport: size.viewport,
      deviceScaleFactor: 2,
      colorScheme,
      reducedMotion: 'reduce',
    });
    for (const { name, path } of PAGES) {
      await page.goto(BASE + path, { waitUntil: 'networkidle' });
      const file = join(out, `${name}-${size.name}-${colorScheme}.png`);
      await page.screenshot({ path: file });
      console.log(file);
    }
    await page.close();
  }
}
await browser.close();
