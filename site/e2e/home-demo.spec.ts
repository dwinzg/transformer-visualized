import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

test('the demo plays on its own', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  const chips = demo.locator('.token-chip');
  const before = await chips.count();
  await expect(chips).toHaveCount(before + 1, { timeout: 5000 });
});

test('picking a guess adds it, pauses and announces it', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  const option = demo.getByRole('option').nth(1);
  const word = ((await option.locator('.prob-word').textContent()) ?? '').trim();
  await option.click();
  await expect(demo.getByRole('button', { name: 'Play demo' })).toBeVisible();
  await expect(demo.locator('[aria-live="polite"]')).toContainText(`Added '${word}'`);
});

test('the pause button stops autoplay', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.getByRole('button', { name: 'Pause demo' }).click();
  const count = await demo.locator('.token-chip').count();
  await page.waitForTimeout(2500);
  await expect(demo.locator('.token-chip')).toHaveCount(count);
});

test('hovering the demo stops autoplay', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.hover();
  const count = await demo.locator('.token-chip').count();
  await page.waitForTimeout(2500);
  await expect(demo.locator('.token-chip')).toHaveCount(count);
});

test('reduced motion starts paused, and Play still works', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await expect(demo.getByRole('button', { name: 'Play demo' })).toBeVisible();
  const count = await demo.locator('.token-chip').count();
  await demo.getByRole('button', { name: 'Play demo' }).click();
  await expect(demo.locator('.token-chip')).toHaveCount(count + 1, { timeout: 5000 });
});

test('the end of the guesses offers to start over', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.getByRole('button', { name: 'Pause demo' }).click();
  for (let i = 0; i < 3; i++) await demo.getByRole('option').first().click();
  await expect(demo.getByRole('option')).toHaveCount(0);
  const start = demo.getByRole('button', { name: 'Start over' });
  await expect(start).toBeVisible();
  const count = await demo.locator('.token-chip').count();
  await start.click();
  await expect(demo.locator('.token-chip')).toHaveCount(count - 3);
});

test('the home page is accessible in both themes and does not scroll sideways', async ({
  page,
}) => {
  await page.goto('./');
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
  await page.setViewportSize({ width: 320, height: 640 });
  await expectNoHorizontalScroll(page);
});

test('the home page ships at most 40 KB of gzipped JavaScript', () => {
  const dist = resolve(process.cwd(), 'dist');
  const html = readFileSync(resolve(dist, 'index.html'), 'utf8');
  const sources = new Set<string>();
  for (const [, src] of html.matchAll(
    /(?:src|component-url|renderer-url)="\/transformer-visualized\/([^"]+\.js)"/g,
  )) {
    sources.add(src);
  }
  // Static imports of those entry chunks count too.
  const queue = [...sources];
  while (queue.length > 0) {
    const file = queue.pop()!;
    const code = readFileSync(resolve(dist, file), 'utf8');
    for (const [, dep] of code.matchAll(/from\s*"\.\/([^"]+\.js)"/g)) {
      const path = file.replace(/[^/]+$/, '') + dep;
      if (!sources.has(path)) {
        sources.add(path);
        queue.push(path);
      }
    }
  }
  const bytes = [...sources].reduce(
    (total, file) => total + gzipSync(readFileSync(resolve(dist, file))).length,
    0,
  );
  expect(sources.size).toBeGreaterThan(0);
  expect(bytes).toBeLessThanOrEqual(40 * 1024);
});
