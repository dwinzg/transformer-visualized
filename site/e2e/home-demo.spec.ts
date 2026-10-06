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

test('Start over resets the sentence at any point', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.getByRole('button', { name: 'Pause demo' }).click();
  const before = await demo.locator('.token-chip').count();
  await demo.getByRole('option').first().click();
  await expect(demo.locator('.token-chip')).toHaveCount(before + 1);
  await demo.getByRole('button', { name: 'Start over' }).click();
  await expect(demo.locator('.token-chip')).toHaveCount(before);
});

test('Play resumes right after the reader picks a guess', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.getByRole('option').first().click();
  const count = await demo.locator('.token-chip').count();
  await demo.getByRole('button', { name: 'Play demo' }).click();
  await expect(demo.locator('.token-chip')).toHaveCount(count + 1, { timeout: 5000 });
});

test('the end of the guesses offers to start over', async ({ page }) => {
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.getByRole('button', { name: 'Pause demo' }).click();
  const before = await demo.locator('.token-chip').count();
  // The first guesses lead to the end of a sentence without loading the model.
  let picks = 0;
  while ((await demo.getByRole('option').count()) > 0 && picks < 30) {
    await demo.getByRole('option').first().click();
    picks++;
  }
  expect(picks).toBeGreaterThan(3);
  await expect(demo.locator('.token-chip').last()).toContainText('.');
  await expect(demo.getByRole('button', { name: 'Next sentence' })).toBeVisible();
  const start = demo.getByRole('button', { name: 'Start over' });
  await expect(start).toBeFocused();
  await start.click();
  await expect(demo.locator('.token-chip')).toHaveCount(before);
});

test('picks past the built guesses load the model once and finish the sentence', async ({
  page,
}) => {
  const fetched: string[] = [];
  page.on('request', (r) => r.url().includes('.safetensors') && fetched.push(r.url()));
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.getByRole('button', { name: 'Pause demo' }).click();
  // The second guess each time leaves the path that was worked out ahead.
  let picks = 0;
  while (picks < 30) {
    const options = demo.getByRole('option');
    await expect(
      options.first().or(demo.getByRole('button', { name: 'Next sentence' })),
    ).toBeVisible({
      timeout: 20_000,
    });
    if ((await options.count()) === 0) break;
    await options.nth((await options.count()) > 1 ? 1 : 0).click();
    picks++;
  }
  await expect(demo.getByRole('button', { name: 'Next sentence' })).toBeVisible();
  expect(picks).toBeGreaterThan(3);
  expect(fetched).toHaveLength(1);
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

test('the home page ships at most 80 KB of gzipped JavaScript', () => {
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
  // About 66 KB of this is the React runtime that every island page loads.
  expect(bytes).toBeLessThanOrEqual(80 * 1024);
});

test('autoplay holds while the demo is scrolled out of view', async ({ page }) => {
  await page.setViewportSize({ width: 1000, height: 500 });
  await page.goto('./');
  const demo = page.locator('[data-home-demo]');
  await demo.locator('.token-chip').first().waitFor();
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  const count = await demo.locator('.token-chip').count();
  await page.waitForTimeout(4000);
  await expect(demo.locator('.token-chip')).toHaveCount(count);
});

test('with reduced motion, every figure shows at once', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('learn/introduction/');
  const figures = page.locator('figure[data-enter]');
  const total = await figures.count();
  expect(total).toBeGreaterThan(1);
  await expect(page.locator('figure[data-enter].is-visible')).toHaveCount(total);
});
