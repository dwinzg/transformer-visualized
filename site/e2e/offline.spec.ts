import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, test } from '@playwright/test';

test.use({ serviceWorkers: 'allow' });
// Service worker control in Playwright is reliable in Chromium only.
test.skip(({ browserName }) => browserName !== 'chromium', 'Chromium only');
// One test changes the built sw.js for a moment, so these run one at a time.
test.describe.configure({ mode: 'serial' });

test('the site has a manifest with installable icons', async ({ page }) => {
  await page.goto('');
  const href = await page.locator('link[rel="manifest"]').getAttribute('href');
  const manifest = await (await page.request.get(href!)).json();
  expect(manifest.start_url).toBe('/transformer-visualized/');
  expect(manifest.icons.map((i: { sizes: string }) => i.sizes)).toContain('512x512');
});

test('pages and the playground work offline after one visit', async ({ page, context }) => {
  await page.goto('');
  await page.evaluate(() => navigator.serviceWorker.ready);
  // The playground keeps the model in the cache the first time it loads it.
  await page.goto('playground/');
  await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });

  await context.setOffline(true);
  await page.goto('learn/tokens/');
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Tokens');
  await page.goto('learn/?depth=formula');
  await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
  await page.goto('playground/');
  await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });
  await expect(page.locator('.sampling-summary')).toContainText('gets 11.5%');
  // A shared link opens offline too.
  await page.goto('playground/?text=The%20cat%20sat');
  await expect(page.getByRole('list', { name: 'Tokens' }).getByRole('listitem')).toHaveCount(3, {
    timeout: 20_000,
  });
});

test('a new version offers a reload, and Reload switches to it', async ({ page }) => {
  await page.goto('');
  await page.evaluate(() => navigator.serviceWorker.ready);
  await page.reload();
  await expect.poll(() => page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);
  const note = page.locator('[data-update-note]');
  await expect(note).toBeHidden();

  // Change the built sw.js, the way a deploy would, and ask the browser to check for it.
  const file = resolve(process.cwd(), 'dist/sw.js');
  const original = readFileSync(file, 'utf8');
  writeFileSync(file, `${original}\n// a new version`);
  try {
    await page.evaluate(async () => (await navigator.serviceWorker.getRegistration())?.update());
    await expect
      .poll(
        () =>
          page.evaluate(async () => !!(await navigator.serviceWorker.getRegistration())?.waiting),
        {
          timeout: 15_000,
        },
      )
      .toBe(true);
    await expect(note).toBeVisible();
    await expect(note).toHaveText(/A new version of this site is ready/);

    const reloaded = page.waitForEvent('load');
    await note.getByRole('button', { name: 'Reload' }).click();
    await reloaded;
    await expect(note).toBeHidden();
  } finally {
    writeFileSync(file, original);
  }
});
