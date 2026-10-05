import { expect, test } from '@playwright/test';

test.use({ serviceWorkers: 'allow' });
// Service worker control in Playwright is reliable in Chromium only.
test.skip(({ browserName }) => browserName !== 'chromium', 'Chromium only');

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
