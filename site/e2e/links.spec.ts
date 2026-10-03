import { expect, test, type Page } from '@playwright/test';

/** Waits until the map island has hydrated and shows the given part. */
async function expectMapAt(page: Page, title: string | null): Promise<void> {
  await expect(page).toHaveURL(/learn\/architecture\//);
  const map = page.locator('.arch-map');
  await map.scrollIntoViewIfNeeded();
  const heading = map.locator('.arch-panel h2');
  if (title) await expect(heading).toHaveText(title);
  else await expect(map.locator('.arch-panel')).toContainText('Tap a part');
}

test('a chapter step links to the map', async ({ page }) => {
  await page.goto('learn/introduction/');
  await page
    .locator('#step-inside-a-transformer-does-the-guessing')
    .getByRole('link', { name: 'See it on the map' })
    .click();
  await expectMapAt(page, null);
});

test('the pipeline figure opens the map at the stage it explains', async ({ page }) => {
  await page.goto('learn/introduction/');
  const fig = page.locator('#step-inside-a-transformer-does-the-guessing figure');
  await fig.scrollIntoViewIfNeeded();
  await expect(fig.locator('astro-island')).not.toHaveAttribute('ssr');
  await fig.getByRole('radio', { name: 'Output' }).click();
  await fig.getByRole('link', { name: 'See it on the map.' }).click();
  await expectMapAt(page, 'Output');
});

test('a glossary term opens the map at its part', async ({ page }) => {
  await page.goto('glossary/');
  await page.locator('#token').getByRole('link', { name: 'See it on the map' }).click();
  await expectMapAt(page, 'Inputs');
});

test('the learn page links to the map', async ({ page }) => {
  await page.goto('learn/');
  await page.getByRole('link', { name: /The transformer at a glance/ }).click();
  await expectMapAt(page, null);
});
