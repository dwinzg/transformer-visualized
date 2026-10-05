import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const PAGE = 'review/';

test('the home page opens the quick review', async ({ page }) => {
  await page.goto('');
  await page.getByRole('link', { name: /Quick review/ }).click();
  await expect(page).toHaveURL(/review\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Quick review');
});

test('one card per part, in the order a sentence passes through', async ({ page }) => {
  await page.goto(PAGE);
  const titles = page.locator('.card h2');
  await expect(titles.first()).toHaveText('Inputs');
  await expect(titles.last()).toHaveText('Output');
  await expect(titles).toHaveCount(11);
  await expect(page.locator('#masked-attn .katex-display')).toBeVisible();
  await expect(page.locator('#masked-attn code')).toContainText('weights');
});

test('a card opens its chapter step and its place on the map', async ({ page }) => {
  await page.goto(PAGE);
  await page.locator('#embedding').getByRole('link', { name: 'Read the chapter' }).click();
  await expect(page).toHaveURL(/learn\/embeddings\/#step-each-token-becomes-a-list-of-numbers$/);
  await page.goto(PAGE);
  await page.locator('#ffn').getByRole('link', { name: 'See it on the map' }).click();
  await expect(page).toHaveURL(/learn\/architecture\/\?part=ffn&view=gpt2$/);
});

test('every chapter link lands on a step that exists', async ({ page }) => {
  await page.goto(PAGE);
  const hrefs = await page
    .locator('.card a[href*="#"]')
    .evaluateAll((links) => links.map((a) => (a as HTMLAnchorElement).href));
  expect(hrefs.length).toBeGreaterThan(5);
  for (const href of hrefs) {
    await page.goto(href);
    await expect(page.locator(`#${new URL(href).hash.slice(1)}`)).toHaveCount(1);
  }
});

test('the page is accessible in both themes', async ({ page }) => {
  await page.goto(PAGE);
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the page never scrolls sideways', async ({ page }) => {
    await page.goto(PAGE);
    await expectNoHorizontalScroll(page);
    await expectNoA11yViolations(page);
  });
});
