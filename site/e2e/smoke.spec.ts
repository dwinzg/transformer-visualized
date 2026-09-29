import { expect, test } from '@playwright/test';

test('the home page is served under the base path', async ({ page }) => {
  const response = await page.goto('./');
  expect(response?.status()).toBe(200);
  await expect(page).toHaveTitle(/Transformer Visualized/);
});
