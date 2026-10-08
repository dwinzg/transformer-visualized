import { expect, test, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

async function playground(page: Page, path = 'playground/') {
  await page.goto(path);
  await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });
}

const topGuess = (page: Page) => page.locator('.prob-bar').first();

test('the map shows the Llama view, with its own parts and code', async ({ page }) => {
  await page.goto('learn/architecture/?part=ffn&view=llama');
  const map = page.locator('.arch-map');
  await expect(map.getByRole('radio', { name: 'Llama style' })).toBeChecked();
  await expect(map.locator('.arch-panel .arch-title')).toHaveText('Feed forward');
  await expect(map.getByRole('radio', { name: 'Grouped-query attention, RoPE' })).toBeVisible();
  await expect(map.getByRole('radio', { name: /Position embedding/ })).toHaveCount(0);
  const article = page.locator('#part-article');
  await expect(article).toContainText('w_gate');
  await expect(article).toContainText('tensor([ 0.2635, -0.2112, -0.2429, -0.2152])');
});

test('the playground switches to the Llama model, and remembers it', async ({ page }) => {
  const fetched: string[] = [];
  page.on('request', (r) => {
    if (r.url().includes('.safetensors')) fetched.push(r.url());
  });
  await playground(page);
  await expect(topGuess(page)).toContainText('ball');
  expect(fetched).toHaveLength(1);

  await page.getByRole('radio', { name: 'Llama style' }).check();
  await expect(topGuess(page)).toContainText('toys', { timeout: 20_000 });
  expect(fetched).toHaveLength(2);
  await expect(page).toHaveURL(/model=llama/);

  // The embeddings have no position numbers, and the final norm is an RMSNorm.
  await page.getByRole('radio', { name: 'Embeddings' }).check();
  await expect(page.getByRole('radio', { name: 'Position' })).toHaveCount(0);
  await page.getByRole('radio', { name: 'Residual stream' }).check();
  await page.getByText('Then the final norm rescales').click();
  await expect(page.locator('.norm-details')).toContainText('root mean square');

  // A new visit without the link keeps the choice.
  await playground(page, 'playground/');
  await expect(page.getByRole('radio', { name: 'Llama style' })).toBeChecked();
  await expect(topGuess(page)).toContainText('toys', { timeout: 20_000 });
});

test('the playground with the Llama model is accessible and fits a phone', async ({ page }) => {
  await playground(page, 'playground/?model=llama');
  await expect(topGuess(page)).toContainText('toys', { timeout: 20_000 });
  await expectNoA11yViolations(page);
  await page.setViewportSize({ width: 320, height: 640 });
  await expectNoHorizontalScroll(page);
});
