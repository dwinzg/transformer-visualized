import { expect, test } from '@playwright/test';
import { expectNoA11yViolations } from './a11y';

test('picking a bar adds its word to the sentence', async ({ page }) => {
  await page.goto('dev/figures/');
  const chips = page.locator('.token-chip');
  const before = await chips.count();
  const first = page.getByRole('option').first();
  const word = (await first.locator('.prob-word').textContent()) ?? '';
  await first.click();
  await expect(chips).toHaveCount(before + 1);
  await expect(chips.last().locator('.token-text')).toHaveText(word);
});

test('the bars work with the keyboard', async ({ page }) => {
  await page.goto('dev/figures/');
  const list = page.getByRole('listbox', { name: 'Next word guesses' });
  await list.focus();
  await page.keyboard.press('ArrowDown');
  await expect(page.getByRole('option').nth(1)).toHaveAttribute('aria-selected', 'true');
  const word = (await page.getByRole('option').nth(1).locator('.prob-word').textContent()) ?? '';
  await page.keyboard.press('Enter');
  await expect(page.locator('.token-chip').last().locator('.token-text')).toHaveText(word);
});

test('rapid taps each add exactly one word', async ({ page }) => {
  await page.goto('dev/figures/');
  const chips = page.locator('.token-chip');
  const before = await chips.count();
  for (let i = 0; i < 3; i++) await page.getByRole('option').first().click();
  await expect(chips).toHaveCount(before + 3);
  await expect(page.getByRole('option')).toHaveCount(0);
});

test('the kit is accessible in both themes', async ({ page }) => {
  await page.goto('dev/figures/');
  // The figure fades in once it enters the viewport (see Figure.astro's data-enter). Wait for
  // that to finish so axe does not score the transition's intermediate, low-contrast frames.
  await expect(page.locator('.figure')).toHaveCSS('opacity', '1');
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test('the figure becomes visible once scrolled into view', async ({ page }) => {
  await page.goto('dev/figures/');
  const figure = page.locator('.figure');
  await figure.scrollIntoViewIfNeeded();
  await expect(figure).toBeVisible();
  await expect(figure).toHaveCSS('opacity', '1');
});
