import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

test('sliding back to the start shows an untrained model', async ({ page }) => {
  await page.goto('training/');
  const figure = page.locator('.train-figure');
  await figure.scrollIntoViewIfNeeded();
  await expect(page.locator('astro-island:has(.train-figure)')).not.toHaveAttribute('ssr');
  await expect(figure).toContainText('After 25,000 steps');
  await expect(figure.locator('.prob-bar').first()).toContainText('ball');

  await figure.getByRole('slider').fill('0');
  await expect(figure).toContainText('Before any training');
  await expect(figure).toContainText('Loss 8.33');
  await expect(figure).toContainText('Remembering Fluffy from an earlier sentence, 0%');
});

test('the Learn menu links to the training page', async ({ page }) => {
  await page.goto('');
  const nav = page.getByRole('navigation', { name: 'Main' });
  await expect(
    nav.getByRole('link', { name: 'Watch it learn', includeHidden: true }),
  ).toHaveAttribute('href', /training\/$/);
});

test('the training page is accessible and fits a phone', async ({ page }) => {
  await page.goto('training/');
  await page.locator('.train-figure').scrollIntoViewIfNeeded();
  await expect(page.locator('astro-island[ssr]')).toHaveCount(0);
  await expectNoA11yViolations(page);
  await page.emulateMedia({ colorScheme: 'dark' });
  await expectNoA11yViolations(page);
  await page.setViewportSize({ width: 320, height: 640 });
  await expectNoHorizontalScroll(page);
});
