import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

test('the Learn page puts the new chapters under Using it today', async ({ page }) => {
  await page.goto('learn/');
  const today = page.getByRole('region', { name: 'Using it today' });
  await expect(today.getByRole('link')).toHaveText([
    /Context and prompts/,
    /Retrieval and tools/,
    /Thinking step by step/,
  ]);
  const core = page.getByRole('region', { name: 'Inside the model' });
  await expect(core.getByRole('link')).toHaveCount(5);
});

test('the note falls out of the window as sentences are added', async ({ page }) => {
  await page.goto('learn/context/');
  const figure = page.locator('.window-figure');
  await figure.scrollIntoViewIfNeeded();
  await expect(page.locator('astro-island:has(.window-figure)')).not.toHaveAttribute('ssr');
  await expect(figure.locator('.prob-bar').first()).toContainText('Fluffy');
  await expect(figure).toContainText('the note is inside its window');

  const slider = figure.getByRole('slider', { name: /Sentences in between/ });
  await slider.fill('6');
  await expect(figure).toContainText('140 tokens in all');
  await expect(figure).toContainText('the note is outside its window');
  await expect(figure.locator('.prob-bar').first()).toContainText('her');
  await expect(figure.locator('.prob-bar', { hasText: 'Fluffy' })).toHaveCount(0);
});

test('a prompt comparison switches between its texts', async ({ page }) => {
  await page.goto('learn/tools/');
  const island = page.locator('astro-island:has(.compare-figure)').nth(1);
  const figure = island.locator('.compare-figure');
  await figure.scrollIntoViewIfNeeded();
  await expect(island).not.toHaveAttribute('ssr');
  await expect(figure.locator('.prob-bar').first()).toContainText('rug');
  await figure.getByRole('radio', { name: 'Without it' }).check();
  await expect(figure.locator('.prob-bar').first()).toContainText('tree');
});

for (const path of ['learn/context/', 'learn/tools/', 'learn/reasoning/']) {
  test(`/${path} is accessible and fits a phone`, async ({ page }) => {
    await page.goto(path);
    for (const island of await page.locator('astro-island').all()) {
      await island.scrollIntoViewIfNeeded();
    }
    await expect(page.locator('astro-island[ssr]')).toHaveCount(0);
    await expectNoA11yViolations(page);
    await page.emulateMedia({ colorScheme: 'dark' });
    await expectNoA11yViolations(page);
    await page.setViewportSize({ width: 320, height: 640 });
    await expectNoHorizontalScroll(page);
  });
}
