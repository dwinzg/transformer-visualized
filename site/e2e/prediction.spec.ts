import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const CHAPTER = 'learn/prediction/';

async function hydrated(page: Page, step: string): Promise<Locator> {
  const figure = page.locator(`#${step} figure`);
  await figure.scrollIntoViewIfNeeded();
  await expect(figure.locator('astro-island')).not.toHaveAttribute('ssr');
  return figure;
}

test('the tied scores table shows the five best tokens and their scores', async ({ page }) => {
  await page.goto(CHAPTER);
  const table = page.locator('#step-one-score-for-every-token table');
  await expect(table.locator('tbody tr')).toHaveCount(5);
  await expect(table.locator('tbody tr').first()).toContainText('ball');
  await expect(table.locator('tbody tr').first()).toContainText('2.60');
});

test('temperature reshapes the chances, and Start over goes back', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-scores-become-chances');
  const summary = figure.locator('.sampling-summary');
  const slider = figure.getByRole('slider', { name: 'Temperature' });
  await expect(summary).toContainText('has a 11.5% chance');
  await expect(summary).toHaveAttribute('aria-live', 'polite');
  await slider.fill('0.1');
  await expect(figure.locator('output')).toHaveText('0.1');
  await expect(summary).toContainText('has a 72.4% chance');
  await slider.focus();
  await page.keyboard.press('ArrowRight');
  await expect(figure.locator('output')).toHaveText('0.2');
  await figure.getByRole('button', { name: 'Start over' }).click();
  await expect(figure.locator('output')).toHaveText('1.0');
  await expect(summary).toContainText('has a 11.5% chance');
});

test('the keep rules take tokens out of the running', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-scores-become-chances');
  const bars = figure.locator('.prob-bar');
  await figure.getByRole('radio', { name: 'Top-k, k = 5' }).check();
  await expect(bars.nth(4)).not.toHaveClass(/is-out/);
  await expect(bars.nth(5)).toHaveClass(/is-out/);
  await expect(bars.nth(5)).toContainText('out');
  await expect(bars.last()).toContainText('out');
  await figure.getByRole('radio', { name: 'Top-p, p = 0.9' }).check();
  await expect(bars.last()).not.toHaveClass(/is-out/);
  await figure.getByRole('radio', { name: 'Greedy' }).check();
  await expect(bars.first()).toContainText('100.0%');
});

test('greedy always samples the top token', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-greedy-top-k-and-top-p');
  await expect(figure.getByRole('radio', { name: 'Greedy' })).toBeChecked();
  const sample = figure.getByRole('button', { name: 'Sample' });
  const picks = figure.locator('.sampling-picks');
  await expect(picks).toContainText('Press Sample');
  await sample.focus();
  await page.keyboard.press('Enter');
  await page.keyboard.press('Enter');
  await expect(picks).toContainText('Picked');
  await expect(picks).toContainText('space ball');
  await expect(picks).toContainText('Before that');
  await expect(picks).not.toContainText('rarer');
});

test('the exercise checks a chance at temperature 0.5', async ({ page }) => {
  await page.goto(CHAPTER);
  const step = page.locator('#step-turning-the-temperature-up-or-down');
  await step.scrollIntoViewIfNeeded();
  const island = step.locator('astro-island', { has: page.getByRole('textbox') });
  await island.scrollIntoViewIfNeeded();
  await expect(island).not.toHaveAttribute('ssr');
  await step.getByRole('textbox').fill('0.98');
  await step.getByRole('button', { name: /Check/ }).click();
  await expect(step).toContainText('Dividing by 0.5 gives 4 and 0');
});

test('the chapter never downloads the model or the tokenizer', async ({ page }) => {
  const fetched: string[] = [];
  page.on('request', (request) => {
    if (/\.safetensors|tokenizer\.json/.test(request.url())) fetched.push(request.url());
  });
  await page.goto(CHAPTER);
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForLoadState('networkidle');
  expect(fetched).toHaveLength(0);
});

test('the chapter is accessible after interaction, in both themes', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-scores-become-chances');
  await figure.getByRole('radio', { name: 'Top-k, k = 5' }).check();
  await figure.getByRole('button', { name: 'Sample' }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test('the map opens this chapter from the Linear block', async ({ page }) => {
  await page.goto('learn/architecture/?part=linear&view=gpt2');
  const map = page.locator('.arch-map');
  await map.scrollIntoViewIfNeeded();
  await map.getByRole('link', { name: 'Read the chapter' }).click();
  await expect(page).toHaveURL(/learn\/prediction\/#step-one-score-for-every-token$/);
});

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the page never scrolls sideways and stays accessible', async ({ page }) => {
    await page.goto(CHAPTER);
    await hydrated(page, 'step-scores-become-chances');
    await expectNoHorizontalScroll(page);
    await page.evaluate(() => window.scrollTo(0, 0));
    await expectNoA11yViolations(page);
  });
});
