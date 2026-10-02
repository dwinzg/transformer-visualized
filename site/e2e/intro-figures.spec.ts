import { expect, test, type Locator } from '@playwright/test';
import { expectNoA11yViolations } from './a11y';

const CHAPTER = 'learn/introduction/';

/**
 * Scrolls a figure into view, which starts `client:visible` hydration, and waits until Astro
 * removes the island's `ssr` attribute. Before that, clicks reach server-rendered markup with no
 * handlers (see the same helper in interactions.spec.ts).
 */
async function hydrated(figure: Locator): Promise<void> {
  await figure.scrollIntoViewIfNeeded();
  await expect(figure.locator('astro-island')).not.toHaveAttribute('ssr');
}

test.beforeEach(async ({ page }) => {
  await page.goto('./');
  await page.evaluate(() => localStorage.clear());
});

test('next word: picking and resetting', async ({ page }) => {
  await page.goto(CHAPTER);
  const fig = page.locator('#step-it-guesses-the-next-piece-of-text figure');
  await hydrated(fig);
  const chips = fig.locator('.token-chip');
  const before = await chips.count();
  await fig.getByRole('option').nth(2).click();
  await expect(chips).toHaveCount(before + 1);
  await expect(fig.locator('[aria-live="polite"]')).toContainText('Added');
  await fig.getByRole('button', { name: 'Start over' }).click();
  await expect(chips).toHaveCount(before);
});

test('next word: exact percentages at the Numbers level', async ({ page }) => {
  await page.goto(CHAPTER);
  const fig = page.locator('#step-it-guesses-the-next-piece-of-text figure');
  await hydrated(fig);
  await page.getByRole('radio', { name: 'Numbers' }).check();
  const value = fig.locator('.prob-value').first();
  await expect(value).toHaveText(/^\d+\.\d%$/);
});

test('next word: rapid taps never skip or double up, and the end offers to start over', async ({
  page,
}) => {
  await page.goto(CHAPTER);
  const fig = page.locator('#step-it-guesses-the-next-piece-of-text figure');
  await hydrated(fig);
  const before = await fig.locator('.token-chip').count();
  for (let i = 0; i < 3; i++) await fig.getByRole('option').first().click();
  await expect(fig.locator('.token-chip')).toHaveCount(before + 3);
  await expect(fig.getByRole('button', { name: 'Start over' })).toBeVisible();
});

test('next word: keyboard focus stays in the figure after the last pick', async ({ page }) => {
  await page.goto(CHAPTER);
  const fig = page.locator('#step-it-guesses-the-next-piece-of-text figure');
  await hydrated(fig);
  await fig.getByRole('listbox').focus();
  for (let i = 0; i < 3; i++) await page.keyboard.press('Enter');
  await expect(fig.getByRole('button', { name: 'Start over' })).toBeFocused();
});

test('generation loop: add a guess and play', async ({ page }) => {
  await page.goto(CHAPTER);
  const fig = page.locator('#step-then-it-does-it-again figure');
  await hydrated(fig);
  await fig.getByRole('button', { name: 'Add a guess' }).click();
  await expect(fig.getByText('Guesses made: 1')).toBeVisible();
  await fig.getByRole('button', { name: 'Play' }).click();
  await expect(fig.getByText('Guesses made: 3')).toBeVisible({ timeout: 5000 });
  await fig.getByRole('button', { name: 'Start over' }).click();
  await expect(fig.getByText('Guesses made: 0')).toBeVisible();
});

test('generation loop works with reduced motion', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto(CHAPTER);
  const fig = page.locator('#step-then-it-does-it-again figure');
  await hydrated(fig);
  await fig.getByRole('button', { name: 'Play' }).click();
  await expect(fig.getByText('Guesses made: 3')).toBeVisible({ timeout: 5000 });
});

test('pipeline: arrow keys pick a stage and show what it does', async ({ page }) => {
  await page.goto(CHAPTER);
  const fig = page.locator('#step-inside-a-transformer-does-the-guessing figure');
  await hydrated(fig);
  await fig.getByRole('radio', { name: 'Tokens' }).click();
  await expect(fig).toContainText('The text is split into tokens');
  await page.keyboard.press('ArrowDown');
  await expect(fig.getByRole('radio', { name: 'Embeddings and position' })).toBeChecked();
  await expect(fig).toContainText('a list of 128 numbers');
});

test('the figures are accessible after interaction in both themes', async ({ page }) => {
  await page.goto(CHAPTER);
  const nextWord = page.locator('#step-it-guesses-the-next-piece-of-text figure');
  await hydrated(nextWord);
  await nextWord.getByRole('option').first().click();
  const pipeline = page.locator('#step-inside-a-transformer-does-the-guessing figure');
  await hydrated(pipeline);
  await pipeline.getByRole('radio', { name: 'Output' }).click();
  // Axe flags a target that the sticky header half covers, and where the clicks above leave the
  // page decides whether a step's tabs sit there. Scan from the top, so the result is stable.
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});
