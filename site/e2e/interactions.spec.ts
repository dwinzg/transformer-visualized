import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectNoA11yViolations } from './a11y';

const CHAPTER = 'learn/introduction/';

/**
 * Scrolls an island's content into view, which starts `client:visible` hydration, and waits until
 * Astro removes the island's `ssr` attribute. Before that, clicks reach server-rendered markup
 * with no handlers, and the exercise form would submit natively.
 */
async function hydrated(page: Page, content: Locator): Promise<void> {
  await content.scrollIntoViewIfNeeded();
  await expect(page.locator('astro-island', { has: content })).not.toHaveAttribute('ssr');
}

test('a glossary term opens its definition', async ({ page }) => {
  await page.goto(CHAPTER);
  await page.getByRole('button', { name: 'language model' }).first().click();
  const card = page.locator('.term-card:popover-open');
  await expect(card).toContainText('gives every possible next piece of text a probability');
  await expect(card.getByRole('link', { name: 'Open the glossary' })).toHaveAttribute(
    'href',
    '/transformer-visualized/glossary/#language-model',
  );
  await page.keyboard.press('Escape');
  await expect(page.locator('.term-card:popover-open')).toHaveCount(0);
});

test('citations link to the references page', async ({ page }) => {
  await page.goto(CHAPTER);
  await expect(page.locator('a.ref', { hasText: 'Shannon, 1951' })).toHaveAttribute(
    'href',
    '/transformer-visualized/references/#shannon1951',
  );
});

test('predict then reveal', async ({ page }) => {
  await page.goto(CHAPTER);
  const block = page.locator('.predict');
  await hydrated(page, block);
  await expect(block).not.toContainText('Most people pick');
  await block.getByRole('button', { name: 'moon' }).click();
  await expect(block).toContainText('The likeliest answer is "mat".');
  await expect(block).toContainText('Most people pick');
  await expect(block.getByRole('button', { name: 'mat' })).toBeDisabled();
});

test('predict then reveal keeps focus after answering by keyboard', async ({ page }) => {
  await page.goto(CHAPTER);
  const block = page.locator('.predict');
  await hydrated(page, block);
  await block.getByRole('button', { name: 'moon' }).focus();
  await page.keyboard.press('Enter');
  await expect(block).toContainText('The likeliest answer is "mat".');
  const activeElementIsBody = await page.evaluate(() => document.activeElement === document.body);
  expect(activeElementIsBody).toBe(false);
  await expect(block.getByRole('button', { name: 'moon' })).toBeFocused();
});

test('exercise checks the answer', async ({ page }) => {
  await page.goto(CHAPTER);
  const form = page.locator('.exercise');
  await hydrated(page, form);
  const input = form.getByRole('textbox');
  await input.fill('0.1');
  await form.getByRole('button', { name: 'Check' }).click();
  await expect(form).toContainText('Not quite');
  await input.fill('5%');
  await form.getByRole('button', { name: 'Check' }).click();
  await expect(form).toContainText('Correct.');
  await expect(form).toContainText('1 minus 0.9 minus 0.05');
});

test('exercise announces a repeated wrong answer again', async ({ page }) => {
  await page.goto(CHAPTER);
  const form = page.locator('.exercise');
  await hydrated(page, form);
  const input = form.getByRole('textbox');
  const feedback = form.locator('.exercise-feedback');
  await input.fill('0.1');
  await form.getByRole('button', { name: 'Check' }).click();
  const firstMessage = await feedback.textContent();
  await input.fill('0.2');
  await form.getByRole('button', { name: 'Check' }).click();
  const secondMessage = await feedback.textContent();
  expect(secondMessage).not.toBe(firstMessage);
  expect(secondMessage).toMatch(/2nd try/);
});

test('the chapter stays accessible with the new components', async ({ page }) => {
  await page.goto(CHAPTER);
  await expectNoA11yViolations(page);
  await page.emulateMedia({ colorScheme: 'dark' });
  await expectNoA11yViolations(page);
});

test.describe('accessibility with an interactive state open', () => {
  for (const colorScheme of ['light', 'dark'] as const) {
    test(`a Term popover is accessible in the ${colorScheme} theme`, async ({ page }) => {
      await page.emulateMedia({ colorScheme });
      await page.goto(CHAPTER);
      await page.getByRole('button', { name: 'language model' }).first().click();
      await expect(page.locator('.term-card:popover-open')).toBeVisible();
      await expectNoA11yViolations(page);
    });

    test(`a revealed PredictReveal is accessible in the ${colorScheme} theme`, async ({ page }) => {
      await page.emulateMedia({ colorScheme });
      await page.goto(CHAPTER);
      const block = page.locator('.predict');
      await hydrated(page, block);
      await block.getByRole('button', { name: 'moon' }).click();
      await expect(block).toContainText('The likeliest answer is "mat".');
      await expectNoA11yViolations(page);
    });

    test(`Exercise feedback is accessible in the ${colorScheme} theme`, async ({ page }) => {
      await page.emulateMedia({ colorScheme });
      await page.goto(CHAPTER);
      const form = page.locator('.exercise');
      await hydrated(page, form);
      await form.getByRole('textbox').fill('0.1');
      await form.getByRole('button', { name: 'Check' }).click();
      await expect(form).toContainText('Not quite');
      await expectNoA11yViolations(page);
    });
  }
});
