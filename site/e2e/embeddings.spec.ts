import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const CHAPTER = 'learn/embeddings/';

async function hydrated(page: Page, step: string): Promise<Locator> {
  const figure = page.locator(`#${step} figure`);
  await figure.scrollIntoViewIfNeeded();
  await expect(figure.locator('astro-island')).not.toHaveAttribute('ssr');
  return figure;
}

test('the number strip shows one number at a time, with token, position and sum', async ({
  page,
}) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-each-token-becomes-a-list-of-numbers');
  const readout = figure.locator('.strip-readout');
  await expect(readout).toContainText('Number 1 of 128. Token 0.008');
  const grid = figure.getByRole('listbox');
  await grid.focus();
  await page.keyboard.press('ArrowUp');
  await expect(readout).toContainText('Number 1 of 128.');
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowDown');
  await expect(readout).toContainText('Number 18 of 128.');
  await page.keyboard.press('End');
  await expect(readout).toContainText('Number 128 of 128.');
  await page.keyboard.press('ArrowDown');
  await expect(readout).toContainText('Number 128 of 128.');
  await figure.getByRole('radio', { name: 'space play', exact: true }).click();
  await expect(grid).toHaveAccessibleName(/token numbers for space play/);
  await figure.getByRole('radio', { name: 'Added together', exact: true }).check();
  await expect(grid).toHaveAccessibleName(/token plus position numbers for space play/);
  await figure.getByRole('button', { name: 'Start over' }).click();
  await expect(figure.getByRole('radio', { name: 'Lily', exact: true })).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await expect(readout).toContainText('Number 1 of 128.');
});

test('token chips move with arrow keys', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-each-token-becomes-a-list-of-numbers');
  await figure.getByRole('radio', { name: 'Lily', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(figure.getByRole('radio', { name: 'space wanted', exact: true })).toBeFocused();
  await expect(figure.getByRole('radio', { name: 'space wanted', exact: true })).toHaveAttribute(
    'aria-checked',
    'true',
  );
});

test('the position step starts on play, in the position view', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-each-place-gets-numbers-too');
  await expect(figure.getByRole('radio', { name: 'space play', exact: true })).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await expect(figure.getByRole('radio', { name: 'Position', exact: true })).toBeChecked();
});

test('picking a word lists its nearest tokens', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-similar-words-get-similar-numbers');
  const rows = figure.locator('.nearest-row');
  await expect(rows).toHaveCount(5);
  await expect(rows.first()).toContainText('boy');
  await expect(rows.first()).toContainText('0.74');
  await figure.getByRole('radio', { name: 'space happy', exact: true }).click();
  await expect(rows.first()).toContainText('glad');
  await figure.getByRole('button', { name: 'Start over' }).click();
  await expect(rows.first()).toContainText('boy');
});

test('word chips move with arrow keys and say the leading space', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-similar-words-get-similar-numbers');
  await figure.getByRole('radio', { name: 'space girl', exact: true }).focus();
  await page.keyboard.press('ArrowRight');
  await expect(figure.getByRole('radio', { name: 'space ran', exact: true })).toBeFocused();
  await expect(figure.locator('.nearest-title')).toContainText('space ran.');
  await page.keyboard.press('End');
  await expect(figure.locator('.nearest-title')).toContainText('space 3 0.88');
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

test('the exercise checks a dot product', async ({ page }) => {
  await page.goto(CHAPTER);
  const step = page.locator('#step-comparing-two-lists');
  await step.scrollIntoViewIfNeeded();
  const island = step.locator('astro-island').last();
  await expect(island).not.toHaveAttribute('ssr');
  await step.getByRole('textbox').fill('4');
  await step.getByRole('button', { name: /Check/ }).click();
  await expect(step).toContainText('Multiply place by place to get 2, -4 and 6');
});

test('the chapter is accessible after interaction, in both themes', async ({ page }) => {
  await page.goto(CHAPTER);
  const strip = await hydrated(page, 'step-each-token-becomes-a-list-of-numbers');
  await strip.getByRole('radio', { name: 'space her', exact: true }).click();
  const near = await hydrated(page, 'step-similar-words-get-similar-numbers');
  await near.getByRole('radio', { name: 'space red', exact: true }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test('the map opens this chapter from the Position block', async ({ page }) => {
  await page.goto('learn/architecture/?part=position&view=gpt2');
  const map = page.locator('.arch-map');
  await map.scrollIntoViewIfNeeded();
  await expect(page.locator('astro-island', { has: map })).not.toHaveAttribute('ssr');
  await page.locator('#part-article').getByRole('link', { name: 'Read the chapter' }).click();
  await expect(page).toHaveURL(/learn\/embeddings\/#step-each-place-gets-numbers-too$/);
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the page never scrolls sideways', async ({ page }) => {
    await page.goto(CHAPTER);
    await hydrated(page, 'step-each-token-becomes-a-list-of-numbers');
    await expectNoHorizontalScroll(page);
  });

  test('a tapped cell shows its number, and the page stays accessible', async ({ page }) => {
    await page.goto(CHAPTER);
    const figure = await hydrated(page, 'step-each-token-becomes-a-list-of-numbers');
    await figure.getByRole('option', { name: /^Number 20,/ }).click();
    await expect(figure.locator('.strip-readout')).toContainText('Number 20 of 128.');
    await page.evaluate(() => window.scrollTo(0, 0));
    await expectNoA11yViolations(page);
  });
});
