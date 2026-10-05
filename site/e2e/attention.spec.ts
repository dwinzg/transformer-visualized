import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const CHAPTER = 'learn/attention/';

async function hydrated(page: Page, step: string): Promise<Locator> {
  const figure = page.locator(`#${step} figure`);
  await figure.scrollIntoViewIfNeeded();
  await expect(figure.locator('astro-island')).not.toHaveAttribute('ssr');
  return figure;
}

test('the toy example steps through attention and back', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-questions-name-tags-and-backpacks');
  const title = figure.locator('.toy-title');
  const next = figure.getByRole('button', { name: 'Next' });
  await expect(title).toHaveText('Step 1 of 5. Match questions with name tags');
  await expect(title.locator('..')).toHaveAttribute('aria-live', 'polite');
  await expect(figure.getByRole('button', { name: 'Previous' })).toHaveAttribute(
    'aria-disabled',
    'true',
  );
  await next.click();
  await next.click();
  await expect(title).toContainText('Step 3 of 5');
  await expect(figure.locator('td.is-hidden')).toHaveCount(3);
  await next.click();
  await expect(figure.locator('tbody tr').last()).toContainText('0.77');
  await next.click();
  await expect(title).toContainText('Step 5 of 5');
  await expect(next).toHaveAttribute('aria-disabled', 'true');
  await expect(figure.getByRole('list', { name: 'New backpacks' })).toBeVisible();
  await figure.getByRole('button', { name: 'Previous' }).click();
  await expect(title).toContainText('Step 4 of 5');
  await figure.getByRole('button', { name: 'Start over' }).click();
  await expect(title).toContainText('Step 1 of 5');
});

test('the toy example works from the keyboard', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-questions-name-tags-and-backpacks');
  await figure.getByRole('button', { name: 'Next' }).focus();
  await page.keyboard.press('Enter');
  await page.keyboard.press('Space');
  await expect(figure.locator('.toy-title')).toContainText('Step 3 of 5');
});

test('the grid starts on her, in the head that looks back at Lily', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-words-need-context');
  const summary = figure.locator('.attn-summary');
  // The summary holds each token twice, drawn with a dot and spoken with "space".
  await expect(summary).toContainText('In layer 3, head 3,');
  await expect(summary).toContainText('space her looks most at');
  await expect(summary).toContainText('Lily 0.66');
  await expect(figure.locator('tr.is-picked')).toContainText('0.66');
  await expect(summary).toHaveAttribute('aria-live', 'polite');
  await expect(figure.getByRole('button', { name: 'Start over' })).toHaveAttribute(
    'aria-disabled',
    'true',
  );
});

test('picking a layer, head or token updates the grid, and Start over goes back', async ({
  page,
}) => {
  await page.goto(CHAPTER);
  const figure = await hydrated(page, 'step-words-need-context');
  const summary = figure.locator('.attn-summary');
  await figure.getByRole('group', { name: 'Layer' }).getByRole('radio', { name: '2' }).check();
  await figure.getByRole('group', { name: 'Head' }).getByRole('radio', { name: '1' }).check();
  await expect(summary).toContainText('In layer 2, head 1');
  await figure.getByRole('radio', { name: 'space with', exact: true }).click();
  await figure.getByRole('radio', { name: 'space with', exact: true }).focus();
  await page.keyboard.press('ArrowLeft');
  await expect(figure.getByRole('radio', { name: 'space play', exact: true })).toBeFocused();
  await expect(summary).toContainText('space play looks most at');
  await figure.getByRole('button', { name: 'Start over' }).click();
  await expect(summary).toContainText('In layer 3, head 3,');
});

test('the softmax exercise checks the first weight', async ({ page }) => {
  await page.goto(CHAPTER);
  const step = page.locator('#step-softmax-turns-scores-into-weights');
  await step.scrollIntoViewIfNeeded();
  const island = step.locator('astro-island').last();
  await expect(island).not.toHaveAttribute('ssr');
  await step.getByRole('textbox').fill('0.79');
  await step.getByRole('button', { name: /Check/ }).click();
  await expect(step).toContainText('The first weight is 7.389 divided by 9.389');
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
  const toy = await hydrated(page, 'step-questions-name-tags-and-backpacks');
  for (let i = 0; i < 4; i++) await toy.getByRole('button', { name: 'Next' }).click();
  const grid = await hydrated(page, 'step-many-heads-many-layers');
  await grid.getByRole('group', { name: 'Head' }).getByRole('radio', { name: '4' }).check();
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test('the map opens this chapter from the Masked attention block', async ({ page }) => {
  await page.goto('learn/architecture/?part=masked-attn&view=gpt2');
  const map = page.locator('.arch-map');
  await map.scrollIntoViewIfNeeded();
  await map.getByRole('link', { name: 'Read the chapter' }).click();
  await expect(page).toHaveURL(/learn\/attention\/#step-words-need-context$/);
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the page never scrolls sideways, even at the last toy step', async ({ page }) => {
    await page.goto(CHAPTER);
    await hydrated(page, 'step-words-need-context');
    const toy = await hydrated(page, 'step-questions-name-tags-and-backpacks');
    for (let i = 0; i < 4; i++) await toy.getByRole('button', { name: 'Next' }).click();
    await expectNoHorizontalScroll(page);
    await page.evaluate(() => window.scrollTo(0, 0));
    await expectNoA11yViolations(page);
  });
});

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the page never scrolls sideways, and the grid can be scrolled from the keyboard', async ({
    page,
  }) => {
    await page.goto(CHAPTER);
    const figure = await hydrated(page, 'step-words-need-context');
    await expectNoHorizontalScroll(page);
    const scroller = figure.getByRole('group', { name: 'Attention grid' });
    await expect(scroller).toHaveAttribute('tabindex', '0');
    await page.evaluate(() => window.scrollTo(0, 0));
    await expectNoA11yViolations(page);
  });
});
