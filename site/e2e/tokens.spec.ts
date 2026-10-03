import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const CHAPTER = 'learn/tokens/';

/** The first tokenizer figure, once the tokenizer has loaded. */
async function firstTokenizer(page: Page): Promise<Locator> {
  const figure = page.locator('figure', { has: page.locator('.tokenizer-figure') }).first();
  await figure.scrollIntoViewIfNeeded();
  // Set after hydration, once the tokenizer has loaded, so typing is never lost to a late render.
  await expect(figure.locator('.tokenizer-figure[data-ready]')).toHaveCount(1);
  return figure;
}

const TOKENIZER = /tokenizer[^/]*\.json$/;

test('the starting sentence shows its tokens before the tokenizer loads', async ({ page }) => {
  // Hold the download back for the whole test.
  await page.route(TOKENIZER, () => {});
  await page.goto(CHAPTER);
  const figure = page.locator('.tokenizer-figure').first();
  await figure.scrollIntoViewIfNeeded();
  const chips = figure.locator('.token-chip');
  await expect(chips).toHaveCount(6);
  await expect(chips.first()).toHaveText(/Lily/);
  await expect(chips.nth(1).locator('.visually-hidden')).toHaveText('space wanted, id 408');
  await expect(figure.locator('.tokenizer-count')).toHaveText('28 characters, 6 tokens');
});

test('a failed download says so, and typing again retries it', async ({ page }) => {
  await page.route(TOKENIZER, (route) => route.abort());
  await page.goto(CHAPTER);
  const figure = page.locator('.tokenizer-figure').first();
  await figure.scrollIntoViewIfNeeded();
  const box = figure.getByRole('textbox', { name: 'Your text' });
  await expect(box).not.toHaveAttribute('readonly');
  await box.fill('Hello');
  await expect(figure.locator('.tokenizer-count')).toHaveText(/did not load/);
  await page.unroute(TOKENIZER);
  await box.fill('Hello there');
  await expect(figure.locator('.tokenizer-count')).toHaveText('11 characters, 2 tokens');
});

test('it works from the keyboard alone', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await firstTokenizer(page);
  const startOver = figure.getByRole('button', { name: 'Start over' });
  const box = figure.getByRole('textbox', { name: 'Your text' });
  // WebKit leaves buttons out of the Tab order by default, so focus each control directly.
  await box.focus();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.type('Lily ran');
  await expect(figure.locator('.tokenizer-count')).toHaveText('8 characters, 2 tokens');
  await startOver.focus();
  await page.keyboard.press('Enter');
  await expect(box).toHaveValue('Lily wanted to play with her');
});

test('typing shows the new tokens and count, and Start over brings the sentence back', async ({
  page,
}) => {
  await page.goto(CHAPTER);
  const figure = await firstTokenizer(page);
  const box = figure.getByRole('textbox', { name: 'Your text' });
  await box.fill('Tokenization is fun!');
  await expect(figure.locator('.tokenizer-count')).toHaveText('20 characters, 7 tokens');
  await expect(figure.locator('.token-chip').first()).toHaveText(/To/);
  await expect(figure.locator('.token-chip').first().locator('.visually-hidden')).toHaveText(
    'To, id 2275',
  );
  await figure.getByRole('button', { name: 'Start over' }).click();
  await expect(box).toHaveValue('Lily wanted to play with her');
  await expect(figure.locator('.tokenizer-count')).toHaveText('28 characters, 6 tokens');
});

test('it explains cleaned punctuation and characters the model never saw', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await firstTokenizer(page);
  const box = figure.getByRole('textbox', { name: 'Your text' });
  await box.fill('It’s a café');
  await expect(figure).toContainText('Curly quotes, long dashes and other fancy punctuation');
  await expect(figure).toContainText('The model never saw é in training');
  await expect(figure.locator('.token-chip', { hasText: 'byte C3' })).toHaveCount(1);
  await box.fill('Hi 😀');
  await expect(figure.locator('.tokenizer-count')).toHaveText(/^4 characters/);
});

test('only the Tokens chapter downloads the tokenizer, once for every figure', async ({ page }) => {
  const fetched: string[] = [];
  page.on('request', (request) => {
    if (TOKENIZER.test(request.url())) fetched.push(request.url());
  });
  await page.goto('learn/introduction/');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForLoadState('networkidle');
  expect(fetched).toHaveLength(0);

  await page.goto(CHAPTER);
  const figures = page.locator('.tokenizer-figure');
  for (const figure of await figures.all()) {
    await figure.scrollIntoViewIfNeeded();
    await expect(figure).toHaveAttribute('data-ready');
  }
  expect(await figures.count()).toBeGreaterThan(1);
  expect(fetched).toHaveLength(1);
});

test('the map opens this chapter from the Inputs block', async ({ page }) => {
  await page.goto('learn/architecture/?part=input&view=gpt2');
  const map = page.locator('.arch-map');
  await map.scrollIntoViewIfNeeded();
  await map.getByRole('link', { name: 'Read the chapter' }).click();
  await expect(page).toHaveURL(/learn\/tokens\/#step-text-becomes-tokens$/);
});

test('the chapter is accessible after typing, in both themes', async ({ page }) => {
  await page.goto(CHAPTER);
  const figure = await firstTokenizer(page);
  await figure.getByRole('textbox', { name: 'Your text' }).fill('Zebras zigzag 😀');
  // Scan from the top, so the sticky header never half covers a target (see intro-figures.spec.ts).
  await page.evaluate(() => window.scrollTo(0, 0));
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('a long text wraps instead of scrolling sideways', async ({ page }) => {
    await page.goto(CHAPTER);
    const figure = await firstTokenizer(page);
    await figure
      .getByRole('textbox', { name: 'Your text' })
      .fill('Supercalifragilisticexpialidocious '.repeat(5));
    await expectNoHorizontalScroll(page);
  });
});
