import { expect, test, type Locator, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const MAP = 'learn/architecture/';

/** Opens the map and waits until the island has hydrated. */
async function openMap(page: Page, query = ''): Promise<Locator> {
  await page.goto('./');
  await page.evaluate(() => localStorage.clear());
  await page.goto(MAP + query);
  const map = page.locator('.arch-map');
  await map.scrollIntoViewIfNeeded();
  await expect(page.locator('astro-island', { has: map })).not.toHaveAttribute('ssr');
  return map;
}

const panel = (map: Locator) => map.locator('.arch-panel');
const title = (map: Locator) => panel(map).locator('.arch-title');
const article = (page: Page) => page.locator('#part-article');

test('picking a block explains it beside the map and in full below it', async ({ page }) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Masked multi-head attention' }).click();
  await expect(title(map)).toHaveText('Masked attention');
  await expect(panel(map)).toContainText('cannot look ahead');
  const full = article(page);
  await expect(full.getByRole('heading', { level: 2 })).toHaveText('Masked attention');
  await expect(full.locator('.katex')).toHaveCount(1);
  for (const name of ['Sizes', 'Formula', 'In PyTorch', 'In our engine']) {
    await expect(full.getByRole('heading', { level: 3, name })).toBeVisible();
  }
  await expect(full.locator('.arch-prints')).toContainText('0.1737');
});

test('the article pages through the tour and moves to the next heading', async ({ page }) => {
  await openMap(page, '?part=final-norm&view=gpt2');
  const full = article(page);
  await expect(full.locator('.arch-prints')).toContainText('tensor([ 0.8167,  2.4633, -3.1536');
  await full.getByRole('button', { name: /^Next/ }).click();
  await expect(full.getByRole('heading', { level: 2 })).toHaveText('Linear');
  await expect(full.getByRole('heading', { level: 2 })).toBeFocused();
  await expect(page.locator('.arch-count')).toHaveText('10 of 12');
  await full.getByRole('button', { name: /^Previous Final norm/ }).click();
  await expect(full.getByRole('heading', { level: 2 })).toHaveText('Final norm');
});

test("the paper's view adds the paper's version of a part", async ({ page }) => {
  await openMap(page, '?part=position');
  const full = article(page);
  await expect(full.getByRole('heading', { name: "The paper's version" })).toBeVisible();
  await expect(full).toContainText('tensor([0.8415, 0.5403, 0.7617, 0.6479])');
});

test('the setup and every snippet can be copied', async ({ page, context, browserName }) => {
  test.skip(browserName !== 'chromium', 'Only Chromium lets a test read the clipboard.');
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await openMap(page, '?part=embedding&view=gpt2');
  const full = article(page);
  await full.getByText('Run the code yourself').click();
  await full.getByRole('button', { name: 'Copy PyTorch setup' }).click();
  await expect(full.getByRole('button', { name: 'Copied PyTorch setup' })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('load_file');
});

test('the tour walks every part and stops at the end', async ({ page }) => {
  const map = await openMap(page);
  const next = map.getByRole('button', { name: 'Next' });
  for (let i = 0; i < 20; i++) await next.click();
  await expect(map.locator('.arch-count')).toHaveText('20 of 20');
  await expect(next).toHaveAttribute('aria-disabled', 'true');
  await expect(title(map)).toHaveText('Output');
  await map.getByRole('button', { name: 'Previous' }).click();
  await expect(map.locator('.arch-count')).toHaveText('19 of 20');
});

test('the GPT-2 view drops the encoder and cross-attention and adds a final norm', async ({
  page,
}) => {
  const map = await openMap(page);
  await expect(map.getByRole('radio', { name: 'Outputs (shifted right)' })).toBeVisible();
  await map.getByRole('radio', { name: 'Multi-head attention', exact: true }).click();
  await expect(title(map)).toHaveText('Cross-attention');
  const original = map.getByRole('radio', { name: 'Original paper' });
  const gpt2 = map.getByRole('radio', { name: 'GPT-2 style (our model)' });
  await original.focus();
  await page.keyboard.press('ArrowRight');
  await expect(gpt2).toBeChecked();
  await expect(map.getByRole('radio', { name: 'Outputs (shifted right)' })).toHaveCount(0);
  await expect(map.getByRole('radio', { name: 'Multi-head attention', exact: true })).toHaveCount(
    0,
  );
  await expect(map.getByRole('radio', { name: 'Final norm' })).toBeVisible();
  // The picked block was hidden, so the panel resets. Focus stays on the toggle.
  await expect(panel(map)).toContainText('Tap a part');
  await expect(map.locator('.arch-count')).toHaveText('12 parts');
  await expect(gpt2).toBeFocused();
  await page.keyboard.press('ArrowLeft');
  await expect(original).toBeChecked();
  await expect(original).toBeFocused();
});

test("a lesson link opens our model's block, not the encoder's", async ({ page }) => {
  const map = await openMap(page, '?part=input&view=gpt2');
  await expect(map.getByRole('radio', { name: 'Inputs' })).toHaveAttribute('aria-checked', 'true');
  const gpt2 = map.getByRole('radio', { name: 'GPT-2 style (our model)' });
  await expect(gpt2).toBeChecked();
  await gpt2.focus();
  await page.keyboard.press('ArrowLeft');
  await map.getByRole('radio', { name: 'Inputs' }).click();
  await expect(panel(map)).toContainText('the sentence to translate');
  await expect(panel(map)).not.toContainText('shifted');
});

test('a link can open the map at one part', async ({ page }) => {
  const map = await openMap(page, '?part=cross-attn&view=gpt2');
  await expect(title(map)).toHaveText('Cross-attention');
  const unknown = await openMap(page, '?part=nope');
  await expect(panel(unknown)).toContainText('Tap a part');
});

test('a part link shows its whole article even though the map loads later', async ({ page }) => {
  await openMap(page, '?part=softmax');
  await expect(article(page).locator('.katex')).toHaveCount(1);
  await expect(article(page)).toContainText('4,096 chances');
});

test('arrow keys move through the parts', async ({ page }) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Inputs' }).first().focus();
  await page.keyboard.press('ArrowUp');
  await expect(map.getByRole('radio', { name: 'Input embedding' })).toBeFocused();
  await expect(map.getByRole('radio', { name: 'Input embedding' })).toHaveAttribute(
    'aria-checked',
    'true',
  );
  await page.keyboard.press('End');
  await expect(map.getByRole('radio', { name: 'Output probabilities' })).toBeFocused();
});

test('the tour keeps its buttons and the text on screen', async ({ page }) => {
  const map = await openMap(page);
  const next = map.getByRole('button', { name: 'Next' });
  for (let i = 0; i < 3; i++) await next.click();
  await expect(next).toBeInViewport();
  await expect(title(map)).toBeInViewport();
});

test('start over at rest does nothing, so the tour still advances after it', async ({ page }) => {
  const map = await openMap(page);
  const next = map.getByRole('button', { name: 'Next' });
  // Playwright treats aria-disabled as disabled, so force the click a reader can still make.
  await map.getByRole('button', { name: 'Start over' }).click({ force: true });
  await next.focus();
  await page.keyboard.press('Enter');
  await page.keyboard.press('Enter');
  await expect(map.locator('.arch-count')).toHaveText('2 of 20');
  await expect(next).toBeFocused();
});

test('start over clears the pick and returns to the original view', async ({ page }) => {
  const map = await openMap(page, '?view=gpt2');
  await map.getByRole('radio', { name: 'Softmax' }).click();
  await map.getByRole('button', { name: 'Start over' }).click();
  await expect(panel(map)).toContainText('Tap a part');
  await expect(map.getByRole('radio', { name: 'Outputs (shifted right)' })).toBeVisible();
});

test('the article links to its chapter and the glossary', async ({ page }) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Feed forward' }).first().click();
  await expect(article(page).getByRole('link', { name: 'Read the chapter' })).toHaveCount(0);
  await map.getByRole('radio', { name: 'Inputs' }).first().click();
  await expect(article(page).getByRole('link', { name: 'Read the chapter' })).toHaveAttribute(
    'href',
    /learn\/tokens\/#step-text-becomes-tokens$/,
  );
  await article(page).getByRole('link', { name: 'What the word means' }).click();
  await expect(page).toHaveURL(/glossary\/#token$/);
});

test('the map is accessible after interaction in both themes', async ({ page }) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Feed forward', exact: true }).click();
  await article(page).getByText('Run the code yourself').click();
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the page never scrolls sideways, with an article open', async ({ page }) => {
    await openMap(page, '?part=masked-attn&view=gpt2');
    await expect(article(page).getByRole('heading', { level: 2 })).toBeVisible();
    await expectNoHorizontalScroll(page);
  });

  test('tapping a block shows its text', async ({ page }) => {
    const map = await openMap(page);
    await map.getByRole('radio', { name: 'Softmax' }).click();
    await expect(title(map)).toBeInViewport();
  });
});
