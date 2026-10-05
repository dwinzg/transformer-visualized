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

test('picking a block explains it, with the formula at the Formula level', async ({ page }) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Masked multi-head attention' }).click();
  await expect(panel(map).getByRole('heading', { name: 'Masked attention' })).toBeVisible();
  await expect(panel(map)).toContainText('cannot look ahead');
  await expect(panel(map).locator('.katex')).toHaveCount(0);
  await page.getByRole('radio', { name: 'Formula', exact: true }).check();
  await expect(panel(map).locator('.katex')).toHaveCount(1);
});

test('the tour walks every part and stops at the end', async ({ page }) => {
  const map = await openMap(page);
  const next = map.getByRole('button', { name: 'Next' });
  for (let i = 0; i < 20; i++) await next.click();
  await expect(map.locator('.arch-count')).toHaveText('20 of 20');
  await expect(next).toHaveAttribute('aria-disabled', 'true');
  await expect(panel(map).getByRole('heading')).toHaveText('Output');
  await map.getByRole('button', { name: 'Previous' }).click();
  await expect(map.locator('.arch-count')).toHaveText('19 of 20');
});

test('the GPT-2 view drops the encoder and cross-attention and adds a final norm', async ({
  page,
}) => {
  const map = await openMap(page);
  await expect(map.getByRole('radio', { name: 'Outputs (shifted right)' })).toBeVisible();
  await map.getByRole('radio', { name: 'Multi-head attention', exact: true }).click();
  await expect(panel(map).getByRole('heading')).toHaveText('Cross-attention');
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
  await expect(panel(map).getByRole('heading')).toHaveText('Cross-attention');
  const unknown = await openMap(page, '?part=nope');
  await expect(panel(unknown)).toContainText('Tap a part');
});

test('a depth link reaches the map even though it loads later', async ({ page }) => {
  const map = await openMap(page, '?part=softmax&depth=formula');
  await expect(panel(map).locator('.katex')).toHaveCount(1);
  await expect(panel(map)).toContainText('4,096 chances');
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
  await expect(panel(map).getByRole('heading')).toBeInViewport();
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

test('the panel links to the glossary, and says when a chapter is still coming', async ({
  page,
}) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Feed forward' }).first().click();
  await expect(panel(map)).toContainText('Chapter coming soon');
  await map.getByRole('radio', { name: 'Inputs' }).first().click();
  await expect(panel(map).getByRole('link', { name: 'Read the chapter' })).toHaveAttribute(
    'href',
    /learn\/tokens\/#step-text-becomes-tokens$/,
  );
  await panel(map).getByRole('link', { name: 'What the word means' }).click();
  await expect(page).toHaveURL(/glossary\/#token$/);
});

test('the map is accessible after interaction in both themes', async ({ page }) => {
  const map = await openMap(page);
  await map.getByRole('radio', { name: 'Feed forward', exact: true }).click();
  await page.getByRole('radio', { name: 'Code', exact: true }).check();
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the page never scrolls sideways', async ({ page }) => {
    await openMap(page);
    await expectNoHorizontalScroll(page);
  });

  test('tapping a block shows its text', async ({ page }) => {
    const map = await openMap(page);
    await map.getByRole('radio', { name: 'Softmax' }).click();
    await expect(panel(map).getByRole('heading')).toBeInViewport();
  });
});
