import { expect, test, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const CHAPTER = 'learn/introduction/';
const LEVELS = ['Story', 'Numbers', 'Formula', 'Code'];

test.beforeEach(async ({ page }) => {
  await page.goto('./');
  await page.evaluate(() => localStorage.clear());
});

test('the learn page lists the introduction', async ({ page }) => {
  await page.goto('learn/');
  await expect(page.getByRole('link', { name: /What a language model does/ })).toHaveAttribute(
    'href',
    '/transformer-visualized/learn/introduction/',
  );
  await expectNoA11yViolations(page);
  await page.emulateMedia({ colorScheme: 'dark' });
  await expectNoA11yViolations(page);
});

test('each step starts at the story level with tabs for the others', async ({ page }) => {
  await page.goto(CHAPTER);
  const step = page.locator('#step-it-guesses-the-next-piece-of-text');
  const tabs = step.getByRole('tab');
  await expect(tabs).toHaveText(['Story', 'Numbers', 'Formula', 'Code']);
  await expect(step.getByRole('tab', { name: 'Story' })).toHaveAttribute('aria-selected', 'true');
  await expect(step.getByRole('tabpanel')).toContainText('your phone suggests the next word');
});

test('arrow keys move between levels and the choice applies to every step', async ({ page }) => {
  await page.goto(CHAPTER);
  const first = page.locator('#step-it-guesses-the-next-piece-of-text');
  await first.getByRole('tab', { name: 'Story' }).focus();
  await page.keyboard.press('ArrowRight');
  await page.keyboard.press('ArrowRight');
  await expect(first.getByRole('tab', { name: 'Formula' })).toBeFocused();
  await expect(first.locator('.katex').first()).toBeVisible();
  const second = page.locator('#step-then-it-does-it-again');
  await expect(second.getByRole('tab', { name: 'Formula' })).toHaveAttribute(
    'aria-selected',
    'true',
  );
  await page.reload();
  await expect(first.getByRole('tab', { name: 'Formula' })).toHaveAttribute(
    'aria-selected',
    'true',
  );
});

test('the depth dial changes every step', async ({ page }) => {
  await page.goto(CHAPTER);
  await page.getByRole('radio', { name: 'Code' }).check();
  await expect(
    page.locator('#step-then-it-does-it-again').getByRole('tab', { name: 'Code' }),
  ).toHaveAttribute('aria-selected', 'true');
});

test('the quick review link opens chapters at the formula level', async ({ page }) => {
  await page.goto('learn/?depth=formula');
  await page.goto(CHAPTER);
  await expect(
    page.locator('#step-it-guesses-the-next-piece-of-text').getByRole('tab', { name: 'Formula' }),
  ).toHaveAttribute('aria-selected', 'true');
});

test('?depth= also works when it lands directly on a chapter page', async ({ page }) => {
  await page.goto(`${CHAPTER}?depth=formula`);
  await expect(
    page.locator('#step-it-guesses-the-next-piece-of-text').getByRole('tab', { name: 'Formula' }),
  ).toHaveAttribute('aria-selected', 'true');
});

test('without JavaScript every level is shown in order and the depth dial is hidden', async ({
  browser,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(CHAPTER);
  const step = page.locator('#step-it-guesses-the-next-piece-of-text');
  await expect(step.locator('.level-label')).toHaveText(LEVELS);
  await expect(page.locator('[data-depth-dial]').first()).toBeHidden();
  await context.close();
});

test('the chapter has no pager when there is no other chapter', async ({ page }) => {
  await page.goto(CHAPTER);
  await expect(page.getByRole('navigation', { name: 'Chapters' })).toHaveCount(0);
});

test('the chapter ends with a recap and links to go deeper', async ({ page }) => {
  await page.goto(CHAPTER);
  await expect(page.getByRole('region', { name: 'Recap' }).getByRole('listitem')).toHaveCount(3);
  await expect(
    page
      .getByRole('region', { name: 'Go deeper' })
      .getByRole('link', { name: 'Vaswani et al., 2017' }),
  ).toHaveAttribute('href', '/transformer-visualized/references/#vaswani2017');
});

// One test per depth level, each scanning both themes: running all four levels and both themes
// in a single test made 8 axe scans, which was slow enough to hit Firefox's 30s test timeout.
for (const level of LEVELS) {
  test(`the chapter has no accessibility violations at the ${level} level in either theme`, async ({
    page,
  }) => {
    await page.goto(CHAPTER);
    for (const colorScheme of ['light', 'dark'] as const) {
      await page.emulateMedia({ colorScheme });
      await page.getByRole('radio', { name: level }).check();
      await expectNoA11yViolations(page);
    }
  });
}

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the learn pages do not scroll sideways at any level', async ({ page }) => {
    await page.goto('learn/');
    await expectNoHorizontalScroll(page);
    await page.goto(CHAPTER);
    for (const level of LEVELS) {
      await page.getByRole('radio', { name: level }).check();
      await expectNoHorizontalScroll(page);
    }
  });
});


test.describe('anchored links land below the sticky header', () => {
  // The nav wraps to more rows at these widths (see SiteHeader.astro), which is what pushed
  // anchor targets under the header before the fix.
  for (const width of [390, 320]) {
    test(`a glossary link and a reference link at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 700 });

      await page.goto(CHAPTER);
      await page.getByRole('button', { name: 'language model' }).first().click();
      await page
        .locator('.term-card:popover-open')
        .getByRole('link', { name: 'Open the glossary' })
        .click();
      await expect(page).toHaveURL(/glossary\/#language-model$/);
      await expectBelowHeader(page, '#language-model');

      await page.goto(CHAPTER);
      await page.locator('a.ref', { hasText: 'Shannon, 1951' }).first().click();
      await expect(page).toHaveURL(/references\/#shannon1951$/);
      await expectBelowHeader(page, '#shannon1951');
    });
  }
});

async function expectBelowHeader(page: Page, targetSelector: string): Promise<void> {
  const header = page.locator('.site-header');
  const target = page.locator(targetSelector);
  const headerBox = await header.boundingBox();
  const targetBox = await target.boundingBox();
  if (!headerBox || !targetBox)
    throw new Error('Expected both the header and the target to be visible');
  expect(targetBox.y, `${targetSelector} top vs header bottom`).toBeGreaterThanOrEqual(
    headerBox.y + headerBox.height,
  );
}
