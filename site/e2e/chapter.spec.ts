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

test('the learn page says what each level adds, and previews it on hover', async ({ page }) => {
  await page.goto('learn/');
  const note = page.locator('[data-depth-note]');
  await expect(note).toHaveText('Plain words and pictures. No math needed.');
  await page.getByRole('radio', { name: 'Formula' }).check();
  await expect(note).toHaveText('Adds the math, with every symbol explained.');
  await page.locator('[data-depth-dial] label', { hasText: 'Code' }).hover();
  await expect(note).toContainText('code that runs each step');
  await page.locator('h1').hover();
  await expect(note).toHaveText('Adds the math, with every symbol explained.');
});

test('each step starts at the story level with tabs for the others', async ({ page }) => {
  await page.goto(CHAPTER);
  const step = page.locator('#step-it-guesses-the-next-piece-of-text');
  const tabs = step.getByRole('tab');
  await expect(tabs).toHaveText(['Story', 'Numbers', 'Formula', 'Code']);
  await expect(step.getByRole('tab', { name: 'Story' })).toHaveAttribute('aria-selected', 'true');
  await expect(step.getByRole('tabpanel')).toContainText('your phone suggests the next word');
});

test('the selected level tab is visually distinct from the others in both themes', async ({
  page,
}) => {
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await page.goto(CHAPTER);
    const step = page.locator('#step-it-guesses-the-next-piece-of-text');
    const selected = step.getByRole('tab', { name: 'Story' });
    const unselected = step.getByRole('tab', { name: 'Numbers' });
    await expect(selected).toHaveAttribute('aria-selected', 'true');
    const [selectedBackground, unselectedBackground] = await Promise.all([
      selected.evaluate((el) => getComputedStyle(el).backgroundColor),
      unselected.evaluate((el) => getComputedStyle(el).backgroundColor),
    ]);
    expect(selectedBackground, colorScheme).not.toBe(unselectedBackground);
  }
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

test('the quick review link opens that page at the formula level', async ({ page }) => {
  await page.goto('learn/?depth=formula');
  await expect(page.getByRole('radio', { name: 'Formula' })).toBeChecked();
});

test('?depth= also works when it lands directly on a chapter page', async ({ page }) => {
  await page.goto(`${CHAPTER}?depth=formula`);
  await expect(
    page.locator('#step-it-guesses-the-next-piece-of-text').getByRole('tab', { name: 'Formula' }),
  ).toHaveAttribute('aria-selected', 'true');
});

test('?depth= sets the level for that page view only, and is not saved', async ({ page }) => {
  await page.goto(`${CHAPTER}?depth=formula`);
  await expect(
    page.locator('#step-it-guesses-the-next-piece-of-text').getByRole('tab', { name: 'Formula' }),
  ).toHaveAttribute('aria-selected', 'true');
  await expect(page).toHaveURL(new RegExp(`${CHAPTER}$`));

  await page.goto('learn/');
  await expect(page.getByRole('radio', { name: 'Story' })).toBeChecked();
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
  await expect(page.locator('[data-chapter-toolbar]')).toBeHidden();
  await context.close();
});

test('the toolbar counts steps as the reader scrolls', async ({ page }) => {
  await page.goto(CHAPTER);
  const count = page.locator('[data-chapter-toolbar] .step-count');
  await expect(count).toHaveText('Step 1 of 3');
  await page.locator('#step-inside-a-transformer-does-the-guessing').scrollIntoViewIfNeeded();
  await page.evaluate(() =>
    document
      .querySelector('#step-inside-a-transformer-does-the-guessing')
      ?.scrollIntoView({ block: 'center' }),
  );
  await expect(count).toHaveText('Step 3 of 3');
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 700 } });

  test('the toolbar sits at the bottom and never covers the end of the page', async ({ page }) => {
    await page.goto(CHAPTER);
    const toolbar = page.locator('[data-chapter-toolbar]');
    const box = await toolbar.boundingBox();
    expect(box && Math.round(box.y + box.height)).toBe(700);
    await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
    const recap = await page.getByRole('region', { name: 'Go deeper' }).boundingBox();
    const bar = await toolbar.boundingBox();
    expect(recap && bar && recap.y + recap.height).toBeLessThanOrEqual(bar!.y);
    const lastLink = await page.locator('footer a').last().boundingBox();
    expect(lastLink && lastLink.y + lastLink.height).toBeLessThanOrEqual(bar!.y);
  });
});

test('the pager links each chapter to its neighbors', async ({ page }) => {
  await page.goto(CHAPTER);
  const pager = page.getByRole('navigation', { name: 'Chapters' });
  await expect(pager.getByRole('link')).toHaveCount(1);
  await pager.getByRole('link', { name: /Tokens/ }).click();
  await expect(page).toHaveURL(/learn\/tokens\/$/);
  await expect(
    page
      .getByRole('navigation', { name: 'Chapters' })
      .getByRole('link', { name: /What a language model does/ }),
  ).toBeVisible();
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
