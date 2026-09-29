import { expect, test } from '@playwright/test';
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

test('without JavaScript every level is shown in order', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto(CHAPTER);
  const step = page.locator('#step-it-guesses-the-next-piece-of-text');
  await expect(step.locator('.level-label')).toHaveText(LEVELS);
  await context.close();
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

test('the chapter has no accessibility violations at any level in either theme', async ({
  page,
}) => {
  await page.goto(CHAPTER);
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    for (const level of LEVELS) {
      await page.getByRole('radio', { name: level }).check();
      await expectNoA11yViolations(page);
    }
  }
});

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
