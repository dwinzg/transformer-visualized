import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

test.describe('home page', () => {
  test('offers ways to start', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('See inside a language model');
    await expect(page.getByRole('link', { name: /Learn from scratch/ })).toHaveAttribute(
      'href',
      '/transformer-visualized/learn/',
    );
    await expect(page.getByRole('link', { name: /Quick review/ })).toHaveAttribute(
      'href',
      '/transformer-visualized/learn/?depth=formula',
    );
    await expect(page.getByText('Playground')).toBeVisible();
  });

  test('has no accessibility violations in either theme', async ({ page }) => {
    await page.goto('./');
    await expectNoA11yViolations(page);
    await page.emulateMedia({ colorScheme: 'dark' });
    await expectNoA11yViolations(page);
  });
});

test.describe('theme toggle', () => {
  test('is hidden without JavaScript', async ({ browser }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    const page = await context.newPage();
    await page.goto('./');
    await expect(page.locator('[data-theme-toggle]').first()).toBeHidden();
    await context.close();
  });

  test('follows the system theme until the reader picks one', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto('./');
    await expect(page.getByRole('button', { name: 'Switch to dark theme' })).toBeVisible();
    await page.emulateMedia({ colorScheme: 'dark' });
    await expect(page.getByRole('button', { name: 'Switch to light theme' })).toBeVisible();
    await page.emulateMedia({ colorScheme: 'light' });
    await expect(page.getByRole('button', { name: 'Switch to dark theme' })).toBeVisible();
  });

  test('switches the theme and remembers it', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto('./');
    await page.getByRole('button', { name: 'Switch to dark theme' }).click();
    const html = page.locator('html');
    const themeColor = page.locator('meta[name="theme-color"][media="all"]');
    await expect(html).toHaveAttribute('data-theme', 'dark');
    await expect(html).toHaveCSS('color-scheme', 'dark');
    await expect(themeColor).toHaveAttribute('content', '#0b0b0d');
    await page.reload();
    await expect(html).toHaveAttribute('data-theme', 'dark');
    await expect(html).toHaveCSS('color-scheme', 'dark');
    await expect(themeColor).toHaveAttribute('content', '#0b0b0d');
    await expect(page.getByRole('button', { name: 'Switch to light theme' })).toBeVisible();
  });
});

test.describe('navigation', () => {
  test('skip link moves focus to the main content', async ({ page, browserName }) => {
    test.skip(browserName === 'webkit', 'WebKit does not move focus to links with Tab by default');
    await page.goto('./');
    await page.keyboard.press('Tab');
    await expect(page.getByRole('link', { name: 'Skip to content' })).toBeFocused();
    await page.keyboard.press('Enter');
    await expect(page.locator('main')).toBeFocused();
  });

  test('unknown pages show the not found page', async ({ page }) => {
    const response = await page.goto('no-such-page/');
    expect(response?.status()).toBe(404);
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('Page not found');
    await expectNoA11yViolations(page);
  });
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('the header fits on one row and still links home by name', async ({ page }) => {
    await page.goto('learn/');
    const home = page.getByRole('link', { name: 'Transformer Visualized' });
    await expect(home).toBeVisible();
    const header = await page.locator('.site-header').boundingBox();
    expect(header!.height).toBeLessThan(70);
    await home.click();
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('See inside a language model');
  });
});

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the home and not found pages do not scroll sideways', async ({ page }) => {
    await page.goto('./');
    await expectNoHorizontalScroll(page);
    await page.goto('no-such-page/');
    await expectNoHorizontalScroll(page);
  });
});
