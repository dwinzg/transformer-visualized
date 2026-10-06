import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

test.describe('home page', () => {
  test('offers ways to start', async ({ page }) => {
    await page.goto('./');
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('See inside a language model');
    const ways = page.getByRole('region', { name: 'Ways to start' });
    await expect(ways.getByRole('link', { name: /Learn from scratch/ })).toHaveAttribute(
      'href',
      '/transformer-visualized/learn/',
    );
    await expect(ways.getByRole('link', { name: /The whole model/ })).toHaveAttribute(
      'href',
      '/transformer-visualized/learn/architecture/',
    );
    await expect(ways.getByRole('link', { name: /Quick review/ })).toHaveAttribute(
      'href',
      '/transformer-visualized/review/',
    );
    await expect(ways.getByRole('link', { name: /Playground/ })).toHaveAttribute(
      'href',
      '/transformer-visualized/playground/',
    );
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

test('figures still appear if the script that reveals them never runs', async ({ page }) => {
  await page.goto('learn/introduction/');
  const figure = page.locator('figure[data-enter]').first();
  // Undo what the script did, as if it had never loaded.
  await figure.evaluate((el) => {
    document.documentElement.removeAttribute('data-enter-ready');
    el.classList.remove('is-visible');
  });
  await expect(figure).toHaveCSS('opacity', '1', { timeout: 5000 });
});

test.describe('the Learn menu', () => {
  test.use({ viewport: { width: 1100, height: 800 } });

  test('opens from its button, lists every way to learn, and closes with Escape', async ({
    page,
  }) => {
    await page.goto('./');
    const button = page.getByRole('button', { name: 'More in Learn' });
    await expect(button).toHaveAttribute('aria-expanded', 'false');
    await button.click();
    await expect(button).toHaveAttribute('aria-expanded', 'true');
    const menu = page.locator('#learn-menu');
    for (const name of ['All chapters', '1. Tokens', 'The whole model', 'Quick review']) {
      await expect(menu.getByRole('link', { name })).toBeVisible();
    }
    await page.keyboard.press('Escape');
    await expect(menu).toBeHidden();
    await expect(button).toBeFocused();
  });

  test('opens on hover with a mouse, and its links go to the page', async ({ page, isMobile }) => {
    test.skip(isMobile, 'A touch screen has no hover. The button opens the menu there.');
    await page.goto('./');
    await page.getByRole('link', { name: 'Learn', exact: true }).hover();
    const menu = page.locator('#learn-menu');
    await menu.getByRole('link', { name: '3. Attention' }).click();
    await expect(page).toHaveURL(/learn\/attention\/$/);
    await expect(menu).toBeHidden();
  });

  test('closes when a click lands outside it', async ({ page }) => {
    await page.goto('./');
    await page.getByRole('button', { name: 'More in Learn' }).click();
    await page.locator('h1').click();
    await expect(page.locator('#learn-menu')).toBeHidden();
  });

  test('without JavaScript, Learn is a plain link and no menu buttons show', async ({
    browser,
  }) => {
    const context = await browser.newContext({ javaScriptEnabled: false });
    const page = await context.newPage();
    await page.goto('./');
    await expect(page.getByRole('button', { name: 'More in Learn' })).toBeHidden();
    await expect(page.locator('#learn-menu')).toBeHidden();
    await page.getByRole('link', { name: 'Learn', exact: true }).click();
    await expect(page).toHaveURL(/learn\/$/);
    await context.close();
  });
});

test.describe('on a phone', () => {
  test.use({ viewport: { width: 390, height: 844 } });

  test('one Menu button opens every link, and Escape closes it', async ({ page }) => {
    await page.goto('./');
    const menu = page.getByRole('button', { name: 'Menu' });
    await expect(page.getByRole('link', { name: 'Glossary' })).toBeHidden();
    await menu.click();
    for (const name of ['Learn', '2. Embeddings and position', 'The whole model', 'Playground']) {
      await expect(page.getByRole('link', { name, exact: true })).toBeVisible();
    }
    await expectNoA11yViolations(page);
    await page.keyboard.press('Escape');
    await expect(page.getByRole('link', { name: 'Glossary' })).toBeHidden();
    await expect(menu).toBeFocused();
  });

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
