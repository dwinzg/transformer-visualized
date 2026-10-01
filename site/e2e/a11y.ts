import AxeBuilder from '@axe-core/playwright';
import { expect, type Page } from '@playwright/test';

/**
 * Elements marked [data-enter] (see Figure.astro) fade in from opacity 0 the first time they
 * scroll into view. Axe can catch that transition's low-contrast intermediate frames, so settle
 * every one of them into view and fully opaque first. Cheap no-op on pages with none.
 */
async function settleEnterMotion(page: Page): Promise<void> {
  const targets = page.locator('[data-enter]');
  const count = await targets.count();
  for (let i = 0; i < count; i++) {
    const target = targets.nth(i);
    await target.scrollIntoViewIfNeeded();
    await expect(target).toHaveCSS('opacity', '1');
  }
}

/** Fails the test if axe finds WCAG 2.2 A or AA violations on the current page. */
export async function expectNoA11yViolations(page: Page): Promise<void> {
  await settleEnterMotion(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'wcag22aa', 'best-practice'])
    .analyze();
  expect(results.violations.map((v) => `${v.id}: ${v.help}`)).toEqual([]);
}

/** Fails the test if the page is wider than its viewport, which makes it scroll sideways. */
export async function expectNoHorizontalScroll(page: Page): Promise<void> {
  const { scrollWidth, clientWidth } = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(scrollWidth, 'page width').toBeLessThanOrEqual(clientWidth);
}
