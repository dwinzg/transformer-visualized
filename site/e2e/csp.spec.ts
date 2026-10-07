import { expect, test } from '@playwright/test';

// The built site carries a content security policy. Each page must load, hydrate and run its
// figures without the browser blocking a script, style or worker.
const PAGES = [
  '',
  'learn/',
  'learn/introduction/',
  'learn/tokens/',
  'learn/embeddings/',
  'learn/attention/',
  'learn/prediction/',
  'learn/context/',
  'learn/tools/',
  'learn/reasoning/',
  'learn/architecture/',
  'playground/',
  'glossary/',
  'references/',
  'review/',
  'search/?q=attention',
  'faq/',
  'no-such-page/',
  'dev/figures/',
];

for (const path of PAGES) {
  test(`the policy blocks nothing on /${path}`, async ({ page }) => {
    await page.addInitScript(() => {
      const seen: string[] = [];
      (window as unknown as { cspSeen: string[] }).cspSeen = seen;
      document.addEventListener('securitypolicyviolation', (e) =>
        seen.push(`${e.effectiveDirective} ${e.blockedURI}`),
      );
    });
    await page.goto(path);
    await expect(page.locator('meta[http-equiv="content-security-policy"]')).toHaveCount(1);
    // A policy in a meta tag only covers what comes after it, so no inline script may come first.
    const before = await page.evaluate(() => {
      const meta = document.querySelector('meta[http-equiv="content-security-policy"]')!;
      return [...document.querySelectorAll('script:not([src])')].filter(
        (s) => s.compareDocumentPosition(meta) & Node.DOCUMENT_POSITION_FOLLOWING,
      ).length;
    });
    expect(before).toBe(0);
    // Bringing each island into view wakes the ones that wait until they are seen.
    for (const island of await page.locator('astro-island').all()) {
      await island.scrollIntoViewIfNeeded();
    }
    await expect(page.locator('astro-island[ssr]')).toHaveCount(0, { timeout: 20_000 });
    await page.waitForLoadState('networkidle');
    if (path === 'playground/') {
      await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });
    }
    expect(await page.evaluate(() => (window as unknown as { cspSeen: string[] }).cspSeen)).toEqual(
      [],
    );
  });
}
