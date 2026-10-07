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
  'learn/architecture/',
  'playground/',
  'glossary/',
  'references/',
  'review/',
  'no-such-page/',
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
    // Scrolling to the end wakes every island that waits until it is seen.
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await page.waitForLoadState('networkidle');
    if (path === 'playground/') {
      await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });
    }
    expect(await page.evaluate(() => (window as unknown as { cspSeen: string[] }).cspSeen)).toEqual(
      [],
    );
  });
}
