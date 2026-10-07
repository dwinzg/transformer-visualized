import { expect, test, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const results = (page: Page) => page.locator('[data-search-results] > li');

test('typing finds a term, and its result opens the glossary entry', async ({ page }) => {
  await page.goto('search/');
  await page.getByLabel(/Search the chapters/).fill('softmax');
  await expect(page.getByRole('status')).toHaveText(/results? for “softmax”/);
  const first = results(page).first();
  await expect(first).toContainText('Glossary');
  await expect(results(page).locator('mark').first()).toHaveText(/softmax/i);
  await expect(page).toHaveURL(/\?q=softmax/);
  await first.getByRole('link').click();
  await expect(page).toHaveURL(/glossary\/#softmax$/);
});

test('a search in the address shows its results, and Enter moves to the first one', async ({
  page,
}) => {
  await page.goto('search/?q=causal+mask');
  await expect(page.getByLabel(/Search the chapters/)).toHaveValue('causal mask');
  await expect(results(page).first()).toBeVisible();
  await page.getByLabel(/Search the chapters/).press('Enter');
  await expect(results(page).first().getByRole('link')).toBeFocused();
});

test('a search with no match says so', async ({ page }) => {
  await page.goto('search/?q=zzzzqx');
  await expect(page.getByRole('status')).toHaveText(
    'Nothing found for “zzzzqx”. Try fewer or shorter words.',
  );
  await expect(results(page)).toHaveCount(0);
});

test('every place in the search index exists on its page', async ({ page, request }) => {
  await page.goto('search/');
  const hrefs = await page.evaluate(() =>
    (JSON.parse(document.getElementById('search-data')!.textContent!) as { href: string }[]).map(
      (e) => e.href,
    ),
  );
  const pages = new Map<string, string>();
  for (const href of hrefs) {
    const [path, hash] = href.split('#');
    if (!pages.has(path)) {
      const response = await request.get(path);
      expect(response.ok(), path).toBe(true);
      pages.set(path, await response.text());
    }
    if (hash) expect(pages.get(path), href).toContain(`id="${hash}"`);
  }
});

test('the header links to search, and the Learn menu to the questions', async ({ page }) => {
  await page.goto('');
  const nav = page.getByRole('navigation', { name: 'Main' });
  const menu = nav.getByRole('button', { name: 'Menu' });
  if (await menu.isVisible()) await menu.click();
  await nav.getByRole('link', { name: 'Search' }).click();
  await expect(page).toHaveURL(/search\/$/);
  await expect(page.getByRole('heading', { level: 1, name: 'Search' })).toBeVisible();
  await expect(
    nav.getByRole('link', { name: 'Questions and answers', includeHidden: true }),
  ).toHaveAttribute('href', /faq\/$/);
});

test('each answer links to steps and sources that exist', async ({ page, request }) => {
  await page.goto('faq/');
  const links = await page
    .locator('.qa a')
    .evaluateAll((as) => as.map((a) => (a as HTMLAnchorElement).getAttribute('href')!));
  expect(links.length).toBeGreaterThan(15);
  for (const href of new Set(links)) {
    const [path, hash] = href.split('#');
    const response = await request.get(path);
    expect(response.ok(), href).toBe(true);
    if (hash) expect(await response.text(), href).toContain(`id="${hash}"`);
  }
});

for (const path of ['search/?q=attention', 'faq/']) {
  test(`/${path} is accessible and fits a phone`, async ({ page }) => {
    await page.goto(path);
    if (path.startsWith('search')) await expect(results(page).first()).toBeVisible();
    await expectNoA11yViolations(page);
    await page.emulateMedia({ colorScheme: 'dark' });
    await expectNoA11yViolations(page);
    await page.setViewportSize({ width: 320, height: 640 });
    await expectNoHorizontalScroll(page);
  });
}
