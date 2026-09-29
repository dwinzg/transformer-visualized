import { expect, test } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

test('the glossary lists terms alphabetically with anchors', async ({ page }) => {
  await page.goto('glossary/');
  const terms = await page.locator('dt').allTextContents();
  expect(terms).toEqual([...terms].sort((a, b) => a.localeCompare(b)));
  await expect(page.locator('#token dt')).toHaveText('Token');
  await expectNoA11yViolations(page);
});

test('the references page groups sources and links to them', async ({ page }) => {
  await page.goto('references/');
  await expect(page.getByRole('heading', { name: 'The transformer' })).toBeVisible();
  const entry = page.locator('#vaswani2017');
  await expect(entry).toContainText('Vaswani, A., Shazeer, N.');
  await expect(entry).toContainText('arXiv:1706.03762v7');
  await expect(entry.getByRole('link', { name: 'Attention Is All You Need' })).toHaveAttribute(
    'href',
    'https://arxiv.org/abs/1706.03762v7',
  );
  await expect(page.locator('#shannon1951')).toContainText(
    'doi:10.1002/j.1538-7305.1951.tb01366.x',
  );
  await expectNoA11yViolations(page);
});

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the glossary and references pages do not scroll sideways', async ({ page }) => {
    for (const path of ['glossary/', 'references/']) {
      await page.goto(path);
      await expectNoHorizontalScroll(page);
    }
  });
});
