import { expect, test, type Page } from '@playwright/test';
import { expectNoA11yViolations, expectNoHorizontalScroll } from './a11y';

const PAGE = 'playground/';
const LONG =
  'Once upon a time, there was a little girl named Lily. She loved to play outside with her dog. One day, they found a big red ball in the park.';

async function ready(page: Page) {
  await page.goto(PAGE);
  await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });
}

test('only the playground downloads the model, once', async ({ page }) => {
  const fetched: string[] = [];
  page.on('request', (request) => {
    if (request.url().includes('.safetensors')) fetched.push(request.url());
  });
  await page.goto('');
  await page.waitForLoadState('networkidle');
  expect(fetched).toHaveLength(0);
  await ready(page);
  expect(fetched).toHaveLength(1);
});

test('the home page opens the playground', async ({ page }) => {
  await page.goto('');
  await page.getByRole('link', { name: /Playground/ }).click();
  await expect(page).toHaveURL(/playground\/$/);
});

test('it runs the model on the starting sentence and on new text', async ({ page }) => {
  await ready(page);
  const tokens = page.getByRole('list', { name: 'Tokens' }).getByRole('listitem');
  await expect(tokens).toHaveCount(6);
  await expect(page.locator('.sampling-summary')).toContainText('gets 11.5%');
  const box = page.getByRole('textbox', { name: 'Your text' });
  await box.fill('Once upon a time, there was a little');
  await expect(tokens).toHaveCount(9);
  await expect(page.locator('.sampling-summary')).toContainText('space boy');
  await page.getByRole('button', { name: 'Start over' }).first().click();
  await expect(tokens).toHaveCount(6);
});

test('a sampled token can be added to the text', async ({ page }) => {
  await ready(page);
  await page.getByRole('radio', { name: 'Greedy' }).check();
  await page.getByRole('button', { name: 'Sample' }).click();
  await page.getByRole('button', { name: /^Add .* to the text$/ }).click();
  await expect(page.getByRole('textbox', { name: 'Your text' })).toHaveValue(
    'Lily wanted to play with her ball',
  );
  await expect(page.getByRole('list', { name: 'Tokens' }).getByRole('listitem')).toHaveCount(7);
  // The Add button goes away, so the focus moves to Sample instead of getting lost.
  await expect(page.getByRole('button', { name: 'Sample' })).toBeFocused();
});

test('the picked layer and head stay while the text changes', async ({ page }) => {
  await ready(page);
  await page.getByRole('radio', { name: 'Attention' }).check();
  await page.getByRole('group', { name: 'Layer' }).getByRole('radio', { name: '3' }).check();
  await page.getByRole('textbox', { name: 'Your text' }).fill('Ben wanted to play with her');
  await expect(page.locator('.attn-summary')).toContainText('In layer 3, head 1');
  await expect(page.locator('.attn-summary')).toContainText('her');
});

test('long text shows the last 24 tokens in the grid', async ({ page }) => {
  await ready(page);
  await page.getByRole('textbox', { name: 'Your text' }).fill(LONG);
  await page.getByRole('radio', { name: 'Attention' }).check();
  await expect(page.locator('.playground-note')).toContainText('last 24 tokens');
  await expect(page.locator('.attn-grid tbody tr')).toHaveCount(24);
  // The cut moves as the text grows, and the picked token stays the same.
  const dog = page
    .getByRole('radiogroup', { name: 'Pick the token that looks' })
    .getByRole('radio', { name: 'space dog', exact: true });
  await dog.click();
  await page.getByRole('textbox', { name: 'Your text' }).fill(`${LONG} It`);
  await expect(page.locator('.tokenizer-count')).toContainText('36 tokens');
  await expect(dog).toHaveAttribute('aria-checked', 'true');
});

test('a link opens the playground with its text, and the address follows the text', async ({
  page,
}) => {
  await page.goto(`${PAGE}?text=${encodeURIComponent('The cat sat on the')}`);
  const box = page.getByRole('textbox', { name: 'Your text' });
  await expect(box).toHaveValue('The cat sat on the');
  await expect(page.getByRole('list', { name: 'Tokens' }).getByRole('listitem')).toHaveCount(5, {
    timeout: 20_000,
  });
  await box.fill('Once upon a time');
  await expect(page).toHaveURL(/\?text=Once\+upon\+a\+time$/);
  await page.getByRole('button', { name: 'Start over' }).first().click();
  await expect(page).toHaveURL(/playground\/$/);
});

test('Copy link copies the address with the text', async ({ page, context, browserName }) => {
  test.skip(browserName !== 'chromium', 'Only Chromium lets a test read the clipboard.');
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await ready(page);
  await page.getByRole('textbox', { name: 'Your text' }).fill('Ben ran');
  await expect(page).toHaveURL(/text=Ben\+ran/);
  await page.getByRole('button', { name: 'Copy link' }).click();
  await expect(page.getByRole('button', { name: 'Link copied' })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toMatch(/text=Ben\+ran$/);
});

test('every stage shows the real numbers', async ({ page }) => {
  await ready(page);
  await page.getByRole('radio', { name: 'Embeddings' }).check();
  await expect(page.locator('.strip-readout')).toContainText('Number 1 of 128.');
  await page.getByRole('radio', { name: 'Attention' }).check();
  await page.getByRole('group', { name: 'Layer' }).getByRole('radio', { name: '3' }).check();
  await page.getByRole('group', { name: 'Head' }).getByRole('radio', { name: '3' }).check();
  await expect(page.locator('.attn-summary')).toContainText('Lily 0.66');
  await page.getByRole('radio', { name: 'Feed forward' }).check();
  await expect(page.locator('.ffn-summary')).toContainText('of 512 neurons');
  await expect(page.locator('.ffn-figure .prob-bar')).toHaveCount(8);
});

test('an empty box asks for text', async ({ page }) => {
  await ready(page);
  await page.getByRole('textbox', { name: 'Your text' }).fill('');
  await expect(page.locator('.playground-status')).toContainText('Type something');
});

test('a failed download says so, and Try again loads it', async ({ page }) => {
  let fail = true;
  await page.route('**/*.safetensors', (route) => (fail ? route.abort() : route.continue()));
  await page.goto(PAGE);
  const retry = page.getByRole('button', { name: 'Try again' });
  await expect(retry).toBeVisible({ timeout: 20_000 });
  fail = false;
  await retry.click();
  await expect(page.getByRole('list', { name: 'Tokens' })).toBeVisible({ timeout: 20_000 });
});

test('the page is accessible in both themes', async ({ page }) => {
  await ready(page);
  for (const colorScheme of ['light', 'dark'] as const) {
    await page.emulateMedia({ colorScheme });
    await expectNoA11yViolations(page);
  }
});

test.describe('at 320px wide', () => {
  test.use({ viewport: { width: 320, height: 640 } });

  test('the page never scrolls sideways, even with long text', async ({ page }) => {
    await ready(page);
    await page.getByRole('textbox', { name: 'Your text' }).fill(LONG);
    await expect(page.locator('.tokenizer-count')).toContainText('35 tokens');
    for (const stage of ['Embeddings', 'Attention', 'Feed forward', 'Next token']) {
      await page.getByRole('radio', { name: stage }).check();
      await expectNoHorizontalScroll(page);
    }
    await page.evaluate(() => window.scrollTo(0, 0));
    await expectNoA11yViolations(page);
  });
});
