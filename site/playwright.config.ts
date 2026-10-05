import { defineConfig, devices } from '@playwright/test';

const PORT = 4321;

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: `http://localhost:${PORT}/transformer-visualized/`,
    trace: 'retain-on-failure',
    // A service worker answers requests before page.route can see them, so tests block it.
    // The offline test turns it back on.
    serviceWorkers: 'block',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'webkit', use: { ...devices['Desktop Safari'] } },
    { name: 'firefox', use: { ...devices['Desktop Firefox'] } },
    { name: 'mobile', use: { ...devices['iPhone 15'] } },
  ],
  webServer: {
    command: `npm run build && npm run preview -- --port ${PORT}`,
    url: `http://localhost:${PORT}/transformer-visualized/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    // Astro 7's `preview` command detects coding-agent shells and daemonizes itself in
    // that case, which exits the process Playwright is waiting on. Force the foreground
    // behavior so the web server stays attached to this process everywhere.
    env: { ASTRO_PREVIEW_BACKGROUND: 'false' },
  },
});
