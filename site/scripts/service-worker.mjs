// Runs after `astro build`. Writes dist/sw.js, which keeps every page and asset for offline use.
// The 5 MB model and the tokenizer are not fetched up front. They are kept the first time a page
// asks for them, so a reader who opened the playground once can use it offline too.
import { generateSW } from 'workbox-build';

const BASE = '/transformer-visualized/';

const { count, size, warnings } = await generateSW({
  globDirectory: 'dist',
  globPatterns: ['**/*.{html,js,css,svg,png,woff2,webmanifest}'],
  // The figure kit page is for development only.
  globIgnores: ['dev/**'],
  swDest: 'dist/sw.js',
  modifyURLPrefix: { '': BASE },
  // Links like learn/?depth=formula open the same page.
  ignoreURLParametersMatching: [/^depth$/, /^part$/, /^view$/, /^utm_/],
  // Astro already puts a hash in these names, so they need no second one.
  dontCacheBustURLsMatching: /\/_astro\//,
  cleanupOutdatedCaches: true,
  // A new version waits until every open tab is closed. Taking over an open page would delete the
  // old files its figures still load.
  clientsClaim: true,
  runtimeCaching: [
    {
      urlPattern: /\.(safetensors|json)$/,
      handler: 'CacheFirst',
      // One model and one tokenizer. A retrained model gets a new name and pushes the old one out.
      options: { cacheName: 'model', expiration: { maxEntries: 2 } },
    },
  ],
});
for (const warning of warnings) console.warn(warning);
console.log(`service worker: ${count} files, ${Math.round(size / 1024)} KB precached`);
