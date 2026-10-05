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
  cleanupOutdatedCaches: true,
  clientsClaim: true,
  skipWaiting: true,
  runtimeCaching: [
    {
      urlPattern: /\.(safetensors|json)$/,
      handler: 'CacheFirst',
      options: { cacheName: 'model' },
    },
  ],
});
for (const warning of warnings) console.warn(warning);
console.log(`service worker: ${count} files, ${Math.round(size / 1024)} KB precached`);
