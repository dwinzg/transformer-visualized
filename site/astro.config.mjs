import mdx from '@astrojs/mdx';
import react from '@astrojs/react';
import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://dwinzg.github.io',
  base: '/transformer-visualized',
  trailingSlash: 'always',
  output: 'static',
  // Code blocks take their colors from the --astro-code-* tokens in src/styles/tokens.css,
  // so they follow the site theme and meet the contrast the token test enforces.
  markdown: { shikiConfig: { theme: 'css-variables' } },
  // GitHub Pages cannot send headers, so the policy goes in a meta tag. Astro hashes every inline
  // script and style, and no inline script may come before the policy (csp.spec.ts checks).
  // Inline style attributes stay allowed, since KaTeX, Shiki and the figures use them, and an
  // attribute cannot run code. That is also why Astro's warning about Shiki and CSP is harmless.
  security: {
    csp: {
      directives: [
        "default-src 'self'",
        "font-src 'self' data:",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
      ],
      styleDirective: {
        resources: [
          { resource: "'self'", kind: 'element' },
          { resource: "'unsafe-inline'", kind: 'attribute' },
        ],
      },
    },
  },
  integrations: [react(), mdx()],
  vite: {
    // The playground worker loads the same model and tokenizer as the page. Naming its copies the
    // same way means one file each, so readers download the model once and the offline cache keeps it.
    worker: {
      rollupOptions: { output: { assetFileNames: '_astro/[name].[hash][extname]' } },
    },
    build: {
      rollupOptions: {
        onLog(level, log, defaultHandler) {
          // @astrojs/mdx injects a "use astro:head-inject" directive into every .mdx module, and
          // Rolldown warns that it does not understand the directive when bundling. It is
          // expected and harmless, so this is the only warning silenced here.
          if (
            log.code === 'MODULE_LEVEL_DIRECTIVE' &&
            log.id?.endsWith('.mdx?astroPropagatedAssets')
          ) {
            return;
          }
          defaultHandler(level, log);
        },
      },
    },
  },
});
