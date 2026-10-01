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
  integrations: [react(), mdx()],
  vite: {
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
