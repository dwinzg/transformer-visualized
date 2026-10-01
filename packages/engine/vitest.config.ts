import { defineConfig } from 'vitest/config';

const config = {
  test: {
    include: ['test/**/*.test.ts'],
    globals: true,
  },
  benchmark: {
    include: ['bench/**/*.bench.ts'],
  },
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default defineConfig(config as any);
