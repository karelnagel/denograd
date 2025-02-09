import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    lib: {
      entry: 'web.ts',
      name: 'dg',
      fileName: (format) => format === 'iife' ? 'denograd.js' : 'denograd.mjs',
      formats: ['iife', 'es'],
    },
    emptyOutDir: false,
    outDir: './public',
    target: 'esnext',
    rollupOptions: {
      output: {
        globals: {},
      },
    },
  },
  publicDir: false,
})
