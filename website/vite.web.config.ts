import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    lib: {
      entry: 'web.ts',
      name: 'Denograd',
      fileName: (format) => format === 'iife' ? 'denograd.js' : 'denograd.es.js',
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
