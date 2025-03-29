import { defineConfig } from 'astro/config'
import react from '@astrojs/react'
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://jsgrad.org',
  output: 'static',
  integrations: [react(), mdx()],
  devToolbar: { enabled: false },

  vite: {
    plugins: [tailwindcss()],
  },
})