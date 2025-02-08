import { defineConfig } from 'astro/config'
import tailwind from '@astrojs/tailwind'
import react from '@astrojs/react'

// https://astro.build/config
export default defineConfig({
  site: 'https://denograd.io',
  output: 'static',
  integrations: [react(), tailwind()],
  devToolbar: { enabled: false },
})
