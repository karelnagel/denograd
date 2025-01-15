import { defineConfig } from 'astro/config'
import react from '@astrojs/react'
import tailwind from '@astrojs/tailwind'

// https://astro.build/config
export default defineConfig({
  site: 'https://karelnagel.github.io',
  base: 'denograd',
  output: 'static',
  integrations: [react(), tailwind()],
  devToolbar: { enabled: false },
})
