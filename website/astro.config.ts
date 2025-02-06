import { defineConfig } from 'astro/config'
import tailwind from '@astrojs/tailwind'
import preact from '@astrojs/preact'

// https://astro.build/config
export default defineConfig({
  site: 'https://karelnagel.github.io',
  output: 'static',
  integrations: [preact({ include: ['**/*[jt]sx'] }), tailwind()],
  devToolbar: { enabled: false },
})
