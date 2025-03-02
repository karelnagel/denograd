#!/usr/bin/env deno run -A
import { chromium } from 'npm:playwright'
import esbuild from 'npm:esbuild'

const build = await esbuild.build({
  entryPoints: [Deno.args[0]],
  format: 'esm',
  bundle: true,
  platform: 'browser',
  write: false,
  logLevel: 'error',
  target: ['chrome100'],
  external: ['./denograd/env/deno.ts', './denograd/env/bun.ts'],
})
const code = build.outputFiles[0].text + ';console.log("ASYNC_CODE_COMPLETE");'

const browser = await chromium.launch({ headless: true, args: ['--disable-web-security'] })
const page = await browser.newPage()
await page.goto('https://denograd.com')
await page.setContent('<html><body></body></html>')

page.on('pageerror', (e) => {
  console.error(e.stack)
  throw e
})
const promise = new Promise<void>((res) => {
  page.on('console', (msg) => (msg.text() === 'ASYNC_CODE_COMPLETE') ? res() : console.log(msg.text()))
})

await page.addScriptTag({ content: code, type: 'module' })

await promise

await browser.close()
