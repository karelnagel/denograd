#!/usr/bin/env -S deno run -A
import { chromium } from 'npm:playwright'
import esbuild from 'npm:esbuild'
import process from 'node:process'
import { string_to_bytes } from '../denograd/helpers.ts'

const FORWARD_ENVS = ['DEBUG', 'D', 'DEVICE', 'JIT', 'BEAM', 'CACHELEVEL']

const [entry, ...args] = Deno.args
const build = await esbuild.build({
  entryPoints: [entry],
  format: 'esm',
  bundle: true,
  platform: 'browser',
  write: false,
  logLevel: 'error',
  target: ['chrome100'],
  external: ['./denograd/env/deno.ts', './denograd/env/bun.ts', './denograd/env/node.ts'],
  define: {
    'window.args': JSON.stringify(args),
    'process': JSON.stringify({ env: Object.fromEntries(FORWARD_ENVS.map((k) => [k, process.env[k]])) }),
  },
})

const code = build.outputFiles[0].text + ';console.log("ASYNC_CODE_COMPLETE");'

const browser = await chromium.launchPersistentContext('.playwright', { headless: !process.env.SHOW, args: ['--disable-web-security', '--enable-unsafe-webgpu', '--enable-features=Vulkan'] })
const page = await browser.newPage()
await page.goto('https://denograd.com') // needed cause indexedDB won't work in about:blank
await page.setContent('<html><body></body></html>')

page.on('pageerror', (e) => {
  console.error(e.stack)
  throw e
})
page.on('dialog', (x) => {
  if (x.type() === 'prompt') x.accept(prompt(x.message())!)
  else throw new Error(`Unhandled dialog: ${x.type()}`)
})
const promise = new Promise<void>((res) => {
  page.on('console', (msg) => {
    const text = msg.text()
    if (text === 'ASYNC_CODE_COMPLETE') return res()
    if (text.includes('\u200B')) Deno.stdout.writeSync(string_to_bytes(text.replace('\u200B', '')))
    else console.log(text)
  })
})

await page.addScriptTag({ content: code, type: 'module' })

await promise

await browser.close()
