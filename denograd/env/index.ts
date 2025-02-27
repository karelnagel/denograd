// deno-lint-ignore-file no-process-global
import type { Environment } from './abstract.ts'
import { WebEnv } from './web.ts'

// @ts-ignore import.meta.env
if (typeof process !== 'undefined' && typeof import.meta.env === 'undefined') import.meta.env = { SSR: true }

let Class
// Server envs
// @ts-ignore import.meta.env
if (import.meta.env.SSR) {
  if (typeof Deno !== 'undefined') Class = await import('./deno.ts').then((x) => x.DenoEnv)
  // else if (typeof Bun !== 'undefined') Class = await import('./deno.ts').then((x) => x.DenoEnv)
  // else if (typeof process !== 'undefined') Class = await import('./deno.ts').then((x) => x.DenoEnv)
} // Browser
else {
  if (typeof window !== 'undefined') Class = WebEnv
}

if (!Class) throw new Error('Unknown environment')

export const Env: Environment = new Class()
if (Env.env.get('DEBUG')) console.log(`Using env ${Env.NAME}`)
