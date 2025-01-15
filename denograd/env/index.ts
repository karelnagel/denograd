// deno-lint-ignore-file no-process-globals
import type { Environment } from './abstract.ts'
import { WebEnv } from './web.ts'
declare const Bun: unknown

let Class
// Server envs
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
