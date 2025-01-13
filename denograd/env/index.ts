// deno-lint-ignore-file no-process-globals
import type { Environment } from './abstract.ts'
declare const Bun: unknown

let Class
if (typeof Deno !== 'undefined') Class = await import('./deno.ts').then((x) => x.DenoEnv)
// else if (typeof Bun !== 'undefined') Class = await import('./deno.ts').then((x) => x.DenoEnv)
// else if (typeof process !== 'undefined') Class = await import('./deno.ts').then((x) => x.DenoEnv)
else if (typeof window !== 'undefined') Class = await import('./web.ts').then((x) => x.WebEnv)
else throw new Error('Unknown environment')

export const Env: Environment = new Class()
