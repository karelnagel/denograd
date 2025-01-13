// deno-lint-ignore-file no-process-globals
import type { Environment } from './abstract.ts'

const ENVS = {
  DENO: () => import('./deno.ts').then((x) => x.DenoEnv),
  BUN: () => import('./deno.ts').then((x) => x.DenoEnv),
  NODE: () => import('./deno.ts').then((x) => x.DenoEnv),
  WEB: () => import('./web.ts').then((x) => x.WebEnv),
}

export type Env = keyof typeof ENVS
const env: Env | undefined = typeof Deno !== 'undefined' ? 'DENO' : typeof Bun !== 'undefined' ? 'BUN' : typeof process !== 'undefined' ? 'NODE' : typeof window !== 'undefined' ? 'WEB' : undefined
if (!env) throw new Error('Unknown environment')

export const Env: Environment = await ENVS[env]().then((Environment) => new Environment(env))
