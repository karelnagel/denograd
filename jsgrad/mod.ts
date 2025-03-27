import { setEnv } from './env/index.ts'

let Env
if (typeof Deno !== 'undefined') Env = await import('./env/deno.ts').then((x) => new x.DenoEnv())
// @ts-ignore Bun
else if (typeof Bun !== 'undefined') Env = await import('./env/bun.ts').then((x) => new x.BunEnv())
else if (typeof window === 'undefined') Env = await import('./env/node.ts').then((x) => new x.NodeEnv())
else Env = await import('./env/web.ts').then((x) => new x.WebEnv())

if (Env) setEnv(Env)

export * from './exports.ts'
