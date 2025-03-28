import { setEnv } from './env/index.ts'

if (typeof Deno !== 'undefined') setEnv(await import('./env/deno.ts').then((x) => new x.DenoEnv()))
// @ts-ignore Bun
else if (typeof Bun !== 'undefined') setEnv(await import('./env/bun.ts').then((x) => new x.BunEnv()))
else if (typeof window === 'undefined') setEnv(await import('./env/node.ts').then((x) => new x.NodeEnv()))
else setEnv(await import('./env/web.ts').then((x) => new x.WebEnv()))

export * from './base.ts'
