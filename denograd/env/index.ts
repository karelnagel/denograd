import { Environment } from './abstract.ts'

let Class
if (typeof Deno !== undefined) Class = await import('./deno.ts').then((x) => x.DenoEnv)
//  else if (typeof Bun !== undefined) Class = await import("./bun.ts").then(x=>x.BunEnv)
// else if (typeof process !== undefined) Class = await import('./node.ts').then((x) => x.NodeEnv)
else if (typeof window !== undefined) Class = await import('./web.ts').then((x) => x.WebEnv)
else throw new Error('Unknown environment')

export const Env: Environment = new Class()
