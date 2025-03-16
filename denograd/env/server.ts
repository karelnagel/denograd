// deno-lint-ignore-file no-process-global
import { setRuntime } from './index.ts'
import { setDevices } from '../device.ts'

let Runtime
if (typeof Deno !== 'undefined') Runtime = await import('./deno.ts').then((x) => new x.DenoEnv())
// @ts-ignore Bun
else if (typeof Bun !== 'undefined') Runtime = await import('./bun.ts').then((x) => new x.BunEnv())
else if (typeof process !== 'undefined') Runtime = await import('./node.ts').then((x) => new x.NodeEnv())

if (Runtime) {
  setRuntime(Runtime)
  setDevices(Runtime.DEVICES)
}
