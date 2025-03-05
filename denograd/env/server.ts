import { setRuntime } from './index.ts'
import { setDevices } from '../device.ts'

let Runtime
if (typeof Deno !== 'undefined') Runtime = await import('./deno.ts').then((x) => new x.DenoEnv())
// @ts-ignore Bun
else if (typeof Bun !== 'undefined') Runtime = await import('./bun.ts').then((x) => new x.BunEnv())

if (Runtime) {
  setRuntime(Runtime)
  setDevices(Runtime.DEVICES)
}
