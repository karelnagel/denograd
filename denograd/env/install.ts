import { setRuntime } from './index.ts'
import { CLANG } from '../runtime/ops_clang.ts'
import { DISK } from '../runtime/ops_disk.ts'
import { setDevices } from '../device.ts'
import { WEBGPU } from '../runtime/ops_webgpu.ts'
import { WASM } from '../runtime/ops_wasm.ts'
import { JS } from '../runtime/ops_js.ts'

if (typeof Deno !== 'undefined') {
  const Env = await import('./deno.ts').then((x) => x.DenoEnv)
  setRuntime(new Env())
  setDevices({ CLANG, WEBGPU, WASM, JS, DISK })
} // @ts-ignore Bun
else if (typeof Bun !== 'undefined') {
  const Env = await import('./deno.ts').then((x) => x.DenoEnv)
  setRuntime(new Env())
  setDevices({ CLANG, WEBGPU, WASM, JS, DISK })
}
