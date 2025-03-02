import { setRuntime } from './index.ts'
import { CLANG } from '../runtime/ops_clang_deno.ts'
import { DISK } from '../runtime/ops_disk_deno.ts'
import { setDevices } from '../device.ts'
import { WEBGPU } from '../runtime/ops_webgpu.ts'
import { WASM } from '../runtime/ops_wasm.ts'
import { JS } from '../runtime/ops_js.ts'
import { CLANG as BunCLANG } from '../runtime/ops_clang_bun.ts'

if (typeof Deno !== 'undefined') {
  setRuntime(new (await import('./deno.ts').then((x) => x.DenoEnv))())
  setDevices({ CLANG, WEBGPU, WASM, JS, DISK })
} // @ts-ignore Bun
else if (typeof Bun !== 'undefined') {
  setRuntime(new (await import('./bun.ts').then((x) => x.BunEnv))())
  setDevices({ CLANG: BunCLANG, WASM, JS, DISK })
}
