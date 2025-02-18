import { CLANG } from './ops_clang.ts'
import { DISK } from './ops_disk.ts'
import { JS } from './ops_js.ts'
import { WASM } from './ops_wasm.ts'
import { WEBGPU } from './ops_webgpu.ts'

export const ALL_DEVICES = {
  CLANG,
  WEBGPU,
  WASM,
  DISK,
  JS,
}

export type AllDevices = keyof typeof ALL_DEVICES
export type DeviceType = AllDevices | `${AllDevices}:${string}`
