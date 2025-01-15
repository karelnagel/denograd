import { CLANG } from './ops_clang.ts'
import { DISK } from './ops_disk.ts'
import { PYTHON } from './ops_python.ts'
import { WEBGPU } from './ops_webgpu.ts'

export const ALL_DEVICES = {
  CLANG,
  WEBGPU,
  DISK,
  PYTHON,
}

export type AllDevices = keyof typeof ALL_DEVICES
export type DeviceType = AllDevices | `${AllDevices}:${string}`
