import { CLANG } from './ops_clang.ts'
import { DISK } from './ops_disk.ts'
import { PYTHON } from './ops_python.ts'
import { WEBGPU } from './ops_webgpu.ts'

export const DEVICES = {
  WEBGPU,
  CLANG,
  DISK,
  PYTHON,
}

export type AllDevices = keyof typeof DEVICES
export type DeviceType = AllDevices | `${AllDevices}:${string}`
