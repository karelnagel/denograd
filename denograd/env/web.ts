// deno-lint-ignore-file no-process-global
import type { AllDevices } from '../device.ts'
import { Environment } from './abstract.ts'

export class WebEnv extends Environment {
  NAME = 'WEB'
  PLATFORM = 'web' as const
  override DEVICES: AllDevices[] = ['WEBGPU', 'WASM', 'JS']
  override CPU_DEVICE: AllDevices = 'JS'
  env = {
    // @ts-ignore import.meta.env
    get: (key: string) => (import.meta.env || process.env)[key] as string,
    set: (key: string, value: string) => {
      // @ts-ignore import.meta.env
      ;(import.meta.env || process.env)[key] = value
    },
  }
  readFileSync = (path: string) => this.notImplemented()
  writeFileSync = (path: string, data: Uint8Array) => this.notImplemented()
  removeSync = (path: string) => this.notImplemented()
  realPathSync = (path: string) => this.notImplemented()
  statSync = (path: string) => this.notImplemented()
  writeStdout = (p: Uint8Array) => this.notImplemented()
  makeTempFileSync = () => this.notImplemented()
  execSync = (command: string, { args }: { args?: string[] } = {}) => this.notImplemented()
  tmpdir = () => '/tmp'
  homedir = () => '/home'
  gunzipSync = (input: ArrayBuffer) => this.notImplemented()
  // TODO: find some sync implementation
  sha256 = (data: Uint8Array) => data
  override readTextFileSync = (path: string) => this.notImplemented()
}
