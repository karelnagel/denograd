import { DeviceType } from '../device.ts'
import { Environment } from './abstract.ts'

export class WebEnv extends Environment {
  name = 'web' as const
  platform = 'web' as const
  override supportedDevices: DeviceType[] = ['PYTHON']
  env = {
    get: (key: string) => (import.meta as any).env?.[key],
    set: (key: string, value: string) => {
      ;(import.meta as any).env[key] = value
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
  sha256 = (data: string | Uint8Array) => this.notImplemented()
  override readTextFileSync = (path: string) => this.notImplemented()
}
