import type { DeviceType } from '../runtime/all.ts'

export abstract class Environment {
  abstract NAME: string
  abstract PLATFORM: 'aix' | 'android' | 'haiku' | 'cygwin' | 'netbsd' | 'darwin' | 'freebsd' | 'linux' | 'openbsd' | 'sunos' | 'win32' | 'web'
  DEVICES: DeviceType[] | undefined
  CPU_DEVICE: DeviceType = 'CLANG'
  notImplemented = () => {
    throw new Error(`This feature is not available in ${this.NAME} environment`)
  }
  abstract readFileSync: (path: string) => Uint8Array
  readTextFileSync = (path: string) => new TextDecoder().decode(this.readFileSync(path))
  abstract writeFileSync: (path: string, data: Uint8Array) => void
  writeTextFileSync = (path: string, data: string) => this.writeFileSync(path, new TextEncoder().encode(data))
  abstract removeSync: (path: string) => void
  abstract realPathSync: (path: string) => string
  abstract statSync: (path: string) => { size: number }
  abstract writeStdout: (p: Uint8Array) => Promise<number>
  abstract execSync: (command: string, args: { args: string[] }) => { success: boolean; stdout: Uint8Array; stderr: Uint8Array }
  abstract tmpdir: () => string
  abstract homedir: () => string
  abstract env: { get: (k: string) => string | undefined; set: (k: string, v: string) => void }
  abstract sha256: (v: Uint8Array) => Uint8Array
}
