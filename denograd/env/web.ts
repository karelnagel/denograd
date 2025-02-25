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
  
  private _db: IDBDatabase | undefined = undefined
  private _open_db = async (): Promise<IDBDatabase> => {
    if (this._db) return this._db
    this._db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open('denograd', this.DB_VERSION)
      req.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result
        db.createObjectStore('denograd', { keyPath: 'key' })
      }
      req.onsuccess = (event) => resolve((event.target as IDBOpenDBRequest).result)
      req.onerror = () => reject(req.error)
    })
    return this._db
  }

  disk_get = async (table: string, key: string): Promise<IDBDatabase | undefined> => {
    const db = await this._open_db()
    const transaction = db.transaction(['denograd'], 'readonly')
    const store = transaction.objectStore('denograd')
    const request = store.get(`${table}_${key}`)

    return await new Promise<any | undefined>((resolve, reject) => {
      request.onsuccess = () => {
        const data = request.result
        resolve(data ? data.value : undefined)
      }
      request.onerror = () => reject(request.error)
    })
  }

  disk_put = async (table: string, key: string, value: any) => {
    const db = await this._open_db()
    const transaction = db.transaction(['denograd'], 'readwrite')
    const store = transaction.objectStore('denograd')
    const request = store.put({ key: `${table}_${key}`, value })

    await new Promise<void>((resolve, reject) => {
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }
}
