// deno-lint-ignore-file no-process-global
import type { AllDevices } from '../device.ts'

export class WebEnv {
  NAME = 'WEB'
  PLATFORM = 'web'
  DEVICES: AllDevices[] = ['WEBGPU', 'WASM', 'JS']
  CPU_DEVICE: AllDevices = 'JS'
  DB_VERSION = 1
  // @ts-ignore import.meta.env
  _env!: Record<string, string> = (typeof import.meta !== 'undefined' ? import.meta?.env : typeof process !== 'undefined' ? process.env : {}) || {}

  // env
  get = (key: string, def?: string) => this._env[key] || def
  get_num = (key: string, def?: number) => Number(this._env[key] || def)
  set = (key: string, value: string) => {
    this._env[key] = value
  }
  notImplemented = () => {
    throw new Error(`This feature is not available in ${this.NAME} environment`)
  }

  // files
  readFile = (path: string): Promise<Uint8Array> => this.notImplemented()
  readTextFile = async (path: string): Promise<string> => new TextDecoder().decode(await this.readFile(path))
  writeFile = (path: string, data: Uint8Array): Promise<void> => this.notImplemented()
  writeTextFile = async (path: string, data: string) => await this.writeFile(path, new TextEncoder().encode(data))
  remove = (path: string): Promise<void> => this.notImplemented()
  realPath = (path: string): Promise<string> => this.notImplemented()
  stat = (path: string): Promise<{ size: number }> => this.notImplemented()
  statSync = (path: string): { size: number } => this.notImplemented()
  tempFile = (): Promise<string> => this.notImplemented()
  writeStdout = (p: Uint8Array) => console.log(new TextDecoder().decode(p))
  homedir = (): string => '/home'

  //
  sha256 = (data: Uint8Array): Uint8Array => data

  // storage
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

  disk_get = async (table: string, key: string): Promise<any | undefined> => {
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

  //
  OSX = this.PLATFORM === 'darwin'
  WINDOWS = this.PLATFORM === 'win32'
  CACHE_DIR = `${this.get('CACHE_DIR') || this.get('XDG_CACHE_HOME') || this.PLATFORM === 'darwin' ? `${this.homedir()}/Library/Caches` : `${this.homedir()}/.cache`}/denograd`
  CACHE_DB = this.get('CACHEDB') || `${this.CACHE_DIR}/denograd.db`
  CI = !!this.get_num('CI')
  DEBUG = this.get_num('DEBUG', 0)
  IMAGE = this.get_num('IMAGE', 0)
  BEAM = this.get_num('BEAM', 0)
  NOOPT = this.get_num('NOOPT', 0)
  JIT = this.get_num('JIT', 1)
  WINO = this.get_num('WINO', 0)
  CAPTURING = this.get_num('CAPTURING', 1)
  TRACEMETA = this.get_num('TRACEMETA', 1)
  USE_TC = this.get_num('TC', 1)
  TC_OPT = this.get_num('TC_OPT', 0)
  AMX = this.get_num('AMX', 0)
  TRANSCENDENTAL = this.get_num('TRANSCENDENTAL', 1)
  FUSE_ARANGE = this.get_num('FUSE_ARANGE', 0)
  FUSE_CONV_BW = this.get_num('FUSE_CONV_BW', 0)
  SPLIT_REDUCEOP = this.get_num('SPLIT_REDUCEOP', 1)
  NO_MEMORY_PLANNER = this.get_num('NO_MEMORY_PLANNER', 0)
  RING = this.get_num('RING', 1)
  PICKLE_BUFFERS = this.get_num('PICKLE_BUFFERS', 1)
  PROFILE = this.get('PROFILE', this.get('VIZ'))
  LRU = this.get_num('LRU', 1)
  CACHELEVEL = this.get_num('CACHELEVEL', 2)
  CAPTURE_PROCESS_REPLAY = this.get('RUN_PROCESS_REPLAY') || this.get('CAPTURE_PROCESS_REPLAY')
}
