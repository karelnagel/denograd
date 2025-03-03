// deno-lint-ignore-file no-process-global
import type { Compiled } from '../runtime/allocator.ts'

// deno-fmt-ignore
export class WebEnv {
  NAME = 'WEB'
  PLATFORM = 'web'
  CPU_DEVICE: string = 'JS'
  DB_VERSION = 1
  DEVICES:Record<string,typeof Compiled> = {}
  // @ts-ignore import.meta.env
  _env!: Record<string, string> = (typeof import.meta?.env !== 'undefined' ? import.meta.env : (typeof process !== 'undefined' && process.env) ? process.env : {}) || {}

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
  homedir = () => '/home'
  gunzip = async (res:Response):Promise<ArrayBuffer> => await new Response(res.body!.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()

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
  get DEVICE() { return this.get('DEVICE') || this.get("D") }
  get OSX(){ return this.PLATFORM === 'darwin'}
  get WINDOWS(){ return this.PLATFORM === 'win32'}
  get CACHE_DIR (){ return `${this.get('CACHE_DIR') || this.get('XDG_CACHE_HOME') || (this.OSX ? `${this.homedir()}/Library/Caches` : `${this.homedir()}/.cache`)}/denograd` }
  get CACHE_DB (){return this.get('CACHE_DB') || `${this.CACHE_DIR}/denograd.db` }
  get CI (){ return !!this.get_num('CI') }
  get DEBUG (){ return this.get_num('DEBUG', 0) }
  get IMAGE (){ return this.get_num('IMAGE', 0) }
  get BEAM (){ return this.get_num('BEAM', 0) }
  get NOOPT (){ return this.get_num('NOOPT', 0) }
  get JIT (){ return this.get_num('JIT', 1) }
  get WINO (){ return this.get_num('WINO', 0) }
  get CAPTURING (){ return this.get_num('CAPTURING', 1) }
  get TRACEMETA (){ return this.get_num('TRACEMETA', 1) }
  get USE_TC (){ return this.get_num('TC', 1) }
  get TC_OPT (){ return this.get_num('TC_OPT', 0) }
  get AMX (){ return this.get_num('AMX', 0) }
  get TRANSCENDENTAL (){ return this.get_num('TRANSCENDENTAL', 1) }
  get FUSE_ARANGE (){ return this.get_num('FUSE_ARANGE', 0) }
  get FUSE_CONV_BW (){ return this.get_num('FUSE_CONV_BW', 0) }
  get SPLIT_REDUCEOP (){ return this.get_num('SPLIT_REDUCEOP', 1) }
  get NO_MEMORY_PLANNER (){ return this.get_num('NO_MEMORY_PLANNER', 0) }
  get RING (){ return this.get_num('RING', 1) }
  get PICKLE_BUFFERS (){ return this.get_num('PICKLE_BUFFERS', 1) }
  get PROFILE (){ return this.get('PROFILE', this.get('VIZ')) }
  get LRU (){ return this.get_num('LRU', 1) }
  get CACHELEVEL (){ return this.get_num('CACHELEVEL', 2) }
  get CAPTURE_PROCESS_REPLAY (){ return this.get('RUN_PROCESS_REPLAY') || this.get('CAPTURE_PROCESS_REPLAY') }
}

export let env = new WebEnv()
export const setRuntime = (e: WebEnv) => env = e
if (env.get('DEBUG')) console.log(`Using env ${env.NAME}`)
