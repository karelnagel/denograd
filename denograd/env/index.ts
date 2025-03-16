// deno-lint-ignore-file no-process-global
import { memsize_to_str } from '../helpers.ts'
import type { Compiled } from '../runtime/allocator.ts'
import { Sha256 } from '../sha256.js'
import { Tqdm, type TqdmOnProgress } from '../tqdm.ts'
export type Stat = { size: number; isFile: boolean }
// deno-fmt-ignore
export class WebEnv {
  NAME = 'web'
  PLATFORM = 'web'
  CPU_DEVICE: string = 'JS'
  DB_VERSION = 1
  DEVICES:Record<string,typeof Compiled> = {}
  // @ts-ignore import.meta.env
  _env!: Record<string, string | number> = (typeof import.meta?.env !== 'undefined' ? import.meta.env : (typeof process !== 'undefined' && process.env) ? process.env : {}) || {}

  // env
  get = (key: string, def?: string) => this._env[key] !== undefined ? this._env[key].toString() : def
  get_num = (key: string, def?: number) => Number(this._env[key] || def)
  set = (key: string, value: string) => {
    this._env[key] = value
  }
  notImplemented = () => {
    throw new Error(`This feature is not available in ${this.NAME} environment`)
  }

  // files
  readFile = async (path: string): Promise<Uint8Array> => await this.disk_get("fs", path)
  readTextFile = async (path: string): Promise<string> => new TextDecoder().decode(await this.readFile(path))
  writeFile = async (path: string, data: Uint8Array): Promise<void> => await this.disk_put("fs", path, data)
  writeTextFile = async (path: string, data: string) => await this.writeFile(path, new TextEncoder().encode(data))
  remove = async (path: string): Promise<void> => {
    const db = await this._open_db()
    const transaction = db.transaction(['denograd'], 'readwrite')
    const store = transaction.objectStore('denograd')
    const request = store.delete(`fs_${path}`)

    await new Promise<void>((resolve, reject) => {
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }
  realPath = async (path: string): Promise<string> =>path
  stat = async (path: string): Promise<Stat> => {
    const res = await this.disk_get("fs", path)
    if (!res || !(res instanceof Uint8Array)) throw new Error(`No entry for ${path}`)
    return { isFile:!!res, size:res.length }
  }
  statSync = (path: string): Stat => this.notImplemented()
  tempFile = async (): Promise<string> => `/tmp/${(Math.random() * 100000000).toFixed(0)}`
  writeStdout = (p:string) => console.log(p+'\u200B')
  homedir = () => '/home'
  gunzip = async (res:Response):Promise<ArrayBuffer> => await new Response(res.body!.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()
  mkdir = async (path:string): Promise<void> => {}
  args = (): string[] => (window as any).args || []
  machine = () => "browser"

  //
  sha256 = (data: Uint8Array): Uint8Array => new Uint8Array(new Sha256().update(data)!.arrayBuffer())

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

  disk_put = async (table: string, key: string, value: string | Uint8Array) => {
    const db = await this._open_db()
    const transaction = db.transaction(['denograd'], 'readwrite')
    const store = transaction.objectStore('denograd')
    const request = store.put({ key: `${table}_${key}`, value })

    await new Promise<void>((resolve, reject) => {
      request.onsuccess = () => resolve()
      request.onerror = () => reject(request.error)
    })
  }
  fetchSave = async (url: string, path: string, dir?: string, onProgress?: TqdmOnProgress) => {
    if (dir){
      path = `${dir}/${path}`
      await this.mkdir(dir)
    }
    if (await this.stat(path).then((x) => x.isFile).catch(() => false)) {
      console.log(`File ${path} already exists, skipping`)
      return path
    }
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Error ${res.status}`)
    const reader = res.body?.getReader()
    if (!reader) throw new Error('Response body not readable!')
    let size = Number(res.headers.get('content-length')), i = 0
    const data = new Uint8Array(size)
    const  t = new Tqdm(size, { onProgress, label: `Downloading ${path}`, format: memsize_to_str })
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      if (value) {
        data.set(value, i)
        i += value.length
        t.render(i)
      }
    }
    this.writeStdout("\n")
    await this.writeFile(path, new Uint8Array(data))
    return path
  }
  get DEVICE() { return this.get('DEVICE') || this.get("D") }
  set DEVICE(val) { this._env.DEVICE = val! }
  get OSX(){ return this.PLATFORM === 'darwin'}
  set OSX(val) { this._env.PLATFORM = val ? 'darwin' : this.PLATFORM }
  get WINDOWS(){ return this.PLATFORM === 'win32'}
  set WINDOWS(val) { this._env.PLATFORM = val ? 'win32' : this.PLATFORM }
  get CACHE_DIR (){ return `${this.get('CACHE_DIR') || this.get('XDG_CACHE_HOME') || (this.OSX ? `${this.homedir()}/Library/Caches` : `${this.homedir()}/.cache`)}/denograd` }
  set CACHE_DIR(val) { this._env.CACHE_DIR = val }
  get CACHE_DB (){return this.get('CACHE_DB') || `${this.CACHE_DIR}/denograd.db` }
  set CACHE_DB(val) { this._env.CACHE_DB = val }
  get CI (){ return !!this.get_num('CI') }
  set CI(val) { this._env.CI = val ? 1 : 0 }
  get DEBUG (){ return this.get_num('DEBUG', 0) }
  set DEBUG(val) { this._env.DEBUG = val }
  get IMAGE (){ return this.get_num('IMAGE', 0) }
  set IMAGE(val) { this._env.IMAGE = val }
  get BEAM (){ return this.get_num('BEAM', 0) }
  set BEAM(val) { this._env.BEAM = val }
  get NOOPT (){ return this.get_num('NOOPT', 0) }
  set NOOPT(val) { this._env.NOOPT = val }
  get JIT (){ return this.get_num('JIT', 1) }
  set JIT(val) { this._env.JIT = val }
  get WINO (){ return this.get_num('WINO', 0) }
  set WINO(val) { this._env.WINO = val }
  get CAPTURING (){ return this.get_num('CAPTURING', 1) }
  set CAPTURING(val) { this._env.CAPTURING = val }
  get TRACEMETA (){ return this.get_num('TRACEMETA', 1) }
  set TRACEMETA(val) { this._env.TRACEMETA = val }
  get USE_TC (){ return this.get_num('TC', 1) }
  set USE_TC(val) { this._env.TC = val }
  get TC_OPT (){ return this.get_num('TC_OPT', 0) }
  set TC_OPT(val) { this._env.TC_OPT = val }
  get AMX (){ return this.get_num('AMX', 0) }
  set AMX(val) { this._env.AMX = val }
  get TRANSCENDENTAL (){ return this.get_num('TRANSCENDENTAL', 1) }
  set TRANSCENDENTAL(val) { this._env.TRANSCENDENTAL = val }
  get FUSE_ARANGE (){ return this.get_num('FUSE_ARANGE', 0) }
  set FUSE_ARANGE(val) { this._env.FUSE_ARANGE = val }
  get FUSE_CONV_BW (){ return this.get_num('FUSE_CONV_BW', 0) }
  set FUSE_CONV_BW(val) { this._env.FUSE_CONV_BW = val }
  get SPLIT_REDUCEOP (){ return this.get_num('SPLIT_REDUCEOP', 1) }
  set SPLIT_REDUCEOP(val) { this._env.SPLIT_REDUCEOP = val }
  get NO_MEMORY_PLANNER (){ return this.get_num('NO_MEMORY_PLANNER', 0) }
  set NO_MEMORY_PLANNER(val) { this._env.NO_MEMORY_PLANNER = val }
  get RING (){ return this.get_num('RING', 1) }
  set RING(val) { this._env.RING = val }
  get PICKLE_BUFFERS (){ return this.get_num('PICKLE_BUFFERS', 1) }
  set PICKLE_BUFFERS(val) { this._env.PICKLE_BUFFERS = val }
  get PROFILE (){ return this.get('PROFILE', this.get('VIZ')) }
  set PROFILE(val) { this._env.PROFILE = val! }
  get LRU (){ return this.get_num('LRU', 1) }
  set LRU(val) { this._env.LRU = val }
  get CACHELEVEL (){ return this.get_num('CACHELEVEL', 2) }
  set CACHELEVEL(val) { this._env.CACHELEVEL = val }
  get CAPTURE_PROCESS_REPLAY (){ return this.get('RUN_PROCESS_REPLAY') || this.get('CAPTURE_PROCESS_REPLAY') }
  set CAPTURE_PROCESS_REPLAY(val) { this._env.CAPTURE_PROCESS_REPLAY = val! }
}

export let env = new WebEnv()
export const setRuntime = (e: WebEnv) => {
  env = e
  if (env.DEBUG === 1) console.log(`Using env ${env.NAME}`)
}

export const withEnv = <Res>(overrides: Record<string, string | number>, fn: () => Res): Res => {
  const old = env._env
  env._env = { ...env._env, ...overrides as any }
  const res = fn()
  env._env = old
  return res
}

export const withEnvAsync = async <Res>(overrides: Record<string, string | number>, fn: () => Promise<Res>): Promise<Res> => {
  const old = env._env
  env._env = { ...env._env, ...overrides as any }
  const res = await fn()
  env._env = old
  return res
}

await withEnv({ AMX: 1 }, async () => {})
