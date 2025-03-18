import { Database } from 'bun:sqlite'
import { CLANG } from '../runtime/ops_clang.ts'
import { JS } from '../runtime/ops_js.ts'
import { WASM } from '../runtime/ops_wasm.ts'
import { CLOUD } from '../runtime/ops_cloud.ts'
import { NodeEnv } from './node.ts'
import { CString, dlopen, FFIType, JSCallback, ptr, toArrayBuffer } from 'bun:ffi'
import { DISK } from '../runtime/ops_disk.ts'
import { DAWN } from '../runtime/ops_dawn.ts'
import type { Dlopen, FFICallback } from './index.ts'

const ffiType = (type: Deno.NativeResultType): FFIType => {
  if (type === 'isize') return FFIType.i64
  if (type === 'usize') return FFIType.u64
  if (typeof type === 'object') return FFIType.buffer
  return FFIType[type]
}

export class BunEnv extends NodeEnv {
  override NAME = 'bun'
  override DEVICES = { CLANG, DAWN, WASM, JS, CLOUD, DISK }
  override args = () => Bun.argv.slice(2)
  override dlopen: Dlopen = (file, args) => {
    return dlopen(
      file,
      Object.fromEntries(
        Object.entries(args).map(([name, args]: any) => [
          name,
          { args: args.parameters.map((x: any) => ffiType(x)), result: ffiType(args.result) },
        ]),
      ),
    ) as any
  }
  override ptr = (buffer: ArrayBuffer) => ptr(buffer)
  override ptrToU64 = (ptr: any) => BigInt(ptr)
  override u64ToPtr = (u64: any) => Number(u64)
  override getCString = (ptr: any) => new CString(ptr).toString()
  override getArrayBuffer = (ptr: any, byteLength: number, offset?: number) => toArrayBuffer(ptr, offset, byteLength)
  override callback: FFICallback = (x, cb) => new JSCallback(cb, { args: x.parameters.map(ffiType), returns: ffiType(x.result) }).ptr

  override gunzip = async (res: Response) => Bun.gunzipSync(new Uint8Array(await res.arrayBuffer())).buffer as ArrayBuffer

  private bunDb?: Database
  private _tables: string[] = []
  private _get_db = async () => {
    if (this.bunDb) return this.bunDb
    await this.mkdir(this.CACHE_DIR)
    this.bunDb = new Database(this.CACHE_DB)
    return this.bunDb
  }
  override disk_get = async (table: string, key: string) => {
    const db = await this._get_db()
    try {
      const name = `${table}_${this.DB_VERSION}`
      const row = db.prepare(`SELECT value FROM ${name} WHERE key = ?`).get(key) as { value: any } | undefined
      return row?.value
    } catch (e) {
      return undefined
    }
  }
  override disk_put = async (table: string, key: string, value: any) => {
    const valueType = typeof value === 'string' ? 'TEXT' : value instanceof Uint8Array ? 'BLOB' : undefined
    if (!valueType) throw new Error(`Invalid value type ${typeof value}`)

    const db = await this._get_db()
    const name = `${table}_${this.DB_VERSION}`
    if (!this._tables.includes(name)) {
      db.exec(`CREATE TABLE IF NOT EXISTS ${name} (key TEXT PRIMARY KEY, value ${valueType})`)
      this._tables.push(name)
    }
    db.prepare(`INSERT INTO ${name} (key, value) VALUES (?, ?)`).run(key, value)
  }
}
