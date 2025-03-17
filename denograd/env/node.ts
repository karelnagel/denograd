import process from 'node:process'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { type Dlopen, WebEnv } from './index.ts'
import { JS } from '../runtime/ops_js.ts'
import { CLOUD } from '../runtime/ops_cloud.ts'
import { random_id, string_to_bytes } from '../helpers.ts'
import fs from 'node:fs/promises'
import { statSync } from 'node:fs'
import path from 'node:path'
import type { DatabaseSync } from 'node:sqlite'
import { CLANG } from '../runtime/ops_clang.ts'
import { exec } from 'node:child_process'

import { Buffer } from 'node:buffer'

const ffiType = (type: Deno.NativeType) => {
  if (type === 'isize') return 'int64'
  if (type === 'usize') return 'uint64'
  if (typeof type === 'object' || type === 'buffer') return 'pointer'
  const typeMap = {
    'void': 'void',
    'i8': 'int8',
    'u8': 'uint8',
    'i16': 'int16',
    'u16': 'uint16',
    'i32': 'int32',
    'u32': 'uint32',
    'f32': 'float',
    'f64': 'double',
  }
  return typeMap[type as keyof typeof typeMap] || 'pointer'
}

export class NodeEnv extends WebEnv {
  override NAME = 'node'
  override CPU_DEVICE = 'JS'
  override PLATFORM = process.platform
  override DEVICES = { CLANG, JS, CLOUD }
  override readFile = async (path: string) => new Uint8Array(await fs.readFile(path))
  override writeFile = fs.writeFile
  override remove = fs.unlink
  override realPath = (...paths: string[]) => paths[0].startsWith('/') ? path.resolve(process.cwd(), ...paths) : path.resolve(...paths)
  override stat = fs.stat
  override statSync = statSync
  override writeStdout = (p: string) => process.stdout.write(string_to_bytes(p))
  override tempFile = async () => `/tmp/dg_tmp_${random_id()}`
  override homedir = os.homedir
  override mkdir = async (path: string) => void await fs.mkdir(path, { recursive: true })
  override args = () => process.argv.slice(2)
  override machine = () => os.machine()
  override exit = (code: number) => process.exit(code)
  override exec = async (cmd: string) => {
    return await new Promise<string>((resolve, reject) => {
      exec(cmd, (error, stdout, stderr) => {
        if (error) reject(stderr)
        else resolve(stdout)
      })
    })
  }
  private _ref: any
  override dlopen: Dlopen = async (file, args) => {
    const ffi = await import('ffi-napi')
    this._ref = await import('ref-napi')
    const symbols = Object.fromEntries(
      Object.entries(args).map(([name, { parameters, result }]: any) => [
        name,
        [ffiType(result), parameters.map(ffiType)],
      ]),
    )
    return {
      symbols: ffi.Library(file, symbols),
      close: () => {},
    }
  }

  override ptr = (buffer: ArrayBuffer) => this._ref.ref(Buffer.from(buffer))

  override sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest() as Uint8Array

  private db?: DatabaseSync
  private tables: string[] = []
  private db_name = (table: string) => `${table}_${this.DB_VERSION}`
  private get_db = async (): Promise<DatabaseSync> => {
    if (this.db) return this.db
    await this.mkdir(this.CACHE_DIR)
    const { DatabaseSync } = await import('node:sqlite')
    this.db = new DatabaseSync(this.CACHE_DB)
    return this.db
  }
  override disk_get = async (table: string, key: string) => {
    const db = await this.get_db()
    try {
      const row = db.prepare(`SELECT * FROM "${this.db_name(table)}" WHERE key = ?`).get(key) as { value: any | undefined }
      return row?.value
    } catch (e) {
      return undefined
    }
  }
  override disk_put = async (table: string, key: string, value: any) => {
    const valueType = typeof value === 'string' ? 'TEXT' : value instanceof Uint8Array ? 'BLOB' : undefined
    if (!valueType) throw new Error(`Invalid value type ${valueType}`)
    try {
      const db = await this.get_db()
      if (!this.tables.includes(this.db_name(table))) {
        db.exec(`CREATE TABLE IF NOT EXISTS "${this.db_name(table)}"  (key TEXT PRIMARY KEY, value ${valueType});`)
        this.tables.push(this.db_name(table))
      }

      db.prepare(`INSERT INTO "${this.db_name(table)}" (key, value) VALUES (?, ?);`).run(key, value)
    } catch (e) {
      console.error(e)
    }
  }
}
