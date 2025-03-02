import process from 'node:process'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { DatabaseSync } from 'node:sqlite'
import { WebEnv } from './index.ts'
import { DISK } from '../runtime/ops_disk_deno.ts'
import { JS } from '../runtime/ops_js.ts'
import { WASM } from '../runtime/ops_wasm.ts'
import { WEBGPU } from '../runtime/ops_webgpu.ts'
import { CLANG } from '../runtime/ops_clang_deno.ts'

export class DenoEnv extends WebEnv {
  override NAME = 'DENO'
  override PLATFORM = process.platform
  override DEVICES = { CLANG, WEBGPU, WASM, JS, DISK }
  override readFile = Deno.readFile
  override writeFile = Deno.writeFile
  override remove = Deno.remove
  override realPath = Deno.realPath
  override stat = Deno.stat
  override statSync = Deno.statSync
  override writeStdout = (p: Uint8Array) => void Deno.stdout.writeSync(p)
  override tempFile = Deno.makeTempFile
  override homedir = os.homedir

  override sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest() as Uint8Array

  private db?: DatabaseSync
  private tables: string[] = []
  private get_db = async () => {
    if (this.db) return this.db
    await Deno.mkdir(this.CACHE_DIR, { recursive: true })
    this.db = new DatabaseSync(this.CACHE_DB)
    return this.db
  }
  override disk_get = async (table: string, key: string) => {
    const db = await this.get_db()
    try {
      const row = db.prepare(`SELECT * FROM ${table}_${this.DB_VERSION} WHERE key = ?`).get(key) as { value: any | undefined }
      return row?.value
    } catch (e) {
      return undefined
    }
  }
  override disk_put = async (table: string, key: string, value: any) => {
    const valueType = typeof value === 'string' ? 'TEXT' : value instanceof Uint8Array ? 'BLOB' : undefined
    if (!valueType) throw new Error(`Invalid value type ${valueType}`)

    const db = await this.get_db()
    const name = `${table}_${this.DB_VERSION}`
    if (!this.tables.includes(name)) {
      db.exec(`CREATE TABLE IF NOT EXISTS ${name} (key TEXT PRIMARY KEY, value ${valueType});`)
      this.tables.push(name)
    }

    db.prepare(`INSERT INTO ${name} (key, value) VALUES (?, ?);`).run(key, value)
  }
}
