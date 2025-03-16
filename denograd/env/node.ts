import process from 'node:process'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { DatabaseSync } from 'node:sqlite'
import { WebEnv } from './index.ts'
import { JS } from '../runtime/ops_js.ts'
import { WASM } from '../runtime/ops_wasm.ts'
import { CLOUD } from '../runtime/ops_cloud.ts'
import { random_id, string_to_bytes } from '../helpers.ts'
import { DAWN } from '../runtime/ops_dawn.ts'
import fs from 'node:fs/promises'
import { statSync } from 'node:fs'
import path from 'node:path'

export class NodeEnv extends WebEnv {
  override NAME = 'node'
  override CPU_DEVICE = 'CLANG'
  override PLATFORM = process.platform
  override DEVICES = { DAWN, WASM, JS, CLOUD }
  override readFile = async (path:string)=>new Uint8Array((await fs.readFile(path)))
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

  override sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest() as Uint8Array

  private db?: DatabaseSync
  private tables: string[] = []
  private db_name = (table: string) => `${table}_${this.DB_VERSION}`
  private get_db = async () => {
    if (this.db) return this.db
    await this.mkdir(this.CACHE_DIR)
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
