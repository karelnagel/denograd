import process from 'node:process'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { Database } from 'bun:sqlite'
import { mkdir, realpath, stat, unlink } from 'fs/promises'
import { statSync } from 'fs'
import { WebEnv } from './index.ts'
import { CLANG } from '../runtime/ops_clang_bun.ts'
import { JS } from '../runtime/ops_js.ts'
import { WASM } from '../runtime/ops_wasm.ts'

export class BunEnv extends WebEnv {
  override NAME = 'BUN'
  override PLATFORM = process.platform
  override DEVICES = { CLANG, WASM, JS }

  override readFile = async (path: string) => new Uint8Array(await Bun.file(path).arrayBuffer())
  override writeFile = (path: string, data: Uint8Array) => Bun.write(path, data)
  override remove = (path: string) => unlink(path)
  override realPath = (path: string) => realpath(path)
  override stat = (path: string) => stat(path)
  override statSync = statSync
  override writeStdout = (p: Uint8Array) => void Bun.stdout.write(p)
  override tempFile = async () => {
    const path = `${os.tmpdir()}/tmp-${Date.now()}-${Math.random().toString(36).substring(2)}`
    await Bun.write(path, '')
    return path
  }
  override homedir = os.homedir
  override gunzip = async (res: Response) => Bun.gunzipSync(new Uint8Array(await res.arrayBuffer()))
  override sha256 = (data: Uint8Array) => createHash('sha256').update(data).digest() as Uint8Array

  private db?: Database
  private tables: string[] = []
  private get_db = async () => {
    if (this.db) return this.db
    await mkdir(this.CACHE_DIR, { recursive: true })
    this.db = new Database(this.CACHE_DB)
    return this.db
  }
  override disk_get = async (table: string, key: string) => {
    const db = await this.get_db()
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

    const db = await this.get_db()
    const name = `${table}_${this.DB_VERSION}`
    if (!this.tables.includes(name)) {
      db.exec(`CREATE TABLE IF NOT EXISTS ${name} (key TEXT PRIMARY KEY, value ${valueType})`)
      this.tables.push(name)
    }
    db.prepare(`INSERT INTO ${name} (key, value) VALUES (?, ?)`).run(key, value)
  }
}
