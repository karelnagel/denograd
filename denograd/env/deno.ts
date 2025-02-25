import { Environment } from './abstract.ts'
import process from 'node:process'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { DatabaseSync } from 'node:sqlite'

export class DenoEnv extends Environment {
  NAME = 'DENO'
  PLATFORM = process.platform
  env = Deno.env
  readFileSync = (path: string) => Deno.readFileSync(path)
  writeFileSync = (path: string, data: Uint8Array) => Deno.writeFileSync(path, data)
  removeSync = (path: string) => Deno.removeSync(path)
  realPathSync = (path: string) => Deno.realPathSync(path)
  statSync = (path: string) => Deno.statSync(path)
  writeStdout = async (p: Uint8Array) => await Deno.stdout.write(p)
  makeTempFileSync = () => Deno.makeTempFileSync()
  execSync = (command: string, { args }: { args?: string[] } = {}) => new Deno.Command(command, { args }).outputSync()
  tmpdir = () => os.tmpdir()
  homedir = () => os.homedir()
  sha256 = (data: string | Uint8Array) => createHash('sha256').update(data).digest() as Uint8Array

  cache_dir = `${this.env.get('XDG_CACHE_HOME') || this.PLATFORM === 'darwin' ? `${os.homedir()}/Library/Caches` : `${os.homedir()}/.cache`}/denograd`
  cache_db = this.env.get('CACHEDB') || `${this.cache_dir}/denograd.db`

  private db?: DatabaseSync
  private tables: string[] = []
  private get_db = async () => {
    if (this.db) return this.db
    await Deno.mkdir(this.cache_dir, { recursive: true })
    this.db = new DatabaseSync(this.cache_db)
    return this.db
  }
  disk_get = async (table: string, key: string) => {
    const db = await this.get_db()
    try {
      const row = db.prepare(`SELECT * FROM ${table}_${this.DB_VERSION} WHERE key = ?`).get(key) as { value: any | undefined }
      return row?.value
    } catch (e) {
      return undefined
    }
  }
  disk_put = async (table: string, key: string, value: any) => {
    const valueType = typeof value === 'string' ? 'TEXT' : value instanceof Uint8Array ? 'BLOB' : undefined
    if (!valueType) throw new Error(`Invalid value type ${valueType}`)

    const db = await this.get_db(), name = `${table}_${this.DB_VERSION}`
    if (!this.tables.includes(name)) {
      db.exec(`CREATE TABLE IF NOT EXISTS ${name} (key TEXT PRIMARY KEY, value ${valueType});`)
      this.tables.push(name)
    }

    db.prepare(`INSERT INTO ${name} (key, value) VALUES (?, ?);`).run(key, value)
  }
}
