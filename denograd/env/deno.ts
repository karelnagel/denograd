import { Environment } from './abstract.ts'
import process from 'node:process'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { gunzipSync } from 'node:zlib'

export class DenoEnv extends Environment {
  platform = process.platform
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
  sha256 = (data: string | Uint8Array) => createHash('sha256').update(data).digest()
  gunzipSync = (input: ArrayBuffer) => gunzipSync(input)
}
