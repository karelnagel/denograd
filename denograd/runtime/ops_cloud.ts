import { Buffer } from 'node:buffer'
import { BufferSpec, Compiler, type ProgramCallArgs } from '../device.ts'
import { env } from '../env/index.ts'
import { bytes_to_string, random_id, string_to_bytes } from '../helpers.ts'
import type { MemoryView } from '../memoryview.ts'
import { ClangRenderer } from '../renderer/cstyle.ts'
import { WATRenderer } from '../renderer/wat.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import { Allocator, Compiled, Program } from './allocator.ts'
import { createHash } from 'node:crypto'

// ***** API *****
export class CloudRequest {}

export class BufferAlloc extends CloudRequest {
  constructor(public buffer_num: number, public size: number, public options: BufferSpec) {
    super()
  }
}
export class BufferFree extends CloudRequest {
  constructor(public buffer_num: number) {
    super()
  }
}
export class CopyIn extends CloudRequest {
  constructor(public buffer_num: number, public datahash: string) {
    super()
  }
}
export class CopyOut extends CloudRequest {
  constructor(public buffer_num: number) {
    super()
  }
}
export class ProgramAlloc extends CloudRequest {
  constructor(public name: string, public datahash: string) {
    super()
  }
}
export class ProgramFree extends CloudRequest {
  constructor(public name: string, public datahash: string) {
    super()
  }
}
export class ProgramExec extends CloudRequest {
  constructor(public name: string, public datahash: string, public bufs: number[], public vals: number[], public global_size?: number[], public local_size?: number[], public wait?: boolean) {
    super()
  }
}
const CLASSES = [BufferSpec, BufferAlloc, BufferFree, CopyIn, CopyOut, ProgramAlloc, ProgramFree, ProgramExec]

export const serialize = (x: any): string | undefined => {
  if (typeof x === 'boolean') return x ? 'True' : 'False'
  if (typeof x === 'undefined') return 'None'
  if (typeof x === 'bigint' || typeof x === 'number') return x.toString()
  if (typeof x === 'string') return `'${x}'`
  if (Array.isArray(x)) return `[${x.map(serialize).join(', ')}]`
  if (typeof x === 'function') return undefined
  if (!CLASSES.some((c) => x instanceof c)) throw new Error(`Can't serialize ${x}`)

  const args = Object.entries(x).map(([k, v]) => [k, serialize(v)]).filter(([k, v]) => k !== 'key' && v !== undefined)
  return `${x.constructor.name}(${args.map(([k, v]) => `${k}=${v}`).join(', ')})`
}

const split_commas = (string: string): string[] => {
  let brackets = 0, current = '', res: string[] = []
  for (const char of string) {
    if (['(', '['].includes(char)) brackets++
    else if ([')', ']'].includes(char)) brackets--

    if (char === ',' && brackets === 0) res.push(current.trim()), current = ''
    else current += char
  }
  return [...res, current.trim()].filter(Boolean)
}
export const deserialize = (x: string): any => {
  if (x === 'True') return true
  if (x === 'False') return false
  if (x === 'None') return undefined
  if (!isNaN(Number(x)) && x.trim() !== '') return Number(x) // TODO: bigint
  if (x.startsWith("'") && x.endsWith("'")) return x.slice(1, -1)
  if (x.startsWith('[') && x.endsWith(']')) return split_commas(x.slice(1, -1)).map(deserialize)

  const [name, argstr] = x.slice(0, -1).split('(')
  const cls = CLASSES.find((c) => c.name === name)
  if (cls) {
    const args = split_commas(argstr).map((x) => x.split('=')).map(([k, v]) => deserialize(v))
    return new (cls as any)(...args)
  }
  return x
}

export class BatchRequest {
  _q: CloudRequest[] = []
  _h: Record<string, Uint8Array> = {}
  h = (d: Uint8Array): string => {
    const binhash = createHash('sha256').update(d).digest()
    const datahash = binhash.toString('hex')
    this._h[datahash] = new Uint8Array([...binhash, ...new Uint8Array(new BigUint64Array([BigInt(d.length)]).buffer), ...d])
    return datahash
  }
  q = (x: CloudRequest) => this._q.push(x)
  serialize = (): Uint8Array => {
    this.h(string_to_bytes(serialize(this._q)!))
    return new Uint8Array(Object.values(this._h).flatMap((x) => [...x]))
  }
  static deserialize = (dat: Uint8Array): BatchRequest => {
    let res = new BatchRequest(), ptr = 0
    while (ptr < dat.length) {
      const datahash = Buffer.from(dat.slice(ptr, ptr + 0x20)).toString('hex')
      const datalen = Number(new BigUint64Array(dat.slice(ptr + 0x20, ptr + 0x28).buffer)[0])
      res._h[datahash] = dat.slice(ptr + 0x28, ptr + 0x28 + datalen)
      ptr += 0x28 + datalen
      res._q = deserialize(bytes_to_string(res._h[datahash]))
    }
    return res
  }
}

// ***** frontend *****
class CloudAllocator extends Allocator<number> {
  constructor(public device: CLOUD) {
    super()
  }
  // TODO: ideally we shouldn't have to deal with images here
  _alloc = (size: number, options: BufferSpec): number => {
    this.device.buffer_num += 1
    this.device.req.q(new BufferAlloc(this.device.buffer_num, size, options))
    return this.device.buffer_num
  }
  // TODO: options should not be here in any Allocator
  _free = (opaque: number, options: BufferSpec) => this.device.req.q(new BufferFree(opaque))
  _copyin = (dest: number, src: MemoryView) => this.device.req.q(new CopyIn(dest, this.device.req.h(src.bytes)))
  _copyout = async (dest: MemoryView, src: number) => {
    this.device.req.q(new CopyOut(src))
    const resp = await this.device.batch_submit().then((x) => x.arrayBuffer())
    if (resp.byteLength !== dest.length) throw new Error(`buffer length mismatch ${resp.byteLength} != ${dest.length}`)
    dest.set(resp)
  }
}
const getCloudProgram = (dev: CLOUD) => {
  return class CloudProgram extends Program {
    datahash!: string
    dev!: CLOUD
    static override init = async (name: string, lib: Uint8Array) => {
      const res = new CloudProgram(name, lib)
      res.dev = dev
      res.datahash = res.dev.req.h(lib)
      res.dev.req.q(new ProgramAlloc(res.name, res.datahash))
      return res
    }
    del = () => this.dev.req.q(new ProgramFree(this.name, this.datahash))

    override call = async (bufs: number[], { global_size, local_size, vals = [] }: ProgramCallArgs, wait = false) => {
      this.dev.req.q(new ProgramExec(this.name, this.datahash, bufs, vals, global_size, local_size, wait))
      if (wait) return Number(await this.dev.batch_submit())
    }
  }
}

const RENDERERS = [ClangRenderer, WGSLRenderer, WATRenderer]

export class CLOUD extends Compiled {
  host = env.DEVICE?.startsWith('CLOUD:') ? env.DEVICE.replace('CLOUD:', '') : env.get('HOST', 'http://127.0.0.1:6667')
  // state for the connection
  session = random_id()
  buffer_num = 0
  req = new BatchRequest()
  _init = false
  constructor(device: string) {
    super(device, undefined, undefined, new Compiler())
    this.allocator = new CloudAllocator(this)
    this.runtime = getCloudProgram(this)

    if (env.DEBUG >= 1) console.log(`cloud with host ${this.host}`)
  }
  override init = async () => {
    // TODO replace _init with renderer
    if (this.renderer) return
    // TODO: how to we have BEAM be cached on the backend? this should just send a specification of the compute. rethink what goes in Renderer
    const clouddev = await this.send('GET', 'renderer').then((x) => x.json())
    const renderer = RENDERERS.find((x) => x.name === clouddev[1])!
    if (!renderer) throw new Error(`Invalid renderer ${clouddev}`)
    this.renderer = new renderer()
    if (env.DEBUG >= 1) console.log(`remote has device ${clouddev}`)
  }
  del = async () => {
    // TODO: this is never being called
    // TODO: should close the whole session
    await this.batch_submit()
  }

  batch_submit = async () => {
    const data = this.req.serialize()
    const ret = await this.send('POST', 'batch', data)
    this.req = new BatchRequest()
    return ret
  }

  send = async (method: string, path: string, data?: Uint8Array) => {
    // TODO: retry logic
    const res = await fetch(this.host + '/' + path, { method, body: data, headers: { 'Cookie': `session=${this.session}` }, keepalive: true })
    if (res.status !== 200) throw new Error(`failed on ${method} ${path} ${await res.text()}`)
    return res
  }
}
