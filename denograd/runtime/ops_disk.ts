import { Allocator, Compiled } from './allocator.ts'
import { assert } from '../helpers.ts'
import process from 'node:process'
import type { DeviceType } from '../device.ts'
import { MemoryView } from '../memoryview.ts'

export class DiskBuffer {
  constructor(public device: DiskDevice, public size: number, public offset = 0) {
  }
  toString = () => `<DiskBuffer size=${this.size} offset=${this.offset}>`
  _buf = (): MemoryView => {
    if (this.device.mem === undefined) throw new Error("DiskBuffer wasn't opened")
    return new MemoryView(this.device.mem).slice(this.offset, this.offset + this.size)
  }
}
// const [MAP_LOCKED, MAP_POPULATE] = [OSX ? 0 : 0x2000, mmap.MAP_POPULATE || (OSX ? 0 : 0x008000)]

export class DiskAllocator extends Allocator<DiskBuffer> {
  constructor(public dev: DiskDevice) {
    super()
  }
  override _alloc = (size: number, options: any) => {
    this.dev._might_open(size)
    return new DiskBuffer(this.dev, size)
  }
  override _free = (opaque: any, options: any) => this.dev._might_close()
  _as_buffer = (src: DiskBuffer) => src._buf()
  write = (buf: MemoryView, offset: number) => {
    const fo = Deno.openSync(this.dev.filename, { write: true, read: true })
    fo.seekSync(offset, Deno.SeekMode.Start)
    fo.writeSync(buf.toBytes())
    fo.close()
  }
  override _copyin = (dest: DiskBuffer, src: MemoryView) => {
    const buf = dest._buf()
    buf.set(src)
    this.write(buf, dest.offset)
  }
  override _copyout = (dest: MemoryView, src: DiskBuffer) => {
    const buf = src._buf()
    dest.set(buf)
    this.write(buf, src.offset)
  }
  _offset = (buf: DiskBuffer, size: number, offset: number) => new DiskBuffer(buf.device, size, offset)
}

export class DiskDevice extends Compiled {
  static _tried_io_uring_init = false
  size?: number
  count = 0
  mem!: Uint8Array
  constructor(device: DeviceType) {
    super(device, undefined, undefined, undefined, undefined)
    if (!DiskDevice._tried_io_uring_init) this._iouring_setup()
    this.allocator = new DiskAllocator(this)
  }
  get filename() {
    return this.device.slice('DISK:'.length)
  }
  _might_open = (size: number) => {
    this.count += 1
    if (this.size !== undefined && size > this.size) throw new Error(`can't reopen Disk tensor with larger size, opened with ${this.size}, tried to open with ${size}`)
    if (this.size !== undefined) return
    this.size = size

    if (process.platform !== 'win32' && this.filename.startsWith('shm:')) {
      throw new Error('Not implemented')
    } else {
      const fo = Deno.openSync(this.filename, { read: true, write: true, create: true })
      this.mem = new Uint8Array(this.size)
      const stat = fo.statSync()
      if (stat.size === 0) fo.writeSync(this.mem)
      else fo.readSync(this.mem)
      fo.close()
    }
  }
  _might_close = () => {
    this.count -= 1
    if (this.count === 0) {
      this.size = undefined
    }
    throw new Error('Not called')
  }
  _iouring_setup = () => {
    DiskDevice._tried_io_uring_init = true
  }
}
