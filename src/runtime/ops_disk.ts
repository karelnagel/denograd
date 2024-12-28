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
    assert(this.device.mem !== undefined, "DiskBuffer wasn't opened")
    return new MemoryView(this.device.mem).slice(this.offset, this.offset + this.size)
  }
}
// const [MAP_LOCKED, MAP_POPULATE] = [OSX ? 0 : 0x2000, mmap.MAP_POPULATE || (OSX ? 0 : 0x008000)]

export class DiskAllocator extends Allocator {
  constructor(public dev: DiskDevice) {
    super()
  }
  override _alloc = (size: number, options: any) => {
    this.dev._might_open(size)
    return new DiskBuffer(this.dev, size)
  }
  override _free = (opaque: any, options: any) => this.dev._might_close()
  _as_buffer = (src: DiskBuffer) => src._buf()
  override _copyin = (dest: DiskBuffer, src: MemoryView) => void dest._buf().set(src)
  override _copyout = (dest: MemoryView, src: DiskBuffer) => {
    // if (OSX && this.dev.fd !== undefined) {
    //       // OSX doesn't seem great at mmap, this === faster
    // with io.FileIO(this.dev.fd, "a+b", closefd=false) as fo:
    //         fo.seek(src.offset)
    //         fo.readinto(dest)
    // } else {
    dest.set(src._buf())
    // }
  }
  _offset = (buf: DiskBuffer, size: number, offset: number) => new DiskBuffer(buf.device, size, offset)
}
export class DiskDevice extends Compiled {
  static _tried_io_uring_init = false
  size?: number
  // fd?: number
  count = 0
  mem!: ArrayBuffer
  constructor(device: DeviceType) {
    super(device, undefined, undefined, undefined, undefined)
    if (!DiskDevice._tried_io_uring_init) this._iouring_setup()
    this.allocator = new DiskAllocator(this)
  }
  _might_open = (size: number) => {
    this.count += 1
    assert(this.size === undefined || size <= this.size, `can't reopen Disk tensor with larger size, opened with ${this.size}, tried to open with ${size}`)
    if (this.size !== undefined) return
    const filename = this.device.slice('DISK:'.length)
    this.size = size

    if (process.platform !== 'win32' && filename.startsWith('shm:')) {
      throw new Error('Not implemented')
    } else {
      try {
        this.mem = Deno.readFileSync(filename)
      } catch {
        Deno.writeFileSync(filename, new Uint8Array(this.size))
        this.mem = Deno.readFileSync(filename)
      }
      const stat = Deno.statSync(filename)
      if (stat.size < this.size) Deno.truncateSync(filename, this.size)
    }
    // const hp = mmap.MADV_HUGEPAGE || undefined
    // if (hasattr(this.mem, 'madvise') && hp !== undefined) this.mem.madvise(hp)
    //       with contextlib.suppress(OSError): this.mem.madvise(hp) // some systems have transparent_hugepage disabled
  }
  _might_close = () => {
    this.count -= 1
    if (this.count === 0) {
      // if (this.fd !== undefined) os.close(this.fd)
      this.size = undefined
    }
  }
  _iouring_setup = () => {
    DiskDevice._tried_io_uring_init = true
  }
}
