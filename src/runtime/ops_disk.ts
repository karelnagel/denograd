// from __future__ import annotations
// import os, sys, mmap, io, ctypes, ctypes.util, contextlib
// from typing import Optional, Generator, Tuple, Callable, List
// from tinygrad.helpers import OSX, round_up
// from tinygrad.device import Compiled, Allocator
// with contextlib.suppress(ImportError):
//   import _posixshmem
//   from tinygrad.runtime.autogen import io_uring, libc

import { Allocator, Compiled } from '../device.ts'
import { assert, OSX } from '../helpers.ts'
import process from 'node:process'
import os from 'node:os'

export class DiskBuffer {
  device: DiskDevice
  size: number
  offset: number

  constructor(device: DiskDevice, size: number, offset = 0) {
    this.device = device, this.size = size, this.offset = offset
  }
  toString = () => `<DiskBuffer size=${this.size} offset=${this.offset}>`
  _buf = (): DataView => {
    assert(this.device.mem !== undefined, "DiskBuffer wasn't opened")
    return new DataView(this.device.mem, this.offset, this.offset + this.size)
  }
}
// const [MAP_LOCKED, MAP_POPULATE] = [OSX ? 0 : 0x2000, mmap.MAP_POPULATE || (OSX ? 0 : 0x008000)]

export class DiskAllocator extends Allocator {
  dev: DiskDevice
  constructor(dev: DiskDevice) {
    super()
    this.dev = dev
  }
  override _alloc = (size: number, options: any) => {
    this.dev._might_open(size)
    return new DiskBuffer(this.dev, size)
  }
  override _free = (opaque: any, options: any) => this.dev._might_close()
  _as_buffer = (src: DiskBuffer) => src._buf()
  override _copyin = (dest: DiskBuffer, src: DataView) => {
    // TODO:
    // dest._buf() = src
  }
  override _copyout = (dest: DataView, src: DiskBuffer) => {
    if (OSX && this.dev.fd !== undefined) {
      //       // OSX doesn't seem great at mmap, this === faster
      // with io.FileIO(this.dev.fd, "a+b", closefd=false) as fo:
      //         fo.seek(src.offset)
      //         fo.readinto(dest)
    } else {
      dest = src._buf()
    }
  }
  _offset = (buf: DiskBuffer, size: number, offset: number) => new DiskBuffer(buf.device, size, offset)
}
export class DiskDevice extends Compiled {
  static _tried_io_uring_init = false
  size?: number
  fd?: number
  count = 0
  mem!: ArrayBuffer
  constructor(device: string) {
    super(device, undefined, undefined, undefined, undefined)
    this.allocator = new DiskAllocator(this)
    if (!DiskDevice._tried_io_uring_init) this._iouring_setup()
  }
  _might_open = (size: number) => {
    this.count += 1
    assert(this.size === undefined || size <= this.size, `can't reopen Disk tensor with larger size, opened with ${this.size}, tried to open with ${size}`)
    if (this.size !== undefined) return
    const filename = this.device.slice('disk:'.length)
    this.size = size

    if (process.platform !== 'win32' && filename.startsWith('shm:')) {
      //       fd = _posixshmem.shm_open("/"+filename[4:].lstrip("/"), os.O_RDWR, 0o600)
      //       this.mem = mmap.mmap(fd, this.size, mmap.MAP_SHARED | MAP_POPULATE | MAP_LOCKED)
      //       os.close(fd)
    } else {
      //       try: this.fd = os.open(filename, os.O_RDWR|os.O_CREAT|getattr(os, "O_DIRECT", 0))
      //       except OSError: this.fd = os.open(filename, os.O_RDWR|os.O_CREAT)
      //       if os.fstat(this.fd).st_size < this.size: os.ftruncate(this.fd, this.size)
      //       this.mem = mmap.mmap(this.fd, this.size)
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
