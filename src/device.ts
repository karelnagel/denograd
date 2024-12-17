import { DType, dtypes, ImageDType, PtrDType } from './dtype.ts'
import { assert, CI, ctypes, DEBUG, diskcache_get, diskcache_put, flat_mv, from_mv, getEnv, getNumberEnv, GlobalCounters, isNone, isNotNone, OSX, resolvePromise } from './helpers.ts'
import process from 'node:process'
import { Renderer } from './renderer/index.ts'

// # **************** Device ****************

const DEVICES = {
  // METAL: () => import('./runtime/ops_metal.ts').then((o) => o.MetalDevice),
  // AMD: () => import('./runtime/ops_amd.ts').then((o) => o.AMDDevice),
  // NV: () => import('./runtime/ops_nv.ts').then((o) => o.NVDevice),
  // CUDA: () => import('./runtime/ops_cuda.ts').then((o) => o.CUDADevice),
  // QCOM: () => import('./runtime/ops_qcom.ts').then((o) => o.QCOMDevice),
  // GPU: () => import('./runtime/ops_gpu.ts').then((o) => o.GPUDevice),
  CLANG: () => import('./runtime/ops_clang.ts').then((o) => o.ClangDevice),
  // LLVM: () => import('./runtime/ops_llvm.ts').then((o) => o.LLVMDevice),
  DISK: () => import('./runtime/ops_disk.ts').then((o) => o.DiskDevice),
}

export class _Device {
  _canonicalize = (device: string): string => {
    const d = device.split(':', 1)[0].toUpperCase()
    return d + device.slice(d.length).replace(':0', '')
  }
  // NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  canonicalize = (device?: string): string => isNotNone(device) ? this._canonicalize(device) : Device.DEFAULT
  get = (ix: string): Compiled => this.__getCanonicalizedItem(this.canonicalize(ix))
  __getCanonicalizedItem = (ix: string): Compiled => {
    // TODO take this progremmatically
    const cpn = 'MainProcess' //multiprocessing.current_process().name
    assert(cpn === 'MainProcess' || ['DISK', 'NPY', 'PYTHON'].includes(ix.split(':')[0]), `can only open device ${ix} from parent, not ${cpn}`)
    const ret = resolvePromise(DEVICES[ix.split(':')[0].toUpperCase() as keyof typeof DEVICES]())
    if (DEBUG >= 1) console.log(`opened device ${ix}`)
    return new ret(ix)
  }
  default = (): Compiled => this.get(this.DEFAULT)
  public *getAvailableDevices(): Generator<string> {
    for (const device in DEVICES) {
      try {
        yield this.get(device).device
      } catch {
        continue
      }
    }
  }
  get DEFAULT(): string {
    const fromEnv = Object.keys(DEVICES).filter((d) => !['DISK', 'NPY'].includes(d) && getNumberEnv(d) === 1)[0]
    if (fromEnv) return fromEnv

    const device = this.getAvailableDevices().next()
    if (!device) throw new Error('no usable devices')
    process.env[device.value] = '1'
    return device.value
  }
}
export const Device = new _Device()

// **************** Buffer + Allocators ****************

export class BufferSpec {
  //   # TODO: move device, size, dtype here?
  constructor(
    public image?: ImageDType,
    public uncached = false,
    public cpu_access = false,
    public host = false,
    public nolru = false,
    public external_ptr?: number,
  ) {}
}
type BufferInput = { device: string; size: number; dtype: DType; opaque?: any; options?: BufferSpec; initial_value?: Uint8Array; lb_refcount?: number; base?: Buffer; offset?: number; preallocate?: boolean }
export class Buffer {
  device: string
  size: number
  dtype: DType
  options?: BufferSpec
  offset: number

  _base?: Buffer
  _lb_refcount = 0
  _buf?: any
  allocator?: Allocator

  constructor({ device, size, dtype, opaque, options, initial_value, lb_refcount = 0, base, offset = 0, preallocate = false }: BufferInput) {
    if (dtype instanceof ImageDType) options = new BufferSpec(dtype) // TODO: image hack shouldn't be here. where should it be?
    else assert(dtype instanceof DType && !(dtype instanceof PtrDType))
    ;[this.device, this.size, this.dtype, this.options, this.offset] = [device, size, dtype, options, offset]
    if (isNone(base)) {
      assert(offset === 0, "base buffers can't have offset")
      this._lb_refcount = lb_refcount
      if (isNotNone(opaque)) this.allocate(opaque)
      if (isNotNone(initial_value)) {
        this.allocate()
        this.copyin(new DataView(initial_value.buffer))
      }
    } else {
      assert(isNone(base._base), "base can't have a base")
      assert(device === base.device, 'base must have the same device')
      this._base = base
    }
    if (preallocate) this.allocate()
  }
  get base(): Buffer {
    return isNotNone(this._base) ? this._base : this
  }
  get lb_refcount() {
    return this.base._lb_refcount
  }
  ref = (cnt: number) => this.base._lb_refcount += cnt
  is_allocated = () => !!this._buf
  ensure_allocated = (): Buffer => !this.is_allocated() ? this.allocate() : this
  allocate = (opaque?: any, external_ptr?: any): Buffer => {
    assert(!this.is_allocated(), "can't allocate already allocated buffer")
    this.allocator = Device.get(this.device).allocator
    if (isNotNone(external_ptr)) {
      this.options = this.options
        ? new BufferSpec(this.options.image, this.options.uncached, this.options.cpu_access, this.options.host, this.options.nolru, external_ptr)
        : new BufferSpec(undefined, undefined, undefined, undefined, undefined, external_ptr)
    }
    if (isNotNone(this._base)) {
      this._base.ensure_allocated()
      if (!this.allocator || !('_offset' in this.allocator)) throw new Error('offset function required for view')
      this._buf = (this.allocator._offset as any)(this.base._buf, this.nbytes, this.offset)
    } else {
      this._buf = isNotNone(opaque) ? opaque : this.allocator?.alloc(this.nbytes, this.options)
      if (!this.device.startsWith('DISK')) GlobalCounters.mem_used += this.nbytes
    }
    return this
  }
  __reduce__ = () => {
    let buf
    if (isNotNone(this._base)) {
      return [Buffer, [this.device, this.size, this.dtype, undefined, undefined, undefined, 0, this.base, this.offset, this.is_allocated()]]
    }
    if (this.device === 'NPY') return [Buffer, [this.device, this.size, this.dtype, this._buf, this.options, undefined, this.lb_refcount]]
    if (this.is_allocated()) {
      buf = new Uint8Array(this.nbytes)
      this.copyout(new DataView(buf.buffer))
    }
    return [Buffer, [this.device, this.size, this.dtype, undefined, this.options, buf, this.lb_refcount]]
  }
  // deno-fmt-ignore
  get nbytes() {
        return this.size * this.dtype.itemsize
    }
  __del__ = () => {
    if (!this.is_allocated()) return
    if (isNone(this._base) && (isNone(this.options) || isNone(this.options.external_ptr))) {
      if (!this.device.startsWith('DISK')) GlobalCounters.mem_used -= this.nbytes
      this.allocator?.free(this._buf, this.nbytes, this.options)
    }
  }
  __repr__ = () => {
    return `<buf real:${this.is_allocated()} device:${this.device} size:${this.size} dtype:${this.dtype}` + (this.base ? ` offset:${this.offset}` : '') + (isNotNone(this.options) ? ` ${this.options}` : '') + '>'
  }
  as_buffer = (allowZeroCopy = false, forceZeroCopy = false) => {
    // zero copy with as_buffer (disabled by default due to use after free)
    if ((forceZeroCopy || allowZeroCopy) && this.allocator && '_asBuffer' in this.allocator && (isNone(this.options) || isNone(this.options.image))) return (this.allocator._asBuffer as any)(this._buf)
    assert(!forceZeroCopy, 'force zero copy was passed, but copy is required')
    return this.copyout(new DataView(new Uint8Array(this.nbytes).buffer))
  }
  copyin = (mv: DataView): Buffer => {
    mv = flat_mv(mv)
    assert(mv.byteLength === this.nbytes, `size mismatch, ${mv.byteLength} != ${this.dtype} ${this.size}`)
    assert(this.is_allocated(), "can't copyin to unallocated buffer")
    this.allocator?._copyin(this._buf, mv)
    return this
  }
  copyout = (mv: DataView): DataView => {
    mv = flat_mv(mv)
    assert(mv.byteLength === this.nbytes, `size mismatch, {len(mv)=} != {this.dtype=} ${this.size}`)
    assert(this.is_allocated(), "can't copyout unallocated buffer")
    this.allocator?._copyout(mv, this._buf)
    return mv
  }
  view = (size: number, dtype: DType, offset: number): Buffer => {
    assert(offset < this.nbytes, 'offset must be less than nbytes')
    if (isNotNone(this._base)) return new Buffer({ device: this.device, size, dtype, base: this._base, offset: this.offset + offset })
    return new Buffer({ device: this.device, size, dtype, base: this, offset: offset })
  }
}

// # TODO: size, dest, src are the same type. can we enforce this?
export abstract class Allocator {
  //   # overriden in LRUAllocator

  alloc = (size: number, options?: BufferSpec) => {
    assert(typeof size !== 'number' || size > 0, `alloc size must be positve, getting {size}`)
    return this._alloc(size, isNotNone(options) ? options : new BufferSpec())
  }
  free = (opaque: number, size: number, options?: BufferSpec) => this._free(opaque, isNotNone(options) ? options : new BufferSpec())

  //   # implemented by the runtime
  abstract _alloc: (size: number, options: BufferSpec) => void
  abstract _free: (opaque: number, options: BufferSpec) => void // if opaque is a Python object, you don't need a free
  abstract _copyin: (dest: any, src: DataView) => void
  abstract _copyout: (dest: DataView, src: any) => void
  // def _as_buffer( src) -> memoryview:
  // def _offset( buf, size:number, offset:number):
  // def _transfer( dest, src, sz:number, src_dev, dest_dev):
}

/**
 * The LRU Allocator is responsible for caching buffers.
 * It ensures that buffers are not freed until it is absolutely necessary, optimizing performance.
 */
export abstract class LRUAllocator extends Allocator {
  cache = new Map<[number, BufferSpec | undefined], any>()
  override alloc = (size: number, options?: BufferSpec) => {
    const c = this.cache.get([size, options])
    if (c.length) return c.pop()
    try {
      assert(typeof size !== 'number' || size > 0, `alloc size must be positve, getting {size}`)
      return this._alloc(size, isNotNone(options) ? options : new BufferSpec())
    } catch {
      this.free_cache()
      assert(typeof size !== 'number' || size > 0, `alloc size must be positve, getting {size}`)
      return this._alloc(size, isNotNone(options) ? options : new BufferSpec())
    }
  }
  free_cache = () => {
    for (const [[sz, options], opaques] of this.cache.entries()) {
      for (const opaque of opaques) {
        this._free(opaque, isNotNone(options) ? options : new BufferSpec())
      }
      opaques.clear()
    }
  }
  override free = (opaque: any, size: number, options?: BufferSpec) => {
    if (getNumberEnv('LRU', 1) && (isNone(options) || !options.nolru)) this.cache.get([size, options]).append(opaque)
    else this._free(opaque, isNotNone(options) ? options : new BufferSpec())
  }
}

export class _MallocAllocator extends LRUAllocator {
  _alloc = (size: number, options: BufferSpec) => options.external_ptr ? (ctypes.c_uint8.mul(size)).fromAddress(options.external_ptr) : (ctypes.c_uint8.mul(size)).call()
  _asBuffer = (src: ArrayBuffer): DataView => flat_mv(new DataView(src))
  _copyin = (dest: any, src: DataView) => ctypes.memmove(dest, from_mv(src), src.byteLength)
  _copyout = (dest: DataView, src: any) => ctypes.memmove(from_mv(dest), src, dest.byteLength)
  _offset = (buf: ArrayBuffer, size: number, offset: number) => from_mv(new DataView(this._asBuffer(buf).buffer, offset, offset + size))
  _free = () => {
    throw new Error('Not implemented')
  }
}
export const MallocAllocator = new _MallocAllocator()

// # **************** for Compiled Devices ****************

export class CompileError extends Error {}

export class Compiler {
  cachekey?: string
  constructor(cachekey?: string) {
    this.cachekey = getEnv('DISABLE_COMPILER_CACHE') ? undefined : cachekey
  }
  compile = (src: string): Uint8Array => new TextEncoder().encode(src) // NOTE: empty compiler is the default
  compile_cached = (src: string): Uint8Array => {
    let lib = this.cachekey ? diskcache_get(this.cachekey, src) : undefined
    if (isNone(lib)) {
      assert(!getEnv('ASSERT_COMPILE'), `tried to compile with ASSERT_COMPILE set\n${src}`)
      lib = this.compile(src)
      if (isNotNone(this.cachekey)) diskcache_put(this.cachekey, src, lib)
    }
    return lib
  }
  disassemble = (lib: Uint8Array) => {/** pass */}
}

export class Compiled {
  constructor(
    public device: string,
    public allocator?: Allocator,
    public renderer: Renderer = new Renderer(),
    public compiler: Compiler = new Compiler(),
    public runtime?: any,
    public graph?: any,
  ) {}
  /**
   * Synchronize all pending operations on the device.
   *
   * This method ensures that all previously queued operations on the device have been completed before proceeding.
   */
  synchronize = () => {
    //     # override this in your device implementation
  }
}

// # TODO: move this to each Device
export const isDTypeSupported = (dtype: DType, device?: string): boolean => {
  if (isNone(device)) device = Device.DEFAULT
  if (dtype === dtypes.bfloat16) {
    // NOTE: this requires bf16 buffer support
    return ['AMD'].includes(device) || ['CUDA', 'NV'].includes(device) && !CI && !getEnv('PTX')
  }
  if (device === 'WEBGPU') return [dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort, dtypes.float, dtypes.int32, dtypes.uint32].includes(dtype)
  // for CI GPU and OSX, cl_khr_fp16 isn't supported
  // for CI LLVM, it segfaults because it can't link to the casting function
  // CI CUDA architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  // PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if (dtype === dtypes.half) {
    if (device === 'GPU') return !CI && !OSX
    if (['CUDA', 'NV'].includes(device)) return !CI
    if (device === 'LLVM') return OSX
    // if device === "PYTHON": return sys.version_info >= (3, 12)
  }
  if (dtype === dtypes.float64) return device !== 'METAL' && !(OSX && device === 'GPU')
  return true
}
