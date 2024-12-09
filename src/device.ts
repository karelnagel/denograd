import { DType, dtypes, ImageDType, PtrDType } from './dtype.ts'
import { assert, bytearray, type bytes, CI, ctypes, DEBUG, diskcache_get, diskcache_put, flat_mv, from_mv, getEnv, getNumberEnv, GlobalCounters, isNone, isNotNone, memoryview, OSX, resolvePromise } from './helpers.ts'
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
  image
  uncached
  cpuAccess
  host
  nolru
  externalPtr
  //   # TODO: move device, size, dtype here?
  constructor(p: { image?: ImageDType; uncached?: boolean; cpuAccess?: boolean; host?: boolean; nolru?: boolean; externalPtr?: number }) {
    this.image = p.image
    this.uncached = p.uncached || false
    this.cpuAccess = p.cpuAccess || false
    this.host = p.host || false
    this.nolru = p.nolru || false
    this.externalPtr = p.externalPtr
  }
}
type BufferInput = { device: string; size: number; dtype: DType; opaque?: any; options?: BufferSpec; initialValue?: ByteLengthQueuingStrategy; lbRefcount?: number; base?: Buffer; offset?: number; preallocate?: boolean }
export class Buffer {
  device: string
  size: number
  dtype: DType
  options?: BufferSpec
  offset: number

  _base?: Buffer
  _lbRefcount = 0
  _buf?: any
  allocator?: Allocator

  constructor({ device, size, dtype, opaque, options, initialValue, lbRefcount = 0, base, offset = 0, preallocate = false }: BufferInput) {
    if (dtype instanceof ImageDType) options = new BufferSpec({ image: dtype }) // TODO: image hack shouldn't be here. where should it be?
    else assert(dtype instanceof DType && !(dtype instanceof PtrDType))
    ;[this.device, this.size, this.dtype, this.options, this.offset] = [device, size, dtype, options, offset]
    if (isNone(base)) {
      assert(offset === 0, "base buffers can't have offset")
      this._lbRefcount = lbRefcount
      if (isNotNone(opaque)) this.allocate(opaque)
      if (isNotNone(initialValue)) {
        this.allocate()
        this.copyin(new memoryview(initialValue))
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
    return this.base._lbRefcount
  }
  ref = (cnt: number) => this.base._lbRefcount += cnt
  is_allocated = () => !!this._buf
  ensure_allocated = (): Buffer => !this.is_allocated() ? this.allocate() : this
  allocate = (opaque?: any, externalPtr?: any): Buffer => {
    assert(!this.is_allocated(), "can't allocate already allocated buffer")
    this.allocator = Device.get(this.device).allocator
    if (isNotNone(externalPtr)) {
      this.options = this.options ? new BufferSpec({ ...this.options, externalPtr }) : new BufferSpec({ externalPtr })
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
    if (this.device === 'NPY') return [Buffer, [this.device, this.size, this.dtype, this._buf, this.options, undefined, this.lb_refcount()]]
    if (this.is_allocated()) {
      buf = new bytearray(this.nbytes)
      this.copyout(new memoryview(buf))
    }
    return [Buffer, [this.device, this.size, this.dtype, undefined, this.options, buf, this.lb_refcount()]]
  }
  // deno-fmt-ignore
  get nbytes() {
        return this.size * this.dtype.itemsize
    }
  __del__ = () => {
    if (!this.is_allocated()) return
    if (isNone(this._base) && (isNone(this.options) || isNone(this.options.externalPtr))) {
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
    return this.copyout(new memoryview(new bytearray(this.nbytes)))
  }
  copyin = (mv: memoryview): Buffer => {
    mv = flat_mv(mv)
    assert(mv.length === this.nbytes, `size mismatch, ${mv.length} != ${this.dtype} ${this.size}`)
    assert(this.is_allocated(), "can't copyin to unallocated buffer")
    this.allocator?._copyin(this._buf, mv)
    return this
  }
  copyout = (mv: memoryview): memoryview => {
    mv = flat_mv(mv)
    assert(mv.length === this.nbytes, `size mismatch, {len(mv)=} != {this.dtype=} ${this.size}`)
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
    return this._alloc(size, isNotNone(options) ? options : new BufferSpec({}))
  }
  free = (opaque: number, size: number, options?: BufferSpec) => this._free(opaque, isNotNone(options) ? options : new BufferSpec({}))

  //   # implemented by the runtime
  abstract _alloc: (size: number, options: BufferSpec) => void
  abstract _free: (opaque: number, options: BufferSpec) => void // if opaque is a Python object, you don't need a free
  abstract _copyin: (dest: string, src: memoryview) => void
  abstract _copyout: (dest: memoryview, src: string) => void
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
      return this._alloc(size, isNotNone(options) ? options : new BufferSpec({}))
    } catch {
      this.free_cache()
      assert(typeof size !== 'number' || size > 0, `alloc size must be positve, getting {size}`)
      return this._alloc(size, isNotNone(options) ? options : new BufferSpec({}))
    }
  }
  free_cache = () => {
    for (const [[sz, options], opaques] of this.cache.entries()) {
      for (const opaque of opaques) {
        this._free(opaque, isNotNone(options) ? options : new BufferSpec({}))
      }
      opaques.clear()
    }
  }
  override free = (opaque: any, size: number, options?: BufferSpec) => {
    if (getNumberEnv('LRU', 1) && (isNone(options) || !options.nolru)) this.cache.get([size, options]).append(opaque)
    else this._free(opaque, isNotNone(options) ? options : new BufferSpec({}))
  }
}

export class _MallocAllocator extends LRUAllocator {
  _alloc = (size: number, options: BufferSpec) => options.externalPtr ? (ctypes.c_uint8.mul(size)).fromAddress(options.externalPtr) : (ctypes.c_uint8.mul(size)).call()
  _asBuffer = (src: any): memoryview => flat_mv(new memoryview(src))
  _copyin = (dest: any, src: memoryview) => ctypes.memmove(dest, from_mv(src), src.length)
  _copyout = (dest: memoryview, src: any) => ctypes.memmove(from_mv(dest), src, dest.length)
  _offset = (buf: any, size: number, offset: number) => from_mv(this._asBuffer(buf).slice(offset, offset + size))
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
  compile = (src: string): bytes => new TextEncoder().encode(src) // NOTE: empty compiler is the default
  compile_cached = (src: string): bytes => {
    let lib = this.cachekey ? diskcache_get(this.cachekey, src) : undefined
    if (isNone(lib)) {
      assert(!getEnv('ASSERT_COMPILE'), `tried to compile with ASSERT_COMPILE set\n${src}`)
      lib = this.compile(src)
      if (isNotNone(this.cachekey)) diskcache_put(this.cachekey, src, lib)
    }
    return lib
  }
  disassemble = (lib: bytes) => {/** pass */}
}

export class Compiled {
  device: string
  allocator?: Allocator
  renderer: Renderer
  compiler: Compiler
  runtime: any
  graph: any
  // deno-fmt-ignore
  constructor(device:string,allocator?:Allocator,renderer?:Renderer ,compiler?:Compiler,runtime?:any,graph?:any){
        this.device=device; this.allocator=allocator; this.renderer=renderer||new Renderer(); this.compiler=compiler || new Compiler(); this.runtime=runtime; this.graph=graph
    }
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
