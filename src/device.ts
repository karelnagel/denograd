import { DType, dtypes, ImageDType, PtrDType } from './dtype.ts'
import { assert, cache, CI, DEBUG, get_env, get_number_env, GlobalCounters, isNone, isNotNone, OSX } from './helpers.ts'
import process from 'node:process'
import { Allocator, BufferSpec, Compiled } from './runtime/allocator.ts'
import { MemoryView } from './memoryview.ts'
export * from './runtime/allocator.ts'
// # **************** Device ****************

const DEVICES = {
  // METAL: () => import('./runtime/ops_metal.ts').then((o) => o.MetalDevice),
  // AMD: () => import('./runtime/ops_amd.ts').then((o) => o.AMDDevice),
  // NV: () => import('./runtime/ops_nv.ts').then((o) => o.NVDevice),
  // CUDA: () => import('./runtime/ops_cuda.ts').then((o) => o.CUDADevice),
  // QCOM: () => import('./runtime/ops_qcom.ts').then((o) => o.QCOMDevice),
  // GPU: () => import('./runtime/ops_gpu.ts').then((o) => o.GPUDevice),
  // LLVM: () => import('./runtime/ops_llvm.ts').then((o) => o.LLVMDevice),
  CLANG: await import('./runtime/ops_clang.ts').then((o) => o.ClangDevice),
  DISK: await import('./runtime/ops_disk.ts').then((o) => o.DiskDevice),
  PYTHON: await import('./runtime/ops_python.ts').then((o) => o.PythonDevice),
}
// KAREL: enable clang
export type AllDevices = keyof typeof DEVICES
export type DeviceType = AllDevices | `${AllDevices}:${string}`

export class _Device {
  @cache
  _canonicalize(device: DeviceType): DeviceType {
    const d = device.split(':', 1)[0].toUpperCase()
    return d + device.slice(d.length).replace(':0', '') as DeviceType
  }
  // NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  canonicalize = (device?: DeviceType) => device !== undefined ? this._canonicalize(device) : Device.DEFAULT
  get = (device: DeviceType): Compiled => {
    const ix = this.canonicalize(device)
    const Device = DEVICES[ix.split(':')[0].toUpperCase() as keyof typeof DEVICES]
    if (DEBUG >= 1) console.log(`opened device ${ix}`)
    return new Device(ix)
  }
  default = () => this.get(this.DEFAULT)
  @cache
  get_available_devices(): DeviceType[] {
    const res: DeviceType[] = []
    for (const device of Object.keys(DEVICES).filter((x) => x !== 'DISK')) {
      try {
        res.push(this.get(device as DeviceType).device)
      } catch (e) {
        console.log(e)
        // Device not working
      }
    }
    return res
  }
  @cache
  get DEFAULT(): DeviceType {
    const fromEnv = Object.keys(DEVICES).filter((d) => !['DISK'].includes(d) && get_number_env(d) === 1)[0]
    if (fromEnv) return fromEnv as DeviceType

    const device = this.get_available_devices()[0]
    if (!device) throw new Error('no usable devices')
    process.env[device] = '1'
    return device
  }
}
export const Device = new _Device()

export class Buffer {
  _base?: Buffer
  _lb_refcount?: number
  _buf?: MemoryView
  allocator?: Allocator

  constructor(
    public device: DeviceType,
    public size: number,
    public dtype: DType,
    public in_opaque?: any,
    public options?: BufferSpec,
    public in_initial_value?: Uint8Array,
    lb_refcount = 0,
    base?: Buffer,
    public offset = 0,
    public in_preallocate = false,
  ) {
    if (dtype instanceof ImageDType) this.options = new BufferSpec(dtype) // TODO: image hack shouldn't be here. where should it be?
    else assert(dtype instanceof DType && !(dtype instanceof PtrDType))
    if (isNone(base)) {
      assert(offset === 0, "base buffers can't have offset")
      this._lb_refcount = lb_refcount
      if (isNotNone(in_opaque)) this.allocate(in_opaque)
      if (isNotNone(in_initial_value)) {
        this.allocate()
        this.copyin(new MemoryView(in_initial_value))
      }
    } else {
      assert(isNone(base._base), "base can't have a base")
      assert(device === base.device, 'base must have the same device')
      this._base = base
    }
    if (in_preallocate) this.allocate()
  }
  get base(): Buffer {
    return isNotNone(this._base) ? this._base : this
  }
  get lb_refcount() {
    return this.base._lb_refcount!
  }
  ref = (cnt: number) => this.base._lb_refcount! += cnt
  is_allocated = () => !!this._buf
  ensure_allocated = (): Buffer => !this.is_allocated() ? this.allocate() : this
  allocate = (opaque?: any, external_ptr?: bigint): Buffer => {
    assert(!this.is_allocated(), "can't allocate already allocated buffer")
    this.allocator = Device.get(this.device).allocator
    if (isNotNone(external_ptr)) {
      this.options = this.options ? new BufferSpec(this.options.image, this.options.uncached, this.options.cpu_access, this.options.host, this.options.nolru, external_ptr) : new BufferSpec(undefined, undefined, undefined, undefined, undefined, external_ptr)
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
    if (this.is_allocated()) {
      buf = new Uint8Array(this.nbytes)
      this.copyout(new MemoryView(buf))
    }
    return [Buffer, [this.device, this.size, this.dtype, undefined, this.options, buf, this.lb_refcount]]
  }
  get nbytes() {
    return this.size * this.dtype.itemsize
  }
  __del__ = () => {
    if (!this.is_allocated()) return
    if (isNone(this._base) && (isNone(this.options) || isNone(this.options.external_ptr))) {
      if (!this.device.startsWith('DISK')) GlobalCounters.mem_used -= this.nbytes
      this.allocator?.free(this._buf!, this.nbytes, this.options)
    }
  }
  toString = () => {
    return `<buf real:${this.is_allocated()} device:${this.device} size:${this.size} dtype:${this.dtype}${this.base ? ` offset:${this.offset}` : ''}${isNotNone(this.options) ? ` ${this.options}` : ''}>`
  }
  as_buffer = (allowZeroCopy = false, forceZeroCopy = false): MemoryView => {
    // zero copy with as_buffer (disabled by default due to use after free)
    if ((forceZeroCopy || allowZeroCopy) && this.allocator && '_asBuffer' in this.allocator && (isNone(this.options) || isNone(this.options.image))) return (this.allocator._asBuffer as any)(this._buf)
    assert(!forceZeroCopy, 'force zero copy was passed, but copy is required')
    return this.copyout(new MemoryView(new Uint8Array(this.nbytes)))
  }
  copyin = (mv: MemoryView): Buffer => {
    mv = mv.flat()
    assert(mv.byteLength === this.nbytes, `size mismatch, ${mv.byteLength} != ${this.dtype} ${this.size}`)
    assert(this.is_allocated(), "can't copyin to unallocated buffer")
    this.allocator?._copyin(this._buf, mv)
    return this
  }
  copyout = (mv: MemoryView): MemoryView => {
    mv = mv.flat()
    assert(mv.byteLength === this.nbytes, `size mismatch, {len(mv)=} != {this.dtype=} ${this.size}`)
    assert(this.is_allocated(), "can't copyout unallocated buffer")
    this.allocator?._copyout(mv, this._buf)
    return mv
  }
  view = (size: number, dtype: DType, offset: number): Buffer => {
    assert(offset < this.nbytes, 'offset must be less than nbytes')
    if (isNotNone(this._base)) return new Buffer(this.device, size, dtype, undefined, undefined, undefined, undefined, this._base, this.offset + offset)
    return new Buffer(this.device, size, dtype, undefined, undefined, undefined, undefined, this, offset)
  }
}

// # TODO: move this to each Device
export const is_dtype_supported = (dtype: DType, device?: string): boolean => {
  if (isNone(device)) device = Device.DEFAULT
  if (dtype === dtypes.bfloat16) {
    // NOTE: this requires bf16 buffer support
    return ['AMD'].includes(device) || ['CUDA', 'NV'].includes(device) && !CI && !get_env('PTX')
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
