import { DType, dtypes, ImageDType, PtrDType } from './dtype.ts'
import { assert, cache, CI, DEBUG, get_env, get_number_env, GlobalCounters, NotImplemented, OSX, PROFILE, random_id } from './helpers.ts'
import { type Allocator, BufferSpec, type Compiled } from './runtime/allocator.ts'
import { MemoryView } from './memoryview.ts'
import { Env } from './env/index.ts'
import { Ops, type UOp } from './ops.ts'

export * from './runtime/allocator.ts'

// **************** Device ****************
export const ALL_DEVICES = {
  CLANG: () => import('./runtime/ops_clang.ts').then((x) => x.CLANG),
  WEBGPU: () => import('./runtime/ops_webgpu.ts').then((x) => x.WEBGPU),
  WASM: () => import('./runtime/ops_wasm.ts').then((x) => x.WASM),
  DISK: () => import('./runtime/ops_disk.ts').then((x) => x.DISK),
  JS: () => import('./runtime/ops_js.ts').then((x) => x.JS),
}

export type AllDevices = keyof typeof ALL_DEVICES
export type DeviceType = AllDevices | `${AllDevices}:${string}`

export class _Device {
  DEVICES: Record<string, typeof Compiled> = {}
  DEFAULT!: AllDevices
  init = async () => {
    for (const [name, imp] of Object.entries(ALL_DEVICES)) {
      if (Env.DEVICES !== undefined && !Env.DEVICES.includes(name as AllDevices)) continue
      try {
        const compiled = await imp()
        await compiled.init()
        this.DEVICES[name] = compiled
      } catch (e) {
        console.log(`Device ${name} failed: ${e}`)
      }
    }

    // Setting default
    const fromEnv = Object.keys(this.DEVICES).filter((d) => !['DISK'].includes(d) && get_number_env(d) === 1)[0]
    if (fromEnv) {
      this.DEFAULT = fromEnv as AllDevices
      return
    }
    const device = Object.keys(this.DEVICES)[0]
    if (!device) throw new Error('no usable devices')
    Env.env.set(device, '1')
    this.DEFAULT = device as AllDevices
  }
  @cache
  _canonicalize(device: DeviceType): DeviceType {
    const d = device.split(':', 1)[0].toUpperCase()
    return d + device.slice(d.length).replace(':0', '') as DeviceType
  }
  // NOTE: you can't cache canonicalize in case Device.DEFAULT changes
  canonicalize = (device?: DeviceType) => device !== undefined ? this._canonicalize(device) : Device.DEFAULT
  get(device: DeviceType): Compiled {
    const ix = this.canonicalize(device)
    const Device = this.DEVICES[ix.split(':')[0].toUpperCase() as AllDevices]!
    if (DEBUG >= 1) console.log(`opened device ${ix}`)
    return new Device(ix)
  }
  default = () => this.get(this.DEFAULT)
  setDefault = (device: AllDevices) => {
    if (this.DEFAULT) Env.env.set(this.DEFAULT, '0')
    Env.env.set(device, '1')
    this.DEFAULT = device
    return device
  }
}
export const Device = new _Device()
await Device.init()

// NOTE: these 3 functions should actually be under UOp, but Buffer caused circular import
export const uop_buffer = (uop: UOp): Buffer => {
  if (uop.op === Ops.VIEW) {
    if (!uop.st!.contiguous) throw new Error("VIEW only works here if it's contiguous")
    return uop_buffer(uop.src[0])
  }
  if (uop.op !== Ops.BUFFER) throw new Error(`must be BUFFER ${uop.op}`)
  if (uop._buf) return uop._buf
  if (Array.isArray(uop.device)) throw new Error(`buffer not supported on multi ${uop.device}`)
  const ret = new Buffer(uop.device, uop.size, uop.dtype instanceof ImageDType ? uop.dtype : uop.dtype.base)
  uop._buf = ret
  return ret
}
export const uop_realized = (uop: UOp): Buffer | undefined => {
  if (uop.op === Ops.VIEW && uop.src.length === 1 && uop.src[0].op === Ops.BUFFER) return uop_realized(uop.src[0])
  return uop.op === Ops.BUFFER ? uop_buffer(uop) : undefined
}
export const uop_is_realized = (uop: UOp) => {
  return uop.base.op === Ops.MULTI ? uop.base.real_lbs.every((x) => uop_realized(x.base) !== undefined) : uop_realized(uop.base) !== undefined
}

export class Buffer<Buf extends object = object> {
  _base?: Buffer<Buf>
  _lb_refcount?: number
  _buf?: Buf
  allocator?: Allocator<Buf>
  key: string
  static register = new FinalizationRegistry((x: { allocator?: Allocator<any>; _buf: any; options?: BufferSpec; size: number }) => {
    x.allocator?.free(x._buf, x.size, x.options)
  })

  constructor(
    public device: DeviceType,
    public size: number,
    public dtype: DType,
    opaque?: any,
    public options?: BufferSpec,
    initial_value?: MemoryView,
    lb_refcount = 0,
    base?: Buffer<Buf>,
    public offset = 0,
    preallocate = false,
  ) {
    this.key = random_id()
    if (dtype instanceof ImageDType) this.options = new BufferSpec(dtype) // TODO: image hack shouldn't be here. where should it be?
    else assert(dtype instanceof DType && !(dtype instanceof PtrDType))
    if (base === undefined) {
      if (offset !== 0) throw new Error("base buffers can't have offset")
      this._lb_refcount = lb_refcount
      if (opaque !== undefined) this.allocate(opaque)
      if (initial_value !== undefined) {
        this.allocate()
        this.copyin(initial_value)
      }
    } else {
      if (base._base !== undefined) throw new Error("base can't have a base")
      if (device !== base.device) throw new Error('base must have the same device')
      this._base = base
    }
    if (preallocate) this.allocate()
  }
  get base(): Buffer<Buf> {
    return this._base !== undefined ? this._base : this
  }
  get lb_refcount() {
    return this.base._lb_refcount!
  }
  ref = (cnt: number) => this.base._lb_refcount! += cnt
  is_allocated = () => !!this._buf
  ensure_allocated = (): Buffer<Buf> => !this.is_allocated() ? this.allocate() : this
  allocate = (opaque?: Buf, external_ptr?: bigint): Buffer<Buf> => {
    if (this.is_allocated()) throw new Error("can't allocate already allocated buffer")
    this.allocator = Device.get(this.device).allocator
    if (external_ptr !== undefined) {
      this.options = this.options ? new BufferSpec(this.options.image, this.options.uncached, this.options.cpu_access, this.options.host, this.options.nolru, external_ptr) : new BufferSpec(undefined, undefined, undefined, undefined, undefined, external_ptr)
    }
    if (this._base !== undefined) {
      this._base.ensure_allocated()
      if (!this.allocator || !this.allocator._offset) throw new Error('offset function required for view')
      this._buf = this.allocator._offset(this.base._buf!, this.nbytes, this.offset)
    } else {
      this._buf = opaque !== undefined ? opaque : this.allocator?.alloc(this.nbytes, this.options)
      Buffer.register.register(this, { _buf: this._buf, allocator: this.allocator, size: this.nbytes, options: this.options })
      if (!this.device.startsWith('DISK')) GlobalCounters.mem_used += this.nbytes
    }
    return this
  }
  deallocate = () => {
    if (!this.is_allocated()) throw new Error('buffer must be allocated to deallocate')
    if (this._base === undefined && (this.options === undefined || this.options.external_ptr === undefined)) {
      if (!this.device.startsWith('DISK')) GlobalCounters.mem_used -= this.nbytes
      this.allocator!.free(this._buf!, this.nbytes, this.options)
      if ('del' in this._buf! && typeof this._buf.del === 'function') this._buf.del()
    }
    delete this._buf
  }
  get nbytes() {
    return this.size * this.dtype.itemsize
  }
  toString = () => {
    return `<buf real:${this.is_allocated()} device:${this.device} size:${this.size} dtype:${this.dtype}${this.base ? ` offset:${this.offset}` : ''}${this.options !== undefined ? ` ${this.options}` : ''}>`
  }
  as_buffer = async (allowZeroCopy = false, forceZeroCopy = false): Promise<MemoryView> => {
    // zero copy with as_buffer (disabled by default due to use after free)
    if ((forceZeroCopy || allowZeroCopy) && this.allocator && this.allocator._as_buffer && (this.options === undefined || this.options.image === undefined)) return this.allocator._as_buffer(this._buf!)
    if (forceZeroCopy) throw new Error('force zero copy was passed, but copy is required')
    return await this.copyout(new MemoryView(this.nbytes))
  }
  copyin = (mv: MemoryView): Buf => {
    mv = mv.flat()
    if (mv.byteLength !== this.nbytes) throw new Error(`size mismatch, ${mv.byteLength} != ${this.dtype} ${this.size}`)
    if (!this.is_allocated()) throw new Error("can't copyin to unallocated buffer")
    this.allocator?._copyin(this._buf!, mv)
    return this._buf!
  }
  copyout = async (mv: MemoryView): Promise<MemoryView> => {
    mv = mv.flat()
    if (mv.byteLength !== this.nbytes) throw new Error(`size mismatch, {len(mv)=} != {this.dtype=} ${this.size}`)
    if (!this.is_allocated()) throw new Error("can't copyout unallocated buffer")
    await this.allocator?._copyout(mv, this._buf!)
    return mv
  }
  view = (size: number, dtype: DType, offset: number): Buffer<Buf> => {
    if (offset >= this.nbytes) throw new Error('offset must be less than nbytes')
    if (this._base !== undefined) return new Buffer(this.device, size, dtype, undefined, undefined, undefined, undefined, this._base, this.offset + offset)
    return new Buffer(this.device, size, dtype, undefined, undefined, undefined, undefined, this, offset)
  }
}

// TODO: move this to each Device
export const is_dtype_supported = (dtype: DType, device?: string): boolean => {
  if (device === undefined) device = Device.DEFAULT
  if (dtype === dtypes.bfloat16) {
    // NOTE: this requires bf16 buffer support
    return ['AMD'].includes(device) || ['CUDA', 'NV'].includes(device) && !CI && !get_env('PTX')
  }
  if (device === 'WEBGPU') return [dtypes.bool, dtypes.char, dtypes.uchar, dtypes.short, dtypes.ushort, dtypes.float, dtypes.int32, dtypes.uint32].includes(dtype)
  // for CI GPU and OSX, cl_khr_fp16 isn't supported
  // for CI LLVM, it segfaults because it can't link to the casting function
  // CI CUDA architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  // JS supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if (dtype === dtypes.half) {
    if (device === 'GPU') return !CI && !OSX
    if (['CUDA', 'NV'].includes(device)) return !CI
    if (device === 'LLVM') return OSX
    // if device === "JS": return sys.version_info >= (3, 12)
  }
  if (dtype === dtypes.float64) return device !== 'METAL' && !(OSX && device === 'GPU')
  return true
}

if (PROFILE) {
  throw new NotImplemented()
}
