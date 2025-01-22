import { DType, dtypes, ImageDType, PtrDType } from './dtype.ts'
import { assert, cache, CI, DEBUG, get_env, get_number_env, GlobalCounters, NotImplemented, OSX, PROFILE } from './helpers.ts'
import { type Allocator, BufferSpec, type Compiled } from './runtime/allocator.ts'
import { MemoryView } from './memoryview.ts'
import { Env } from './env/index.ts'
import { ALL_DEVICES, type DeviceType } from './runtime/all.ts'

export * from './runtime/allocator.ts'
export type { AllDevices, DeviceType } from './runtime/all.ts'

// # **************** Device ****************
const DEVICES = Object.fromEntries(Object.entries(ALL_DEVICES).filter(([name]) => Env.DEVICES === undefined || Env.DEVICES.includes(name as DeviceType))) as typeof ALL_DEVICES

// Importing all the supported devices for current environment
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
    const Device = DEVICES[ix.split(':')[0].toUpperCase() as keyof typeof DEVICES]!
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
    Env.env.set(device, '1')
    return device
  }
}
export const Device = new _Device()

export class Buffer<Buf extends object = object> {
  _base?: Buffer<Buf>
  _lb_refcount?: number
  _buf?: Buf
  allocator?: Allocator<Buf>

  constructor(
    public device: DeviceType,
    public size: number,
    public dtype: DType,
    opaque?: any,
    public options?: BufferSpec,
    initial_value?: Uint8Array,
    lb_refcount = 0,
    base?: Buffer<Buf>,
    public offset = 0,
    preallocate = false,
  ) {
    if (dtype instanceof ImageDType) this.options = new BufferSpec(dtype) // TODO: image hack shouldn't be here. where should it be?
    else assert(dtype instanceof DType && !(dtype instanceof PtrDType))
    if (base === undefined) {
      if (offset !== 0) throw new Error("base buffers can't have offset")
      this._lb_refcount = lb_refcount
      if (opaque !== undefined) this.allocate(opaque)
      if (initial_value !== undefined) {
        this.allocate()
        this.copyin(new MemoryView(initial_value))
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
      if (!this.allocator || !('_offset' in this.allocator)) throw new Error('offset function required for view')
      this._buf = (this.allocator._offset as any)(this.base._buf, this.nbytes, this.offset)
    } else {
      this._buf = opaque !== undefined ? opaque : this.allocator?.alloc(this.nbytes, this.options)
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
      delete this._buf
    }
  }
  get nbytes() {
    return this.size * this.dtype.itemsize
  }
  del = () => (!this.is_allocated()) || this.deallocate()

  toString = () => {
    return `<buf real:${this.is_allocated()} device:${this.device} size:${this.size} dtype:${this.dtype}${this.base ? ` offset:${this.offset}` : ''}${this.options !== undefined ? ` ${this.options}` : ''}>`
  }
  as_buffer = async (allowZeroCopy = false, forceZeroCopy = false): Promise<MemoryView> => {
    // zero copy with as_buffer (disabled by default due to use after free)
    if ((forceZeroCopy || allowZeroCopy) && this.allocator && '_asBuffer' in this.allocator && (this.options === undefined || this.options.image === undefined)) return (this.allocator._asBuffer as any)(this._buf)
    if (forceZeroCopy) throw new Error('force zero copy was passed, but copy is required')
    return await this.copyout(new MemoryView(new Uint8Array(this.nbytes)))
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

// # TODO: move this to each Device
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

if (PROFILE) {
  throw new NotImplemented()
}
