// deno-lint-ignore-file require-await
import type { ImageDType } from '../dtype.ts'
import { ArrayMap, diskcache_get, diskcache_put, get_env, get_key, LRU, NotImplemented, PROFILE, string_to_bytes, WeakValueMap } from '../helpers.ts'
import { Renderer } from '../renderer/index.ts'
import type { DeviceType } from '../device.ts'
import { MemoryView } from '../memoryview.ts'

// **************** Buffer + Allocators ****************
export class BufferSpec {
  key: string
  static cache = new WeakValueMap<string, BufferSpec>()
  // TODO: move device, size, dtype here?
  constructor(
    public image?: ImageDType,
    public uncached = false,
    public cpu_access = false,
    public host = false,
    public nolru = false,
    public external_ptr?: bigint,
  ) {
    this.key = get_key(image, uncached, cpu_access, host, nolru, external_ptr)
    Object.freeze(this)
    return BufferSpec.cache.setDefault(this.key, this)
  }
}

// TODO: size, dest, src are the same type. can we enforce this?
export abstract class Allocator<Buf> {
  // overriden in LRUAllocator
  alloc(size: number, options?: BufferSpec): Buf {
    if (size <= 0) throw new Error(`alloc size must be positve, getting ${size}`)
    return this._alloc(size, options !== undefined ? options : new BufferSpec())
  }

  free(opaque: Buf, size: number, options?: BufferSpec) {
    return this._free(opaque, options !== undefined ? options : new BufferSpec())
  }

  // implemented by the runtime
  abstract _alloc: (size: number, options: BufferSpec) => Buf
  abstract _free: (opaque: Buf, options: BufferSpec) => void // if opaque is a Python object, you don't need a free
  abstract _copyin: (dest: Buf, src: MemoryView) => void
  abstract _copyout: (dest: MemoryView, src: Buf) => Promise<void> | void
  // def _as_buffer( src) -> memoryview:
  // def _offset( buf, size:number, offset:number):
  // def _transfer( dest, src, sz:number, src_dev, dest_dev):
}

/**
 * The LRU Allocator is responsible for caching buffers.
 * It ensures that buffers are not freed until it is absolutely necessary, optimizing performance.
 */
export abstract class LRUAllocator extends Allocator<MemoryView> {
  cache = new ArrayMap<[number, BufferSpec?], MemoryView[]>()
  override alloc = (size: number, options?: BufferSpec) => {
    const c = this.cache.setDefault([size, options], [])
    if (c.length) return c.pop()!
    try {
      return super.alloc(size, options)
    } catch {
      this.free_cache()
      return super.alloc(size, options)
    }
  }
  free_cache = () => {
    for (const [[size, options], opaques] of this.cache.entries()) {
      for (const opaque of opaques) super.free(opaque, size, options)
      opaques.splice(0, opaques.length)
    }
  }
  // KAREL: TODO: free gets never called
  override free = (opaque: MemoryView, size: number, options?: BufferSpec) => {
    if (LRU && (options === undefined || !options.nolru)) {
      this.cache.setDefault([size, options], []).push(opaque)
    } else super.free(opaque, size, options)
  }
}

export class _MallocAllocator extends LRUAllocator {
  _alloc = (size: number, options: BufferSpec): MemoryView => {
    const mv = new MemoryView(size)
    if (options.external_ptr) throw new Error(`TODO: external_ptr:${options.external_ptr}`)
    return mv
  }
  _as_buffer = (src: ArrayBuffer): MemoryView => new MemoryView(src).flat()
  _copyin = (dest: MemoryView, src: MemoryView) => dest.set(src)
  _copyout = (dest: MemoryView, src: MemoryView) => void dest.set(src)
  _offset = (buf: MemoryView, size: number, offset: number) => buf.slice(offset, offset + size)
  _free = () => {
    throw new NotImplemented()
  }
}
export const MallocAllocator = new _MallocAllocator()

// NOTE: MAP_JIT is added to mmap module in python 3.13
export const MAP_JIT = 0x0800

export type ProgramCallArgs = { global_size?: number[]; local_size?: number[]; vals?: number[] }
export class Program {
  constructor(public name: string, public lib: Uint8Array) {
  }
  call = async (bufs: any[], args: ProgramCallArgs, wait: boolean): Promise<number> => {
    throw new NotImplemented()
  }
}

// CPUProgram is a jit/shellcode program that can be just mmapped and jumped to
export class CPUProgram extends Program {
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    throw new NotImplemented()
  }
  override call = (bufs: any[], args: ProgramCallArgs, wait = false) => {
    throw new NotImplemented()
  }
}
// **************** for Compiled Devices ****************

export class CompileError extends Error {}

export class Compiler {
  cachekey?: string
  constructor(cachekey?: string) {
    this.cachekey = get_env('DISABLE_COMPILER_CACHE') ? undefined : cachekey
  }
  compile = (src: string): Uint8Array => string_to_bytes(src) // NOTE: empty compiler is the default
  compile_cached = (src: string): Uint8Array => {
    let lib = this.cachekey ? diskcache_get(this.cachekey, src) : undefined
    if (lib === undefined) {
      if (get_env('ASSERT_COMPILE')) throw new Error(`tried to compile with ASSERT_COMPILE set\n${src}`)
      lib = this.compile(src)
      if (this.cachekey !== undefined) diskcache_put(this.cachekey, src, lib)
    }
    return lib
  }
  disassemble = (lib: Uint8Array) => {/** pass */}
}
export class ProfileEvent {}

export class ProfileDeviceEvent extends ProfileEvent {
  constructor(public device: string, public comp_tdiff: number = 0, public copy_tdiff = 0) {
    super()
  }
}

export class ProfileRangeEvent extends ProfileEvent {
  constructor(public device: string, public name: string, public st: number, public en: number, public is_copy: boolean) {
    super()
  }
}
export class ProfileGraphEntry {
  constructor(public device: string, public name: string, public st_id: number, public en_id: number, public is_copy: boolean) {}
}

export class ProfileGraphEvent extends ProfileEvent {
  constructor(public ents: ProfileGraphEntry[], public deps: number[][], public sigs: number[]) {
    super()
  }
}
export class ProfileResult {
  constructor(public st?: number, public en?: number) {}
}

// @contextlib.contextmanager
export const cpu_profile = (name: string, device = 'CPU', is_copy = false, display = true): ProfileResult => {
  const st = performance.now(), res = new ProfileResult(st)
  const en = performance.now()
  res.en = en
  if (PROFILE && display) {
    Compiled.profile_events = [...Compiled.profile_events, new ProfileRangeEvent(device, name, st / 1000, en / 1000, is_copy)]
  }
  return res
}

export class Compiled {
  static profile_events: ProfileEvent[] = [new ProfileDeviceEvent('CPU')]

  constructor(
    public device: DeviceType,
    public allocator?: Allocator<any>,
    public renderer: Renderer = new Renderer(),
    public compiler: Compiler = new Compiler(),
    public runtime?: typeof Program,
    public graph?: any,
  ) {}
  /**
   * Synchronize all pending operations on the device.
   *
   * This method ensures that all previously queued operations on the device have been completed before proceeding.
   */
  synchronize = () => {
    throw new NotImplemented()
  }
  /**
   * Called at the end of profiling to allow the device to finalize any profiling.
   */
  _at_profile_finalize = () => {
    throw new NotImplemented()
  }
}
