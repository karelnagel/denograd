// deno-lint-ignore-file require-await
import { ImageDType } from '../dtype.ts'
import { assert, dataclass, diskcache_get, diskcache_put, get_env, get_number_env, isNone, isNotNone, stringToBytes } from '../helpers.ts'
import { Renderer } from '../renderer/index.ts'
import type { DeviceType } from '../device.ts'
import { MemoryView } from '../memoryview.ts'

// **************** Buffer + Allocators ****************
@dataclass
export class BufferSpec {
  //   # TODO: move device, size, dtype here?
  constructor(
    public image?: ImageDType,
    public uncached = false,
    public cpu_access = false,
    public host = false,
    public nolru = false,
    public external_ptr?: bigint,
  ) {}
}

// # TODO: size, dest, src are the same type. can we enforce this?
export abstract class Allocator<AllocRes = MemoryView> {
  //   # overriden in LRUAllocator

  alloc = (size: number, options?: BufferSpec): AllocRes => {
    assert(typeof size !== 'number' || size > 0, `alloc size must be positve, getting {size}`)
    return this._alloc(size, isNotNone(options) ? options : new BufferSpec())
  }
  free = (opaque: MemoryView, size: number, options?: BufferSpec) => this._free(opaque, isNotNone(options) ? options : new BufferSpec())

  //   # implemented by the runtime
  abstract _alloc: (size: number, options: BufferSpec) => AllocRes
  abstract _free: (opaque: MemoryView, options: BufferSpec) => void // if opaque is a Python object, you don't need a free
  abstract _copyin: (dest: any, src: MemoryView) => any
  abstract _copyout: (dest: MemoryView, src: any) => any
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
    // KAREL: TODO: enable cache
    // const c = this.cache.get([size, options])
    // console.log(c)
    // if (c.length) return c.pop()
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
  override free = (opaque: MemoryView, size: number, options?: BufferSpec) => {
    if (get_number_env('LRU', 1) && (isNone(options) || !options.nolru)) this.cache.get([size, options]).append(opaque)
    else this._free(opaque, isNotNone(options) ? options : new BufferSpec())
  }
}

export class _MallocAllocator extends LRUAllocator {
  _alloc = (size: number, options: BufferSpec): MemoryView => {
    const mv = new MemoryView(size)
    if (options.external_ptr) throw new Error(`TODO: external_ptr:${options.external_ptr}`)
    return mv
  }
  _asBuffer = (src: ArrayBuffer): MemoryView => new MemoryView(src).flat()
  _copyin = (dest: MemoryView, src: MemoryView) => dest.set(src)
  _copyout = (dest: MemoryView, src: MemoryView) => dest.set(src)
  _offset = (buf: MemoryView, size: number, offset: number) => buf.slice(offset, offset + size)
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
    this.cachekey = get_env('DISABLE_COMPILER_CACHE') ? undefined : cachekey
  }
  compile = (src: string): Uint8Array => stringToBytes(src) // NOTE: empty compiler is the default
  compile_cached = (src: string): Uint8Array => {
    let lib = this.cachekey ? diskcache_get(this.cachekey, src) : undefined
    if (isNone(lib)) {
      assert(!get_env('ASSERT_COMPILE'), `tried to compile with ASSERT_COMPILE set\n${src}`)
      lib = this.compile(src)
      if (isNotNone(this.cachekey)) diskcache_put(this.cachekey, src, lib)
    }
    return lib
  }
  disassemble = (lib: Uint8Array) => {/** pass */}
}

export class Compiled {
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
    //     # override this in your device implementation
  }
}

export type ProgramCallInput = { global_size?: number[]; local_size?: number[]; vals?: number[] }
export class Program {
  constructor(public name: string, public lib: Uint8Array) {
  }
  call = async (bufs: any[], vals: ProgramCallInput, wait: boolean): Promise<number> => {
    throw new Error('not implemented')
  }
}
