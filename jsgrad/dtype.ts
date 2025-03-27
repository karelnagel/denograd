import { cache, cache_fn, type ConstType, get_key, intersection, is_less_than, vars, WeakValueMap } from './helpers.ts'
import { type FmtStr, MemoryView } from './memoryview.ts'
export type { FmtStr } from './memoryview.ts'
export type { ConstType } from './helpers.ts'

export const bitcast = (data: (number | bigint | boolean)[], srcFmt: FmtStr, destFmt: FmtStr) => {
  const src = new MemoryView.ARRAYS[srcFmt](data as any)
  return [...new MemoryView.ARRAYS[destFmt](src.buffer)]
}

export class DType {
  key: string
  static cache = new WeakValueMap<string, DType>()
  constructor(public priority: number, public itemsize: number, public name: string, public fmt: undefined | FmtStr, public count: number, public _scalar?: DType, kwargs: any[] = []) {
    this.key = get_key(priority, count, itemsize, name, _scalar, fmt, ...kwargs)
    return DType.cache.setDefault(this.key, this)
  }

  static new = (priority: number, itemsize: number, name: string, fmt?: FmtStr) => new DType(priority, itemsize, name, fmt, 1, undefined)
  toString = () => `dtypes.${INVERSE_DTYPES_DICT[this.scalar().name]}${this.count > 1 ? `.vec(${this.count})` : ''}`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  lt = (o: DType) => is_less_than(...[this, o].map((x) => [x.priority, x.itemsize, x.name, x.fmt, x.count]) as [number[], number[]])
  get base(): DType {
    return this
  }
  get vcount() {
    return this.count
  }
  vec = cache((sz: number): DType => {
    if (this.count !== 1) throw new Error(`can't vectorize ${this} with size ${sz}`)
    if (sz === 1 || this === dtypes.void) return this // void doesn't vectorize, and sz=1 is scalar
    return new DType(this.priority, this.itemsize * sz, `${INVERSE_DTYPES_DICT[this.name]}${sz}`, undefined, sz, this)
  })
  ptr = (size = -1, local = false) => new PtrDType(this.priority, this.itemsize, this.name, this.fmt, this.count, undefined, this, local, 1, size)
  scalar = () => this._scalar || this
}

export class PtrDType extends DType {
  constructor(
    priority: number,
    itemsize: number,
    name: string,
    fmt: undefined | FmtStr,
    count: number,
    _scalar: undefined | DType,
    public _base: DType,
    public local: boolean,
    public v: number,
    public size = -1,
    kwargs: any[] = [],
  ) {
    super(priority, itemsize, name, fmt, count, _scalar, [_base, local, v, size, ...kwargs])
  }
  override get base() {
    return this._base
  }
  override vec = cache((sz: number): PtrDType => {
    if (this.v !== 1) throw new Error(`can't vectorize ptr ${this} with size ${sz}`)
    if (sz === 1) return this
    return new PtrDType(this.priority, this.itemsize, this.name, this.fmt, this.count, this, this.base, this.local, sz)
  })
  override ptr = (size = -1, local = false): PtrDType => {
    throw new Error("can't make a pointer from a pointer")
  }
  override get vcount() {
    return this.v
  }
  override toString = () => `${this.base.toString()}.ptr(${this.size}${this.local ? ', true' : ''})${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

export class ImageDType extends PtrDType {
  constructor(
    priority: number,
    itemsize: number,
    name: string,
    fmt: undefined | FmtStr,
    count: number,
    _scalar: undefined | DType,
    _base: DType,
    local: boolean,
    v: number,
    size = -1,
    public shape: number[],
  ) {
    super(priority, itemsize, name, fmt, count, _scalar, _base, local, v, size, [shape])
  }
  override ptr = (size = -1, local = false) => {
    if (local) throw new Error("images can't be local")
    return this
  }
  override vec = cache((sz: number): ImageDType => {
    if (this.v !== 1) throw new Error(`can't vectorize ptr ${this} with size ${sz}`)
    if (sz === 1) return this
    return new ImageDType(this.priority, this.itemsize, this.name, this.fmt, this.count, this, this.base, this.local, sz, this.size, this.shape)
  })
  override toString = () => `dtypes.${this.name}((${this.shape.join(', ')}))${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

export class dtypes {
  static is_float = cache((x: DType) => {
    return dtypes.floats.includes(x.scalar()) || x instanceof ImageDType
  })
  static is_int = cache((x: DType) => {
    return dtypes.ints.includes(x.scalar())
  })
  static is_big_int = cache((x: DType) => {
    return dtypes.bigints.includes(x.scalar())
  })
  static is_unsigned = cache((x: DType) => {
    return dtypes.uints.includes(x.scalar())
  })
  static from_js = (x: number | boolean | bigint | (number | bigint | boolean)[]): DType => {
    if (typeof x === 'number') return Number.isInteger(x) ? dtypes.default_int : dtypes.default_float
    if (typeof x === 'bigint') return dtypes.int64
    if (typeof x === 'boolean') return dtypes.bool
    //  put this in the last is faster because there are more items than lists/tuples to check
    if (Array.isArray(x)) return x.length ? x.map((x) => dtypes.from_js(x)).reduce((max, curr) => max.lt(curr) ? curr : max) : dtypes.default_float
    throw new Error(`Could not infer dtype of ${x} with type ${typeof x}`)
  }
  static verify = (val: ConstType, dtype: DType) => {
    if (dtypes.is_big_int(dtype)) return typeof val === 'bigint'
    if (dtypes.is_int(dtype)) return Number.isInteger(val)
    if (dtypes.is_float(dtype)) return typeof val === 'number'
    return typeof val === 'boolean'
  }
  static as_const(val: ConstType | ConstType[], dtype: DType): ConstType | ConstType[] {
    if (Array.isArray(val)) {
      if (val.length !== dtype.count) throw new Error(`mismatch (${val.map((val) => typeof val === 'boolean' ? (val ? 'True' : 'False') : val)},) ${dtype.toString()}`)
      return val.map((x) => dtypes.as_const(x, dtype) as ConstType)
    }

    // TODO: should truncate here
    if (dtypes.is_big_int(dtype)) return typeof val === 'bigint' ? val : BigInt(Math.trunc(Number(val)))
    if (dtypes.is_int(dtype)) return Math.trunc(Number(val))
    else if (dtypes.is_float(dtype)) return Number(val)
    else if (Number.isNaN(val)) return true //python bool(math.nan) returns True
    else return Boolean(val)
  }
  static min = cache((dtype: DType) => {
    if (dtypes.is_big_int(dtype)) return dtypes.is_unsigned(dtype) ? 0n : (-2n) ** (BigInt(dtype.itemsize) * 8n - 1n)
    if (dtypes.is_int(dtype)) return dtypes.is_unsigned(dtype) ? 0 : (-2) ** (dtype.itemsize * 8 - 1)
    return dtypes.is_float(dtype) ? -Infinity : false
  })
  static max = cache((dtype: DType) => {
    if (dtypes.is_big_int(dtype)) return 2n ** (BigInt(dtype.itemsize) * 8n) - 1n + BigInt(dtypes.min(dtype))
    if (dtypes.is_int(dtype)) return 2 ** (dtype.itemsize * 8) - 1 + Number(dtypes.min(dtype))
    return dtypes.is_float(dtype) ? Infinity : true
  })
  /**
   * @returns [exponent, mantissa]
   */
  static finfo(x: DType): [number, number] {
    if (!dtypes.is_float(x)) throw new Error(`${x} is not a floating point type`)
    return new Map<DType, [number, number]>([[dtypes.float16, [5, 10]], [dtypes.bfloat16, [8, 7]], [dtypes.float32, [8, 23]], [dtypes.float64, [11, 52]]]).get(x)!
  }
  static fields = () => DTYPES_DICT
  static void = DType.new(-1, 0, 'void', undefined)
  static bool = DType.new(0, 1, 'bool', '?')
  static int8 = DType.new(1, 1, 'signed char', 'b')
  static uint8 = DType.new(2, 1, 'unsigned char', 'B')
  static int16 = DType.new(3, 2, 'short', 'h')
  static uint16 = DType.new(4, 2, 'unsigned short', 'H')
  static int32 = DType.new(5, 4, 'int', 'i')
  static uint32 = DType.new(6, 4, 'unsigned int', 'I')
  static int64 = DType.new(7, 8, 'long', 'q')
  static uint64 = DType.new(8, 8, 'unsigned long', 'Q')
  static float16 = DType.new(9, 2, 'half', 'e')
  // bfloat16 has higher priority than float16, so least_upper_dtype(dtypes.int64, dtypes.uint64) = dtypes.float16
  static bfloat16 = DType.new(10, 2, '__bf16', undefined)
  static float32 = DType.new(11, 4, 'float', 'f')
  static float64 = DType.new(12, 8, 'double', 'd')

  // dtype aliases
  static half = dtypes.float16
  static float = dtypes.float32
  static double = dtypes.float64
  static uchar = dtypes.uint8
  static ushort = dtypes.uint16
  static uint = dtypes.uint32
  static ulong = dtypes.uint64
  static char = dtypes.int8
  static short = dtypes.int16
  static int = dtypes.int32
  static long = dtypes.int64

  // NOTE: these are image dtypes
  static imageh = (...shp: number[]) => new ImageDType(100, 1, 'imageh', 'e', 2, undefined, dtypes.float32, false, 1, shp.reduce((acc, x) => acc * x, 1), shp)
  static imagef = (...shp: number[]) => new ImageDType(100, 4, 'imagef', 'f', 1, undefined, dtypes.float32, false, 1, shp.reduce((acc, x) => acc * x, 1), shp)

  static default_float = dtypes.float32
  static default_int = dtypes.int32

  static floats = [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]
  static uints = [dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
  static sints = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]
  static ints = [...dtypes.uints, ...dtypes.sints]
  static bigints = [dtypes.uint64, dtypes.int64]
}
const envDefaultFloat = vars.get('DEFAULT_FLOAT', '')
if (envDefaultFloat) {
  dtypes.default_float = dtypes[envDefaultFloat as keyof dtypes]
  if (!dtypes.is_float(dtypes.default_float)) throw new Error(`${envDefaultFloat} is not a float dtype`)
}

export type DTypeLike = string | DType
export const to_dtype = (x: DTypeLike): DType => (x instanceof DType) ? x : dtypes[x as 'float16']

// https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
// we don't support weak type and complex type
export const promoLattice = new Map<DType, DType[]>([
  [dtypes.bool, [dtypes.int8, dtypes.uint8]],
  [dtypes.int8, [dtypes.int16]],
  [dtypes.int16, [dtypes.int32]],
  [dtypes.int32, [dtypes.int64]],
  [dtypes.int64, [dtypes.float16, dtypes.bfloat16]],
  [dtypes.uint8, [dtypes.int16, dtypes.uint16]],
  [dtypes.uint16, [dtypes.int32, dtypes.uint32]],
  [dtypes.uint32, [dtypes.int64, dtypes.uint64]],
  [dtypes.uint64, [dtypes.float16, dtypes.bfloat16]],
  [dtypes.float16, [dtypes.float32]],
  [dtypes.bfloat16, [dtypes.float32]],
  [dtypes.float32, [dtypes.float64]],
])

export const _get_recursive_parents = cache_fn((dtype: DType): DType[] => {
  if (dtype === dtypes.float64) return [dtypes.float64]
  return [...new Set([dtype, ...promoLattice.get(dtype)!.flatMap(_get_recursive_parents)])]
})

export const least_upper_dtype = cache_fn((...ds: DType[]): DType => {
  const images = ds.filter((d) => (d instanceof ImageDType))
  if (images.length) return images[0]
  const res = [...intersection(...ds.flatMap((d) => new Set(_get_recursive_parents(d))))]
  return res.reduce((min, curr) => min.lt(curr) ? min : curr)
})
export const least_upper_float = (dt: DType) => dtypes.is_float(dt) ? dt : least_upper_dtype(dt, dtypes.float32)

export const DTYPES_DICT: Record<string, DType> = Object.fromEntries(Object.entries(dtypes).filter(([k, v]) => v instanceof DType && !k.startsWith('default') && k !== 'void'))
export const INVERSE_DTYPES_DICT: Record<string, string> = { ...Object.fromEntries(Object.entries(DTYPES_DICT).map(([k, v]) => [v.name, k])), 'void': 'void' }

export const sum_acc_dtype = (dt: DType) => {
  // default acc dtype for sum
  if (dtypes.is_unsigned(dt)) return least_upper_dtype(dt, dtypes.uint)
  if (dtypes.is_int(dt) || dt === dtypes.bool) return least_upper_dtype(dt, dtypes.int)
  return least_upper_dtype(dt, dtypes.float)
}

export const truncate = new Map<DType, (x: any) => any>([
  [dtypes.bool, (x: boolean) => Boolean(x)],
  // TODO: bfloat16 (tinygrad)
  // @ts-ignore Float16Array exists in deno
  [dtypes.float16, (x: number) => new (typeof Float16Array !== 'undefined' ? Float16Array : Float32Array)([x])[0]],
  [dtypes.float32, (x: number) => new Float32Array([x])[0]],
  [dtypes.float64, (x: number) => new Float64Array([x])[0]],
  [dtypes.uint8, (x: number) => new Uint8Array([x])[0]],
  [dtypes.uint16, (x: number) => new Uint16Array([x])[0]],
  [dtypes.uint32, (x: number) => new Uint32Array([x])[0]],
  [dtypes.uint64, (x: bigint) => BigInt.asUintN(64, x)],
  [dtypes.int8, (x: number) => new Int8Array([x])[0]],
  [dtypes.int16, (x: number) => new Int16Array([x])[0]],
  [dtypes.int32, (x: number) => new Int32Array([x])[0]],
  [dtypes.int64, (x: bigint) => BigInt.asIntN(64, x)],
])
