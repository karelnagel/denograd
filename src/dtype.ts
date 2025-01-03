import { assert, checkCached, get_env, intersection, isEq, isLessThan, max, sorted } from './helpers.ts'
import { FmtStr, MemoryView } from './memoryview.ts'
export type { FmtStr } from './memoryview.ts'

export type ConstType<This = never> = number | boolean | This
export type TypedArrays = Uint8Array | Int8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | BigInt64Array | BigUint64Array | Float32Array | Float64Array

export const bitcast = (data: (number | bigint | boolean)[], srcFmt: FmtStr, destFmt: FmtStr) => {
  const src = new MemoryView.ARRAYS[srcFmt](data as any)
  return [...new MemoryView.ARRAYS[destFmt](src.buffer)]
}

export class DType {
  static dcache: Record<string, DType> = {}
  constructor(public priority: number, public itemsize: number, public name: string, public fmt: undefined | FmtStr, public count: number, public _scalar?: DType, kwargs?: any) {
    return checkCached({ priority, count, itemsize, name, _scalar, fmt, kwargs }, DType.dcache, this)
  }

  static new = (priority: number, itemsize: number, name: string, fmt?: FmtStr) => new DType(priority, itemsize, name, fmt, 1, undefined)
  reduce = (): [typeof DType, any[]] => [DType, Object.entries(this).filter((x) => typeof x[1] !== 'function').map((x) => x[1])]
  toString = () => `dtypes.${INVERSE_DTYPES_DICT[this.scalar().name]}${this.count > 1 ? `.vec(${this.count})` : ''}`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  lt = (o: DType) => isLessThan(...[this, o].map((x) => [x.priority, x.itemsize, x.name, x.fmt, x.count]) as [number[], number[]])
  get base(): DType {
    return this
  }
  get vcount() {
    return this.count
  }
  vec(sz: number) {
    assert(this.count === 1, `can't vectorize ${this} with size ${sz}`)
    if (sz === 1 || isEq(this, dtypes.void)) return this // void doesn't vectorize, and sz=1 is scalar
    return new DType(this.priority, this.itemsize * sz, `${INVERSE_DTYPES_DICT[this.name]}${sz}`, undefined, sz, this)
  }
  ptr = (local = false) => new PtrDType(this.priority, this.itemsize, this.name, this.fmt, this.count, undefined, this, local, 1)
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
    kwargs?: any,
  ) {
    super(priority, itemsize, name, fmt, count, _scalar, { _base, local, v, ...kwargs })
    this._base = _base
    this.local = local
    this.v = v
  }
  override get base() {
    return this._base
  }
  override vec(sz: number): PtrDType {
    assert(this.v === 1, `can't vectorize ptr ${this} with size ${sz}`)
    if (sz === 1) return this
    return new PtrDType(this.priority, this.itemsize, this.name, this.fmt, this.count, this, this.base, this.local, sz)
  }
  override ptr = (local = false): PtrDType => {
    throw new Error("can't make a pointer from a pointer")
  }
  override get vcount() {
    return this.v
  }
  override toString = () => `${this.base.toString()}.ptr(${this.local ? 'true' : ''})${this.v !== 1 ? `.vec(${this.v})` : ''}`
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
    public shape: number[],
  ) {
    super(priority, itemsize, name, fmt, count, _scalar, _base, local, v, { shape })
  }
  override ptr = (local = false) => {
    assert(!local, "images can't be local")
    return this
  }
  override vec(sz: number): ImageDType {
    assert(this.v === 1, `can't vectorize ptr ${this} with size ${sz}`)
    if (sz === 1) return this
    return new ImageDType(this.priority, this.itemsize, this.name, this.fmt, this.count, this, this.base, this.local, sz, this.shape)
  }
  override toString = () => `dtypes.${this.name}((${this.shape.join(', ')}))${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

export class dtypes {
  static is_float = (x: DType) => dtypes.floats.includes(x.scalar()) || x instanceof ImageDType
  static is_int = (x: DType) => dtypes.ints.includes(x.scalar())
  static is_big_int = (x: DType) => dtypes.bigints.includes(x.scalar())
  static is_unsigned = (x: DType) => dtypes.uints.includes(x.scalar())
  static from_js = (x: number | boolean | (number | boolean)[]): DType => {
    if (typeof x === 'number') return Number.isInteger(x) ? dtypes.default_int : dtypes.default_float
    if (typeof x === 'bigint') return dtypes.int64
    if (typeof x === 'boolean') return dtypes.bool
    //  put this in the last is faster because there are more items than lists/tuples to check
    if (Array.isArray(x)) return x ? max(x.map((x) => dtypes.from_js(x))) : dtypes.default_float
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
      assert(val.length === dtype.count, `mismatch (${val.map((val) => typeof val === 'boolean' ? (val ? 'True' : 'False') : val)},) ${dtype.toString()}`)
      return val.map((x) => dtypes.as_const(x, dtype) as ConstType)
    }

    // TODO: should truncate here
    if (dtypes.is_big_int(dtype)) return BigInt(Math.trunc(Number(val))) as any
    if (dtypes.is_int(dtype)) return Math.trunc(Number(val))
    else if (dtypes.is_float(dtype)) return Number(val)
    else if (Number.isNaN(val)) return true //python bool(math.nan) returns True
    else return Boolean(val)
  }
  static min(x: DType) {
    if (dtypes.is_int(x)) return dtypes.is_unsigned(x) ? 0 : (-2) ** (x.itemsize * 8 - 1)
    return dtypes.is_float(x) ? -Infinity : false
  }
  static max(dtype: DType) {
    if (dtypes.is_int(dtype)) return 2 ** (dtype.itemsize * 8) - 1 + Number(dtypes.min(dtype))
    return dtypes.is_float(dtype) ? Infinity : true
  }
  /**
   * @returns [exponent, mantissa]
   */
  static finfo(x: DType): [number, number] {
    assert(dtypes.is_float(x), `${x} is not a floating point type`)
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
  static imageh = (...shp: number[]) => new ImageDType(100, 1, 'imageh', 'e', 2, undefined, dtypes.float32, false, 1, shp)
  static imagef = (...shp: number[]) => new ImageDType(100, 4, 'imagef', 'f', 1, undefined, dtypes.float32, false, 1, shp)

  static default_float = dtypes.float32
  static default_int = dtypes.int32

  static floats = [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]
  static uints = [dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
  static sints = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]
  static ints = [...dtypes.uints, ...dtypes.sints]
  static bigints = [dtypes.uint64, dtypes.int64]
}
const envDefaultFloat = get_env('DEFAULT_FLOAT', '')
if (envDefaultFloat) {
  dtypes.default_float = dtypes[envDefaultFloat as keyof dtypes]
  assert(dtypes.is_float(dtypes.default_float), `${envDefaultFloat} is not a float dtype`)
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
export const _getRecursiveParents = (dtype: DType): DType[] => {
  if (isEq(dtype, dtypes.float64)) return [dtypes.float64]
  return [...new Set([dtype, ...promoLattice.get(dtype)!.flatMap(_getRecursiveParents)])]
}

export const least_upper_dtype = (...ds: DType[]): DType => {
  const images = ds.filter((d) => (d instanceof ImageDType))
  if (images.length) return images[0]
  const res = [...intersection(...ds.flatMap((d) => new Set(_getRecursiveParents(d))))]
  return sorted(res)[0]
}
export const least_upper_float = (dt: DType) => dtypes.is_float(dt) ? dt : least_upper_dtype(dt, dtypes.float32)

export const DTYPES_DICT: Record<string, DType> = Object.fromEntries(Object.entries(dtypes).filter(([k, v]) => v instanceof DType && !k.startsWith('default') && k !== 'void'))
export const INVERSE_DTYPES_DICT: Record<string, string> = { ...Object.fromEntries(Object.entries(DTYPES_DICT).map(([k, v]) => [v.name, k])), 'void': 'void' }

export const sum_acc_dtype = (dt: DType) => {
  // default acc dtype for sum
  if (dtypes.is_unsigned(dt)) return least_upper_dtype(dt, dtypes.uint)
  if (dtypes.is_int(dt) || isEq(dt, dtypes.bool)) return least_upper_dtype(dt, dtypes.int)
  return least_upper_dtype(dt, dtypes.float)
}

export const truncate = new Map<DType, (x: any) => any>([
  [dtypes.bool, (x: boolean) => Boolean(x)],
  // TODO: bfloat16 (tinygrad)
  [dtypes.float16, (x: number) => new Float16Array([x])[0]],
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
