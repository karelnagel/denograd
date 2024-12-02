import { assert, getEnv, intersection, isEq, isLessThan, max, sorted } from './helpers.ts'

export type ConstType<This = never> = number | boolean | This

export type DTypeArgs = { priority: number; itemsize: number; name: string; fmt?: string; count: number; _scalar?: DType }
export class DType {
  static dcache = new Map<string, DType>()
  priority!: number
  itemsize!: number
  name!: string
  fmt?: string
  count!: number
  _scalar?: DType
  // deno-fmt-ignore
  constructor({priority,count,itemsize,name,fmt,_scalar}: DTypeArgs) {
    const key = JSON.stringify({priority,count,itemsize,name,_scalar,fmt})
    if (DType.dcache.has(key)) return DType.dcache.get(key)!
    this.priority=priority; this.itemsize=itemsize; this.name=name; this.fmt=fmt; this.count=count; this._scalar=_scalar
    DType.dcache.set(key,this)
  }

  static new = (...[priority, itemsize, name, fmt]: [number, number, string, string | undefined]) => new DType({ priority, itemsize, name, fmt, count: 1, _scalar: undefined })
  reduce = (): [typeof DType, any[]] => [DType, Object.entries(this).filter((x) => typeof x[1] !== 'function').map((x) => x[1])]
  toString = () => `dtypes.${INVERSE_DTYPES_DICT[this.scalar().name]}${this.count > 1 ? `.vec(${this.count})` : ''}`
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
    return new DType({ priority: this.priority, itemsize: this.itemsize * sz, name: `${INVERSE_DTYPES_DICT[this.name]}${sz}`, fmt: undefined, count: sz, _scalar: this })
  }
  ptr = (local = false) => new PtrDType({ ...this, _scalar: undefined, base: this, local, v: 1 })
  scalar = () => this._scalar || this
}
export type PtrDTypeArgs = DTypeArgs & { base: DType; local: boolean; v: number }

export class PtrDType extends DType {
  _base: DType
  local: boolean
  v: number
  constructor({ base, local, v, ...args }: PtrDTypeArgs) {
    super(args)
    this._base = base
    this.local = local
    this.v = v
  }
  override get base() {
    return this._base
  }
  override vec(sz: number): DType {
    assert(this.v === 1, `can't vectorize ptr ${self} with size ${sz}`)
    if (sz === 1) return this
    return new PtrDType({ ...this, v: sz, _scalar: this, base: this.base, local: this.local })
  }
  override ptr = (local = false): PtrDType => {
    throw new Error("can't make a pointer from a pointer")
  }
  override get vcount() {
    return this.v
  }
  override toString = () => `${this.base.toString()}.ptr(${this.local ? 'local=True' : ''})${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

export class ImageDType extends PtrDType {
  shape: number[]
  constructor({ shape, ...args }: PtrDTypeArgs & { shape: number[] }) {
    super(args)
    this.shape = shape
  }
  override ptr = (local = false) => {
    assert(!local, "images can't be local")
    return this
  }
  override toString = () => `dtypes.${this.name}((${this.shape.join(', ')}))${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

export class dtypes {
  static isFloat = (x: DType) => dtypes.floats.includes(x.scalar()) || x instanceof ImageDType
  static isInt = (x: DType) => dtypes.ints.includes(x.scalar())
  static isUnsigned = (x: DType) => dtypes.uints.includes(x.scalar())
  static fromJS = (x: number | boolean | (number | boolean)[]): DType => {
    if (typeof x === 'number') return Number.isInteger(x) ? dtypes.defaultInt : dtypes.defaultFloat
    if (typeof x === 'boolean') return dtypes.bool
    //  put this in the last is faster because there are more items than lists/tuples to check
    if (Array.isArray(x)) return x ? max(x.map((x) => dtypes.fromJS(x))) : dtypes.defaultFloat
    throw new Error(`Could not infer dtype of ${x} with type ${typeof x}`)
  }
  static asConst(val: ConstType | ConstType[], dtype: DType): ConstType | ConstType[] {
    if (Array.isArray(val)) {
      if (val.length !== dtype.count) throw new Error(`mismatch (${val.map((val) => typeof val === 'boolean' ? (val ? 'True' : 'False') : val)},) ${dtype.toString()}`)
      return val.map((x) => dtypes.asConst(x, dtype) as ConstType)
    }

    // TODO: should truncate here (tinygrad)
    if (dtypes.isInt(dtype)) return Math.floor(Number(val)) //TODO: floor????? - seems ok
    else if (dtypes.isFloat(dtype)) return Number(val)
    else return Boolean(val)
  }
  static min(x: DType) {
    if (dtypes.isInt(x)) return dtypes.isUnsigned(x) ? 0 : (-2) ** (x.itemsize * 8 - 1)
    return dtypes.isFloat(x) ? -Infinity : false
  }
  static max(x: DType) {
    if (dtypes.isInt(x)) return (2 ** (x.itemsize * 8 - (dtypes.isUnsigned(x) ? 0 : 1))) - 1
    return dtypes.isFloat(x) ? Infinity : true
  }
  /**
   * @returns [exponent, mantissa]
   */
  static finfo(x: DType): [number, number] {
    assert(dtypes.isFloat(x), `${x} is not a floating point type`)
    if (isEq(x, dtypes.float16)) return [5, 10]
    if (isEq(x, dtypes.bfloat16)) return [8, 7]
    if (isEq(x, dtypes.float32)) return [8, 23]
    if (isEq(x, dtypes.float64)) return [11, 52]
    throw new Error(`Invalid dtype ${x} for finfo`)
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
  static imageh = (...shp: number[]) => new ImageDType({ priority: 100, itemsize: 2, name: 'imageh', fmt: 'e', count: 1, _scalar: undefined, base: dtypes.float32, local: false, v: 1, shape: shp })
  static imagef = (...shp: number[]) => new ImageDType({ priority: 100, itemsize: 4, name: 'imagef', fmt: 'f', count: 1, _scalar: undefined, base: dtypes.float32, local: false, v: 1, shape: shp })

  static defaultFloat = dtypes.float32
  static defaultInt = dtypes.int32

  static floats = [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]
  static uints = [dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
  static sints = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]
  static ints = [...dtypes.uints, ...dtypes.sints]
}
const envDefaultFloat = getEnv('DEFAULT_FLOAT', '')
if (envDefaultFloat) {
  dtypes.defaultFloat = dtypes[envDefaultFloat as keyof dtypes]
  assert(dtypes.isFloat(dtypes.defaultFloat), `${envDefaultFloat} is not a float dtype`)
}

type DTypeLike = string | DType
export const toDType = (x: DTypeLike): DType => (x instanceof DType) ? x : dtypes[x as 'float16']

// https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
// we don't support weak type and complex type
// deno-fmt-ignore
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
  [dtypes.float32, [dtypes.float64]]
])
export const _getRecursiveParents = (dtype: DType): DType[] => {
  if (isEq(dtype, dtypes.float64)) return [dtypes.float64]
  return [...new Set([dtype, ...promoLattice.get(dtype)!.flatMap(_getRecursiveParents)])]
}

export const leastUpperDType = (...ds: DType[]): DType => {
  const images = ds.filter((d) => (d instanceof ImageDType))
  if (images.length) return images[0]
  const res = [...intersection(...ds.flatMap((d) => new Set(_getRecursiveParents(d))))]
  return sorted(res)[0]
}
export const leastUpperFloat = (dt: DType) => dtypes.isFloat(dt) ? dt : leastUpperDType(dt, dtypes.float32)

export const DTYPES_DICT: Record<string, DType> = Object.fromEntries(Object.entries(dtypes).filter(([k, v]) => v instanceof DType && !k.startsWith('default') && k !== 'void'))
export const INVERSE_DTYPES_DICT: Record<string, string> = { ...Object.fromEntries(Object.entries(DTYPES_DICT).map(([k, v]) => [v.name, k])), 'void': 'void' }

export const sumAccDType = (dt: DType) => {
  // default acc dtype for sum
  if (dtypes.isUnsigned(dt)) return leastUpperDType(dt, dtypes.uint)
  if (dtypes.isInt(dt) || isEq(dt, dtypes.bool)) return leastUpperDType(dt, dtypes.int)
  return leastUpperDType(dt, dtypes.float)
}

// deno-fmt-ignore
export const truncate=(dtype:DType) => {
    switch (dtype){
    case dtypes.bool: return (x: any) => Boolean(x)
    // TODO: bfloat16 (tinygrad)
    case dtypes.float16: return (x: number) => new Float16Array([x])[0]
    case dtypes.float32: return (x: number) => new Float32Array([x])[0]
    case dtypes.float64: return (x: number) => new Float64Array([x])[0]
    case dtypes.uint8: return (x: number) => new Uint8Array([x])[0]
    case dtypes.uint16: return (x: number) => new Uint16Array([x])[0]
    case dtypes.uint32: return (x: number) => new Uint32Array([x])[0]
    case dtypes.uint64: return (x: bigint) => BigInt.asUintN(64, x)
    case dtypes.int8: return (x: number) => new Int8Array([x])[0]
    case dtypes.int16: return (x: number) => new Int16Array([x])[0]
    case dtypes.int32: return (x: number) => new Int32Array([x])[0]
    case dtypes.int64: return (x: bigint) => BigInt.asIntN(64, x)
    default: throw new Error(`Invalid dtype ${dtype}`)
    }
}
