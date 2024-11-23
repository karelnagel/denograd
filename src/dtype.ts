import { assert } from './helpers.ts'

export type ConstType = number | boolean

// TODO: all DTypes should only be created once, DTypeMetaClass
const cache = new Map<string, any>()
const cachedOrNew = <T extends typeof DType>(Class: any, args: ConstructorParameters<T>[0]): T => {
    const key = JSON.stringify(args)
    const cached = cache.get(key)
    if (cached) return cached

    const dtype = new Class(args)
    cache.set(key, dtype)
    return dtype as any
}
export type DTypeArgs = { priority: number; itemsize: number; name: string; fmt: string | null; count: number; scalar: DType | null }

export class DType {
    // deno-fmt-ignore
    constructor(args: DTypeArgs) {
        this.priority = args.priority; this.itemsize = args.itemsize;this.name = args.name; this.fmt = args.fmt; this.count = args.count; this._scalar = args.scalar
    }
    priority: number
    itemsize: number
    name: string
    fmt: string | null
    count: number
    _scalar: DType | null

    static new = (...[priority, itemsize, name, fmt]: [number, number, string, string | null]) => new DType({ priority, itemsize, name, fmt, count: 1, scalar: null })
    reduce = (): [typeof DType, any[]] => [DType, Object.entries(this).filter((x) => typeof x[1] !== 'function').map((x) => x[1])]
    toString = () => `dtypes.${INVERSE_DTYPES_DICT[this.scalar().name as 'bool']}${this.count > 1 ? `.vec(${this.count})` : ''}`
    lt = (o: DType) => [this.priority, this.itemsize, this.name, this.fmt, this.count] < [o.priority, o.itemsize, o.name, o.fmt, o.count]
    get base(): DType {
        return this
    }
    get vcount() {
        return this.count
    }
    // @functools.lru_cache(None)
    vec(sz: number) {
        assert(this.count === 1, `can't vectorize ${this} with size ${sz}`)
        if (sz === 1 || this === dtypes.void) return this // void doesn't vectorize, and sz=1 is scalar
        return new DType({ priority: this.priority, itemsize: this.itemsize * sz, name: `${INVERSE_DTYPES_DICT[this.name as 'bool']}${sz}`, fmt: null, count: sz, scalar: this })
    }
    ptr = (local = false) => new PtrDType({ ...this, scalar: null, base: this, local, v: 1 })
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
    // @functools.lru_cache(None)
    override vec(sz: number): DType {
        assert(this.v === 1, `can't vectorize ptr ${self} with size ${sz}`)
        if (sz === 1) return this
        return new PtrDType({ ...this, v: sz, scalar: this, base: this.base, local: this.local })
    }
    override ptr = (local = false): PtrDType => {
        throw new Error("can't make a pointer from a pointer")
    }
    override get vcount() {
        return this.v
    }
    override toString = () => `${this.base.toString()}.ptr(${this.local ? 'local=True' : ''})${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

// @dataclass(frozen=True, eq=False)
class ImageDType extends PtrDType {
    shape: number[]
    constructor({ shape, ...args }: PtrDTypeArgs & { shape: number[] }) {
        super(args)
        this.shape = shape
    }
    override ptr = (local = false) => {
        assert(!local, "images can't be local")
        return this
    }
    override toString = () => `dtypes.${this.name}(${this.shape})${this.v !== 1 ? `.vec(${this.v})` : ''}`
}

class dtypes {
    //   @functools.lru_cache(None)
    static isFloat = (x: DType) => dtypes.floats.includes(x.scalar()) || x instanceof ImageDType
    //   @functools.lru_cache(None)
    static isInt = (x: DType) => dtypes.ints.includes(x.scalar())
    //   @functools.lru_cache(None)
    static isUnsigned = (x: DType) => dtypes.uints.includes(x.scalar())
    //   @staticmethod
    static fromJS = (x: any): DType => {
        if (typeof x === 'number') return Number.isInteger(x) ? dtypes.default_int : dtypes.default_float
        if (typeof x === 'boolean') return dtypes.bool
        //  put this in the last is faster because there are more items than lists/tuples to check
        if (Array.isArray(x)) return x ? x.map((x) => dtypes.fromJS(x)).sort()[0] : dtypes.default_float
        throw new Error(`Could not infer dtype of ${x} with type ${typeof x}`)
    }
    static asConst(val: ConstType | ConstType[], dtype: DType): ConstType | ConstType[] {
        if (Array.isArray(val)) {
            if (val.length !== dtype.count) throw new Error(`mismatch ${val} ${JSON.stringify(dtype)}`)
            return val.map((x) => dtypes.asConst(x, dtype) as ConstType)
        }

        // TODO: should truncate here (tinygrad)
        if (dtypes.isInt(dtype)) return Number(val)
        else if (dtypes.isFloat(dtype)) return Number(val)
        else return Boolean(val)
    }
    //   @functools.lru_cache(None)
    static min(x: DType) {
        if (dtypes.isInt(x)) return dtypes.isUnsigned(x) ? 0 : (-2) ** (x.itemsize * 8 - 1)
        return dtypes.isFloat(x) ? -Infinity : false
    }
    //   @functools.lru_cache(None)
    static max(x: DType) {
        if (dtypes.isInt(x)) return (2 ** (x.itemsize * 8 - (dtypes.isUnsigned(x) ? 0 : 1))) - 1
        return dtypes.isFloat(x) ? Infinity : true
    }
    /**
     * @returns [exponent, mantissa]
     */
    static finfo(x: DType): [number, number] {
        assert(dtypes.isFloat(x), `${x} is not a floating point type`)
        if (x === dtypes.float16) return [5, 10]
        if (x === dtypes.bfloat16) return [8, 7]
        if (x === dtypes.float32) return [8, 23]
        if (x === dtypes.float64) return [11, 52]
        throw new Error(`Invalid dtype ${x} for finfo`)
    }
    static fields = () => DTYPES_DICT
    static void = DType.new(-1, 0, 'void', null)
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
    static bfloat16 = DType.new(10, 2, '__bf16', null)
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

    //   # NOTE: these are image dtypes
    static imageh = (shp: number[]) => new ImageDType({ priority: 100, itemsize: 2, name: 'imageh', fmt: 'e', count: 1, scalar: null, base: dtypes.float32, local: false, v: 1, shape: shp })
    static imagef = (shp: number[]) => new ImageDType({ priority: 100, itemsize: 4, name: 'imagef', fmt: 'f', count: 1, scalar: null, base: dtypes.float32, local: false, v: 1, shape: shp })

    static default_float = dtypes.float32
    static default_int = dtypes.int32

    static floats = [dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64]
    static uints = [dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
    static sints = [dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64]
    static ints = [...dtypes.uints, ...dtypes.sints]
}
// TODO
// if (env_default_float := getenv("DEFAULT_FLOAT", "")):
//   dtypes.default_float = getattr(dtypes, env_default_float.lower())
//   assert dtypes.is_float(dtypes.default_float), f"{env_default_float} is not a float dtype"

type DTypeLike = string | DType
export const toDType = (x: DTypeLike): DType => (x instanceof DType) ? x : dtypes[x as 'float16']

// # https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html
// # we don't support weak type and complex type
// deno-fmt-ignore
export const promoLattice = { bool: [dtypes.int8, dtypes.uint8], int8: [dtypes.int16], int16: [dtypes.int32], int32: [dtypes.int64],
  int64: [dtypes.float16, dtypes.bfloat16], uint8: [dtypes.int16, dtypes.uint16], uint16: [dtypes.int32, dtypes.uint32],
  uint32: [dtypes.int64, dtypes.uint64], uint64: [dtypes.float16, dtypes.bfloat16],
  float16: [dtypes.float32], bfloat16: [dtypes.float32], float32: [dtypes.float64], }

// @functools.lru_cache(None)
export const _getRecursiveParents = (x: DType): Set<DType> => {
    if (x === dtypes.float64) return new Set([dtypes.float64])
    return new Set([...(promoLattice[INVERSE_DTYPES_DICT[x.name as 'bool'] as 'bool'] || []).map((x) => [..._getRecursiveParents(x)]).flat(), x])
}
// @functools.lru_cache(None)
export const leastUpperDType = (...ds: DType[]): DType => {
    const images = ds.filter((d) => (d instanceof ImageDType))
    if (images.length) return images[0]
    return [...ds.map((d) => _getRecursiveParents(d)).reduce((acc, set) => new Set([...acc].filter((x) => set.has(x))))].sort()[0]
}
export const leastUpperFloat = (dt: DType) => dtypes.isFloat(dt) ? dt : leastUpperDType(dt, dtypes.float32)

// TODO get these types programmatically
const DTYPES_DICT = {
    'bool': dtypes.bool,
    'int8': dtypes.char,
    'uint8': dtypes.uchar,
    'int16': dtypes.short,
    'uint16': dtypes.ushort,
    'int32': dtypes.int,
    'uint32': dtypes.uint,
    'int64': dtypes.long,
    'uint64': dtypes.ulong,
    'float16': dtypes.half,
    'bfloat16': dtypes.bfloat16,
    'float32': dtypes.float,
    'float64': dtypes.double,
    'half': dtypes.half,
    'float': dtypes.float,
    'double': dtypes.double,
    'uchar': dtypes.uchar,
    'ushort': dtypes.ushort,
    'uint': dtypes.uint,
    'ulong': dtypes.ulong,
    'char': dtypes.char,
    'short': dtypes.short,
    'int': dtypes.int,
    'long': dtypes.long,
}
const INVERSE_DTYPES_DICT = {
    'bool': 'bool',
    'signed char': 'char',
    'unsigned char': 'uchar',
    'short': 'short',
    'unsigned short': 'ushort',
    'int': 'int',
    'unsigned int': 'uint',
    'long': 'long',
    'unsigned long': 'ulong',
    'half': 'half',
    '__bf16': 'bfloat16',
    'float': 'float',
    'double': 'double',
    'void': 'void',
}

export const sumAccDType = (dt: DType) => {
    // default acc dtype for sum
    if (dtypes.isUnsigned(dt)) return leastUpperDType(dt, dtypes.uint)
    if (dtypes.isInt(dt) || dt === dtypes.bool) return leastUpperDType(dt, dtypes.int)
    return leastUpperDType(dt, dtypes.float)
}
export const truncateFp16 = (x: any) => {
    try {
        // TODO
        throw new Error('Not implemented')
        // return struct.unpack("@e", struct.pack("@e", float(x)))[0]
    } catch {
        return Math.abs(Infinity) * Math.sign(x)
    }
}

export const truncate = {
    bool: (x: any) => Boolean(x),
    // TODO: bfloat16 (tinygrad)
    float16: (x: any) => truncateFp16(x),
    float32: (x: number) => new Float32Array([x])[0],
    float64: (x: number) => new Float64Array([x])[0],
    uint8: (x: number) => new Uint8Array([x])[0],
    uint16: (x: number) => new Uint16Array([x])[0],
    uint32: (x: number) => new Uint32Array([x])[0],
    uint64: (x: bigint) => BigInt.asUintN(64, x),
    int8: (x: number) => new Int8Array([x])[0],
    int16: (x: number) => new Int16Array([x])[0],
    int32: (x: number) => new Int32Array([x])[0],
    int64: (x: bigint) => BigInt.asIntN(64, x),
}
