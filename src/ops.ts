import { createHash } from 'node:crypto'
import { type ConstType, DType, dtypes, ImageDType, PtrDType, truncate } from './dtype.ts'
import { allSame, assert, isNone, isNotNone, isSubset, mathGcd, partition, permutations, prod, raise, range } from './helpers.ts'
import { Buffer } from 'node:buffer'
import { readFileSync } from 'node:fs'
import { asdict } from '../test/helpers.ts'

export type sint = number | UOp
export type Variable = UOp
export type ConstLike<This = never> = ConstType<This> | Variable | ConstType[]

class SimpleMathTrait {
    //   # required to implement
    alu = (arg: Ops, ...src: typeof this[]): typeof this => raise('Not implemented')
    constLike = (b: ConstLike): typeof this => raise('Not implemented')

    //   # great functions you get!
    ufix = (x: ConstType<typeof this>): typeof this => x instanceof MathTrait ? x : this.constLike(x as any) //ignoring this error, cause not sure
    _binop = (op: Ops, x: ConstType<typeof this>, reverse: boolean) => reverse ? this.ufix(x).alu(op, this) : this.alu(op, this.ufix(x))
    logicalNot = () => this.ne(true)
    neg = () => {
        const dtype = 'dtype' in this && this.dtype instanceof DType ? this.dtype : null
        if (isNone(dtype)) throw new Error('MathTraits __neg__ requires a dtype')
        return dtype.scalar() === dtypes.bool ? this.logicalNot() : this.mul(-1)
    }
    add = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.ADD, x, reverse)
    mul = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.MUL, x, reverse)
    bitwiseAnd = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.AND, x, reverse)
    bitwiseOr = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.OR, x, reverse)
    xor = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.XOR, x, reverse)
    idiv = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.IDIV, x, reverse)
    sub = (x: ConstType<typeof this>, reverse = false) => reverse ? this.ufix(x).alu(Ops.ADD, this.neg()) : this.alu(Ops.ADD, this.ufix(x).neg())
    div = (x: ConstType<typeof this>, reverse = false) => reverse ? this.ufix(x).mul(this.alu(Ops.RECIP)) : this.mul(this.ufix(x).alu(Ops.RECIP))

    lt = (x: ConstType<typeof this>) => this.alu(Ops.CMPLT, this.ufix(x))
    gt = (x: ConstType<typeof this>) => this.ufix(x).alu(Ops.CMPLT, this)
    ne = (x: ConstType<typeof this>) => this.alu(Ops.CMPNE, this.ufix(x))
    ge = (x: ConstType<typeof this>) => this.lt(x).logicalNot()
    le = (x: ConstType<typeof this>) => this.gt(x).logicalNot()
    eq = (x: ConstType<typeof this>) => this.ne(x).logicalNot()
}

class MathTrait extends SimpleMathTrait {
    // TODO: move to Tensor when new backward is done (tinygrad)
    lshift = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.SHL, x, reverse)
    rshift = (x: ConstType<typeof this>, reverse = false) => this._binop(Ops.SHR, x, reverse)

    //   # not in Tensor
    mod = (x: ConstType<typeof this>, reverse = false) => !reverse ? this.alu(Ops.MOD, this.ufix(x)) : this.ufix(x).alu(Ops.MOD, this)
    maximum = (x: ConstType<typeof this>) => this.alu(Ops.MAX, this.ufix(x))
    minimum = (x: ConstType<typeof this>) => this.neg().maximum(this.ufix(x).neg()).neg()
    where = (x: ConstType<typeof this>, y: ConstType<typeof this>) => this.alu(Ops.WHERE, this.ufix(x), this.ufix(y))
    threefry = (seed: ConstType<typeof this>) => this.alu(Ops.THREEFRY, this.ufix(seed))
    reciprocal = () => this.alu(Ops.RECIP)
    sqrt = () => this.alu(Ops.SQRT)
    sin = () => this.alu(Ops.SIN)
    log2 = () => this.alu(Ops.LOG2)
    exp2 = () => this.alu(Ops.EXP2)
}

// # the order of these Ops controls the order of the toposort
// deno-fmt-ignore
enum Ops{
    // uops that aren't rendered
    SINK, CONTIGUOUS, PRELOAD,

    // MetaOps
    COPY, EMPTY, BUFFER_VIEW,

    EXPAND, CONTRACT, VIEW, DEFINE_GLOBAL, BUFFER, DEFINE_VAR, 
    DEFINE_LOCAL, DEFINE_ACC, VALID, SPECIAL, NOOP,

    // reduce
    REDUCE_AXIS,

    // helper ops
    GEP, VECTORIZE,

    // UnaryOps
    CAST, BITCAST, EXP2, LOG2, SIN, SQRT, RECIP, NEG,

    // loads before math
    LOAD,

    // math ops
    WMMA,

    // BinaryOps
    ADD, MUL, IDIV,MAX, MOD, CMPLT, CMPNE, XOR,
    SHL, SHR, OR, AND, THREEFRY, SUB, FDIV,

    // TernaryOps
    WHERE, MULACC,

    // assignment ops
    STORE, ASSIGN, BIND,

    // late INDEX
    INDEX,

    // control flow ops
    BARRIER, IF, RANGE,

    // ops that are not graph nodes
    ENDRANGE, ENDIF,

    // consts last!
    VCONST, CONST,
}
const getEnumString = (op: Ops) => {
    for (const key in Ops) if (Ops[key] === op as unknown as keyof Ops) return key
    return undefined
}
class GroupOp {
    static Unary = [Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG]
    static Binary = [Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV]
    static Ternary = [Ops.WHERE, Ops.MULACC]
    static ALU = [...this.Unary, ...this.Binary, ...this.Ternary]

    static Irreducible = [Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE]

    //   # meta ops
    static Meta = [Ops.COPY, Ops.EMPTY, Ops.BUFFER_VIEW]
    static Buffer = [Ops.LOAD, Ops.PRELOAD, Ops.STORE, Ops.VALID]

    //   # BinaryOps that can be flipped
    static Commutative = [Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR]

    //   # do not preserve f(0) = 0
    static UnsafePad = [Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV]
}

// # https://en.wikipedia.org/wiki/Identity_element
// const identityElement=(op:Ops, dt:DType) =>  dtypes.asConst({Ops.ADD:0, Ops.MUL:1, Ops.MAX:dtypes.min(dt)}[op], dt)

export const canPad = (u: UOp) => {
    for (const x of u.sparents().keys()) if (GroupOp.UnsafePad.includes(x.op)) return true
    return false
}

export const END_FOR_UOP = new Map([[Ops.IF, [Ops.STORE, Ops.ENDIF]], [Ops.RANGE, [Ops.ASSIGN, Ops.ENDRANGE]]])

// With True as the default, this matches the old symbolic behavior
export const resolve = (x: UOp, def = false) => {
    if (!(x instanceof UOp)) return Boolean(x)
    if (x.dtype.name !== 'bool') throw new Error('UOp in resolve must be bool')
    // NOTE: generating the text for the exception is expensive, so we do this
    const sx = x.simplify()
    // TODO this Boolean() is probably broken
    return sx.vmin() === sx.vmax() ? Boolean(sx.vmin()) : def
}

// # smax/smin are replacements for max/min that preserve symbolic
const _suop = (lst: (UOp | UOp[])[], uop_fxn: (...x: UOp[]) => UOp, python_fxn: (a: UOp[][]) => UOp): UOp[] | UOp => {
    const [maxUop, maxNum] = partition(lst, (x) => x instanceof UOp) as [UOp[], UOp[][]]
    if (maxUop.length) return (maxNum.length ? [...maxUop, python_fxn(maxNum)] : maxUop).reduce((prev, curr) => uop_fxn(prev, curr)).ssimplify()
    return python_fxn(maxNum)
}
// TODO: really unsure about these
export const smax = (...lst: (UOp | UOp[])[]) => _suop(Array.isArray(lst[0]) ? lst[0] as UOp[] : lst as UOp[], (...x) => x.reduce((prev, curr) => curr.maximum(prev)), (x) => UOp.max(...x.flat()))
export const smin = (...lst: (UOp | UOp[])[]) => _suop(Array.isArray(lst[0]) ? lst[0] as UOp[] : lst as UOp[], (...x) => x.reduce((prev, curr) => curr.maximum(prev)), (x) => UOp.min(...x.flat()))

export const ssimplify = (uop: UOp) => uop instanceof UOp ? uop.ssimplify() : uop
export const symInfer = (uop: UOp | number, varVals: Map<UOp, number>): number => uop instanceof UOp ? uop.symInfer(varVals) : uop

// AI generated
// used for UOp and UPat
export const prettyPrint = (x: any, rep: (x: any) => string, srcfn: (x: any) => any[] = (x) => x.src, cache: Map<any, [number, number, boolean]> = new Map(), d = 0): string => {
    const dfs = (x: any, cache: Map<any, [number, number, boolean]>) => {
        for (const s of srcfn(x) || []) {
            if (!cache.has(s)) cache.set(s, [cache.size, 0, false])
            const entry = cache.get(s)!
            entry[1]++
            if (entry[1] === 1) dfs(s, cache)
        }
    }

    if (cache.size === 0) dfs(x, cache)

    if (!cache.has(x)) cache.set(x, [0, 0, false])
    const cx = cache.get(x)!
    if (cx[2]) return `${' '.repeat(d)} x${cx[0]}`

    cx[2] = true
    const srcs = isNone(srcfn(x)) ? 'None' : (srcfn(x) || [])
        .map((s) => `\n${prettyPrint(s, rep, srcfn, cache, d + 2)},`)
        .join('')

    return `${' '.repeat(d)}${cx[1] > 1 ? `x${cx[0]}:=` : ''}${rep(x)}`.replace('%s', srcs)
}

// TODO:
// class UOpMetaClass(type):
//   ucache:WeakValueDictionary[Tuple, UOp] = WeakValueDictionary()
//   def __call__(cls, op:Ops, dtype:DType=dtypes.void, src:Tuple[UOp,...]=tuple(), arg:Any=None):
//     if (ret:=UOpMetaClass.ucache.get(key:=(op, dtype, src, arg), None)) is not None: return ret
//     UOpMetaClass.ucache[key] = ret = super().__call__(op, dtype, src, arg)
//     return ret
type UOpInput = { op: Ops; dtype?: DType; src?: UOp[]; arg?: any }
type UOpTuple = [Ops, any, DType, UOpTuple[]]
class UOp extends MathTrait {
    dtype: DType
    op: Ops
    src: UOp[]
    arg: any
    // deno-fmt-ignore
    constructor({ op, dtype=dtypes.void, src=[], arg=null}:UOpInput) {
        super(); this.op = op; this.dtype = dtype; this.src = src; this.arg = arg;
    }
    __reduce__ = () => [UOp, [this.op, this.dtype, this.src, this.arg]] as const
    replace = (args: Partial<UOpInput>) => {
        const oldArgs: UOpInput = { dtype: this.dtype, arg: this.arg, op: this.op, src: this.src }
        const newArgs: UOpInput = { ...oldArgs, ...args }
        return (Object.entries(oldArgs).every(([k, v]) => v === newArgs[k as keyof UOpInput])) ? this : new UOp(newArgs)
    }
    key = (): Buffer => {
        const hash = createHash('sha256')
        hash.update(Buffer.from(JSON.stringify([this.op, this.dtype, this.arg])))
        for (const s of this.src) hash.update(s.key())
        return hash.digest()
    }
    __repr__ = () => prettyPrint(this, (x) => `UOp(${x.op}, ${x.dtype}, arg=${x.argstr()}, src=(%s))`)
    argstr = () => this.op === Ops.REDUCE_AXIS ? `(${this.arg.map((x: any) => x.toString()).join(', ')})` : this.arg
    parents = () => {
        const map = new Map<UOp, null>()
        for (const x of this.src) map.set(x, null)
        for (const x of this.src) for (const [k] of x.parents()) map.set(k, null)
        return map
    }
    sparents = () => new Map([...this.parents().entries(), [this, null]])
    tuplize = (): UOpTuple => [this.op, this.arg, this.dtype, this.src.map((src) => src.tuplize())]

    //   # *** uop shape stuff ***
    // TODO: ignored the shape stuff for now
    // has_st = () => ![Ops.DEFINE_LOCAL, Ops.DEFINE_GLOBAL, Ops.BUFFER, Ops.CONST, Ops.DEFINE_VAR].includes(this.op)
    //   @functools.cached_property
    //   def st(self) -> Optional[ShapeTracker]:
    //     if not self.has_st: return None
    //     if self.op in GroupOp.Buffer: return self.st_arg
    //     if self.op is Ops.VIEW: return self.arg
    //     src_sts = [x.st for x in self.src if x.st is not None]
    //     assert all_same([x.shape for x in src_sts]), f"UOp parents must have the same shape {self} {[x.shape for x in src_sts]}"
    //     from tinygrad.shape.shapetracker import ShapeTracker
    //     return ShapeTracker.from_shape(src_sts[0].reduce(self.axis_arg)) if self.op is Ops.REDUCE_AXIS else src_sts[0]
    //   @functools.cached_property
    //   def full_shape(self) -> Tuple[sint, ...]:
    //     return self.arg.shape if self.op is Ops.VIEW else tuple(smax(x) for x in zip(*[x.full_shape for x in self.src if x.has_st]))

    //   # *** uop evaluation ***

    simplify = (): UOp => {
        // TODO: with Context(TRACK_MATCH_STATS=0):
        return graphRewrite(this, symbolic)
    }
    ssimplify = (): UOp => {
        const ret = this.simplify()
        return ret.op === Ops.CONST ? ret.arg : ret
    }
    _eval = <T extends new (...args: any[]) => void>(dtypes: DType[], expectedType: T): InstanceType<T> => {
        if (!dtypes.includes(this.dtype)) throw new Error(`eval with wrong dtype ${this}`)
        const simpleSelf = this.simplify()
        const [vmin, vmax] = simpleSelf._minMax()
        if (vmin !== vmax) throw new Error(`eval failed to be a single number, range is ${vmin} to ${vmax} in ${simpleSelf.render()}`)
        if ((vmin instanceof expectedType)) throw new Error(`vmin is wrong dtype ${typeof vmin} != ${expectedType}`)
        return vmin as InstanceType<T>
    }
    __bool__ = () => this._eval([dtypes.bool], Boolean)
    __int__ = () => this._eval(dtypes.ints, Number)
    __float__ = () => this._eval(dtypes.floats, Number)
    substitute = (dvars: Map<UOp, UOp>) => {
        // TODO:  with Context(TRACK_MATCH_STATS=0):
        return graphRewrite(this, _substitute, dvars)
    }

    //   # *** uop syntactic sugar ***
    // TODO: returns ShapeTracker
    stArg = () => {
        if (!(GroupOp.Buffer.includes(this.op))) throw new Error(`st_arg called on ${this.op}`)
        const ret = this.src[this.op === Ops.VALID ? 0 : 1]
        if (ret.op !== Ops.VIEW) throw new Error(`st_arg trying to return ${ret}`)
        return ret.arg
    }
    axisArg = (): number[] => {
        if (![Ops.REDUCE_AXIS, Ops.WMMA].includes(this.op)) throw new Error(`axis_arg called on ${this.op}`)
        const ret = this.op === Ops.REDUCE_AXIS ? this.arg[1] : this.arg[7]
        if (!(Array.isArray(ret) && ret.every((x) => typeof x === 'number'))) throw new Error(`axis_arg trying to return ${ret}`)
        return ret
    }
    sink = (...srcs: UOp[]) => new UOp({ op: Ops.SINK, dtype: dtypes.void, src: [this, ...srcs] })
    index = (idx: UOp, valid?: UOp) => new UOp({ op: Ops.INDEX, dtype: this.dtype, src: isNotNone(valid) ? [this, idx, valid] : [this, idx] })
    override constLike = (b: ConstLike<typeof this>): typeof this => UOp.const(this.dtype, b) as typeof this
    broadcast = (count: number) => {
        if (this.dtype.count !== 1) throw new Error(`dtype.count !==1`)
        if (count === 1) return this
        return new UOp({ op: Ops.VECTORIZE, dtype: this.dtype.vec(count), src: range(count).map(() => this) })
    }
    cast = (dtype: DType) => new UOp({ op: Ops.CAST, dtype, src: [this] })
    bitcast = (dtype: DType) => new UOp({ op: Ops.BITCAST, dtype, src: [this] })
    gep = (i: number[] | number) => {
        if (!Array.isArray(i)) {
            // NOTE: these are just shortcuts to not have to create and fold later
            if (this.op === Ops.VECTORIZE) return this.src[i]
            if (this.op === Ops.VCONST) return UOp.const(this.dtype.scalar(), this.arg[i])
            if (this.op === Ops.CONST) return UOp.const(this.dtype.scalar(), this.arg)
            i = [i]
        }
        if (this.dtype.vcount === i.length && i === range(i.length) || this.dtype === dtypes.void) return this
        return new UOp({ op: Ops.GEP, dtype: i.length > 1 ? this.dtype.scalar().vec(i.length) : this.dtype.scalar(), src: [this], arg: i })
    }
    load = (src: UOp[], kwargs?: Record<string, any>) => new UOp({ op: Ops.LOAD, src: [this, ...src], ...kwargs })
    store = (src: UOp[], kwargs?: Record<string, any>) => new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [this, ...src], ...kwargs })
    override alu = (arg: Ops, ...src: typeof this[]): typeof this => {
        let outDType = [this, ...src].at(-1)?.dtype
        if ([Ops.CMPLT, Ops.CMPNE].includes(arg) && isNotNone(outDType)) {
            outDType = outDType.count > 1 ? dtypes.bool.vec(outDType.count) : dtypes.bool
        }
        return new UOp({ op: arg, dtype: outDType, src: [this, ...src] }) as typeof this
    }
    static const = (dtype: DType, b: ConstLike) => {
        if (b instanceof UOp) return b.unbind()[0]
        if (Array.isArray(b) && allSame(b)) b = b[0]
        return new UOp({ op: Array.isArray(b) ? Ops.VCONST : Ops.CONST, dtype, arg: isNotNone(dtype) ? dtypes.asConst(b, dtype) : b })
    }
    //   @staticmethod
    static range = (dtype: DType, start: ConstType<UOp>, end: ConstType<UOp>, idx: number) => {
        return new UOp({ op: Ops.RANGE, dtype: dtype, src: [!(start instanceof UOp) ? UOp.const(dtype, start) : start, !(end instanceof UOp) ? UOp.const(dtype, end) : end], arg: [idx, false] })
    }
    r = (op: Ops, axis: number[]) => new UOp({ op: Ops.REDUCE_AXIS, dtype: this.dtype, src: [this], arg: [op, axis] })
    assign = (x: UOp) => new UOp({ op: Ops.ASSIGN, dtype: this.dtype, src: [this, x] })
    contiguous = () => new UOp({ op: Ops.CONTIGUOUS, dtype: this.dtype, src: [this] })
    isContiguousBase = () => this.op === Ops.CONTIGUOUS && !(this.src[0].base().op === Ops.VIEW && this.src[0].base().src.length === 2)

    //   # *** from LazyBuffer ***

    //   @staticmethod
    //   def const_with_shape(dtype:DType, val:ConstLike, shape:Tuple[sint,...]) -> UOp:
    //     from tinygrad.shape.shapetracker import ShapeTracker
    //     return UOp(Ops.VALID, dtypes.bool, (ShapeTracker.from_shape(()).reshape((1,)*len(shape)).expand(shape).to_uop(),)).where(UOp.const(dtype, val), 0)

    //   # *** uop movement ops ***

    base = () => this.op === Ops.VIEW && this.src.length === 1 ? this.src[0] : this
    //   def view(self, st:ShapeTracker) -> UOp:
    //     assert self.op is not Ops.STORE, "VIEW of STORE is invalid, STORE is always base"
    //     return self if self.st is None or self.st === st else UOp(Ops.VIEW, self.dtype, (self,), st)
    //   def reshape(self, arg:Tuple[sint, ...]) -> UOp: return self.view(unwrap(self.st).reshape(arg))

    //   # *** uop Buffer stuff ***

    static newBuffer = (device: string, size: number, dtype: DType, num = -1) => new UOp({ op: Ops.BUFFER, dtype: dtype.ptr(), src: [], arg: [num, [device, size, dtype]] })
    // deno-fmt-ignore
    device = ():string => {
        switch (this.op){
            case Ops.COPY: return this.arg
            case Ops.BUFFER: return this .arg[1][0]
            default: return this.src[0].device()
        }
    }
    size = (): number => this.bufUOp().arg[1][1]
    bufUOp = () => {
        if (this.op === Ops.BUFFER) return this
        if (!([...GroupOp.Buffer, Ops.ASSIGN, Ops.VIEW].includes(this.op) && this.src[0].op === Ops.BUFFER)) throw new Error(`buf_uop called on ${this.op}`)
        return this.src[0]
    }
    //   # *** uop Variable stuff ***

    static variable = (name: string, minVal: ConstType<UOp>, maxVal: ConstType<UOp>, dtype = dtypes.int) => {
        if (!(!(minVal instanceof UOp) && !(maxVal instanceof UOp))) throw new Error(`can't create Variable ${name} with ${minVal}/${maxVal}`)
        return new UOp({ op: Ops.DEFINE_VAR, dtype, arg: [name, minVal, maxVal] })
    }
    expr = () => {
        if (!(this.op === Ops.DEFINE_VAR)) throw new Error(`op is ${this.op}, need DEFINE_VAR`)
        return this.arg[0]
    }
    bind = (val: number) => {
        if (this.op !== Ops.DEFINE_VAR) throw new Error(`op is ${this.op}, need DEFINE_VAR`)
        if (!(this.arg[1] <= val && val <= this.arg[2])) throw new Error(`bind ${val} not in range [${this.arg[1]}, ${this.arg[2]}]`)
        return new UOp({ op: Ops.BIND, dtype: this.dtype, src: [this, this.constLike(val)] })
    }
    unbind = (): [Variable, number] => {
        if (!(this.op === Ops.BIND && this.src[0].op === Ops.DEFINE_VAR && this.src[1].op === Ops.CONST)) throw new Error(`can't unbind ${this}`)
        return [this.src[0], this.src[1].arg]
    }
    val = () => this.unbind()[1]
    vars = (): Set<UOp> => {
        const sparents = [...this.sparents().keys()]
        const boundVars = new Set(sparents.filter((x) => x.op === Ops.BIND && x.src[0].op === Ops.DEFINE_VAR))
        const boundVarBase = new Set([...boundVars].map((x) => x.src[0]))
        const allVars = new Set(sparents.filter((x) => x.op === Ops.DEFINE_VAR))
        return boundVars.union(new Set([...allVars].filter((x) => !boundVarBase.has(x))))
    }
    variables = (): Variable[] => {
        const stVars: Set<Variable>[] = [...this.sparents().keys()].filter((x) => GroupOp.Buffer.includes(x.op)).map((x) => x.stArg().vars())
        const idk = new Set([...this.vars()].map((x) => x.op !== Ops.DEFINE_VAR ? x.unbind()[0] : x))
        return [...new Set([...stVars.flatMap((x) => [...x]), ...idk])].sort((a, b) => b.arg - a.arg)
    }

    //   # *** uop symbolic stuff ***

    /**largest known int that divides self */
    constFactor = (): number => {
        if (this.op === Ops.CONST) return this.arg
        if (this.op === Ops.VCONST) return mathGcd(...this.arg)
        if (this.op === Ops.ADD) return mathGcd(this.src[0].constFactor(), this.src[1].constFactor())
        if (this.op === Ops.MUL) return this.src[1].op === Ops.CONST ? this.src[0].op === Ops.CONST ? this.src[0].arg : this.src[1].arg : 1
        return 1
    }
    divides = (v: number): UOp | null => {
        if (v === 1) return this
        if (this.op === Ops.CONST) return this.arg % v === 0 ? this.constLike(Math.floor(this.arg / v)) : null
        if (this.op === Ops.VCONST) return this.arg.every((x: number) => x % v === 0) ? this.constLike(this.arg.map((x: number) => Math.floor(x / v))) : null

        const d0 = this.src[0].divides(v)
        const d1 = this.src[1].divides(v)
        if (this.op === Ops.ADD) return isNotNone(d0) && isNotNone(d1) ? d0.add(d1) : null
        if (this.op === Ops.MUL) {
            if (isNotNone(d0)) return d0.mul(this.src[1])
            if (isNotNone(d1)) return this.src[0].mul(d1)
        }
        return null // generic None if we aren't sure
    }

    static min = (...args: UOp[]) => args.reduce((min, current) => min.lt(current) ? min : current)
    static max = (...args: UOp[]) => args.reduce((max, current) => max.gt(current) ? max : current)

    vmin = () => this._minMax()[0]
    vmax = () => this._minMax()[1]
    _minMax = (): [UOp, UOp] => {
        if (GroupOp.Binary.includes(this.op) && !dtypes.isFloat(this.dtype)) {
            const [[s0Vmin, s0Vmax], [s1Vmin, s1Vmax]] = [this.src[0]._minMax(), this.src[1]._minMax()]
            if (this.op === Ops.ADD) return [s0Vmin.add(s1Vmin), s0Vmax.add(s1Vmax)]
            if (this.op === Ops.MUL) {
                const vals = [s0Vmin.mul(s1Vmin), s0Vmin.mul(s1Vmax), s0Vmax.mul(s1Vmin), s0Vmax.mul(s1Vmax)]
                return [UOp.min(...vals), UOp.max(...vals)]
            }
            if (this.op === Ops.MOD && s1Vmin.gt(0)) return [this.constLike(0), s1Vmax.sub(1)]
            if (this.op === Ops.IDIV && s1Vmin === s1Vmax) { // min/max are equal in a CONST
                if (s1Vmin.gt(0)) return [s0Vmin.idiv(s1Vmin), s0Vmax.idiv(s1Vmin)]
                if (s1Vmin.lt(0) && s0Vmin.ge(0)) return [(s0Vmax.idiv(s1Vmin.neg())).neg(), (s0Vmin.idiv(s1Vmin.neg())).neg()]
            }
            if (this.op === Ops.MAX) return [UOp.max(s0Vmin, s1Vmin), UOp.max(s0Vmax, s1Vmax)]
            if (this.op === Ops.CMPLT) return [s0Vmax.lt(s1Vmin), s0Vmin.lt(s1Vmax)]
            if (this.op === Ops.CMPNE) return [(s0Vmax.lt(s1Vmin)).bitwiseOr(s1Vmax.lt(s0Vmin)), (s0Vmin.eq(s0Vmax).bitwiseAnd(s0Vmax.eq(s1Vmin)).bitwiseAnd(s1Vmin.eq(s1Vmax))).neg()]
            if (this.dtype === dtypes.bool) {
                if (this.op === Ops.OR) return [s0Vmin.bitwiseOr(s1Vmin), s0Vmax.bitwiseOr(s1Vmax)]
                if (this.op === Ops.AND) return [s0Vmin.bitwiseAnd(s1Vmin), s0Vmax.bitwiseAnd(s1Vmax)]
            }
        }
        // float has NAN issue and we use explicit NAN in transcendental
        if (this.op === Ops.WHERE && dtypes.isInt(this.dtype)) return [UOp.min(this.src[1].vmin(), this.src[2].vmin()), UOp.max(this.src[1].vmax(), this.src[2].vmax())]
        // NOTE: returned UOp is assumed to be CONST
        if (this.op === Ops.DEFINE_VAR && this.arg) return [this.arg[1], this.arg[2]]
        if (this.op === Ops.RANGE) return [this.src[0].vmin(), (this.src[1].sub(1)).vmax()]
        if (this.op === Ops.BIND) return this.src[0]._minMax() // ignore the bound value
        if ([Ops.EXPAND, Ops.VECTORIZE].includes(this.op)) return [UOp.min(...this.src.map((x) => x.vmin())), UOp.max(...this.src.map((x) => x.vmax()))]
        // TODO: UOps.SPECIAL is UOps.DEFINE_VAR
        if (this.op === Ops.SPECIAL) return [this.constLike(0), typeof this.arg[1] === 'number' ? this.constLike(this.arg[1] - 1) : this.constLike(dtypes.max(this.dtype))]
        if (this.op === Ops.CONST) return [this.arg, this.arg]
        if (this.op === Ops.VCONST) return [UOp.min(this.constLike(this.arg)), UOp.max(this.arg)]
        return [this.constLike(dtypes.min(this.dtype)), this.constLike(dtypes.max(this.dtype))]
    }

    _sym_fxn = (): [(m: Map<UOp, number>) => number, any[]] => {
        const sself = this.simplify()
        const varnames = []
        for (const x of sself.parents().keys()) if (x.op === Ops.DEFINE_VAR) varnames.push(x.arg[0])
        // TODO: check this return type ,used eval before
        return [(m) => {
            return +sself.render()
        }, varnames]
    }
    symInfer = (varVals: Map<UOp, number>) => {
        const [fxn, varnames] = this._sym_fxn()
        const map = new Map<UOp, number>()
        for (const [k, v] of varVals.entries()) if (varnames.includes(k.arg[0])) map.set(k.arg[0], v)
        return fxn(map)
    }

    render = (simplify = true) => {
        const ret = graphRewrite(simplify ? this.simplify() : this, renderer)
        return ret.op === Ops.NOOP ? ret.arg : ret.toString()
    }
}

export class KernelInfo {
    local_dims = 0 // number of local dimensions  (this is remapping RANGE to SPECIAL)
    upcasted = 0 // count that are upcasted     (this is remapping RANGE to EXPAND)
    dont_use_locals = false // don't use local indexing
}

// # ***** ops in python *****
// deno-fmt-ignore
const hookOverflow = <T extends any[]>(dv: number, fxn: (...args: T) => number) =>  (...args: T) => { try { return fxn(...args) } catch { return dv } }

export const pythonAlu: { [key in Ops]?: (...x: number[]) => number } = {
    [Ops.LOG2]: (x) => x === 0 ? x > 0 ? Math.log2(2) : -Infinity : NaN,
    [Ops.EXP2]: hookOverflow(Infinity, (x: number) => 2 ** x),
    [Ops.SQRT]: (x) => x >= 0 ? Math.sqrt(x) : NaN,
    [Ops.RECIP]: (x) => x !== 0 ? 1 / x : x >= 0 ? Infinity : -Infinity,
    [Ops.SIN]: (x) => isFinite(x) ? Math.sin(x) : NaN,
    [Ops.NEG]: (x) => -x,
    [Ops.ADD]: (x, y) => x + y,
    [Ops.SUB]: (x, y) => x - y,
    [Ops.MUL]: (x, y) => x * y,
    [Ops.MOD]: (x, y) => Math.abs(Math.floor(x)) % Math.abs(Math.floor(y)) * (x < 0 ? -1 : 1),
    [Ops.IDIV]: (x, y) => y !== 0 ? Math.floor(Math.abs(x) / Math.abs(y)) * ((x * y < 0) ? -1 : 1) : x * Infinity,
    [Ops.MAX]: (...args) => Math.max(...args),
    [Ops.CMPNE]: (x, y) => Number(x !== y),
    [Ops.CMPLT]: (x, y) => Number(x < y),
    [Ops.XOR]: (x, y) => x ^ y,
    [Ops.OR]: (x, y) => x | y,
    [Ops.AND]: (x, y) => x & y,
    [Ops.SHR]: (x, y) => x >> y,
    [Ops.SHL]: (x, y) => x << y,
    [Ops.MULACC]: (x, y, z) => (x * y) + z,
    [Ops.WHERE]: (x, y, z) => x ? y : z,
}

const execAlu = (op: Ops, dtype: DType, operands: number[], truncateOutput = true): any => {
    if (dtype.count > 1) return range(dtype.count).map((i) => execAlu(op, dtype.scalar(), operands.map((x) => Array.isArray(x) ? x[i] : x)))
    const alu = pythonAlu[op]!(...operands)
    return truncateOutput ? truncate(dtype)(alu) : alu
}
// # ***** uop helpers *****

export const printUOps = (uops: UOp[]) => {
    for (const [i, u] of uops.entries()) {
        const formattedParents = u.src.map((x) => uops.includes(x) ? x.op !== Ops.CONST ? uops.indexOf(x) : `${x.arg}` : '--')
        console.log(`${i.toString().padStart(4)} ${u.op.toString().padEnd(20)} ${u.dtype.toString().padEnd(30)} ${formattedParents.toString().padEnd(32)} ${u.arg}`)
    }
}

export const flopsMem = (uops: UOp[], ignoreIndexing = false): [UOp, UOp] => {
    let flops = uops[0].constLike(0)
    let mem = uops[0].constLike(0)
    let mults = uops[0].constLike(1)
    const multStack: UOp[] = []
    let dontCount = new Set<UOp>()
    if (ignoreIndexing) {
        for (const u of uops) {
            if ([Ops.LOAD, Ops.STORE].includes(u.op)) {
                dontCount = dontCount.union(u.src[0].sparents())
                if (u.src.length > 2) dontCount = dontCount.union(u.src[2].sparents())
            } else if (u.op === Ops.IF) dontCount = dontCount.union(u.src[0].sparents())
        }
    }
    for (const u of uops) {
        if (u.op === Ops.RANGE) {
            multStack.push(mults)
            mults = (u.src[1].sub(u.src[0])).ssimplify().mul(mults)
        } else if (u.op === Ops.ENDRANGE) mults = multStack.pop()!
        else if (u.op === Ops.SPECIAL) mults = u.arg[1].mul(mults) // NOTE: we don't push to the mult_stack here, you can't end these
        else if (u.op === Ops.LOAD) mem = (mults.mul(u.dtype.itemsize)).add(mem)
        else if (u.op === Ops.STORE) mem = (mults.mul(u.src[1].dtype.itemsize)).add(mem)
        else if (GroupOp.ALU.includes(u.op) && !dontCount.has(u)) flops = flops.add((mults.mul(u.op === Ops.MULACC ? 2 : 1)).mul(u.dtype.count))
        else if (u.op === Ops.WMMA && !dontCount.has(u)) flops = mults.mul(Math.floor(prod(u.arg[1]) / u.arg[5])).mul(2).add(flops)
    }
    return [flops, mem]
}
// # ***** pattern matcher *****

const getLocation = (): [string, number] => {
    // const frm = sys._getframe(1)
    // // find the real frame in the file that has the UPat, TODO: is there a better way to do this?
    // while (frm.f_back !== null && ['ops.py', 'uopgraph.py', 'schedule.py', 'lowerer.py', 'cstyle.py'].includes(pathlib.Path(frm.f_back.f_code.co_filename).name)) {
    //     frm = frm.f_back
    // }
    // return frm.f_code.co_filename, frm.f_lineno
    return ['', 0]
}
const lines = (fn: string): string[] => {
    return readFileSync(fn).toString().split('\n')
}

type UPatInput = { op?: Ops | Ops[]; dtype?: DType | DType[]; src?: UPat | UPat[]; arg?: any; name?: string; allowAnyLen?: boolean; location?: any; customEarlyReject?: Ops[] }
class UPat extends MathTrait {
    //   __slots__ = ["op", "dtype", "arg", "name", "src"]
    op?: Ops[]
    dtype?: DType[]
    arg?: any
    name?: string
    _inSrc?: UPat | UPat[]
    customEarlyReject?: Ops[]
    src?: UPat[][]
    allowedLen: number
    location: [string, number]
    earlyReject: Ops[]

    constructor({ op, dtype, arg, location, name, src, allowAnyLen = false, customEarlyReject }: UPatInput) {
        super()
        assert(isNone(op) || !(!Array.isArray(op) && Object.values(Ops).includes(op)) || !(Array.isArray(op) && Object.values(Ops).includes(op[0])), 'op must be Ops or tuple of Ops')
        this.op = Array.isArray(op) ? op : !isNone(op) ? [op] : undefined
        this.dtype = Array.isArray(dtype) ? dtype : !isNone(dtype) ? [dtype] : undefined
        this.arg = arg
        this.name = name
        this._inSrc = src
        this.customEarlyReject = customEarlyReject
        assert(self.name !== 'ctx', "UPat can't be named ctx")

        // try all permutations if it's a list
        if (Array.isArray(src)) this.src = !allSame(src) ? permutations(src) : [src]
        // repeat if it's a UPat
        else if (src instanceof UPat) this.src = [[src]]

        this.allowedLen = allowAnyLen || src instanceof UPat || isNone(src) ? -1 : src.length
        this.location = location || getLocation()

        if (isNotNone(customEarlyReject)) this.earlyReject = customEarlyReject
        else {
            const upatMatch = src instanceof UPat ? [src] : (isNone(src) ? [] : this.src![0])
            this.earlyReject = upatMatch.filter((pp) => isNotNone(pp.op) && pp.op.length === 1).map((pp) => pp.op![0])
        }
    }

    named = (name: string) => new UPat({ op: this.op, dtype: this.dtype, src: this._inSrc, arg: this.arg, name, allowAnyLen: this.allowedLen === -1, customEarlyReject: this.customEarlyReject })

    static any = (src: UPatInput['src']) => new UPatAny({ src: src })

    static var = (name?: string, dtype?: DType | DType[]) => new UPat({ dtype: dtype, name: name })
    static cvar = (name?: string, dtype?: DType, vec = true) => new UPat({ op: vec ? [Ops.CONST, Ops.VCONST] : Ops.CONST, dtype, name })
    static const = (dtype?: DType | DType[], b?: ConstLike) => new UPat({ op: Ops.CONST, dtype: dtype, arg: b })

    //   # copied from UOp
    index = (idx: UPat, valid?: UPat) => new UPat({ op: Ops.INDEX, dtype: this.dtype, src: isNotNone(valid) ? [this, idx, valid] : [this, idx] })
    view = (st?: any, kwargs?: any) => new UPat({ op: Ops.VIEW, dtype: this.dtype, src: [this], arg: st, ...kwargs })
    cast = (dtype?: DType) => new UPat({ op: Ops.CAST, dtype, src: [this] })
    bitcast = (dtype?: DType) => new UPat({ op: Ops.BITCAST, dtype, src: [this] })
    gep = (i: number) => new UPat({ op: Ops.GEP, src: [this], arg: [i] })
    load = (src: UPat[], kwargs?: any) => new UPat({ op: Ops.LOAD, src: [this, ...src], ...kwargs })
    store = (src: UPat[], kwargs?: any) => new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [this, ...src], ...kwargs })
    assign = (x: UPat) => new UPat({ op: Ops.ASSIGN, dtype: this.dtype, src: [this, x] })

    override constLike = (b: ConstLike): typeof this => UPat.const(this.dtype, b) as typeof this
    override alu = (op: Ops, ...src: this[]) => {
        const asrc = [this, ...src]
        return new UPat({ op, dtype: [Ops.CMPLT, Ops.CMPNE].includes(op) ? undefined : asrc.pop()?.dtype, src: GroupOp.Commutative.includes(op) ? asrc : asrc }) as typeof this
    }

    // deno-fmt-ignore
    printable = ():string => {
        try{ return lines(this.location[0])[this.location[1]-1].trim() }
        catch { return "<missing>"}
    }
    __repr__ = (level = 1): string => {
        const rep = (x: UPat) => {
            const op = isNone(x.op) ? 'None' : `(${x.op.map((x) => `Ops.${getEnumString(x)}`).join(', ')})`
            const dtype = x.dtype ? `{${[...new Set(x.dtype)]}}` : 'None'
            const len = x.allowedLen === 0 ? 'True' : 'False'
            const space = range(level * 2).map((x) => ' ').join('')
            const src = x.src ? x.src.flat().length ? `\n${space}${[...new Set(x.src.flat())].map((u) => u.__repr__(level + 1)).join(',\n' + space)},` : '' : 'None'
            const name = x.name ? `'${x.name}'` : 'None'
            const form = `UPat(${op}, ${isNotNone(x.arg) ? x.arg : 'None'}, name=${name}, dtype=${dtype}, allow_any_len=${len}, src=(${src}))`
            return form
        }
        // TODO: not quite right, check it
        return prettyPrint(this, rep, (x) => !x.src ? null : x.src[0])
    }
    match = (uop: UOp, store: Record<string, UOp>): Record<string, UOp>[] => {
        if (
            (isNotNone(this.op) && !this.op.includes(uop.op)) ||
            (isNotNone(this.name) && (store[self.name] || uop) !== uop) ||
            (isNotNone(this.dtype) && !this.dtype.includes(uop.dtype) && !this.dtype.includes(uop.dtype.scalar())) ||
            (isNotNone(this.arg) && this.arg !== uop.arg) ||
            (this.allowedLen !== -1 && uop.src.length !== this.allowedLen)
        ) return []
        if (isNone(this.src)) return [store]
        let res: Record<string, UOp>[] = []
        for (const vp of this.src) {
            let [stores, newStores] = [[{ ...store }], [] as Record<string, UOp>[]]
            for (const [uu, vv] of uop.src.map((uu, i) => [uu, vp[i]] as const)) {
                for (const s of stores) newStores = [...newStores, ...vv.match(uu, s)]
                stores = newStores
                newStores = []
            }
            res = [...res, ...stores]
        }
        return res
    }
}

class UPatAny extends UPat {
    override match = (uop: UOp, store: Record<string, UOp>) => {
        let ret: Record<string, UOp>[] = []
        for (const x of this.src?.[0] || []) {
            const match = x.match(uop, { ...store })
            if (match) ret = [...ret, ...match]
        }
        return ret
    }
}

const deconstructFunction = (fxn: () => void): string[] => {
    //   const newGlobals = {k:v for k,v in fxn.__globals__.items() if k in fxn.__code__.co_names}
    //   for co in fxn.__code__.co_consts:
    //     if isinstance(co, types.CodeType): new_globals.update({k:v for k,v in fxn.__globals__.items() if k in co.co_names})
    //   # NOTE: optional round trip through pickle!
    //   assert fxn.__closure__ is None, "closures are not supported in pattern matchers"
    //   ret = fxn.__code__, new_globals, fxn.__name__, fxn.__defaults__
    //   return ret
    return []
}
type Pattern = [UPat, (...args: any[]) => any]
class PatternMatcher {
    patterns: Pattern[]
    pdict = new Map<Ops, ([UPat, (...args: any[]) => void, Set<any>, boolean][])>()
    constructor(patterns: Pattern[]) {
        this.patterns = patterns
        // NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!
        // for (const [p,fxn] of self.patterns){
        //   if (isNotNone(p.op)) throw new Error("assert")
        //   const tuple_fxn = Array.isArray(fxn) ? fxn : deconstructFunction(fxn)
        //   const real_fxn = types.FunctionType(...tuple_fxn)
        //   for (const uop in p.op) self.pdict.setdefault(uop, []).append((p, real_fxn, p.early_reject, 'ctx' in inspect.signature(real_fxn).parameters))
        // }
    }

    // __reduce__ = () => this.patterns.map(([x, fxn]) => [x, fxn.__name__ === '<lambda>' ? deconstructFunction(fxn) : fxn])

    __add__ = (more: PatternMatcher) => new PatternMatcher([...this.patterns, ...more.patterns])

    rewrite = (uop: UOp, ctx?: any): UOp | null => {
        //     const ler = uop.src.map((u) => u.op)
        //     for (const [p, fxn, early_reject, has_ctx] of this.pdict.get(uop.op, [])!) {
        //         if (!early_reject.issubset(ler)) continue
        //         for (const match of p.match(uop, {})) {
        //             const ret = has_ctx ? fxn(ctx, ...match) : fxn(...match)
        //             if (isNotNone(ret)) return ret
        //         }
        //     }
        return null
    }
}
// # *** tracking pattern matcher ***

// const TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if getenv("VIZ") else 0)
const TRACK_MATCH_STATS = 2
const matchStats = new Map<UPat, number[]>()
const setMap = <K, V>(map: Map<K, V>, key: K, fn: (x: V) => V) => {
    const newVal = fn(map.get(key)!)
    map.set(key, newVal)
    return newVal
}

// @dataclass(frozen=True)
class TrackedRewriteContext {
    loc: [string, number] // location that called graph_rewrite
    sink: UOp // the sink passed into the rewrite
    matches: [UOp, UOp | null, UPat | null, number][] = [] // all matches of sparents
    constructor(loc: [string, number], sink: UOp, matches?: [UOp, UOp | null, UPat | null, number][]) {
        this.loc = loc
        this.sink = sink
        if (matches) this.matches = matches
    }
}
const rewriteStack: [any, TrackedRewriteContext[]][] = []
const contexts: [any, TrackedRewriteContext[]][] = []
const _rewriteCnt: Record<string, number> = {}
const trackRewrites = (named = false) => {
    const _decorator = (func: (...args: any[]) => void) => {
        const __wrapper = (...args: any[]) => {
            if (TRACK_MATCH_STATS >= 2) {
                if (named) _rewriteCnt[func.name] = (_rewriteCnt[func.name] || 0) + 1
                rewriteStack.push([named ? `{(n:=func.__name__)}_{_rewrite_cnt[n]}` : this, []])
            }
            let ret
            try {
                ret = func(...args)
            } finally { // NOTE: save everything in the stack
                if (TRACK_MATCH_STATS >= 2) contexts.push(rewriteStack.pop()!)
            }
            return ret
        }
        return __wrapper
    }
    return _decorator
}

export class TrackedPatternMatcher extends PatternMatcher {
    constructor(patterns: Pattern[]) {
        super(patterns)
        for (const [p] of this.patterns) {
            if (!matchStats.has(p)) matchStats.set(p, [0, 0, 0.0, 0.0])
        }
    }
    override rewrite = (uop: UOp, ctx?: any): UOp | null => {
        const ret = null
        const ler = uop.src.map((u) => u.op)
        for (const [p, fxn, earlyReject, hasCtx] of this.pdict.get(uop.op)!) {
            const st = performance.now()
            if (!isSubset(earlyReject, new Set(ler))) {
                setMap(matchStats, p, (o) => [o[0], o[1], o[2] + (performance.now() - st), o[3]])
                continue
            }
            const old = matchStats.get(p)!
            matchStats.set(p, [old[0], old[1] + 1, old[3], old[4]])
            for (const match of p.match(uop, {})) {
                const ret: any = hasCtx ? fxn({ ctx, ...match }) : fxn(match)
                if (isNotNone(ret)) {
                    const et = performance.now() - st
                    setMap(matchStats, p, (o) => [o[0] + 1, o[1], o[2], o[3] + et])
                    if (TRACK_MATCH_STATS >= 3) console.log(`${(et * 1e6).toFixed(2)} us -- `, p.printable())
                    if (TRACK_MATCH_STATS >= 2 && rewriteStack.length !== 0 && ret instanceof UOp) rewriteStack.at(-1)?.at(1).at(-1).matches.append([uop, ret, p, et])
                    return ret // NOTE: if it returns None, we keep trying to match
                }
            }
            setMap(matchStats, p, (o) => [o[0], o[1], o[2] + performance.now() - st, o[3]])
        }
        if (TRACK_MATCH_STATS >= 2 && rewriteStack.length !== 0) rewriteStack.at(-1)!.at(1).at(-1).matches.append([uop, ret, null, 0])
        return null
    }
}

// TODO: later
// if TRACK_MATCH_STATS:
//   PatternMatcher = TrackedPatternMatcher  # type: ignore
//   import atexit
//   @atexit.register
//   def print_match_stats():
//     if TRACK_MATCH_STATS >= 2:
//       with open(fn:=temp("rewrites.pkl"), "wb") as f:
//         print(f"rewrote {len(contexts)} graphs and matched {sum(len(r.matches) for _,x in contexts for r in x)} times, saved to {fn}")
//         pickle.dump(contexts, f)
//     if getenv("VIZ"):
//       os.environ["VIZ"] = "0"
//       os.execv(sys.executable, [sys.executable] + [os.path.join(os.path.dirname(__file__), ".", "viz", "serve.py"), temp("rewrites.pkl")])
//     if getenv("PRINT_MATCH_STATS", 1):
//       ret = [0,0,0.0,0.0]
//       for k,v in sorted(list(match_stats.items()), key=lambda x: x[1][2]+x[1][3]):
//         loc_str = f"{k.location[0].split('/')[-1]}:{k.location[1]}"
//         if v[1] !== 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
//         ret = [x+y for x,y in zip(ret, v)]
//       print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")

// # *** simple graph rewrite engine ***

class RewriteContext {
    pm: PatternMatcher
    ctx: any
    replace = new Map<UOp, UOp>()
    constructor(pm: PatternMatcher, ctx: any) {
        this.pm = pm
        this.ctx = ctx
    }
    rewrite = (n: UOp): UOp => {
        const rn = this.replace.get(n)
        if (isNotNone(rn)) return rn
        const newSrc = n.src.map((x) => this.rewrite(x))
        const newN = newSrc === n.src ? this.pm.rewrite(n, this.ctx) : new UOp({ op: n.op, dtype: n.dtype, src: newSrc, arg: n.arg })
        const ret = isNone(newN) ? n : this.rewrite(newN)
        this.replace.set(n, ret)
        return ret
    }
}
const graphRewrite = (sink: UOp, pm: PatternMatcher, ctx?: Map<UOp, UOp>): UOp => {
    if (TRACK_MATCH_STATS >= 2 && rewriteStack.length !== 0) {
        // TODO fix this
        const frm = { fCode: { coFilename: 'idk.ts' }, fLineno: 2 }
        rewriteStack.at(-1)?.at(1).append(new TrackedRewriteContext([frm.fCode.coFilename, frm.fLineno], sink))
    }
    return new RewriteContext(pm, ctx).rewrite(sink)
}
// # ***** uop type spec *****

// # this is the matcher for the final rendered UOps
// # matcher functions returns True or False (or None to not match)
export const spec = new PatternMatcher([
    [new UPat({ op: Ops.DEFINE_GLOBAL, name: 'x' }), (x) => (x.dtype instanceof PtrDType || x.dtype instanceof ImageDType) && !x.dtype.local],
    [new UPat({ op: Ops.DEFINE_LOCAL, name: 'x' }), (x) => x.dtype instanceof PtrDType && x.dtype.local],
    [new UPat({ op: Ops.DEFINE_ACC, src: [UPat.var('c')], name: 'x', allowAnyLen: true }), (x, c) => x.src.slice(1).every((y: any) => y.op === Ops.RANGE) && c.dtype === x.dtype],
    [new UPat({ op: Ops.DEFINE_VAR, src: [], name: 'x' }), (x) => typeof x.arg[1] === 'number' && typeof x.arg[2] === 'number'],
    [new UPat({ op: Ops.RANGE, src: [new UPat({ name: 'x' }), new UPat({ name: 'y' })], name: 'rng' }), (rng, x, y) => rng.dtype === x.dtype && x.dtype === y.dtype],
    [new UPat({ op: Ops.SPECIAL, src: [] }), () => true],

    //   # TODO: confirm the args of both of these are shapetrackers
    [new UPat({ op: Ops.VIEW, dtype: dtypes.void, src: [] }), () => true],
    [new UPat({ op: Ops.VIEW, src: [UPat.var('src')], name: 'x' }), (x, src) => src.op !== Ops.STORE && x.dtype === src.dtype],
    [new UPat({ op: Ops.VALID, dtype: dtypes.bool, src: [new UPat({ op: Ops.VIEW })] }), () => true],
    [new UPat({ op: Ops.CONST, name: 'x' }), (x) => x.dtype === x.dtype.scalar() && typeof x.arg === typeof dtypes.asConst(x.arg, x.dtype)],

    //   # early LOAD has a <buf, shapetracker, store?>
    [new UPat({ op: Ops.LOAD, src: [new UPat({ op: [Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL] }), new UPat({ op: Ops.VIEW })] }), () => true],
    [new UPat({ op: Ops.LOAD, src: [new UPat({ op: [Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL] }), new UPat({ op: Ops.VIEW }), new UPat({ op: Ops.STORE })] }), () => true],
    //   # early STORE has a <buf, shapetracker, val>
    [new UPat({ op: Ops.STORE, src: [new UPat({ op: [Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL] }), new UPat({ op: Ops.VIEW }), new UPat({})] }), () => true],
    //   # **** new style load/store ****

    //   # INDEX is used in new style load/store
    [new UPat({ op: Ops.INDEX, src: [new UPat({ op: [Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL] }), new UPat({})] }), () => true],
    //   # LOAD takes a <bufidx, alt?, gate?, barrier?>
    [new UPat({ op: Ops.LOAD, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] })] }), () => true],
    [new UPat({ op: Ops.LOAD, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] }), new UPat({ op: [Ops.IF, Ops.BARRIER] })] }), () => true],
    [new UPat({ op: Ops.LOAD, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] }), new UPat({ name: 'alt' }), new UPat({ dtype: dtypes.bool })], name: 'ld' }), (ld, alt) => ld.dtype === alt.dtype],

    //   # STORE takes a <bufidx, val, gate?>
    [new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] }), new UPat({})] }), () => true],
    [new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] }), new UPat({}), new UPat({ dtype: dtypes.bool })] }), () => true],
    [new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] }), new UPat({}), new UPat({ op: Ops.IF })] }), () => true],

    //   # most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
    [new UPat({ op: Ops.WHERE, name: 'w', src: [new UPat({ dtype: dtypes.bool }), new UPat({ name: 'x' }), new UPat({ name: 'y' })] }), (w, x, y) => w.dtype === x.dtype && x.dtype === y.dtype],
    [new UPat({ op: [Ops.CMPLT, Ops.CMPNE], dtype: dtypes.bool, src: [new UPat({ name: 'x' }), new UPat({ name: 'y' })] }), (x, y) => x.dtype === y.dtype],

    //   # and SHL/SHR, the shift distance can be an int
    [new UPat({ op: [Ops.SHL, Ops.SHR], src: [new UPat({ name: 'x' }), new UPat({ name: 'y' })], name: 'a' }), (a, x, y) => a.dtype === x.dtype && [x.dtype, dtypes.uint].includes(y.dtype)],
    [new UPat({ op: Ops.IDIV, name: 'x' }), (x) => dtypes.isInt(x.dtype) ? null : false],
    [new UPat({ op: GroupOp.ALU, name: 'x' }), (x) => x.src!.every((y: any) => x.dtype === y.dtype)],
    [new UPat({ op: Ops.ASSIGN, src: [new UPat({ op: [Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL] }), new UPat({})] }), () => true],
    [new UPat({ op: Ops.ENDRANGE, dtype: dtypes.void, src: [new UPat({ op: Ops.RANGE })] }), () => true],

    //   # all WMMA has 3 args, <x, w, acc>
    [new UPat({ op: Ops.WMMA, src: [new UPat({}), new UPat({}), new UPat({})] }), () => true],
    [new UPat({ op: Ops.CONTRACT, name: 'x' }), (x) => x.dtype.count === prod(x.arg.map((y: any) => y[1]))],
    [new UPat({ op: Ops.EXPAND, name: 'x' }), (x) => x.src![0].dtype.count === prod(x.arg.map((y: any) => y[1]))],

    //   # if has a <gate, barrier?>

    [new UPat({ op: Ops.IF, dtype: dtypes.void, src: [new UPat({})] }), () => true],
    [new UPat({ op: Ops.IF, dtype: dtypes.void, src: [new UPat({}), new UPat({ op: Ops.BARRIER })] }), () => true],
    [new UPat({ op: Ops.ENDIF, dtype: dtypes.void, src: [new UPat({ op: Ops.IF })] }), () => true],
    [new UPat({ op: Ops.REDUCE_AXIS, name: 'x' }), (x) => Array.isArray(x.arg) && x.arg.length === 2 && [Ops.ADD, Ops.MUL, Ops.MAX].includes(x.arg[0])],
    [new UPat({ op: Ops.GEP, src: [new UPat({ name: 'src' })], name: 'gep' }), (gep, src) => gep.dtype === src.dtype.scalar()],
    [new UPat({ op: Ops.VECTORIZE, name: 'x' }), (x) => x.src!.length > 1 && x.src!.length === x.dtype.count && x.src!.every((y: any) => x.dtype === y.dtype.vec(x.src?.length))],
    [new UPat({ op: [Ops.BITCAST, Ops.CAST], src: [new UPat({})], name: 'x' }), (x) => isNone(x.arg)],
    [new UPat({ op: Ops.BARRIER, dtype: dtypes.void, src: new UPat({ op: Ops.STORE, allowAnyLen: true }) }), () => true], // NOTE: all pointers must be local
    //   # NOTE: for testing, we let sinks be anything
    // [new UPat({ op: UOps.SINK, src: new UPat({ op: UOps.STORE }) }), () => true],
    [new UPat({ op: Ops.SINK, dtype: dtypes.void }), () => true],
    [new UPat({ op: Ops.NOOP }), () => true],

    //   # PTX LOAD/STORE
    [new UPat({ op: [Ops.LOAD, Ops.STORE], src: [new UPat({ dtype: dtypes.int64 })], allowAnyLen: true }), () => true],
    [new UPat({ op: Ops.BARRIER, dtype: dtypes.void, src: new UPat({ op: Ops.STORE, src: [new UPat({ dtype: dtypes.int64 })], allowAnyLen: true }) }), () => true],
])

export const typeVerify = (uops: UOp[]) => {
    for (const [i, u] of uops.entries()) {
        if (!spec.rewrite(u)) {
            printUOps(uops)
            throw new Error(`UOp verification failed at ${i} on ${u.op} ${u.dtype} ${u.src.length} ${u.src.map((x) => x.op)} ${u.arg}`)
        }
    }
}
// # *** uop helpers ***

// def cast_float_to_bf16(x: UOp) -> UOp:
//   assert x.dtype === dtypes.float, "cast float -> bf16 must start with float"
//   x = x.bitcast(dtypes.uint)
//   x = (-x & 0x7f800000).where(x + ((x >> 16) & 1) + 0x7fff, (x & 0xffff).where((x | 0x10000), x))
//   return (x >> 16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)

// # *** most of symbolic lives here now ***

export function* splitUOp(x: UOp, sep: Ops): Generator<UOp> {
    if (x.op === sep) { for (const s of x.src) yield* splitUOp(s, sep) }
    else yield x
}

export const modFolding = (x: UOp, c: number): UOp | null => {
    // simplify x % c, None means no change

    // simple cancel mod case
    const quotient = x.vmin().idiv(c)
    if (0 < c && x.vmin().ge(0) && quotient === x.vmax().idiv(c)) return x.sub(quotient).mul(c)

    let [remainder, somethingChanged] = [[], false] as [UOp[], boolean]
    for (const u of splitUOp(x, Ops.ADD)) {
        const factor = u.constFactor()
        if (factor % c !== factor) {
            const divides = u.divides(factor)?.mul(factor % c)
            if (isNone(divides)) throw new Error('assert')
            remainder.push(divides)
            somethingChanged = true
        } else if (u.op === Ops.MOD && u.src[1].op === Ops.CONST && u.src[1].arg % c === 0) {
            remainder.push(u.src[0])
            somethingChanged = true
        } else remainder.push(u)
    }
    if (!somethingChanged) return null
    return remainder ? remainder.reduce((prev, curr) => prev.add(curr)).mod(c) : x.constLike(0)
}

export const divFolding = (x: UOp, c: number): UOp | null => {
    // simplify x // c, None means no change

    // simple cancel div case
    if (x.vmin().ge(0) && x.vmax().lt(c)) return x.constLike(0)

    let [quotient, remainder, remConst, somethingChanged, gcd, divisor] = [[] as UOp[], [] as UOp[], 0, false, c, 1]
    for (const u of splitUOp(x, Ops.ADD)) {
        if (u.op === Ops.CONST) {
            // add all const together first
            if (remConst !== 0) somethingChanged = true
            remConst += u.arg
        } else {
            const factor = u.constFactor()
            if (factor % c === 0) {
                if (factor) {
                    const divides = u.divides(c)
                    if (isNone(divides)) throw new Error('assert')
                    quotient.push(divides)
                }
                somethingChanged = true
            } else {
                // divisor is the smallest common divisor of all MULs
                if (u.op === Ops.MUL && factor > 1 && c % factor === 0 && (divisor === 1 || divisor > factor)) divisor = factor
                remainder.push(u)
                gcd = mathGcd(gcd, factor)
            }
        }
    }
    // handle the const
    if (remConst % c !== remConst) {
        somethingChanged = true
        quotient.push(x.constLike(remConst).idiv(c))
        remConst = remConst % c
    }
    if (remConst !== 0) remainder.push(x.constLike(remConst))

    // x // c -> quotient + (remainder // div) // (c // div)
    const div = gcd > 1 ? gcd : divisor

    if (!somethingChanged) {
        const newx = divFolding(x, div)
        return 1 < div && div < c && isNotNone(newx) ? newx.idiv(x.constLike(c).idiv(div)) : null
    }
    const rem = remainder ? remainder.reduce((prev, curr) => prev.add(curr)) : null
    const quo = quotient ? quotient.reduce((prev, curr) => prev.add(curr)) : null
    if (isNone(quo)) return isNone(rem) ? x.constLike(0) : divFolding(rem, div)?.idiv(x.constLike(c).idiv(div)) || null
    return isNone(rem) ? quo : divFolding(rem, div)?.idiv(x.constLike(c).idiv(div)).add(quo) || null
}

const ltFolding = (x: UOp, c: number): UOp | null => {
    const [p, np] = partition(splitUOp(x, Ops.ADD).toArray(), (u) => u.constFactor() === 1)
    const d = mathGcd(...np.map((u) => u.constFactor()), c)
    if (np && d > 1 && p.map((u) => u.vmin()).reduce((p, c) => c.add(p)).ge(0) && p.map((u) => u.vmax()).reduce((p, c) => p.add(c)).lt(d)) {
        return np.reduce((p, c) => p.add(c)).divides(d)?.lt(Math.floor(c / d)) || null
    }
    return null
}
const foldUnrolledDivs = (divs: UOp) => {
    // div pattern in unrolled arange
    // example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
    let [addChain, denominator, seenConst, ans] = [splitUOp(divs, Ops.ADD), null as number | null, [] as number[], null as null | UOp]
    for (const u of addChain) {
        if (!(u.op === Ops.IDIV && u.src[1].op === Ops.CONST)) return null
        if (isNone(denominator)) denominator = u.src[1].arg
        if (denominator !== u.src[1].arg) return null
        // assumed CONST is the last of an ADD
        let s0 = u.src[0]
        if (s0.op === Ops.ADD && s0.src[1].op === Ops.CONST && s0.src[1].op === Ops.CONST) {
            seenConst.push(s0.src[1].arg)
            s0 = s0.src[0]
        } else seenConst.push(0)
        if (isNone(ans)) ans = s0
        if (ans !== s0) return null
    }
    if (isNone(denominator)) return null
    // the first (denominator-len(seen_const)) terms may have been folded to 0 already
    for (const i of range(denominator - seenConst.length)) {
        if (isNotNone(ans) && ans.vmin().ge(0) && ans.vmax().add(i).lt(denominator)) seenConst.push(i)
    }
    return isNotNone(ans) && seenConst.sort((a, b) => b - a) === range(denominator) ? ans : null
}
const canonicalizeSimplex = (X: UOp): UOp | null => {
    // (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
    // returns x0 + x1 + ... in such case, or None if not
    let [changed, ret] = [false, [] as UOp[]]
    for (let u of splitUOp(X, Ops.ADD)) {
        // assumed the const is the last src of MUL
        if (u.op === Ops.MUL && u.src[1].op === Ops.CONST && u.src[1].arg > 0) {
            changed = true
            u = u.src[0]
        }
        if (!(GroupOp.Irreducible.includes(u.op) && u.vmin().ge(0))) return null
        ret.push(u)
    }
    return changed ? ret.reduce((p, c) => p.add(c)) : null
}

const isIncreasing = (f: UOp): boolean => {
    // is f a monotonically increasing function regards its input
    if (GroupOp.Irreducible.includes(f.op)) return true
    if (f.op === Ops.ADD) return isIncreasing(f.src[0]) && isIncreasing(f.src[1])
    if ([Ops.MUL, Ops.IDIV].includes(f.op) && f.src[1].op === Ops.CONST && f.src[1].arg >= 0) return isIncreasing(f.src[0])
    return false // False if not sure
}

const parseValid = (valid: UOp): [UOp, boolean, number] => {
    // if it's X <= c, returns X, True, c
    // if it's X >= c, returns X, False, c

    // (X < c).ne(True) -> X >= c
    const s0 = valid.src[0]
    if (valid.op === Ops.CMPNE && valid.src[1].op === Ops.CONST && valid.src[1].arg === 1 && s0.op === Ops.CMPLT && s0.src[1].op === Ops.CONST) return [s0.src[0], false, s0.src[1].arg]
    // X < c -> X <= c-1
    if (valid.op === Ops.CMPLT && valid.src[1].op === Ops.CONST) return [valid.src[0], true, valid.src[1].arg - 1]
    throw new Error(`not able to parse ${valid}`)
}

const uopGivenValid = (valid: UOp, uop: UOp): UOp | null => {
    // return None if valid is always False, otherwise the simplified uop (might be the same as input)

    // first, parse valid into {expr: (lower_bound, upper_bound)}
    const bounds = new Map<UOp, ConstType[]>()
    for (const stmt of splitUOp(valid, Ops.AND)) {
        try {
            const [expr, isUpper, c] = parseValid(stmt)
            setMap(bounds, expr, (o) => o.map((o, i) => i === Number(isUpper) ? c : o))
        } catch {
            return uop
        } // give up if we cannot parse the valid
    }
    // simplify uop given that valid is True
    for (const [expr, v] of bounds.entries()) {
        // some expr has lower bound > upper bound -> valid is an empty set and we return None
        if (isNotNone(v[0]) && isNotNone(v[1]) && v[0] > v[1]) return null

        // every candidate is a set of contrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
        const candidates: [UOp, UOp][][] = []
        if (expr.op === Ops.ADD && v[0] === 1 && splitUOp(expr, Ops.ADD).every((u) => GroupOp.Irreducible.includes(u.op))) {
            // if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
            candidates.push(splitUOp(expr, Ops.ADD).toArray().map((Xi) => [Xi, UOp.variable('fake', 1, Xi.vmax(), Xi.dtype)]))
        }
        // try checking the whole clause
        if ([...uop.sparents().keys()].includes(expr)) {
            candidates.push([[expr, UOp.variable('fake', isNone(v[0]) ? expr.vmin() : v[0], isNone(v[1]) ? expr.vmax() : v[1], expr.dtype)]])
        }
        for (const candidate of candidates) {
            // if every branch in candidate gives the same simplified uop, we can rewrite the uop
            const newuops = candidate.map(([X, newX]) => uop.substitute(new Map([[X, newX]])).simplify().substitute(new Map([[X, newX]])).simplify())
            if (uop.op === Ops.VECTORIZE && uop.src.length === 2) {
                if (allSame(newuops.map((uops) => uops.src[0]))) uop = uop.replace({ src: [newuops[0].src[0], uop.src[1]] })
                if (allSame(newuops.map((uops) => uops.src[1]))) uop = uop.replace({ src: [uop.src[0], newuops[0].src[1]] })
            } else if (allSame(newuops)) uop = newuops[0]
        }
    }
    return uop
}

const _validPriority = (v: UOp, valids: UOp[]): number => {
    // we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
    try {
        return valids.map((other) => other.parents().has(parseValid(v)[0]) ? -1 : 0 as number).reduce((p, c) => p + c)
    } catch {
        return 0
    }
}
export const simplifyValid = (valid: UOp): UOp | null => {
    const ret: UOp[] = []
    let somethingChanged = false
    const valids = splitUOp(valid, Ops.AND).toArray()
    for (const stmt of valids.sort((a, b) => _validPriority(b, valids) - _validPriority(a, valids))) {
        const newstmt = uopGivenValid(ret.reduce((p, c) => p.bitwiseAnd(c)), stmt)
        ret.push(ret && isNotNone(newstmt) ? newstmt : stmt)
        if (ret[-1] !== stmt) somethingChanged = true
    }
    return somethingChanged ? ret.reduce((p, c) => p.bitwiseAnd(c)) : null
}
export const maxVarConst = (x: UOp, c1: UOp, c2: UOp) => {
    if (x.vmin().ge(0)) return c1.arg >= c2.arg ? x.mul(c1) : x.mul(c2)
    if (x.vmax().le(0)) return c1.arg >= c2.arg ? x.mul(c2) : x.mul(c1)
}
export const sintToUPp = (x: sint) => typeof x === 'number' ? UOp.const(dtypes.int, x) : x

export const symbolicSimple = new PatternMatcher([
    //   // ** self folding **
    [UPat.const(undefined, 0).add(UPat.var('x')), (x) => x], // x+0 -> x
    [UPat.var('x').mul(1), (x) => x], // x*1 -> x
    [UPat.var('x').idiv(UPat.var('x')), (x) => x.constLike(1)], // x//x -> 1
    [UPat.var('x').idiv(1), (x) => x], // x//1 -> x
    [UPat.var('x').idiv(-1), (x) => -x], // x//-1 -> -x
    [UPat.var('x').div(UPat.var('x')), (x) => x.constLike(1)], // x/x -> 1
    [(UPat.var('x').mul(UPat.var('x2'))).div(UPat.var('x2')), (x, x2) => x], // (x*x2)/x2 -> x
    [(UPat.var().mod(UPat.var('y'))).named('base').mod(UPat.var('y')), (base, y) => base], // (x%y)%y = -> x%y (rewritten with base for speed)
    [UPat.var('x').mod(UPat.cvar('c')).add(UPat.var('x').idiv(UPat.cvar('c'))).mul(UPat.cvar('c')), (x, c) => x], // (x%c)+(x//c)*c = x
    [UPat.var('x', dtypes.bool).bitwiseAnd(UPat.cvar('c', undefined, false)), (x, c) => c.arg ? x : c],
    [UPat.var('x', dtypes.bool).bitwiseOr(UPat.cvar('c', undefined, false)), (x, c) => c.arg ? c : x],
    [UPat.var('x').maximum(UPat.var('x')), (x) => x],
    [UPat.var('x').bitwiseAnd(UPat.var('x')), (x) => x],
    [UPat.var('x').bitwiseOr(UPat.var('x')), (x) => x],
    [UPat.var('x', dtypes.bool).logicalNot().logicalNot(), (x) => x],
    [UPat.var('x', dtypes.bool).where(UPat.const(dtypes.bool, true), UPat.const(dtypes.bool, false)), (x) => x],
    //   // ** zero folding **
    [UPat.var('x').lt(UPat.var('x')), (x) => UOp.const(dtypes.bool.vec(x.dtype.count), false)], // x < x -> False
    [UPat.var('x', dtypes.ints).ne(UPat.var('x', dtypes.ints)), (x) => UOp.const(dtypes.bool.vec(x.dtype.count), false)], // x != x -> False (only ints)
    //   // x*0 -> 0 or 0*x -> 0
    //   // if x is nan or inf it should render the nan value.
    //   // NOTE: this can be wrong for loaded NaN
    [UPat.var('x').mul(0), (x) => x.constLike(typeof x.arg === 'number' && (isNaN(x.arg) || !isFinite(x.arg)) ? NaN : 0)],
    //   // ** constant folding **
    [new UPat({ op: GroupOp.ALU, name: 'a', src: new UPat({ op: [Ops.VCONST, Ops.CONST] }) }), (a) => a.constLike(execAlu(a.op, a.dtype, a.src.map((x: any) => x.arg), false))],
    //   // bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
    [UPat.var('x', dtypes.bool).mul(UPat.var('y', dtypes.bool)), (x, y) => x.bitwiseAnd(y)],
    [UPat.var('x', dtypes.bool).add(UPat.var('y', dtypes.bool)), (x, y) => x.bitwiseOr(y)],
    [UPat.var('x', dtypes.bool).maximum(UPat.var('y', dtypes.bool)), (x, y) => x.bitwiseOr(y)],
    //   // *** cast ***
    [new UPat({ op: Ops.CAST, name: 'root', src: UPat.cvar('c') }), (root, c) => root.constLike(c.arg)],
    [new UPat({ op: Ops.CAST, name: 'root' }), (root) => root.dtype === root.src![0].dtype ? root.src![0] : null],
])

export const symbolic = symbolicSimple.__add__(
    new PatternMatcher([
        //   // ** COMMUTATIVE flipping **
        [new UPat({ op: GroupOp.Commutative, name: 'x' }), (x) => x.src[1].tuplize < x.src[0].tuplize ? x.replace([...x.src].reverse()) : null],
        //   // group like
        [(UPat.var('x').add(UPat.var('y'))).add(UPat.var('x').mul(UPat.cvar('c'))), (x, y, c) => (x.add(x.mul(c))).add(y)],
        //   // ** boolean algebra **
        [UPat.var('x').bitwiseOr(UPat.var('x').bitwiseAnd(UPat.var())), (x) => x], // x|(x&y) -> x
        //   // ** combine terms **
        [UPat.var('x').mul(UPat.cvar('c0')).add(UPat.var('x').mul(UPat.cvar('c1'))), (x, c0, c1) => x.mul(c0.plus(c1))], // (x*c0)+(x*c1) -> x*(c0+c1)
        [UPat.var('x').add(UPat.var('x').mul(UPat.cvar('c'))), (x, c) => x.mul(c.add(1))], // (x+x*c)-> x*(c+1)
        [UPat.var('x').add(UPat.var('x')), (x) => x.mul(2)], // (x+x)-> x*2
        [(UPat.var('x').div(UPat.var('x2'))).div(UPat.var('x3')), (x, x2, x3) => x.div(x2.mul(x3))], // (x/x2)/x3 -> x/(x2*x3)
        [UPat.var('x').mul(-1, true).add(UPat.cvar('c')), (x, c) => x.neg().add(c.neg())], // -(x+c) -> -x + -c
        //   // a conditional with the same results either way is a noop, also fold const conditionals
        [UPat.var().where(UPat.var('val'), UPat.var('val')), (val) => val],
        [UPat.cvar('gate', undefined, false).where(UPat.var('c0'), UPat.var('c1')), (gate, c0, c1) => gate.arg ? c0 : c1],
        //   // ALU min==max -> CONST (slow!)
        [new UPat({ op: GroupOp.ALU, name: 'x' }), (x) => x.vmin === x.vmax ? x.constLike(x.vmin) : null],
        //   // max folding
        [UPat.var('x').maximum(UPat.var('y')), (x, y) => x.vmax <= y.vmin ? x.vmin >= y.vmax ? x : y : null],
        //   // TODO: why does this rule break beautiful_mnist?
        //   //((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
        [UPat.var('x').mul(UPat.cvar('c1')).maximum(UPat.var('x').mul(UPat.cvar('c2'))), maxVarConst],
        //   // ** two stage ALU folding **
        [(UPat.var('x').add(UPat.cvar('c1'))).add(UPat.cvar('c2')), (x, c1, c2) => x.add(c1.add(c2))],
        [(UPat.var('x').mul(UPat.cvar('c1'))).mul(UPat.cvar('c2')), (x, c1, c2) => x.mul(c1.mul(c2))],
        [(UPat.var('x').bitwiseAnd(UPat.cvar('c1'))).bitwiseAnd(UPat.cvar('c2')), (x, c1, c2) => x.bitwiseAnd(c1.bitwiseAnd(c2))],
        [(UPat.var('x').bitwiseOr(UPat.cvar('c1'))).bitwiseOr(UPat.cvar('c2')), (x, c1, c2) => x.bitwiseOr(c1.bitwiseOr(c2))],
        [(UPat.cvar('c0').add(UPat.var('x'))).lt(UPat.cvar('c1')), (x, c0, c1) => x.lt(c1.sub(c0))], // c0 + x < c1 -> x < c1 - c0
        [(UPat.var('x').idiv(UPat.cvar('c1'))).idiv(UPat.cvar('c2')), (x, c1, c2) => x.idiv(c1.add(c2))], // (x//c1)//c2 -> x//(c1*c2)
        //   // ** lt **
        //   // c0*x<c1 for positive int c0,c1
        [(UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)), (x, c0, c1) => c0.arg > 0 && c1.arg > 0 ? x.lt(Math.ceil(c1.arg / c0.arg)) : null],
        //   // c0*x<c1 for negative int c0 and non-positive c1
        [(UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)), (x, c0, c1) => c0.arg < 0 && c0.arg !== -1 && c1.arg <= 0 ? x.neg().lt(-Math.floor(-c1.arg / -c0.arg)) : null],
        //   // x//c0<c1 for positive int c0
        [(UPat.var('x', dtypes.ints).idiv(UPat.cvar('c0', undefined, false))).lt(UPat.cvar('c1', undefined, false)), (x, c0, c1) => c0.arg > 0 ? x.lt(c1.arg * c0.arg) : null],
        //   // mul add lt
        [((UPat.cvar('c0', undefined, false).mul(UPat.var('x'))).add(UPat.var('x2'))).lt(UPat.cvar('c1', undefined, false)), (x, x2, c0, c1) => c1.arg % c0.arg === 0 && c0.arg > x2.vmax && x2.vmin >= 0 ? x.lt(c1.idiv(c0)) : null],
        //   // ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
        [new UPat({ op: Ops.ADD, src: [UPat.var('x'), UPat.cvar('c1')] }).add(UPat.var('y')), (x, c1, y) => (x.add(y)).add(c1)],
        [new UPat({ op: Ops.MUL, src: [UPat.var('x'), UPat.cvar('c1')] }).mul(UPat.var('y')), (x, c1, y) => (x.mul(y)).mul(c1)],
        //   // *** rules from symbolic ***
        //   // unrolled arange div folding
        [new UPat({ op: Ops.ADD, name: 'divs', src: [new UPat({}), new UPat({ op: Ops.IDIV })] }), foldUnrolledDivs],
        //   // generic lt folding
        [UPat.var('x', dtypes.sints).lt(UPat.cvar('c', undefined, false)), (x, c) => 0 < c.arg ? ltFolding(x, c.arg) : null],
        //   // canonicalize a simplex with positive coefficients > 0
        //   // not x < 1 -> X > 0
        [UPat.var('x', dtypes.ints).lt(1).ne(true), (x) => {
            const newx = canonicalizeSimplex(x)
            return isNotNone(newx) ? newx.lt(1).ne(true) : null
        }],
        //   // ** div **
        //   // // div folding
        [UPat.var('x', dtypes.sints).idiv(UPat.cvar('c', undefined, false)), (x, c) => {
            const newx = divFolding(x, c.arg)
            return 0 < c.arg && isNotNone(newx) ? newx : null
        }],
        //   // ** mod **
        //   // mod folding
        [UPat.var('x').mod(UPat.cvar('c', undefined, false)), (x, c) => {
            const newx = modFolding(x, c.arg)
            return 0 < c.arg && isNotNone(newx) ? newx : null
        }],
    ]),
)

export const symbolicFlat = symbolic.__add__(
    new PatternMatcher([
        // ** combine terms (opinionated) **
        [UPat.var('x').mul(-1, true).add(UPat.var('y')), (x, y) => x.neg().add(y.neg())], // -(x+y) -> -x + -y
        //   # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
        [UPat.var('x', dtypes.ints).add(UPat.var('y')).mul(UPat.cvar('c')), (x, y, c) => x.mul(c).add(y.mul(c))],
    ]),
)

export const _substitute = new PatternMatcher([[new UPat({ op: Object.values(Ops) as Ops[], name: 'x' }), (ctx, x) => ctx.get(x, null)]])

// # for debug
const syms = new Map([[Ops.ADD, '+'], [Ops.SUB, '-'], [Ops.IDIV, '//'], [Ops.MOD, '%'], [Ops.SHL, '<<'], [Ops.SHR, '>>'], [Ops.MUL, '*'], [Ops.CMPLT, '<'], [Ops.CMPNE, '!='], [Ops.AND, '&'], [Ops.OR, '|'], [Ops.XOR, '^']])
export const renderer = new PatternMatcher([
    [new UPat({ op: [Ops.DEFINE_VAR, Ops.SPECIAL], name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: x.arg[0] })],
    [new UPat({ op: Ops.RANGE, name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: `ridx${x.arg[0]}` })],
    [new UPat({ op: Ops.CONST, name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: x.arg.toString() })],
    [new UPat({ op: Ops.BIND, src: new UPat({ op: Ops.NOOP }), name: 'x' }), (x) => x.src[0]],
    [new UPat({ op: Ops.NEG, src: new UPat({ op: Ops.NOOP }), name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: `(-${x.src[0].arg})` })],
    [new UPat({ op: Ops.MAX, src: new UPat({ op: Ops.NOOP }), name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: `max(${x.src[0].arg}, ${x.src[1].arg})` })],
    [new UPat({ op: Ops.MULACC, src: new UPat({ op: Ops.NOOP }), name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: `(${x.src[0].arg}*${x.src[1].arg}+${x.src[2].arg})` })],
    [new UPat({ op: Ops.WHERE, src: new UPat({ op: Ops.NOOP }), name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: `(${x.src[1].arg} if ${x.src[0].arg} else ${x.src[2].arg})` })],
    [new UPat({ op: GroupOp.ALU, src: new UPat({ op: Ops.NOOP }), name: 'x' }), (x) => new UOp({ op: Ops.NOOP, arg: `(${x.src[0].arg}${syms.get(x.op)}${x.src[1].arg})` })],
])

// # *** what was symbolic.py ***
