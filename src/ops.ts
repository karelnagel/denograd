import { createHash } from 'node:crypto'
import { DType, dtypes, ImageDType, PtrDType } from './dtype.ts'
import { allSame, assert, partition, permutations, prod, raise } from './helpers.ts'
import { Buffer } from 'node:buffer'

type ConstType<This = never> = number | boolean | This
export type sint = number | UOp
export type Variable = UOp
export type ConstLike<This = never> = ConstType<This> | Variable | ConstType[]

class SimpleMathTrait {
    //   # required to implement
    alu = (arg: Ops, ...src: typeof this[]): typeof this => raise('Not implemented')
    constLike = (b: ConstLike<typeof this>): typeof this => raise('Not implemented')

    //   # great functions you get!
    ufix = (x: ConstType<typeof this>): typeof this => x instanceof MathTrait ? x : this.constLike(x)
    _binop = (op: Ops, x: ConstType<typeof this>, reverse: boolean) => reverse ? this.ufix(x).alu(op, this) : this.alu(op, this.ufix(x))
    logicalNot = () => this.ne(true)
    neg = () => {
        const dtype = 'dtype' in this && this.dtype instanceof DType ? this.dtype : null
        if (dtype === null) throw new Error('MathTraits __neg__ requires a dtype')
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
    maximum = (...x: ConstType<typeof this>[]) => this.alu(Ops.MAX, this.ufix(x))
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

const canPad = (u: UOp) => {
    for (const x of u.sparents().keys()) if (GroupOp.UnsafePad.includes(x.op)) return true
    return false
}

// deno-fmt-ignore
const END_FOR_UOP = new Map([[Ops.IF, [Ops.STORE, Ops.ENDIF]], [Ops.RANGE, [Ops.ASSIGN, Ops.ENDRANGE]]])

// With True as the default, this matches the old symbolic behavior
const resolve = (x: UOp, def = false) => {
    if (!(x instanceof UOp)) return Boolean(x)
    if (x.dtype.name !== 'bool') throw new Error('UOp in resolve must be bool')
    // NOTE: generating the text for the exception is expensive, so we do this
    const sx = x.simplify()
    // TODO this Boolean() is probably broken
    return sx.vmin() === sx.vmax() ? Boolean(sx.vmin()) : def
}

// # smax/smin are replacements for max/min that preserve symbolic
const _suop = <T>(lst: T[], uop_fxn: (x: T) => T[], python_fxn: (a: T[]) => T): T[] | T => {
    const [maxUop, maxNum] = partition(lst, (x) => x instanceof UOp)
    if (maxUop.length) return (maxNum.length ? [...maxUop, python_fxn(maxNum)] : maxUop).reduce((prev, curr) => [...prev, ...uop_fxn(curr)], [] as T[]).ssimplify()
    return python_fxn(maxNum)
}
// TODO: Array.isArray should be `if isinstance(lst[0], (tuple, list))`
// TODO: really unsure about these
const smax = (...lst: UOp[]) => _suop(Array.isArray(lst[0]) ? lst[0] : lst, (x) => x.maximum(), (...x) => Math.max(...x))
const smin = (...lst: UOp[]) => _suop(Array.isArray(lst[0]) ? lst[0] : lst, (x) => x.minimum(), (...x) => Math.min(...x))

const ssimplify = (uop: UOp) => uop instanceof UOp ? uop.ssimplify() : uop
const symInfer = (uop: UOp | number, varVals: Map<UOp, number>): number => uop instanceof UOp ? uop.symInfer(varVals) : uop

// AI generated
// used for UOp and UPat
const prettyPrint = (x: any, rep: (x: any) => string, srcfn: (x: any) => any[] = (x) => x.src, cache: Map<any, [number, number, boolean]> = new Map(), d = 0): string => {
    const dfs = (x: any, cache: Map<any, [number, number, boolean]>) => {
        for (const s of srcfn(x) || []) {
            if (!cache.has(s)) {
                cache.set(s, [cache.size, 0, false])
            }
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
    const srcs = srcfn(x) === null ? 'None' : (srcfn(x) || [])
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
        // TODO: instant check rules here make debugging easier (tinygrad)
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

    has_st = () => ![Ops.DEFINE_LOCAL, Ops.DEFINE_GLOBAL, Ops.BUFFER, Ops.CONST, Ops.DEFINE_VAR].includes(this.op)
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
    ssimplify = () => {
        const ret = this.simplify()
        return ret.op === Ops.CONST ? ret.arg : ret
    }
    //   def ssimplify(self) -> Union[UOp, ConstType]: return ret.arg if (ret:=self.simplify()).op is Ops.CONST else ret
    //   def _eval(self, dtype, expected_type:Type[T]) -> T:
    //     assert self.dtype in dtype, f"eval with wrong dtype {self}"
    //     vmin, vmax = (simple_self:=self.simplify())._min_max
    //     if vmin != vmax: raise ValueError(f"eval failed to be a single number, range is {vmin} to {vmax} in {simple_self.render()}")
    //     assert isinstance(vmin, expected_type), f"vmin is wrong dtype {type(vmin)} != {expected_type}"
    //     return vmin
    //   def __bool__(self): return self._eval((dtypes.bool,), bool)
    //   def __int__(self): return self._eval(dtypes.ints, int)
    //   def __float__(self): return self._eval(dtypes.floats, float)
    //   def substitute(self, dvars:Dict[UOp, UOp]):
    //     with Context(TRACK_MATCH_STATS=0):
    //       return graph_rewrite(self, _substitute, dvars)

    //   # *** uop syntactic sugar ***

    //   @property
    //   def st_arg(self) -> ShapeTracker:
    //     assert self.op in GroupOp.Buffer, f"st_arg called on {self.op}"
    //     ret = self.src[0 if self.op is Ops.VALID else 1]
    //     assert ret.op is Ops.VIEW, f"st_arg trying to return {ret}"
    //     return ret.arg
    //   @property
    //   def axis_arg(self) -> Tuple[int, ...]:
    //     assert self.op in {Ops.REDUCE_AXIS, Ops.WMMA}, f"axis_arg called on {self.op}"
    //     ret = self.arg[1] if self.op is Ops.REDUCE_AXIS else self.arg[7]
    //     assert isinstance(ret, tuple) and all(isinstance(x, int) for x in ret), f"axis_arg trying to return {ret}"
    //     return ret
    //   def sink(self, *srcs:UOp): return UOp(Ops.SINK, dtypes.void, (self,)+srcs)
    //   def index(self, idx:UOp, valid:Optional[UOp]=None): return UOp(Ops.INDEX, self.dtype, (self,idx,valid) if valid is not None else (self,idx))
    override constLike = (b: ConstLike<typeof this>): typeof this => UOp.const(this.dtype, b) as typeof this
    //   def broadcast(self, count:int):
    //     assert self.dtype.count == 1
    //     if count == 1: return self
    //     return UOp(Ops.VECTORIZE, self.dtype.vec(count), (self,)*count)
    cast = (dtype: DType) => new UOp({ op: Ops.CAST, dtype, src: [this] })
    bitcast = (dtype: DType) => new UOp({ op: Ops.BITCAST, dtype, src: [this] })
    //   def gep(self, i:Union[Tuple[int, ...], int]):
    //     if isinstance(i, int):
    //       # NOTE: these are just shortcuts to not have to create and fold later
    //       if self.op is Ops.VECTORIZE: return self.src[i]
    //       if self.op is Ops.VCONST: return UOp.const(self.dtype.scalar(), self.arg[i])
    //       if self.op is Ops.CONST: return UOp.const(self.dtype.scalar(), self.arg)
    //       i = (i,)
    //     if (self.dtype.vcount == len(i) and i == tuple(range(len(i)))) or self.dtype == dtypes.void: return self
    //     return UOp(Ops.GEP, self.dtype.scalar().vec(len(i)) if len(i) > 1 else self.dtype.scalar(), (self,), i)
    load = (src: UOp[], kwargs?: Record<string, any>) => new UOp({ op: Ops.LOAD, src: [this, ...src], ...kwargs })
    store = (src: UOp[], kwargs?: Record<string, any>) => new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [this, ...src], ...kwargs })
    //   def alu(self, arg, *src:UOp):
    //     out_dtype = (self, *src)[-1].dtype
    //     if arg in {Ops.CMPLT, Ops.CMPNE} and out_dtype is not None:
    //       out_dtype = dtypes.bool.vec(out_dtype.count) if out_dtype.count > 1 else dtypes.bool
    //     return UOp(arg, out_dtype, (self,)+src)
    static const = (dtype: DType, b: ConstLike) => {
        if (b instanceof UOp) return b.unbind()[0]
        if (Array.isArray(b) && allSame(b)) b = b[0]
        return new UOp({ op: Array.isArray(b) ? Ops.VCONST : Ops.CONST, dtype, arg: dtype ? dtypes.asConst(b, dtype) : b })
    }
    //   @staticmethod
    //   def range(dtype:DType, start:ConstType|UOp, end:ConstType|UOp, idx:int):
    //     return UOp(Ops.RANGE, dtype=dtype, src=(UOp.const(dtype, start) if not isinstance(start, UOp) else start,
    //                                              UOp.const(dtype, end) if not isinstance(end, UOp) else end), arg=(idx, False))
    //   def r(self, op:Ops, axis:Tuple[int, ...]): return UOp(Ops.REDUCE_AXIS, self.dtype, (self,), (op, axis))
    //   def assign(self, x:UOp): return UOp(Ops.ASSIGN, self.dtype, (self,x))
    //   def contiguous(self): return UOp(Ops.CONTIGUOUS, self.dtype, (self,))
    //   @property
    //   def is_contiguous_base(self): return self.op is Ops.CONTIGUOUS and not (self.src[0].base.op is Ops.VIEW and len(self.src[0].base.src) == 2)

    //   # *** from LazyBuffer ***

    //   @staticmethod
    //   def const_with_shape(dtype:DType, val:ConstLike, shape:Tuple[sint,...]) -> UOp:
    //     from tinygrad.shape.shapetracker import ShapeTracker
    //     return UOp(Ops.VALID, dtypes.bool, (ShapeTracker.from_shape(()).reshape((1,)*len(shape)).expand(shape).to_uop(),)).where(UOp.const(dtype, val), 0)

    //   # *** uop movement ops ***

    //   @property
    //   def base(self) -> UOp: return self.src[0] if self.op is Ops.VIEW and len(self.src) == 1 else self
    //   def view(self, st:ShapeTracker) -> UOp:
    //     assert self.op is not Ops.STORE, "VIEW of STORE is invalid, STORE is always base"
    //     return self if self.st is None or self.st == st else UOp(Ops.VIEW, self.dtype, (self,), st)
    //   def reshape(self, arg:Tuple[sint, ...]) -> UOp: return self.view(unwrap(self.st).reshape(arg))

    //   # *** uop Buffer stuff ***

    //   @staticmethod
    //   def new_buffer(device:str, size:int, dtype:DType, num=-1): return  UOp(Ops.BUFFER, dtype.ptr(), (), (num, (device, size, dtype)))
    //   @functools.cached_property
    //   def device(self) -> str:
    //     match self.op:
    //       case Ops.COPY: return self.arg
    //       case Ops.BUFFER: return self.arg[1][0]
    //       case _: return self.src[0].device
    //   @property
    //   def size(self) -> int: return self.buf_uop.arg[1][1]
    //   @property
    //   def buf_uop(self) -> UOp:
    //     if self.op is Ops.BUFFER: return self
    //     assert self.op in {*GroupOp.Buffer, Ops.ASSIGN, Ops.VIEW} and self.src[0].op is Ops.BUFFER, f"buf_uop called on {self.op}"
    //     return self.src[0]

    //   # *** uop Variable stuff ***

    //   @staticmethod
    //   def variable(name:str, min_val:ConstType, max_val:ConstType, dtype:DType=dtypes.int):
    //     assert not isinstance(min_val, UOp) and not isinstance(max_val, UOp), f"can't create Variable {name} with {min_val}/{max_val}"
    //     return UOp(Ops.DEFINE_VAR, dtype, arg=(name, min_val, max_val))
    //   @property
    //   def expr(self):
    //     assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    //     return self.arg[0]
    //   def bind(self, val:int):
    //     assert self.op is Ops.DEFINE_VAR, f"op is {self.op}, need DEFINE_VAR"
    //     assert self.arg[1] <= val and val <= self.arg[2], f"bind {val} not in range [{self.arg[1]}, {self.arg[2]}]"
    //     return UOp(Ops.BIND, self.dtype, (self, self.const_like(val)))
    unbind = (): [Variable, number] => {
        if (!(this.op === Ops.BIND && this.src[0].op === Ops.DEFINE_VAR && this.src[1].op === Ops.CONST)) throw new Error(`can't unbind ${this}`)
        return [this.src[0], this.src[1].arg]
    }
    //   def unbind(self) -> Tuple[Variable, int]:
    //     assert self.op is Ops.BIND and self.src[0].op is Ops.DEFINE_VAR and self.src[1].op is Ops.CONST, f"can't unbind {self}"
    //     return self.src[0], self.src[1].arg
    //   @property
    //   def val(self) -> int: return self.unbind()[1]
    //   def vars(self) -> Set[UOp]:
    //     bound_vars = set([x for x in self.sparents if x.op is Ops.BIND and x.src[0].op is Ops.DEFINE_VAR])
    //     bound_var_base = set(x.src[0] for x in bound_vars)
    //     all_vars = set([x for x in self.sparents if x.op is Ops.DEFINE_VAR])
    //     return bound_vars.union(set([x for x in all_vars if x not in bound_var_base]))
    //   def variables(self) -> List[Variable]:
    //     st_vars: List[Set[Variable]] = [x.st_arg.vars() for x in self.sparents if x.op in GroupOp.Buffer]
    //     return sorted(set.union(*st_vars, [x.unbind()[0] if x.op is not Ops.DEFINE_VAR else x for x in self.vars()]), key=lambda v: v.arg)

    //   # *** uop symbolic stuff ***

    //   def const_factor(self) -> int:
    //     """largest known int that divides self"""
    //     if self.op is Ops.CONST: return self.arg
    //     if self.op is Ops.VCONST: return math.gcd(*self.arg)
    //     if self.op is Ops.ADD: return math.gcd(self.src[0].const_factor(), self.src[1].const_factor())
    //     if self.op is Ops.MUL: return self.src[0].arg if self.src[0].op is Ops.CONST else self.src[1].arg if self.src[1].op is Ops.CONST else 1
    //     return 1
    //   def divides(self, v) -> Optional[UOp]:
    //     if v==1: return self
    //     if self.op is Ops.CONST: return self.const_like(self.arg//v) if self.arg%v == 0 else None
    //     if self.op is Ops.VCONST: return self.const_like(tuple(x//v for x in self.arg)) if all(x%v == 0 for x in self.arg) else None
    //     if self.op is Ops.ADD: return d0+d1 if (d0:=self.src[0].divides(v)) is not None and (d1:=self.src[1].divides(v)) is not None else None
    //     if self.op is Ops.MUL:
    //       if (d0:=self.src[0].divides(v)) is not None: return d0 * self.src[1]
    //       if (d1:=self.src[1].divides(v)) is not None: return self.src[0] * d1
    //     return None # generic None if we aren't sure

    static min = (...args: UOp[]) => args.reduce((min, current) => min.lt(current) ? min : current)
    static max = (...args: UOp[]) => args.reduce((max, current) => max.gt(current) ? max : current)

    vmin = () => this._minMax()[0]
    vmax = () => this._minMax()[1]
    _minMax = (): [UOp, UOp] => {
        if (GroupOp.Binary.includes(this.op) && !dtypes.isFloat(this.dtype)) {
            const [[s0_vmin, s0_vmax], [s1_vmin, s1_vmax]] = [this.src[0]._minMax(), this.src[1]._minMax()]
            if (this.op === Ops.ADD) return [s0_vmin.add(s1_vmin), s0_vmax.add(s1_vmax)]
            if (this.op === Ops.MUL) {
                const vals = [s0_vmin.mul(s1_vmin), s0_vmin.mul(s1_vmax), s0_vmax.mul(s1_vmin), s0_vmax.mul(s1_vmax)]
                return [UOp.min(...vals), UOp.max(...vals)]
            }
            if (this.op === Ops.MOD && s1_vmin.gt(0)) return [this.constLike(0), s1_vmax.sub(1)]
            if (this.op === Ops.IDIV && s1_vmin === s1_vmax) { // min/max are equal in a CONST
                if (s1_vmin.gt(0)) return [s0_vmin.idiv(s1_vmin), s0_vmax.idiv(s1_vmin)]
                if (s1_vmin.lt(0) && s0_vmin.ge(0)) return [(s0_vmax.idiv(s1_vmin.neg())).neg(), (s0_vmin.idiv(s1_vmin.neg())).neg()]
            }
            if (this.op === Ops.MAX) return [UOp.max(s0_vmin, s1_vmin), UOp.max(s0_vmax, s1_vmax)]
            if (this.op === Ops.CMPLT) return [s0_vmax.lt(s1_vmin), s0_vmin.lt(s1_vmax)]
            if (this.op === Ops.CMPNE) return [(s0_vmax.lt(s1_vmin)).bitwiseOr(s1_vmax.lt(s0_vmin)), (s0_vmin.eq(s0_vmax).bitwiseAnd(s0_vmax.eq(s1_vmin)).bitwiseAnd(s1_vmin.eq(s1_vmax))).neg()]
            if (this.dtype == dtypes.bool) {
                if (this.op === Ops.OR) return [s0_vmin.bitwiseOr(s1_vmin), s0_vmax.bitwiseOr(s1_vmax)]
                if (this.op === Ops.AND) return [s0_vmin.bitwiseAnd(s1_vmin), s0_vmax.bitwiseAnd(s1_vmax)]
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
        // TODO:
        //     return eval("lambda "+','.join(varnames)+": "+sself.render()), varnames  # pylint: disable=eval-used
    }
    //   def _sym_fxn(self):
    //     sself = self.simplify()
    //     varnames = tuple(x.arg[0] for x in sself.sparents if x.op is Ops.DEFINE_VAR)
    //     # TODO: sanitize varnames, or don't use naked eval while staying fast
    //     return eval("lambda "+','.join(varnames)+": "+sself.render()), varnames  # pylint: disable=eval-used

    symInfer = (varVals: Map<UOp, number>) => {
        const [fxn, varnames] = this._sym_fxn()
        const map = new Map<UOp, number>()
        for (const [k, v] of varVals.entries()) if (varnames.includes(k.arg[0])) map.set(k.arg[0], v)
        return fxn(map)
    }
    //   def sym_infer(self, var_vals:Dict[UOp, int]):
    //     fxn, varnames = self._sym_fxn
    //     return fxn(**{k.arg[0]:v for k,v in var_vals.items() if k.arg[0] in varnames})

    //   def render(self, simplify=True) -> str:
    //     ret = graph_rewrite(self.simplify() if simplify else self, renderer)
    //     return ret.arg if ret.op is Ops.NOOP else str(ret)
}

// @dataclass(frozen=True)
class KernelInfo {
    //   local_dims: int = 0           # number of local dimensions  (this is remapping RANGE to SPECIAL)
    //   upcasted: int = 0             # count that are upcasted     (this is remapping RANGE to EXPAND)
    //   dont_use_locals: bool = False # don't use local indexing
}

// # ***** ops in python *****

const hookOverflow = <T extends any[]>(dv: number, fxn: (...args: T) => number) => {
    // deno-fmt-ignore
    return (...args: T) => { try { return fxn(...args) } catch { return dv } }
}

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

// def exec_alu(op:Ops, dtype:DType, operands, truncate_output=True):
//   if dtype.count > 1:
//     return tuple([exec_alu(op, dtype.scalar(), [x[i] if isinstance(x, tuple) else x for x in operands]) for i in range(dtype.count)])
//   alu = python_alu[op](*operands)
//   return truncate.get(dtype, lambda x: x)(alu) if truncate_output else alu

// # ***** uop helpers *****

// def print_uops(uops:List[UOp]):
//   for i,u in enumerate(uops):
//     formatted_parents = [(uops.index(x) if x.op is not Ops.CONST else f"{x.arg}") if x in uops else "--" for x in u.src]
//     print(f"{i:4d} {str(u.op):20s}: {str(u.dtype):30s} " f"{str(formatted_parents):32s} {u.arg}")

// def flops_mem(uops:List[UOp], ignore_indexing=False) -> Tuple[sint, sint]:
//   flops: sint = 0
//   mem: sint = 0
//   mults: sint = 1
//   mult_stack: List[sint] = []
//   dont_count: Set[UOp] = set()
//   if ignore_indexing:
//     for u in uops:
//       if u.op in {Ops.LOAD, Ops.STORE}:
//         dont_count = dont_count.union(u.src[0].sparents)
//         if len(u.src) > 2: dont_count = dont_count.union(u.src[2].sparents)
//       elif u.op is Ops.IF:
//         dont_count = dont_count.union(u.src[0].sparents)
//   for u in uops:
//     if u.op is Ops.RANGE:
//       mult_stack.append(mults)
//       mults *= (u.src[1] - u.src[0]).ssimplify()
//     elif u.op is Ops.ENDRANGE:
//       mults = mult_stack.pop(-1)
//     elif u.op is Ops.SPECIAL:
//       mults *= u.arg[1] # NOTE: we don't push to the mult_stack here, you can't end these
//     elif u.op is Ops.LOAD:
//       mem += u.dtype.itemsize * mults
//     elif u.op is Ops.STORE:
//       mem += u.src[1].dtype.itemsize * mults
//     elif u.op in GroupOp.ALU and u not in dont_count:
//       flops += (mults * (2 if u.op is Ops.MULACC else 1)) * u.dtype.count
//     elif u.op is Ops.WMMA and u not in dont_count:
//       flops += 2 * prod(u.arg[1]) // u.arg[5] * mults
//   return flops, mem

// # ***** pattern matcher *****

// def get_location() -> Tuple[str, int]:
//   frm = sys._getframe(1)
//   # find the real frame in the file that has the UPat, TODO: is there a better way to do this?
//   while frm.f_back is not None and pathlib.Path(frm.f_back.f_code.co_filename).name in {"ops.py", "uopgraph.py", "schedule.py",
//                                                                                         "lowerer.py", "cstyle.py"}:
//     frm = frm.f_back
//   return frm.f_code.co_filename, frm.f_lineno
// @functools.lru_cache(None)
// def lines(fn) -> List[str]:
//   with open(fn) as f: return f.readlines()

type UPatInput = { op?: Ops | Ops[]; dtype?: DType | DType[]; src?: UPat | UPat[]; arg?: any; name?: string; allowAnyLen?: boolean; location?: any; customEarlyReject?: Ops[] }
class UPat extends MathTrait {
    //   __slots__ = ["op", "dtype", "arg", "name", "src"]
    op?: Ops[]
    dtype: DType[]
    arg: any
    name: string
    _inSrc: UPat | UPat[]
    customEarlyReject?: Ops[]
    src?: UPat[][]
    allowedLen: number
    location: [string, number]
    earlyReject: Ops[]

    constructor({ op, dtype = [], arg, location, name = '', src = [], allowAnyLen = false, customEarlyReject }: UPatInput) {
        super()
        assert(!op || !(!Array.isArray(op) && Object.values(Ops).includes(op)) || !(Array.isArray(op) && Object.values(Ops).includes(op[0])), 'op must be Ops or tuple of Ops')
        this.op = !op ? undefined : Array.isArray(op) ? op : [op]
        this.dtype = Array.isArray(dtype) ? dtype : [dtype]
        this.arg = arg
        this.name = name
        this._inSrc = src
        this.customEarlyReject = customEarlyReject
        assert(self.name != 'ctx', "UPat can't be named ctx")

        // try all permutations if it's a list
        if (Array.isArray(src)) this.src = allSame(src) ? permutations(src) : [src]
        //     # repeat if it's a UPat
        else if (src instanceof UPat) this.src = [[src]]

        this.allowedLen = allowAnyLen || src instanceof UPat || !src ? -1 : src.length
        this.location = location || getLocation()

        if (customEarlyReject) this.earlyReject = customEarlyReject
        else {
            const upatMatch = src instanceof UPat ? [src] : !this.src ? [] : this.src[0]
            this.earlyReject = upatMatch.filter((pp) => !!pp.op && pp.op.length === 1).map((pp) => pp.op![0])
        }
    }

    named = (name: string) => new UPat({ op: this.op, dtype: this.dtype, src: this._inSrc, arg: this.arg, name, allowAnyLen: this.allowedLen === -1, customEarlyReject: this.customEarlyReject })

    static any = (src: UPatInput['src']) => new UPatAny({ src: src })

    static var = (name?: string, dtype?: DType | DType[]) => new UPat({ dtype: dtype, name: name })
    static cvar = (name?: string, dtype?: DType, vec = true) => new UPat({ op: vec ? [Ops.CONST, Ops.VCONST] : Ops.CONST, dtype, name })
    static const = (dtype?: DType | DType[], b?: ConstLike) => new UPat({ op: Ops.CONST, dtype: dtype, arg: b })

    //   # copied from UOp
    index = (idx: UPat, valid?: UPat) => new UPat({ op: Ops.INDEX, dtype: this.dtype, src: valid ? [this, idx, valid] : [this, idx] })
    view = (st?: any, kwargs?: any) => new UPat({ op: Ops.VIEW, dtype: this.dtype, src: [this], arg: st, ...kwargs })
    cast = (dtype?: DType) => new UPat({ op: Ops.CAST, dtype, src: [this] })
    bitcast = (dtype?: DType) => new UPat({ op: Ops.BITCAST, dtype, src: [this] })
    gep = (i: number) => new UPat({ op: Ops.GEP, src: [this], arg: [i] })
    load = (src: UPat[], kwargs?: any) => new UPat({ op: Ops.LOAD, src: [this, ...src], ...kwargs })
    store = (src: UPat[], kwargs?: any) => new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [this, ...src], ...kwargs })
    assign = (x: UPat) => new UPat({ op: Ops.ASSIGN, dtype: this.dtype, src: [this, x] })

    const_like = (b: ConstLike) => UPat.const(this.dtype, b)
    override alu = (op: Ops, ...src: this[]) => {
        const asrc = [this, ...src]
        return new UPat({ op, dtype: [Ops.CMPLT, Ops.CMPNE].includes(op) ? undefined : asrc.pop()?.dtype, src: GroupOp.Commutative.includes(op) ? asrc : asrc })
    }

    // deno-fmt-ignore
    printable = ():string => {
        try{ return lines(this.location[0])[this.location[1]-1].strip() }
        catch { return "<missing>"}
    }
    __repr = () => {
        const rep = (x: UPat) => {
            const form = `UPat(${!x.op ? null : x.op.map((x) => x.toString()).join(', ')}, ${x.arg}, name=${repr(x.name)}, dtype=${x.dtype ? new Set(x.dtype) : null}, allow_any_len=${
                x.allowedLen === 0
            }, src=(???"[%s]" if x.src and len(x.src)>1 else "(%s)"???))`
            return form
        }
        // TODO: not quite right, check it
        return prettyPrint(this, rep, (x) => !x.src ? null : x.src[0])
    }
    match = (uop: UOp, store: Record<string, UOp>): Record<string, UOp>[] => {
        if (
            (!!this.op && !this.op.includes(uop.op)) ||
            (!!this.name && store.setdefault(self.name, uop) !== uop) ||
            (!!this.dtype && !this.dtype.includes(uop.dtype) && !this.dtype.includes(uop.dtype.scalar())) ||
            (!!this.arg && this.arg !== uop.arg) ||
            (this.allowedLen !== -1 && uop.src.length !== this.allowedLen)
        ) return []
        if (!this.src) return [store]
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

// def deconstruct_function(fxn:Callable) -> Tuple:
//   new_globals = {k:v for k,v in fxn.__globals__.items() if k in fxn.__code__.co_names}
//   for co in fxn.__code__.co_consts:
//     if isinstance(co, types.CodeType): new_globals.update({k:v for k,v in fxn.__globals__.items() if k in co.co_names})
//   # NOTE: optional round trip through pickle!
//   assert fxn.__closure__ is None, "closures are not supported in pattern matchers"
//   ret = fxn.__code__, new_globals, fxn.__name__, fxn.__defaults__
//   return pickle.loads(pickle.dumps(ret)) if getenv("TEST_PICKLE") else ret

type Pattern = [UPat, (...args: any[]) => any]
class PatternMatcher {
    patterns: Pattern[]
    pdict = new Map<Ops, ([UPat, () => void, Set<any>, boolean][])>()
    constructor(patterns: Pattern[]) {
        this.patterns = patterns
        // NOTE: use of DefaultDict here is very dangerous! all keys will live for the lifetime of the PatternMatcher!

        //     self.pdict: Dict[Ops, List[Tuple[UPat, Callable, Set, bool]]] = {}
        //     # uop is required, arg is optional
        //     for p,fxn in self.patterns:
        //       assert p.op is not None
        //       tuple_fxn = fxn if isinstance(fxn, tuple) else deconstruct_function(fxn)
        //       real_fxn = types.FunctionType(*tuple_fxn)
        //       for uop in p.op: self.pdict.setdefault(uop, []).append((p, real_fxn, p.early_reject, 'ctx' in inspect.signature(real_fxn).parameters))
    }

    //   def __reduce__(self): return PatternMatcher, ([(x,deconstruct_function(fxn) if fxn.__name__ == "<lambda>" else fxn) for x,fxn in self.patterns],)

    //   @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
    __add__ = (more: PatternMatcher) => new PatternMatcher([...this.patterns, ...more.patterns])

    //   def rewrite(self, uop:UOp, ctx=None) -> Optional[UOp]:
    //     ler = {u.op for u in uop.src}
    //     for p,fxn,early_reject,has_ctx in self.pdict.get(uop.op, []):
    //       if not early_reject.issubset(ler): continue
    //       for match in p.match(uop, {}):
    //         if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None: return ret
    //     return None
}
// # *** tracking pattern matcher ***

// TRACK_MATCH_STATS = ContextVar("TRACK_MATCH_STATS", 2 if getenv("VIZ") else 0)
// match_stats:Dict[UPat, List[Union[int, float]]] = dict()
// @dataclass(frozen=True)
class TrackedRewriteContext {
    //   loc: Tuple[str, int]                                                                              # location that called graph_rewrite
    //   sink: UOp                                                                                         # the sink passed into the rewrite
    //   matches: List[Tuple[UOp, Optional[UOp], Optional[UPat], float]] = field(default_factory=list)     # all matches of sparents
}
// rewrite_stack: List[Tuple[Any, List[TrackedRewriteContext]]] = []
// contexts: List[Tuple[Any, List[TrackedRewriteContext]]] = []
// _rewrite_cnt: Dict[str, int] = {}
// def track_rewrites(named=False):
//   def _decorator(func):
//     def __wrapper(self, *args, **kwargs):
//       if TRACK_MATCH_STATS >= 2:
//         if named: _rewrite_cnt[func.__name__] = _rewrite_cnt.setdefault(func.__name__, 0)+1
//         rewrite_stack.append((f"{(n:=func.__name__)}_{_rewrite_cnt[n]}" if named else self, []))
//       try: ret = func(self, *args, **kwargs)
//       finally: # NOTE: save everything in the stack
//         if TRACK_MATCH_STATS >= 2: contexts.append(rewrite_stack.pop())
//       return ret
//     return __wrapper
//   return _decorator

class TrackedPatternMatcher extends PatternMatcher {
    //   def __init__(self, patterns:List[Tuple[UPat, Callable]]):
    //     super().__init__(patterns)
    //     for p,_ in self.patterns:
    //       if p not in match_stats: match_stats[p] = [0,0,0.0,0.0]

    //   def rewrite(self, uop:UOp, ctx=None) -> Optional[UOp]:
    //     ret = None
    //     ler = {u.op for u in uop.src}
    //     for p,fxn,early_reject,has_ctx in self.pdict.get(uop.op, []):
    //       st = time.perf_counter()
    //       if not early_reject.issubset(ler):
    //         match_stats[p][2] += time.perf_counter()-st
    //         continue
    //       match_stats[p][1] += 1
    //       for match in p.match(uop, {}):
    //         if (ret:=(fxn(ctx=ctx, **match) if has_ctx else fxn(**match))) is not None:
    //           match_stats[p][0] += 1
    //           match_stats[p][3] += (et:=time.perf_counter()-st)
    //           if TRACK_MATCH_STATS >= 3: print(f"{et*1e6:7.2f} us -- ", p.printable())
    //           if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0 and isinstance(ret, UOp): rewrite_stack[-1][1][-1].matches.append((uop, ret, p, et))
    //           return ret # NOTE: if it returns None, we keep trying to match
    //       match_stats[p][2] += time.perf_counter()-st
    //     if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0: rewrite_stack[-1][1][-1].matches.append((uop, ret, None, 0))
    //     return None
}
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
//         if v[1] != 0: print(f"{v[0]:6d} / {v[1]:7d} -- {v[3]*1000.:9.2f} / {(v[2]+v[3])*1000.:9.2f} ms -- {loc_str:15s}", k.printable())
//         ret = [x+y for x,y in zip(ret, v)]
//       print(f"{ret[0]:6d} / {ret[1]:7d} -- {ret[3]*1000.:9.2f} / {(ret[2]+ret[3])*1000.:9.2f} ms -- TOTAL")

// # *** simple graph rewrite engine ***

class RewriteContext {
    //   def __init__(self, pm, ctx):
    //     self.pm: PatternMatcher = pm
    //     self.ctx = ctx
    //     self.replace: Dict[UOp, UOp] = {}
    //   def rewrite(self, n:UOp) -> UOp:
    //     if (rn := self.replace.get(n)) is not None: return rn
    //     new_src = tuple(map(self.rewrite, n.src))
    //     new_n = self.pm.rewrite(n, self.ctx) if new_src == n.src else UOp(n.op, n.dtype, new_src, n.arg)
    //     self.replace[n] = ret = n if new_n is None else self.rewrite(new_n)
    //     return ret
}
// def graph_rewrite(sink:UOp, pm:PatternMatcher, ctx=None) -> UOp:
//   if TRACK_MATCH_STATS >= 2 and len(rewrite_stack) != 0:
//     rewrite_stack[-1][1].append(TrackedRewriteContext(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink))
//   return RewriteContext(pm, ctx).rewrite(sink)

// # ***** uop type spec *****

// # this is the matcher for the final rendered UOps
// # matcher functions returns True or False (or None to not match)
export const spec = new PatternMatcher([
    [new UPat({ op: Ops.DEFINE_GLOBAL, name: 'x' }), (x) => (x.dtype instanceof PtrDType || x.dtype instanceof ImageDType) && !x.dtype.local],
    [new UPat({ op: Ops.DEFINE_LOCAL, name: 'x' }), (x) => x.dtype instanceof PtrDType && x.dtype.local],
    [new UPat({ op: Ops.DEFINE_ACC, src: [UPat.var('c')], name: 'x', allowAnyLen: true }), (x, c) => x.src.slice(1).every((y) => y.op === Ops.RANGE) && c.dtype == x.dtype],
    [new UPat({ op: Ops.DEFINE_VAR, src: [], name: 'x' }), (x) => typeof x.arg[1] === 'number' && typeof x.arg[2] === 'number'],
    [new UPat({ op: Ops.RANGE, src: [new UPat({ name: 'x' }), new UPat({ name: 'y' })], name: 'rng' }), (rng, x, y) => rng.dtype == x.dtype && x.dtype == y.dtype],
    [new UPat({ op: Ops.SPECIAL, src: [] }), () => true],

    //   # TODO: confirm the args of both of these are shapetrackers
    [new UPat({ op: Ops.VIEW, dtype: dtypes.void, src: [] }), () => true],
    [new UPat({ op: Ops.VIEW, src: [UPat.var('src')], name: 'x' }), (x, src) => src.op !== Ops.STORE && x.dtype == src.dtype],
    [new UPat({ op: Ops.VALID, dtype: dtypes.bool, src: [new UPat({ op: Ops.VIEW })] }), () => true],
    [new UPat({ op: Ops.CONST, name: 'x' }), (x) => x.dtype == x.dtype.scalar() && typeof x.arg === typeof dtypes.asConst(x.arg, x.dtype)],

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
    [new UPat({ op: Ops.LOAD, src: [new UPat({ op: [Ops.INDEX, Ops.CAST] }), new UPat({ name: 'alt' }), new UPat({ dtype: dtypes.bool })], name: 'ld' }), (ld, alt) => ld.dtype == alt.dtype],

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
    [new UPat({ op: GroupOp.ALU, name: 'x' }), (x) => x.src!.every((y) => x.dtype === y.dtype)],
    [new UPat({ op: Ops.ASSIGN, src: [new UPat({ op: [Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL] }), new UPat({})] }), () => true],
    [new UPat({ op: Ops.ENDRANGE, dtype: dtypes.void, src: [new UPat({ op: Ops.RANGE })] }), () => true],

    //   # all WMMA has 3 args, <x, w, acc>
    [new UPat({ op: Ops.WMMA, src: [new UPat({}), new UPat({}), new UPat({})] }), () => true],
    [new UPat({ op: Ops.CONTRACT, name: 'x' }), (x) => x.dtype.count === prod(x.arg.map((y) => y[1]))],
    [new UPat({ op: Ops.EXPAND, name: 'x' }), (x) => x.src![0].dtype.count === prod(x.arg.map((y) => y[1]))],

    //   # if has a <gate, barrier?>

    [new UPat({ op: Ops.IF, dtype: dtypes.void, src: [new UPat({})] }), () => true],
    [new UPat({ op: Ops.IF, dtype: dtypes.void, src: [new UPat({}), new UPat({ op: Ops.BARRIER })] }), () => true],
    [new UPat({ op: Ops.ENDIF, dtype: dtypes.void, src: [new UPat({ op: Ops.IF })] }), () => true],
    [new UPat({ op: Ops.REDUCE_AXIS, name: 'x' }), (x) => Array.isArray(x.arg) && x.arg.length == 2 && [Ops.ADD, Ops.MUL, Ops.MAX].includes(x.arg[0])],
    [new UPat({ op: Ops.GEP, src: [new UPat({ name: 'src' })], name: 'gep' }), (gep, src) => gep.dtype == src.dtype.scalar()],
    [new UPat({ op: Ops.VECTORIZE, name: 'x' }), (x) => x.src!.length > 1 && x.src!.length === x.dtype.count && x.src!.every((y) => x.dtype === y.dtype.vec(x.src?.length))],
    [new UPat({ op: (Ops.BITCAST, Ops.CAST), src: [new UPat({})], name: 'x' }), (x) => !x.arg],
    [new UPat({ op: Ops.BARRIER, dtype: dtypes.void, src: new UPat({ op: Ops.STORE, allowAnyLen: true }) }), () => true], // NOTE: all pointers must be local
    //   # NOTE: for testing, we let sinks be anything
    // [new UPat({ op: UOps.SINK, src: new UPat({ op: UOps.STORE }) }), () => true],
    [new UPat({ op: Ops.SINK, dtype: dtypes.void }), () => true],
    [new UPat({ op: Ops.NOOP }), () => true],

    //   # PTX LOAD/STORE
    [new UPat({ op: (Ops.LOAD, Ops.STORE), src: [new UPat({ dtype: dtypes.int64 })], allowAnyLen: true }), () => true],
    [new UPat({ op: Ops.BARRIER, dtype: dtypes.void, src: new UPat({ op: Ops.STORE, src: [new UPat({ dtype: dtypes.int64 })], allowAnyLen: true }) }), () => true],
])

// def type_verify(uops:List[UOp]):
//   for i,u in enumerate(uops):
//     if not spec.rewrite(u):
//       print_uops(uops)
//       raise RuntimeError(f"UOp verification failed at {i} on {u.op} {u.dtype} {len(u.src)} {[x.op for x in u.src]} {u.arg}")

// # *** uop helpers ***

// def cast_float_to_bf16(x: UOp) -> UOp:
//   assert x.dtype == dtypes.float, "cast float -> bf16 must start with float"
//   x = x.bitcast(dtypes.uint)
//   x = (-x & 0x7f800000).where(x + ((x >> 16) & 1) + 0x7fff, (x & 0xffff).where((x | 0x10000), x))
//   return (x >> 16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)

// # *** most of symbolic lives here now ***

// def split_uop(x:UOp, sep:Ops):
//   if x.op is sep:
//     for s in x.src: yield from split_uop(s, sep)
//   else: yield x

// def mod_folding(x:UOp, c:int) -> Optional[UOp]:
//   # simplify x % c, None means no change

//   # simple cancel mod case
//   if 0 < c and 0 <= x.vmin and (quotient:=x.vmin//c) == x.vmax//c: return x-quotient*c

//   remainder, something_changed = [], False
//   for u in split_uop(x, Ops.ADD):
//     if (factor:=u.const_factor())%c != factor:
//       divides = u.divides(factor)*(factor%c)
//       assert divides is not None
//       remainder.append(divides)
//       something_changed = True
//     elif u.op is Ops.MOD and (s1:=u.src[1]).op is Ops.CONST and s1.arg%c == 0:
//       remainder.append(u.src[0])
//       something_changed = True
//     else: remainder.append(u)
//   if not something_changed: return None
//   return functools.reduce(operator.add, remainder)%c if remainder else x.const_like(0)

// def div_folding(x:UOp, c:int) -> Optional[UOp]:
//   # simplify x // c, None means no change

//   # simple cancel div case
//   if 0 <= x.vmin and x.vmax < c: return x.const_like(0)

//   quotient, remainder, rem_const, something_changed, gcd, divisor = [], [], 0, False, c, 1
//   for u in split_uop(x, Ops.ADD):
//     if u.op is Ops.CONST:
//       # add all const together first
//       if rem_const != 0: something_changed = True
//       rem_const += u.arg
//     elif (factor:=u.const_factor())%c == 0:
//       if factor:
//         divides = u.divides(c)
//         assert divides is not None
//         quotient.append(divides)
//       something_changed = True
//     else:
//       # divisor is the smallest common divisor of all MULs
//       if u.op is Ops.MUL and factor > 1 and c % factor == 0 and (divisor == 1 or divisor > factor): divisor = factor
//       remainder.append(u)
//       gcd = math.gcd(gcd, factor)

//   # handle the const
//   if rem_const%c != rem_const:
//     something_changed = True
//     quotient.append(x.const_like(rem_const//c))
//     rem_const = rem_const%c
//   if rem_const != 0: remainder.append(x.const_like(rem_const))

//   # x // c -> quotient + (remainder // div) // (c // div)
//   div = gcd if gcd > 1 else divisor

//   if not something_changed: return newx//(c//div) if 1 < div < c and (newx:=div_folding(x, div)) is not None else None
//   rem:Optional[UOp] = functools.reduce(operator.add, remainder) if remainder else None
//   quo:Optional[UOp] = functools.reduce(operator.add, quotient) if quotient else None
//   if quo is None: return x.const_like(0) if rem is None else cast(UOp, div_folding(rem, div))//(c//div)
//   return quo if rem is None else cast(UOp, div_folding(rem, div))//(c//div)+quo

// def lt_folding(x:UOp, c:int) -> Optional[UOp]:
//   p, np = partition(split_uop(x, Ops.ADD), lambda u: u.const_factor() == 1)
//   if np and (d:=math.gcd(*[u.const_factor() for u in np], c)) > 1 and 0 <= sum(u.vmin for u in p) and sum(u.vmax for u in p) < d:
//     return cast(UOp, functools.reduce(operator.add, np).divides(d)).lt(c//d)
//   return None

// def fold_unrolled_divs(divs:UOp):
//   # div pattern in unrolled arange
//   # example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
//   add_chain, denominator, seen_const, ans = list(split_uop(divs, Ops.ADD)), None, [], None
//   for u in add_chain:
//     if not (u.op is Ops.IDIV and u.src[1].op is Ops.CONST): return None
//     if denominator is None: denominator = u.src[1].arg
//     if denominator != u.src[1].arg: return None
//     # assumed CONST is the last of an ADD
//     if (s0:=u.src[0]).op is Ops.ADD and s0.src[1].op is Ops.CONST and s0.src[1].op is Ops.CONST:
//       seen_const.append(s0.src[1].arg)
//       s0 = s0.src[0]
//     else: seen_const.append(0)
//     if ans is None: ans = s0
//     if ans is not s0: return None
//   if denominator is None: return None
//   # the first (denominator-len(seen_const)) terms may have been folded to 0 already
//   for i in range(denominator-len(seen_const)):
//     if ans is not None and 0 <= ans.vmin and ans.vmax + i < denominator: seen_const.append(i)
//   return ans if ans is not None and sorted(seen_const)==list(range(denominator)) else None

// def canonicalize_simplex(X:UOp) -> Optional[UOp]:
//   # (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
//   # returns x0 + x1 + ... in such case, or None if not
//   changed, ret = False, []
//   for u in split_uop(X, Ops.ADD):
//     # assumed the const is the last src of MUL
//     if u.op is Ops.MUL and u.src[1].op is Ops.CONST and u.src[1].arg > 0:
//       changed = True
//       u = u.src[0]
//     if not (u.op in GroupOp.Irreducible and u.vmin >= 0): return None
//     ret.append(u)
//   return functools.reduce(operator.add, ret) if changed else None

// def is_increasing(f:UOp) -> bool:
//   # is f a monotonically increasing function regards its input
//   if f.op in GroupOp.Irreducible: return True
//   if f.op is Ops.ADD: return is_increasing(f.src[0]) and is_increasing(f.src[1])
//   if f.op in (Ops.MUL, Ops.IDIV) and f.src[1].op is Ops.CONST and f.src[1].arg >= 0: return is_increasing(f.src[0])
//   return False  # False if not sure

// def parse_valid(valid:UOp) -> Tuple[UOp, bool, int]:
//   # if it's X <= c, returns X, True, c
//   # if it's X >= c, returns X, False, c

//   # (X < c).ne(True) -> X >= c
//   if valid.op is Ops.CMPNE and valid.src[1].op is Ops.CONST and valid.src[1].arg == 1 and \
//     (s0:=valid.src[0]).op is Ops.CMPLT and s0.src[1].op is Ops.CONST: return s0.src[0], False, s0.src[1].arg
//   # X < c -> X <= c-1
//   if valid.op is Ops.CMPLT and valid.src[1].op is Ops.CONST: return valid.src[0], True, valid.src[1].arg-1
//   raise ValueError(f"not able to parse {valid=}")

// def uop_given_valid(valid:UOp, uop:UOp) -> Optional[UOp]:
//   # return None if valid is always False, otherwise the simplified uop (might be the same as input)

//   # first, parse valid into {expr: (lower_bound, upper_bound)}
//   bounds:DefaultDict[UOp, List[Optional[ConstType]]] = defaultdict(lambda: [None, None])
//   for stmt in split_uop(valid, Ops.AND):
//     try: expr, is_upper, c = parse_valid(stmt)
//     except ValueError: return uop  # give up if we cannot parse the valid
//     bounds[expr][int(is_upper)] = c

//   # simplify uop given that valid is True
//   for expr,v in bounds.items():
//     # some expr has lower bound > upper bound -> valid is an empty set and we return None
//     if v[0] is not None and v[1] is not None and v[0] > v[1]: return None

//     # every candidate is a set of contrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
//     candidates = []
//     if expr.op is Ops.ADD and v[0] == 1 and all(u.op in GroupOp.Irreducible for u in split_uop(expr, Ops.ADD)):
//       # if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
//       candidates.append([(Xi, UOp.variable("fake", 1, Xi.vmax, Xi.dtype)) for Xi in split_uop(expr, Ops.ADD)])
//     # try checking the whole clause
//     if expr in uop.sparents:
//       candidates.append([(expr, UOp.variable("fake", expr.vmin if v[0] is None else v[0], expr.vmax if v[1] is None else v[1], expr.dtype))])

//     for candidate in candidates:
//       # if every branch in candidate gives the same simplified uop, we can rewrite the uop
//       newuops = [uop.substitute({X:newX}).simplify().substitute({newX:X}).simplify() for X,newX in candidate]
//       if uop.op is Ops.VECTORIZE and len(uop.src) == 2:
//         if all_same([uops.src[0] for uops in newuops]): uop = uop.replace(src=(newuops[0].src[0], uop.src[1]))
//         if all_same([uops.src[1] for uops in newuops]): uop = uop.replace(src=(uop.src[0], newuops[0].src[1]))
//       elif all_same(newuops): uop = newuops[0]

//   return uop

// def _valid_priority(v: UOp, valids:List[UOp]):
//   # we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
//   try: return sum(-1 if parse_valid(v)[0] in other.parents else 0 for other in valids)
//   except ValueError: return 0

// def simplify_valid(valid:UOp) -> Optional[UOp]:
//   ret:List[UOp] = []
//   something_changed = False
//   valids = list(split_uop(valid, Ops.AND))
//   for stmt in sorted(valids, key=lambda v: _valid_priority(v, valids)):
//     ret.append(newstmt if ret and (newstmt:=uop_given_valid(functools.reduce(operator.and_, ret), stmt)) is not None else stmt)
//     if ret[-1] is not stmt: something_changed = True
//   return functools.reduce(operator.and_, ret) if something_changed else None

// def max_var_const(x:UOp, c1:UOp, c2:UOp):
//   if x.vmin >= 0: return x*c1 if c1.arg >= c2.arg else x*c2
//   if x.vmax <= 0: return x*c2 if c1.arg >= c2.arg else x*c1

// def sint_to_uop(x:sint) -> UOp: return UOp.const(dtypes.int, x) if isinstance(x, int) else x

export const symbolicSimple = new PatternMatcher([
    //   // ** self folding **
    [UPat.var('x').add(0), (x) => x], // x+0 -> x
    [UPat.var('x').mul(1), (x) => x], // x*1 -> x
    [UPat.var('x').idiv(UPat.var('x')), (x) => x.const_like(1)], // x//x -> 1
    [UPat.var('x').idiv(1), (x) => x], // x//1 -> x
    [UPat.var('x').idiv(-1), (x) => -x], // x//-1 -> -x
    [UPat.var('x').div(UPat.var('x')), (x) => x.const_like(1)], // x/x -> 1
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
    [UPat.var('x').mul(0), (x) => x.const_like(typeof x.arg === 'number' && (isNaN(x.arg) || !isFinite(x.arg)) ? NaN : 0)],
    //   // ** constant folding **
    [new UPat({ op: GroupOp.ALU, name: 'a', src: new UPat({ op: [Ops.VCONST, Ops.CONST] }) }), (a) => a.const_like(exec_alu(a.op, a.dtype, a.src!.map((x) => x.arg), false))],
    //   // bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
    [UPat.var('x', dtypes.bool).mul(UPat.var('y', dtypes.bool)), (x, y) => x.bitwiseAnd(y)],
    [UPat.var('x', dtypes.bool).add(UPat.var('y', dtypes.bool)), (x, y) => x.bitwiseOr(y)],
    [UPat.var('x', dtypes.bool).maximum(UPat.var('y', dtypes.bool)), (x, y) => x.bitwiseOr(y)],
    //   // *** cast ***
    [new UPat({ op: Ops.CAST, name: 'root', src: UPat.cvar('c') }), (root, c) => root.const_like(c.arg)],
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
        [new UPat({ op: GroupOp.ALU, name: 'x' }), (x) => x.vmin == x.vmax ? x.const_like(x.vmin) : null],
        //   // max folding
        [UPat.var('x').maximum(UPat.var('y')), (x, y) => x.vmax <= y.vmin ? x.vmin >= y.vmax ? x : y : null],
        //   // TODO: why does this rule break beautiful_mnist?
        //   //((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
        [UPat.var('x').mul(UPat.cvar('c1')).maximum(UPat.var('x').mul(UPat.cvar('c2'))), max_var_const],
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
        [(UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)), (x, c0, c1) => c0.arg < 0 && c0.arg != -1 && c1.arg <= 0 ? x.neg().lt(-Math.floor(-c1.arg / -c0.arg)) : null],
        //   // x//c0<c1 for positive int c0
        [(UPat.var('x', dtypes.ints).idiv(UPat.cvar('c0', undefined, false))).lt(UPat.cvar('c1', undefined, false)), (x, c0, c1) => c0.arg > 0 ? x.lt(c1.arg * c0.arg) : null],
        //   // mul add lt
        [((UPat.cvar('c0', undefined, false).mul(UPat.var('x'))).add(UPat.var('x2'))).lt(UPat.cvar('c1', undefined, false)), (x, x2, c0, c1) => c1.arg % c0.arg == 0 && c0.arg > x2.vmax && x2.vmin >= 0 ? x.lt(c1.idiv(c0)) : null],
        //   // ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
        [new UPat({ op: Ops.ADD, src: [UPat.var('x'), UPat.cvar('c1')] }).add(UPat.var('y')), (x, c1, y) => (x.add(y)).add(c1)],
        [new UPat({ op: Ops.MUL, src: [UPat.var('x'), UPat.cvar('c1')] }).mul(UPat.var('y')), (x, c1, y) => (x.mul(y)).mul(c1)],
        //   // *** rules from symbolic ***
        //   // unrolled arange div folding
        [new UPat({ op: Ops.ADD, name: 'divs', src: [new UPat({}), new UPat({ op: Ops.IDIV })] }), fold_unrolled_divs],
        //   // generic lt folding
        [UPat.var('x', dtypes.sints).lt(UPat.cvar('c', undefined, false)), (x, c) => 0 < c.arg ? lt_folding(x, c.arg) : null],
        //   // canonicalize a simplex with positive coefficients > 0
        //   // not x < 1 -> X > 0
        [UPat.var('x', dtypes.ints).lt(1).ne(true), (x) => {
            const newx = canonicalize_simplex(x)
            return newx !== null ? newx.lt(1).ne(true) : null
        }],
        //   // ** div **
        //   // // div folding
        [UPat.var('x', dtypes.sints).idiv(UPat.cvar('c', undefined, false)), (x, c) => {
            const newx = div_folding(x, c.arg)
            return 0 < c.arg && newx !== null ? newx : null
        }],
        //   // ** mod **
        //   // mod folding
        [UPat.var('x').mod(UPat.cvar('c', undefined, false)), (x, c) => {
            const newx = mod_folding(x, c.arg)
            return 0 < c.arg && newx !== null ? newx : null
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
