import { type ConstType, DType, dtypes, ImageDType, PtrDType, truncate } from './dtype.ts'
import { all_same, assert, bytesToString, checkCached, counter, DataClass, divmod, Enum, isEq, isInf, isLessThan, isNone, isNotNone, isSubset, listStr, mathGcd, partition, permutations, prod, raise, range, setDefault, setMap, sha256, sum, zip } from './helpers.ts'
import { ShapeTracker } from './shape/shapetracker.ts'
import { argfix } from './helpers.ts'

export type Variable = UOp
export type ConstLike<This = never> = ConstType<This> | Variable | ConstType[]

export class SimpleMathTrait<T extends SimpleMathTrait<T>> {
  //   # required to implement
  alu = (_arg: Ops, ..._src: T[]): T => raise('Not implemented')
  const_like = (_b: ConstLike): T => raise('Not implemented')

  //   # great functions you get!
  ufix = (x: ConstType<T>): T => x instanceof MathTrait ? x : this.const_like(x as any) //ignoring this error, cause not sure
  _binop = (op: Ops, x: ConstType<T>, reverse: boolean) => reverse ? this.ufix(x).alu(op, this as any) : this.alu(op, this.ufix(x))
  logical_not = () => this.ne(true)
  neg = () => {
    const dtype = 'dtype' in this && this.dtype instanceof DType ? this.dtype : undefined
    if (isNone(dtype)) throw new Error(`MathTraits __neg__ requires a dtype, ${this}`)
    return isEq(dtype.scalar(), dtypes.bool) ? this.logical_not() : this.mul(-1)
  }
  add = (x: ConstType<T>, reverse = false) => this._binop(Ops.ADD, x, reverse)
  mul = (x: ConstType<T>, reverse = false) => this._binop(Ops.MUL, x, reverse)
  bitwise_and = (x: ConstType<T>, reverse = false) => this._binop(Ops.AND, x, reverse)
  bitwise_or = (x: ConstType<T>, reverse = false) => this._binop(Ops.OR, x, reverse)
  xor = (x: ConstType<T>, reverse = false) => this._binop(Ops.XOR, x, reverse)
  idiv = (x: ConstType<T>, reverse = false) => this._binop(Ops.IDIV, x, reverse)
  sub = (x: ConstType<T>, reverse = false) => reverse ? this.ufix(x).alu(Ops.ADD, this.neg()) : this.alu(Ops.ADD, typeof x === 'number' || typeof x === 'boolean' || typeof x === 'bigint' ? this.ufix(-x) : x.neg())
  div = (x: ConstType<T>, reverse = false) => reverse ? this.ufix(x).mul(this.alu(Ops.RECIP)) : this.mul(this.ufix(x).alu(Ops.RECIP))

  lt = (x: ConstType<T>) => this.alu(Ops.CMPLT, this.ufix(x))
  gt = (x: ConstType<T>) => this.ufix(x).alu(Ops.CMPLT, this as any)
  ge = (x: ConstType<T>) => this.lt(x).logical_not()
  le = (x: ConstType<T>) => this.gt(x).logical_not()
  ne = (x: ConstType<T>) => this.alu(Ops.CMPNE, this.ufix(x))
  eq = (x: ConstType<T>) => this.ne(x).logical_not()
}

export class MathTrait<T extends MathTrait<any>> extends SimpleMathTrait<T> {
  // TODO: move to Tensor when new backward is done (tinygrad)
  lshift = (x: ConstType<T>, reverse = false) => this._binop(Ops.SHL, x, reverse)
  rshift = (x: ConstType<T>, reverse = false) => this._binop(Ops.SHR, x, reverse)

  //   # not in Tensor
  mod = (x: ConstType<T>, reverse = false) => !reverse ? this.alu(Ops.MOD, this.ufix(x)) : this.ufix(x).alu(Ops.MOD, this)
  maximum = (x: ConstType<T>) => this.alu(Ops.MAX, this.ufix(x))
  minimum = (x: ConstType<T>) => this.neg().maximum(typeof x === 'number' || typeof x === 'boolean' || typeof x === 'bigint' ? this.ufix(-x) : x.neg()).neg()
  where = (x: ConstType<T>, y: ConstType<T>) => this.alu(Ops.WHERE, this.ufix(x), this.ufix(x).ufix(y))
  threefry = (seed: ConstType<T>) => this.alu(Ops.THREEFRY, this.ufix(seed))
  reciprocal = () => this.alu(Ops.RECIP)
  sqrt = () => this.alu(Ops.SQRT)
  sin = () => this.alu(Ops.SIN)
  log2 = () => this.alu(Ops.LOG2)
  exp2 = () => this.alu(Ops.EXP2)
}

// # the order of these Ops controls the order of the toposort
export class Ops<Name extends string = string, Value extends number = number> extends Enum {
  private static VALUES: Ops[] = []
  static values = () => [...Ops.VALUES]
  constructor(name: Name, value: Value) {
    super(name, value)
    Ops.VALUES.push(this)
    assert(value === Ops.VALUES.length)
  }
  // uops that aren't rendered
  static readonly SINK = new Ops('SINK', 1)
  static readonly CONTIGUOUS = new Ops('CONTIGUOUS', 2)
  static readonly PRELOAD = new Ops('PRELOAD', 3)

  // MetaOps
  static readonly COPY = new Ops('COPY', 4)
  static readonly EMPTY = new Ops('EMPTY', 5)
  static readonly BUFFER_VIEW = new Ops('BUFFER_VIEW', 6)

  // blocks in linearizer
  static readonly BLOCK = new Ops('BLOCK', 7)
  static readonly BLOCKSTART = new Ops('BLOCKSTART', 8)
  static readonly BLOCKFORK = new Ops('BLOCKFORK', 9)
  static readonly BLOCKEND = new Ops('BLOCKEND', 10)

  // misc ops
  static readonly EXPAND = new Ops('EXPAND', 11)
  static readonly CONTRACT = new Ops('CONTRACT', 12)
  static readonly VIEW = new Ops('VIEW', 13)
  static readonly DEFINE_GLOBAL = new Ops('DEFINE_GLOBAL', 14)
  static readonly BUFFER = new Ops('BUFFER', 15)
  static readonly DEFINE_VAR = new Ops('DEFINE_VAR', 16)
  static readonly DEFINE_LOCAL = new Ops('DEFINE_LOCAL', 17)
  static readonly DEFINE_ACC = new Ops('DEFINE_ACC', 18)
  static readonly VALID = new Ops('VALID', 19)
  static readonly SPECIAL = new Ops('SPECIAL', 20)
  static readonly NOOP = new Ops('NOOP', 21)

  // reduce
  static readonly REDUCE_AXIS = new Ops('REDUCE_AXIS', 22)

  // helper ops
  static readonly GEP = new Ops('GEP', 23)
  static readonly VECTORIZE = new Ops('VECTORIZE', 24)

  // UnaryOps
  static readonly CAST = new Ops('CAST', 25)
  static readonly BITCAST = new Ops('BITCAST', 26)
  static readonly EXP2 = new Ops('EXP2', 27)
  static readonly LOG2 = new Ops('LOG2', 28)
  static readonly SIN = new Ops('SIN', 29)
  static readonly SQRT = new Ops('SQRT', 30)
  static readonly RECIP = new Ops('RECIP', 31)
  static readonly NEG = new Ops('NEG', 32)

  // load/store before math
  static readonly LOAD = new Ops('LOAD', 33)
  static readonly STORE = new Ops('STORE', 34)

  // early INDEX
  static readonly INDEX = new Ops('INDEX', 35)

  // math ops
  static readonly WMMA = new Ops('WMMA', 36)

  // BinaryOps
  static readonly ADD = new Ops('ADD', 37)
  static readonly MUL = new Ops('MUL', 38)
  static readonly IDIV = new Ops('IDIV', 39)
  static readonly MAX = new Ops('MAX', 40)
  static readonly MOD = new Ops('MOD', 41)
  static readonly CMPLT = new Ops('CMPLT', 42)
  static readonly CMPNE = new Ops('CMPNE', 43)
  static readonly XOR = new Ops('XOR', 44)
  static readonly SHL = new Ops('SHL', 45)
  static readonly SHR = new Ops('SHR', 46)
  static readonly OR = new Ops('OR', 47)
  static readonly AND = new Ops('AND', 48)
  static readonly THREEFRY = new Ops('THREEFRY', 49)
  static readonly SUB = new Ops('SUB', 50)
  static readonly FDIV = new Ops('FDIV', 51)

  // TernaryOps
  static readonly WHERE = new Ops('WHERE', 52)
  static readonly MULACC = new Ops('MULACC', 53)

  // assignment ops
  static readonly ASSIGN = new Ops('ASSIGN', 54)
  static readonly BIND = new Ops('BIND', 55)

  // control flow ops
  static readonly BARRIER = new Ops('BARRIER', 56)
  static readonly RANGE = new Ops('RANGE', 57)
  static readonly IF = new Ops('IF', 58)
  static readonly ENDRANGE = new Ops('ENDRANGE', 59)
  static readonly ENDIF = new Ops('ENDIF', 60)

  // consts last!
  static readonly VCONST = new Ops('VCONST', 61)
  static readonly CONST = new Ops('CONST', 62)
}

export class GroupOp {
  static Unary = [Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG]
  static Binary = [Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV]
  static Ternary = [Ops.WHERE, Ops.MULACC]
  static ALU = [...this.Unary, ...this.Binary, ...this.Ternary]

  static Irreducible = [Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE]

  //   # meta ops
  static Meta = [Ops.COPY, Ops.EMPTY, Ops.BUFFER_VIEW]
  static Buffer = [Ops.LOAD, Ops.PRELOAD, Ops.STORE, Ops.VALID]
  static Block = [Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKFORK, Ops.BLOCKSTART]

  //   # BinaryOps that can be flipped
  static Commutative = [Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.AND, Ops.OR]

  //   # do not preserve f(0) = 0
  static UnsafePad = [Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV]
}

// # https://en.wikipedia.org/wiki/Identity_element
export const identity_element = (op: Ops, dt: DType) => dtypes.as_const(new Map([[Ops.ADD, 0], [Ops.MUL, 1], [Ops.MAX, dtypes.min(dt)]]).get(op)!, dt)

export const can_pad = (u: UOp, edges: Map<UOp, UOp>, visisted: Set<UOp>): boolean => {
  if (GroupOp.UnsafePad.includes(u.op)) return false
  if ((u.src.length === 2 && edges.has(u.src[0])) || visisted.has(u)) return true
  visisted.add(u)
  return u.src.every((x) => can_pad(x.base, edges, visisted))
}
export const END_FOR_UOP = new Map([[Ops.IF, [Ops.STORE, Ops.ENDIF]], [Ops.RANGE, [Ops.ASSIGN, Ops.ENDRANGE]]])

// With True as the default, this matches the old symbolic behavior
export const resolve = (x: ConstType<UOp>, def = true) => {
  if (!(x instanceof UOp)) return Boolean(x)
  if (x.dtype.name !== 'bool') throw new Error('UOp in resolve must be bool')
  // NOTE: generating the text for the exception is expensive, so we do this
  const sx = x.simplify()
  return sx.vmin === sx.vmax ? Boolean(sx.vmin) : def
}

// # smax/smin are replacements for max/min that preserve symbolic
const _suop = (lst: sint[], uop_fxn: (...x: UOp[]) => UOp, python_fxn: (...a: number[]) => number): sint => {
  const [uops, nums] = partition(lst, (x) => x instanceof UOp) as [UOp[], number[]]
  return ssimplify((nums ? [...uops, python_fxn(...nums)] as UOp[] : []).reduce((acc, x) => uop_fxn(acc, x)))
}
export const smax = (...lst: (sint | sint[])[]) => _suop(argfix(...lst), (...x) => x.reduce((acc, x) => acc.maximum(x)), (...x) => Math.max(...x))
export const smin = (...lst: (sint | sint[])[]) => _suop(argfix(...lst), (...x) => x.reduce((acc, x) => acc.minimum(x)), (...x) => Math.min(...x))

export const ssimplify = (uop: UOp) => uop instanceof UOp ? uop.ssimplify() : uop
export const sym_infer = (uop: sint, varVals: Map<UOp, number>): number => uop instanceof UOp ? uop.symInfer(varVals) : uop

type UOpInput = { op: Ops; dtype?: DType; src?: UOp[]; arg?: any }
type UOpTuple = [number, any, DType, UOpTuple[]]
export class UOp extends MathTrait<UOp> {
  static ucache: Record<string, UOp> = {}
  constructor(public op: Ops, public dtype = dtypes.void, public src: UOp[] = [], public arg?: any) {
    super()
    // KAREL: this is a hack, for some reason sometime it sends in int
    if (typeof this.op === 'number') op = this.op = Ops.values().find((x) => x.value === op as any)!
    return checkCached({ op, dtype, src, arg }, UOp.ucache, this)
  }
  override toString = (indent = 2): string => {
    const src = !this.src ? 'undefined' : this.src.length === 0 ? '[]' : `[\n${' '.repeat(indent)}${this.src.map((s) => s.toString(indent + 2)).join(',\n' + ' '.repeat(indent))}\n${' '.repeat(indent - 2)}]`
    return `new UOp(${this.op.toString()}, ${this.dtype}, ${src}, ${listStr(this.arg)})`
  };
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  __reduce__ = () => [UOp, [this.op, this.dtype, this.src, this.arg]] as const
  replace = (args: Partial<UOpInput>) => new UOp(args.op || this.op, args.dtype || this.dtype, args.src || this.src, args.arg || this.arg)
  get key(): string {
    let data = JSON.stringify([this.op, this.dtype, this.arg])
    for (const s of this.src) data += s.key
    return bytesToString(sha256(data))
  }
  get toposort(): Set<UOp> {
    let nodes = new Set<UOp>()
    // NOTE: this is a lot faster than the comprehension in parents
    for (const parent of this.src) nodes = nodes.union(parent.toposort)
    nodes.add(this)
    return nodes
  }
  tuplize = (): UOpTuple => [this.op.value, this.arg, this.dtype, this.src.map((x) => x.tuplize())]

  //   # *** uop shape stuff ***
  get has_st() {
    return ![Ops.DEFINE_LOCAL, Ops.DEFINE_GLOBAL, Ops.BUFFER, Ops.CONST, Ops.DEFINE_VAR].includes(this.op)
  }

  get st(): undefined | ShapeTracker {
    if (this.op === Ops.VIEW) return this.arg
    // buffer ops can have a non contiguous shapetracker
    let src_sts = this.src.filter((x) => x.op === Ops.VIEW).map((x) => x.st!)
    if (GroupOp.Buffer.includes(this.op) && src_sts.length !== 0) return src_sts[0]
    src_sts = this.src.filter((x) => isNotNone(x.st)).map((x) => x.st!)
    if (src_sts.length === 0) return undefined
    assert(all_same(src_sts.map((x) => x.shape)), `UOp parents must have the same shape ${this} ${listStr(src_sts.map((x) => x.shape))}`)
    // all other ops have a contiguous shapetracker
    return ShapeTracker.from_shape([Ops.REDUCE_AXIS, Ops.WMMA].includes(this.op) ? src_sts[0].reduce(this.axis_arg) : src_sts[0].shape)
  }
  get shape() {
    return this.st!.shape
  }
  get size() {
    return this.op === Ops.BUFFER ? this.arg[1][1] : this.st!.size
  }
  get full_shape(): sint[] {
    return this.op === Ops.VIEW ? this.shape : zip(...this.src.filter((x) => x.has_st).map((x) => x.full_shape)).map((x) => smax(x))
  }
  //   # *** uop evaluation ***

  simplify = (): UOp => graph_rewrite(this, symbolic)
  ssimplify = (): UOp => {
    const ret = this.simplify()
    return ret.op === Ops.CONST ? ret.arg : ret
  }
  _eval = <T extends new (...args: any[]) => void>(dtypes: DType[], expectedType: T): InstanceType<T> => {
    if (!dtypes.includes(this.dtype)) throw new Error(`eval with wrong dtype ${this}`)
    const simpleThis = this.simplify()
    const [vmin, vmax] = simpleThis._minMax()
    if (!isEq(vmin, vmax)) throw new Error(`eval failed to be a single number, range is ${vmin} to ${vmax} in ${simpleThis.render()}`)
    // if ((vmin instanceof expectedType)) throw new Error(`vmin is wrong dtype ${typeof vmin} != ${expectedType}`)
    return vmin as InstanceType<T>
  }
  __bool__ = () => this._eval([dtypes.bool], Boolean)
  __int__ = () => this._eval(dtypes.ints, Number)
  __float__ = () => this._eval(dtypes.floats, Number)
  substitute = (dvars: Map<UOp, UOp>) => {
    return graph_rewrite(this, _substitute, dvars, true)
  }

  //   # *** uop syntactic sugar ***
  get st_arg(): ShapeTracker {
    if (!(GroupOp.Buffer.includes(this.op))) throw new Error(`st_arg called on ${this.op.toString()}`)
    const ret = this.src[this.op === Ops.VALID ? 0 : 1]
    if (ret.op !== Ops.VIEW) throw new Error(`st_arg trying to return ${ret}`)
    return ret.arg
  }
  get axis_arg() {
    if (![Ops.REDUCE_AXIS, Ops.WMMA].includes(this.op)) throw new Error(`axis_arg called on ${this.op}`)
    const ret = this.op === Ops.REDUCE_AXIS ? this.arg[1] : this.arg[7]
    if (!(Array.isArray(ret) && ret.every((x) => typeof x === 'number'))) throw new Error(`axis_arg trying to return ${ret}`)
    return ret
  }
  static sink = (...srcs: UOp[]) => new UOp(Ops.SINK, dtypes.void, [...srcs])
  index = (idx: UOp, valid?: UOp) => new UOp(Ops.INDEX, this.dtype, isNotNone(valid) ? [this, idx, valid] : [this, idx])
  override const_like = (b: ConstLike<UOp>) => (isNone(this.st) ? UOp.const(this.dtype, b) : UOp.const_with_shape(this.dtype, b, this.shape)) as UOp
  broadcast = (count: number) => {
    if (this.dtype.count !== 1) throw new Error(`dtype.count !==1`)
    if (count === 1) return this
    return new UOp(Ops.VECTORIZE, this.dtype.vec(count), range(count).map(() => this))
  }
  cast = (dtype: DType) => new UOp(Ops.CAST, dtype, [this])
  bitcast = (dtype: DType) => new UOp(Ops.BITCAST, dtype, [this])
  gep = (i: number[] | number) => {
    if (!Array.isArray(i)) {
      // NOTE: these are just shortcuts to not have to create and fold later
      if (this.op === Ops.VECTORIZE) return this.src[i]
      if (this.op === Ops.VCONST) return UOp.const(this.dtype.scalar(), this.arg[i])
      if (this.op === Ops.CONST) return UOp.const(this.dtype.scalar(), this.arg)
      i = [i]
    }
    if (this.dtype.vcount === i.length && isEq(i, range(i.length)) || isEq(this.dtype, dtypes.void)) return this
    return new UOp(Ops.GEP, i.length > 1 ? this.dtype.scalar().vec(i.length) : this.dtype.scalar(), [this], i)
  }
  load = (src: UOp[], kwargs?: Partial<UOpInput>) => new UOp(kwargs?.op || Ops.LOAD, kwargs?.dtype, kwargs?.src || [this, ...src], kwargs?.arg)
  store = (src: UOp[], kwargs?: Partial<UOpInput>) => new UOp(kwargs?.op || Ops.STORE, kwargs?.dtype || dtypes.void, kwargs?.src || [this, ...src], kwargs?.arg)
  override alu = (arg: Ops, ...src: UOp[]): UOp => {
    let out_dtype = [this, ...src].at(-1)!.dtype
    if ([Ops.CMPLT, Ops.CMPNE].includes(arg)) out_dtype = out_dtype.count > 1 ? dtypes.bool.vec(out_dtype.count) : dtypes.bool
    return new UOp(arg, out_dtype, [this, ...src]) as UOp
  }
  static const = (dtype: DType, b: ConstLike) => {
    if (b instanceof UOp) return b.unbind()[0]
    if (Array.isArray(b) && all_same(b)) b = b[0]
    return new UOp(Array.isArray(b) ? Ops.VCONST : Ops.CONST, dtype, undefined, dtypes.as_const(b, dtype))
  }
  static int = (b: number) => UOp.const(dtypes.int, b)
  static bool = (b: boolean) => UOp.const(dtypes.bool, b)
  static float = (b: number) => UOp.const(dtypes.float, b)
  static range = (dtype: DType, start: sint, end: sint, idx: number) => new UOp(Ops.RANGE, dtype, [sint_to_uop(start), sint_to_uop(end)], idx)
  r = (op: Ops, axis: number[]) => new UOp(Ops.REDUCE_AXIS, this.dtype, [this], [op, axis])
  assign = (x: UOp) => new UOp(Ops.ASSIGN, this.dtype, [this, x])
  contiguous = () => new UOp(Ops.CONTIGUOUS, this.dtype, [this])

  //   # *** from LazyBuffer ***

  static const_with_shape = (dtype: DType, val: ConstLike, shape: sint[]): UOp => {
    return new UOp(Ops.VALID, dtypes.bool, [ShapeTracker.from_shape([]).reshape(range(shape.length).map(() => 1)).expand(shape).to_uop()]).where(UOp.const(dtype, val), 0)
  }
  //   # *** uop movement ops ***
  get base(): UOp {
    return this.op === Ops.VIEW && this.src.length === 1 && this.src[0].op !== Ops.BUFFER ? this.src[0] : this
  }

  view = (new_st: ShapeTracker): UOp => {
    assert(isNotNone(this.st) && isNotNone(this.base.st), `must have shape ${this}`)
    const ret = new UOp(Ops.VIEW, this.dtype, [this.base], new_st)
    // instant folding rules
    if (this.st?.size === 0 || (isNotNone(new_st.views.at(-1)!.mask) && new_st.views.at(-1)!.mask?.some((x) => sub(x[1], x[0]) === 0))) return ret.const_like(0)
    if (new_st.contiguous && isEq(this.base.st?.shape, new_st.shape)) return this.base
    return ret
  }
  reshape = (arg: sint[]) => this.view(this.st!.reshape(arg))
  pad = (arg: [sint, sint][]) => this.view(this.st!.pad(arg))
  expand = (arg: sint[]) => this.view(this.st!.expand(arg))
  permute = (arg: number[]) => this.view(this.st!.permute(arg))
  shrink = (arg: [sint, sint][]) => this.view(this.st!.shrink(arg))
  stride = (arg: number[]) => this.view(this.st!.stride(arg))

  //   # *** uop Buffer stuff ***
  static buffer_num = counter(0)
  static new_buffer = (device: string, size: number, dtype: DType) => new UOp(Ops.BUFFER, dtype.ptr(), [], [UOp.buffer_num.next().value, [device, size, dtype]])
  get device(): string {
    return this._device!
  }
  get _device(): string | undefined {
    const dsrcs = this.src.filter((x) => isNotNone(x._device))
    return this.op === Ops.BUFFER ? this.arg[1][0] : dsrcs.length !== 0 ? dsrcs[0]._device : undefined
  }
  get buf_uop(): UOp {
    if (this.op === Ops.BUFFER) return this
    assert([...GroupOp.Buffer, Ops.ASSIGN, Ops.VIEW].includes(this.op) && this.src[0].op === Ops.BUFFER, `buf_uop called on ${this.op}`)
    return this.src[0]
  }
  //   # *** uop Variable stuff ***

  static variable = (name: string, minVal: ConstType<UOp> = dtypes.min(dtypes.int), maxVal: ConstType<UOp> = dtypes.max(dtypes.int), dtype = dtypes.int) => {
    assert(!(minVal instanceof UOp) && !(maxVal instanceof UOp), `can't create Variable ${name} with ${minVal}/${maxVal}`)
    return new UOp(Ops.DEFINE_VAR, dtype, undefined, [name, minVal, maxVal])
  }
  get expr() {
    assert(this.op === Ops.DEFINE_VAR, `op is ${this.op}, need DEFINE_VAR`)
    return this.arg[0]
  }
  bind = (val: number) => {
    assert(this.op === Ops.DEFINE_VAR, `op is ${this.op}, need DEFINE_VAR`)
    assert(this.arg[1] <= val && val <= this.arg[2], `bind ${val} not in range [${this.arg[1]}, ${this.arg[2]}]`)
    return new UOp(Ops.BIND, this.dtype, [this, this.const_like(val)])
  }
  unbind = (): [Variable, number] => {
    assert(this.op === Ops.BIND && this.src[0].op === Ops.DEFINE_VAR && this.src[1].op === Ops.CONST, `can't unbind ${this}`)
    return [this.src[0], this.src[1].arg]
  }
  val = () => this.unbind()[1]
  vars = (): UOp[] => {
    const sparents = [...this.toposort]
    const boundVars = new Set(sparents.filter((x) => x.op === Ops.BIND && x.src[0].op === Ops.DEFINE_VAR))
    const boundVarBase = new Set([...boundVars].map((x) => x.src[0]))
    const allVars = new Set(sparents.filter((x) => x.op === Ops.DEFINE_VAR))
    return [...boundVars.union(new Set([...allVars].filter((x) => !boundVarBase.has(x))))]
  }
  variables = (): Variable[] => {
    const stVars: Variable[] = [...this.toposort].filter((x) => GroupOp.Buffer.includes(x.op)).flatMap((x) => x.st_arg.vars())
    const idk = new Set(this.vars().map((x) => x.op !== Ops.DEFINE_VAR ? x.unbind()[0] : x))
    return [...new Set([...stVars, ...idk])].sort((a, b) => b.arg - a.arg)
  }

  //   # *** uop symbolic stuff ***

  /**largest known int that divides this */
  constFactor = (): number => {
    if (this.op === Ops.CONST) return this.arg
    if (this.op === Ops.VCONST) return mathGcd(...this.arg)
    if (this.op === Ops.ADD) return mathGcd(this.src[0].constFactor(), this.src[1].constFactor())
    if (this.op === Ops.MUL) return this.src[1].op === Ops.CONST ? this.src[0].op === Ops.CONST ? this.src[0].arg : this.src[1].arg : 1
    return 1
  }
  divides = (v: number): UOp | undefined => {
    if (v === 1) return this
    if (this.op === Ops.CONST) return this.arg % v === 0 ? this.const_like(idiv(this.arg, v)) : undefined
    if (this.op === Ops.VCONST) return this.arg.every((x: number) => x % v === 0) ? this.const_like(this.arg.map((x: number) => idiv(x, v))) : undefined

    if (this.op === Ops.ADD) {
      const d0 = this.src[0].divides(v)
      const d1 = this.src[1].divides(v)
      return d0 !== undefined && d1 !== undefined ? d0.add(d1) : undefined
    }
    if (this.op === Ops.MUL) {
      const d0 = this.src[0].divides(v)
      const d1 = this.src[1].divides(v)
      if (d0 !== undefined) return d0.mul(this.src[1])
      if (d1 !== undefined) return this.src[0].mul(d1)
    }
    return undefined // generic None if we aren't sure
  }
  get vmin() {
    return this._minMax()[0]
  }
  get vmax() {
    return this._minMax()[1]
  }
  // Actually can return boolean as well, but types don't like it
  _minMax = (): [number, number] => {
    if (GroupOp.Binary.includes(this.op) && !dtypes.is_float(this.dtype)) {
      const [[s0_vmin, s0_vmax], [s1_vmin, s1_vmax]] = [this.src[0]._minMax(), this.src[1]._minMax()]
      if (this.op === Ops.ADD) return [s0_vmin + s1_vmin, s0_vmax + s1_vmax]
      if (this.op === Ops.MUL) {
        const vals = [s0_vmin * s1_vmin, s0_vmin * s1_vmax, s0_vmax * s1_vmin, s0_vmax * s1_vmax]
        return [Math.min(...vals), Math.max(...vals)]
      }
      if (this.op === Ops.MOD && s1_vmin > 0) return [0, s1_vmax - 1]
      if (this.op === Ops.IDIV) {
        if (s1_vmin === s1_vmax) { // min/max are equal in a CONST
          if (s1_vmin > 0) return [idiv(s0_vmin, s1_vmin), idiv(s0_vmax, s1_vmin)]
          if (s1_vmin < 0 && s0_vmin >= 0) return [-idiv(s0_vmax, -s1_vmin), -idiv(s0_vmin, -s1_vmin)]
        }
        // don't know exact bounds, but know the sign
        if ((s0_vmax <= 0 && s1_vmin < 0) || (s0_vmin >= 0 && s1_vmin > 0)) return [0, Number(dtypes.max(this.dtype))]
        if ((s0_vmax <= 0 && s1_vmin > 0) || (s0_vmin >= 0 && s1_vmin < 0)) return [Number(dtypes.min(this.dtype)), 0]
      }
      if (this.op === Ops.MAX) return [Math.max(s0_vmin, s1_vmin), Math.max(s0_vmax, s1_vmax)]
      if (this.op === Ops.CMPLT) return [Number(s0_vmax < s1_vmin), Number(s0_vmin < s1_vmax)]
      if (this.op === Ops.CMPNE) return [Number((s0_vmax < s1_vmin) || (s1_vmax < s0_vmin)), Number(!(s0_vmin === s0_vmax && s0_vmax === s1_vmin && s1_vmin === s1_vmax))]
      if (this.dtype === dtypes.bool) {
        if (this.op === Ops.OR) return [Number(s0_vmin || s1_vmin), Number(s0_vmax || s1_vmax)]
        if (this.op === Ops.AND) return [Number(s0_vmin && s1_vmin), Number(s0_vmax && s1_vmax)]
      }
    }
    // float has NAN issue and we use explicit NAN in transcendental
    if (this.op === Ops.WHERE && dtypes.is_int(this.dtype)) return [Math.min(this.src[1].vmin, this.src[2].vmin), Math.max(this.src[1].vmax, this.src[2].vmax)]
    // NOTE: returned UOp is assumed to be CONST
    if (this.op === Ops.DEFINE_VAR && this.arg) return [this.arg[1], this.arg[2]]
    if (this.op === Ops.RANGE) return [this.src[0].vmin, (this.src[1].sub(1)).vmax]
    if (this.op === Ops.BIND) return this.src[0]._minMax() // ignore the bound value
    if ([Ops.EXPAND, Ops.VECTORIZE].includes(this.op)) return [Math.min(...this.src.map((x) => x.vmin)), Math.max(...this.src.map((x) => x.vmax))]
    // TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if (this.op === Ops.SPECIAL) return [0, typeof this.arg[1] === 'number' ? (this.arg[1] - 1) : Number(dtypes.max(this.dtype))]
    if (this.op === Ops.CONST) return [this.arg, this.arg]
    if (this.op === Ops.VCONST) return [Math.min(...this.arg), Math.max(...this.arg)]
    return [Number(dtypes.min(this.dtype)), Number(dtypes.max(this.dtype))]
  }

  _sym_fxn = (): [(m: Record<string, number>) => number, string[]] => {
    const sthis = this.simplify()
    const varnames: string[] = [...sthis.toposort].filter((x) => x.op === Ops.DEFINE_VAR).map((x) => x.arg[0])
    // TODO: sanitize varnames, or don't use naked eval while staying fast
    return [eval(`({${varnames.join(',')}})=>${sthis.render()}`), varnames]
  }
  symInfer = (varVals: Map<UOp, number>) => {
    const [fxn, varnames] = this._sym_fxn()
    const args = Object.fromEntries(varVals.entries().filter(([k, _]) => varnames.includes(k.arg[0])).map(([k, v]) => [k.arg[0] as string, v]))
    return fxn(args)
  }

  render = (simplify = true): string => {
    const ret = graph_rewrite(simplify ? this.simplify() : this, renderer)
    return ret.op === Ops.NOOP ? ret.arg : ret.toString()
  }
}

@DataClass
export class KernelInfo {
  constructor(public local_dims = 0, public upcasted = 0, public dont_use_locals = false) {}
}

// # ***** ops in python *****
const safe_exp2 = (x: number) => {
  try {
    return 2 ** x
  } catch {
    return Infinity
  }
}
export const python_alu = new Map<Ops, (...x: number[]) => number>([
  [Ops.LOG2, (x) => x === 0 ? x > 0 ? Math.log2(2) : -Infinity : NaN],
  [Ops.EXP2, safe_exp2],
  [Ops.SQRT, (x) => x >= 0 ? Math.sqrt(x) : NaN],
  [Ops.RECIP, (x) => x !== 0 ? 1 / x : x >= 0 ? Infinity : -Infinity],
  [Ops.SIN, (x) => !isInf(x) ? Math.sin(x) : NaN],
  [Ops.NEG, (x) => -x],
  [Ops.ADD, (x, y) => x + y],
  [Ops.SUB, (x, y) => x - y],
  [Ops.MUL, (x, y) => x * y],
  [Ops.CMPNE, (x, y) => Number(x !== y)],
  [Ops.CMPLT, (x, y) => Number(x < y)],
  [Ops.XOR, (x, y) => x ^ y],
  [Ops.OR, (x, y) => x | y],
  [Ops.AND, (x, y) => x & y],
  [Ops.SHR, (x, y) => x >> y],
  [Ops.SHL, (x, y) => x << y],
  [Ops.MAX, (...args) => Math.max(...args)],
  [Ops.MOD, (x, y) => Math.abs(Math.trunc(x)) % Math.abs(Math.trunc(y)) * (x < 0 ? -1 : 1)],
  [Ops.IDIV, (x, y) => y !== 0 ? idiv(Math.abs(x), Math.abs(y)) * ((x * y < 0) ? -1 : 1) : x * Infinity],
  [Ops.MULACC, (x, y, z) => (x * y) + z],
  [Ops.WHERE, (x, y, z) => x ? y : z],
])

export const exec_alu = (op: Ops, dtype: DType, operands: number[], truncateOutput = true): any => {
  if (dtype.count > 1) return range(dtype.count).map((i) => exec_alu(op, dtype.scalar(), operands.map((x) => Array.isArray(x) ? x[i] : x)))
  const alu = python_alu.get(op)!(...operands)
  return truncateOutput ? truncate.get(dtype)!(alu) : alu
}
// # ***** uop helpers *****

export const print_uops = (uops: UOp[]) => {
  for (const [i, u] of uops.entries()) {
    const formattedParents = u.src.map((x) => uops.includes(x) ? x.op !== Ops.CONST ? uops.indexOf(x) : `${x.arg}` : '--')
    console.log(`${i.toString().padStart(4)} ${u.op.toString().padEnd(20)} ${u.dtype.toString().padEnd(30)} ${listStr(formattedParents).padEnd(32)} ${listStr(u.arg)}`)
  }
}

export const flops_mem = (uops: UOp[], ignoreIndexing = false): [UOp, UOp] => {
  let flops = uops[0].const_like(0)
  let mem = uops[0].const_like(0)
  let mults = uops[0].const_like(1)
  const multStack: UOp[] = []
  let dontCount = new Set<UOp>()
  if (ignoreIndexing) {
    for (const u of uops) {
      if ([Ops.LOAD, Ops.STORE].includes(u.op)) {
        dontCount = dontCount.union(u.src[0].toposort)
        if (u.src.length > 2) dontCount = dontCount.union(u.src[2].toposort)
      } else if (u.op === Ops.IF) dontCount = dontCount.union(u.src[0].toposort)
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
    else if (u.op === Ops.WMMA && !dontCount.has(u)) flops = add(flops, mul(idiv(mul(2, prod(u.arg[1])), u.arg[5]), mults))
  }
  return [flops, mem]
}
// # ***** pattern matcher *****
function getLocation(): [string, number] {
  const [file, line] = new Error().stack!.split('\n')[2]?.split('file://')[1]?.split(')')[0]?.split(':')
  return [file, Number(line)]
}
const lines = (fn: string): string[] => {
  return Deno.readFileSync(fn).toString().split('\n')
}

export type UPatInput = { op?: Ops | Ops[]; dtype?: DType | DType[]; src?: UPat | UPat[] | [UPat[]]; arg?: any; name?: string; allow_any_len?: boolean; location?: any; custom_early_reject?: Ops[] }
export class UPat extends MathTrait<UPat> {
  op?: Ops[]
  dtype?: DType[]
  _in_src?: UPat | UPat[] | [UPat[]]
  src?: UPat[][]
  allowed_len: number
  location: [string, number]
  early_reject: Ops[]
  constructor(op?: Ops | Ops[], dtype?: DType | DType[], src?: UPat | UPat[] | [UPat[]], public arg?: any, public name?: string, allow_any_len?: boolean, location?: any, public custom_early_reject?: Ops[]) {
    super()
    assert(isNone(op) || !(!Array.isArray(op) && Object.values(Ops).includes(op)) || !(Array.isArray(op) && Object.values(Ops).includes(op[0])), 'op must be Ops or tuple of Ops')
    this.op = Array.isArray(op) ? op : !isNone(op) ? [op] : undefined
    this.dtype = Array.isArray(dtype) ? dtype : !isNone(dtype) ? [dtype] : undefined
    this._in_src = src
    assert(this.name !== 'ctx', "UPat can't be named ctx")

    // try all permutations if it's a list (we use src[][])
    if (Array.isArray(src) && Array.isArray(src[0])) this.src = !all_same(src[0]) ? permutations(src[0]) : [src[0]]
    // only one if it's a tuple (we use src[])
    else if (Array.isArray(src)) this.src = [src as UPat[]]
    // repeat if it's a UPat
    else if (src instanceof UPat) this.src = [range(100).map(() => src!) as UPat[]] // KAREL: this is a hack

    // NOTE: This is here because we can't differentaite between list and tuple so we use Upat[][] to achieve the same thing as list. but after this part the difference isn't needed anymore so we convert back to UPat[]
    if (Array.isArray(src) && src?.length === 1 && Array.isArray(src[0])) src = src[0]

    this.allowed_len = (allow_any_len || src instanceof UPat || isNone(src)) ? -1 : src.length
    this.location = location || getLocation()

    if (isNotNone(custom_early_reject)) this.early_reject = custom_early_reject
    else {
      const upatMatch = src instanceof UPat ? [src] : (isNone(src) ? [] : this.src![0])
      this.early_reject = upatMatch.filter((pp) => isNotNone(pp.op) && pp.op.length === 1).map((pp) => pp.op![0])
    }
  }

  named = (name: string) => new UPat(this.op, this.dtype, this._in_src, this.arg, name, this.allowed_len === -1, undefined, this.custom_early_reject)

  static any = (src: UPat[]) => new UPatAny(undefined, undefined, src)

  static var = (name?: string, dtype?: DType | DType[]) => new UPat(undefined, dtype, undefined, undefined, name)
  static cvar = (name?: string, dtype?: DType, vec = true) => new UPat(vec ? [Ops.CONST, Ops.VCONST] : Ops.CONST, dtype, undefined, undefined, name)
  static const = (dtype?: DType | DType[], b?: ConstLike) => new UPat(Ops.CONST, dtype, undefined, b)

  //   # copied from UOp
  index = (idx: UPat, valid?: UPat) => new UPat(Ops.INDEX, this.dtype, isNotNone(valid) ? [this, idx, valid] : [this, idx])
  view = (st?: ShapeTracker, kwargs: Partial<UPatInput> = {}) => new UPat(kwargs.op || Ops.VIEW, kwargs.dtype || this.dtype, kwargs.src || [this], kwargs.arg || st, kwargs.name, kwargs.allow_any_len, kwargs.location, kwargs.custom_early_reject)
  cast = (dtype?: DType) => new UPat(Ops.CAST, dtype, [this])
  bitcast = (dtype?: DType) => new UPat(Ops.BITCAST, dtype, [this])
  gep = (i: number) => new UPat(Ops.GEP, undefined, [this], [i])
  load = (src?: UPat[], kwargs: Partial<UPatInput> = {}) => new UPat(kwargs.op || Ops.LOAD, kwargs.dtype, kwargs.src || [this, ...(src || [])], kwargs.arg, kwargs.name, kwargs.allow_any_len, kwargs.location, kwargs.custom_early_reject)
  store = (src: UPat[], kwargs: Partial<UPatInput> = {}) => new UPat(kwargs.op || Ops.STORE, kwargs.dtype || dtypes.void, kwargs.src || [this, ...src], kwargs.arg, kwargs.name, kwargs.allow_any_len, kwargs.location, kwargs.custom_early_reject)
  assign = (x: UPat) => new UPat(Ops.ASSIGN, this.dtype, [this, x])

  override const_like = (b: ConstLike): UPat => UPat.const(this.dtype, b)
  override alu = (op: Ops, ...src: UPat[]) => {
    const asrc = [this, ...src]
    return new UPat(op, [Ops.CMPLT, Ops.CMPNE].includes(op) ? dtypes.bool : asrc.at(-1)?.dtype, GroupOp.Commutative.includes(op) ? [asrc] : asrc)
  }

  printable = (): string => {
    try {
      return lines(this.location[0])[this.location[1] - 1].trim()
    } catch {
      return '<missing>'
    }
  }
  override toString = () => `new UPat(${listStr(this.op?.map((o) => o.toString()))}, ${listStr(this.dtype)}, ${listStr(this.src)}, ${listStr(this.arg)}, ${this.name}, ${this.allowed_len === 0})`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  match = (uop: UOp, store: Map<string, UOp>): Map<string, UOp>[] => {
    if (
      (isNotNone(this.op) && !this.op.includes(uop.op)) ||
      (isNotNone(this.name) && !isEq(setDefault(store, this.name, uop), uop)) ||
      (isNotNone(this.dtype) && !this.dtype.includes(uop.dtype) && !this.dtype.includes(uop.dtype.scalar())) ||
      (isNotNone(this.arg) && !isEq(this.arg, uop.arg)) ||
      (this.allowed_len !== -1 && uop.src.length !== this.allowed_len)
    ) return []
    if (isNone(this.src)) return [store]
    let res: Map<string, UOp>[] = []
    for (const vp of this.src) {
      let stores = [store]
      let new_stores: typeof stores = []
      for (const [uu, vv] of zip(uop.src, vp)) {
        for (const s of stores) new_stores = [...new_stores, ...vv.match(uu, s)]
        stores = new_stores
        new_stores = []
      }
      res = [...res, ...stores]
    }
    return res
  }
}

export class UPatAny extends UPat {
  override match = (uop: UOp, store: Map<string, UOp>) => {
    let ret: Map<string, UOp>[] = []
    for (const x of this.src?.[0] || []) {
      const match = x.match(uop, new Map(store))
      if (match) ret = [...ret, ...match]
    }
    return ret
  }
}

export class PatternMatcher<Args = Record<string, UOp>, Res = UOp | undefined, Fn extends ((args: Args) => Res) = (args: Args) => Res> {
  patterns: [UPat, Fn][]
  pdict = new Map<Ops, ([UPat, Fn, Set<Ops>, boolean][])>()
  constructor(patterns: [UPat, Fn][]) {
    this.patterns = patterns
    for (const [p, fxn] of this.patterns) {
      assert(isNotNone(p.op))
      for (const uop of p.op || []) setDefault(this.pdict, uop, []).push([p, fxn, new Set(p.early_reject), fxn.toString().includes('ctx')])
    }
  }

  add = <NewArgs, NewRes>(more: PatternMatcher<NewArgs, NewRes>) => new PatternMatcher<Args | NewArgs, Res | NewRes>([...this.patterns, ...more.patterns] as any)

  rewrite = (uop: UOp, ctx?: any): Res | undefined => {
    const ler = new Set(uop.src.map((u) => u.op))
    for (const [p, fxn, early_reject, hasCtx] of this.pdict.get(uop.op) || []) {
      const index = this.patterns.findIndex((pattern) => pattern[0] === p)
      if (!isSubset(ler, early_reject)) {
        continue
      }
      for (const match of p.match(uop, new Map())) {
        const ret = hasCtx ? fxn({ ctx, ...Object.fromEntries(match) } as any) : fxn(Object.fromEntries(match) as any)
        if (isNotNone(ret)) return ret
      }
    }
    return undefined
  }
}

// # *** simple graph rewrite engine ***
export class RewriteContext {
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
    const newN = isEq(newSrc, n.src) ? this.pm.rewrite(n, this.ctx) : new UOp(n.op, n.dtype, newSrc, n.arg)
    const ret = isNone(newN) ? n : this.rewrite(newN)
    this.replace.set(n, ret)
    return ret
  }
  bottom_up_rewrite = (n: UOp): UOp => {
    const rn = this.replace.get(n)
    if (isNotNone(rn)) return rn
    let new_n: UOp | undefined = n
    let last_n!: UOp
    while (new_n !== undefined) {
      ;[last_n, new_n] = [new_n, this.pm.rewrite(new_n, this.ctx)]
    }
    const new_src = last_n.src.map((x) => this.bottom_up_rewrite(x))
    const ret = isEq(new_src, last_n.src) ? last_n : this.bottom_up_rewrite(new UOp(last_n.op, last_n.dtype, new_src, last_n.arg))
    this.replace.set(n, ret)
    return ret
  }
}
export const graph_rewrite = (sink: UOp, pm: PatternMatcher<any, any>, ctx?: any, bottom_up = false): UOp => {
  return bottom_up ? new RewriteContext(pm, ctx).bottom_up_rewrite(sink) : new RewriteContext(pm, ctx).rewrite(sink)
}
// # ***** uop type spec *****

// # this is the matcher for the final rendered UOps
// # matcher functions returns True or False (or None to not match)
export const spec = new PatternMatcher<Record<string, UOp>, boolean | undefined>([
  [new UPat(Ops.DEFINE_GLOBAL).named('x'), ({ x }) => (x.dtype instanceof PtrDType || x.dtype instanceof ImageDType) && !x.dtype.local],
  [new UPat(Ops.DEFINE_LOCAL).named('x'), ({ x }) => x.dtype instanceof PtrDType && x.dtype.local],
  [new UPat(Ops.DEFINE_ACC, undefined, [UPat.var('c')], undefined, 'x', true), ({ x, c }) => x.src.slice(1).every((y) => y.op === Ops.RANGE) && c.dtype === x.dtype],
  [new UPat(Ops.DEFINE_VAR, undefined, [], undefined, 'x'), ({ x }) => typeof x.arg[1] === 'number' && typeof x.arg[2] === 'number'],

  [
    new UPat(Ops.RANGE, undefined, [new UPat(undefined).named('x'), new UPat(undefined).named('y')], undefined, 'rng'),
    ({ rng, x, y }) => rng.dtype === x.dtype && x.dtype === y.dtype && typeof rng.arg === 'number',
  ],
  [new UPat(Ops.SPECIAL, undefined, []), () => true],

  //   # TODO: confirm the args of both of these are shapetrackers
  [new UPat(Ops.VIEW, dtypes.void, []), () => true],
  [new UPat(Ops.VIEW, undefined, [UPat.var('src')], undefined, 'x'), ({ x, src }) => src.op !== Ops.STORE && x.dtype === src.dtype],
  [new UPat(Ops.VALID, dtypes.bool, [new UPat(Ops.VIEW)]), () => true],
  [new UPat(Ops.CONST).named('x'), ({ x }) => x.dtype === x.dtype.scalar() && dtypes.verify(x.arg, x.dtype)], // NOTE: this is slightly different from python, int(1) != float(1) in py but it is the same in TS

  //   # early LOAD has a <buf, shapetracker, store?>
  [new UPat(Ops.LOAD, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat(Ops.VIEW)]), () => true],
  [new UPat(Ops.LOAD, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat(Ops.VIEW), new UPat(Ops.STORE)]), () => true],
  //   # early STORE has a <buf, shapetracker, val>
  [new UPat(Ops.STORE, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat(Ops.VIEW), new UPat()]), () => true],
  //   # **** new style load/store ****

  //   # INDEX is used in new style load/store
  [new UPat(Ops.INDEX, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat()]), () => true],
  //   # LOAD takes a <bufidx, alt?, gate?, barrier?>
  [new UPat(Ops.LOAD, undefined, [new UPat([Ops.INDEX, Ops.CAST])]), () => true],
  [new UPat(Ops.LOAD, undefined, [new UPat([Ops.INDEX, Ops.CAST]), new UPat([Ops.IF, Ops.BARRIER])]), () => true],
  [new UPat(Ops.LOAD, undefined, [new UPat([Ops.INDEX, Ops.CAST]), new UPat(undefined).named('alt'), new UPat(undefined, dtypes.bool)], undefined, 'ld'), ({ ld, alt }) => ld.dtype === alt.dtype],

  //   # STORE takes a <bufidx, val, gate?>
  [new UPat(Ops.STORE, dtypes.void, [new UPat([Ops.INDEX, Ops.CAST]), new UPat()]), () => true],
  [new UPat(Ops.STORE, dtypes.void, [new UPat([Ops.INDEX, Ops.CAST]), new UPat(), new UPat(undefined, dtypes.bool)]), () => true],
  [new UPat(Ops.STORE, dtypes.void, [new UPat([Ops.INDEX, Ops.CAST]), new UPat(), new UPat(Ops.IF)]), () => true],

  //   # most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  [
    new UPat(Ops.WHERE, undefined, [new UPat(undefined, dtypes.bool), new UPat(undefined).named('x'), new UPat(undefined).named('y')], undefined, 'w'),
    ({ w, x, y }) => w.dtype === x.dtype && x.dtype === y.dtype,
  ],
  [new UPat([Ops.CMPLT, Ops.CMPNE], dtypes.bool, [new UPat(undefined).named('x'), new UPat(undefined).named('y')]), ({ x, y }) => x.dtype === y.dtype],

  //   # and SHL/SHR, the shift distance can be an int
  [
    new UPat([Ops.SHL, Ops.SHR], undefined, [new UPat(undefined).named('x'), new UPat(undefined).named('y')], undefined, 'a'),
    ({ a, x, y }) => a.dtype === x.dtype && [x.dtype, dtypes.uint].includes(y.dtype),
  ],
  [new UPat(Ops.IDIV).named('x'), ({ x }) => dtypes.is_int(x.dtype) ? undefined : false],
  [new UPat(GroupOp.ALU).named('x'), ({ x }) => x.src!.every((y) => x.dtype === y.dtype)],
  [new UPat(Ops.ASSIGN, undefined, [new UPat([Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL]), new UPat()]), () => true],
  [new UPat(Ops.ENDRANGE, dtypes.void, [new UPat(Ops.RANGE)]), () => true],

  //   # all WMMA has 3 args, <x, w, acc>
  [new UPat(Ops.WMMA, undefined, [new UPat(), new UPat(), new UPat()]), () => true],
  [new UPat(Ops.CONTRACT).named('x'), ({ x }) => x.dtype.count === prod(x.arg.map((y: any) => y[1]))],
  [new UPat(Ops.EXPAND).named('x'), ({ x }) => x.src![0].dtype.count === prod(x.arg.map((y: any) => y[1]))],

  //   # if has a <gate, barrier?>

  [new UPat(Ops.IF, dtypes.void, [new UPat()]), () => true],
  [new UPat(Ops.IF, dtypes.void, [new UPat(), new UPat(Ops.BARRIER)]), () => true],
  [new UPat(Ops.ENDIF, dtypes.void, [new UPat(Ops.IF)]), () => true],
  [new UPat(Ops.REDUCE_AXIS).named('x'), ({ x }) => Array.isArray(x.arg) && x.arg.length === 2 && [Ops.ADD, Ops.MUL, Ops.MAX].includes(x.arg[0])],
  [new UPat(Ops.GEP, undefined, [new UPat(undefined).named('src')], undefined, 'gep'), ({ gep, src }) => gep.dtype === src.dtype.scalar()],
  [new UPat(Ops.VECTORIZE).named('x'), ({ x }) => x.src?.length > 1 && x.src?.length === x.dtype.count && x.src!.every((y) => x.dtype === y.dtype.vec(x.src?.length))],
  [new UPat([Ops.BITCAST, Ops.CAST], undefined, [new UPat()], undefined, 'x'), ({ x }) => isNone(x.arg)],
  [new UPat(Ops.BARRIER, dtypes.void, new UPat(Ops.STORE, undefined, undefined, undefined, undefined, true)), () => true], // NOTE: all pointers must be local
  //   # NOTE: for testing, we let sinks be anything
  // [new UPat(UOps.SINK, undefined, new UPat(UOps.STORE)), () => true],
  [new UPat(Ops.SINK, dtypes.void), () => true],
  [new UPat(Ops.NOOP), () => true],

  //   # PTX LOAD/STORE
  [new UPat([Ops.LOAD, Ops.STORE], undefined, [new UPat(undefined, dtypes.int64)], undefined, undefined, true), () => true],
  [new UPat(Ops.BARRIER, dtypes.void, new UPat(Ops.STORE, undefined, [new UPat(undefined, dtypes.int64)], undefined, undefined, true)), () => true],
])

export const type_verify = (uops: UOp[]) => {
  for (const [i, u] of uops.entries()) {
    if (!spec.rewrite(u)) {
      print_uops(uops)
      throw new Error(`UOp verification failed at ${i} on ${u}`)
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

export const div_and_mod_folding = (x: UOp, c: number, which: typeof Ops.MOD | typeof Ops.IDIV, split_rem = false): undefined | UOp => {
  // simplify x // c or x % c, None means no change, c must be > 0
  assert(c > 0)
  if (x.dtype.count > 1) return undefined
  // simple cancel div/mod case
  const q = idiv(x.vmin, c)
  if (q === idiv(x.vmax, c)) {
    if (which === Ops.MOD) return x.sub(q * c)
    return x.const_like(q)
  }
  let [svars, factors, quotients, remainders, gcd, div, const2, offset, something_changed] = [[], [], [], [], c, 1, 0, 0, false] as [UOp[], number[], number[], number[], number, number, number, number, boolean]
  for (let u of splitUOp(x, Ops.ADD)) {
    if (u.op === Ops.MOD && which === Ops.MOD && u.src[1].op === Ops.CONST && u.src[1].arg % c === 0) {
      u = u.src[0]
      something_changed = true
    }
    const f = u.constFactor()
    const v = u.divides(f)!
    const [q, r] = divmod(f, c)
    if (r === 0 || ((which === Ops.MOD || split_rem || u.op === Ops.CONST) && r !== f)) something_changed = true
    offset += r * v.vmin
    if (u.op === Ops.CONST) const2 += f
    else { // div is the smallest common divisor of all terms
      if (f > 1 && c % f === 0 && (div === 1 || div > f)) div = f
      gcd = mathGcd(r, gcd)
      factors.push(f)
      svars.push(v)
      quotients.push(q)
      remainders.push(r)
    }
  }
  offset = offset % c
  let ubound = offset
  let lbound = offset
  // we can fold if the expression has only one non-constant term and this term can only take on two values
  let v = svars[0]
  if (svars.length === 1 && v.vmax - v.vmin === 1) {
    const r = (offset + remainders[0]) % c - offset % c
    offset -= r * v.vmin
    if (which === Ops.MOD) return add(mul(r, v), offset) as UOp
    return add(mul(idiv(factors[0] - r, c), v), idiv(const2 - offset, c)) as UOp
  }
  // a//c = (a-a%c)/c, if we can fold a%c, we can fold a//c
  // within a mod we can freely subtract multiples of c, we use this to see if a is congruent to an expression whose vmin/vmax are between 0 and c
  let exitedWithBreak = false
  let r
  for ([r, v] of zip(remainders, svars)) {
    if (r > idiv(c, 2)) {
      r = r - c
      lbound = lbound + r * (v.vmax - v.vmin)
      if (lbound < 0) {
        exitedWithBreak = true
        break
      }
    } else {
      ubound = ubound + r * (v.vmax - v.vmin)
      if (ubound >= c) {
        exitedWithBreak = true
        break
      }
    }
    offset -= r * v.vmin // determine what the new offset would be
  }
  if (!exitedWithBreak) { // vmin/vmax of the remainder is between 0 and c, we can remove the mod/div
    remainders = remainders.map((r) => Math.min(Math.abs(r), Math.abs(r - c)))
    if (which === Ops.MOD) return zip(remainders, svars).reduce((acc, [r, v]) => acc.add(mul(r, v)), x.const_like(offset))
    return zip(factors, remainders, svars).reduce((acc, [f, r, v]) => acc.add(mul(idiv(f - r, c), v)), x.const_like(idiv(const2 - offset, c)))
  }

  if (gcd !== 1) something_changed = true
  if (!something_changed) {
    if (which === Ops.IDIV && (1 < div && div < c)) {
      const newx = div_and_mod_folding(x, div, Ops.IDIV)
      if (isNotNone(newx)) return newx.idiv(idiv(c, div))
    }
    return undefined
  }
  let [quo, rem] = [x.const_like(idiv(const2, c)), x.const_like(idiv(const2 % c, gcd))]
  for (const [q, r, f, v] of zip(quotients, remainders, factors, svars)) {
    if (which === Ops.IDIV && (!split_rem) && r !== 0) rem = rem.add(mul(idiv(f, gcd), v))
    else {
      rem = rem.add(mul(idiv(r, gcd), v))
      quo = quo.add(mul(q, v))
    }
  }
  if (which === Ops.MOD) return add(mul(gcd, mod(rem, idiv(c, gcd))), mod(const2, gcd)) as UOp
  return add(idiv(rem, idiv(c, gcd)), quo) as UOp
}

const lt_folding = (x: UOp, c: number): UOp | undefined => {
  const [p, np] = partition(splitUOp(x, Ops.ADD).toArray(), (u) => u.constFactor() === 1)
  const d = mathGcd(...np.map((u) => u.constFactor()), c)
  if (np && d > 1 && 0 <= p.map((u) => u.vmin).reduce((p, c) => c + p) && p.map((u) => u.vmax).reduce((p, c) => p + c) < d) {
    return np.reduce((p, c) => p.add(c), UOp.int(0)).divides(d)!.lt(idiv(c, d))
  }
  return undefined
}
const fold_unrolled_divs = ({ divs }: { divs: UOp }) => {
  // div pattern in unrolled arange
  // example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
  let [addChain, denominator, seenConst, ans] = [splitUOp(divs, Ops.ADD), undefined as number | undefined, [] as number[], undefined as undefined | UOp]
  for (const u of addChain) {
    if (!(u.op === Ops.IDIV && u.src[1].op === Ops.CONST)) return undefined
    if (isNone(denominator)) denominator = u.src[1].arg
    if (denominator !== u.src[1].arg) return undefined
    // assumed CONST is the last of an ADD
    let s0 = u.src[0]
    if (s0.op === Ops.ADD && s0.src[1].op === Ops.CONST && s0.src[1].op === Ops.CONST) {
      seenConst.push(s0.src[1].arg)
      s0 = s0.src[0]
    } else seenConst.push(0)
    if (isNone(ans)) ans = s0
    if (!isEq(ans, s0)) return undefined
  }
  if (isNone(denominator)) return undefined
  // the first (denominator-len(seen_const)) terms may have been folded to 0 already
  for (const i of range(denominator - seenConst.length)) {
    if (isNotNone(ans) && 0 <= ans.vmin && ans.vmax + i < denominator) seenConst.push(i)
  }
  return isNotNone(ans) && isEq(seenConst.sort((a, b) => b - a), range(denominator)) ? ans : undefined
}
const canonicalizeSimplex = (X: UOp): UOp | undefined => {
  // (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  // returns x0 + x1 + ... in such case, or None if not
  let [changed, ret] = [false, [] as UOp[]]
  for (let u of splitUOp(X, Ops.ADD)) {
    // assumed the const is the last src of MUL
    if (u.op === Ops.MUL && u.src[1].op === Ops.CONST && u.src[1].arg > 0) {
      changed = true
      u = u.src[0]
    }
    if (!(GroupOp.Irreducible.includes(u.op) && u.vmin >= 0)) return undefined
    ret.push(u)
  }
  return changed ? ret.reduce((p, c) => p.add(c)) : undefined
}

export const isIncreasing = (f: UOp): boolean => {
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

export const uop_given_valid = (valid: UOp, uop: UOp): UOp | undefined => {
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
    if (isNotNone(v[0]) && isNotNone(v[1]) && v[0] > v[1]) return undefined

    // every candidate is a set of contrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
    const candidates: [UOp, UOp][][] = []
    if (expr.op === Ops.ADD && v[0] === 1 && splitUOp(expr, Ops.ADD).every((u) => GroupOp.Irreducible.includes(u.op))) {
      // if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
      candidates.push(splitUOp(expr, Ops.ADD).toArray().map((Xi) => [Xi, UOp.variable('fake', 1, Xi.vmax, Xi.dtype)]))
    }
    // try checking the whole clause
    if (uop.toposort.has(expr)) {
      candidates.push([[expr, UOp.variable('fake', isNone(v[0]) ? expr.vmin : v[0], isNone(v[1]) ? expr.vmax : v[1], expr.dtype)]])
    }
    for (const candidate of candidates) {
      // if every branch in candidate gives the same simplified uop, we can rewrite the uop
      const newuops = candidate.map(([X, newX]) => uop.substitute(new Map([[X, newX]])).simplify().substitute(new Map([[X, newX]])).simplify())
      if (uop.op === Ops.VECTORIZE && uop.src.length === 2) {
        if (all_same(newuops.map((uops) => uops.src[0]))) uop = uop.replace({ src: [newuops[0].src[0], uop.src[1]] })
        if (all_same(newuops.map((uops) => uops.src[1]))) uop = uop.replace({ src: [uop.src[0], newuops[0].src[1]] })
      } else if (all_same(newuops)) uop = newuops[0]
    }
  }
  return uop
}

const _validPriority = (v: UOp, valids: UOp[]): number => {
  // we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
  try {
    return valids.map((other) => other.toposort.has(parseValid(v)[0]) ? -1 : 0 as number).reduce((p, c) => p + c)
  } catch {
    return 0
  }
}
export const simplify_valid = (valid: UOp): UOp | undefined => {
  const ret: UOp[] = []
  let somethingChanged = false
  const valids = splitUOp(valid, Ops.AND).toArray()
  for (const stmt of valids.sort((a, b) => _validPriority(b, valids) - _validPriority(a, valids))) {
    ret.push(ret.length ? (uop_given_valid(ret.reduce((p, c) => p.bitwise_and(c)), stmt) || stmt) : stmt)
    if (!isEq(ret.at(-1), stmt)) somethingChanged = true
  }
  return somethingChanged ? ret.reduce((p, c) => p.bitwise_and(c)) : undefined
}
export const maxVarConst = ({ x, c1, c2 }: { x: UOp; c1: UOp; c2: UOp }) => {
  if (x.vmin >= (0)) return c1.arg >= c2.arg ? x.mul(c1) : x.mul(c2)
  if (x.vmax <= (0)) return c1.arg >= c2.arg ? x.mul(c2) : x.mul(c1)
}
export const sint_to_uop = (x: sint, dtype = dtypes.int) => (typeof x === 'number' || typeof x === 'bigint') ? UOp.const(dtype, x) : x

export const symbolic_simple = new PatternMatcher([
  //   // ** this folding **
  [UPat.var('x').add(0), ({ x }) => x], // x+0 -> x
  [UPat.var('x').mul(1), ({ x }) => x], // x*1 -> x
  [UPat.var('x').idiv(UPat.var('x')), ({ x }) => x.const_like(1)], // x//x -> 1
  [UPat.var('x').idiv(1), ({ x }) => x], // x//1 -> x
  [UPat.var('x').idiv(-1), ({ x }) => x.neg()], // x//-1 -> -x
  [UPat.var('x').div(UPat.var('x')), ({ x }) => x.const_like(1)], // x/x -> 1
  [(UPat.var('x').mul(UPat.var('x2'))).div(UPat.var('x2')), ({ x, x2: _x2 }) => x], // (x*x2)/x2 -> x
  [(UPat.var().mod(UPat.var('y'))).named('base').mod(UPat.var('y')), ({ base, y: _y }) => base], // (x%y)%y = -> x%y (rewritten with base for speed)
  [UPat.var('x').mod(UPat.cvar('c')).add(UPat.var('x').idiv(UPat.cvar('c')).mul(UPat.cvar('c'))), ({ x, c: _c }) => x], // (x%c)+(x//c)*c = x
  [(UPat.var('x').idiv(UPat.cvar('c1'))).mul(UPat.cvar('c3')).add(UPat.var('x').mod(UPat.cvar('c1')).mul(UPat.cvar('c2'))), ({ x, c1, c2, c3 }) => c1.arg * c2.arg === c3.arg ? x.mul(c2) : undefined], // (x%c1)*c2+(x//c1)*c3 = x*c2 if c1*c2==c3
  [UPat.var('x', dtypes.bool).bitwise_and(UPat.cvar('c', undefined, false)), ({ x, c }) => c.arg ? x : c],
  [UPat.var('x', dtypes.bool).bitwise_or(UPat.cvar('c', undefined, false)), ({ x, c }) => c.arg ? c : x],
  [UPat.var('x').maximum(UPat.var('x')), ({ x }) => x],
  [UPat.var('x').bitwise_and(UPat.var('x')), ({ x }) => x],
  [UPat.var('x').bitwise_or(UPat.var('x')), ({ x }) => x],
  [UPat.var('x', dtypes.bool).logical_not().logical_not(), ({ x }) => x],
  [UPat.var('x', dtypes.bool).where(UPat.const(dtypes.bool, true), UPat.const(dtypes.bool, false)), ({ x }) => x],
  //   // ** zero folding **
  [UPat.var('x').lt(UPat.var('x')), ({ x }) => UOp.const(dtypes.bool.vec(x.dtype.count), false)], // x < x -> False
  [UPat.var('x', dtypes.ints).ne(UPat.var('x', dtypes.ints)), ({ x }) => UOp.const(dtypes.bool.vec(x.dtype.count), false)], // x != x -> False (only ints)
  //   // x*0 -> 0 or 0*x -> 0
  //   // if x is nan or inf it should render the nan value.
  //   // NOTE: this can be wrong for loaded NaN
  [UPat.var('x').mul(0), ({ x }) => x.const_like(typeof x.arg === 'number' && (isNaN(x.arg) || isInf(x.arg)) ? NaN : 0)],
  //   // ** constant folding **
  [new UPat(GroupOp.ALU, undefined, new UPat([Ops.VCONST, Ops.CONST]), undefined, 'a'), ({ a }) => a.const_like(exec_alu(a.op, a.dtype, a.src?.map((x) => x.arg), false))],
  //   // bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  [UPat.var('x', dtypes.bool).mul(UPat.var('y', dtypes.bool)), ({ x, y }) => x.bitwise_and(y)],
  [UPat.var('x', dtypes.bool).add(UPat.var('y', dtypes.bool)), ({ x, y }) => x.bitwise_or(y)],
  [UPat.var('x', dtypes.bool).maximum(UPat.var('y', dtypes.bool)), ({ x, y }) => x.bitwise_or(y)],
  //   // *** cast ***
  [new UPat(Ops.CAST, undefined, UPat.cvar('c'), undefined, 'root'), ({ root, c }) => root.const_like(c.arg)],
  [new UPat(Ops.CAST).named('root'), ({ root }) => isEq(root.dtype, root.src![0].dtype) ? root.src![0] : undefined], //24
])

export const symbolic = symbolic_simple.add(
  new PatternMatcher([
    // ** COMMUTATIVE flipping **
    [new UPat(GroupOp.Commutative).named('x'), ({ x }) => isLessThan(x.src[1].tuplize(), x.src[0].tuplize()) ? x.replace({ src: x.src.toReversed() }) : undefined],
    //   // group like
    [(UPat.var('x').add(UPat.var('y'))).add(UPat.var('x').mul(UPat.cvar('c'))), ({ x, y, c }) => (x.add(x.mul(c))).add(y)],
    //   // ** boolean algebra **
    [UPat.var('x').bitwise_or(UPat.var('x').bitwise_and(UPat.var())), ({ x }) => x], // x|(x&y) -> x
    //   // ** combine terms **
    [UPat.var('x').mul(UPat.cvar('c0')).add(UPat.var('x').mul(UPat.cvar('c1'))), ({ x, c0, c1 }) => x.mul(c0.add(c1))], // (x*c0)+(x*c1) -> x*(c0+c1)
    [UPat.var('x').add(UPat.var('x').mul(UPat.cvar('c'))), ({ x, c }) => x.mul(c.add(1))], // (x+x*c)-> x*(c+1)
    [UPat.var('x').add(UPat.var('x')), ({ x }) => x.mul(2)], // (x+x)-> x*2
    [(UPat.var('x').div(UPat.var('x2'))).div(UPat.var('x3')), ({ x, x2, x3 }) => x.div(x2.mul(x3))], // (x/x2)/x3 -> x/(x2*x3)
    [UPat.var('x').add(UPat.cvar('c')).mul(-1, true), ({ x, c }) => x.neg().add(c.neg())], // -(x+c) -> -x + -c
    // //   // a conditional with the same results either way is a noop, also fold const conditionals
    [UPat.var().where(UPat.var('val'), UPat.var('val')), ({ val }) => val],
    [UPat.cvar('gate', undefined, false).where(UPat.var('c0'), UPat.var('c1')), ({ gate, c0, c1 }) => gate.arg ? c0 : c1],
    // alu of two where with same conds can combine, only do if true branch or false branch is const
    [
      new UPat(GroupOp.Binary, undefined, [UPat.var('c').where(UPat.var('t'), UPat.var('f')), UPat.var('c').where(UPat.var('tt'), UPat.var('ff'))], undefined, 'alu'),
      ({ alu, c, t, tt, f, ff }) => (t.op === tt.op && tt.op === Ops.CONST) || (f.op === ff.op && ff.op === Ops.CONST) ? c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) : undefined,
    ],
    // ALU min==max -> CONST (slow!)
    [new UPat(GroupOp.ALU).named('x'), ({ x }) => x.vmin === x.vmax ? x.const_like(x.vmin) : undefined],
    // max folding
    [UPat.var('x').maximum(UPat.var('y')), ({ x, y }) => x.vmax <= y.vmin ? x.vmin >= y.vmax ? x : y : undefined],
    // TODO: why does this rule break beautiful_mnist?
    //((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
    [UPat.var('x').mul(UPat.cvar('c1')).maximum(UPat.var('x').mul(UPat.cvar('c2'))), ({ x, c1, c2 }) => maxVarConst({ x, c1, c2 })],
    //   // ** two stage ALU folding **
    [(UPat.var('x').add(UPat.cvar('c1'))).add(UPat.cvar('c2')), ({ x, c1, c2 }) => x.add(c1.add(c2))],
    [(UPat.var('x').mul(UPat.cvar('c1'))).mul(UPat.cvar('c2')), ({ x, c1, c2 }) => x.mul(c1.mul(c2))],
    [(UPat.var('x').bitwise_and(UPat.cvar('c1'))).bitwise_and(UPat.cvar('c2')), ({ x, c1, c2 }) => x.bitwise_and(c1.bitwise_and(c2))],
    [(UPat.var('x').bitwise_or(UPat.cvar('c1'))).bitwise_or(UPat.cvar('c2')), ({ x, c1, c2 }) => x.bitwise_or(c1.bitwise_or(c2))],
    [(UPat.cvar('c0').add(UPat.var('x'))).lt(UPat.cvar('c1')), ({ x, c0, c1 }) => x.lt(c1.sub(c0))], // c0 + x < c1 -> x < c1 - c0
    [(UPat.var('x').idiv(UPat.cvar('c1'))).idiv(UPat.cvar('c2')), ({ x, c1, c2 }) => x.idiv(c1.mul(c2))], // (x//c1)//c2 -> x//(c1*c2)
    // //   // ** lt **
    // //   // c0*x<c1 for positive int c0,c1
    [(UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)), ({ x, c0, c1 }) => c0.arg > 0 && c1.arg > 0 ? x.lt(Math.ceil(c1.arg / c0.arg)) : undefined],
    //   // c0*x<c1 for negative int c0 and non-positive c1
    [(UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)), ({ x, c0, c1 }) => c0.arg < 0 && c0.arg !== -1 && c1.arg <= 0 ? x.neg().lt(-Math.floor(-c1.arg / -c0.arg)) : undefined],
    //   // x//c0<c1 for positive int c0
    [(UPat.var('x', dtypes.ints).idiv(UPat.cvar('c0', undefined, false))).lt(UPat.cvar('c1', undefined, false)), ({ x, c0, c1 }) => c0.arg > 0 ? x.lt(c1.arg * c0.arg) : undefined],
    //   // ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
    [new UPat(Ops.ADD, undefined, [UPat.var('x'), UPat.cvar('c1')]).add(UPat.var('y')), ({ x, c1, y }) => (x.add(y)).add(c1)],
    [new UPat(Ops.MUL, undefined, [UPat.var('x'), UPat.cvar('c1')]).mul(UPat.var('y')), ({ x, c1, y }) => (x.mul(y)).mul(c1)],
    //   // *** rules from symbolic ***
    //   // unrolled arange div folding
    [new UPat(Ops.ADD, undefined, [[new UPat(), new UPat(Ops.IDIV)]], undefined, 'divs'), ({ divs }) => fold_unrolled_divs({ divs })],
    // generic lt folding
    [UPat.var('x', dtypes.sints).lt(UPat.cvar('c', undefined, false)), ({ x, c }) => 0 < c.arg ? lt_folding(x, c.arg) : undefined],
    // canonicalize a simplex with positive coefficients > 0
    // not x < 1 -> X > 0
    [UPat.var('x', dtypes.ints).lt(1).ne(true), ({ x }) => {
      const newx = canonicalizeSimplex(x)
      return isNotNone(newx) ? newx.lt(1).ne(true) : undefined
    }],
    // div folding
    [UPat.var('x').idiv(UPat.cvar('c')).add(UPat.cvar('a')).idiv(UPat.cvar('d')), ({ x, c, a, d }) => (x.add(a.mul(c))).idiv(c.mul(d))], // (x//c+a)//d -> (x+a*c)//(c*d)
    [UPat.var('x', dtypes.sints).idiv(UPat.cvar('c', undefined, false)), ({ x, c }) => 0 < c.arg ? div_and_mod_folding(x, c.arg, Ops.IDIV) : undefined],
    // ** mod **
    // mod folding
    [UPat.var('x').mod(UPat.cvar('c', undefined, false)), ({ x, c }) => 0 < c.arg ? div_and_mod_folding(x, c.arg, Ops.MOD) : undefined],
  ]),
)

export const symbolic_flat = symbolic.add(
  new PatternMatcher<Record<string, UOp>, UOp | undefined>([
    // ** combine terms (opinionated) **
    [UPat.var('x').add(UPat.var('y')).mul(-1, true), ({ x, y }) => x.neg().add(y.neg())], // -(x+y) -> -x + -y
    //   # (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
    [UPat.var('x', dtypes.ints).add(UPat.var('y')).mul(UPat.cvar('c')), ({ x, y, c }) => x.mul(c).add(y.mul(c))],
  ]),
)

export const _substitute = new PatternMatcher<{ x: UOp; ctx: Map<UOp, UOp> }>([[new UPat(Ops.values()).named('x'), ({ ctx, x }) => ctx.get(x)]])

// # for debug
const syms = new Map([[Ops.ADD, '+'], [Ops.SUB, '-'], [Ops.IDIV, '//'], [Ops.MOD, '%'], [Ops.SHL, '<<'], [Ops.SHR, '>>'], [Ops.MUL, '*'], [Ops.CMPLT, '<'], [Ops.CMPNE, '!='], [Ops.AND, '&'], [Ops.OR, '|'], [Ops.XOR, '^']])
// KAREL: probably all these should be JS specific
export const renderer = new PatternMatcher<Record<string, UOp>, UOp>([
  [new UPat([Ops.DEFINE_VAR, Ops.SPECIAL]).named('x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, x.arg[0])],
  [new UPat(Ops.RANGE).named('x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, `ridx(${x.arg},)`)],
  [new UPat(Ops.CONST).named('x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, x.arg.toString())],
  [new UPat(Ops.BIND, undefined, new UPat(Ops.NOOP), undefined, 'x'), ({ x }) => x.src[0]],
  [new UPat(Ops.NEG, undefined, new UPat(Ops.NOOP), undefined, 'x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(-${x.src[0].arg})`)],
  [new UPat(Ops.MAX, undefined, new UPat(Ops.NOOP), undefined, 'x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, `max(${x.src[0].arg}, ${x.src[1].arg})`)],
  [new UPat(Ops.MULACC, undefined, new UPat(Ops.NOOP), undefined, 'x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(${x.src[0].arg}*${x.src[1].arg}+${x.src[2].arg})`)],
  [new UPat(Ops.WHERE, undefined, new UPat(Ops.NOOP), undefined, 'x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(${x.src[1].arg} if ${x.src[0].arg} else ${x.src[2].arg})`)],
  [new UPat(GroupOp.ALU, undefined, new UPat(Ops.NOOP), undefined, 'x'), ({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(${x.src[0].arg}${syms.get(x.op)}${x.src[1].arg})`)],
])

// # *** what was symbolic.py ***

export type sint = number | UOp
export const add = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.add(b) : typeof b !== 'number' ? b.add(a, true) : a + b) as A | B
export const sub = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.sub(b) : typeof b !== 'number' ? b.const_like(a).sub(b) : a - b) as A | B
export const mul = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.mul(b) : typeof b !== 'number' ? b.mul(a, true) : a * b) as A | B
export const div = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.div(b) : typeof b !== 'number' ? b.div(a, true) : a / b) as A | B
export const idiv = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.idiv(b) : typeof b !== 'number' ? b.idiv(a, true) : Math.floor(a / b)) as A | B
export const neg = <A extends sint>(a: A) => (typeof a !== 'number' ? a.mul(-1) : a * -1)

export const lt = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.lt(b) : typeof b !== 'number' ? b.const_like(a).lt(b) : Number(a < b)) as A | B
export const gt = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.gt(b) : typeof b !== 'number' ? b.const_like(a).gt(b) : Number(a > b)) as A | B

export const le = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.le(b) : typeof b !== 'number' ? b.const_like(a).le(b) : Number(a <= b)) as A | B
export const ge = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.ge(b) : typeof b !== 'number' ? b.const_like(a).ge(b) : Number(a >= b)) as A | B

export const mod = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.mod(b) : typeof b !== 'number' ? b.mod(a, true) : a % b) as A | B
export const ne = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.ne(b) : typeof b !== 'number' ? b.ne(a) : Number((a as number) !== b)) as A | B
export const eq = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.eq(b) : typeof b !== 'number' ? b.eq(a) : Number((a as number) === b)) as A | B
export const and = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.bitwise_and(b) : typeof b !== 'number' ? b.bitwise_and(a, true) : Number(a && b)) as A | B
export const or = <A extends sint, B extends sint>(a: A, b: B) => (typeof a !== 'number' ? a.bitwise_or(b) : typeof b !== 'number' ? b.bitwise_or(a, true) : Number(a || b)) as A | B

export const sint_polyN = <A extends sint>(x: A, p: number[]): A => p.reduce((acc, c) => add(mul(acc, x), c), 0 as sint) as A

export const sint_prod = (x: sint[]) => x.reduce((acc, curr) => mul(acc, curr), 1)
export const sint_sum = (x: sint[]) => x.reduce((acc, curr) => add(acc, curr), 0)
export const sint_sorted = (items: sint[], reverse = false) => items.toSorted((a, b) => lt(a, b) ? (!reverse ? -1 : 1) : (!reverse ? 1 : -1))

export const sint_ceildiv = (num: sint, amt: sint): sint => neg(idiv(num, neg(amt)))

// *** uop swizzling ***

export const merge_views = new PatternMatcher<Record<string, UOp>, UOp>([[new UPat(Ops.VIEW).named('s0').view(undefined, { name: 's1' }), ({ s0, s1 }) => s0.replace({ arg: s0.st?.add(s1.st!) })]])

// push VIEW to loads
export const view_left = merge_views.add(
  new PatternMatcher<Record<string, UOp>, UOp>([
    // VIEW before elementwise ops
    [
      new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN]).named('e').view(undefined, { name: 'v' }),
      ({ e, v }) => e.replace({ src: e.src.map((s) => !s.has_st ? s : s === s.base ? s.view(v.st!) : s.base.view(s.st!.add(v.st!))) }),
    ],
    // early merge VIEW buffer ops
    [new UPat(GroupOp.Buffer).named('b').view(undefined, { name: 'v' }), ({ b, v }) => b.replace({ src: b.src.map((s) => s.op === Ops.VIEW ? (s.st!.add(v.st!)).to_uop() : s) })],
  ]),
)
