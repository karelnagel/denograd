// deno-lint-ignore-file no-this-alias
import type { Buffer, DeviceType } from './device.ts'
import { DType, dtypes, ImageDType, PtrDType, truncate } from './dtype.ts'
import { Env } from './env/index.ts'
import { accumulate, add, AMX, and, cache_fn, type ConstType, dedup, DefaultMap, div, flatten, floatString, ge, get_env, idiv, is_less_than, isConst, isinstance, lshift, lt, mod, mul, ne, neg, NotImplemented, or, pairwise, polyN, prod, product, rshift, slice, sorted, sub, sum, TRANSCENDENTAL, WeakKeyMap, xor } from './helpers.ts'
import { _METADATA, abs, all_int, all_same, assert, cache, counter, DEBUG, divmod, Enum, get_key, get_number_env, is_eq, is_subset, isInf, list_str, math_gcd, max, type Metadata, min, partition, permutations, range, set_default, sin, SPLIT_REDUCEOP, sqrt, trunc, WeakValueMap, zip } from './helpers.ts'
import type { Renderer } from './renderer/index.ts'
import { ShapeTracker } from './shape/shapetracker.ts'

export type Variable = UOp
export type ConstLike<This = never> = ConstType<This> | Variable | ConstType[]

class SimpleMathTrait<T extends SimpleMathTrait<T>> {
  // required to implement
  alu = (_arg: Ops, ..._src: T[]): T => {
    throw new NotImplemented()
  }
  const_like = (_b: ConstLike): T => {
    throw new NotImplemented()
  }

  // great functions you get!
  ufix = (x: ConstType<T>): T => x instanceof MathTrait ? x : this.const_like(x as any) //ignoring this error, cause not sure
  _binop = (op: Ops, x: ConstType<T>, reverse: boolean) => reverse ? this.ufix(x).alu(op, this as any) : this.alu(op, this.ufix(x))
  logical_not = () => this.ne(true)
  neg = () => {
    const dtype = 'dtype' in this && this.dtype instanceof DType ? this.dtype : undefined
    if (dtype === undefined) throw new Error(`MathTraits __neg__ requires a dtype, ${this}`)
    return dtype.scalar() === dtypes.bool ? this.logical_not() : this.mul(-1)
  }
  add = (x: ConstType<T>, reverse = false) => this._binop(Ops.ADD, x, reverse)
  mul = (x: ConstType<T>, reverse = false) => this._binop(Ops.MUL, x, reverse)
  bitwise_and = (x: ConstType<T>, reverse = false) => this._binop(Ops.AND, x, reverse)
  bitwise_or = (x: ConstType<T>, reverse = false) => this._binop(Ops.OR, x, reverse)
  xor = (x: ConstType<T>, reverse = false) => this._binop(Ops.XOR, x, reverse)
  idiv = (x: ConstType<T>, reverse = false) => this._binop(Ops.IDIV, x, reverse)
  mod = (x: ConstType<T>, reverse = false) => !reverse ? this.alu(Ops.MOD, this.ufix(x)) : this.ufix(x).alu(Ops.MOD, this as any)
  sub = (x: ConstType<T>, reverse = false) => reverse ? this.ufix(x).alu(Ops.ADD, this.neg()) : this.alu(Ops.ADD, isConst(x) ? this.ufix(-x) : x.neg())
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

  // not in Tensor
  maximum = (x: ConstType<T>) => this.alu(Ops.MAX, this.ufix(x))
  minimum = (x: ConstType<T>) => this.neg().maximum(isConst(x) ? this.ufix(-x) : x.neg()).neg()
  where = (x: ConstType<T>, y: ConstType<T>) => this.alu(Ops.WHERE, this.ufix(x), this.ufix(x).ufix(y))
  threefry = (seed: ConstType<T>) => this.alu(Ops.THREEFRY, this.ufix(seed))
  reciprocal = () => this.alu(Ops.RECIP)
  sqrt = () => this.alu(Ops.SQRT)
  sin = () => this.alu(Ops.SIN)
  log2 = () => this.alu(Ops.LOG2)
  exp2 = () => this.alu(Ops.EXP2)
}

// the order of these Ops controls the order of the toposort
export class Ops<Name extends string = string> extends Enum {
  private static VALUES: Ops[] = []
  static values = () => [...Ops.VALUES]
  key: string
  constructor(name: Name) {
    super(name, Ops.VALUES.length + 1)
    Ops.VALUES.push(this)
    this.key = get_key(name, this.value)
  }
  // uops that aren't rendered
  static readonly SINK = new Ops('SINK')
  static readonly CONTIGUOUS = new Ops('CONTIGUOUS')
  static readonly CONTIGUOUS_BACKWARD = new Ops('CONTIGUOUS_BACKWARD')
  static readonly DETACH = new Ops('DETACH')
  static readonly PRELOAD = new Ops('PRELOAD')

  // TODO: empty continues to exist because of tensor
  static readonly EMPTY = new Ops('EMPTY')

  // MetaOps
  static readonly COPY = new Ops('COPY')
  static readonly BUFFER_VIEW = new Ops('BUFFER_VIEW')

  // blocks in linearizer
  static readonly BLOCK = new Ops('BLOCK')
  static readonly BLOCKSTART = new Ops('BLOCKSTART')
  static readonly BLOCKFORK = new Ops('BLOCKFORK')
  static readonly BLOCKEND = new Ops('BLOCKEND')

  // movement ops!
  static readonly RESHAPE = new Ops('RESHAPE')
  static readonly PERMUTE = new Ops('PERMUTE')
  static readonly EXPAND = new Ops('EXPAND')
  static readonly PAD = new Ops('PAD')
  static readonly SHRINK = new Ops('SHRINK')
  static readonly STRIDE = new Ops('STRIDE')

  // misc ops
  static readonly UNROLL = new Ops('UNROLL')
  static readonly CONTRACT = new Ops('CONTRACT')
  static readonly VIEW = new Ops('VIEW')
  static readonly DEFINE_GLOBAL = new Ops('DEFINE_GLOBAL')
  static readonly BUFFER = new Ops('BUFFER')
  static readonly DEFINE_VAR = new Ops('DEFINE_VAR')
  static readonly DEFINE_LOCAL = new Ops('DEFINE_LOCAL')
  static readonly DEFINE_ACC = new Ops('DEFINE_ACC')
  static readonly VALID = new Ops('VALID')
  static readonly SPECIAL = new Ops('SPECIAL')
  static readonly NOOP = new Ops('NOOP')

  // reduce
  static readonly REDUCE_AXIS = new Ops('REDUCE_AXIS')

  // helper ops
  static readonly GEP = new Ops('GEP')
  static readonly VECTORIZE = new Ops('VECTORIZE')

  // UnaryOps
  static readonly CAST = new Ops('CAST')
  static readonly BITCAST = new Ops('BITCAST')
  static readonly EXP2 = new Ops('EXP2')
  static readonly LOG2 = new Ops('LOG2')
  static readonly SIN = new Ops('SIN')
  static readonly SQRT = new Ops('SQRT')
  static readonly RECIP = new Ops('RECIP')
  static readonly NEG = new Ops('NEG')

  // load/store before math
  static readonly LOAD = new Ops('LOAD')
  static readonly STORE = new Ops('STORE')

  // early INDEX
  static readonly INDEX = new Ops('INDEX')

  // math ops
  static readonly WMMA = new Ops('WMMA')

  // BinaryOps
  static readonly ADD = new Ops('ADD')
  static readonly MUL = new Ops('MUL')
  static readonly IDIV = new Ops('IDIV')
  static readonly MAX = new Ops('MAX')
  static readonly MOD = new Ops('MOD')
  static readonly CMPLT = new Ops('CMPLT')
  static readonly CMPNE = new Ops('CMPNE')
  static readonly XOR = new Ops('XOR')
  static readonly SHL = new Ops('SHL')
  static readonly SHR = new Ops('SHR')
  static readonly OR = new Ops('OR')
  static readonly AND = new Ops('AND')
  static readonly THREEFRY = new Ops('THREEFRY')
  static readonly SUB = new Ops('SUB')
  static readonly FDIV = new Ops('FDIV')

  // TernaryOps
  static readonly WHERE = new Ops('WHERE')
  static readonly MULACC = new Ops('MULACC')

  // assignment ops
  static readonly ASSIGN = new Ops('ASSIGN')
  static readonly BIND = new Ops('BIND')

  // control flow ops
  static readonly BARRIER = new Ops('BARRIER')
  static readonly RANGE = new Ops('RANGE')
  static readonly IF = new Ops('IF')
  static readonly ENDRANGE = new Ops('ENDRANGE')
  static readonly ENDIF = new Ops('ENDIF')

  // consts last!
  static readonly VCONST = new Ops('VCONST')
  static readonly CONST = new Ops('CONST')

  // device
  static readonly DEVICE = new Ops('DEVICE')
  static readonly MULTI = new Ops('MULTI')
}

export class GroupOp {
  static Unary = [Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIP, Ops.NEG]
  static Binary = [Ops.ADD, Ops.MUL, Ops.IDIV, Ops.MAX, Ops.MOD, Ops.CMPLT, Ops.CMPNE, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.THREEFRY, Ops.SUB, Ops.FDIV]
  static Ternary = [Ops.WHERE, Ops.MULACC]
  static ALU = [...this.Unary, ...this.Binary, ...this.Ternary]

  static Irreducible = [Ops.CONST, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.RANGE]
  static Movement = [Ops.RESHAPE, Ops.PERMUTE, Ops.EXPAND, Ops.PAD, Ops.SHRINK, Ops.STRIDE]

  static Buffer = [Ops.LOAD, Ops.STORE, Ops.VALID, Ops.PRELOAD]
  static Block = [Ops.BLOCK, Ops.BLOCKEND, Ops.BLOCKFORK, Ops.BLOCKSTART]

  // BinaryOps that can be flipped
  static Commutative = [Ops.ADD, Ops.MUL, Ops.MAX, Ops.CMPNE, Ops.XOR, Ops.OR, Ops.AND]

  // BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  static Associative = [Ops.AND, Ops.ADD, Ops.MUL, Ops.OR]

  // BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  static Idempotent = [Ops.AND, Ops.MAX, Ops.OR]

  // do not preserve f(0) = 0
  static UnsafePad = [Ops.RECIP, Ops.LOG2, Ops.EXP2, Ops.IDIV]
}

// some BUFFER ops can be processed with only a view
export const view_supported_devices = ['LLVM', 'CLANG', 'CUDA', 'NV', 'AMD', 'METAL', 'QCOM', 'DSP', 'DISK']

// https://en.wikipedia.org/wiki/Identity_element
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

// smax/smin are replacements for max/min that preserve symbolic
const _suop = (lst: sint[], uop_fxn: (...x: UOp[]) => UOp, python_fxn: (...a: number[]) => number): sint => {
  const [uops, nums] = partition(lst, (x) => x instanceof UOp) as [UOp[], number[]]
  return ssimplify((nums ? [...uops, python_fxn(...nums)] as UOp[] : []).reduce((acc, x) => uop_fxn(acc, x)))
}
export const smax = (...lst: sint[]) => _suop(lst, (...x) => x.reduce((acc, x) => acc.maximum(x)), (...x) => max(x))
export const smin = (...lst: sint[]) => _suop(lst, (...x) => x.reduce((acc, x) => acc.minimum(x)), (...x) => min(x))

export const ssimplify = (uop: UOp) => uop instanceof UOp ? uop.ssimplify() : uop
export const sym_infer = (uop: sint, varVals: Map<UOp, number>): number => uop instanceof UOp ? uop.sym_infer(varVals) : uop

type UOpInput = { op: Ops; dtype?: DType; src?: UOp[]; arg?: any }

export const buffers = new WeakKeyMap<UOp, Buffer>()
export const all_metadata = new WeakKeyMap<UOp, Metadata>()

export class UOp extends MathTrait<UOp> {
  static register = new FinalizationRegistry<string>((key) => {
    if (buffers.map.has(key)) buffers.map.get(key)?.[1].ref(-1)
  })
  static cache = new WeakValueMap<string, UOp>()
  key: string
  children = new WeakValueMap<string, UOp>()
  constructor(public op: Ops, public dtype = dtypes.void, public src: UOp[] = [], public arg?: any, _buffer?: Buffer) {
    super()
    this.key = get_key(this.op, this.dtype, this.arg, this.src)
    const ret = UOp.cache.get(this.key)
    if (ret !== undefined) {
      if (!is_eq(this.arg, ret.arg)) throw new Error(`Args fucked: \nthis=${this.arg}, ${this.arg?.key} \nret=${ret.arg}, ${ret.arg?.key}`)
      if (this.op !== ret.op) throw new Error(`Op fucked: ${this.op} !== ${ret.op}`)
      if (this.dtype !== ret.dtype) throw new Error(`DType fucked: ${this.dtype} !== ${ret.dtype}`)
      for (const [a, b] of zip(this.src, ret.src)) if (a !== b) throw new Error(`Src fucked: \na=${a}, ${a.key} \nb=${b}, ${b.key}`)
      return ret
    }

    for (const s of src) s.children.set(this.key, this)
    // NOTE: this will soon be set by Tensor once we remove function.py
    const metadata = _METADATA.get()
    if (metadata !== undefined) all_metadata.set(this, metadata)
    // NOTE: this value is set by pickle when pickling a realized tensor
    if (_buffer !== undefined) {
      if (op !== Ops.BUFFER) throw new Error(`trying to set Buffer ${_buffer} for ${op}`)
      buffers.set(this, _buffer)
    }
    UOp.register.register(this, this.key)
    UOp.cache.set(this.key, this)
    Object.freeze(this)
  }
  @cache
  override toString(): string {
    const src = !this.src ? 'undefined' : this.src.length === 0 ? '[]' : `[\n${this.src.map((s) => s.toString()).join(',\n').split('\n').map((s) => '  ' + s).join('\n')}\n]`
    return `new UOp(${this.op.toString()}, ${this.dtype}, ${src}, ${list_str(this.arg)})`
  }
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  replace = (args: Partial<UOpInput>) => new UOp(args.op !== undefined ? args.op : this.op, args.dtype !== undefined ? args.dtype : this.dtype, args.src !== undefined ? args.src : this.src, args.arg !== undefined ? args.arg : this.arg)

  get toposort(): Set<UOp> {
    const _toposort = (u: UOp, cache: Set<UOp>): Set<UOp> => {
      if (cache.has(u)) return new Set()
      const nodes = new Set<UOp>()
      //  NOTE: this is a lot faster than the comprehension in parents
      for (const parent of u.src) _toposort(parent, cache).values().forEach((x) => nodes.add(x))
      nodes.add(u)
      cache.add(u)
      return nodes
    }
    return _toposort(this, new Set())
  }
  @cache
  get tuplize(): any[] {
    return [this.op.value, this.arg, this.dtype, this.src.map((x) => x.tuplize)]
  }

  // *** uop shape stuff ***
  @cache
  get st(): undefined | ShapeTracker {
    if (this.op === Ops.MULTI) {
      return ShapeTracker.from_shape(this.real_lbs[0].shape.map((s, a) => a === this.axis ? sum(this.real_lbs.map((y) => y.shape[a])) : s))
    }
    // these ops define a ShapeTracker from the arg
    if (this.op === Ops.VIEW) return this.arg
    if (GroupOp.Movement.includes(this.op)) return this.src[0].st!.mop(this.op, this.arg)
    // buffer ops return the ShapeTracker from sources
    if (GroupOp.Buffer.includes(this.op)) return this.src.filter((x) => x.op === Ops.VIEW).map((x) => x.st)[0]
    const src_sts = this.src.filter((x) => x.st !== undefined).map((x) => x.st!)
    if (!src_sts.length) return undefined
    if (!all_same(src_sts.map((x) => x.shape))) throw new Error(`UOp sources must have the same shape ${this} ${src_sts.map((x) => x.shape)}`)
    let shape
    if ([Ops.BUFFER_VIEW, Ops.BITCAST].includes(this.op)) {
      shape = src_sts[0].shape
      if (this.dtype.itemsize !== this.src[0].dtype.itemsize) shape = [...shape.slice(0, -1), idiv(shape.at(-1)! as number * this.src[0].dtype.itemsize, this.dtype.itemsize)]
    } // only reduce ops are allowed to change shape, everything else derives shape from sources
    else if ([Ops.REDUCE_AXIS, Ops.WMMA].includes(this.op)) shape = src_sts[0].reduce(this.axis_arg)
    else shape = src_sts[0].shape
    return ShapeTracker.from_shape(shape)
  }
  @cache
  get full_shape(): sint[] {
    if (this.op === Ops.VIEW) return this.shape
    //  TODO: this should check if st is None, it cannot because local reduce has implicit movement ops
    return zip(...this.src.filter((x) => ![Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.CONST].includes(x.op)).map((x) => x.full_shape)).map((x) => smax(...x))
  }
  get shape() {
    return this.st!.shape
  }
  get size(): number {
    return this.op === Ops.BUFFER ? this.arg[1] : this.st!.size
  }
  // *** uop evaluation ***

  simplify = (): UOp => graph_rewrite(this, symbolic)
  ssimplify = (): UOp => {
    const ret = this.simplify()
    return ret.op === Ops.CONST ? ret.arg : ret
  }
  _eval = <T extends new (...args: any[]) => void>(dtypes: DType[], expectedType: T): InstanceType<T> => {
    if (!dtypes.includes(this.dtype)) throw new Error(`eval with wrong dtype ${this}`)
    const simpleThis = this.simplify()
    const [vmin, vmax] = simpleThis._min_max
    if (is_eq(vmin, vmax)) throw new Error(`eval failed to be a single number, range is ${vmin} to ${vmax} in ${simpleThis.render()}`)
    // if ((vmin instanceof expectedType)) throw new Error(`vmin is wrong dtype ${typeof vmin} != ${expectedType}`)
    return vmin as InstanceType<T>
  }
  __bool__ = () => this._eval([dtypes.bool], Boolean)
  __int__ = () => this._eval(dtypes.ints, Number)
  __float__ = () => this._eval(dtypes.floats, Number)
  substitute = (dvars: Map<UOp, UOp>) => {
    return graph_rewrite(this, _substitute, dvars, true)
  }

  // *** uop syntactic sugar ***
  get st_arg(): ShapeTracker {
    if (!GroupOp.Buffer.includes(this.op)) throw new Error(`st_arg called on ${this.op.toString()}`)
    return this.st!
  }
  get const_arg(): ConstType {
    let ret
    if (this.base.op === Ops.CONST) ret = this.base.arg
    else throw new Error(`const_arg called on ${this.base.op}`)
    if (!isConst(ret)) throw new Error(`const_arg trying to return ${ret}`)
    return ret
  }
  get axis_arg() {
    if (![Ops.REDUCE_AXIS, Ops.WMMA].includes(this.op)) throw new Error(`axis_arg called on ${this.op}`)
    const ret = this.op === Ops.REDUCE_AXIS ? this.arg[1] : this.arg[7]
    if (!(Array.isArray(ret) && ret.every((x) => typeof x === 'number'))) throw new Error(`axis_arg trying to return ${ret}`)
    return ret
  }
  static sink = (...srcs: UOp[]) => new UOp(Ops.SINK, dtypes.void, [...srcs])
  detach = () => new UOp(Ops.DETACH, this.dtype, [this])
  index = (idx: UOp, valid?: UOp) => new UOp(Ops.INDEX, this.dtype, valid !== undefined ? [this, idx, valid] : [this, idx])
  override const_like = (b: ConstLike<UOp>) => {
    // constants can optionally have a DEVICE source
    if (this._device === undefined) return UOp.const(this.dtype, b)
    if (Array.isArray(this.device)) return UOp.multi(this.device.map((d) => UOp.metaop(Ops.CONST, this.shape, this.dtype, d, b)), undefined)
    return UOp.metaop(Ops.CONST, this.shape, this.dtype, this.device, b)
  }
  broadcast = (count: number) => {
    if (this.dtype.count !== 1) throw new Error(`dtype.count !==1`)
    if (count === 1) return this
    return new UOp(Ops.VECTORIZE, this.dtype.vec(count), range(count).map(() => this))
  }
  cast = (dtype: DType) => new UOp(Ops.CAST, dtype, [this])
  bitcast = (dtype: DType) => new UOp(Ops.BITCAST, dtype, [this])
  gep = (i: number[] | number) => {
    if (typeof i === 'number') {
      // NOTE: these are just shortcuts to not have to create and fold later
      if (this.op === Ops.VECTORIZE) return this.src[i]
      if (this.op === Ops.VCONST) return UOp.const(this.dtype.scalar(), this.arg[i])
      if (this.op === Ops.CONST) return UOp.const(this.dtype.scalar(), this.arg)
      i = [i]
    }
    if ((this.dtype.vcount === i.length && is_eq(i, range(i.length))) || this.dtype === dtypes.void) return this
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
  valid = (st: ShapeTracker) => {
    if (![Ops.CONST, Ops.DEFINE_VAR].includes(this.op)) throw new Error(`can only create VALID from a constant, got ${this.op}`)
    return new UOp(Ops.VALID, dtypes.bool, [st.to_uop()]).where(this, 0)
  }
  static range = (dtype: DType, start: sint, end: sint, idx: number) => new UOp(Ops.RANGE, dtype, [sint_to_uop(start), sint_to_uop(end)], idx)
  _reduce_op = (op: Ops, axis: number[]) => {
    axis = sorted(axis.filter((x) => resolve(ne(this.shape[x], 1))))
    return axis.length === 0 ? this : new UOp(Ops.REDUCE_AXIS, this.dtype, [this], [op, axis])
  }
  r = (op: Ops, axis: number[]): UOp => {
    const new_shape = this.st!.reduce(axis)

    // TODO: can we split symbolic shape if the reduce axis is not symbolic?
    // TODO: this shouldn't be here, it belongs in scheduler! that's why it broke multi
    if (!SPLIT_REDUCEOP || Array.isArray(this._device) || !all_int(this.shape) || this.shape.includes(0) || (idiv(prod(this.shape), prod(new_shape)) as number) < get_number_env('REDUCEOP_SPLIT_THRESHOLD', 32768)) {
      return this._reduce_op(op, axis)
    }

    // if there are few globals, make some reduces into globals by splitting into two kernels
    // cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
    //   ~2**10 should be enough if GROUP is used
    // 256 split maximum should be "negligible reduce" for low prod(new_shape), 8 split minimum.
    // split is moved to the end to provide maximum locality for the second phase reduce.
    const self_real_strides = this.st!.real_strides(true)
    const split_candidates = range(Math.min(256, (2 ** get_number_env('REDUCEOP_SPLIT_SIZE', 22), prod(new_shape) as number)), 8 - 1, -1)
      .flatMap((x) => axis.filter((i) => mod(this.shape[i] as number, x) === 0 && self_real_strides[i] !== 0).map((i) => [i, x]))
    if (!split_candidates.length) return this._reduce_op(op, axis)
    const [dim_to_split, divisor] = split_candidates[0]
    const splitted_shape = [...this.shape.slice(0, dim_to_split), divisor, idiv(this.shape[dim_to_split], divisor), ...this.shape.slice(dim_to_split + 1)]
    const splitted = this.reshape(splitted_shape).permute([...range(splitted_shape.length).filter((x) => x !== dim_to_split), dim_to_split])
    if (DEBUG >= 3) console.log(`split ${divisor}: ${this.shape} -> ${splitted.shape} -> ${new_shape}`)
    return splitted._reduce_op(op, axis)._reduce_op(op, [new_shape.length]).reshape(new_shape) // reduce original axes, then split  assign = (x: UOp) => new UOp(Ops.ASSIGN, this.dtype, [this, x])
  }
  assign = (x: UOp) => new UOp(Ops.ASSIGN, this.dtype, [this, x])
  contiguous = () => this.alu(Ops.CONTIGUOUS)
  contiguous_backward = () => this.alu(Ops.CONTIGUOUS_BACKWARD)

  // *** from MultiLazyBuffer ***

  static multi = (more: UOp[], axis?: number, real?: boolean[]) => {
    const parents = [...more]
    if (!all_same(parents.map((x) => x.dtype))) throw new Error('multi parents must have the same dtype')
    return new UOp(Ops.MULTI, more[0].dtype, parents, [axis, real !== undefined ? real : range(parents.length).map(() => true)])
  }

  get bounds() {
    if (this.axis === undefined) throw new Error('bounds is not defined when axis is None')
    return pairwise(accumulate(this.src.map((lb) => lb.shape[this.axis]), undefined, 0))
  }
  get axis() {
    assert(this.op === Ops.MULTI)
    return this.arg[0]
  }
  get real() {
    assert(this.op === Ops.MULTI)
    return this.arg[1]
  }
  get real_lbs() {
    return zip(this.src, this.real).filter(([lb, r]) => r).map(([lb]) => lb)
  }

  shard = (devices: DeviceType[], axis?: number): UOp => {
    let lbs: UOp[]
    if (axis === undefined) lbs = range(devices.length).map(() => this)
    else {
      if (mod(this.shape[axis], devices.length) !== 0) throw new Error(`multi axis uneven: this.shape[axis]=${this.shape[axis]} axis=${axis} devices.length=${devices.length}`)
      let sz = idiv(this.shape[axis] as number, devices.length)
      const sizes = range(devices.length).map((i) => max([0, min([sz, this.shape[axis!] as number - sz * i])]))
      lbs = []
      let off = 0
      for (sz of sizes) {
        lbs.push(this.shrink(this.shape.map((s, i) => i !== axis ? [0, s] : [off, off + sz])))
      }
      off += sz
    }
    const sharded_lbs = zip(lbs, devices).map(([lb, d]) => lb.copy_to_device(d))
    // NOTE: this contiguous is making it impossible for the scheduler to do late const folding
    return UOp.multi(sharded_lbs.map((lb) => lb.contiguous()), axis)
  }

  // *** from LazyBuffer ***

  static metaop = (op: Ops, shape: sint[], dtype: DType, device: string, arg?: any): UOp => {
    // Tensor const is CONST(VIEW(DEVICE)) -> RESHAPE -> EXPAND
    if (op === Ops.CONST) {
      if (!isConst(arg)) throw new Error(`trying to create CONST with arg=${arg}`)
      return UOp.const(dtype, arg!)
        .replace({ src: [new UOp(Ops.VIEW, dtypes.void, [new UOp(Ops.DEVICE, undefined, undefined, device)], ShapeTracker.from_shape([]))] })
        .reshape(range(shape.length).map((x) => 1)).expand(shape)
    }
    // Tensor variable binding is BIND(VAR(VIEW(DEVICE)), CONST(VIEW(DEVICE)))
    if (op === Ops.BIND) {
      const [variable, val] = arg.unbind()
      return variable.replace({ src: [new UOp(Ops.VIEW, dtypes.void, [new UOp(Ops.DEVICE, undefined, undefined, device)], ShapeTracker.from_shape(shape))] }).bind(val)
    }
    // otherwise it's just a VIEW(BUFFER)
    const st = ShapeTracker.from_shape(shape)
    return new UOp(Ops.VIEW, dtype, [UOp.new_buffer(device, st.size, dtype)], st)
  }
  copy_to_device = (device: DeviceType | DeviceType[], clone = false): UOp => {
    // if it's a shrink, do the shrink before the copy with CONTIGUOUS
    if (prod(this.shape) < prod(this.base.shape)) return this.contiguous().copy_to_device(device)
    // COPY is COPY(DEVICE, copyin.base) -> VIEW(copyin.st)
    let ret = new UOp(Ops.COPY, this.base.dtype, [new UOp(Ops.DEVICE, undefined, undefined, device), this.base], clone)
    let op_arg: [Ops, any][] = []
    let mop: UOp = this
    while (mop !== this.base) {
      op_arg.push([mop.op, mop.arg])
      mop = mop.src[0]
    }
    for (const [op, arg] of op_arg.toReversed()) ret = new UOp(op, ret.dtype, [ret], arg)
    return ret
  }
  clone = (): UOp => this.copy_to_device(this.device, true)
  get metadata() {
    return all_metadata.get(this)
  }

  // *** uop movement ops ***
  get base(): UOp {
    if (GroupOp.Movement.includes(this.op)) return this.src[0].base
    return this.op === Ops.VIEW && this.src.length === 1 && this.src[0].op !== Ops.BUFFER ? this.src[0].base : this
  }
  view = (new_st: ShapeTracker) => new UOp(Ops.VIEW, this.dtype, [this.base], new_st)

  _mop = (op: Ops, arg: any) => {
    const ret = new UOp(op, this.dtype, [this], arg)
    if (this.st === ret.st) return this // ignore NOOPs, also check ret.st
    return ret
  }
  reshape = (arg: sint[]) => this._mop(Ops.RESHAPE, arg)
  pad = (arg: [sint, sint][]) => this._mop(Ops.PAD, arg)
  expand = (arg: sint[]) => this._mop(Ops.EXPAND, arg)
  permute = (arg: sint[]) => this._mop(Ops.PERMUTE, arg)
  shrink = (arg: [sint, sint][]) => this._mop(Ops.SHRINK, arg)
  stride = (arg: sint[]) => this._mop(Ops.STRIDE, arg)

  // *** uop Buffer stuff ***
  static buffer_num = counter(0)
  static new_buffer = (device: string, size: number, dtype: DType) => new UOp(Ops.BUFFER, dtype, [new UOp(Ops.DEVICE, undefined, undefined, device)], [UOp.buffer_num.next().value, size])
  get device(): DeviceType | DeviceType[] {
    return this._device!
  }
  @cache
  get _device(): DeviceType | DeviceType[] | undefined {
    if (this.op === Ops.DEVICE) return this.arg
    if (this.op === Ops.MULTI) return this.src.map((x) => x.device as DeviceType)
    return this.src.filter((x) => x._device !== undefined)[0]?._device
  }
  get buf_uop(): UOp {
    if (this.op === Ops.BUFFER) return this
    if (![...GroupOp.Buffer, Ops.ASSIGN, Ops.VIEW].includes(this.base.op)) throw new Error(`buf_uop called on ${this.op}`)
    return this.src[0].buf_uop
  }
  buf_uop_view = (): UOp => this.buf_uop.view(this.st!)

  // *** uop Variable stuff ***

  static variable = (name: string, minVal: ConstType<UOp> = dtypes.min(dtypes.int), maxVal: ConstType<UOp> = dtypes.max(dtypes.int), dtype = dtypes.int) => {
    if ((minVal instanceof UOp) || (maxVal instanceof UOp)) throw new Error(`can't create Variable ${name} with ${minVal}/${maxVal}`)
    return new UOp(Ops.DEFINE_VAR, dtype, undefined, [name, minVal, maxVal])
  }
  get expr() {
    if (this.op !== Ops.DEFINE_VAR) throw new Error(`op is ${this.op}, need DEFINE_VAR`)
    return this.arg[0]
  }
  bind = (val: number) => {
    if (this.op !== Ops.DEFINE_VAR) throw new Error(`op is ${this.op}, need DEFINE_VAR`)
    if (this.arg[1] > val || val > this.arg[2]) throw new Error(`bind ${val} not in range [${this.arg[1]}, ${this.arg[2]}]`)
    return new UOp(Ops.BIND, this.dtype, [this, this.const_like(val)])
  }
  unbind = (): [Variable, number] => {
    if (this.op !== Ops.BIND || this.src[0].op !== Ops.DEFINE_VAR || this.src[1].op !== Ops.CONST) throw new Error(`can't unbind ${this}`)
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

  // *** uop symbolic stuff ***

  /**largest known int that divides this */
  constFactor = (): number => {
    if (this.op === Ops.CONST) return this.arg
    if (this.op === Ops.VCONST) return math_gcd(...this.arg)
    if (this.op === Ops.ADD) return math_gcd(this.src[0].constFactor(), this.src[1].constFactor())
    if (this.op === Ops.MUL) return this.src[1].op === Ops.CONST ? this.src[0].op === Ops.CONST ? this.src[0].arg : this.src[1].arg : 1
    return 1
  }
  divides = (v: number): UOp | undefined => {
    if (v === 1) return this
    if (this.op === Ops.CONST) return mod(this.arg, v) === 0 ? this.const_like(idiv(this.arg, v)) : undefined
    if (this.op === Ops.VCONST) return this.arg.every((x: number) => mod(x, v) === 0) ? this.const_like(this.arg.map((x: number) => idiv(x, v))) : undefined

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
    return this._min_max[0]
  }
  get vmax() {
    return this._min_max[1]
  }
  // Actually can return boolean as well, but types don't like it
  @cache
  get _min_max(): [number | bigint, number | bigint] {
    if (GroupOp.Binary.includes(this.op) && !dtypes.is_float(this.dtype)) {
      const [s0_vmin, s0_vmax] = this.src[0]._min_max, [s1_vmin, s1_vmax] = this.src[1]._min_max
      if (this.op === Ops.ADD) return [add(s0_vmin, s1_vmin), add(s0_vmax, s1_vmax)]
      if (this.op === Ops.MUL) {
        const vals = [mul(s0_vmin, s1_vmin), mul(s0_vmin, s1_vmax), mul(s0_vmax, s1_vmin), mul(s0_vmax, s1_vmax)]
        return [min(vals), max(vals)]
      }
      if (this.op === Ops.SHL && s1_vmin === s1_vmax && all_int([s0_vmin, s0_vmax, s1_vmin])) return [lshift(s0_vmin, s1_vmin), lshift(s0_vmax, s1_vmin)]
      if (this.op === Ops.SHR && s1_vmin === s1_vmax && all_int([s0_vmin, s0_vmax, s1_vmin])) return [rshift(s0_vmin, s1_vmin), rshift(s0_vmax, s1_vmin)]
      if (this.op === Ops.MOD && s1_vmin > 0) return s0_vmin >= 0 ? [0, sub(s1_vmax, 1)] : [-sub(s1_vmax, 1), sub(s1_vmax, 1)]
      if (this.op === Ops.IDIV) {
        if (s1_vmin === s1_vmax) { // min/max are equal in a CONST
          if (s1_vmin > 0) return [idiv(s0_vmin, s1_vmin), idiv(s0_vmax, s1_vmin)]
          if (s1_vmin < 0 && s0_vmin >= 0) return [-idiv(s0_vmax, -s1_vmin), -idiv(s0_vmin, -s1_vmin)]
        }
        // don't know exact bounds, but know the sign
        if ((s0_vmax <= 0 && s1_vmin < 0) || (s0_vmin >= 0 && s1_vmin > 0)) return [0, Number(dtypes.max(this.dtype))]
        if ((s0_vmax <= 0 && s1_vmin > 0) || (s0_vmin >= 0 && s1_vmin < 0)) return [Number(dtypes.min(this.dtype)), 0]
      }
      if (this.op === Ops.MAX) return [max([s0_vmin, s1_vmin]), max([s0_vmax, s1_vmax])]
      if (this.op === Ops.CMPLT) return [Number(s0_vmax < s1_vmin), Number(s0_vmin < s1_vmax)]
      if (this.op === Ops.CMPNE) return [Number((s0_vmax < s1_vmin) || (s1_vmax < s0_vmin)), Number(!(s0_vmin === s0_vmax && s0_vmax === s1_vmin && s1_vmin === s1_vmax))]
      if (this.dtype === dtypes.bool) {
        if (this.op === Ops.OR) return [Number(s0_vmin || s1_vmin), Number(s0_vmax || s1_vmax)]
        if (this.op === Ops.AND) return [Number(s0_vmin && s1_vmin), Number(s0_vmax && s1_vmax)]
      }
    }
    // float has NAN issue and we use explicit NAN in transcendental
    if (this.op === Ops.WHERE && dtypes.is_int(this.dtype)) return [min([this.src[1].vmin, this.src[2].vmin]), max([this.src[1].vmax, this.src[2].vmax])]
    // NOTE: returned UOp is assumed to be CONST
    if (this.op === Ops.DEFINE_VAR && this.arg) return [this.arg[1], this.arg[2]]
    if (this.op === Ops.RANGE) return [this.src[0].vmin, (this.src[1].sub(1)).vmax]
    if (this.op === Ops.BIND) return this.src[0]._min_max // ignore the bound value
    if ([Ops.UNROLL, Ops.VECTORIZE].includes(this.op)) return [min(this.src.map((x) => x.vmin)), max(this.src.map((x) => x.vmax))]
    // TODO: UOps.SPECIAL is UOps.DEFINE_VAR
    if (this.op === Ops.SPECIAL) return [0, typeof this.arg[1] === 'number' ? (this.arg[1] - 1) : this.arg[1].vmax]
    if (this.op === Ops.CONST) return [this.arg, this.arg]
    if (this.op === Ops.VCONST) return [min(this.arg), max(this.arg)]
    return [Number(dtypes.min(this.dtype)), Number(dtypes.max(this.dtype))]
  }
  @cache
  get _sym_fxn(): [(m: Record<string, number>) => number, string[]] {
    const sthis = this.simplify()
    const varnames: string[] = [...sthis.toposort].filter((x) => x.op === Ops.DEFINE_VAR).map((x) => x.arg[0])
    // TODO: sanitize varnames, or don't use naked eval while staying fast
    return [eval(`({${varnames.join(',')}})=>${sthis.render()}`), varnames]
  }
  sym_infer = (varVals: Map<UOp, number>) => {
    const [fxn, varnames] = this._sym_fxn
    const args = Object.fromEntries(varVals.entries().filter(([k, _]) => varnames.includes(k.arg[0])).map(([k, v]) => [k.arg[0] as string, v]))
    return fxn(args)
  }

  render = (simplify = true): string => {
    const ret = graph_rewrite(simplify ? this.simplify() : this, renderer)
    return ret.op === Ops.NOOP ? ret.arg : ret.toString()
  }
}

export class KernelInfo {
  key: string
  static cache = new WeakValueMap<string, KernelInfo>()
  constructor(
    public local_dims = 0, // number of local dimensions  (this is remapping RANGE to SPECIAL)
    public upcasted = 0, // count that are upcasted     (this is remapping RANGE to UNROLL)
    public dont_use_locals = false, // don't use local indexing
  ) {
    this.key = get_key(local_dims, upcasted, dont_use_locals)
    Object.freeze(this)
    return KernelInfo.cache.setDefault(this.key, this)
  }
  toString = () => `new KernelInfo(${this.local_dims}, ${this.upcasted}, ${this.dont_use_locals})`
}

// ***** ops in python *****
const safe_exp2 = (x: ConstType) => {
  try {
    return typeof x !== 'bigint' ? 2 ** Number(x) : 2n ** x
  } catch {
    return Infinity
  }
}
export const python_alu = new Map<Ops, (...x: ConstType[]) => ConstType>([
  [Ops.LOG2, (x) => x === 0 ? x > 0 ? Math.log2(2) : -Infinity : NaN],
  [Ops.EXP2, safe_exp2],
  [Ops.SQRT, (x) => ge(x, 0) ? sqrt(x) : NaN],
  [Ops.RECIP, (x) => ne(x, 0) ? div(1, x) : ge(x, 0) ? Infinity : -Infinity],
  [Ops.SIN, (x) => !isInf(x as number) ? sin(x) : NaN],
  [Ops.NEG, (x) => neg(x)],
  [Ops.ADD, (x, y) => add(x, y)],
  [Ops.SUB, (x, y) => sub(x, y)],
  [Ops.MUL, (x, y) => mul(x, y)],
  [Ops.CMPNE, (x, y) => ne(x, y)],
  [Ops.CMPLT, (x, y) => lt(x, y)],
  [Ops.XOR, (x, y) => xor(x, y)],
  [Ops.OR, (x, y) => or(x, y)],
  [Ops.AND, (x, y) => and(x, y)],
  [Ops.SHR, (x, y) => rshift(x, y)],
  [Ops.SHL, (x, y) => lshift(x, y)],
  [Ops.MAX, (...args) => max(args)],
  [Ops.MOD, (x, y) => mul(mod(abs(trunc(x)), abs(trunc(y))), lt(x, 0) ? -1 : 1)],
  [Ops.IDIV, (x, y) => ne(y, 0) ? mul(idiv(abs(x), abs(y)), (mul(x, y) < 0) ? -1 : 1) : 0],
  [Ops.MULACC, (x, y, z) => add(mul(x, y), z)],
  [Ops.WHERE, (x, y, z) => x ? y : z],
])

export const exec_alu = (op: Ops, dtype: DType, operands: ConstType[], truncateOutput = true): any => {
  if (dtype.count > 1) return range(dtype.count).map((i) => exec_alu(op, dtype.scalar(), operands.map((x) => Array.isArray(x) ? x[i] : x)))
  const alu = python_alu.get(op)!(...operands)
  return truncateOutput ? truncate.get(dtype)!(alu) : alu
}
// ***** uop helpers *****

export const print_uops = (uops: UOp[]) => {
  for (const [i, u] of uops.entries()) {
    const formattedParents = u.src.map((x) => uops.includes(x) ? x.op !== Ops.CONST ? uops.indexOf(x) : `${x.arg}` : '--')
    console.log(`${i.toString().padStart(4)} ${u.op.toString().padEnd(20)} ${u.dtype.toString().padEnd(30)} ${list_str(formattedParents).padEnd(32)} ${list_str(u.arg)}`)
  }
}

// ***** pattern matcher *****
function getLocation(): [string, number] {
  // TODO: Doesn't work in browser
  // const [file, line] = new Error().stack!.split('\n')[2]?.split('file://')[1]?.split(')')[0]?.split(':')
  const [file, line] = ['todo.ts', 1]
  return [file, Number(line)]
}
const lines = (fn: string): string[] => {
  return Env.readFileSync(fn).toString().split('\n')
}

export type UPatInput = { op?: Ops | Ops[]; dtype?: DType | DType[]; src?: UPat | UPat[] | [UPat[]]; arg?: any; name?: string; allow_any_len?: boolean; location?: any; custom_early_reject?: Ops[] }
export type UPatFn<Ctx = unknown, Res = UOp | undefined> = (args: Record<string, UOp> & { ctx: Ctx }) => Res
export type Pattern<Ctx = unknown, Res = UOp | undefined> = [UPat, UPatFn<Ctx, Res>]
export class UPat extends MathTrait<UPat> {
  op?: Ops[]
  dtype?: DType[]
  _in_src?: UPat | UPat[] | [UPat[]]
  src?: UPat[][]
  allowed_len: number
  location: [string, number]
  early_reject: Ops[]
  fn = <Ctx = unknown, Res = UOp | undefined>(fn: UPatFn<Ctx, Res>): Pattern<Ctx, Res> => [this, fn]
  constructor(op?: Ops | Ops[], dtype?: DType | DType[], src?: UPat | UPat[] | [UPat[]], public arg?: any, public name?: string, allow_any_len?: boolean, location?: any, public custom_early_reject?: Ops[]) {
    super()
    // TODO reverse this condition
    if (!(op === undefined || !(!Array.isArray(op) && Object.values(Ops).includes(op)) || !(Array.isArray(op) && Object.values(Ops).includes(op[0])))) throw new Error('op must be Ops or tuple of Ops')
    this.op = Array.isArray(op) ? op : op !== undefined ? [op] : undefined
    this.dtype = Array.isArray(dtype) ? dtype : dtype !== undefined ? [dtype] : undefined
    this._in_src = src
    if (this.name === 'ctx') throw new Error("UPat can't be named ctx")

    // try all permutations if it's a list (we use src[][])
    if (Array.isArray(src) && Array.isArray(src[0])) this.src = !all_same(src[0]) ? permutations(src[0]) : [src[0]]
    // only one if it's a tuple (we use src[])
    else if (Array.isArray(src)) this.src = [src as UPat[]]
    // repeat if it's a UPat
    else if (src instanceof UPat) this.src = [range(100).map(() => src!) as UPat[]] // KAREL: this is a hack

    // NOTE: This is here because we can't differentaite between list and tuple so we use Upat[][] to achieve the same thing as list. but after this part the difference isn't needed anymore so we convert back to UPat[]
    if (Array.isArray(src) && src?.length === 1 && Array.isArray(src[0])) src = src[0]
    this.allowed_len = (allow_any_len || src instanceof UPat || src === undefined) ? -1 : src.length
    this.location = location || getLocation()

    if (custom_early_reject !== undefined) this.early_reject = custom_early_reject
    else {
      const upatMatch = src instanceof UPat ? [src] : (src === undefined ? [] : this.src![0])
      this.early_reject = upatMatch.filter((pp) => pp.op !== undefined && pp.op.length === 1).map((pp) => pp.op![0])
    }
  }
  named = (name?: string) => {
    this.name = name
    return this
  }
  static any = (src: UPat[]) => new UPatAny(undefined, undefined, src)
  static var = (name?: string, dtype?: DType | DType[]) => new UPat(undefined, dtype, undefined, undefined, name)
  static cvar = (name?: string, dtype?: DType, vec = true) => new UPat(vec ? [Ops.CONST, Ops.VCONST] : Ops.CONST, dtype).named(name)
  static const = (dtype?: DType | DType[], b?: ConstLike) => new UPat(Ops.CONST, dtype, undefined, b)

  // copied from UOp
  index = (idx: UPat, valid?: UPat) => new UPat(Ops.INDEX, this.dtype, valid !== undefined ? [this, idx, valid] : [this, idx])
  static index = (self: UPat, idx: UPat, valid?: UPat) => self.index(idx, valid)
  view = (st?: ShapeTracker, kwargs: Partial<UPatInput> = {}) => new UPat(kwargs.op || Ops.VIEW, kwargs.dtype || this.dtype, kwargs.src || [this], kwargs.arg || st, kwargs.name, kwargs.allow_any_len, kwargs.location, kwargs.custom_early_reject)
  cast = (dtype?: DType) => new UPat(Ops.CAST, dtype, [this])
  bitcast = (dtype?: DType) => new UPat(Ops.BITCAST, dtype, [this])
  gep = (i: number) => new UPat(Ops.GEP, undefined, [this], [i])
  load = (src?: UPat[], kwargs: Partial<UPatInput> = {}) => new UPat(kwargs.op || Ops.LOAD, kwargs.dtype, kwargs.src || [this, ...(src || [])], kwargs.arg, kwargs.name, kwargs.allow_any_len, kwargs.location, kwargs.custom_early_reject)
  static load = (src: UPat[], kwargs: Partial<UPatInput> = {}) => src[0].load(src.slice(1), kwargs)
  store = (src: UPat[], kwargs: Partial<UPatInput> = {}) => new UPat(kwargs.op || Ops.STORE, kwargs.dtype || dtypes.void, kwargs.src || [this, ...src], kwargs.arg, kwargs.name, kwargs.allow_any_len, kwargs.location, kwargs.custom_early_reject)
  static store = (src: UPat[], kwargs: Partial<UPatInput> = {}) => src[0].store(src.slice(1), kwargs)
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
  override toString = () => `new UPat(${list_str(this.op?.map((o) => o.toString()))}, ${list_str(this.dtype)}, ${list_str(this.src)}, ${list_str(this.arg)}, ${this.name}, ${this.allowed_len === 0})`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  match = (uop: UOp, store: Map<string, UOp>): Map<string, UOp>[] => {
    if (
      (this.op !== undefined && !this.op.includes(uop.op)) ||
      (this.name !== undefined && set_default(store, this.name, uop) !== uop) ||
      (this.dtype !== undefined && !this.dtype.includes(uop.dtype) && !this.dtype.includes(uop.dtype.scalar())) ||
      (this.arg !== undefined && !is_eq(this.arg, uop.arg)) ||
      (this.allowed_len !== -1 && uop.src.length !== this.allowed_len)
    ) return []
    if (this.src === undefined) return [store]
    let res: Map<string, UOp>[] = []
    for (const vp of this.src) {
      let stores = [new Map(store)]
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
  override match = (uop: UOp, store: Map<string, UOp>): Map<string, UOp>[] => {
    const matches = this.src![0].map((x) => x.match(uop, new Map(store)))
    return flatten(matches.filter((x) => x !== undefined))
  }
}

export class PatternMatcher<Ctx = unknown, Res = UOp | undefined> {
  patterns: [UPat, UPatFn<Ctx, Res>][]
  pdict = new Map<Ops, ([UPat, UPatFn<Ctx, Res>, Set<Ops>, boolean][])>()
  constructor(patterns: [UPat, UPatFn<Ctx, Res>][]) {
    this.patterns = patterns
    for (const [p, fxn] of this.patterns) {
      assert(p.op !== undefined)
      for (const uop of p.op || []) set_default(this.pdict, uop, []).push([p, fxn, new Set(p.early_reject), fxn.toString().includes('ctx')])
    }
  }

  add = <NewCtx, NewRes>(more: PatternMatcher<NewCtx, NewRes>) => new PatternMatcher<Ctx | NewCtx, Res | NewRes>([...this.patterns, ...more.patterns] as any)

  rewrite = (uop: UOp, ctx?: any): Res | undefined => {
    const ler = new Set(uop.src.map((u) => u.op))
    for (const [p, fxn, early_reject, hasCtx] of this.pdict.get(uop.op) || []) {
      if (!is_subset(ler, early_reject)) continue
      for (const match of p.match(uop, new Map())) {
        const ret = hasCtx ? fxn({ ctx, ...Object.fromEntries(match) } as any) : fxn(Object.fromEntries(match) as any)
        if (ret !== undefined) return ret
      }
    }
    return undefined
  }
}

const TRACK_MATCH_STATS = get_number_env('TRACK_MATCH_STATS', get_env('VIZ') ? 2 : 0)
const match_stats = new Map<UPat, number[]>()
export class TrackedGraphRewrite {
  loc!: [string, number] // location that called graph_rewrite
  sink!: UOp // the sink input to graph_rewrite
  matches!: [UOp, UOp, UPat][] // before+after of all the matches
}
const tracked_keys: any[] = []
const tracked_ctxs: TrackedGraphRewrite[][] = []
const _name_cnt = new Map<string, number>()
const track_rewrites = (named = false) => {
  throw new NotImplemented()
}

export class TrackedPatternMatcher<Ctx> extends PatternMatcher<Ctx> {
  override rewrite = (uop: UOp, ctx?: Ctx): UOp | undefined => {
    throw new NotImplemented()
  }
}
if (TRACK_MATCH_STATS) {
  throw new NotImplemented()
}

export const launch_viz = (env_str: string, data: string) => {
  throw new NotImplemented()
}

// *** simple graph rewrite engine ***
export class RewriteContext<Ctx> {
  pm: PatternMatcher<Ctx>
  ctx?: Ctx
  replace = new Map<UOp, UOp>()
  constructor(pm: PatternMatcher<Ctx>, ctx?: Ctx) {
    this.pm = pm
    this.ctx = ctx
  }
  top_down_rewrite = (n: UOp): UOp => {
    const rn = this.replace.get(n)
    if (rn !== undefined) return rn
    const new_src = n.src.map((x) => this.top_down_rewrite(x))
    const new_n = is_eq(new_src, n.src) ? this.pm.rewrite(n, this.ctx) : new UOp(n.op, n.dtype, new_src, n.arg)
    const ret = new_n === undefined ? n : this.top_down_rewrite(new_n)
    this.replace.set(n, ret)
    return ret
  }
  bottom_up_rewrite = (n: UOp): UOp => {
    const rn = this.replace.get(n)
    if (rn !== undefined) return rn
    let new_n: UOp | undefined = n
    let last_n!: UOp
    while (new_n !== undefined) [last_n, new_n] = [new_n, this.pm.rewrite(new_n, this.ctx)]
    const new_src = last_n.src.map((x) => this.bottom_up_rewrite(x))
    const ret = is_eq(new_src, last_n.src) ? last_n : this.bottom_up_rewrite(new UOp(last_n.op, last_n.dtype, new_src, last_n.arg))
    this.replace.set(n, ret)
    return ret
  }
}
export const graph_rewrite = <Ctx>(sink: UOp, pm: PatternMatcher<Ctx>, ctx?: Ctx, bottom_up = false): UOp => {
  if (TRACK_MATCH_STATS >= 2 && !bottom_up && tracked_ctxs.length !== 0) { // TODO: make viz work with bottom_up=True
    // tracked_ctxs[-1].append(TrackedGraphRewrite(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink))
    throw new NotImplemented()
  }
  return bottom_up ? new RewriteContext(pm, ctx).bottom_up_rewrite(sink) : new RewriteContext(pm, ctx).top_down_rewrite(sink)
}
export const graph_rewrite_map = <Ctx>(sink: UOp, pm: PatternMatcher<Ctx>, ctx?: Ctx, bottom_up = false): Map<UOp, UOp> => {
  if (TRACK_MATCH_STATS >= 2 && !bottom_up && tracked_ctxs.length !== 0) { // TODO: make viz work with bottom_up=True
    //     tracked_ctxs[-1].append(TrackedGraphRewrite(((frm:=sys._getframe(1)).f_code.co_filename, frm.f_lineno), sink))
    throw new NotImplemented()
  }
  const rewrite_ctx = new RewriteContext(pm, ctx)
  return new Map([...sink.toposort].toReversed().map((k) => [k, bottom_up ? rewrite_ctx.bottom_up_rewrite(k) : rewrite_ctx.top_down_rewrite(k)]))
}
// ***** uop type spec *****

// this is the matcher for the final rendered UOps
// matcher functions returns True or False (or None to not match)
export const spec = new PatternMatcher<unknown, boolean | undefined>([
  new UPat(Ops.DEFINE_GLOBAL).named('x').fn(({ x }) => (x.dtype instanceof PtrDType || x.dtype instanceof ImageDType) && !x.dtype.local),
  new UPat(Ops.DEFINE_LOCAL).named('x').fn(({ x }) => x.dtype instanceof PtrDType && x.dtype.local),
  new UPat(Ops.DEFINE_ACC, undefined, [UPat.var('c')], undefined, 'x', true).fn(({ x, c }) => x.src.slice(1).every((y) => y.op === Ops.RANGE) && c.dtype === x.dtype),
  new UPat(Ops.DEFINE_VAR, undefined, []).named('x').fn(({ x }) => typeof x.arg[1] === 'number' && typeof x.arg[2] === 'number'),

  new UPat(Ops.RANGE, undefined, [new UPat(undefined).named('x'), new UPat(undefined).named('y')]).named('rng').fn(
    ({ rng, x, y }) => rng.dtype === x.dtype && x.dtype === y.dtype && typeof rng.arg === 'number',
  ),
  new UPat(Ops.SPECIAL, undefined, []).fn(() => true),

  // TODO: confirm the args of both of these are shapetrackers
  new UPat(Ops.VIEW, dtypes.void, []).fn(() => true),
  new UPat(Ops.VIEW, undefined, [UPat.var('src')]).named('x').fn(({ x, src }) => src.op !== Ops.STORE && x.dtype === src.dtype),
  new UPat(Ops.VALID, dtypes.bool, [new UPat(Ops.VIEW)]).fn(() => true),
  new UPat(Ops.CONST).named('x').fn(({ x }) => x.dtype === x.dtype.scalar() && dtypes.verify(x.arg, x.dtype)), // NOTE: this is slightly different from python, int(1) != float(1) in py but it is the same in TS

  // early LOAD has a <buf, shapetracker, store?>
  new UPat(Ops.LOAD, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat(Ops.VIEW)]).fn(() => true),
  new UPat(Ops.LOAD, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat(Ops.VIEW), new UPat(Ops.STORE)]).fn(() => true),
  // early STORE has a <buf, shapetracker, val>
  new UPat(Ops.STORE, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat(Ops.VIEW), new UPat()]).fn(() => true),
  // **** new style load/store ****

  // INDEX is used in new style load/store
  new UPat(Ops.INDEX, undefined, [new UPat([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL]), new UPat()]).fn(() => true),
  // LOAD takes a <bufidx, alt?, gate?, barrier?>
  new UPat(Ops.LOAD, undefined, [new UPat([Ops.INDEX, Ops.CAST])]).fn(() => true),
  new UPat(Ops.LOAD, undefined, [new UPat([Ops.INDEX, Ops.CAST]), new UPat([Ops.IF, Ops.BARRIER])]).fn(() => true),
  new UPat(Ops.LOAD, undefined, [new UPat([Ops.INDEX, Ops.CAST]), new UPat(undefined).named('alt'), new UPat(undefined, dtypes.bool)]).named('ld').fn(({ ld, alt }) => ld.dtype === alt.dtype),

  // STORE takes a <bufidx, val, gate?>
  new UPat(Ops.STORE, dtypes.void, [new UPat([Ops.INDEX, Ops.CAST]), new UPat()]).fn(() => true),
  new UPat(Ops.STORE, dtypes.void, [new UPat([Ops.INDEX, Ops.CAST]), new UPat(), new UPat(undefined, dtypes.bool)]).fn(() => true),
  new UPat(Ops.STORE, dtypes.void, [new UPat([Ops.INDEX, Ops.CAST]), new UPat(), new UPat(Ops.IF)]).fn(() => true),

  // most ALUs have all matching dtypes, except CMPLT, CMPNE, and WHERE
  new UPat(Ops.WHERE, undefined, [new UPat(undefined, dtypes.bool), new UPat(undefined).named('x'), new UPat(undefined).named('y')]).named('w').fn(
    ({ w, x, y }) => w.dtype === x.dtype && x.dtype === y.dtype,
  ),
  new UPat([Ops.CMPLT, Ops.CMPNE], dtypes.bool, [new UPat(undefined).named('x'), new UPat(undefined).named('y')]).fn(({ x, y }) => x.dtype.base === y.dtype.base),

  // and SHL/SHR, the shift distance can be an int
  new UPat([Ops.SHL, Ops.SHR], undefined, [new UPat(undefined).named('x'), new UPat(undefined).named('y')]).named('a').fn(
    ({ a, x, y }) => a.dtype === x.dtype && [x.dtype, dtypes.uint].includes(y.dtype),
  ),
  new UPat(Ops.IDIV).named('x').fn(({ x }) => dtypes.is_int(x.dtype) ? undefined : false),
  new UPat(GroupOp.ALU).named('x').fn(({ x }) => x.src!.every((y) => x.dtype.base === y.dtype.base)),
  new UPat(Ops.ASSIGN, undefined, [new UPat([Ops.DEFINE_ACC, Ops.DEFINE_GLOBAL]), new UPat()]).fn(() => true),
  new UPat(Ops.ENDRANGE, dtypes.void, [new UPat(Ops.RANGE)]).fn(() => true),

  // WMMA has a <a, b, acc>
  new UPat(Ops.WMMA, undefined, [new UPat(), new UPat(), new UPat()]).named('x').fn(({ x }) => Array.isArray(x.arg) && x.arg.length === 8),
  new UPat(Ops.CONTRACT).named('x').fn(({ x }) => x.dtype.count === prod(x.arg.map((y: any) => y[1]))),
  new UPat(Ops.UNROLL).named('x').fn(({ x }) => x.src![0].dtype.count === prod(x.arg.map((y: any) => y[1]))),

  // if has a <gate, barrier?>

  new UPat(Ops.IF, dtypes.void, [new UPat()]).fn(() => true),
  new UPat(Ops.IF, dtypes.void, [new UPat(), new UPat(Ops.BARRIER)]).fn(() => true),
  new UPat(Ops.ENDIF, dtypes.void, [new UPat(Ops.IF)]).fn(() => true),
  new UPat(Ops.REDUCE_AXIS).named('x').fn(({ x }) => Array.isArray(x.arg) && x.arg.length === 2 && [Ops.ADD, Ops.MUL, Ops.MAX].includes(x.arg[0])),
  new UPat(Ops.GEP, undefined, [new UPat(undefined).named('src')]).named('gep').fn(({ gep, src }) => gep.dtype === src.dtype.scalar()),
  new UPat(Ops.VECTORIZE).named('x').fn(({ x }) => x.src.length > 1 && x.src.length === x.dtype.count && x.src.every((y) => x.dtype === y.dtype.vec(x.src.length))),
  new UPat([Ops.BITCAST, Ops.CAST], undefined, [new UPat()]).named('x').fn(({ x }) => x.arg === undefined),
  new UPat(Ops.BARRIER, dtypes.void, new UPat(Ops.STORE, undefined, undefined, undefined, undefined, true)).fn(() => true), // NOTE: all pointers must be local
  // NOTE: for testing, we let sinks be anything
  // new UPat(UOps.SINK, undefined, new UPat(UOps.STORE)).fn(() => true),
  new UPat(Ops.SINK, dtypes.void).fn(() => true),
  new UPat(Ops.NOOP).fn(() => true),

  // PTX LOAD/STORE
  new UPat([Ops.LOAD, Ops.STORE], undefined, [new UPat(undefined, dtypes.int64)], undefined, undefined, true).fn(() => true),
])

export const type_verify = (uops: UOp[], extra_specs: PatternMatcher<unknown, boolean | undefined>[] = []) => {
  const specs = [spec, ...extra_specs]
  for (const [i, u] of uops.entries()) {
    const spec_ret: (boolean | undefined)[] = specs.map((s) => s.rewrite(u))
    if (spec_ret.some((ret) => ret === false) || spec_ret.every((ret) => ret === undefined)) {
      print_uops(uops)
      throw new Error(`UOp verification failed at ${i} on ${u}`)
    }
  }
}

// *** most of symbolic lives here now ***

export function* split_uop(x: UOp, sep: Ops): Generator<UOp> {
  if (x.op === sep) { for (const s of x.src) yield* split_uop(s, sep) }
  else yield x
}

export const div_and_mod_folding = (x: UOp, y: UOp, which: typeof Ops.MOD | typeof Ops.IDIV, split_rem = false): undefined | UOp => {
  // simplify x // y or x % y, None means no change
  // simple cancel div/mod case
  if (y.vmin !== 0 && y.vmax !== 0) {
    const q = idiv(x.vmin, y.vmin)
    if (all_same([q, idiv(x.vmin, y.vmax), idiv(x.vmax, y.vmin), idiv(x.vmax, y.vmax)])) {
      return which === Ops.MOD ? sub(x, mul(q, y)) : x.const_like(q)
    }
  }
  const c: number = y.arg
  if ((y.op !== Ops.CONST) || c <= 0 || (x.dtype.count > 1)) return undefined

  let svars: UOp[] = [], factors: number[] = [], quotients: number[] = [], remainders: number[] = [], gcd = c, div = 1, const2 = 0, offset: number | bigint = 0, something_changed = false
  for (let u of split_uop(x, Ops.ADD)) {
    if (u.op === Ops.MOD && which === Ops.MOD && u.src[1].op === Ops.CONST && mod(u.src[1].arg, c) === 0) {
      u = u.src[0]
      something_changed = true
    }
    const f = u.constFactor(), v = u.divides(f)!
    const [q, r] = divmod(f, c)
    if (r === 0 || ((which === Ops.MOD || split_rem || u.op === Ops.CONST) && r !== f)) something_changed = true
    offset = add(offset, mul(r, v.vmin))
    if (u.op === Ops.CONST) const2 += f
    else { // div is the smallest common divisor of all terms
      if (f > 1 && mod(c, f) === 0 && (div === 1 || div > f)) div = f
      gcd = math_gcd(r, gcd)
      factors.push(f), svars.push(v), quotients.push(q), remainders.push(r)
    }
  }
  offset = mod(offset, c)
  let ubound = offset, lbound = offset
  // we can fold if the expression has only one non-constant term and this term can only take on two values
  let v = svars[0]
  if (svars.length === 1 && sub(v.vmax, v.vmin) === 1) {
    const r = sub(mod(add(offset, remainders[0]), c), mod(offset, c))
    offset = sub(offset, mul(r, v.vmin))
    if (which === Ops.MOD) return add(mul(r, v), offset)
    return add(mul(idiv(sub(factors[0], r), c), v), idiv(sub(const2, offset), c))
  }
  // a//c = (a-a%c)/c, if we can fold a%c, we can fold a//c
  // within a mod we can freely subtract multiples of c, we use this to see if a is congruent to an expression whose vmin/vmax are between 0 and c
  let exitedWithBreak = false
  let r
  for ([r, v] of zip(remainders, svars)) {
    if (r > idiv(c, 2)) {
      r = r - c
      lbound = add(lbound, mul(r, sub(v.vmax, v.vmin)))
      if (lbound < 0) {
        exitedWithBreak = true
        break
      }
    } else {
      ubound = add(ubound, mul(r, sub(v.vmax, v.vmin)))
      if (ubound >= c) {
        exitedWithBreak = true
        break
      }
    }
    offset = sub(offset, mul(r, v.vmin)) // determine what the new offset would be
  }
  if (!exitedWithBreak) { // vmin/vmax of the remainder is between 0 and c, we can remove the mod/div
    remainders = remainders.map((r) => Math.min(Math.abs(r), Math.abs(r - c)))
    if (which === Ops.MOD) return zip(remainders, svars).reduce((acc, [r, v]) => acc.add(mul(r, v)), x.const_like(offset))
    return zip(factors, remainders, svars).reduce((acc, [f, r, v]) => acc.add(mul(idiv(f - r, c), v)), x.const_like(idiv(sub(const2, offset), c)))
  }

  if (gcd !== 1) something_changed = true
  if (!something_changed) {
    if (which === Ops.IDIV && (1 < div && div < c)) {
      const newx = div_and_mod_folding(x, UOp.const(dtypes.int, div), Ops.IDIV)
      if (newx !== undefined) return newx.idiv(idiv(c, div))
    }
    return undefined
  }
  let quo = x.const_like(idiv(const2, c)), rem = x.const_like(idiv(mod(const2, c), gcd))
  for (const [q, r, f, v] of zip(quotients, remainders, factors, svars)) {
    if (which === Ops.IDIV && !split_rem && r !== 0) {
      rem = rem.add(mul(idiv(f, gcd), v))
    } else {
      rem = rem.add(mul(idiv(r, gcd), v))
      quo = quo.add(mul(q, v))
    }
  }
  if (which === Ops.MOD) return add(mul(gcd, mod(rem, idiv(c, gcd))), mod(const2, gcd))
  return add(idiv(rem, idiv(c, gcd)), quo)
}

const lt_folding = (x: UOp, c: number): UOp | undefined => {
  const [p, np] = partition(split_uop(x, Ops.ADD).toArray(), (u) => u.constFactor() === 1)
  const d = math_gcd(...np.map((u) => u.constFactor()), c)
  if (np && d > 1 && 0 <= sum(p.map((u) => u.vmin)) && sum(p.map((u) => u.vmax)) < d) {
    return np.reduce((p, c) => p.add(c), UOp.int(0)).divides(d)!.lt(idiv(c, d))
  }
  return undefined
}
const fold_unrolled_divs = (divs: UOp) => {
  // div pattern in unrolled arange
  // example: (x//4+(x+1)//4+(x+2)//4+(x+3)//4 -> x
  let add_chain = [...split_uop(divs, Ops.ADD)], denominator: number | undefined = undefined, seen_const: number[] = [], ans: UOp | undefined = undefined
  for (const u of add_chain) {
    if (!(u.op === Ops.IDIV && u.src[1].op === Ops.CONST)) return undefined
    if (denominator === undefined) denominator = u.src[1].arg
    if (denominator !== u.src[1].arg) return undefined
    // assumed CONST is the last of an ADD
    let s0 = u.src[0]
    if (s0.op === Ops.ADD && s0.src[1].op === Ops.CONST && s0.src[1].op === Ops.CONST) {
      seen_const.push(s0.src[1].arg)
      s0 = s0.src[0]
    } else seen_const.push(0)
    if (ans === undefined) ans = s0
    if (ans !== s0) return undefined
  }
  if (denominator === undefined) return undefined
  // the first (denominator-len(seen_const)) terms may have been folded to 0 already
  for (const i of range(denominator - seen_const.length)) {
    if (ans !== undefined && 0 <= ans.vmin && add(ans.vmax, i) < denominator) seen_const.push(i)
  }
  return ans !== undefined && is_eq(sorted(seen_const), range(denominator)) ? ans : undefined
}
export const canonicalize_simplex = (X: UOp): UOp | undefined => {
  // (X := a0*x0 + a1*x1 + ...) > 0 is equivalent to x0 + x1 + ... > 0 if xi >= 0 and ai > 0 for ints.
  // returns x0 + x1 + ... in such case, or None if not
  let [changed, ret] = [false, [] as UOp[]]
  for (let u of split_uop(X, Ops.ADD)) {
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

export const is_increasing = (f: UOp): boolean => {
  // is f a monotonically increasing function regards its input
  if (GroupOp.Irreducible.includes(f.op)) return true
  if (f.op === Ops.ADD) return is_increasing(f.src[0]) && is_increasing(f.src[1])
  if ([Ops.MUL, Ops.IDIV].includes(f.op) && f.src[1].op === Ops.CONST && f.src[1].arg >= 0) return is_increasing(f.src[0])
  return false // False if not sure
}

export const parse_valid = (valid: UOp): [UOp, boolean, number] => {
  // if it's X <= c, returns X, True, c
  // if it's X >= c, returns X, False, c

  // (X < c).ne(True) -> X >= c
  const s0 = valid.src[0]
  if (valid.op === Ops.CMPNE && valid.src[1].op === Ops.CONST && valid.src[1].arg === 1 && s0.op === Ops.CMPLT && s0.src[1].op === Ops.CONST) return [s0.src[0], false, s0.src[1].arg]
  // X < c -> X <= c-1
  if (valid.op === Ops.CMPLT && valid.src[1].op === Ops.CONST && dtypes.is_int(valid.src[0].dtype)) return [valid.src[0], true, valid.src[1].arg - 1]
  throw new Error(`not able to parse ${valid}`)
}

export const uop_given_valid = (valid: UOp, uop: UOp): UOp | undefined => {
  // return None if valid is always False, otherwise the simplified uop (might be the same as input)

  // first, parse valid into {expr: (lower_bound, upper_bound)}
  // KAREL: should be DefaultDict but for some reason I get segment fault with it
  const bounds = new Map<UOp, ConstType[]>()
  for (const stmt of split_uop(valid, Ops.AND)) {
    try {
      const [expr, isUpper, c] = parse_valid(stmt)
      bounds.set(expr, bounds.get(expr)!.map((o, i) => i === Number(isUpper) ? c : o))
    } catch {
      return uop
    } // give up if we cannot parse the valid
  }
  // simplify uop given that valid is True
  for (const [expr, v] of bounds.entries()) {
    // some expr has lower bound > upper bound -> valid is an empty set and we return None
    if (v[0] !== undefined && v[1] !== undefined && v[0] > v[1]) return undefined

    // every candidate is a set of contrained UOp based on valid, and if every item in a set simplifies the uop into a same output, we rewrite uop
    const candidates: [UOp, UOp][][] = []
    if (expr.op === Ops.ADD && v[0] === 1 && split_uop(expr, Ops.ADD).every((u) => GroupOp.Irreducible.includes(u.op))) {
      // if the constraint is a simplex: X0 + X1 + ... > 0, we can check if all Xi > 0 simplify into the same output
      candidates.push(split_uop(expr, Ops.ADD).toArray().map((Xi) => [Xi, UOp.variable('fake', 1, Xi.vmax, Xi.dtype)]))
    }
    // try checking the whole clause
    if (uop.toposort.has(expr)) {
      candidates.push([[expr, UOp.variable('fake', v[0] === undefined ? expr.vmin : v[0], v[1] === undefined ? expr.vmax : v[1], expr.dtype)]])
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

const _valid_priority = (v: UOp, valids: UOp[]): number => {
  // we want valid that's in other valids' parents to be first, so it's more likely the other valids get simplified
  try {
    return valids.map((other) => other.toposort.has(parse_valid(v)[0]) ? -1 : 0 as number).reduce((p, c) => p + c)
  } catch {
    return 0
  }
}
export const simplify_valid = (valid: UOp): UOp | undefined => {
  const ret: UOp[] = []
  let somethingChanged = false
  const valids = split_uop(valid, Ops.AND).toArray()
  for (const stmt of valids.sort((a, b) => _valid_priority(b, valids) - _valid_priority(a, valids))) {
    ret.push(ret.length ? (uop_given_valid(ret.reduce((p, c) => p.bitwise_and(c)), stmt) || stmt) : stmt)
    if (ret.at(-1) !== stmt) somethingChanged = true
  }
  return somethingChanged ? ret.reduce((p, c) => p.bitwise_and(c)) : undefined
}
// export const max_var_const = ({ x, c1, c2 }: { x: UOp; c1: UOp; c2: UOp }) => {
//   if (x.vmin >= (0)) return c1.arg >= c2.arg ? x.mul(c1) : x.mul(c2)
//   if (x.vmax <= (0)) return c1.arg >= c2.arg ? x.mul(c2) : x.mul(c1)
// }
export const sint_to_uop = (x: sint, dtype = dtypes.int) => isConst(x) ? UOp.const(dtype, x) : x

export const symbolic_simple = new PatternMatcher([
  //   // ** this folding **
  UPat.var('x').add(0).fn(({ x }) => x), // x+0 -> x
  UPat.var('x').mul(1).fn(({ x }) => x), // x*1 -> x
  UPat.var('x').idiv(UPat.var('x')).fn(({ x }) => x.const_like(1)), // x//x -> 1
  UPat.var('x').idiv(1).fn(({ x }) => x), // x//1 -> x
  UPat.var('x').idiv(-1).fn(({ x }) => x.neg()), // x//-1 -> -x
  UPat.var('x').div(UPat.var('x')).fn(({ x }) => x.const_like(1)), // x/x -> 1
  (UPat.var('x').mul(UPat.var('x2'))).div(UPat.var('x2')).fn(({ x, x2: _x2 }) => x), // (x*x2)/x2 -> x
  (UPat.var().mod(UPat.var('y'))).named('base').mod(UPat.var('y')).fn(({ base, y: _y }) => base), // (x%y)%y = -> x%y (rewritten with base for speed)
  UPat.var('x').mod(UPat.cvar('c')).add(UPat.var('x').idiv(UPat.cvar('c')).mul(UPat.cvar('c'))).fn(({ x, c: _c }) => x), // (x%c)+(x//c)*c = x
  (UPat.var('x').idiv(UPat.cvar('c1'))).mul(UPat.cvar('c3')).add(UPat.var('x').mod(UPat.cvar('c1')).mul(UPat.cvar('c2'))).fn(({ x, c1, c2, c3 }) => c1.arg * c2.arg === c3.arg ? x.mul(c2) : undefined), // (x%c1)*c2+(x//c1)*c3 = x*c2 if c1*c2==c3
  UPat.var('x', dtypes.bool).bitwise_and(UPat.cvar('c', undefined, false)).fn(({ x, c }) => c.arg ? x : c),
  UPat.var('x', dtypes.bool).bitwise_or(UPat.cvar('c', undefined, false)).fn(({ x, c }) => c.arg ? c : x),
  new UPat(GroupOp.Idempotent, undefined, [UPat.var('x'), UPat.var('x')]).fn(({ x }) => x),
  UPat.var('x', dtypes.bool).logical_not().logical_not().fn(({ x }) => x),
  UPat.var('x', dtypes.bool).where(UPat.const(dtypes.bool, true), UPat.const(dtypes.bool, false)).fn(({ x }) => x),
  //   // ** zero folding **
  UPat.var('x').lt(UPat.var('x')).fn(({ x }) => x.const_like(false).cast(dtypes.bool.vec(x.dtype.count))), // x < x -> False
  UPat.var('x', dtypes.ints).ne(UPat.var('x', dtypes.ints)).fn(({ x }) => UOp.const(dtypes.bool.vec(x.dtype.count), false)), // x != x -> False (only ints)
  //   // x*0 -> 0 or 0*x -> 0
  //   // if x is nan or inf it should render the nan value.
  //   // NOTE: this can be wrong for loaded NaN
  UPat.var('x').mul(0).fn(({ x }) => x.const_like(typeof x.arg === 'number' && (isNaN(x.arg) || isInf(x.arg)) ? NaN : 0)),
  //   // ** constant folding **
  // TODO: add const folding for Ops.THREEFRY
  new UPat(GroupOp.ALU, undefined, new UPat([Ops.VCONST, Ops.CONST])).named('a').fn(({ a }) => a.op !== Ops.THREEFRY ? a.const_like(exec_alu(a.op, a.dtype, a.src.map((x) => x.arg), false)) : undefined),
  //   // bool MUL is AND, ADD/MAX is OR. prevents other rules to rewrite bool ADD/MUL incorrectly
  UPat.var('x', dtypes.bool).mul(UPat.var('y', dtypes.bool)).fn(({ x, y }) => x.bitwise_and(y)),
  UPat.var('x', dtypes.bool).add(UPat.var('y', dtypes.bool)).fn(({ x, y }) => x.bitwise_or(y)),
  UPat.var('x', dtypes.bool).maximum(UPat.var('y', dtypes.bool)).fn(({ x, y }) => x.bitwise_or(y)),
  // *** cast ***
  new UPat(Ops.CAST, undefined, UPat.cvar('c'), undefined, 'root').fn(({ root, c }) => root.const_like(c.arg)),
  new UPat(Ops.CAST).named('root').fn(({ root }) => root.dtype === root.src![0].dtype ? root.src![0] : undefined), //24
])

export const symbolic = symbolic_simple.add(
  new PatternMatcher([
    // ** COMMUTATIVE flipping **
    new UPat(GroupOp.Commutative).named('x').fn(({ x }) => is_less_than(x.src[1].tuplize, x.src[0].tuplize) ? x.replace({ src: x.src.toReversed() }) : undefined),
    //   // ** boolean algebra **
    UPat.var('x').bitwise_or(UPat.var('x').bitwise_and(UPat.var())).fn(({ x }) => x), // x|(x&y) -> x
    //   // ** combine terms **
    UPat.var('x').mul(UPat.cvar('c0')).add(UPat.var('x').mul(UPat.cvar('c1'))).fn(({ x, c0, c1 }) => x.mul(c0.add(c1))), // (x*c0)+(x*c1) -> x*(c0+c1)
    (UPat.var('y').add(UPat.var('x').mul(UPat.cvar('c0')))).add(UPat.var('x').mul(UPat.cvar('c1'))).fn(({ x, y, c0, c1 }) => y.add(x.mul(c0.add(c1)))),
    UPat.var('x').add(UPat.var('x').mul(UPat.cvar('c'))).fn(({ x, c }) => x.mul(c.add(1))), // (x+x*c)-> x*(c+1)
    (UPat.var('y').add(UPat.var('x'))).add(UPat.var('x').mul(UPat.cvar('c'))).fn(({ x, y, c }) => y.add(x.mul(c.add(1)))),
    UPat.var('x').add(UPat.var('x')).fn(({ x }) => x.mul(2)), // (x+x)-> x*2
    (UPat.var('y').add(UPat.var('x'))).add(UPat.var('x')).fn(({ y, x }) => y.add(x.mul(2))),
    (UPat.var('x').div(UPat.var('x2'))).div(UPat.var('x3')).fn(({ x, x2, x3 }) => x.div(x2.mul(x3))), // (x/x2)/x3 -> x/(x2*x3)
    UPat.var('x').add(UPat.cvar('c')).mul(-1, true).fn(({ x, c }) => x.neg().add(c.neg())), // -(x+c) -> -x + -c
    // //   // a conditional with the same results either way is a noop, also fold const conditionals
    UPat.var().where(UPat.var('val'), UPat.var('val')).fn(({ val }) => val),
    UPat.cvar('gate', undefined, false).where(UPat.var('c0'), UPat.var('c1')).fn(({ gate, c0, c1 }) => gate.arg ? c0 : c1),
    // alu of two where with same conds can combine, only do if true branch or false branch is const
    [
      new UPat(GroupOp.Binary, undefined, [UPat.var('c').where(UPat.var('t'), UPat.var('f')), UPat.var('c').where(UPat.var('tt'), UPat.var('ff'))], undefined, 'alu'),
      ({ alu, c, t, tt, f, ff }) => (t.op === tt.op && tt.op === Ops.CONST) || (f.op === ff.op && ff.op === Ops.CONST) ? c.where(t.alu(alu.op, tt), f.alu(alu.op, ff)) : undefined,
    ],
    // ALU min==max -> CONST (slow!)
    new UPat(GroupOp.ALU).named('x').fn(({ x }) => x.vmin === x.vmax ? x.const_like(x.vmin) : undefined),
    // max folding
    UPat.var('x').maximum(UPat.var('y')).fn(({ x, y }) => x.vmax <= y.vmin ? x.vmin >= y.vmax ? x : y : undefined),
    // TODO: why does this rule break beautiful_mnist?
    //((UPat.var("x")+UPat.var("z")).maximum(UPat.var("y")+UPat.var("z")), lambda x,y,z: x.maximum(y) + z),
    // UPat.var('x').mul(UPat.cvar('c1')).maximum(UPat.var('x').mul(UPat.cvar('c2'))).fn(({ x, c1, c2 }) => max_var_const({ x, c1, c2 })),
    //   // ** two stage ALU folding **
    ...GroupOp.Associative.map((op) => [UPat.var('x').alu(op, UPat.cvar('c1')).alu(op, UPat.cvar('c2')).named('f'), ({ f, x, c1, c2 }) => x.alu(f.op, c1.alu(f.op, c2))] as [UPat, (p: Record<string, UOp>) => UOp]),
    (UPat.cvar('c0').add(UPat.var('x'))).lt(UPat.cvar('c1')).fn(({ x, c0, c1 }) => x.lt(c1.sub(c0))), // c0 + x < c1 -> x < c1 - c0
    (UPat.var('x').idiv(UPat.cvar('c1'))).idiv(UPat.cvar('c2')).fn(({ x, c1, c2 }) => x.idiv(c1.mul(c2))), // (x//c1)//c2 -> x//(c1*c2)
    // //   // ** lt **
    // //   // c0*x<c1 for positive int c0,c1
    (UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)).fn(({ x, c0, c1 }) => c0.arg > 0 && c1.arg > 0 ? x.lt(Math.ceil(c1.arg / c0.arg)) : undefined),
    //   // c0*x<c1 for negative int c0 and non-positive c1
    (UPat.cvar('c0', undefined, false).mul(UPat.var('x', dtypes.ints))).lt(UPat.cvar('c1', undefined, false)).fn(({ x, c0, c1 }) => c0.arg < 0 && c0.arg !== -1 && c1.arg <= 0 ? x.neg().lt(-Math.floor(-c1.arg / -c0.arg)) : undefined),
    //   // x//c0<c1 for positive int c0
    (UPat.var('x', dtypes.ints).idiv(UPat.cvar('c0', undefined, false))).lt(UPat.cvar('c1', undefined, false)).fn(({ x, c0, c1 }) => c0.arg > 0 ? x.lt(c1.arg * c0.arg) : undefined),
    //   // ** move add/mul consts to end (NOTE: this is still happening before constant folding) **
    new UPat(Ops.ADD, undefined, [UPat.var('x'), UPat.cvar('c1')]).add(UPat.var('y')).fn(({ x, c1, y }) => (x.add(y)).add(c1)),
    new UPat(Ops.MUL, undefined, [UPat.var('x'), UPat.cvar('c1')]).mul(UPat.var('y')).fn(({ x, c1, y }) => (x.mul(y)).mul(c1)),
    //   // *** rules from symbolic ***
    //   // unrolled arange div folding
    new UPat(Ops.ADD, undefined, [[new UPat(), new UPat(Ops.IDIV)]]).named('divs').fn(({ divs }) => fold_unrolled_divs(divs)),
    // generic lt folding
    UPat.var('x', dtypes.sints).lt(UPat.cvar('c', undefined, false)).fn(({ x, c }) => 0 < c.arg ? lt_folding(x, c.arg) : undefined),
    // canonicalize a simplex with positive coefficients > 0
    // not x < 1 -> X > 0
    [UPat.var('x', dtypes.ints).lt(1).ne(true), ({ x }) => {
      const newx = canonicalize_simplex(x)
      return newx !== undefined ? newx.lt(1).ne(true) : undefined
    }],
    // div folding
    UPat.var('x').idiv(UPat.cvar('c')).add(UPat.cvar('a')).idiv(UPat.cvar('d')).fn(({ x, c, a, d }) => (x.add(a.mul(c))).idiv(c.mul(d))), // (x//c+a)//d -> (x+a*c)//(c*d)
    UPat.var('x', dtypes.sints).idiv(UPat.var('y')).fn(({ x, y }) => div_and_mod_folding(x, y, Ops.IDIV)),
    // ** mod **
    // mod folding
    UPat.var('x').mod(UPat.var('y')).fn(({ x, y }) => div_and_mod_folding(x, y, Ops.MOD)),
  ]),
)

export const symbolic_flat = symbolic.add(
  new PatternMatcher<unknown, UOp | undefined>([
    // ** combine terms (opinionated) **
    UPat.var('x').add(UPat.var('y')).mul(-1, true).fn(({ x, y }) => x.neg().add(y.neg())), // -(x+y) -> -x + -y
    // (x+y)*c -> x*c+y*c. only for int, float has inf*0=nan issue
    UPat.var('x', dtypes.ints).add(UPat.var('y')).mul(UPat.cvar('c')).fn(({ x, y, c }) => x.mul(c).add(y.mul(c))),
  ]),
)

export const _substitute = new PatternMatcher<Map<UOp, UOp>>([[new UPat(Ops.values()).named('x'), ({ ctx, x }) => ctx.get(x)]])

// for debug
const syms = new Map([[Ops.ADD, '+'], [Ops.SUB, '-'], [Ops.IDIV, '//'], [Ops.MOD, '%'], [Ops.SHL, '<<'], [Ops.SHR, '>>'], [Ops.MUL, '*'], [Ops.CMPLT, '<'], [Ops.CMPNE, '!='], [Ops.AND, '&'], [Ops.OR, '|'], [Ops.XOR, '^']])

export const renderer = new PatternMatcher([
  new UPat([Ops.DEFINE_VAR, Ops.SPECIAL]).named('x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, x.arg[0])),
  new UPat(Ops.RANGE).named('x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, `ridx(${x.arg},)`)),
  new UPat(Ops.CONST).named('x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, dtypes.is_float(x.dtype) ? floatString(x.arg) : x.arg.toString())),
  new UPat(Ops.BIND, undefined, new UPat(Ops.NOOP), undefined, 'x').fn(({ x }) => x.src[0]),
  new UPat(Ops.NEG, undefined, new UPat(Ops.NOOP), undefined, 'x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(-${x.src[0].arg})`)),
  new UPat(Ops.MAX, undefined, new UPat(Ops.NOOP), undefined, 'x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, `max(${x.src[0].arg}, ${x.src[1].arg})`)),
  new UPat(Ops.MULACC, undefined, new UPat(Ops.NOOP), undefined, 'x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(${x.src[0].arg}*${x.src[1].arg}+${x.src[2].arg})`)),
  new UPat(Ops.WHERE, undefined, new UPat(Ops.NOOP), undefined, 'x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(${x.src[0].arg} ? ${x.src[1].arg} : ${x.src[2].arg})`)),
  new UPat(GroupOp.ALU, undefined, new UPat(Ops.NOOP), undefined, 'x').fn(({ x }) => new UOp(Ops.NOOP, undefined, undefined, `(${x.src[0].arg}${syms.get(x.op)}${x.src[1].arg})`)),
])

// *** what was symbolic.py ***
export type sint = number | UOp

// *** UOp merge views and swizzling ***

export const merge_views = new PatternMatcher([
  // VIEW(VIEW) merges to a single VIEW
  new UPat(Ops.VIEW, undefined, [new UPat(Ops.VIEW).named('vm2')]).named('vm1').fn(({ vm1, vm2 }) => vm2.replace({ arg: vm2.st!.add(vm1.st!) })),
  new UPat(Ops.VIEW, undefined, [UPat.var('x')]).named('vm').fn(({ vm, x }) => vm.st!.contiguous && x.st !== undefined && is_eq(x.shape, vm.shape) ? x : undefined),
])

// push VIEW to parents
export const view_left = merge_views.add(
  new PatternMatcher([
    // VIEW before elementwise/buffer ops
    [
      new UPat(Ops.VIEW, undefined, [new UPat([Ops.CAST, Ops.BITCAST, ...GroupOp.ALU, Ops.ASSIGN]).named('e')]).named('vm'),
      ({ e, vm }) => e.replace({ src: e.src.map((s) => s.st === undefined ? s : s === s.base ? s.view(vm.st!) : s.base.view(s.st!.add(vm.st!))) }),
    ],
    [
      new UPat(Ops.VIEW, undefined, [new UPat(GroupOp.Buffer).named('b')]).named('vm'),
      ({ b, vm }) => b.replace({ src: b.src.map((s) => s.op === Ops.VIEW ? s.st!.add(vm.st!).to_uop() : s) }),
    ],
  ]),
)

// ============================================================
//                    transcendental.ts
// ============================================================

export const TRANSCENDENTAL_SUPPORTED_DTYPES = [dtypes.float16, dtypes.float32, dtypes.float64]

/**replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio*/
export const _lazy_map_numbers = (x: UOp, inf: UOp, _inf: UOp, nan: UOp, ratio: UOp) => x.ne(Infinity).where(x.ne(x).where(nan, x.ne(-Infinity).where(ratio, _inf)), inf)

// *** helper functions for bit manipulation ***
export const mantissa_bits = (d: DType): number => dtypes.finfo(d)[1]
export const exponent_bias = (d: DType): number => new Map([[dtypes.float64, 1023], [dtypes.float32, 127], [dtypes.float16, 15]] as const).get(d)!
export const exponent_mask = (d: DType): number => new Map([[dtypes.float64, 2047], [dtypes.float32, 255], [dtypes.float16, 31]] as const).get(d)!

// **** utils ****
export const shr = (x: UOp, y: number): UOp => x.idiv(2 ** y)
export const shl = (x: UOp, y: number): UOp => x.mul(2 ** y)

/**round d:float to int away from 0*/
export const rintk = (d: UOp): UOp => {
  const out_dtype = new Map([[dtypes.float64, dtypes.int64], [dtypes.float32, dtypes.int32], [dtypes.float16, dtypes.int16]]).get(d.dtype)!
  return (d.add(d.lt(0.0).where(d.const_like(-0.5), d.const_like(0.5)))).cast(out_dtype)
}

/**cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]*/
export const pow2if = (q: UOp, float_dtype: DType) => {
  const out_dtype = new Map([[dtypes.int64, dtypes.float64], [dtypes.int32, dtypes.float32], [dtypes.int16, float_dtype]]).get(q.dtype)!
  return shl(q.add(exponent_bias(out_dtype)), mantissa_bits(out_dtype)).bitcast(out_dtype)
}
/**calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf).*/
export const ilogb2k = (d: UOp): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  const dint = d.bitcast(new Map([[dtypes.float64, dtypes.int64], [dtypes.float32, dtypes.int32], [dtypes.float16, dtypes.int16]]).get(d.dtype)!)
  // -1 <= ilog2bk(d) <= 128
  return (shr(dint, mantissa_bits(d.dtype)).bitwise_and(exponent_mask(d.dtype))).sub(exponent_bias(d.dtype))
}
/**d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number.*/
export const ldexp3k = (d: UOp, e: UOp): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype) && TRANSCENDENTAL_SUPPORTED_DTYPES.includes(e.dtype))
  const cast_map = new Map([[dtypes.float64, dtypes.int64], [dtypes.float32, dtypes.int32], [dtypes.float16, dtypes.int16]])
  const m1 = d.bitcast(cast_map.get(d.dtype)!)
  const m2 = shl(e.cast(cast_map.get(d.dtype)!), mantissa_bits(d.dtype))
  return (m1.add(m2)).bitcast(d.dtype).cast(d.dtype)
}
/**d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal.*/
export const ldexp2k = (d: UOp, e: UOp): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype) && [dtypes.int16, dtypes.int32, dtypes.int64].includes(e.dtype))
  return (d.mul(pow2if(shr(e, 1), d.dtype))).mul(pow2if(e.sub(shr(e, 1)), d.dtype))
}
/** frexp(v) -> (mantissa, exponent) assuming v != 0 */
export const frexp = (v: UOp): [UOp, UOp] => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(v.dtype))
  // m1 = masks for mantissa, m2 = masks to normalize the mantissa.
  const m1 = new Map([[dtypes.float64, 0x000FFFFFFFFFFFFF], [dtypes.float32, 0x807FFFFF], [dtypes.float16, 0x83FF]]).get(v.dtype)!
  const m2 = new Map([[dtypes.float64, 0x3FE0000000000000], [dtypes.float32, 0x3F000000], [dtypes.float16, 0x3800]]).get(v.dtype)!
  const bits = v.bitcast(new Map([[dtypes.float64, dtypes.uint64], [dtypes.float32, dtypes.uint32], [dtypes.float16, dtypes.uint16]]).get(v.dtype)!)
  const exponent = shr(bits, mantissa_bits(v.dtype)).bitwise_and(exponent_mask(v.dtype))
  // Set the exponent bits appropriately to normalize the mantissa into the range of [0.5, 1.0).
  const mantissa = ((bits.bitwise_and(m1)).bitwise_or(m2)).bitcast(v.dtype)
  const exp = exponent.sub(exponent_bias(v.dtype)).add(1)
  return [mantissa, exp]
}

// *** reduction algorithms for sine ***
/**
 * Performs Payne-Hanek Reduction: computes the remainder of `d` modulo pi/2 for the values `d` where 39800.0 <= d <= +Inf
 * Returns a tuple of `(r, q)`:
 * - `r`[d.dtype] is the reminder value corresponding to `round_to_nearest(x % pi/2)`.
 * - `q`[int32] is an integer, and q % 4 is corresponding to the quadrant of the original angle `d`.
 */
export const payne_hanek_reduction = (d: UOp): [UOp, UOp] => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  // https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
  // 190 bits of 2/pi for Payne-Hanek style argument reduction
  const two_over_pi_f = [0x00000000, 0x28be60db, 0x9391054a, 0x7f09d5f4, 0x7d4d3770, 0x36d8a566, 0x4f10e410]

  const intermediate_dtype = d.dtype === dtypes.float16 ? dtypes.float32 : d.dtype

  let [f, e] = frexp(d)
  const ia = (f.cast(intermediate_dtype).mul(4.294967296e9)).cast(dtypes.uint64)
  // extract 96 relevant bits of 2/pi based on magnitude of argument
  const i = shr(e.cast(dtypes.uint64), 5)
  e = e.cast(dtypes.int32).bitwise_and(31)
  const offset = e.sub(32, true)

  /** an = two_over_pi_f[i+offset] */
  const _take = (an: UOp, offset: number, count = 0): UOp => {
    if (count + offset < two_over_pi_f.length - 1) {
      an = i.ne(count).where(_take(an, offset, count + 1), an.const_like(two_over_pi_f[count + offset]))
    }
    return an
  }
  const _shl_lazy = (x: UOp, y: UOp) => (x.cast(dtypes.uint64).mul(pow2if(y, d.dtype).cast(dtypes.uint64))).cast(dtypes.uint32)
  const _shr_lazy = (x: UOp, y: UOp) => (x.cast(dtypes.uint64).idiv(pow2if(y, d.dtype).cast(dtypes.uint64))).cast(dtypes.uint32)

  const a = range(4).map((i) => _take(UOp.const(dtypes.uint32, 0), i))
  //  (two_over_pi_f[Int(i) + n] << e) | (two_over_pi_f[Int(i) + n+1] >> (nbits - e))
  // Note: e >= 1 for all numbers d >= 1.0. assume e != 0
  const hi = _shl_lazy(a[0], e).bitwise_or(_shr_lazy(a[1], offset))
  const mi = _shl_lazy(a[1], e).bitwise_or(_shr_lazy(a[2], offset))
  const lo = _shl_lazy(a[2], e).bitwise_or(_shr_lazy(a[3], offset))

  const _hp_mul = (x: UOp, y: UOp) => x.cast(dtypes.uint64).mul(y.cast(dtypes.uint64))
  // compute x * 2/pi
  let p = shl(_hp_mul(ia, hi), 32).add(_hp_mul(ia, mi)).add(shr(_hp_mul(ia, lo), 32))

  // round quotient to nearest
  const q = shr(p, 62).cast(dtypes.int32)
  p = p.bitwise_and(0x3fffffffffffffffn)
  const r = (p.cast(intermediate_dtype).mul(3.4061215800865545e-19)).cast(d.dtype)

  // if fraction >= 0.5, r -= pi/2, q += 1
  return [(f.lt(0.5)).where(r, r.sub(Math.PI / 2)), (f.lt(0.5)).where(q, q.add(1))]
}

/**
 * Performs Cody-Waite Reduction: computes the reminder of `d` modulo pi/2 for the values `d` where 0 <= abs(d) <= 39800.0
 * Returns a tuple of `(r, q)`, where the output format is the same as that of `payne_hanek_reduction`.
 */
export const cody_waite_reduction = (d: UOp): [UOp, UOp] => {
  const m_1_pi = 0.318309886183790671537767526745028724
  const qdh = (d.mul(m_1_pi / 2.0 ** 24)).cast(dtypes.int64).cast(d.dtype).mul(2.0 ** 24)
  const _reduce_d = (x: UOp, q: UOp) => {
    // https://github.com/shibatch/sleef/blob/4e08851f59fc2b545f9c393c6a23dfd311a26308/src/libm/sleefdp.c#L789-L823
    if (x.dtype === dtypes.float64) {
      // https://github.com/shibatch/sleef/blob/f6d8a841fbfddd26ce712834d4da220cd76048fb/src/common/misc.h#L77
      const [PI_A, PI_B, PI_C, PI_D] = [3.1415926218032836914, 3.1786509424591713469e-08, 1.2246467864107188502e-16, 1.2736634327021899816e-24]
      d = qdh.sub(PI_A).add(x)
      d = q.mul(-PI_A).add(d)
      d = qdh.mul(-PI_B).add(d)
      d = q.mul(-PI_B).add(d)
      d = qdh.mul(-PI_C).add(d)
      d = q.mul(-PI_C).add(d)
      d = (qdh.add(q)).mul(-PI_D).add(d)
    } else if (x.dtype === dtypes.float16) {
      // [FIXME] when reducing `d`, FP16 needs FP32 precision to achieve 1.0 ULP precision.
      d = _reduce_d(x.cast(dtypes.float32), q.cast(dtypes.float32)).cast(dtypes.float16)
    } else {
      // https://github.com/shibatch/sleef/blob/4e08851f59fc2b545f9c393c6a23dfd311a26308/src/libm/sleefsp.c#L464-L503
      d = q.mul(-3.1414794921875).add(x)
      d = q.mul(-0.00011315941810607910156).add(d)
      d = q.mul(-1.9841872589410058936e-09).add(d)
      d = q.mul(-1.2154201256553420762e-10).add(d)
    }
    return d
  }
  const quadrant = d.dtype === dtypes.float64 ? rintk(d.mul(m_1_pi).sub(qdh)) : rintk(d.mul(m_1_pi))
  return [_reduce_d(d, quadrant.cast(d.dtype)), quadrant.cast(dtypes.int32)]
}
// *** approximate sine on small angle. ***
export const trig_poly = (d: UOp, coeff32: number[], coeff64: number[]) => d.mul(d.dtype === dtypes.float64 ? polyN(d.mul(d), coeff64) : polyN(d.mul(d), coeff32))
// approximate sine on [-pi/2, pi/2]
// deno-fmt-ignore
export const sin_poly = (d: UOp): UOp => {
  return trig_poly(d,
    [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0],
    [-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10, -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815, -0.166666666666666657414808, 1.0],
  )
}

const _ifand = (q: UOp, n: number) => (q.bitwise_and(n)).ne(0)

export const sin_poly_small = (d: UOp, q: UOp): UOp => {
  const r = sin_poly(d)
  return r.mul(_ifand(q, 1).where(r.const_like(-1), r.const_like(1)))
}

export const sin_poly_large = (d: UOp, q: UOp): UOp => {
  const r = sin_poly(d.add(_ifand(q, 1).where(d.const_like(Math.PI / 2), d.const_like(0))))
  return r.mul(_ifand(q, 2).where(r.const_like(-1), r.const_like(1)))
}

// *** toplevel functions for xsin/xlog2/xexp2 ***
/**
 * Implements a 1.0 ULP approximation for Ops.SIN.
 * - fast=True assumes x <= switch_over.
 * - switch_over is the threshold for switching to payne_hanek_reduction.
 */
export const xsin = ({ d, fast = false, switch_over = 30.0 }: { d: UOp; fast?: boolean; switch_over?: number }) => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  //  mask +-inf/nan as zero
  const x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  //  x_sign = sign(x)
  const x_sign = x.ne(0).where((x.lt(0)).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  const x_abs = x.mul(x_sign)
  const [r, q] = (fast ? cody_waite_reduction : payne_hanek_reduction)(x_abs)
  let result
  if (fast) result = sin_poly_small(r, q)
  else {
    // Payne Hanek Reduction assumes abs(x) >= pi/4, so for smaller values, use cody_waite_reduction.
    const [r_small, q_small] = cody_waite_reduction(x_abs)
    result = (x_abs.lt(switch_over)).where(sin_poly_small(r_small, q_small), sin_poly_large(r, q))
  }
  // adjusts the sign for abs(x)
  result = result.mul(x_sign)
  // sin(Inf) = NaN, sin(-Inf) = NaN, sin(NaN) = NaN
  return _lazy_map_numbers(d, d.const_like(NaN), d.const_like(NaN), d.const_like(NaN), result)
}

/**
 * Implements a 1.0 ULP approximation for Ops.EXP2
 * Paper: https://arxiv.org/pdf/2001.09258
 */
export const xexp2 = ({ d }: { d: UOp }): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  // mask +=inf/nan as zero.
  const x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  const q = rintk(x)
  // s = d - round(d)
  const s = x.sub(q.cast(x.dtype))
  // a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  let u
  if (d.dtype === dtypes.float64) {
    // deno-fmt-ignore
    u = polyN(s, [0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4,
                    0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0,
                    0.6931471805599452862e+0, 0.1000000000000000000e+1])
  } else u = polyN(s, [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0])
  u = ldexp2k(u, q) // u*2^q
  const [upper, lower] = new Map([[dtypes.float64, [1024, -2000]], [dtypes.float32, [128, -150]], [dtypes.float16, [23, -22]]]).get(d.dtype)!
  // Replace x >= upper with +inf
  u = (d.ge(upper)).where(d.const_like(Infinity), u)
  // Replace x < lower with zero.
  u = (d.lt(lower)).where(d.const_like(0.0), u)
  // exp2(NaN) = NaN
  return d.ne(d).where(d.const_like(NaN), u)
}

/**
 * Implements a 1.0 ULP approximation for Ops.LOG2
 * Paper: https://arxiv.org/pdf/2001.09258 5.5
 */
export const xlog2 = ({ d }: { d: UOp }): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  // TODO: float16 denormal need float32 to achieve precision
  if (d.dtype === dtypes.float16) return xlog2({ d: d.cast(dtypes.float32) }).cast(dtypes.float16)
  const FLT_MIN = d.const_like(d.dtype === dtypes.float16 ? 1e-6 : 1e-4)
  const is_denormal = d.lt(FLT_MIN)
  const a = is_denormal.where(d.mul(2 ** 64), d)

  let e = ilogb2k(a.mul(1.0 / 0.75)).cast(a.dtype)
  const m = ldexp3k(a, e.neg())
  e = is_denormal.where(e.sub(64), e)

  const x = (m.sub(1.0)).div(m.add(1.0))
  const x2 = x.mul(x)
  let t, s_hi, s_lo
  if (d.dtype === dtypes.float64) {
    // deno-fmt-ignore
    t = polyN(x2, [0.2211941750456081490e+0, 0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0,
                       0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449]);
    ;[s_hi, s_lo] = [e.add(x.mul(2.885390081777926774)), e.const_like(0)]
  } else {
    t = polyN(x2, [0.4374550283e+0, 0.5764790177e+0, 0.9618012905120])
    ;[s_hi, s_lo] = [e.add(x.mul(2.8853900432586669922)), x.mul(3.2734474483568488616e-08)]
  }
  let r = t.mul(x.mul(x2)).add(s_hi.add(s_lo))

  // log2(Inf) = Inf
  r = d.ne(Infinity).where(r, r.const_like(Infinity))
  // log2(x) = NaN for x < 0
  r = (d.lt(-0.0)).where(r.const_like(NaN), r)
  // log2(0) = -Inf, but we will compare using the value of y because 1e-200==0 is true.
  // log2_zero = the value of unmasked xlog2(0.0).
  const log2_zero = new Map([[dtypes.float64, -1087], [dtypes.float32, -191], [dtypes.float16, -79]]).get(d.dtype)!
  r = r.ne(log2_zero).where(r, r.const_like(-Infinity))
  // log2(NaN) = NaN
  r = d.ne(d).where(r.const_like(NaN), r)
  // log2(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
  return d.reciprocal().ne(-Infinity).where(r, r.const_like(-Infinity))
}

// ============================================================
//                      rewriter.ts
// ============================================================

// ***** float4/image store handling *****
export const fold_expanded = (ex: UOp, buf: UOp) => {
  if (buf.dtype.base !== dtypes.float && buf.dtype.base !== dtypes.half && !isinstance(buf.dtype, ImageDType)) return undefined
  let new_srcs: (UOp | undefined)[] = dedup([...ex.src])
  const old_new_srcs = [...new_srcs]
  const [is_load, is_image] = [new_srcs[0]?.op === Ops.LOAD, isinstance(buf.dtype, ImageDType)]

  // first, extract all the relevant offsets
  const offsets_rootsrc = new DefaultMap<UOp, Map<string, number>>(undefined, () => new Map())
  for (const [i, s] of new_srcs.entries()) {
    const idx = s!.src[0].src[1]
    let root_src: any, arg
    if (s!.dtype.count !== 1 || (is_image && idx.dtype.count === 2)) continue
    if (idx.op === Ops.ADD && idx.src[1].op === Ops.CONST) [root_src, arg] = [idx.src[0], idx.src[1].arg]
    else if (idx.op === Ops.CONST) [root_src, arg] = ['CONST', idx.arg]
    else [root_src, arg] = [idx, 0]
    // add gates for gated
    if (s!.src[0].src.length === 3) root_src = [s!.src[0].src[2], root_src]
    if (set_default(offsets_rootsrc, root_src, new Map()).has(arg)) throw new Error(`${offsets_rootsrc.get(root_src)!.get(arg)} != ${i} with ${s?.src.length} sources`)
    offsets_rootsrc.get(root_src)!.set(arg, i)
  }
  // then rewrite everything we can
  const lengths = is_image ? [4] : (buf.dtype.base === dtypes.half && get_env('ALLOW_HALF8') ? [8, 4, 2] : (AMX ? [16, 8, 4, 2] : [4, 2]))
  let used: [UOp, any][] = []
  for (const [rootsrc, offsets] of offsets_rootsrc.entries()) {
    for (const o of offsets.keys()) {
      for (const fold_length of lengths) {
        if (range(fold_length).every((i) => !used.some(([a, b]) => a === rootsrc && b === (o + i)) && offsets.has(o + i))) {
          const load_1 = new_srcs[offsets.get(o)!]!
          const new_src = [...load_1.src]
          const oidx = new_src[0].src[1]
          if (oidx.divides(fold_length) === undefined) continue
          if (is_image) {
            // for images, we rewrite the index. it must evenly divide 4 from the above check
            new_src[0] = buf.index(
              new UOp(Ops.VECTORIZE, dtypes.int.vec(2), [oidx.idiv(4).mod((buf.dtype as ImageDType).shape[1]), oidx.idiv(4 * (buf.dtype as ImageDType).shape[1])]),
              isinstance(rootsrc, Array) ? rootsrc[0] as UOp : undefined,
            )
          } else {
            // for non image, we upcast the index pointer
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr(idiv((new_src[0].dtype as PtrDType).size, fold_length), (new_src[0].dtype as PtrDType).local))
          }
          // generate the folded new_srcs
          if (is_load) {
            const new_load = new UOp(Ops.LOAD, load_1!.dtype.vec(fold_length), new_src)
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = new_load.gep(i)
          } else { // vectorize the store
            new_src[1] = new UOp(Ops.VECTORIZE, new_src[1].dtype.vec(fold_length), range(fold_length).map((i) => new_srcs[offsets.get(o + i)!]!.src[1]))
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = i === 0 ? new UOp(Ops.STORE, dtypes.void, new_src) : undefined
          }
          used = [...used, ...range(fold_length).map((i) => [rootsrc, o + i as any] as [UOp, UOp])]
        }
      }
    }
  }
  // dedup expand for LOAD
  if (is_load && old_new_srcs.length !== ex.src.length) new_srcs = ex.src.map((s) => new_srcs[old_new_srcs.indexOf(s)])
  // remove Nones for STORE
  return used.length ? new UOp(ex.op, ex.dtype, [...new_srcs.filter((x) => x !== undefined)], ex.arg) : undefined
}

export const fix_unfoldable_image_load = (load: UOp, buf: UOp) => {
  const oidx = load.src[0].src[1]
  if (!isinstance(buf.dtype, ImageDType) || oidx.dtype.count === 2) return undefined
  const id4 = oidx.mod(4)
  const new_src = [...load.src]
  // TODO: copied logic from above
  new_src[0] = load.src[0].src[0].index(
    new UOp(Ops.VECTORIZE, dtypes.int.vec(2), [(oidx.idiv(4)).mod(buf.dtype.shape[1]), oidx.idiv(4 * buf.dtype.shape[1])]),
    load.src[0].src.length === 3 ? load.src[0].src[2] : undefined,
  )
  const vec_load = new UOp(Ops.LOAD, load.dtype.vec(4), [...new_src])
  return range(4).reduce((ret, i) => id4.ne(i).where(ret, vec_load.gep(i)), load.const_like(NaN))
}

export const buf_idx_pat = new UPat(Ops.INDEX, undefined, [UPat.var('buf')], undefined, undefined, true)
export const float4_folding = new PatternMatcher([
  new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.LOAD, undefined, [buf_idx_pat], undefined, undefined, true)).named('ex').fn(({ ex, buf }) => fold_expanded(ex, buf)),
  new UPat([Ops.BARRIER, Ops.SINK], undefined, new UPat(Ops.STORE, undefined, [buf_idx_pat], undefined, undefined, true)).named('ex').fn(({ ex, buf }) => fold_expanded(ex, buf)),
])

// ***** image load valid simplification *****

export const simplify_valid_load = (buf: UOp, start_idx: UOp, valid: UOp): undefined | UOp => {
  const idx = uop_given_valid(valid, start_idx)
  if (idx === undefined) return buf.const_like(0)
  if (!isinstance(buf.dtype, ImageDType)) return idx === start_idx ? undefined : buf.index(idx, valid)

  // wait for it to be image indexed before running simplification
  if (start_idx.dtype.count !== 2) return undefined

  // can drop valid if idx is out of bound when valid is False
  let drop_stmt = []
  for (const stmt of split_uop(valid, Ops.AND)) {
    const [X, is_upper_bound, c] = parse_valid(stmt)

    // for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if (!is_upper_bound && c === 1 && split_uop(X, Ops.ADD).every((u) => GroupOp.Irreducible.includes(u.op) && u.vmin === 0)) {
      const testidx = split_uop(X, Ops.ADD).reduce((nowidx, u) => nowidx.substitute(new Map([[u, u.const_like(0)]])), idx).simplify()
      if (testidx.gep(0).vmax < 0 || testidx.gep(1).vmax < 0) {
        drop_stmt.push(stmt)
        continue
      }
    }
    // if X <= c, check if it's out of bound when X = c+1
    // if X >= c, check if it's out of bound when X = c-1
    const test_value = is_upper_bound ? c + 1 : c - 1
    for (const [i, b] of zip(idx.src, [buf.dtype.shape[1], buf.dtype.shape[0]])) {
      if (is_increasing(i)) {
        const rw = i.substitute(new Map([[X, X.const_like(test_value)]])).simplify()
        if (rw.vmin >= b || rw.vmax < 0) {
          drop_stmt.push(stmt)
          break
        }
      }
    }
  }
  if (!drop_stmt && idx === start_idx) return undefined
  const ss = [...split_uop(valid, Ops.AND)].filter((s) => !drop_stmt.includes(s))
  const new_valid = ss.length ? ss.reduce((acc, s) => acc.add(s)) : undefined
  return buf.index(idx, new_valid)
}
// ***** optional patterns *****

const powers_of_two = new Map(range(64).map((i) => [2 ** i, i]))
export const get_late_rewrite_patterns = cache_fn((ops: Ops[], force_transcendental = false) => {
  let pat: Pattern[] = ([[Ops.EXP2, xexp2], [Ops.LOG2, xlog2], [Ops.SIN, xsin]] as const).filter(([op, f]) => !ops.includes(op) || force_transcendental)
    .map(([op, f]) => new UPat(op, TRANSCENDENTAL_SUPPORTED_DTYPES, [UPat.var('d')]).fn((x) => f(x as any)))
  // rewrite MOD to AND (which should always be supported, but not for generic in tests): x % (2**y) -> x & (2**y-1)
  if (ops.includes(Ops.AND)) {
    pat = [...pat, UPat.var('x', dtypes.ints).mod(UPat.cvar('c')).fn(({ x, c }) => powers_of_two.has(c.arg) ? x.bitwise_and(sub(c.arg, 1)) : undefined)]
  }
  // rewrite MUL/IDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if (ops.includes(Ops.SHL) && ops.includes(Ops.SHR)) {
    pat = [
      ...pat,
      UPat.var('x', dtypes.ints).mul(UPat.cvar('c')).fn(({ c, x }) => powers_of_two.has(c.arg) ? x.lshift(powers_of_two.get(c.arg)!) : undefined),
      UPat.var('x', dtypes.ints).idiv(UPat.cvar('c')).fn(({ x, c }) => powers_of_two.has(c.arg) ? x.rshift(powers_of_two.get(c.arg)!) : undefined),
    ]
  }
  if (ops.includes(Ops.NEG)) {
    pat = [...pat, UPat.var('x').mul(-1).fn(({ x }) => x.alu(Ops.NEG))]
    if (ops.includes(Ops.SUB)) pat = [...pat, UPat.var('x').add(UPat.var('y').alu(Ops.NEG)).fn(({ x, y }) => x.alu(Ops.SUB, y))]
  }
  if (ops.includes(Ops.MULACC)) {
    pat = [...pat, UPat.var('a').mul(UPat.var('b')).add(UPat.var('c')).fn(({ a, b, c }) => a.alu(Ops.MULACC, b, c))]
  }
  return new PatternMatcher(pat)
})
// ***** threefry *****

export const threefry2x32 = (x: UOp, key: UOp) => {
  // split x into two uint32, since x in a uint64
  const [x0, x1] = [(x.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((x.idiv(2 ** 32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]

  const rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  const [key0, key1] = [(key.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((key.idiv(2 ** 32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]
  const ks = [key1, key0.xor(key1).xor(0x1BD11BDA), key0]
  let xr = [x0.add(ks.at(-1)!), x1.add(ks[0])]
  for (const i of range(5)) {
    for (const r of rotations[mod(i, 2)]) {
      const x0 = xr[0].add(xr[1])
      ;[xr[0], xr[1]] = [x0, x0.xor(xr[1].mul(2 ** r).add(xr[1].idiv(2 ** (32 - r))))]
    }
    xr = [xr[0].add(ks[mod(i, 3)]), xr[1].add(ks[mod(i + 1, 3)]).add(i).add(1)]
  }
  return xr[1].cast(dtypes.uint64).mul(2n ** 32n).bitwise_or(xr[0].cast(dtypes.uint64))
}

// ***** other math rewrite ****

export const sigmoid_like = (x: UOp, y: UOp) => {
  const t = div(1, add(x, 1))
  return t.mul(sub(1, t)).mul(y)
}

// ***** main rewriter *****

export const loop_collapse = (compval: UOp, multconst: UOp, rng: UOp, acc: UOp, idx2?: UOp, idx3?: UOp, extra?: UOp, vec?: UOp, ne?: UOp, add = UOp.const(dtypes.int, 0), mul = UOp.const(dtypes.int, 1)) => {
  if (get_env('DISABLE_LOOP_COLLAPSE') || !acc.src.includes(rng)) return undefined // must be the right REDUCE
  let [loop_start, loop_end] = rng.src
  if (loop_start.arg !== 0) {
    // TODO: support and test this with other mul and loop_starts
    if (DEBUG >= 1) console.log(`WARNING, NOT FOLDING: mul:${mul.arg} loop_start:${loop_start.arg}`)
    return undefined
  }
  if (idx2 !== undefined) add = add.add(idx2)
  if (idx3 !== undefined) add = add.add(idx3)
  if (vec !== undefined) {
    // add, mul, loop_start, loop_end
    const dvec = (x: UOp) => {
      if (x.op === Ops.CONST) return UOp.const(x.dtype.vec(vec.dtype.count), x.arg)
      return new UOp(Ops.VECTORIZE, x.dtype.vec(vec.dtype.count), range(vec.dtype.count).map(() => x))
    }
    ;[add, mul, loop_start, loop_end] = [dvec(add), dvec(mul), dvec(loop_start), dvec(loop_end)]
  }

  let comprange
  if (mul.vmin > 0 && ne !== undefined) {
    comprange = loop_end.minimum(add.sub(compval).idiv(mul).add(loop_end.sub(loop_start)).maximum(loop_start))
  } else if (mul.vmax < 0 && ne === undefined) {
    comprange = loop_end.minimum(add.sub(compval).sub(mul).idiv(mul).add(loop_end.sub(loop_start)).maximum(loop_start))
  } else return undefined
  const new_reduce_op = comprange.cast(multconst.dtype).mul(multconst)
  // TODO: what does it mean to have the same numbered DEFINE_ACC with different ranges?
  const new_acc = acc.replace({ src: [...acc.src.slice(0, 1), ...acc.src.slice(1).filter((x) => x !== rng)] })
  let ret = new_acc.assign(new_acc.add(new_reduce_op))
  if (extra !== undefined) ret = ret.add(acc.assign(acc.add(extra)))
  return ret
}

export const index_collapse = (idx: UOp, rng: UOp, buf: UOp, ld: UOp, acc: UOp, add = UOp.const(dtypes.int, 0), mul = UOp.const(dtypes.int, 1)) => {
  if (!acc.src.includes(rng)) return undefined
  const new_load = buf.index(add.add(mul.mul(idx)), idx.ge(rng.src[0]).bitwise_and(idx.lt(rng.src[1]))).load([], { dtype: ld.dtype })
  const new_acc = acc.replace({ src: [acc.src[0], ...acc.src.slice(1).filter((x) => x !== rng)] })
  return new_acc.assign(new_acc.add(new_load))
}
// TODO: there's a lot shared with no_vectorized_wmma here
export const gep_through_wmma = (gep: UOp, wmma: UOp) => {
  const out_sz: number = prod(wmma.arg[6].at(-1)!.map((x: number[]) => x[1]))
  const wmma_idxs: number[] = slice(gep.arg, { step: out_sz })
  for (const i of range(out_sz)) {
    if (!is_eq(slice(gep.arg, { start: i, step: out_sz }).map((x: any) => sub(x, i)), wmma_idxs)) return undefined
  }
  const tsrcs: UOp[] = []
  for (const [s, sz] of zip(wmma.src, wmma.arg[6] as number[][][])) {
    let src_args: number[] = []
    const ssz = prod(sz.map((x) => x[1]))
    for (const w of wmma_idxs) src_args = [...src_args, ...range(mul(idiv(w, out_sz), ssz), add(mul(idiv(w, out_sz), ssz), ssz))]
    tsrcs.push(s.gep(src_args))
  }
  return new UOp(Ops.WMMA, gep.dtype, [...tsrcs], wmma.arg)
}
export const no_vectorized_wmma = (wmma: UOp) => {
  const out_sz: number = prod(wmma.arg[6].at(-1)!.map((x: any) => x[1]))
  if (wmma.dtype.count === out_sz) return undefined
  let tsrcs: UOp[][] = []
  for (const [s, sz] of zip(wmma.src, wmma.arg[6] as number[][][])) {
    const ssz = prod(sz.map((x) => x[1]))
    tsrcs.push(range(0, s.dtype.count, ssz).map((grp) => s.gep(range(grp, grp + ssz))))
  }
  const wmmas = zip(...tsrcs).map((tsrc) => new UOp(Ops.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg))
  const wmma_ex = flatten(wmmas.map((e) => range(out_sz).map((i) => e.gep(i))))
  return new UOp(Ops.VECTORIZE, wmma.dtype, wmma_ex)
}
export const reduce_collapse = (acc: UOp, ret: UOp, alu: UOp) => {
  const [reduce_parented, reduce_unparented] = partition(acc.src.slice(1), (x) => ret.toposort.has(x))
  if (reduce_unparented.length === 0) return undefined
  const new_acc = acc.replace({ src: [acc.src[0], ...reduce_parented] })
  ret = new_acc.assign(new_acc.alu(alu.op, ret))
  if (alu.op === Ops.ADD) {
    for (const r of reduce_unparented) ret = ret.mul((r.src[1].sub(r.src[0])).cast(ret.dtype.scalar()).broadcast(ret.dtype.count))
  }
  return ret
}
export const acc_pat = new UPat(Ops.DEFINE_ACC).named('acc'), rng_pat = new UPat(Ops.RANGE).named('rng')
export const rng_aug = UPat.any([rng_pat, UPat.var('add').add(rng_pat), UPat.var('mul').mul(rng_pat), UPat.var('add').add(UPat.var('mul').mul(rng_pat))])

export const index_load = UPat.var('buf').index(rng_aug).load(undefined, { name: 'ld' })

export const arange_augrng = UPat.any([rng_aug, rng_aug.add(UPat.var('idx2')), rng_aug.add(UPat.var('idx2')).add(UPat.var('idx3')), new UPat(Ops.VECTORIZE, undefined, rng_aug, undefined, 'vec')])
export const arange_m = ((arange_augrng.lt(UPat.cvar('compval'))).ne(new UPat(Ops.CONST, undefined, undefined, true, 'ne'))).where(UPat.cvar('multconst'), UPat.const(undefined, 0))

// this moves the accumulation variable down an unrolled add chain which allows for more efficient accumulation using mulacc
export const mulacc_unrolled = new PatternMatcher([
  UPat.var('x').add(UPat.var('y')).add(acc_pat).fn(({ x, y, acc }) => y.op !== Ops.DEFINE_ACC ? acc.add(x).add(y) : undefined),
])

// this is symbolic 2.0
export const sym = symbolic_flat.add(
  new PatternMatcher([
    // self ASSIGN is just self
    new UPat(Ops.ASSIGN, undefined, [UPat.var('x'), UPat.var('x')]).fn(({ x }) => x),
    // VECTORIZE/CONST, VECTORIZE/GEP
    new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.CONST), undefined, 'vec').fn(({ vec }) => UOp.const(vec.dtype, vec.src.map((x) => x.arg))),
    new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.GEP, undefined, [new UPat(undefined).named('x')]), undefined, 'vec').fn(({ vec, x }) => x.gep(vec.src.map((y) => y.arg[0]))),
    // reorder ALU/VECTORIZE
    new UPat(GroupOp.ALU, undefined, [new UPat(Ops.VECTORIZE, undefined, new UPat(undefined).named('x')), new UPat(Ops.VECTORIZE, undefined, new UPat(undefined).named('y'))], undefined, 'alu').fn(
      ({ x, y, alu }) => new UOp(Ops.VECTORIZE, alu.dtype, range(alu.dtype.count).map(() => new UOp(alu.op, alu.dtype.scalar(), [x, y]))),
    ),
    // VECTORIZE of a single element is just that element
    new UPat(Ops.VECTORIZE, undefined, [new UPat(undefined).named('x')]).fn(({ x }) => x),
    // VECTORIZE void is SINK
    new UPat(Ops.VECTORIZE, dtypes.void, new UPat(Ops.BARRIER).named('b')).fn(({ b }) => b),
    new UPat(Ops.VECTORIZE, dtypes.void).named('x').fn(({ x }) => new UOp(Ops.SINK, dtypes.void, x.src)),
    // GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
    new UPat(Ops.GEP, undefined, [new UPat(Ops.GEP).named('g2')], undefined, 'g1').fn(({ g1, g2 }) => g2.src[0].gep(range(g1.dtype.count).map((i) => g2.arg[g1.arg[i]]))),
    new UPat(Ops.GEP, undefined, [new UPat(Ops.VECTORIZE).named('vec')], undefined, 'gep').fn(
      ({ gep, vec }) => gep.arg.length > 1 ? new UOp(Ops.VECTORIZE, gep.dtype, gep.arg.map((i: number) => vec.src[i])) : vec.src[gep.arg[0]],
    ),
    new UPat(Ops.GEP, undefined, [UPat.cvar('c', undefined, false)], undefined, 'gep').fn(({ gep, c }) => gep.const_like(c.arg)),
    new UPat(Ops.GEP, undefined, [new UPat(Ops.VCONST).named('c')], undefined, 'gep').fn(({ gep, c }) => gep.const_like(gep.arg.map((x: any) => c.arg[x]))),
    // push all GEPs through ALUs (fix arange stuff)
    new UPat(Ops.GEP, undefined, [new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST]).named('alu')], undefined, 'gep').fn(
      ({ gep, alu }) => new UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), alu.src.map((x) => x.gep(gep.arg)), alu.arg),
    ),
    // push some GEPs through WMMAs
    new UPat(Ops.GEP, undefined, [new UPat(Ops.WMMA).named('wmma')], undefined, 'gep').fn(({ wmma, gep }) => gep_through_wmma(gep, wmma)),
    // tensor core with a 0 input is acc
    new UPat(Ops.WMMA, undefined, [UPat.const(undefined, 0.0), UPat.var(), UPat.var('acc')]).fn(({ acc }) => acc),
    new UPat(Ops.WMMA, undefined, [UPat.var(), UPat.const(undefined, 0.0), UPat.var('acc')]).fn(({ acc }) => acc),
    // tensor core cleanups
    UPat.var('add').add(new UPat(Ops.WMMA).named('wmma')).fn(({ add, wmma }) => new UOp(wmma.op, wmma.dtype, [wmma.src[0], wmma.src[1], wmma.src[2].add(add)], wmma.arg)),
    // threefry + remove longs
    new UPat(Ops.THREEFRY, dtypes.uint64, [UPat.var('x'), UPat.var('key')]).fn(({ x, key }) => threefry2x32(x, key)),
    UPat.var('x', dtypes.uint32).cast(dtypes.uint64).cast(dtypes.uint32).fn(({ x }) => x), // cast there and back is noop (TODO: genericize)
    (UPat.var('x', dtypes.uint64).bitwise_and(0xFFFFFFFF)).cast(dtypes.uint32).fn(({ x }) => x.cast(dtypes.uint32)), // cast does truncation
    (UPat.var(undefined, dtypes.uint64).mul(1n << 32n).bitwise_or(UPat.var('y', dtypes.uint32).cast(dtypes.uint64))).cast(dtypes.uint32).fn(({ y }) => y),
    ((UPat.var('x', dtypes.uint64).mul(1n << 32n)).bitwise_or(UPat.var(undefined, dtypes.uint32).cast(dtypes.uint64))).idiv(1n << 32n).fn(({ x }) => x),
    // hacks for threefry long removal when padded (TODO: genericize)
    UPat.var('x', dtypes.uint32).cast(dtypes.uint64).mul(UPat.var('y').where(UPat.const(dtypes.uint64, 1n << 32n), UPat.const(dtypes.uint64, 0))).fn(({ x, y }) => y.where(x, UOp.const(dtypes.uint32, 0)).cast(dtypes.uint64).mul(1n << 32n)),
    (UPat.var('x', dtypes.uint64).bitwise_and(UPat.var('y').where(UPat.const(dtypes.uint64, 0xFFFFFFFF), UPat.const(dtypes.uint64, 0)))).cast(dtypes.uint32).fn(({ x, y }) => y.where(x.cast(dtypes.uint32), UOp.const(dtypes.uint32, 0))),
    // arange loop folding
    acc_pat.assign(UPat.any([arange_m, arange_m.add(UPat.var('extra'))]).add(acc_pat)).fn(({ compval, multconst, rng, acc, idx2, idx3, extra, vec, ne, add, mul }) => loop_collapse(compval, multconst, rng, acc, idx2, idx3, extra, vec, ne, add, mul)),
    // indexing, with cast or where
    acc_pat.assign(UPat.var('idx').eq(new UPat(Ops.RANGE).named('rng')).cast().mul(index_load).add(acc_pat)).fn(({ idx, rng, buf, ld, axx, add, mul }) => index_collapse(idx, rng, buf, ld, axx, add, mul)),
    acc_pat.assign(UPat.var('idx').eq(new UPat(Ops.RANGE).named('rng')).where(index_load, UPat.const(undefined, 0.0)).add(acc_pat)).fn(({ idx, rng, buf, ld, acc, add, mul }) => index_collapse(idx, rng, buf, ld, acc, add, mul)),
    // parentless reduce  # TODO: add MUL
    acc_pat.assign(new UPat([Ops.ADD, Ops.MAX], undefined, [[acc_pat, UPat.var('ret')]]).named('alu')).fn(({ acc, ret, alu }) => reduce_collapse(acc, ret, alu)),
    // ** self folding **
    new UPat(Ops.DEFINE_ACC, undefined, [UPat.var('x')]).fn(({ x }) => x), // a DEFINE_ACC without ranges is a CONST
    new UPat(Ops.ASSIGN, undefined, [UPat.cvar(), UPat.var('x')]).fn(({ x }) => x), // an ASSIGN to a const is a NOOP
    // x!=0 -> (bool)x
    UPat.var('x').ne(0).fn(({ x }) => x.cast(dtypes.bool.vec(x.dtype.count))),
    // ** load/store folding **
    new UPat(Ops.INDEX).named('index').store([new UPat(Ops.INDEX).named('index').load()]).fn(({ index }) => new UOp(Ops.NOOP)),
    new UPat(Ops.INDEX).named('index').store([UPat.var('gate').where(UPat.var('alt'), new UPat(Ops.INDEX).named('index').load())]).fn(({ index, gate, alt }) => index.src[0].index(index.src[1], gate).store([alt])),
    // fold gated LOAD/STORE
    new UPat().index(new UPat(), UPat.const(dtypes.bool, true)).named('idx').fn(({ idx }) => idx.replace({ src: idx.src.slice(0, 2) })), // remove True
    new UPat().index(new UPat(), UPat.const(dtypes.bool, false)).named('idx').fn(({ idx }) => idx.const_like(0)), //False -> NULL pointer
    new UPat(Ops.LOAD, undefined, [UPat.const(undefined, 0)], undefined, 'x', true).fn(({ x }) => x.const_like(0)), // NULL pointer load loads 0
    new UPat(Ops.STORE, undefined, [UPat.const(undefined, 0)], undefined, undefined, true).fn(() => new UOp(Ops.NOOP)), // NULL pointer store does nothing
    // remove NOOPs from SINK
    new UPat(Ops.SINK).named('root').fn(({ root }) => {
      const a = root.src.filter((x) => x.op !== Ops.NOOP)
      return a.length !== root.src.length ? new UOp(Ops.SINK, root.dtype, a, root.arg) : undefined
    }),
    // remove VECTORIZE from SINK/BARRIER
    new UPat(Ops.BARRIER, undefined, [new UPat([Ops.VECTORIZE, Ops.SINK]).named('sink')]).fn(({ sink }) => new UOp(Ops.BARRIER, dtypes.void, sink.src)),
    new UPat(Ops.SINK).named('root').fn(({ root }) => root.src.some((x) => [Ops.SINK, Ops.UNROLL].includes(x.op)) ? new UOp(Ops.SINK, root.dtype, flatten(root.src.map((x) => [Ops.SINK, Ops.UNROLL].includes(x.op) ? x.src : [x])), root.arg) : undefined),
    // stable sigmoid
    UPat.var('x').mul(((UPat.var('x').add(1)).mul(UPat.var('x').add(1))).reciprocal()).fn(({ x }) => sigmoid_like(x, x.const_like(1))),
    UPat.var('x').mul(((UPat.var('x').add(1)).mul(UPat.var('x').add(1))).reciprocal().mul(UPat.var('y'))).fn(({ x, y }) => sigmoid_like(x, y)),
    UPat.var('x').mul(((UPat.var('x').add(1)).mul(UPat.var('x').add(1)).mul(UPat.var('x').add(1))).reciprocal()).fn(({ x }) => sigmoid_like(x, x.add(1).reciprocal())),
  ]),
)

// *** uop expander ***
export const _expand_arg_to_idx = (args: [number, number][], rpk: Map<number, number>): number => {
  let [idx, mul] = [0, 1]
  for (const [axis, m] of args.toReversed()) {
    idx += rpk.get(axis)! * mul
    mul *= m
  }
  return idx
}
export const _choices_from_args = (args: [number, number][]): Map<number, number>[] => {
  return product(...args.map(([axis, m]) => zip(range(100).map(() => axis), range(m)))).map((x) => new Map(x))
}
export const _swizzle_args = cache_fn((cargs: [number, number][], eargs: [number, number][], exclude_args: number[]): number[] => {
  return _choices_from_args(cargs).map((rpk) => _expand_arg_to_idx(eargs, exclude_args ? new Map([...rpk.entries(), ...exclude_args.map((x) => [x, 0] as [number, number])]) : rpk))
})
export const do_expand = (root: UOp) => {
  const expands = root.src.filter((x) => x.op === Ops.UNROLL)
  if (expands.length === 0) return undefined
  // NOTE: we 0 out the reduce axis for WMMA. in theory they should all be the same, but is this always correct?
  const exclude_args = root.op === Ops.WMMA ? dedup([...root.arg.at(-1)!, ...flatten(root.arg.at(-2)).map((y: any) => y[0])]) : []
  const expands_args = expands.map((x) => x.arg)
  let expand_args: [number, number][]
  if (all_same(expands_args) && exclude_args.length === 0) {
    // if there's only one expand arg, it's okay to use it (optimization)
    expand_args = expands[0].arg
  } // otherwise, we sort them and GEP
  else expand_args = sorted(dedup<[number, number]>(flatten(expands_args))).filter((x) => !exclude_args.includes((x as any)[0]))
  const expand_sz = prod(expand_args.map((x) => x[1]))
  const new_srcs = []
  for (const [i, src] of root.src.entries()) {
    if (src.op === Ops.UNROLL) {
      // IF means OR on first arg to IF
      if (root.op === Ops.IF && i === 0) new_srcs.push(range(expand_sz).map((i) => src.src[0].gep(i)).reduce((acc, x) => acc.bitwise_or(x)))
      // just remove the expand
      else if (is_eq(expand_args, src.arg)) new_srcs.push(src.src[0])
      else {
        let lst = _swizzle_args(expand_args, src.arg, exclude_args)
        // if the base dtype is > 1, put those at the end
        if (src.dtype.count > 1) lst = lst.flatMap((i) => range(src.dtype.count).map((j) => i * src.dtype.count + j))
        new_srcs.push(src.src[0].gep([...lst]))
      }
    } else { // non-UNROLL input
      // for the first arg of IF, just pass them through ignoring UNROLLS
      if (root.op === Ops.IF) new_srcs.push(src)
      // put any input dtype > 1 grouped together
      else if (src.dtype.count > 1) new_srcs.push(new UOp(Ops.VECTORIZE, src.dtype.scalar().vec(expand_sz * src.dtype.count), range(expand_sz).flatMap(() => range(src.dtype.count).map((i) => src.gep(i)))))
      // repeat the arg
      else new_srcs.push(src.broadcast(expand_sz))
    }
  }
  let new_arg = root.arg
  if (root.op === Ops.GEP) {
    assert(root.dtype.count === 1)
    // is this right?
    new_arg = range(root.arg[0], new_srcs[0].dtype.count, idiv(new_srcs[0].dtype.count, expand_sz))
  }
  const nsrc = new UOp(root.op, root.dtype.scalar().vec(root.dtype.count * expand_sz), new_srcs, new_arg)
  return new UOp(Ops.UNROLL, root.dtype, [nsrc], expand_args)
}
export const do_contract = (con: UOp) => {
  const ex = con.src[0]
  // CONTRACT without UNROLL repeats the element VECTORIZED
  if (ex.op !== Ops.UNROLL) return new UOp(Ops.VECTORIZE, con.dtype, range(con.dtype.count).flatMap(() => con.src))
  // CONTRACT may remove several axes from UNROLL
  if (con.dtype.count !== prod(con.arg.map((x: any) => x[1]))) throw new Error('dtype is wrong')
  let idxs: number[] = []
  const new_ex_args = ex.arg.filter((x: any) => !con.arg.some((arg: any[]) => is_eq(arg, x)))
  for (const rpk of _choices_from_args(new_ex_args)) {
    idxs = [...idxs, ..._choices_from_args(con.arg).map((lrpk) => _expand_arg_to_idx(ex.arg, new Map([...rpk.entries(), ...lrpk.entries()])))]
  }
  return new UOp(Ops.UNROLL, con.dtype, [ex.src[0].gep([...idxs])], new_ex_args)
}
export const no_vectorized_alu = (alu: UOp) => {
  if (alu.dtype.vcount === 1) return undefined
  const alus = range(alu.dtype.vcount).map((i) => new UOp(alu.op, alu.dtype.scalar(), alu.src.map((s) => s.gep(i)), alu.arg))
  return new UOp(Ops.VECTORIZE, alu.dtype, alus)
}

const _gate_srcs = cache_fn((u: UOp, gate: UOp): UOp => {
  if (u.op === Ops.BARRIER) return u
  if (u.op === Ops.LOAD && u.src.at(-1)!.op === Ops.BARRIER) return new UOp(u.op, u.dtype, [...u.src.toReversed(), new UOp(Ops.IF, dtypes.void, [gate, u.src.at(-1)!])], u.arg)
  const replace_source = u.src.map((x) => _gate_srcs(x, gate))
  return is_eq(replace_source, u.src) ? u : new UOp(u.op, u.dtype, replace_source, u.arg)
})
export const create_gate = (root: UOp): undefined | UOp => {
  let idx = root.src[0]
  if (idx.op === Ops.CAST) idx = idx.src[0]
  const ret = _gate_srcs(root, idx.src[2])
  return idx.op !== Ops.INDEX || idx.src.length === 2 || ret === root ? undefined : ret
}
export const expander = new PatternMatcher([
  // double expand
  new UPat(Ops.UNROLL, undefined, [new UPat(Ops.UNROLL).named('inner')]).named('outer').fn(
    ({ outer, inner }) => new UOp(Ops.UNROLL, outer.dtype, [inner.src[0]], [...inner.arg, ...outer.arg]),
  ),
  // do expansion
  new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WMMA, Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.ASSIGN, Ops.VECTORIZE, Ops.IF], undefined, undefined, undefined, 'root', undefined, undefined, [Ops.UNROLL]).fn(
    ({ root }) => do_expand(root),
  ),
  new UPat(Ops.CONTRACT).named('con').fn(({ con }) => do_contract(con)),
  // vectorize DEFINE_ACC
  new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.DEFINE_ACC).named('acc')).named('v').fn(({ acc, v }) => acc.replace({ dtype: v.dtype })),
  // BARRIERs aren't actually expanded
  new UPat(Ops.BARRIER, undefined, [new UPat(Ops.UNROLL).named('ex')]).fn(
    ({ ex }) => new UOp(Ops.UNROLL, dtypes.void, range(ex.src.length).map(() => new UOp(Ops.BARRIER, dtypes.void, ex.src)), ex.arg),
  ),
  // empty EXPAND is NOOP
  new UPat(Ops.UNROLL, undefined, [UPat.var('x')], []).fn(({ x }) => x),
  // EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  new UPat(Ops.UNROLL, undefined, range(AMX ? 256 : 8).map((i) => UPat.var('x').gep(i).add(UPat.var('y').gep(i)))).named('ex').fn(
    ({ ex, x, y }) => new UOp(Ops.UNROLL, ex.dtype, range(AMX ? 256 : 8).map((i) => x.add(y).gep(i)), ex.arg),
  ),
])

export const no_vectorized_load_store = (ls: UOp) => {
  const idx = ls.src[0]
  if (!isinstance(idx.dtype, PtrDType)) throw new Error()
  if (idx.dtype.v === 1) return undefined
  const tv = range(idx.dtype.v).map((i) => new UOp(ls.op, ls.dtype.scalar(), ls.src.map((j) => j.gep(i))))
  return new UOp(Ops.VECTORIZE, ls.dtype, tv)
}
export const no_vectorized_acc = (acc: UOp) => {
  if (acc.dtype.count === 1) return undefined
  const alus = range(acc.dtype.count).map((i) => new UOp(acc.op, acc.dtype.scalar(), [...acc.src.entries()].map(([j, s]) => j === 0 ? s.gep(i) : s), [...acc.arg, i]))
  return new UOp(Ops.VECTORIZE, acc.dtype, alus)
}
export const devectorize = new PatternMatcher([
  // no ALU on vectorized dtypes
  new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX]).named('alu').fn(({ alu }) => no_vectorized_alu(alu)),
  new UPat(Ops.WMMA).named('wmma').fn(({ wmma }) => no_vectorized_wmma(wmma)),
  new UPat(Ops.DEFINE_ACC).named('acc').fn(({ acc }) => no_vectorized_acc(acc)),
  new UPat([Ops.LOAD, Ops.STORE]).named('ls').fn(({ ls }) => no_vectorized_load_store(ls)),
])

export const delete_redundant_gates = (buf: UOp, idx: UOp, val: UOp, store_gate: UOp, cast?: UOp): undefined | UOp => {
  if (![...val.toposort].filter((gate) => gate.op === Ops.IF).map((gate) => gate.src[0]).includes(store_gate)) return undefined
  // remove the gate from the index
  return (cast !== undefined ? buf.index(idx).cast(cast.dtype) : buf.index(idx)).store([val])
}
const _stidx = UPat.var('buf').index(UPat.var('idx'), UPat.var('store_gate'))
export const load_store_indexing = new PatternMatcher([
  // late fixup of unfoldable image loads
  new UPat(Ops.LOAD, undefined, [UPat.var('buf'), new UPat()], undefined, 'load', true).fn(({ load, buf }) => fix_unfoldable_image_load(load, buf)),
  // simplify valid
  new UPat(Ops.AND).named('valid').fn(({ valid }) => simplify_valid(valid)),
  // image load valid idx simplification
  new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('start_idx'), UPat.var('valid')]).fn(({ buf, start_idx, valid }) => simplify_valid_load(buf, start_idx, valid)),
  // delete_redundant_gates (after expand)
  new UPat(Ops.STORE, undefined, [UPat.any([_stidx, _stidx.cast(undefined).named('cast')]), UPat.var('val')]).fn(({ buf, idx, val, store_gate, cast }) => delete_redundant_gates(buf, idx, val, store_gate, cast)),
])

export const migrate_indexing = new PatternMatcher([
  // create gate MUST BE BEFORE expander
  new UPat(Ops.STORE).named('root').fn(({ root }) => create_gate(root)),
])

export const move_mask = (x: UOp, buf: UOp, idx: UOp, mask: UOp, cast?: UOp): UOp => {
  // this moves the mask from the indexing to the load/store op for rendering
  const nidx = cast !== undefined ? buf.index(idx).cast(cast.dtype) : buf.index(idx)
  return x.op === Ops.LOAD ? nidx.load([x.const_like(0), mask, ...x.src.slice(1)], { dtype: x.dtype }) : nidx.store([x.src[1], mask, ...x.src.slice(2)])
}
const _masked_index = new UPat(Ops.INDEX, undefined, [new UPat(undefined).named('buf'), new UPat(undefined).named('idx'), new UPat(undefined).named('mask')])
export const pm_render = new PatternMatcher([
  // for rendering, we use explicit VECTORIZE
  new UPat(Ops.CONST).named('c').fn(({ c }) => c.dtype.vcount > 1 ? new UOp(Ops.VECTORIZE, c.dtype, range(c.dtype.vcount).map(() => UOp.const(c.dtype.scalar(), c.arg))) : undefined),
  new UPat(Ops.VCONST).named('c').fn(({ c }) => new UOp(Ops.VECTORIZE, c.dtype, c.arg.map((x: number) => UOp.const(c.dtype.scalar(), x)))),
  new UPat(Ops.GEP).named('gep').fn(({ gep }) => gep.arg.length > 1 ? new UOp(Ops.VECTORIZE, gep.dtype, gep.arg.map((x: number) => gep.src[0].gep(x))) : undefined),
  new UPat(Ops.VECTORIZE, undefined, [new UPat(undefined).named('x')]).fn(({ x }) => x),
  // move masks of loads/stores
  new UPat([Ops.LOAD, Ops.STORE], undefined, [UPat.any([_masked_index, _masked_index.cast(undefined).named('cast')])], undefined, 'x', true).fn(
    ({ x, buf, idx, mask, cast }) => move_mask(x, buf, idx, mask, cast),
  ),
  // gate any stores that aren't gated with ifs
  new UPat(Ops.STORE, dtypes.void, [new UPat(), new UPat(), new UPat(undefined, dtypes.bool)], undefined, 'store').fn(
    ({ store }) => new UOp(Ops.STORE, undefined, [...store.src.slice(0, 2), new UOp(Ops.IF, undefined, [store.src[2]])]),
  ),
])

// *** uop graph ***

export const full_graph_rewrite = (sink: UOp, opts?: Renderer): UOp => {
  if (sink.op !== Ops.SINK) throw new Error(`sink isn't sink, it's ${sink.op}`)
  const supported_ops = opts !== undefined ? [...opts.code_for_op.keys()] : []
  const extra_matcher = opts !== undefined && opts.extra_matcher !== undefined ? opts.extra_matcher : new PatternMatcher([])

  // initial symbolic + migrate indexing (remove this)
  sink = graph_rewrite(sink, sym.add(migrate_indexing))
  // expand
  sink = graph_rewrite(sink, sym.add(expander))

  // devectorize + load_store_indexing
  sink = graph_rewrite(sink, sym.add((opts !== undefined && opts.supports_float4) ? devectorize.add(float4_folding) : devectorize).add(load_store_indexing).add(mulacc_unrolled))

  // final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple.add(get_late_rewrite_patterns(supported_ops, TRANSCENDENTAL >= 2)).add(pm_render).add(extra_matcher))
  return sink
}
