import { type DType, dtypes, PtrDType } from '../dtype.ts'
import { type ConstType, DefaultMap, range } from '../helpers.ts'
import { Ops, PatternMatcher, type UOp, UPat } from '../ops.ts'
import { Renderer } from './index.ts'

const prefixes = new Map([
  [Ops.RANGE, 'ridx'],
  [Ops.WMMA, 'wmma'],
  [Ops.DEFINE_LOCAL, 'temp'],
  // [Ops.CONST, 'const'],
  [Ops.CAST, 'cast'],
  [Ops.BITCAST, 'cast'],
  [Ops.GEP, 'gep'],
  [Ops.VECTORIZE, 'cast'],
  // [Ops.NOOP, 'precast'],
  [Ops.INDEX, 'bidx'],
  [Ops.DEFINE_ACC, 'acc'],
  [Ops.LOAD, 'val'],
])
// [float, int, uint]
const _alus = new Map([
  [Ops.ADD, ['add', 'add', 'add']],
  [Ops.SUB, ['sub', 'sub', 'sub']],
  [Ops.MUL, ['mul', 'mul', 'mul']],
  [Ops.FDIV, ['div', undefined, undefined]],
  [Ops.IDIV, [undefined, 'div_s', 'div_u']],
  [Ops.MOD, [undefined, 'rem_s', 'rem_u']],
  [Ops.NEG, ['neg', undefined, undefined]],
  [Ops.SQRT, ['sqrt', undefined, undefined]],
  [Ops.MAX, ['max', undefined, undefined]],
  [Ops.SHL, [undefined, 'shl', 'shl']],
  [Ops.SHR, [undefined, 'shr_s', 'shr_u']],
  [Ops.XOR, [undefined, 'xor', 'xor']],
  [Ops.OR, [undefined, 'or', 'or']],
  [Ops.AND, [undefined, 'and', 'and']],
  [Ops.CMPLT, ['lt', 'lt_s', 'lt_u']],
  [Ops.CMPNE, ['ne', 'ne', 'ne']],
])
const _render_dtype = new Map([
  [dtypes.int32, 'i32'],
  [dtypes.int64, 'i64'],
  [dtypes.uint32, 'i32'],
  [dtypes.uint64, 'i64'],
  [dtypes.float, 'f32'],
  [dtypes.float32, 'f32'],
  [dtypes.float64, 'f64'],
  [dtypes.bool, 'i32'],
  [dtypes.uchar, 'i32'],

  [dtypes.int32.vec(4), 'i32x4'],
  [dtypes.float32.vec(4), 'f32x4'],
  [dtypes.float32.vec(2), 'f32x4'],
  [dtypes.float32.vec(3), 'f32x4'],
])
const _loads = new Map([
  [dtypes.int32, 'i32.load'],
  [dtypes.int64, 'i64.load'],
  [dtypes.uint32, 'i32.load'],
  [dtypes.uint64, 'i64.load'],
  [dtypes.float32, 'f32.load'],
  [dtypes.float64, 'f64.load'],
  [dtypes.int8, 'i32.load8_s'],
  [dtypes.uint8, 'i32.load8_u'],
  [dtypes.bool, 'i32.load8_u'],
  [dtypes.uchar, 'i32.load8_u'],
  [dtypes.int16, 'i32.load16_s'],
  [dtypes.uint16, 'i32.load16_u'],

  [dtypes.int32.vec(4), 'v128.load'],
  [dtypes.float32.vec(4), 'v128.load'],
  [dtypes.float32.vec(3), 'v128.load'],
  [dtypes.float32.vec(2), 'v128.load'],
])
const _stores = new Map([
  [dtypes.int32, 'i32.store'],
  [dtypes.int64, 'i64.store'],
  [dtypes.uint32, 'i32.store'],
  [dtypes.uint64, 'i64.store'],
  [dtypes.float32, 'f32.store'],
  [dtypes.float64, 'f64.store'],
  [dtypes.int8, 'i32.store8'],
  [dtypes.uint8, 'i32.store8'],
  [dtypes.bool, 'i32.store8'],
  [dtypes.uchar, 'i32.store8'],
  [dtypes.int16, 'i32.store16'],
  [dtypes.uint16, 'i32.store16'],

  [dtypes.int32.vec(4), 'v128.store'],
  [dtypes.float32.vec(4), 'v128.store'],
  [dtypes.float32.vec(3), 'v128.store'],
  [dtypes.float32.vec(2), 'v128.store'],
])

const get_dtype = (dtype: DType) => {
  const res = _render_dtype.get(dtype.base)
  if (!res) throw new Error(`WASM doesn't support ${dtype} dtype`)
  return res
}
const cast = (from: UOp, to: UOp, ctx: WASMRenderer): string[] => {
  if (to.dtype === dtypes.bool) return [`(${get_dtype(from.dtype)}.ne`, `(${get_dtype(from.dtype)}.const 0)`, ...ctx.get_var(from), ')']
  if (from.dtype instanceof PtrDType && to.dtype instanceof PtrDType) return [`;; Should be casted from ${from.dtype} to ${to.dtype}`, ...ctx.get_var(from)]
  try {
    const a = get_dtype(from.dtype), b = get_dtype(to.dtype), sign = dtypes.is_unsigned(from.dtype) || dtypes.is_unsigned(to.dtype) ? 'u' : 's'
    if (a === 'i32' && b === 'i64') return [`(i64.extend_i32_${sign}`, ...ctx.get_var(from), ')']
    if (a === 'i64' && b === 'i32') return [`(i32.wrap_i64`, ...ctx.get_var(from), ')']
    if (a === 'f32' && b === 'f64') return [`(f64.promote_f32`, ...ctx.get_var(from), ')']
    if (a === 'f64' && b === 'f32') return [`(f32.demote_f64`, ...ctx.get_var(from), ')']
    if (['f32', 'f64'].includes(a) && ['i32', 'i64'].includes(b)) return [`(${b}.trunc_${a}_${sign}`, ...ctx.get_var(from), ')']
    if (['i32', 'i64'].includes(a) && ['f32', 'f64'].includes(b)) return [`(${b}.convert_${a}_${sign}`, ...ctx.get_var(from), ')']
    if (a === b) return ctx.get_var(from)
    throw new Error(`Can't cast ${from} to ${to}`)
  } catch (e: any) {
    throw new Error(`${e}\n${from}\n${to}`)
  }
}

const load_fn = (dtype: DType) => {
  const load = _loads.get(dtype)
  if (!load) throw new Error(`Loading dtype ${dtype} is not supported in wasm`)
  return load
}
const store_fn = (dtype: DType) => {
  const store = _stores.get(dtype)
  if (!store) throw new Error(`Storing dtype ${dtype} is not supported in wasm`)
  return store
}

const constant = (num: ConstType) => {
  if (typeof num === 'boolean') return Number(num).toString()
  if (typeof num === 'bigint') return num.toString()
  if (typeof num === 'number') return isNaN(num) ? 'nan' : num === Infinity ? 'inf' : num === -Infinity ? '-inf' : Number(num).toString()
  throw new Error(`Invalid const ${num}`)
}
// TODO: handle NaNs and Infinity
// TODO: handle uints correcly, should use '..._u' functions for those
const string_rewrite = new PatternMatcher<WASMRenderer, string[] | undefined>([
  new UPat(Ops.DEFINE_ACC).named('acc').fn(({ acc, ctx }) => [`(local.set ${ctx.get(acc)} ${ctx.get(acc.src[0])})`]),
  // ALU
  new UPat(Ops.WHERE, undefined, [UPat.var('cond'), UPat.var('a'), UPat.var('b')]).fn(({ ctx, a, b, cond }) => ['(select', ...ctx.get_var(a), ...ctx.get_var(b), ...ctx.get_var(cond), ')']),
  new UPat(Ops.RECIP, undefined, [UPat.var('x')]).fn(({ x, ctx }) => ['(f32.div', '(f32.const 1.0)', ...ctx.get_var(x), ')']),
  new UPat([..._alus.keys()]).named('alu').fn(({ alu, ctx }) => {
    const first = alu.src[0], index = dtypes.is_float(first.dtype) ? 0 : !dtypes.is_unsigned(first.dtype) ? 1 : 2
    const fn = _alus.get(alu.op)?.[index]
    return fn ? [`(${get_dtype(first.dtype)}.${fn}`, ...alu.src.flatMap((a) => ctx.get_var(a)), ')'] : undefined
  }),
  // TODO: EXP2, LOG2, SIN
  new UPat(Ops.GEP, undefined, [UPat.var('base')]).named('gep').fn(({ gep, base, ctx }) => {
    const dtype = get_dtype(base.dtype)
    if (dtype === 'f32x4' || dtype === 'i32x4') return [`(${dtype}.extract_lane ${gep.arg[0]}`, ...ctx.get_var(base), `)`]
    return [
      `(${dtype}.add`,
      ...ctx.get_var(base),
      `(${get_dtype(gep.dtype)}.const ${gep.arg[0] * gep.dtype.itemsize})`,
      `)`,
    ]
  }),
  new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]).named('index').fn(({ index, buf, idx, ctx }) => [
    `(${get_dtype(idx.dtype)}.add`,
    `(${get_dtype(idx.dtype)}.mul`,
    ...ctx.get_var(idx),
    `(${get_dtype(idx.dtype)}.const ${index.dtype.itemsize})`,
    `)`,
    ...ctx.get_var(buf),
    `)`,
  ]),
  new UPat(Ops.ASSIGN).named('x').fn(({ x, ctx }) => [`(local.set ${ctx.get(x.src[0])}`, ...ctx.get_var(x.src[1]), ')']),
  new UPat(Ops.LOAD).named('load').fn(({ load, ctx }) => [`(${load_fn(load.dtype)}`, ...ctx.get_var(load.src[0]), ')']),
  new UPat(Ops.STORE).named('store').fn(({ store, ctx }) => [`(${store_fn(store.src[1].dtype)}`, ...ctx.get_var(store.src[0]), ...ctx.get_var(store.src[1]), ')']),
  new UPat(Ops.RANGE, undefined, [UPat.var('from'), UPat.var('to')]).named('range').fn(({ ctx, range, from, to }) => [
    `(local.set ${ctx.get(range)} ${ctx.get(from)})`,
    `(block $block${range.arg}`,
    `(loop $loop${range.arg}`,
    `(br_if $block${range.arg}`,
    `(${get_dtype(range.dtype)}.eq`,
    ...ctx.get_var(range),
    ...ctx.get_var(to),
    `)`,
    `)`,
  ]),
  new UPat(Ops.ENDRANGE, undefined, [new UPat(Ops.RANGE).named('range')]).named('endrange').fn(({ endrange, range, ctx }) => [
    `(br $loop${range.arg}`,
    `(local.set ${ctx.get(range)}`,
    `(${get_dtype(range.dtype)}.add`,
    ...ctx.get_var(range),
    `(${get_dtype(range.dtype)}.const 1)`,
    `)`,
    `)`,
    `)`,
    ')',
    ')',
  ]),
  new UPat(Ops.CAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => cast(from, to, ctx)),
  new UPat(Ops.BITCAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => [`(${get_dtype(to.dtype)}.reinterpret_${get_dtype(from.dtype)}`, ...ctx.get_var(from), ')']),
  new UPat(Ops.VECTORIZE, undefined, range(4).map(() => UPat.const())).named('x').fn(({ x, ctx }) => [
    `(v128.const ${get_dtype(x.dtype.scalar())}x4 ${x.src.map((x) => x.arg).join(' ')})`,
  ]),
  new UPat(Ops.VECTORIZE).named('x').fn(({ x, ctx }) => [
    '(v128.const f32x4 0 0 0 0)',
    ...x.src.flatMap((x, i) => [...ctx.get_var(x), `(f32x4.replace_lane ${i})`]),
  ]),
])
export class WASMRenderer extends Renderer {
  override has_local = false
  override has_shared = false
  override extra_matcher = new PatternMatcher([
    new UPat(Ops.MAX, dtypes.ints, [UPat.var('a'), UPat.var('b')]).fn(({ a, b }) => (a.lt(b)).where(b, a)),
    // TODO: needed cause i32.trunc_f32_s(0.99) will return 0
    new UPat(Ops.WHERE, undefined, [UPat.var('cond'), UPat.var('a'), UPat.var('b')]).fn(({ cond, a, b }) => (cond.op === Ops.CAST) ? (cond.src[0].ne(0)).where(a, b) : undefined),
  ])

  string_rewrite = string_rewrite
  r?: Map<UOp, string[]>
  get = (uop: UOp): string => this.r!.get(uop)!.join(' ')
  get_var = (uop: UOp): string[] => {
    const res = this.r!.get(uop)
    if (!res?.length) throw new Error(`UOp ${uop} not in r!`)
    return res[0].startsWith('$') ? [`(local.get ${res[0]})`] : res
  }
  override render = (name: string, uops: UOp[]) => {
    let lines: string[] = []
    this.r = new Map<UOp, string[]>()

    // Finding out if value is used more than once and we should store it as a variable
    const prefix_count = new DefaultMap<string, number>(undefined, () => 0)
    const child_count = new DefaultMap<UOp, number>(undefined, () => 0)
    for (const u of uops) for (const v of u.src) child_count.set(v, child_count.get(v) + 1)

    const defs: string[] = []
    for (const uop of uops) {
      // These don't need to be written into lines
      if (uop.op === Ops.DEFINE_GLOBAL) {
        this.r.set(uop, [`$data${uop.arg}`])
        defs.push(`(param ${this.get(uop)} i32)`)
        continue
      } // Using src[0] as the value
      else if (uop.op === Ops.ASSIGN) {
        this.r.set(uop, this.r.get(uop.src[0])!)
      } // Skipping consts
      else if (uop.op === Ops.CONST) {
        let dtype = get_dtype(uop.dtype)
        if (dtype === 'f32x4') throw new Error(`${dtype} ${uop}, ${uop.dtype === dtypes.float}`)
        if (dtype === 'i32x4') throw new Error()
        this.r.set(uop, [`(${dtype}.const ${constant(uop.arg)})`])
        continue
      } //If used more than once or needs a variable
      else if ([Ops.RANGE, Ops.DEFINE_ACC].includes(uop.op) || child_count.get(uop) > 1) {
        const prefix = prefixes.get(uop.op) || 'alu', count = prefix_count.get(prefix)
        const name = `$${prefix}${count}`
        this.r.set(uop, [name])
        let dtype = get_dtype(uop.dtype)
        if (uop.op === Ops.INDEX) dtype = 'i32'
        if (dtype === 'f32x4' && uop.op === Ops.CAST) dtype = 'i32'
        if (dtype === 'i32x4' && uop.op === Ops.CAST) dtype = 'i32'
        if (dtype === 'f32x4' || dtype === 'i32x4') dtype = 'v128'
        defs.push(`(local ${name} ${dtype})`)
        prefix_count.set(prefix, count + 1)
      }

      let str = this.string_rewrite.rewrite(uop, this)
      if (!str) throw new Error(`No matcher for ${uop}`)

      // add to lines for these Ops, others add to context
      if ([Ops.RANGE, Ops.ENDRANGE, Ops.STORE, Ops.ASSIGN, Ops.DEFINE_ACC].includes(uop.op)) {
        lines = [...lines, ';; ' + uop.op.toString(), ...str]
      } // writing these as variables
      else if ([Ops.RANGE, Ops.DEFINE_ACC].includes(uop.op) || child_count.get(uop) > 1) {
        lines = [...lines, ';; ' + uop.op.toString(), `(local.set ${this.get(uop)}`, ...str, ')']
      } // just saving the result for future use
      else {
        this.r.set(uop, str)
      }
    }

    lines = [`(module`, `(import "env" "memory" (memory 1))`, `(func (export "${name}")`, ...defs, ...lines, ')', ')']
    let res = '', indent = 0
    for (const line of lines) {
      const change = line.split('(').length - line.split(')').length
      if (change < 0) indent += change
      res += '  '.repeat(indent) + line + '\n'
      if (change > 0) indent += change
    }
    this.r = undefined
    return res
  }
}
