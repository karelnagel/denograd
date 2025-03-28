import { type DType, dtypes, PtrDType } from '../dtype.ts'
import { type ConstType, DefaultMap, range } from '../helpers/helpers.ts'
import { Ops, PatternMatcher, type UOp, UPat } from '../ops.ts'
import { Renderer } from './index.ts'

const prefixes = new Map([
  [Ops.RANGE, 'ridx'],
  [Ops.WMMA, 'wmma'],
  [Ops.DEFINE_LOCAL, 'temp'],
  [Ops.CAST, 'cast'],
  [Ops.BITCAST, 'cast'],
  [Ops.GEP, 'gep'],
  [Ops.VECTORIZE, 'cast'],
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

const size = (dtype: DType) => dtype.itemsize === 8 ? '64' : '32'
const get_dtype = (dtype: DType): string => {
  dtype = dtype.base
  if (dtype === dtypes.float32 || dtype === dtypes.float64) return `f${size(dtype)}`
  if (dtypes.ints.includes(dtype) || dtype === dtypes.bool) return `i${size(dtype)}`
  if (dtype.vcount > 1) return `${get_dtype(dtype.scalar())}x4`
  throw new Error(`${dtype} not supported`)
}
const cast = (from: UOp, to: UOp, ctx: WATRenderer): string[] => {
  if (to.dtype === dtypes.bool) return [`(${get_dtype(from.dtype)}.ne`, `(${get_dtype(from.dtype)}.const 0)`, ...ctx.get_var(from), ')']
  if (from.dtype instanceof PtrDType && to.dtype instanceof PtrDType) return [`;; Should be casted from ${from.dtype} to ${to.dtype}`, ...ctx.get_var(from)]

  const a = get_dtype(from.dtype), b = get_dtype(to.dtype), sign = dtypes.is_unsigned(from.dtype) || dtypes.is_unsigned(to.dtype) ? 'u' : 's'
  if (a === 'i32' && b === 'i64') return [`(i64.extend_i32_${sign}`, ...ctx.get_var(from), ')']
  if (a === 'i64' && b === 'i32') return [`(i32.wrap_i64`, ...ctx.get_var(from), ')']
  if (a === 'f32' && b === 'f64') return [`(f64.promote_f32`, ...ctx.get_var(from), ')']
  if (a === 'f64' && b === 'f32') return [`(f32.demote_f64`, ...ctx.get_var(from), ')']
  if (['f32', 'f64'].includes(a) && ['i32', 'i64'].includes(b)) return [`(${b}.trunc_${a}_${sign}`, ...ctx.get_var(from), ')']
  if (['i32', 'i64'].includes(a) && ['f32', 'f64'].includes(b)) return [`(${b}.convert_${a}_${sign}`, ...ctx.get_var(from), ')']
  if (a === b) return ctx.get_var(from)
  throw new Error(`Can't cast ${from} to ${to}`)
}
const load_fn = (dtype: DType) => {
  if (dtype === dtypes.float32 || dtype === dtypes.float64) return `f${size(dtype)}.load`
  if (dtypes.uints.includes(dtype) || dtype === dtypes.bool) return `i${size(dtype)}.load${dtype.itemsize < 4 ? `${dtype.itemsize * 8}_u` : ''}`
  if (dtypes.ints.includes(dtype)) return `i${size(dtype)}.load${dtype.itemsize < 4 ? `${dtype.itemsize * 8}_s` : ''}`
  if (dtype.vcount > 1) return `v128.load`
  throw new Error(`Loading ${dtype} not supported`)
}
const store_fn = (dtype: DType) => {
  if (dtype === dtypes.float32 || dtype === dtypes.float64) return `f${size(dtype)}.store`
  if (dtypes.ints.includes(dtype) || dtype === dtypes.bool) return `i${size(dtype)}.store${dtype.itemsize < 4 ? dtype.itemsize * 8 : ''}`
  if (dtype.vcount > 1) return `v128.store`
  throw new Error(`Storing ${dtype} not supported`)
}

const constant = (num: ConstType) => {
  if (typeof num === 'boolean') return Number(num).toString()
  if (typeof num === 'bigint') return num.toString()
  if (typeof num === 'number') return isNaN(num) ? 'nan' : num === Infinity ? 'inf' : num === -Infinity ? '-inf' : Number(num).toString()
  throw new Error(`Invalid const ${num}`)
}
// TODO: handle NaNs and Infinity
// TODO: handle uints correcly, should use '..._u' functions for those
const string_rewrite = new PatternMatcher<WATRenderer, string[] | undefined>([
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
    return [`(${dtype}.add`, ...ctx.get_var(base), `(${get_dtype(gep.dtype)}.const ${gep.arg[0] * gep.dtype.itemsize})`, `)`]
  }),
  new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]).named('index').fn(({ index, buf, idx, ctx }) => [`(${get_dtype(idx.dtype)}.add`, `(${get_dtype(idx.dtype)}.mul`, ...ctx.get_var(idx), `(${get_dtype(idx.dtype)}.const ${index.dtype.itemsize})`, `)`, ...ctx.get_var(buf), `)`]),
  new UPat(Ops.ASSIGN).named('x').fn(({ x, ctx }) => [`(local.set ${ctx.get(x.src[0])}`, ...ctx.get_var(x.src[1]), ')']),
  new UPat(Ops.LOAD).named('load').fn(({ load, ctx }) => [`(${load_fn(load.dtype)}`, ...ctx.get_var(load.src[0]), ')']),
  new UPat(Ops.STORE).named('store').fn(({ store, ctx }) => [`(${store_fn(store.src[1].dtype)}`, ...ctx.get_var(store.src[0]), ...ctx.get_var(store.src[1]), ')']),
  new UPat(Ops.RANGE, undefined, [UPat.var('from'), UPat.var('to')]).named('range').fn(({ ctx, range, from, to }) => [`(local.set ${ctx.get(range)} ${ctx.get(from)})`, `(block $block${range.arg}`, `(loop $loop${range.arg}`, `(br_if $block${range.arg}`, `(${get_dtype(range.dtype)}.eq`, ...ctx.get_var(range), ...ctx.get_var(to), `)`, `)`]),
  new UPat(Ops.ENDRANGE, undefined, [new UPat(Ops.RANGE).named('range')]).named('endrange').fn(({ endrange, range, ctx }) => [`(br $loop${range.arg}`, `(local.set ${ctx.get(range)}`, `(${get_dtype(range.dtype)}.add`, ...ctx.get_var(range), `(${get_dtype(range.dtype)}.const 1)`, `)`, `)`, `)`, ')', ')']),
  new UPat(Ops.CAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => cast(from, to, ctx)),
  new UPat(Ops.BITCAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => [`(${get_dtype(to.dtype)}.reinterpret_${get_dtype(from.dtype)}`, ...ctx.get_var(from), ')']),
  new UPat(Ops.VECTORIZE, undefined, range(4).map(() => UPat.const())).named('x').fn(({ x, ctx }) => [`(v128.const ${get_dtype(x.dtype.scalar())}x4 ${x.src.map((x) => x.arg).join(' ')})`]),
  new UPat(Ops.VECTORIZE).named('x').fn(({ x, ctx }) => ['(v128.const f32x4 0 0 0 0)', ...x.src.flatMap((x, i) => [...ctx.get_var(x), `(f32x4.replace_lane ${i})`])]),
])

export class WATRenderer extends Renderer {
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
        if (dtype === 'f32x4' || dtype === 'i32x4') throw new Error(`Vec consts aren't handled, uop: ${uop}`)
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
        lines = [...lines, ...str]
      } // writing these as variables
      else if (child_count.get(uop) > 1) {
        lines = [...lines, `(local.set ${this.get(uop)}`, ...str, ')']
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
