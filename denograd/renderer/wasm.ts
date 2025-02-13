import { type DType, dtypes, type PtrDType } from '../dtype.ts'
import { GroupOp, Ops, PatternMatcher, type UOp, UPat } from '../ops.ts'
import { Renderer } from './index.ts'

const prefixes = new Map([[Ops.DEFINE_GLOBAL, 'data'], [Ops.RANGE, 'ridx']])
const _render_dtype = new Map([[dtypes.int32, 'i32'], [dtypes.int64, 'i64'], [dtypes.uint32, 'i32'], [dtypes.uint64, 'i64'], [dtypes.float32, 'f32'], [dtypes.float64, 'f64'], [dtypes.bool, 'i32']])
const get_dtype = (dtype: DType) => {
  const res = _render_dtype.get(dtype.base) || _render_dtype.get(dtype.base.scalar())
  if (!res) throw new Error(`WASM doesn't support ${dtype} dtype`)
  return res
}
const cast = (from: DType, to: DType) => {
  const a = get_dtype(from), b = get_dtype(to), sign = dtypes.is_unsigned(from) || dtypes.is_unsigned(to) ? 'u' : 's'
  if (a === 'i32' && b === 'i64') return `i64.extend_i32_${sign}`
  if (a === 'i64' && b === 'i32') return `i32.wrap_i64`
  if (a === 'f32' && b === 'f64') return `f64.promote_f32`
  if (a === 'f64' && b === 'f32') return `f32.demote_f64`
  if (['f32', 'f64'].includes(a) && ['i32', 'i64'].includes(b)) return `${b}.trunc_${a}_${sign}`
  if (['i32', 'i64'].includes(a) && ['f32', 'f64'].includes(b)) return `${b}.convert_${a}_${sign}`
  if (a === b) return undefined
  throw new Error(`Can't cast ${from} to ${to}`)
}
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
const float = (num: number) => isNaN(num) ? 'nan' : num === Infinity ? 'inf' : num === -Infinity ? '-inf' : Number(num).toString()
// TODO: handle NaNs and Infinity
// TODO: handle uints correcly, should use '..._u' functions for those
const string_rewrite = new PatternMatcher<WASMRenderer, string[] | undefined>([
  new UPat(Ops.CONST).named('c').fn(({ c, ctx }) => [`(${get_dtype(c.dtype)}.const ${float(c.arg)})`]),
  // ALU
  new UPat(Ops.WHERE, undefined, [UPat.var('cond'), UPat.var('a'), UPat.var('b')]).fn(({ ctx, a, b, cond }) => ['(select', ...ctx.var(a), ...ctx.var(b), ...ctx.var(cond), ')']),
  new UPat(Ops.RECIP, undefined, [UPat.var('x')]).fn(({ x, ctx }) => ['(f32.div', '(f32.const 1.0)', ...ctx.var(x), ')']),
  new UPat([..._alus.keys()]).named('alu').fn(({ alu, ctx }) => {
    const first = alu.src[0], index = dtypes.is_float(first.dtype) ? 0 : !dtypes.is_unsigned(first.dtype) ? 1 : 2
    const fn = _alus.get(alu.op)?.[index]
    return fn ? [`(${get_dtype(first.dtype)}.${fn}`, ...alu.src.flatMap((a) => ctx.var(a)), ')'] : undefined
  }),
  // TODO: EXP2, LOG2, SIN
  new UPat(Ops.GEP, undefined, [UPat.var('base')]).named('gep').fn(({ gep, base, ctx }) => [`(${get_dtype(base.dtype)}.add`, ...ctx.var(base), `(${get_dtype(base.dtype)}.const ${gep.arg[0] * gep.dtype.itemsize})`, `)`]),
  new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]).named('index').fn(({ index, buf, idx, ctx }) => [
    `(${get_dtype(idx.dtype)}.add`,
    `(${get_dtype(idx.dtype)}.mul`,
    ...ctx.var(idx),
    `(${get_dtype(idx.dtype)}.const ${index.dtype.itemsize})`,
    `)`,
    ...ctx.var(buf),
    `)`,
  ]),
  new UPat(Ops.LOAD).named('load').fn(({ load, ctx }) => [`(${get_dtype(load.dtype)}.load`, ...ctx.var(load.src[0]), ')']),
  new UPat(Ops.STORE).named('store').fn(({ store, ctx }) => [`(${get_dtype(store.src[1].dtype)}.store`, ctx.first(store.src[0]), ...ctx.var(store.src[1]), ')']),
  new UPat(Ops.RANGE).named('range').fn(({ ctx, range }) => [
    `(block $block${range.arg}`,
    `(loop $loop${range.arg}`,
    `(br_if $block${range.arg}`,
    `(${get_dtype(range.dtype)}.eq`,
    ...ctx.var(range),
    ...ctx.var(range.src[1]),
    `)`,
    `)`,
  ]),
  new UPat(Ops.ENDRANGE, undefined, [new UPat(Ops.RANGE).named('range')]).named('endrange').fn(({ endrange, range, ctx }) => [
    `(br $loop${range.arg}`,
    `(local.set ${ctx.first(range)}`,
    `(${get_dtype(range.dtype)}.add`,
    ...ctx.var(range),
    `(${get_dtype(range.dtype)}.const 1)`,
    `)`,
    `)`,
    `)`,
    ')',
    ')',
  ]),
  new UPat(Ops.CAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => {
    const fn = cast(from.dtype, to.dtype)
    return fn ? [`(${fn}`, ...ctx.var(from), ')'] : ctx.var(from)
  }),
])
export class WASMRenderer extends Renderer {
  override has_local = false
  override has_shared = false
  override extra_matcher = new PatternMatcher([
    new UPat(Ops.MULACC, undefined, [UPat.var('a'), UPat.var('b'), UPat.var('c')]).named('mulacc').fn(({ a, b, c }) => a.mul(b).add(c)),
    new UPat(Ops.MAX, dtypes.ints, [UPat.var('a'), UPat.var('b')]).fn(({ a, b }) => (a.gt(b)).where(a, b)),
  ])

  string_rewrite = string_rewrite
  r?: Map<UOp, string[]>
  first = (uop: UOp): string => this.r!.get(uop)!.join(' ')
  var = (uop: UOp): string[] => {
    if (prefixes.has(uop.op)) return [`(local.get ${this.first(uop)})`]
    return [...this.r!.get(uop)!]
  }
  override render = (name: string, uops: UOp[]) => {
    let lines: string[] = []
    this.r = new Map<UOp, string[]>()
    const defs: string[] = []
    for (const uop of uops) {
      // TODO add these into one
      if (uop.op === Ops.DEFINE_GLOBAL) {
        this.r.set(uop, [`$${prefixes.get(uop.op)}${uop.arg}`])
        defs.push(`(param ${this.first(uop)} i32)`)
        continue
      }
      if (uop.op === Ops.RANGE) {
        this.r.set(uop, [`$${prefixes.get(uop.op)}${uop.arg}`])
        defs.push(`(local ${this.first(uop)} ${get_dtype(uop.dtype)})`)
      }
      const str = this.string_rewrite.rewrite(uop, this)
      if (!str) throw new Error(`No matcher for ${uop}`)

      // add to lines for these Ops, others add to context
      if ([Ops.RANGE, Ops.ENDRANGE, Ops.STORE].includes(uop.op)) {
        lines = [...lines, '', `;; ${uop.op}`, ...str]
      } else if ([Ops.INDEX, Ops.LOAD, ...GroupOp.ALU, Ops.CAST, Ops.CONST, Ops.GEP].includes(uop.op)) {
        this.r.set(uop, str)
      } else throw new Error(`Invalid op ${uop.op}`)
    }

    lines = [`(module`, `(import "env" "memory" (memory 1))`, `(func (export "${name}")`, ...defs, ...lines, ')', ')']
    let res = '', indent = 0
    for (const line of lines) {
      const change = line.split('(').length - line.split(')').length
      if (change < 0) indent += change
      res += '  '.repeat(indent) + line + '\n'
      if (change > 0) indent += change
    }
    console.log(res)
    this.r = undefined
    return res
  }
}
