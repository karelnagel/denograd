import { type DType, dtypes, type PtrDType } from '../dtype.ts'
import { GroupOp, Ops, PatternMatcher, type UOp, UPat } from '../ops.ts'
import { Renderer } from './index.ts'

const prefixes = new Map([[Ops.DEFINE_GLOBAL, 'data'], [Ops.RANGE, 'ridx'], [Ops.CONST, 'const']])
const _render_dtype = new Map([[dtypes.int32, 'i32'], [dtypes.int64, 'i64'], [dtypes.uint32, 'i32'], [dtypes.uint64, 'i64'], [dtypes.float32, 'f32'], [dtypes.float64, 'f64'], [dtypes.bool, 'i32']])
const get_dtype = (dtype: DType) => {
  const res = _render_dtype.get(dtype.base)
  if (!res) throw new Error(`WASM doesn't support ${dtype} dtype`)
  return res
}
const cast = (from: DType, to: DType): string => {
  const a = get_dtype(from), b = get_dtype(to), sign = dtypes.is_unsigned(from) || dtypes.is_unsigned(to) ? 'u' : 's'
  if (a === 'i32' && b === 'i64') return `${b}.extend_${a}_${sign}`
  if (a === 'i64' && b === 'i32') return `${b}.wrap_${a}`
  if (a === 'f32' && b === 'f64') return `${b}.promote_${a}`
  if (a === 'f64' && b === 'f32') return `${b}.demote_${a}`
  if (['f32', 'f64'].includes(a) && ['i32', 'i64'].includes(b)) return `${b}.convert_${a}_${sign}`
  if (['i32', 'i64'].includes(a) && ['f32', 'f64'].includes(b)) return `${b}.trunc_${a}_${sign}`
  throw new Error(`Can't cast ${from} to ${to}`)
}
const alu_overrides = new Map([[Ops.MOD, 'rem_s'], [Ops.CMPLT, 'lt_s'], [Ops.CMPNE, 'ne'], [Ops.FDIV, 'div']])
// TODO: handle NaNs and Infinity
// TODO: handle uints correcly, should use '..._u' functions for those
export class WASMRenderer extends Renderer {
  override has_local = false
  override has_shared = false
  override extra_matcher = new PatternMatcher([])

  string_rewrite = new PatternMatcher<WASMRenderer, string[]>([
    new UPat(Ops.CONST).named('c').fn(({ ctx, c }) => [`(local.set ${ctx.first(c)} (${get_dtype(c.dtype)}.const ${c.arg}))`]),
    // ALU
    new UPat([Ops.ADD, Ops.SQRT, Ops.NEG, Ops.MUL, Ops.IDIV, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.SUB]).named('alu').fn(({ alu, ctx }) => [...alu.src.flatMap((a) => [...ctx.get(a), '']), `${get_dtype(alu.dtype)}.${alu.op.name.toLowerCase()}`]),
    new UPat([...alu_overrides.keys()]).named('alu').fn(({ alu, ctx }) => [...alu.src.flatMap((a) => [...ctx.get(a), '']), `${get_dtype(alu.dtype)}.${alu_overrides.get(alu.op)}`]),
    new UPat([Ops.WHERE], undefined, [UPat.var('a'), UPat.var('b'), UPat.var('cond')]).named('alu').fn(({ alu, ctx, a, b, cond }) => [...ctx.get(a), ...ctx.get(b), ...ctx.get(cond), 'select']),
    new UPat([Ops.MULACC], undefined, [UPat.var('a'), UPat.var('b'), UPat.var('c')]).named('alu').fn(({ alu, ctx, a, b, c }) => [...ctx.get(a), ...ctx.get(b), `${get_dtype(alu.dtype)}.mul`, ...ctx.get(c), `${get_dtype(alu.dtype)}.add`]),
    new UPat([Ops.MAX], dtypes.floats).named('alu').fn(({ alu, ctx }) => [...alu.src.flatMap((a) => [...ctx.get(a), '']), `${get_dtype(alu.dtype)}.max`]),
    new UPat([Ops.MAX], dtypes.ints, [UPat.var('a'), UPat.var('b')]).named('alu').fn(({ alu, ctx, a, b }) => [...ctx.get(a), ...ctx.get(b), 'i32.gt_s', '(if (result i32)', '(then', ...ctx.get(a), ')', '(else', ...ctx.get(b), ')', ')']),
    new UPat([Ops.RECIP], undefined, [UPat.var('x')]).named('alu').fn(({ alu, x, ctx }) => ['f32.const 1.0', ...ctx.get(x), 'f32.div']),
    // TODO: EXP2, LOG2, SIN

    new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]).named('index').fn(({ index, buf, idx, ctx }) => [
      `local.get ${ctx.first(buf)}`,
      `local.get ${ctx.first(idx)}`,
      `${get_dtype(idx.dtype)}.const ${(index.dtype as PtrDType).itemsize}`,
      `${get_dtype(idx.dtype)}.mul`,
      `${get_dtype(idx.dtype)}.add`,
    ]),
    new UPat(Ops.LOAD).named('load').fn(({ load, ctx }) => [...ctx.get(load.src[0]), `${get_dtype(load.dtype)}.load`]),
    new UPat(Ops.STORE).named('store').fn(({ store, ctx }) => [...ctx.get(store.src[0]), '', ...ctx.get(store.src[1]), `${get_dtype(store.src[1].dtype)}.store`]),
    new UPat(Ops.RANGE).named('range').fn(({ ctx, range }) => [
      `(block $block${range.arg}`,
      `(loop $loop${range.arg}`,
      `local.get ${ctx.first(range)}`,
      `local.get ${ctx.first(range.src[1])}`,
      `${get_dtype(range.dtype)}.eq`,
      `br_if $block${range.arg}`,
    ]),
    new UPat(Ops.ENDRANGE, undefined, [new UPat(Ops.RANGE).named('range')]).named('endrange').fn(({ endrange, range, ctx }) => [
      `local.get ${ctx.first(range)}`,
      `${get_dtype(range.dtype)}.const 1`,
      `${get_dtype(range.dtype)}.add`,
      `local.set ${ctx.first(range)}`,
      `br $loop${range.arg}`,
      ')',
      ')',
    ]),
    new UPat(Ops.CAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => [...ctx.get(from), cast(from.dtype, to.dtype)]),
  ])
  r?: Map<UOp, string[]>
  get = (uop: UOp): string[] => this.r!.get(uop)!
  first = (uop: UOp): string => this.r!.get(uop)![0]
  override render = (name: string, uops: UOp[]) => {
    let lines: string[] = []
    this.r = new Map<UOp, string[]>()
    const defs: string[] = []
    for (const uop of uops) {
      if (prefixes.has(uop.op)) {
        this.r.set(uop, [`$${prefixes.get(uop.op)}${uop.arg}`])
        defs.push(`(${uop.op === Ops.DEFINE_GLOBAL ? 'param' : 'local'} ${this.first(uop)} ${get_dtype(uop.dtype)})`)
        if (uop.op === Ops.DEFINE_GLOBAL) continue
      }

      const str = this.string_rewrite.rewrite(uop, this)
      if (!str) throw new Error(`No matcher for ${uop}`)

      // add to lines for these Ops, others add to context
      if ([Ops.CONST, Ops.RANGE, Ops.ENDRANGE, Ops.STORE].includes(uop.op)) {
        lines = [...lines, '', `;; ${uop.op}`, ...str]
      } else if ([Ops.INDEX, Ops.LOAD, ...GroupOp.ALU, Ops.CAST].includes(uop.op)) {
        this.r.set(uop, str)
      } else throw new Error(`Invalid op ${uop.op}`)
    }

    lines = [`(module`, `(import "env" "memory" (memory 1))`, `(func (export "${name}")`, ...defs, ...lines, ')', ')']
    let res = '', indent = 0
    for (const line of lines) {
      res += '  '.repeat(indent) + line + '\n'
      indent += line.split('(').length - line.split(')').length
    }
    console.log(res)
    this.r = undefined
    return res
  }
}
