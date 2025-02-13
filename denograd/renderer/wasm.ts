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
const variable = (uop: UOp, ctx: WASMRenderer): string[] => {
  if ([Ops.CONST].includes(uop.op)) return [`(${get_dtype(uop.dtype)}.const ${uop.arg})`]
  if ([Ops.RANGE, Ops.DEFINE_GLOBAL].includes(uop.op)) return [`(local.get ${ctx.first(uop)})`]
  return [...ctx.get(uop)]
}
// TODO: handle NaNs and Infinity
// TODO: handle uints correcly, should use '..._u' functions for those
export class WASMRenderer extends Renderer {
  override has_local = false
  override has_shared = false
  override extra_matcher = new PatternMatcher([])

  string_rewrite = new PatternMatcher<WASMRenderer, string[]>([
    // ALU
    new UPat([Ops.ADD, Ops.SQRT, Ops.NEG, Ops.MUL, Ops.IDIV, Ops.XOR, Ops.SHL, Ops.SHR, Ops.OR, Ops.AND, Ops.SUB]).named('alu').fn(({ alu, ctx }) => [`(${get_dtype(alu.dtype)}.${alu.op.name.toLowerCase()}`, ...alu.src.flatMap((a) => variable(a, ctx)), ')']),
    // new UPat([...alu_overrides.keys()]).named('alu').fn(({ alu, ctx }) => [...alu.src.flatMap((a) => [...ctx.get(a), '']), `${get_dtype(alu.dtype)}.${alu_overrides.get(alu.op)}`]),
    // new UPat([Ops.WHERE], undefined, [UPat.var('a'), UPat.var('b'), UPat.var('cond')]).named('alu').fn(({ alu, ctx, a, b, cond }) => [...ctx.get(a), ...ctx.get(b), ...ctx.get(cond), 'select']),
    // new UPat([Ops.MULACC], undefined, [UPat.var('a'), UPat.var('b'), UPat.var('c')]).named('alu').fn(({ alu, ctx, a, b, c }) => [...ctx.get(a), ...ctx.get(b), `${get_dtype(alu.dtype)}.mul`, ...ctx.get(c), `${get_dtype(alu.dtype)}.add`]),
    // new UPat([Ops.MAX], dtypes.floats).named('alu').fn(({ alu, ctx }) => [...alu.src.flatMap((a) => [...ctx.get(a), '']), `${get_dtype(alu.dtype)}.max`]),
    // new UPat([Ops.MAX], dtypes.ints, [UPat.var('a'), UPat.var('b')]).named('alu').fn(({ alu, ctx, a, b }) => [...ctx.get(a), ...ctx.get(b), 'i32.gt_s', '(if (result i32)', '(then', ...ctx.get(a), ')', '(else', ...ctx.get(b), ')', ')']),
    // new UPat([Ops.RECIP], undefined, [UPat.var('x')]).named('alu').fn(({ alu, x, ctx }) => ['f32.const 1.0', ...ctx.get(x), 'f32.div']),
    // TODO: EXP2, LOG2, SIN

    new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]).named('index').fn(({ index, buf, idx, ctx }) => [
      `(${get_dtype(idx.dtype)}.add`,
      `(${get_dtype(idx.dtype)}.mul`,
      ...variable(idx, ctx),
      `(${get_dtype(idx.dtype)}.const ${(index.dtype as PtrDType).itemsize})`,
      `)`,
      ...variable(buf, ctx),
      `)`,
    ]),
    new UPat(Ops.LOAD).named('load').fn(({ load, ctx }) => [`(${get_dtype(load.dtype)}.load`, ...ctx.get(load.src[0]), ')']),
    new UPat(Ops.STORE).named('store').fn(({ store, ctx }) => [`(${get_dtype(store.src[1].dtype)}.store`, ...ctx.get(store.src[0]), ...ctx.get(store.src[1]), ')']),
    new UPat(Ops.RANGE).named('range').fn(({ ctx, range }) => [
      `(block $block${range.arg}`,
      `(loop $loop${range.arg}`,
      `(br_if $block${range.arg}`,
      `(${get_dtype(range.dtype)}.eq`,
      ...variable(range, ctx),
      ...variable(range.src[1], ctx),
      `)`,
      `)`,
    ]),
    new UPat(Ops.ENDRANGE, undefined, [new UPat(Ops.RANGE).named('range')]).named('endrange').fn(({ endrange, range, ctx }) => [
      `(br $loop${range.arg}`,
      `(local.set ${ctx.first(range)}`,
      `(${get_dtype(range.dtype)}.add`,
      ...variable(range, ctx),
      `(${get_dtype(range.dtype)}.const 1)`,
      `)`,
      `)`,
      `)`,
      ')',
      ')',
    ]),
    new UPat(Ops.CAST, undefined, [UPat.var('from')]).named('to').fn(({ from, to, ctx }) => [`(${cast(from.dtype, to.dtype)}`, ...ctx.get(from), ')']),
  ])
  r?: Map<UOp, string[]>
  get = (uop: UOp): string[] => this.r!.get(uop)!
  first = (uop: UOp): string => this.r!.get(uop)!.join(' ')
  override render = (name: string, uops: UOp[]) => {
    let lines: string[] = []
    this.r = new Map<UOp, string[]>()
    const defs: string[] = []
    for (const uop of uops) {
      if (uop.op === Ops.DEFINE_GLOBAL) {
        this.r.set(uop, [`$${prefixes.get(uop.op)}${uop.arg}`])
        defs.push(`(param ${this.first(uop)} ${get_dtype(uop.dtype)})`)
        continue
      }
      if (uop.op === Ops.RANGE) {
        this.r.set(uop, [`$${prefixes.get(uop.op)}${uop.arg}`])
        defs.push(`(local ${this.first(uop)} ${get_dtype(uop.dtype)})`)
      }
      if (uop.op === Ops.CONST) {
        this.r.set(uop, [`$${prefixes.get(uop.op)}${uop.arg}`])
        continue
      }

      const str = this.string_rewrite.rewrite(uop, this)
      if (!str) throw new Error(`No matcher for ${uop}`)

      // add to lines for these Ops, others add to context
      if ([Ops.RANGE, Ops.ENDRANGE, Ops.STORE].includes(uop.op)) {
        lines = [...lines, '', `;; ${uop.op}`, ...str]
      } else if ([Ops.INDEX, Ops.LOAD, ...GroupOp.ALU, Ops.CAST].includes(uop.op)) {
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
