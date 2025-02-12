import { type DType, dtypes, type PtrDType } from '../dtype.ts'
import { Ops, PatternMatcher, UOp, UPat } from '../ops.ts'
import { Renderer } from './index.ts'

export class WASMRenderer extends Renderer {
  override has_local = false
  override has_shared = false
  render_dtype = new Map([[dtypes.int32, 'i32'], [dtypes.float32, 'f32'], [dtypes.int64, 'i64'], [dtypes.float64, 'f64']])
  override extra_matcher = new PatternMatcher([])

  string_rewrite = new PatternMatcher<Map<UOp, string>, string>([
    // new UPat(Ops.DEFINE_GLOBAL).named('global').fn(({ global, ctx }) => `(param ${ctx.get(global)} i32)`),
    new UPat(Ops.CONST).named('c').fn(({ ctx, c }) => `
    ;; const
    ${this.render_dtype.get(c.dtype)}.const ${c.arg}
    local.set ${ctx.get(c)}
    `),
    new UPat(Ops.INDEX).named('index').fn(({ index, ctx }) => `
        ;; index
        local.get ${ctx.get(index.src[0])}
        local.get ${ctx.get(index.src[1])}
        i32.const ${(index.dtype as PtrDType).size}
        ${this.render_dtype.get(index.dtype.base)}.mul
        ${this.render_dtype.get(index.dtype.base)}.add
        `),
    new UPat(Ops.LOAD).named('load').fn(({ load, ctx }) => `
        ${ctx.get(load.src[0])}

        ;;load
        ${this.render_dtype.get(load.dtype)}.load
    `),
    new UPat(Ops.ADD).named('add').fn(({ add, ctx }) => `
        ${ctx.get(add.src[0])}

        ${ctx.get(add.src[1])}

        ;; add 
        ${this.render_dtype.get(add.dtype)}.add
    `),
    new UPat(Ops.STORE).named('store').fn(({ store, ctx }) => `
        ${ctx.get(store.src[0])}

        ${ctx.get(store.src[1])}

        ;; store
        ${this.render_dtype.get(store.src[1].dtype)}.store
    `),
    new UPat(Ops.RANGE).named('range').fn(({ ctx, range }) => `
    ;; range
    block $block${range.arg}
      loop $loop${range.arg}
        ;; checking if loop should break
        local.get ${ctx.get(range)}
        local.get ${ctx.get(range.src[1])}
        i32.eq
        br_if $block${range.arg}`),
    new UPat(Ops.ENDRANGE, undefined, [new UPat(Ops.RANGE).named('range')]).named('endrange').fn(({ endrange, range, ctx }) => `
        ;; endrange
        local.get ${ctx.get(range)}
        i32.const 1
        i32.add
        local.set ${ctx.get(range)}

        br $loop${range.arg}
      end
    end`),
  ])
  override render = (name: string, uops: UOp[]) => {
    const lines: string[] = []
    const ctx = new Map<UOp, string>()
    const defs: { type: 'param' | 'local'; name: string; dtype: DType }[] = []
    for (const uop of uops) {
      if (uop.op === Ops.DEFINE_GLOBAL) {
        ctx.set(uop, `$data${uop.arg}`)
        defs.push({ type: 'param', name: ctx.get(uop)!, dtype: uop.dtype })
        continue
      }
      if (uop.op === Ops.RANGE) {
        ctx.set(uop, `$ridx${uop.arg}`)
        defs.push({ type: 'local', name: ctx.get(uop)!, dtype: uop.dtype })
      }
      if (uop.op === Ops.CONST) {
        ctx.set(uop, `$const${uop.arg}`)
        defs.push({ type: 'local', name: ctx.get(uop)!, dtype: uop.dtype })
      }

      const str = this.string_rewrite.rewrite(uop, ctx)
      if (!str) throw new Error(`No matcher for ${uop}`)

      // we write lines for these Ops, else add to context
      if ([Ops.CONST, Ops.RANGE, Ops.ENDRANGE, Ops.STORE].includes(uop.op)) {
        lines.push(str)
      } else {
        ctx.set(uop, str)
      }
    }

    let res = `(module
  (import "env" "memory" (memory 1))

  (func (export "${name}")
    ${defs.map((d) => `(${d.type} ${d.name} ${this.render_dtype.get(d.dtype.base)})`).join('\n    ')}

    ${lines.join('\n    ')}
  )
)`
    console.log(res)
    return res
  }
}
