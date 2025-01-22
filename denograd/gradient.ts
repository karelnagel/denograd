import { dtypes, sum_acc_dtype } from './dtype.ts'
import { argsort, cache_fn, zip } from './helpers.ts'
import { add, type sint, sub } from './mod.ts'
import { Ops, PatternMatcher, type UOp, UPat } from './ops.ts'

export const reduce_gradient = (ctx: UOp, ret: UOp) => {
  if (ret.arg[0] === Ops.ADD) return [ctx.expand(ret.src[0].shape)]
  if (ret.arg[0] === Ops.MAX) {
    const max_is_1s = ret.src[0].ne(ret.expand(ret.src[0].shape)).ne(ret.src[0].const_like(1).cast(dtypes.bool)).cast(ctx.dtype)
    const div = max_is_1s.r(Ops.ADD, ret.arg[1]).expand(ret.src[0].shape)
    return [(max_is_1s.div(div)).mul(ctx.expand(ret.src[0].shape))]
  }
  if (ret.arg[0] === Ops.MUL) return [ctx.mul(ret).expand(ret.src[0].shape).div(ret.src[0])]
}
// ctx is grad_output
export const pm_gradient = new PatternMatcher<UOp, (UOp | undefined)[] | undefined>([
  [new UPat(Ops.CAST).named('ret'), ({ ctx, ret }) => [ctx.cast(ret.src[0].dtype)]],
  [new UPat(Ops.RECIP).named('ret'), ({ ctx, ret }) => [ctx.neg().mul(ret).mul(ret)]],
  [new UPat(Ops.SIN).named('ret'), ({ ctx, ret }) => [ret.src[0].sub(Math.PI / 2, true).sin().mul(ctx)]],
  [new UPat(Ops.LOG2).named('ret'), ({ ctx, ret }) => [ctx.div(ret.src[0].mul(Math.log(2)))]],
  [new UPat(Ops.EXP2).named('ret'), ({ ctx, ret }) => [ret.mul(ctx).mul(Math.log(2))]],
  [new UPat(Ops.SQRT).named('ret'), ({ ctx, ret }) => [ctx.div(ret.mul(2))]],
  [new UPat((Ops.CMPLT, Ops.CMPNE)), () => [undefined, undefined]],
  [new UPat(Ops.ADD), ({ ctx }) => [ctx, ctx]],
  [new UPat(Ops.MAX).named('ret'), ({ ctx, ret }) => [
    (ret.src[0].gt(ret.src[1])).where(ctx, (ret.src[0].ne(ret.src[1])).where(ctx.const_like(0), ctx.mul(0.5))),
    (ret.src[0].lt(ret.src[1])).where(ctx, (ret.src[0].ne(ret.src[1])).where(ctx.const_like(0), ctx.mul(0.5))),
  ]],
  [new UPat(Ops.MUL).named('ret'), ({ ctx, ret }) => [ret.src[1].mul(ctx), ret.src[0].mul(ctx)]],
  [new UPat(Ops.WHERE).named('ret'), ({ ctx, ret }) => [undefined, ret.src[0].where(ctx, ctx.const_like(0)), ret.src[0].where(ctx.const_like(0), ctx)]],
  [new UPat(Ops.REDUCE_AXIS).named('ret'), ({ ctx, ret }) => reduce_gradient(ctx, ret)],
  [new UPat(Ops.CONTIGUOUS), ({ ctx }) => [ctx]],
  [new UPat(Ops.RESHAPE).named('ret'), ({ ctx, ret }) => [ctx.reshape(ret.src[0].shape)]],
  [new UPat(Ops.PERMUTE).named('ret'), ({ ctx, ret }) => [ctx.permute(argsort(ret.arg))]],
  [new UPat(Ops.PAD).named('ret'), ({ ctx, ret }) => [ctx.shrink(zip(ret.src[0].shape, ret.arg as [sint, sint][]).map(([s, p]) => [p[0], add(s, p[0])]))]],
  [new UPat(Ops.SHRINK).named('ret'), ({ ctx, ret }) => [ctx.pad(zip(ret.src[0].shape, ret.arg as [sint, sint][]).map(([s, p]) => [p[0], sub(s, p[1])]))]],
  [new UPat(Ops.STRIDE).named('ret'), ({ ctx, ret }) => [ret.arg.every((x: number) => [-1, 1].includes(x)) ? ctx.stride(ret.arg) : undefined]],
  // TODO: this cast can be removed by putting the casts around the EXPAND
  [new UPat(Ops.EXPAND).named('ret'), ({ ctx, ret }) => [ctx.cast(sum_acc_dtype(ctx.dtype)).r(Ops.ADD, [...zip(ret.src[0].shape, ret.arg).entries()].filter(([i, [si, so]]) => si !== so).map(([i]) => i)).cast(ctx.dtype)]],
  // there's no gradient for...is this ASSIGN?
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER), new UPat(Ops.BUFFER_VIEW)]), () => [undefined, undefined]],
  // also no gradient for bitcast
  [new UPat(Ops.BITCAST), ({ ctx }) => [undefined]],
])

// copied from tensor.py, get relevant toposort of gradients
const is_in_target_path = cache_fn((x: UOp, targets: Set<UOp>): boolean => x.src.some((u) => targets.has(u) || is_in_target_path(u, targets)))
export const _deepwalk = (root: UOp, targets: Set<UOp>): UOp[] => {
  const _walk = (node: UOp, visited: Set<UOp>): UOp[] => {
    visited.add(node)
    if (node.op === Ops.DETACH) return []
    let res: UOp[] = []
    if (is_in_target_path(node, visited)) {
      for (const i of node.src) {
        if (!visited.has(i)) res = [...res, ..._walk(i, visited)]
      }
      res.push(node)
    }
    return res
  }
  return _walk(root, new Set())
}
export const compute_gradient = (root: UOp, root_grad: UOp, targets: Set<UOp>): Map<UOp, UOp> => {
  const grads = new Map([[root, root_grad]])
  for (const t0 of _deepwalk(root, targets).toReversed()) {
    if (!grads.has(t0)) continue
    const lgrads = pm_gradient.rewrite(t0, grads.get(t0))
    if (lgrads === undefined) throw new Error(`failed to compute gradient for ${t0.op}\n\nin ${t0.toString().slice(0, 1000)}...`)
    if (lgrads.length !== t0.src.length) throw new Error(`got {len(lgrads)} gradient, expected {len(t0.src)}`)
    for (const [k, v] of zip(t0.src, lgrads)) {
      if (v === undefined) continue
      if (grads.has(k)) grads.set(k, grads.get(k)!.add(v))
      else grads.set(k, v)
    }
  }
  return grads
}
