import { dtypes } from '../dtype.ts'
import { assert, isEq, range } from '../helpers.ts'
import { getNumberEnv, isNone, isNotNone, mergeMaps, zip } from '../helpers.ts'
import { graphRewrite, idiv, mod, mul, Ops, simplifyValid, type sint, splitUOp, symbolicFlat, UOp, uop_given_valid, type Variable } from '../ops.ts'
import { strides_for_shape, View } from './view.ts'

const views_to_indexed_uops = (views: View[], _idxs?: UOp[]): [UOp, UOp] => {
  let [idx, valid] = views.at(-1)!.to_indexed_uops(_idxs)
  for (let view of views.slice(0, -1).toReversed()) {
    view = view.minify()
    let [acc, idxs] = [1 as sint, [] as UOp[]]
    for (const d of view.shape.toReversed()) {
      idxs.push(mod(idiv(idx, acc), d) as UOp)
      acc = mul(acc, d)
    }
    ;[idx, valid] = view.to_indexed_uops(idxs.toReversed(), valid)
  }
  return [idx, valid]
}
const views_to_real_strides = (views: View[], ignore_valid = false): (undefined | sint)[] => {
  // NOTE: if a stride is not always valid, it will be None
  if (views.length === 1 && isNone(views.at(-1)!.mask)) return views.at(-1)!.strides
  let ret: (undefined | sint)[] = range(views.at(-1)!.shape.length).map((x) => undefined)
  let [idx, valid] = views_to_indexed_uops(views).map((u) => graphRewrite(u, symbolicFlat))
  // TODO: always apply these in to_indexed_uops?
  const newvalid = simplifyValid(valid)
  if (isNotNone(newvalid)) valid = newvalid
  const newidx = uop_given_valid(valid, idx)
  if (isNotNone(newidx)) idx = graphRewrite(newidx, symbolicFlat)
  for (const c of splitUOp(idx, Ops.ADD)) {
    if (c.op === Ops.RANGE) ret[c.arg] = 1
    if (c.op === Ops.MUL && c.src[0].op === Ops.RANGE && c.src[1].op === Ops.CONST) ret[c.src[0].arg] = c.src[1].arg
    if (c.op === Ops.MUL && c.src[1].op === Ops.RANGE && c.src[0].op === Ops.CONST) ret[c.src[1].arg] = c.src[0].arg
  }
  const used_ranges = [...idx.toposort].filter((x) => x.op === Ops.RANGE).map((x) => x.arg)
  ret = ret.map((x, i) => used_ranges.includes(i) ? x : 0)
  if (!ignore_valid) {
    for (const masked_axis of [...valid.toposort].filter((x) => x.op === Ops.RANGE).map((x) => x.arg)) ret[masked_axis] = undefined
  }
  return ret
}
export class ShapeTracker {
  views: View[]
  constructor(views: View[]) {
    this.views = views
  }
  __add__ = (st: ShapeTracker): ShapeTracker => {
    let ret = new ShapeTracker(this.views)
    for (const v of st.views) ret = new ShapeTracker([...ret.views, v]).simplify() // one view at a time = better simplification
    return ret
  }
  invert = (out_shape: sint[]): undefined | ShapeTracker => {
    const inverted_views: View[] = []
    for (const [v, s] of zip(this.views.toReversed(), [...this.views.toReversed().slice(1).map((x) => x.shape), out_shape])) {
      const inverted = v.invert(s)
      if (isNone(inverted)) return undefined
      inverted_views.push(inverted)
    }
    return new ShapeTracker(inverted_views).reshape(out_shape)
  }
  static from_shape = (shape: sint[]) => new ShapeTracker([View.create(shape)])

  // deno-fmt-ignore
  get contiguous() {return this.views.length === 1 && this.views[0].contiguous}
  get consecutive() {
    const v = this.views[0]
    return this.views.length === 1 && isNone(v.mask) && isEq(v.strides, strides_for_shape(v.shape))
  }
  // deno-fmt-ignore
  get shape() {return this.views.at(-1)!.shape}
  // deno-fmt-ignore
  get size() {return this.views.at(-1)!.size()}

  reduce = (axis: number[]) => this.shape.map((s, i) => axis.includes(i) ? 1 : s)

  to_uop = () => new UOp({ op: Ops.VIEW, dtype: dtypes.void, src: [], arg: this })

  to_indexed_uops = (_idxs?: UOp[]): [UOp, UOp] => views_to_indexed_uops(this.views, isNotNone(_idxs) ? _idxs : undefined)

  // to_indexed_uops = (_idxs?: UOp[]): [UOp, UOp] => {
  //   let [idx, valid] = this.views.at(-1)!.to_indexed_uops(_idxs)
  //   for (let view of this.views.slice(0, -1).toReversed()) {
  //     view = view.minify()
  //     let [acc, idxs] = [1 as sint, [] as UOp[]]
  //     for (const d of view.shape.toReversed()) {
  //       idxs.push((idx.idiv(acc)).mod(d))
  //       acc = mul(acc, d)
  //     }
  //     ;[idx, valid] = view.to_indexed_uops(idxs.toReversed(), valid)
  //   }
  //   return [idx, valid]
  // }
  real_size = (): number => {
    if (this.shape.includes(0)) return 0
    const [idx, valid] = this.to_indexed_uops()
    if (!valid.vmax) return 0
    assert(idx.vmax < 1e12, `real_size broken for ${self}`)
    return Math.floor(idx.vmax + 1)
  }
  vars = (): Variable[] => [...new Set(this.views.flatMap((v) => v.vars()))]

  get var_vals(): Map<Variable, number> {
    return mergeMaps(this.vars().map((v) => new Map([v.unbind()])))
  }

  unbind = (): [ShapeTracker, Map<Variable, number>] => {
    const un = this.views.map((v) => v.unbind())
    const [unbound_views, var_vals] = [un.map((x) => x[0]), un.map((v) => v[1])]
    return [new ShapeTracker(unbound_views), mergeMaps(var_vals)]
  }
  //   # NOTE: if a stride is not always valid, it will be None
  real_strides = (ignore_valid = false) => views_to_real_strides(this.views, ignore_valid)

  // real_strides = (ignore_valid = false): (undefined | sint)[] => {
  //   if (this.views.length === 1 && isNone(this.views.at(-1)?.mask)) return this.views.at(-1)!.strides
  //   let ret: (sint | undefined)[] = range(this.shape.length).map((x) => undefined)
  //   let [idx, valid] = this.to_indexed_uops().map((u) => graphRewrite(u, symbolicFlat))
  //   //     # TODO: always apply these in to_indexed_uops?
  //   const newvalid = simplifyValid(valid)
  //   if (isNotNone(newvalid)) valid = newvalid
  //   const newidx = uop_given_valid(valid, idx)
  //   if (isNotNone(newidx)) idx = graphRewrite(newidx, symbolicFlat)
  //   for (const c of splitUOp(idx, Ops.ADD)) {
  //     if (c.op === Ops.RANGE) ret[c.arg[0]] = 1
  //     if (c.op === Ops.MUL && c.src[0].op === Ops.RANGE && c.src[1].op === Ops.CONST) ret[c.src[0].arg[0]] = c.src[1].arg
  //     if (c.op === Ops.MUL && c.src[1].op === Ops.RANGE && c.src[0].op === Ops.CONST) ret[c.src[1].arg[0]] = c.src[0].arg
  //   }
  //   const used_ranges = [...idx.sparents().keys().filter((x) => x.op === Ops.RANGE).map((x) => x.arg[0])]
  //   ret = ret.map((x, i) => used_ranges.includes(i) ? x : 0)
  //   if (!ignore_valid) {
  //     for (const masked_axis of valid.sparents().keys().filter((x) => x.op === Ops.RANGE).map((x) => x.arg[0])) ret[masked_axis] = undefined
  //   }
  //   return ret
  // }
  unit_stride_axes = (ignore_valid = false): number[] => [...this.real_strides(ignore_valid).entries()].filter(([i, st]) => st === 1).map(([i, st]) => i)

  axis_is_masked = (axis: number): boolean => {
    const [_, valid] = this.to_indexed_uops()
    return [...graphRewrite(valid, symbolicFlat).toposort].filter((x) => x.op === Ops.RANGE).map((x) => x.arg).includes(axis)
  }
  simplify = (): ShapeTracker => {
    const new_view = this.views.at(-2)?.__add__(this.views.at(-1)!)
    if (this.views.length >= 2 && isNotNone(new_view)) return new ShapeTracker([...this.views.slice(0, -2), new_view]).simplify()
    return this
  }
  //   # *** under this line are the movement ops ***

  pad = (arg: [sint, sint][]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.pad(arg)])
  shrink = (arg: [sint, sint][]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.shrink(arg)])
  expand = (new_shape: sint[]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.expand(new_shape)])
  permute = (axis: number[]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.permute(axis)])
  stride = (multi: number[]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.stride(multi)])

  reshape = (new_shape: sint[]): ShapeTracker => {
    const new_view = this.views.at(-1)?.reshape(new_shape)
    if (getNumberEnv('MERGE_VIEW', 1) && isNotNone(new_view)) return new ShapeTracker([...this.views.slice(0, -1), new_view])
    return new ShapeTracker([...this.views, View.create(new_shape)])
  }
}
