import { dtypes } from '../dtype.ts'
import { env, withEnv } from '../env/index.ts'
import { cache_fn, get_key, gt, is_eq, list_str, lt, range, WeakValueMap } from '../helpers.ts'
import { merge_maps, zip } from '../helpers.ts'
import { graph_rewrite, Ops, simplify_valid, type sint, sint_to_uop, split_uop, sym, symbolic_flat, UOp, uop_given_valid, type Variable } from '../ops.ts'
import { strides_for_shape, unravel, View } from './view.ts'

export const overflow = (u: UOp) => gt(u.vmax, dtypes.max(dtypes.int)) || lt(u.vmin, dtypes.min(dtypes.int))

// If a node overflow, its srcs need to be checked to see if this overflow is the result of an ALU operation,
// or that the node simply inherits the dtype from srcs. Upcast is either `Ops.CAST`+`replace` or just `replace`.
export const upcast = (u: UOp): UOp => {
  const srcs = u.src.map((_src) => upcast(_src))
  if (u.dtype.scalar() === dtypes.int) {
    const dtype = u.dtype.count > 1 ? dtypes.int64.vec(u.dtype.count) : dtypes.int64
    const upcasted = u.replace({ dtype: dtype, src: srcs.map((_src) => _src.cast(dtype)) })
    if (overflow(u)) return upcasted
    // Check the original src, new srcs has Ops.CAST whose vmin, vmax change the real bounds
    // Cast back is required because if the node is in range, siblings would never be upcasted
    if (u.src.some((src) => overflow(src))) return upcasted.cast(u.dtype)
  }
  return u.replace({ src: srcs })
}
// pooling op may overflow before folding causing unnecessary upcast
export const folded_upcast = (u: UOp) => {
  return withEnv({ TRACK_MATCH_STATS: 0 }, () => upcast(graph_rewrite(u, sym, new Map())))
}

const views_to_indexed_uops = cache_fn((views: View[], _idxs?: UOp[]): [UOp, UOp] => {
  let [idx, valid] = views.at(-1)!.to_indexed_uops(_idxs)
  for (let view of views.slice(0, -1).toReversed()) {
    view = view.minify()
    ;[idx, valid] = view.to_indexed_uops(unravel(view.shape, idx).map((i) => sint_to_uop(i)), valid)
  }
  return [idx, valid]
})

const views_to_real_strides = cache_fn((views: View[], ignore_valid = false): (undefined | sint)[] => {
  // NOTE: if a stride is not always valid, it will be None
  if (views.length === 1 && views.at(-1)!.mask === undefined) return views.at(-1)!.strides
  let ret: (undefined | sint)[] = range(views.at(-1)!.shape.length).map((x) => undefined)
  let [idx, valid] = views_to_indexed_uops(views).map((u) => graph_rewrite(u, symbolic_flat))
  // TODO: always apply these in to_indexed_uops?
  const newvalid = simplify_valid(valid)
  if (newvalid !== undefined) valid = newvalid
  const newidx = uop_given_valid(valid, idx)
  if (newidx !== undefined) idx = graph_rewrite(newidx, symbolic_flat)
  for (const c of split_uop(idx, Ops.ADD)) {
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
})

export class ShapeTracker {
  key: string
  static cache = new WeakValueMap<string, ShapeTracker>()
  constructor(public views: View[]) {
    this.key = get_key(...views)
    Object.freeze(this)
    const res = ShapeTracker.cache.setDefault(this.key, this)
    if (!is_eq(this.views, res.views)) throw new Error(`Views don't match: \nthis=${list_str(this.views)} \nres=${list_str(res.views)}`)
    return res
  }
  add = (st: ShapeTracker): ShapeTracker => {
    let ret = new ShapeTracker(this.views)
    for (const v of st.views) ret = new ShapeTracker([...ret.views, v]).simplify() // one view at a time = better simplification
    return ret
  }
  toString = () => `new ShapeTracker(${list_str(this.views)})`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  invert = (out_shape: sint[]): undefined | ShapeTracker => {
    const inverted_views: View[] = []
    for (const [v, s] of zip(this.views.toReversed(), [...this.views.toReversed().slice(1).map((x) => x.shape), out_shape])) {
      const inverted = v.invert(s)
      if (inverted === undefined) return undefined
      inverted_views.push(inverted)
    }
    return new ShapeTracker(inverted_views).reshape(out_shape)
  }
  static from_shape = (shape: sint[]) => new ShapeTracker([View.create(shape)])

  get contiguous(): boolean {
    return this.views.length === 1 && !!this.views[0].contiguous
  }
  get consecutive() {
    const v = this.views[0]
    return this.views.length === 1 && v.mask === undefined && is_eq(v.strides, strides_for_shape(v.shape))
  }
  get shape() {
    return this.views.at(-1)!.shape
  }
  get size() {
    return this.views.at(-1)!.size()
  }

  reduce = (axis: number[]) => this.shape.map((s, i) => axis.includes(i) ? 1 : s)

  to_uop = () => new UOp(Ops.VIEW, dtypes.void, [], this)

  to_indexed_uops = (_idxs?: UOp[]): [UOp, UOp] => {
    const [idx, valid] = views_to_indexed_uops(this.views, _idxs !== undefined ? [..._idxs] : undefined)
    return [folded_upcast(idx), folded_upcast(valid)]
  }

  // upper bound on buffer size required to fit this shapetracker
  real_size = (): number => {
    if (this.shape.includes(0)) return 0
    const v = this.views[0], view = v.mask ? v.shrink(v.mask) : v, [idx] = views_to_indexed_uops([view])
    if (idx.vmax >= 1e12) throw new Error(`real_size broken for ${self}`)
    return Math.trunc(idx.vmax as number + 1)
  }
  vars = (): Variable[] => [...new Set(this.views.flatMap((v) => v.vars()))]

  get var_vals(): Map<Variable, number> {
    return merge_maps(this.vars().map((v) => new Map([v.unbind()])))
  }

  unbind = (): [ShapeTracker, Map<Variable, number>] => {
    const [unbound_views, var_vals] = zip(...this.views.map((v) => v.unbind()))
    return [new ShapeTracker(unbound_views as View[]), merge_maps(var_vals as Map<UOp, number>[])]
  }
  // NOTE: if a stride is not always valid, it will be None
  real_strides = (ignore_valid = false) => views_to_real_strides(this.views, ignore_valid)

  unit_stride_axes = (ignore_valid = false): number[] => [...this.real_strides(ignore_valid).entries()].filter(([i, st]) => st === 1).map(([i, st]) => i)

  axis_is_masked = (axis: number): boolean => {
    const [_, valid] = this.to_indexed_uops()
    return [...graph_rewrite(valid, symbolic_flat).toposort].filter((x) => x.op === Ops.RANGE).map((x) => x.arg).includes(axis)
  }
  simplify = (): ShapeTracker => {
    if (this.views.length >= 2) {
      const new_view = this.views.at(-2)?.add(this.views.at(-1)!)
      if (new_view !== undefined) return new ShapeTracker([...this.views.slice(0, -2), new_view]).simplify()
    }
    return this
  }
  // *** under this line are the movement ops ***

  pad = (arg: [sint, sint][]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.pad(arg)])
  shrink = (arg: [sint, sint][]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.shrink(arg)])
  expand = (new_shape: sint[]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.expand(new_shape)])
  permute = (axis: number[]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.permute(axis)])
  stride = (multi: number[]) => new ShapeTracker([...this.views.slice(0, -1), this.views.at(-1)!.stride(multi)])

  reshape = (new_shape: sint[]): ShapeTracker => {
    if (env.get_num('MERGE_VIEW', 1)) {
      const new_view = this.views.at(-1)?.reshape(new_shape)
      if (new_view !== undefined) return new ShapeTracker([...this.views.slice(0, -1), new_view])
    }
    return new ShapeTracker([...this.views, View.create(new_shape)])
  }
  // deno-fmt-ignore
  mop = (op: Ops, arg: any) => {
    switch (op) {
      case Ops.RESHAPE: return this.reshape(arg);
      case Ops.PERMUTE: return this.permute(arg); 
      case Ops.EXPAND: return this.expand(arg);
      case Ops.SHRINK: return this.shrink(arg);
      case Ops.STRIDE: return this.stride(arg); 
      case Ops.PAD: return this.pad(arg);
      default: throw new Error(`Invalid op ${op}`)
    }
    }
}
