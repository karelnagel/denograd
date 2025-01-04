import { Buffer } from '../device.ts'
import { ConstType, dtypes, ImageDType } from '../dtype.ts'
import { all_int, all_same, assert, colored, DEBUG, dedup, FUSE_ARANGE, FUSE_CONV_BW, get_env, isEq, isinstance, listStr, merge_maps, merge_sets, Metadata, prod, range, setDefault } from '../helpers.ts'
import { can_pad, ge, lt, resolve, sint_prod, sub, UPatInput } from '../ops.ts'
import { graph_rewrite, GroupOp, merge_views, Ops, PatternMatcher, UOp, UPat, Variable, view_left } from '../ops.ts'
import { ShapeTracker } from '../shape/shapetracker.ts'
import { LazyBuffer } from './lazy.ts'

// creation can recurse a lot
// sys.setrecursionlimit(10000)

const BUF_LIMIT = { 'METAL': 32 }

// **** ScheduleItem return type

// @DataClass //KAREL: not sure
export class ScheduleItem {
  constructor(public ast: UOp, public bufs: Buffer[], public metadata: Metadata[], public assign_preloads: Set<UOp>) {}
  /**
   * Read/write || write only buffers in the schedule.
   */
  get outputs(): Buffer[] {
    return this.bufs.filter((b, i) => this.output_idxs.includes(i))
  }
  /**
   * Read only buffers in the schedule.
   */
  get inputs(): Buffer[] {
    return this.bufs.filter((b, i) => !this.output_idxs.includes(i))
  }
  get output_idxs(): number[] {
    return this.ast.op === Ops.SINK ? this.ast.src.map((x) => x.src[0].arg) : [0]
  }
  toString = () => `new ScheduleItem(${this.ast}, ${listStr(this.bufs)}, ${listStr(this.metadata)}, ${listStr([...this.assign_preloads])})`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
}

// // **** Schedule context && big graph

// @DataClass // maybe needed
export class ScheduleContext {
  constructor(
    public lazybufs = new Map<UOp, LazyBuffer>(), // this maps BUFFER uops of this schedule to the underlying lazybuffer
    public var_vals = new Map<Variable, number>(), // this maps a BIND's DEFINE_VAR to its value
    public assigns = new Set<UOp>(), // this holds all the BUFFER uops we ASSIGN to in this schedule
    public realizes = new Map<UOp, UOp>(), // this holds all the BUFFER uops we mutate in this schedule
    public allbufs = new Map<UOp, UOp>(), // this maps BUFFER uops the actual op
    public ops_metadata = new Map<UOp, Metadata>(), // this maps fused ops to Metadata
    public children = new Map<UOp, Map<UOp, undefined>>(),
  ) {}
}
export const is_scheduled = (u: UOp): boolean => u?.op === Ops.VIEW && u?.src.length === 2

export const to_uop = (buf: LazyBuffer, ctx: ScheduleContext, buffers: Map<UOp, Buffer>, cache: Map<LazyBuffer, UOp>): UOp => {
  const r = cache.get(buf)
  if (r !== undefined) return r
  let op, ret, ubuf
  if (buf !== buf.base) {
    ret = to_uop(buf.base, ctx, buffers, cache).view(buf.st)
    cache.set(buf, ret)
    return ret
  }
  assert(buf.op !== undefined, `base must be base itthis ${buf}`)
  //   // make things that can't be images !images
  let dtype = buf.buffer!.dtype
  if (isinstance(dtype, ImageDType) && ((prod(buf.shape as number[]) !== prod(dtype.shape)) || !buf.st.unit_stride_axes().some((x) => buf.shape[x] as number % 4 === 0))) {
    assert(buf.realized === undefined, "can't fixup allocated buffer")
    if (DEBUG >= 2) console.log(`forcing image ${dtype} with shape ${buf.shape} to ${dtype.base}`)
    dtype = buf.dtype.base
    //     // hack the underlying buffer too
    buf.buffer!.dtype = dtype
    buf.buffer!.options = undefined
  }
  if (buf.is_realized) {
    ubuf = UOp.new_buffer(buf.device, buf.size, dtype!)
    buffers.set(ubuf, buf.buffer!)
    op = undefined
  } else if (buf.op === Ops.ASSIGN) {
    const [target, new_val] = buf.srcs!.map((x) => to_uop(x, ctx, buffers, cache))
    ubuf = target.base.buf_uop
    ctx.assigns.add(ubuf)
    op = new UOp(Ops.ASSIGN, dtype.base, [ubuf, new_val], buf.arg)
  } else {
    ubuf = UOp.new_buffer(buf.device, buf.size, dtype)
    buffers.set(ubuf, buf.buffer!)
    op = new UOp(buf.op!, GroupOp.Meta.includes(buf.op!) ? dtype : dtype.base, buf.srcs?.map((x) => to_uop(x, ctx, buffers, cache)), buf.arg)
  }
  ret = new UOp(Ops.VIEW, dtype.base, op === undefined ? [ubuf] : [ubuf, buf.forced_realize ? op.contiguous() : op], buf.st)
  cache.set(buf, ret)
  if (op !== undefined) {
    buf.buffer!.ref(1)
    ctx.lazybufs.set(ubuf, buf)
    ctx.allbufs.set(ubuf, ret)
    for (const x of op.src) {
      if (is_scheduled(x.base)) setDefault(ctx.children, x.base.buf_uop, new Map()).set(ubuf, undefined)
    }
  }
  return ret
}
// // **** AST graph rewrite

// // ** movement ops

export const apply_swizzle = (u: UOp, arg: ShapeTracker): UOp => {
  assert(u === u.base, `must be base to swizzle ${u}`)
  // with Context(TRACK_MATCH_STATS=0): return graph_rewrite(u.view(arg), view_left)
  return graph_rewrite(u.view(arg), view_left)
}

export const swizzle_r = (r: UOp, src: UOp, st: ShapeTracker): UOp => {
  throw new Error()
}
export const push_swizzle_down_through_reduce = (r: UOp, v: UOp, src: UOp): UOp => {
  throw new Error()
}
export const push_swizzle_down_through_elementwise = (root: UOp): UOp | undefined => {
  const swizzles = root.src.filter((x) => x.base !== x)
  if (!swizzles.length) return undefined
  const swizzle_shapes = swizzles.map((x) => [x.st!.shape, x.src[0].st!.shape])
  assert(all_same(swizzle_shapes.map(([x, y]) => [x, sint_prod(y)])), `swizzles must have the same size ${swizzle_shapes}`)
  const [new_shape, new_input_shape] = swizzle_shapes[0]
  const new_src = root.src.map((x) => !x.has_st ? x : swizzles.includes(x) ? x.src[0] : apply_swizzle(x, ShapeTracker.from_shape(new_input_shape)))
  let ret = root.replace({ src: new_src })
  //   // update the ASSIGN offset to match the new shape
  if (ret.op === Ops.ASSIGN && ret.arg !== undefined) ret = ret.replace({ arg: ret.arg.add(ShapeTracker.from_shape(new_input_shape)) })
  return ret.op === Ops.STORE ? ret : ret.view(ShapeTracker.from_shape(new_shape))
}
export const merge_double_reduce = (root: UOp, first_reduce: UOp): UOp => {
  throw new Error()
}
// // push VIEW to stores
export const view_right = merge_views.add(
  new PatternMatcher<Record<string, UOp>, UOp | undefined>([
    //   // ASSIGN with offset swizzles STORE
    [new UPat(Ops.STORE, undefined, [UPat.var('b'), UPat.var('st'), new UPat(Ops.ASSIGN).named('a')]), ({ a, b, st }) => a.arg === undefined ? undefined : apply_swizzle(b.store([st, a.replace({ arg: undefined })]), a.arg)],
    //   // non contiguous VIEW on a reduce creates a new VIEW
    [new UPat(Ops.REDUCE_AXIS, undefined, [UPat.var('src')], undefined, 'r').view(undefined, { name: 'v' }), ({ v, r, src }) => v.st?.contiguous ? undefined : swizzle_r(r, src, v.st!)],
    //   // push a VIEW down to STORE, through a reduce (ONLY reshapes)
    [new UPat(Ops.REDUCE_AXIS, undefined, [UPat.var('src').view(undefined, { name: 'v' })], undefined, 'r'), ({ r, v, src }) => push_swizzle_down_through_reduce(r, v, src)],
    //   // push VIEW(s) down to STORE, through an elementwise op (ONLY reshapes)
    [new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE]).named('root'), ({ root }) => push_swizzle_down_through_elementwise(root)],
    [new UPat(Ops.REDUCE_AXIS, undefined, [new UPat(Ops.REDUCE_AXIS).named('first_reduce')], undefined, 'root'), ({ root, first_reduce }) => merge_double_reduce(root, first_reduce)],
  ]),
)

// // ** ScheduleItem context builder

// @DataClass // for some reason causes issues, but should be here
export class ScheduleItemContext {
  constructor(
    public lazybufs = new Map<UOp, LazyBuffer>(),
    public ops_metadata = new Map<UOp, Metadata>(),
    public assigns = new Set<UOp>(),
    public var_vals = new Map<Variable, number>(),
    public sinked = new Map<UOp, UOp>(),
    public sts = new Set<ShapeTracker>(),
    public bufs: UOp[] = [],
    public metadata = new Set<Metadata>(),
    public assign_adj = new Map<UOp, UOp[]>(),
  ) {}
}
export const _append_st_vars = (ctx: ScheduleItemContext, x: UOp): UOp | undefined => {
  if (ctx.sts.has(x.st!)) return undefined
  const [st, var_vals] = x.st!.simplify().unbind()
  var_vals.forEach((v, k) => ctx.var_vals.set(k, v))
  ctx.sts.add(st)
  return st !== x.st ? st.to_uop() : undefined
}
export const _append_buf = (ctx: ScheduleItemContext, x: UOp): UOp => {
  ctx.bufs.push(x)
  return new UOp(Ops.DEFINE_GLOBAL, x.dtype, [], ctx.bufs.length - 1)
}
export const append_bufs = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleItemContext }, UOp | undefined>([
  [new UPat(Ops.BUFFER).named('x'), ({ ctx, x }) => _append_buf(ctx, x)],
])
export const _append_preload = (ctx: ScheduleItemContext, x: UOp, b: UOp): UOp => {
  const adj_loads = setDefault(ctx.assign_adj, b, [])
  adj_loads.push(x)
  if (!all_same(adj_loads.map((x) => x.op))) throw new Error(`Detected cycle when fusing ${adj_loads}. Can only fuse PRELOAD || LOAD of ${b}`)
  return x.replace({ op: Ops.LOAD })
}
export const check_preload = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleItemContext }, UOp | undefined>([
  [new UPat(Ops.PRELOAD, undefined, [UPat.var('b'), new UPat()], undefined, 'x'), ({ x, b, ctx }) => _append_preload(ctx, x, b)],
])

export const to_si = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleItemContext }, UOp | undefined>([
  [new UPat(Ops.VIEW).named('x'), ({ ctx, x }) => _append_st_vars(ctx, x)],
  [new UPat(Ops.SINK, undefined, [UPat.var('b').store([new UPat(), new UPat(GroupOp.Meta).named('x')])]), ({ ctx, b, x }) => x.replace({ src: [b, ...x.src] })],
])

// // ** fusion

export const lazy = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleItemContext }, UOp | undefined>([
  [new UPat(Ops.values()).named('x'), ({ ctx, x }) => ctx.ops_metadata.get(x) !== undefined ? void ctx.metadata.add(ctx.ops_metadata.get(x)!) : undefined],
  [new UPat(Ops.CONTIGUOUS, undefined, [UPat.var('x')]), ({ ctx, x }) => x],
])

export const multioutput = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleItemContext }, UOp | undefined>([
  [UPat.var('b').load([new UPat()]), ({ ctx, b }) => ctx.sinked.get(b)],
])

export const append_load = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleItemContext }, undefined>([
  [UPat.var('b').load([new UPat()], { name: 'x' }), ({ ctx, b, x }) => ctx.assigns.has(b) ? void setDefault(ctx.assign_adj, b, []).push(x) : undefined],
])

export const full_ast_rewrite = (pre: UOp, ctx: ScheduleContext): [UOp, ScheduleItemContext] => {
  const si_ctx = new ScheduleItemContext(
    ctx.lazybufs,
    ctx.ops_metadata,
    ctx.assigns,
    ctx.var_vals,
    new Map(pre.src.map((x) => [x.buf_uop, x.src[2]])),
    undefined,
    undefined,
    new Set(pre.src.filter((x) => ctx.lazybufs.get(x.buf_uop)?.metadata !== undefined).map((x) => ctx.lazybufs.get(x.buf_uop)!.metadata!)),
  )

  //   // fuse && fold store -> loads
  const ops_folding = si_ctx.sinked.size === 1 ? lazy : lazy.add(multioutput)
  let sink = graph_rewrite(pre, si_ctx.assigns.size === 0 ? ops_folding : ops_folding.add(append_load), si_ctx)
  //   // do movement ops
  sink = graph_rewrite(graph_rewrite(sink, view_left), view_right)
  //   // convert to AST
  sink = graph_rewrite(graph_rewrite(sink, si_ctx.assigns.size !== 0 ? to_si.add(check_preload) : to_si, si_ctx), append_bufs, si_ctx)
  //   // assert(buffer count limit)
  const device = si_ctx.bufs[0].device
  const limit = BUF_LIMIT[device as 'METAL']
  if (limit !== undefined && si_ctx.bufs.length >= limit) {
    if (DEBUG >= 3) console.log(sink)
    throw new Error(`Kernel for ${si_ctx.metadata} exceeded the ${limit} buffer count limit for ${device} with ${si_ctx.bufs.length} buffers.`)
  }
  //   // we also allow masked views. if it has a single view && it's equal when you shrink a contig, it's fine
  for (const [ubuf, ops] of si_ctx.assign_adj.entries()) {
    if (
      si_ctx.sinked.get(ubuf) !== undefined && !ops.every((x) => {
        const [s, m] = [x.st_arg, x.st_arg.views[0].mask]
        return s.contiguous || (s.views.length === 1 && m !== undefined && ShapeTracker.from_shape(s.shape).shrink(m) === s.shrink(m))
      })
    ) {
      throw new Error('this operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n' + colored('   - a += a.T\n', 'red') + colored('   + a += a.T.contiguous()', 'green'))
    }
  }
  if (get_env('RUN_PROCESS_REPLAY')) PROCESS_REPLAY_CAPTURE.push([[pre, ctx.assigns], sink])
  return [sink, si_ctx]
}
export const PROCESS_REPLAY_CAPTURE: [[UOp, Set<UOp>], UOp][] = []
// if (getenv("RUN_PROCESS_REPLAY")) {
//   @atexit.register
//   const save_process_replay = (): undefined => {
//     for (const x,ret of PROCESS_REPLAY_CAPTURE){ diskcache_put("schedule_process_replay", string(x[0].key), (*x, {}, ret))

// // **** Schedule grouping

export const uval = (u: UOp): UOp => {
  assert(is_scheduled(u), `must be a scheduled op ${u}`)
  const r = u.src[1]
  return r.op === Ops.CONTIGUOUS && !(r.src[0].base.op === Ops.VIEW && r.src[0].base.src.length === 2) ? r.src[0] : r
}
/**
 * recursively search the uop for groupable children, realize the UOp if a child can't group
 */
export const recursive_group = (
  tr: UOp,
  st: ShapeTracker,
  r: UOp,
  children: Map<UOp, Map<UOp, undefined>>,
  allbufs: Map<UOp, UOp>,
  realizes: Map<UOp, UOp>,
  reduce_for_op: Map<UOp, UOp>,
  group: Map<UOp, undefined>,
  cache: Map<[UOp, ShapeTracker], undefined>,
): undefined => {
  if (cache.has([tr, st])) return
  cache.set([tr, st], undefined)
  const rsize = allbufs.get(r)!.st?.size
  if (realizes.has(tr) && tr !== r) {
    //     // can only fuse contiguous
    //     // max one reduceop per kernel
    if (!st.contiguous || (st.size !== rsize) || reduce_for_op.has(tr)) setDefault(group, r, undefined)
    return setDefault(group, tr, undefined)
  }
  for (const tr_next of children.get(tr)!.keys()) {
    //     // max one reduceop per kernel
    const tr_next_uop = uval(allbufs.get(tr_next)!).base
    if (tr_next_uop.op === Ops.REDUCE_AXIS) return setDefault(group, r, undefined)
    //     // can only fuse contiguous
    const st_childs = dedup(tr_next_uop.src.filter((x) => is_scheduled(x.base) && x.base.buf_uop === tr).map((x) => x.st!))
    if (st_childs.length > 1) return setDefault(group, r, undefined)
    recursive_group(tr_next, st.add(st_childs[0]), r, children, allbufs, realizes, reduce_for_op, group, cache)
  }
}
export const get_isolated_children = (r: UOp, reduce_for_op: Map<UOp, UOp>, children: Map<UOp, Map<UOp, undefined>>, allbufs: Map<UOp, UOp>, realizes: Map<UOp, UOp>, group: Map<UOp, undefined>): Map<UOp, undefined> => {
  const [rc_parents, cache] = [new Map(group), new Set()]
  while (rc_parents.size) {
    const current = rc_parents.keys().next().value!
    rc_parents.delete(current)
    const p = uval(current)
    if (cache.has(p)) continue
    cache.add(p)
    //     // max one reduceop per kernel
    if (p.op === Ops.REDUCE_AXIS) return new Map()
    p.src.filter((x) => is_scheduled(x.base) && x.base.buf_uop !== r).forEach((x) => rc_parents.set(x.base.buf_uop, undefined))
  }
  //   // search descendants of the reduceop that can cleanly group
  const descendants = new Map<UOp, undefined>()
  for (const tr of group.keys()) recursive_group(tr, allbufs.get(tr)!.st!, tr, children, allbufs, realizes, reduce_for_op, descendants, new Map())
  return merge_maps([group, [...descendants].some(([tr]) => group.has(tr)) ? new Map() : descendants])
}
/**
 * search the big graph for all the reduceops that need to realize, sometimes group/fuse the reduceop
 */
export const group_realizes = (ctx: ScheduleContext): UOp[][] => {
  //   // find all reduces, && pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  const reduce_for_op = new Map<UOp, UOp>()
  const reduce_of_const: UOp[] = []
  const double_reduces: UOp[] = []
  for (let [r, r_uop] of ctx.allbufs.entries()) {
    r_uop = uval(r_uop)
    if (r_uop.op !== Ops.REDUCE_AXIS) continue
    const x = r_uop.src[0]
    if (FUSE_CONV_BW && uval(x.base).op === r_uop.op && x.base !== x) double_reduces.push(r)
    if (ctx.realizes.has(r)) continue
    let group = new Map<UOp, undefined>()
    recursive_group(r, r_uop.st!, r, ctx.children, ctx.allbufs, ctx.realizes, reduce_for_op, group, new Map())
    //     // max one reduceop per kernel
    let can_chase = [...group].every(([tr]) => !reduce_for_op.has(tr))
    //     // TODO: forced_realize exists because the scheduler === incapable of checking for this-contained DAGs
    let forced_realize = group.has(r)
    if (!forced_realize && group.size > 1) group = get_isolated_children(r, reduce_for_op, ctx.children, ctx.allbufs, ctx.realizes, group)
    //     // can only fuse assign if no other assign_target === used in the kernel
    if (!forced_realize && [...group].some(([x]) => ctx.assigns.has(x))) {
      const parents = new Map([[r, undefined], ...group])
      while (parents.size && !forced_realize) {
        const p = parents.keys().next().value!
        parents.delete(p)
        let p_uop = ctx.allbufs.get(p)
        if (p_uop === undefined) continue
        p_uop = uval(p_uop)
        if (p_uop.op === Ops.ASSIGN && !group.has(p)) [forced_realize, can_chase] = [true, false]
        if (ctx.realizes.has(p)) continue
        p_uop.src.filter((x) => x.base.op === Ops.VIEW && x.base.src.length !== 0).forEach((x) => parents.set(x.base.src[0], undefined))
      }
    }
    if (forced_realize || !group.size) {
      let tr = r
      if (can_chase) {
        //         // can chase this down to contiguous children
        let st = r_uop.st!
        while (ctx.children.get(tr)!.size === 1) {
          const tr_next = ctx.children.get(tr)!.values().next().value!
          const tr_next_uop = uval(ctx.allbufs.get(tr_next)!)
          const st_childs = dedup(tr_next_uop.src.filter((x) => is_scheduled(x.base) && x.base.buf_uop === tr).map((x) => x.st!))
          if (st_childs.length > 1) break
          if (st.size !== st_childs[0].size) break
          st = st.add(st_childs[0])
          if (!st.contiguous || tr_next_uop.op === Ops.REDUCE_AXIS) break
          tr = tr_next
        }
        //         // don't cast to higher size before store (tr can!be realized if forced_realize)
        const tr_uop = uval(ctx.allbufs.get(tr)!)
        if (tr_uop.op === Ops.CAST && tr_uop.dtype.base.itemsize > tr_uop.src[0].dtype.base.itemsize) tr = tr_uop.src[0].base.buf_uop
      }
      group = new Map([[tr, undefined]])
      ctx.realizes.set(tr, tr)
    }
    group.keys().forEach((tr) => reduce_for_op.set(tr, r))
    if (FUSE_ARANGE && r_uop.arg[0] === Ops.ADD && uval(r_uop.src[0].base).op === Ops.CONST) reduce_of_const.push(r)
  }
  //   // fuse double reduces with no other child
  for (const reduceop of double_reduces) {
    const top_reduce = uval(ctx.allbufs.get(reduceop)!).src[0].base.buf_uop
    if (ctx.children.get(top_reduce)!.size === 1) ctx.realizes.delete(top_reduce)
  }
  //   // maybe fuse arange with its children
  //   // KAREL: not needed for mnist
  //   // for (const rbuf of reduce_of_const){
  //   //   group = {tr:undefined for tr,rop in reduce_for_op.items() if rop === rbuf}
  //   //   if any(ctx.lazybufs[tr].forced_realize for (const tr of group)){ continue
  //   //   kernel_children = {c for tr in group for c in ctx.children[tr] if uval(ctx.allbufs[c]).op !in {Ops.COPY, Ops.BUFFER_VIEW}}
  //   //   if (kernel_children.length === 0) continue
  //   //   for (const tr of group){ del ctx.realizes[tr]
  //   // group BUFFER uops into kernels
  const output_groups = new Map<UOp, UOp[]>()
  for (const ubuf of ctx.realizes.keys()) setDefault(output_groups, reduce_for_op.get(ubuf) || ubuf, []).push(ubuf)
  return [...output_groups.values()]
}
// // **** Schedule creation && BFS toposort

// // ** ops in the big graph can either be pre-realized || scheduled (fused/realized)

export class UPatRealized extends UPat {
  constructor() {
    super(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b')], undefined, 'base')
  }
}
export class UPatScheduled extends UPat {
  constructor(args: Partial<UPatInput> = {}) {
    super(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b'), new UPat(args.op, args.dtype, args.src, args.arg, 'to_store', args.allow_any_len, args.location, args.custom_early_reject)], undefined, 'base')
  }
}
// // ** this === schedule level const folding

export const _as_const = (u: UOp, val: ConstType): UOp => {
  assert(is_scheduled(u), `must be scheduled to fold ${u}`)
  const base = ShapeTracker.from_shape([])
  const st = base.reshape(range(u.shape.length).map((x) => 1)).expand(u.shape)
  return new UOp(Ops.VIEW, u.dtype, [u.buf_uop, UOp.const(u.dtype, val)], base).view(st)
}
export const ops_folding = new PatternMatcher([
  //   // op with size 0 === zero
  [new UPatScheduled({}), ({ ctx, b, to_store, base }) => base.size === 0 ? _as_const(base, 0) : undefined],
])

// // ** this decides which ops get realized

export const realize = (ctx: Map<UOp, UOp>, b: UOp, to_store: UOp, _base: UOp): undefined => {
  if (![Ops.CONST, Ops.BIND].includes(to_store.op)) ctx.set(b, to_store)
}
export const realize_view = (ctx: Map<UOp, UOp>, base: UOp, view: UOp, to_store: UOp, b: UOp): undefined => {
  if ([Ops.CONST, Ops.BIND].includes(to_store.op)) return undefined
  const base_shape = base.st!.shape
  const st = view.st!
  //   // fold simple pads
  const m = st.views.at(-1)!.mask
  if (st.views.length === 1 && m !== undefined && all_int(base_shape) && resolve(ge(sint_prod(base_shape), sint_prod(m.map(([x, y]) => sub(y, x)))))) {
    return can_pad(base, ctx, new Set()) ? undefined : realize(ctx, b, to_store, base)
  }
  //   // early realize before expand
  if (resolve(lt(sint_prod(base_shape), sint_prod(st.shape)))) return realize(ctx, b, to_store, base)
  //   // otherwise safety check pads
  return st.views.every((v) => v.mask === undefined) || can_pad(base, ctx, new Set()) ? undefined : realize(ctx, b, to_store, base)
}
export const fold_img_cast = (ctx: Map<UOp, UOp>, xb: UOp, view: UOp, b: UOp, to_cast: UOp): UOp | undefined => {
  if (!isinstance(xb.dtype, ImageDType) || !ctx.has(b) || !ctx.has(xb) || GroupOp.Meta.includes(uval(to_cast).op)) return undefined
  ctx.delete(b)
  return to_cast.view(view.st!)
}
export const init_big_graph = (ctx: Map<UOp, UOp>, sink: UOp): UOp | undefined => {
  const new_src = sink.src.filter((x) => is_scheduled(x.base) && uval(x.base).op !== Ops.CONST).map((x) => x.base)
  return isEq(new_src, sink.src) ? undefined : new_src.length === 0 ? new UOp(Ops.NOOP) : UOp.sink(...new_src)
}
export const do_realize = new PatternMatcher<Record<string, UOp> & { ctx: Map<UOp, UOp> }, UOp | undefined>([
  //   // always realize sinked ops
  [new UPat(Ops.SINK).named('sink'), ({ sink, ctx }) => init_big_graph(ctx, sink)],
  //   // always realize meta ops
  [new UPatScheduled({ op: [Ops.ASSIGN, Ops.CONTIGUOUS, ...GroupOp.Meta] }), ({ ctx, b, to_store, base }) => realize(ctx, b, to_store, base)],
  //   // realize before expand || unsafe pad ops
  [new UPatScheduled({}).view(undefined, { name: 'view' }), ({ ctx, base, view, to_store, b }) => realize_view(ctx, base, view, to_store, b)],
  //   // don't realize image to image casts
  [
    new UPatScheduled({ op: Ops.CAST, src: [new UPat(Ops.VIEW, undefined, [UPat.var('xb'), new UPat()], undefined, 'to_cast')], dtype: dtypes.float }).view(undefined, { name: 'view' }),
    ({ ctx, xb, view, b, to_cast }) => fold_img_cast(ctx, xb, view, b, to_cast),
  ],
  //   // realize before COPY || BUFFER_VIEW
  [new UPat([Ops.COPY, Ops.BUFFER_VIEW], undefined, [UPat.any([new UPatScheduled(), new UPatScheduled().view()])]), ({ ctx, b, to_store, base }) => realize(ctx, b, to_store, base)],
])

// // ** this breaks down realized ops into STOREs && rewrites the ops to LOADs

export const generate_valid = (ctx: ScheduleContext, _b: UOp, to_store: UOp, base: UOp): UOp => {
  const val = to_store.arg
  if (isinstance(val, UOp)) ctx.var_vals.set(...val.unbind())
  return UOp.const_with_shape(base.dtype, val, base.st!.shape)
}
export const append_realize = (ctx: ScheduleContext, b: UOp, to_store: UOp, base: UOp): UOp => {
  const st = base.st!
  ctx.realizes.set(b, b.store([ShapeTracker.from_shape(st.shape).to_uop(), append_op(ctx, b, to_store)]))
  return new UOp(Ops.LOAD, base.dtype, [b, st.to_uop()])
}
export const append_op = (ctx: ScheduleContext, b: UOp, to_store: UOp): UOp => {
  const m = ctx.lazybufs.get(b)!.metadata
  if (m !== undefined) ctx.ops_metadata.set(to_store, m)
  return to_store
}
export const break_sched = new PatternMatcher<Record<string, UOp> & { ctx: ScheduleContext }, UOp | undefined>([
  //   // consts are always fused && generated
  [new UPatScheduled({ op: [Ops.CONST, Ops.BIND] }), ({ ctx, b, to_store, base }) => generate_valid(ctx, b, to_store, base)],
  //   // everything else === a VIEW of BUFFER that either realizes || fuses
  [new UPatScheduled({}), ({ ctx, b, to_store, base }) => ctx.realizes.has(b) ? append_realize(ctx, b, to_store, base) : append_op(ctx, b, to_store)],
  //   // just load realized buffers
  [new UPatRealized(), ({ ctx, b, base }) => new UOp(ctx.assigns.has(b) ? Ops.PRELOAD : Ops.LOAD, base.dtype, [b, base.st!.to_uop()])],
])

// @track_rewrites(named=true)
export const create_schedule_with_vars = (outs: LazyBuffer[]): [ScheduleItem[], Map<Variable, number>] => {
  outs = dedup(outs.filter((x) => x.base.realized === undefined && x.base.op !== Ops.CONST).map((x) => x.base))
  if (outs.length === 0) return [[], new Map()]
  //   // create the big graph
  const ctx = new ScheduleContext()
  const cache = new Map<LazyBuffer, UOp>()
  const buffers = new Map<UOp, Buffer>()
  let big_graph = UOp.sink(...outs.map((x) => to_uop(x, ctx, buffers, cache)))
  for (const u of big_graph.src) ctx.realizes.set(u.buf_uop, u)
  big_graph = graph_rewrite(big_graph, ops_folding.add(do_realize), ctx.realizes)
  //   // group realizes into kernels
  const store_groups = group_realizes(ctx)
  graph_rewrite(big_graph, break_sched, ctx)
  //   // preschedule realize groups
  const prescheduled: ScheduleItem[] = []
  for (const store_uops of store_groups) {
    const stores = store_uops.filter((u) => ctx.realizes.get(u)!.op === Ops.STORE).map((u) => ctx.realizes.get(u))
    if (stores.length !== 0) {
      const [ast, ast_ctx] = full_ast_rewrite(UOp.sink(...stores.map((s) => s!)), ctx)
      prescheduled.push(
        new ScheduleItem(ast, ast_ctx.bufs.filter((u) => u.size !== 0).map((u) => buffers.get(u)!), [...ast_ctx.metadata], new Set(ast_ctx.assign_adj.entries().filter(([_, ops]) => ops.some((x) => x.op === Ops.PRELOAD)).map(([ubuf]) => ubuf))),
      )
      for (const u of ast_ctx.sinked.keys()) {
        ast_ctx.lazybufs.get(u)!.srcs?.map((s) => s.__del__()) // can only schedule once
        delete ast_ctx.lazybufs.get(u)!.srcs
      }
    }
  }
  //   // do BFS
  const schedule_targets = new Map(prescheduled.flatMap((si) => si.outputs.map((out) => [out, si])))
  const graph = new Map<ScheduleItem, ScheduleItem[]>()
  const in_degree = new Map<ScheduleItem, number>()
  for (const si of prescheduled) {
    //     // realize outputs before a parent === assigned to
    const parents_assigns = dedup([...si.assign_preloads].map((x) => schedule_targets.get(buffers.get(x)!)!).filter((xsi) => xsi && xsi !== si))
    for (const assign of parents_assigns) {
      setDefault(graph, si, []).push(assign)
      in_degree.set(assign, setDefault(in_degree, assign, 0) + 1)
    }
    //     // realize outputs after all parents are realized
    const scheduled_parents = dedup(si.inputs.map((x) => schedule_targets.get(x)!).filter((xsi) => xsi !== undefined && !parents_assigns.includes(xsi)))
    for (const x of scheduled_parents) {
      setDefault(graph, x, []).push(si)
      in_degree.set(si, setDefault(in_degree, si, 0) + 1)
    }
  }
  const queue = prescheduled.filter((si) => setDefault(in_degree, si, 0) === 0)
  const schedule: ScheduleItem[] = []
  while (queue.length) {
    const si = queue.shift()
    schedule.push(si!)
    for (const x of setDefault(graph, si!, [])) {
      in_degree.set(x, in_degree.get(x)! - 1)
      if (in_degree.get(x) === 0) queue.push(x)
    }
  }
  //   // confirm everything was scheduled correctly
  const groups = prescheduled.length
  if (schedule.length !== groups) throw new Error(`cycle detected in graph, grouped ${groups} but only scheduled ${schedule.length}`)
  if (DEBUG >= 1 && schedule.length >= 10) console.log(`scheduled ${schedule.length} kernels`)
  return [schedule, ctx.var_vals]
}
export const create_schedule = (_outs: LazyBuffer[]): ScheduleItem[] => {
  throw new Error()
}
