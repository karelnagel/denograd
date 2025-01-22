import { Buffer } from '../device.ts'
import { DType, dtypes, ImageDType } from '../dtype.ts'
import { all_int, all_same, cache, CAPTURE_PROCESS_REPLAY, colored, DEBUG, dedup, FUSE_ARANGE, FUSE_CONV_BW, get_env, is_eq, isinstance, list_str, merge_maps, Metadata, NotImplemented, range, set_default, zip } from '../helpers.ts'
import { add, buffers, can_pad, ge, identity_element, lt, mul, pow, prod, resolve, sint, sub, symbolic_simple, type_verify, UPatInput } from '../ops.ts'
import { graph_rewrite, GroupOp, merge_views, Ops, PatternMatcher, UOp, UPat, Variable, view_left } from '../ops.ts'
import { ShapeTracker } from '../shape/shapetracker.ts'
import { strides_for_shape, View } from '../shape/view.ts'

// **** big graph spec

export const tensor_uop_spec = new PatternMatcher<unknown, boolean>([
  [new UPat(Ops.DEVICE, dtypes.void, []).named('device'), ({ device }) => typeof device.arg === 'string'],
  [new UPat(Ops.BUFFER, undefined, [new UPat(Ops.DEVICE)]).named('buf'), ({ buf }) => Array.isArray(buf.arg) && buf.arg.length === 2 && all_int(buf.arg) && buf.dtype instanceof DType],
  [new UPat(GroupOp.Movement, undefined, [UPat.var('x')]).named('mv'), ({ mv, x }) =>
    //  # naturally correct
    (Array.isArray(mv.arg) && mv.dtype === x.dtype) ||
    // "make things that can't be images not images" can change the buffer dtype
    // this is fine as long as it's a realized buffer and base dtypes match.
    ((mv.dtype instanceof ImageDType || x.dtype instanceof ImageDType) && x.dtype.base === mv.dtype.base && x.is_realized)],
  // Tensor variable bindings
  [new UPat(Ops.BIND, dtypes.int, [new UPat(Ops.DEFINE_VAR), UPat.cvar(undefined, dtypes.int)], undefined), () => true],
  [new UPat(Ops.DEFINE_VAR, undefined, [new UPat(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([]))], undefined), () => true],
  // Tensor const has an unmasked ShapeTracker of stride 0 and a device
  [
    new UPat(Ops.CONST, undefined, [new UPat(Ops.VIEW, undefined, [new UPat(Ops.DEVICE)]).named('st')]),
    ({ st }) => st.st!.views.length === 1 && st.st!.views[0].strides.every((s) => s === 0) && st.st!.views[0].mask === undefined,
  ],
  // DETACH and CONTIGUOUS change how we interpret the source UOp
  // CONTIGUOUS ensures the source UOp realizes
  [new UPat([Ops.DETACH, Ops.CONTIGUOUS], undefined, [UPat.var('x')], undefined).named('root'), ({ root, x }) => root.dtype === x.dtype],
  // COPY
  // NOTE: the arg here specifies clone=True, which prevents folding same device copy
  [new UPat(Ops.COPY, undefined, [new UPat(Ops.DEVICE), UPat.var('x')]).named('copy'), ({ copy, x }) => typeof copy.arg === 'boolean' && copy.dtype === x.dtype],
  // VIEW(BUFFER) applies a ShapeTracker on top of the underlying device buffer
  // NOTE: VIEW size exactly matches the underlying BUFFER, tensor doesn't apply movement ops to the VIEW
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('buf')]).named('view'), ({ view, buf }) => view.dtype === buf.dtype && view.size === buf.size && view.st!.contiguous],
  // ASSIGN changes the value of a realized buffer
  [new UPat(Ops.ASSIGN, undefined, [UPat.var('target'), UPat.var('new_val')]).named('assign'), ({ assign, target, new_val }) => (target.op === Ops.BUFFER || target.is_realized) && (assign.dtype === target.dtype && target.dtype === new_val.dtype)],
  // TODO: BUFFER_VIEW is overloaded, it should be removed.
  // BUFFER_VIEW shares the device buffer with its source, it uses a subbuffer of the underlying source buffer
  [new UPat(Ops.BUFFER_VIEW, undefined, [UPat.var('x')]).named('root'), ({ root, x }) =>
    // BUFFER_VIEW can replace contiguous, keeping dtype the same
    (root.dtype === x.dtype) ||
    // it can also replace bitcast, this changes the dtype, but the itemsize stays the same
    (root.dtype !== x.dtype && root.dtype.itemsize === x.dtype.itemsize) ||
    // it can also represent shape changing bitcast (only on DISK)
    (root.dtype !== x.dtype && root.dtype.itemsize !== x.dtype.itemsize && x.device.startsWith('DISK'))],
])

// **** ScheduleItem return type

export class ScheduleItem {
  constructor(
    public ast: UOp,
    public bufs: Buffer[],
    public metadata: Metadata[],
    public assign_preloads: UOp[],
  ) {}
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
  @cache
  get output_idxs(): number[] {
    return this.ast.op === Ops.SINK ? this.ast.src.map((x) => x.src[0].arg) : [0]
  }
  toString = () => `new ScheduleItem(${this.ast}, ${list_str(this.bufs)}, ${list_str(this.metadata)}, ${list_str([...this.assign_preloads])})`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
}

// **** Schedule context && big graph

export class ScheduleContext {
  constructor(
    public tensor_uops = new Map<UOp, UOp[]>(), // this maps BUFFER uops of this schedule to the underlying lazybuffer
    public var_vals = new Map<Variable, number>(), // this maps a BIND's DEFINE_VAR to its value
    public assigns = new Set<UOp>(), // this holds all the BUFFER uops we ASSIGN to in this schedule
    public realizes = new Map<UOp, UOp>(), // this holds all the BUFFER uops we mutate in this schedule
    public allbufs = new Map<UOp, UOp>(), // this maps BUFFER uops the actual op
    public ops_metadata = new Map<UOp, Metadata>(), // this maps fused ops to Metadata
    public contiguous = new Map<UOp, UOp>(), // this maps roots to places they are made contiguous
    public children = new Map<UOp, Map<UOp, undefined>>(),
    public becomes_map = new Map<UOp, UOp>(),
  ) {}
}

// wrap tensor uops around a VIEW(BUFFER, <uop>)
// this BUFFER preserves a link back to the uop on the tensor after the scheduler rewrites it.
const add_buffers = (buf: UOp, ctx: ScheduleContext, cache: Map<UOp, UOp>): UOp => {
  if (cache.has(buf)) return cache.get(buf)!
  if (buf.op === Ops.SINK) return UOp.sink(...buf.src.map((x) => add_buffers(x, ctx, cache)))
  // shapeless op is passthrough
  // realized is passthrough
  // constants are passthrough
  if (buf.st === undefined || buf.base.is_realized || [Ops.CONST, Ops.BIND].includes(buf.base.op)) return buf
  // view is passthrough
  if (buf !== buf.base) {
    const ret = add_buffers(buf.base, ctx, cache).view(buf.st)
    cache.set(buf, ret)
    return ret
  }
  // make things that can't be images not images
  let dtype = buf.dtype
  if (dtype instanceof ImageDType && (prod(buf.shape) !== prod(dtype.shape) || !buf.st.unit_stride_axes().some((x) => buf.shape[x] as number % 4 === 0))) {
    if (DEBUG >= 2) console.log(`forcing image ${dtype} with shape ${buf.shape} to ${dtype.base}`)
    dtype = buf.dtype.base
  }
  // ASSIGN already has a target buffer, otherwise we create a new one
  const buf_uop = buf.op === Ops.ASSIGN ? buf.buf_uop : UOp.new_buffer(buf.device, buf.size, dtype)
  const op = buf.replace({ dtype: dtype.base, src: buf.src.map((x) => add_buffers(x, ctx, cache)) })
  // track the underlying tensor uop for this op
  ctx.tensor_uops.set(buf_uop, [buf])
  // (early) bufferize
  const ret = new UOp(Ops.VIEW, dtype.base, [buf_uop, buf.forced_realize ? op.alu(Ops.CONTIGUOUS) : op], buf.st)
  cache.set(buf, ret)
  return ret
}
// **** AST graph rewrite
// ** movement ops

export const apply_swizzle = (u: UOp): UOp => {
  // TODO add "with Context(TRACK_MATCH_STATS=0):""
  return graph_rewrite(u, view_left)
}

const swizzle_r = (r: UOp, src: UOp, st: ShapeTracker): UOp => {
  const input_st = ShapeTracker.from_shape(src.st!.shape)
  const tmp = input_st.permute([...range(input_st.shape.length).filter((i) => !r.axis_arg.includes(i)), ...r.axis_arg])
  const rshape = tmp.shape.slice(-r.axis_arg.length), prshape = prod(rshape)
  const strides = strides_for_shape(rshape)
  const nv = st.views.map((v) => View.create([...v.shape, ...rshape], [...v.strides.map((x) => mul(x, prshape)), ...strides], mul(v.offset, prshape), v.mask !== undefined ? [...v.mask, ...rshape.map((s) => [0, s] as [sint, sint])] : undefined))
  // update input_st and axis
  const new_input_st = tmp.add(new ShapeTracker([...nv]))
  const new_axis = range(st.shape.length, st.shape.length + r.axis_arg.length)
  return apply_swizzle(src.view(new_input_st)).r(r.arg[0], new_axis).view(ShapeTracker.from_shape(st.shape))
}
const reduceop_view_right = (r: UOp, v: UOp, src: UOp): UOp => {
  if (!v.st!.contiguous || v.size !== src.size) throw new Error(`can't push ${v} down through ${src}`)
  const output_shape = v.st!.reduce(r.axis_arg)
  return src.r(r.arg[0], [...zip(src.shape, output_shape).entries()].filter(([i, [s, u]]) => s !== u).map(([i]) => i)).view(ShapeTracker.from_shape(output_shape))
}
const elementwise_view_right = (root: UOp): UOp | undefined => {
  const swizzles = root.src.filter((x) => x.base !== x)
  if (swizzles.length === 0) return undefined
  if (!swizzles.every((x) => x.base.st !== undefined)) throw new Error(`found shapeless VIEW src in ${root}`)
  if (!all_same(swizzles.map((x) => x.base.size))) throw new Error(`swizzle inputs must have the same size {swizzles}`)
  // push the swizzle from src to root
  const output_swizzle = swizzles[0]
  const new_input_st = ShapeTracker.from_shape(output_swizzle.base.shape)
  const ret = root.replace({ src: root.src.map((x) => x.st === undefined ? x : swizzles.includes(x) ? x.base : apply_swizzle(x.view(new_input_st))) })
  // NOTE: swizzle resolves once we hit STORE
  return ret.op === Ops.STORE ? ret : ret.view(ShapeTracker.from_shape(output_swizzle.shape))
}
const merge_double_reduce = (root: UOp, first_reduce: UOp): UOp => {
  if (root.arg[0] !== first_reduce.arg[0]) throw new Error("can't merge reduceops with different alu")
  if ([...first_reduce.src[0].toposort].some((x) => x.op === Ops.REDUCE_AXIS)) throw new Error("can't merge more than two reduceops at a time")
  return first_reduce.replace({ arg: [first_reduce.arg[0], [...root.axis_arg, ...first_reduce.axis_arg]] })
}
// push VIEW to stores
export const view_right = merge_views.add(
  new PatternMatcher([
    // STORE(.., ASSIGN(VIEW(BUFFER), new_val)) -> STORE(.., new_val).view()
    [new UPat(Ops.STORE, undefined, [UPat.var('b'), UPat.var('st'), UPat.var('target').assign(UPat.var('val'))]), ({ b, target, st, val }) => apply_swizzle(b.store([st, val]).view(target.st!))],
    // REDUCE(src.view(contiguous=False)) -> REDUCE(src.view(contiguous=True)).view()
    [new UPat(Ops.REDUCE_AXIS, undefined, [UPat.var('src')]).named('r').view(undefined, { name: 'v' }), ({ v, r, src }) => v.st!.contiguous ? undefined : swizzle_r(r, src, v.st!)],
    // REDUCE(src.view()) -> REDUCE(src).view()
    [new UPat(Ops.REDUCE_AXIS, undefined, [UPat.var('src').view(undefined, { name: 'v' })]).named('r'), ({ r, v, src }) => reduceop_view_right(r, v, src)],
    // ALU(src.view()) -> ALU(src).view()
    [new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.CONTIGUOUS, Ops.STORE]).named('root'), ({ root }) => elementwise_view_right(root)],
    // double reduce op collapses to a single reduce op
    [new UPat(Ops.REDUCE_AXIS, undefined, [new UPat(Ops.REDUCE_AXIS).named('first_reduce')]).named('root'), ({ root, first_reduce }) => merge_double_reduce(root, first_reduce)],
  ]),
)

// ** ScheduleItem context builder

// @DataClass // for some reason causes issues, but should be here
export class ScheduleItemContext {
  constructor(
    public var_vals = new Map<Variable, number>(),
    public sts = new Set<ShapeTracker>(),
    public bufs: UOp[] = [],
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
  return new UOp(Ops.DEFINE_GLOBAL, x.dtype.ptr(x.size), [], ctx.bufs.length - 1)
}

export const to_si = new PatternMatcher<ScheduleItemContext>([
  // BUFFER -> DEFINE_GLOBAL
  [new UPat(Ops.BUFFER).named('x'), ({ ctx, x }) => _append_buf(ctx, x)],
  // simplify and unbind the final VIEWs
  [new UPat(Ops.VIEW).named('x'), ({ x, ctx }) => _append_st_vars(ctx, x)],
  // don't need SINK on COPY or BUFFER_VIEW
  [new UPat(Ops.SINK, undefined, [UPat.store([UPat.var('b'), new UPat(), new UPat(GroupOp.Meta).named('x')])]), ({ b, x }) => x.replace({ src: [b, ...x.src] })],
  // don't need contiguous or assign anymore
  [new UPat(Ops.CONTIGUOUS, undefined, [UPat.var('x')]), ({ x }) => x],
  [new UPat(Ops.ASSIGN, undefined, [new UPat(), UPat.var('x')]), ({ x }) => x],
  // PRELOAD becomes LOAD
  [new UPat(Ops.PRELOAD).named('root'), ({ root }) => root.replace({ op: Ops.LOAD })],
])

// LOAD(BUFFER) -> the STORE value if it's we're doing the STORE in the same kernel
export const multioutput = new PatternMatcher<Map<UOp, UOp>>([[UPat.load([UPat.var('b'), new UPat()]), ({ ctx, b }) => ctx.get(b)]])

const schedule_uop = (pre: UOp, ctx: ScheduleContext): ScheduleItem => {
  // remove movement ops + substitute LOAD of fused STORE with just the value
  const store_bufs = new Map(pre.src.map((x) => [x.buf_uop, x.src[2]])), sink = graph_rewrite(graph_rewrite(pre, multioutput.add(view_left), store_bufs), view_right)
  // remove extra uops from SINK + substitue BUFFER with DEFINE_GLOBAL
  const si_ctx = new ScheduleItemContext(ctx.var_vals), ast = graph_rewrite(sink, to_si, si_ctx)
  // deal with ASSIGN
  const assign_preloads: UOp[] = []
  if (ctx.assigns.size !== 0) {
    for (const x of [...sink.toposort].toReversed()) {
      // we only allow a kernel to depend on either the before ASSIGN or after ASSIGN version of a BUFFER
      if (x.op === Ops.LOAD && assign_preloads.includes(x.buf_uop)) throw new Error('cycle detected in graph')
      // PRELOAD tells the toposort this kernel should run before ASSIGN
      if (x.op === Ops.PRELOAD) {
        assign_preloads.push(x.buf_uop)
        // if this kernel also assigns to the buffer, we only allow either contiguous or masked views for the LOAD
        const st = x.st_arg!
        if (store_bufs.has(x.buf_uop) && !st.contiguous) {
          // if it has a single view and it's equal when you shrink a contig, it's fine
          const mask = st.views[0].mask
          if (st.views.length !== 1 || mask === undefined || ShapeTracker.from_shape(st.shape).shrink(mask) !== st.shrink(mask)) {
            throw new Error('self operand of augmented assign must be contiguous.\nhelp: consider using .contiguous():\n' + colored('   - a += a.T\n', 'red') + colored('   + a += a.T.contiguous()', 'green'))
          }
        }
      }
    }
  }
  // capture process replay
  if (CAPTURE_PROCESS_REPLAY) {
    throw new NotImplemented()
  }
  return new ScheduleItem(ast, si_ctx.bufs.filter((u) => u.size !== 0).map((u) => u.buffer), dedup([...pre.toposort].map((x) => ctx.ops_metadata.get(x)).filter((m) => m !== undefined)), dedup(assign_preloads))
}

export const PROCESS_REPLAY_CAPTURE = new Map<string, Uint8Array>()
if (get_env('RUN_PROCESS_REPLAY')) {
  throw new NotImplemented()
}
// **** Schedule grouping

export const is_scheduled = (u: UOp): boolean => u.op === Ops.VIEW && u.src.length === 2 && u.src[0].op === Ops.BUFFER
export const uval = (u: UOp): UOp => {
  if (!is_scheduled(u)) throw new Error(`must be a scheduled op ${u}`)
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
  if (cache.keys().some(([_tr, _st]) => tr === _tr && st === _st)) return
  cache.set([tr, st], undefined)
  const rsize = allbufs.get(r)!.st?.size
  if (realizes.has(tr) && tr !== r) {
    // can only fuse contiguous
    // max one reduceop per kernel
    if (!st.contiguous || (st.size !== rsize) || reduce_for_op.has(tr)) set_default(group, r, undefined)
    return set_default(group, tr, undefined)
  }
  for (const tr_next of children.get(tr)!.keys()) {
    // max one reduceop per kernel
    const tr_next_uop = uval(allbufs.get(tr_next)!).base
    if (tr_next_uop.op === Ops.REDUCE_AXIS) return set_default(group, r, undefined)
    // can only fuse contiguous
    const st_childs = dedup(tr_next_uop.src.filter((x) => is_scheduled(x.base) && x.base.buf_uop === tr).map((x) => x.st!))
    if (st_childs.length > 1) return set_default(group, r, undefined)
    recursive_group(tr_next, st.add(st_childs[0]), r, children, allbufs, realizes, reduce_for_op, group, cache)
  }
}
export const get_isolated_children = (r: UOp, reduce_for_op: Map<UOp, UOp>, children: Map<UOp, Map<UOp, undefined>>, allbufs: Map<UOp, UOp>, realizes: Map<UOp, UOp>, group: Map<UOp, undefined>): Map<UOp, undefined> => {
  let [rc_parents, cache] = [[...group.keys()], new Set()]
  while (rc_parents.length) {
    const p = uval(allbufs.get(rc_parents.pop()!)!)
    if (cache.has(p)) continue
    cache.add(p)
    // max one reduceop per kernel
    if (p.op === Ops.REDUCE_AXIS) return new Map()
    rc_parents = [...rc_parents, ...p.src.filter((x) => is_scheduled(x.base) && x.base.buf_uop !== r).map((x) => x.base.buf_uop)]
  }
  // search descendants of the reduceop that can cleanly group
  const descendants = new Map<UOp, undefined>()
  for (const tr of group.keys()) recursive_group(tr, allbufs.get(tr)!.st!, tr, children, allbufs, realizes, reduce_for_op, descendants, new Map())
  return merge_maps([group, [...descendants].some(([tr]) => group.has(tr)) ? new Map() : descendants])
}

/**
 * search the big graph for all the reduceops that need to realize, sometimes group/fuse the reduceop
 */
export const group_realizes = (ctx: ScheduleContext): UOp[][] => {
  // find all reduces, && pair them to a elementwise op. if they can't be cleanly paired, force realize the reduce (or a contig child)
  const reduce_for_op = new Map<UOp, UOp>()
  const reduce_of_const: UOp[] = []
  const double_reduces: UOp[] = []
  for (let [r, r_uop] of ctx.allbufs.entries()) {
    r_uop = uval(r_uop)
    if (r_uop.op !== Ops.REDUCE_AXIS) continue
    if (FUSE_CONV_BW) {
      const x = r_uop.src[0]
      if (is_scheduled(x.base) && uval(x.base).op === r_uop.op && x.base !== x) double_reduces.push(r)
    }
    if (ctx.realizes.has(r)) continue
    let group = new Map<UOp, undefined>()
    recursive_group(r, r_uop.st!, r, ctx.children, ctx.allbufs, ctx.realizes, reduce_for_op, group, new Map())
    // max one reduceop per kernel
    let can_chase = group.keys().every((tr) => !reduce_for_op.has(tr))
    // TODO: forced_realize exists because the scheduler === incapable of checking for this-contained DAGs
    let forced_realize = group.has(r)
    if (!forced_realize && group.size > 1) group = get_isolated_children(r, reduce_for_op, ctx.children, ctx.allbufs, ctx.realizes, group)
    // can only fuse assign if no other assign_target === used in the kernel
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
        // can chase this down to contiguous children
        let st = r_uop.st!
        while (ctx.children.get(tr)!.size === 1) {
          const tr_next = ctx.children.get(tr)!.keys().next().value!
          const tr_next_uop = uval(ctx.allbufs.get(tr_next)!)
          const st_childs = dedup(tr_next_uop.src.filter((x) => is_scheduled(x.base) && x.base.buf_uop === tr).map((x) => x.st!))
          if (st_childs.length > 1) break
          if (st.size !== st_childs[0].size) break
          st = st.add(st_childs[0])
          if (!st.contiguous || tr_next_uop.op === Ops.REDUCE_AXIS) break
          tr = tr_next
        }
        // don't cast to higher size before store (tr can!be realized if forced_realize)
        const tr_uop = uval(ctx.allbufs.get(tr)!)
        if (tr_uop.op === Ops.CAST && tr_uop.dtype.base.itemsize > tr_uop.src[0].dtype.base.itemsize) tr = tr_uop.src[0].base.buf_uop
      }
      group = new Map([[tr, undefined]])
      ctx.realizes.set(tr, tr)
    }
    group.keys().forEach((tr) => reduce_for_op.set(tr, r))
    if (FUSE_ARANGE && r_uop.arg[0] === Ops.ADD && r_uop.src[0].base.op === Ops.CONST) reduce_of_const.push(r)
  }
  // fuse double reduces with no other child
  for (const reduceop of double_reduces) {
    const top_reduce = uval(ctx.allbufs.get(reduceop)!).src[0].base.buf_uop
    if (ctx.children.get(top_reduce)!.size === 1) ctx.realizes.delete(top_reduce)
  }
  for (const rbuf of reduce_of_const) {
    const group = new Map(reduce_for_op.entries().filter(([tr, rop]) => rop === rbuf).map(([tr, rop]) => [tr, undefined]))
    if (group.keys().flatMap((tr) => ctx.tensor_uops.get(tr)!.map((luop) => luop.forced_realize)).some(Boolean)) continue
    const kernel_children = [...group.keys()].flatMap((tr) => [...ctx.children.get(tr)!.keys()]).filter((c) => ![Ops.COPY, Ops.BUFFER_VIEW].includes(uval(ctx.allbufs.get(c)!).op))
    if (kernel_children.length === 0) continue
    for (const [tr] of group) ctx.realizes.delete(tr)
  }
  const output_groups = new Map<UOp, UOp[]>()
  for (const ubuf of ctx.realizes.keys()) set_default(output_groups, reduce_for_op.get(ubuf) || ubuf, []).push(ubuf)
  return [...output_groups.values()]
}
// **** Schedule creation && BFS toposort

// ** ops in the big graph can either be pre-realized || scheduled (fused/realized)

export class UPatScheduled extends UPat {
  constructor(args: Partial<UPatInput> = {}) {
    super(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b'), new UPat(args.op, args.dtype, args.src, args.arg, args.name || 'to_store', args.allow_any_len, args.location, args.custom_early_reject)], undefined, 'base')
  }
}

// ** this is schedule level const folding

const simplify_reduceop = (reduce: UOp, x: UOp): UOp | undefined => {
  if (!all_int(x.shape)) return undefined
  // remove reduce on unmasked const
  const prshape: sint = prod(reduce.arg[1].map((i: number) => x.st!.shape[i]))
  let ret = x.const_arg as sint
  if (reduce.arg[0] === Ops.ADD) ret = mul(ret, prshape)
  else if (reduce.arg[0] === Ops.MUL) ret = pow(ret, prshape)
  else if (reduce.arg[0] === Ops.MAX) { /** NOTE: Ops.MAX is passthrough */ }
  else return undefined
  return reduce.const_like(ret)
}
const found_contiguous = (ctx: ScheduleContext, contig: UOp, base: UOp, b: UOp): undefined => {
  if (contig.src[0].op === Ops.VIEW && contig.src[0].src.length) {
    const old_base = contig.src[0].src[0]
    if (old_base.op === Ops.VIEW && (contig.src[0].st!.invert(old_base.shape)) !== undefined) ctx.contiguous.set(old_base, base.view(contig.src[0].st!))
  }
}
const replace_contiguous = (ctx: ScheduleContext, alu: UOp) => {
  const new_src = [...alu.src]
  for (const [i, s] of alu.src.entries()) {
    if (ctx.contiguous.has(s)) new_src[i] = ctx.contiguous.get(s)!
  }
  if (!is_eq(new_src, alu.src)) return alu.replace({ src: new_src })
}
export const ops_folding = symbolic_simple.add(
  new PatternMatcher<ScheduleContext>([
    // op with size 0 is zero
    [new UPatScheduled(), ({ b, to_store, base }) => base.size === 0 ? base.const_like(0) : undefined],
    // if the uop folded to a CONST we can delete the BUFFER
    [new UPatScheduled({ op: Ops.CONST, name: 'constt' }), ({ b, base, constt }) => base.const_like(constt.const_arg)],
    // DETACH is a NOOP here
    [new UPat(Ops.DETACH).named('detach'), ({ detach }) => detach.src[0]],
    // reduce of size 0 is the identity element
    [new UPat(Ops.REDUCE_AXIS, undefined, [UPat.var('x')]).named('reduce'), ({ reduce, x }) => x.size === 0 && reduce.size !== 0 ? reduce.const_like(identity_element(reduce.arg[0], reduce.dtype)) : undefined],
    // reduce of const is collapsed (TODO: make this a generic rule for stride0)
    [new UPat(Ops.REDUCE_AXIS, undefined, [UPat.cvar('x')]).named('reduce'), ({ reduce, x }) => simplify_reduceop(reduce, x)],
    // CONST doesn't need COPY
    [new UPat(Ops.COPY, undefined, [new UPat(), UPat.cvar('x')]), ({ x }) => x],
    // no COPY to same device, except clone (arg is True)
    [new UPatScheduled({ op: Ops.COPY, src: [new UPat(), new UPat(Ops.VIEW).named('copyin')], name: 'copy' }), ({ base, b, copyin, copy }) => copyin.device === copy.device && copy.arg !== true ? copyin : undefined],
    // support for using a contiguous permuted view instead of the parent view if one exists
    [new UPatScheduled({ op: Ops.CONTIGUOUS, name: 'contig' }), ({ ctx, contig, base, b }) => found_contiguous(ctx, contig, base, b)],
    [new UPat(GroupOp.ALU).named('alu'), ({ ctx, alu }) => replace_contiguous(ctx, alu)],
    // remove CONST/BIND/BUFFER/VIEW from SINK
    [new UPat(Ops.SINK).named('root'), ({ root }) => {
      const new_src = root.src.filter((x) => !x.is_realized && ![Ops.CONST, Ops.BIND].includes(x.base.op))
      return !is_eq(new_src, root.src) ? new UOp(Ops.SINK, root.dtype, new_src, root.arg) : undefined
    }],
  ]),
)

// # ** buffer merging

const merge = (ctx: ScheduleContext, v1: UOp, b1: UOp, v2: UOp, b2: UOp): UOp => {
  if (!(v1.st !== undefined && v2.st !== undefined && v1.st === v2.st)) throw new Error(`implicit movementop ${v1.st} ${v2.st}`)
  // if b2 is realized also realize b1
  if (ctx.realizes.has(b2)) {
    ctx.realizes.set(b1, b1)
    ctx.realizes.delete(b2)
  }
  // ops referring to b2 now ref to b1
  ctx.tensor_uops.set(b1, [...ctx.tensor_uops.get(b1)!, ...ctx.tensor_uops.get(b2)!])
  ctx.tensor_uops.delete(b2)
  // merge
  return v1
}

const merge_realized = (ctx: ScheduleContext, v1: UOp, b1: UOp, v2: UOp, b2: UOp) => {
  // early become
  for (const luop of [...(ctx.tensor_uops.get(b1) || []), ...(ctx.tensor_uops.get(b2) || [])]) ctx.becomes_map.set(luop, b1.view(luop.st!))
  return v1
}
export const merge_bufs = new PatternMatcher<ScheduleContext>([
  // merge base
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b2'), new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b1'), new UPat()]).named('v1')]).named('v2'), ({ ctx, v1, b1, v2, b2 }) => merge(ctx, v1, b1, v2, b2)],
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b2'), new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b1')]).named('v1')]).named('v2'), ({ ctx, v1, b1, v2, b2 }) => merge_realized(ctx, v1, b1, v2, b2)],
  // collapse view
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER), new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER), new UPat()]).view(undefined, { name: 'mv' })]), ({ mv }) => mv],
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER), new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER)]).view(undefined, { name: 'mv' })]), ({ mv }) => mv],
])

// ** this decides which ops get realized

const realize = (ctx: ScheduleContext, b: UOp, to_store: UOp) => void ctx.realizes.set(b, to_store)

export const realize_view = (ctx: ScheduleContext, view: UOp, src: UOp, b: UOp) => {
  if (src.st === undefined) return undefined
  const st = view.st!
  // fold simple pads
  if (st.views.length === 1 && st.views.at(-1)!.mask !== undefined && all_int(src.shape) && resolve(ge(prod(src.shape), prod(st.views.at(-1)!.mask!.map(([x, y]) => sub(y, x)))))) {
    return can_pad(src, ctx.realizes, new Set()) ? undefined : realize(ctx, b, src)
  }
  // early realize before expanz
  if (resolve(lt(prod(src.shape), prod(st.shape))) && !get_env('DONT_REALIZE_EXPAND')) return realize(ctx, b, src)
  // otherwise safety check pads
  return (st.views.every((v) => v.mask === undefined) || can_pad(src, ctx.realizes, new Set())) ? undefined : realize(ctx, b, src)
}
const fold_img_cast = (ctx: ScheduleContext, xb: UOp, view: UOp, b: UOp, to_cast: UOp): UOp | undefined => {
  if (!(xb.dtype instanceof ImageDType) || !ctx.realizes.has(b) || !ctx.realizes.has(xb) || GroupOp.Meta.includes(uval(to_cast).op)) return undefined
  ctx.realizes.delete(b)
  return to_cast.view(view.st!)
}

const sink_outputs = (ctx: ScheduleContext, sink: UOp): undefined => {
  for (const x of sink.src) realize(ctx, x.buf_uop, x)
}
export const do_realize = new PatternMatcher<ScheduleContext>([
  // always realize sinked ops
  [new UPat(Ops.SINK).named('sink'), ({ ctx, sink }) => sink_outputs(ctx, sink)],
  // always realize meta ops
  [new UPatScheduled({ op: [Ops.ASSIGN, Ops.CONTIGUOUS, ...GroupOp.Meta] }), ({ ctx, b, to_store }) => realize(ctx, b, to_store)],
  // realize before expand or unsafe pad ops
  [new UPatScheduled({ name: 'src' }).view(undefined, { name: 'view' }), ({ ctx, view, src, b }) => realize_view(ctx, view, src, b)],
  // don't realize image to image casts
  [new UPatScheduled({ op: Ops.CAST, src: [new UPat(Ops.VIEW, undefined, [UPat.var('xb'), new UPat()]).named('to_cast')], dtype: dtypes.float }).view(undefined, { name: 'view' }), ({ ctx, xb, view, b, to_cast }) => fold_img_cast(ctx, xb, view, b, to_cast)],
  // realize before COPY or BUFFER_VIEW
  [new UPat(Ops.COPY, undefined, [new UPat(), UPat.any([new UPatScheduled(), new UPatScheduled().view()])]), ({ ctx, b, to_store }) => realize(ctx, b, to_store)],
  [new UPat(Ops.BUFFER_VIEW, undefined, [UPat.any([new UPatScheduled(), new UPatScheduled().view()])]), ({ ctx, b, to_store }) => realize(ctx, b, to_store)],
])

// **** rewrite VIEW into LOAD/STORE/VALID or fuse the underlying UOp

const unbind_variable = (ctx: ScheduleContext, bind: UOp, varr: UOp, val: UOp) => {
  if (typeof val.src[1].const_arg !== 'number') throw new Error(`expected BIND value to be int ${val}`)
  const ret = varr.replace({ src: [] })
  ctx.var_vals.set(ret, val.src[1].const_arg)
  return ret.valid(bind.st!)
}

const load_realized = (ctx: ScheduleContext, b: UOp, st: UOp) => {
  // NOTE: if we're assigning to the BUFFER too, PRELOAD tells toposort to place this load before the ASSIGN
  return new UOp(ctx.assigns.has(b) ? Ops.PRELOAD : Ops.LOAD, b.dtype.base, [b, st.st!.to_uop()])
}

const store_or_fuse = (ctx: ScheduleContext, b: UOp, x: UOp, st: UOp) => {
  const m = ctx.tensor_uops.get(b)![0].metadata
  if (m !== undefined) ctx.ops_metadata.set(x, m)
  if (!ctx.realizes.has(b)) return x // collapse BUFFER
  ctx.realizes.set(b, b.store([ShapeTracker.from_shape(st.shape).to_uop(), x]))
  return new UOp(Ops.LOAD, x.dtype, [b, st.st!.to_uop()])
}
export const break_sched = new PatternMatcher<ScheduleContext>([
  // CONST is always fused and generated
  [new UPat(Ops.CONST, undefined, [new UPat(Ops.VIEW).named('st')]).named('x'), ({ x, st }) => UOp.const(x.dtype.base, x.const_arg).valid(st.st!)],
  [new UPat(Ops.BIND, undefined, [UPat.var('varr'), UPat.var('val')]).named('bind'), ({ ctx, bind, varr, val }) => unbind_variable(ctx, bind, varr, val)],
  // VIEW of BUFFER either becomes a LOAD/STORE or we fuse it
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b')]).named('st'), ({ ctx, b, st }) => load_realized(ctx, b, st)],
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('b'), UPat.var('x')]).named('st'), ({ ctx, b, x, st }) => store_or_fuse(ctx, b, x, st)],
])

// # **** Schedule context builder

const append_uop = (ctx: ScheduleContext, view: UOp, buf_uop: UOp): undefined => {
  ctx.allbufs.set(buf_uop, view)
  const op = uval(view)
  if (op.op === Ops.ASSIGN) ctx.assigns.add(buf_uop)
  for (const x of op.base.src) {
    if (is_scheduled(x.base)) set_default(ctx.children, x.base.buf_uop, new Map()).set(buf_uop, undefined)
  }
  // BUFFER_VIEW overrides the underlying buffer
  // TODO: this should be a shrink on the buffer
  if (op.op === Ops.BUFFER_VIEW) {
    const x = op.src[0]
    buffers.set(buf_uop.key, x.buf_uop.buffer.view(view.size, view.dtype, x.st!.views[0].offset as number * x.dtype.itemsize))
  }
  buf_uop.buffer.ref(1)
}
export const create_ctx = new PatternMatcher<ScheduleContext>([[new UPat(Ops.VIEW, undefined, [new UPat(Ops.BUFFER).named('buf_uop'), new UPat()]).named('view'), ({ ctx, view, buf_uop }) => append_uop(ctx, view, buf_uop)]])
// # **** movement ops

export const remove_movement_ops = new PatternMatcher([
  // NOTE: movement ops are always applied to base
  [new UPat(GroupOp.Movement, undefined, [UPat.any([UPat.var('x').view(), UPat.var('x')])]).named('mov'), ({ x, mov }) => x.view(mov.st!)],
  // some masked views can collapse to 0, VIEW(x) -> CONST(VIEW)
  [new UPat(Ops.VIEW).named('view'), ({ view }) => {
    const vm = view.st!.views.at(-1)!.mask
    return vm !== undefined && vm.some((x) => sub(x[1], x[0]) === 0) ? view.const_like(0) : undefined
  }],
  // merge one src views.
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.VIEW, undefined, [new UPat()]).named('v1')]).named('v2'), ({ v1, v2 }) => v1.replace({ arg: add(v1.arg, v2.arg) })],
  // merge unmasked const views
  [new UPat(Ops.VIEW, undefined, [new UPat(Ops.CONST, undefined, [new UPat(Ops.VIEW).named('st')]).named('constt')]).named('view'), ({ st, constt, view }) => (st.st!.add(view.st!)).views.every((v) => v.mask === undefined) ? constt.replace({ src: [st.replace({ arg: st.st!.add(view.st!) })] }) : undefined],
])

// @track_rewrites(named=true)
export const create_schedule_with_vars = (outs: UOp[], skip_check = !DEBUG): [ScheduleItem[], Map<Variable, number>, Map<UOp, UOp>] => {
  if (!skip_check) type_verify([...UOp.sink(...outs).toposort], [tensor_uop_spec])
  // to_uop is removing (many) of the movement ops
  const ctx = new ScheduleContext()
  let sink = add_buffers(UOp.sink(...outs), ctx, new Map())
  // const folding and fusion
  sink = graph_rewrite(sink, remove_movement_ops.add(ops_folding).add(do_realize), ctx)
  sink = graph_rewrite(sink, merge_bufs, ctx)
  // create the scheduler context
  graph_rewrite(sink, create_ctx, ctx)
  // group realizes into kernels
  const store_groups = group_realizes(ctx)
  graph_rewrite(sink, break_sched, ctx)
  // preschedule realize groups
  const prescheduled: ScheduleItem[] = []
  for (const store_uops of store_groups) {
    const stores = store_uops.filter((u) => ctx.realizes.get(u)!.op === Ops.STORE).map((u) => ctx.realizes.get(u)!)
    if (stores.length === 0) continue
    prescheduled.push(schedule_uop(UOp.sink(...stores), ctx))
    // can only schedule once
    for (const buf_uop of store_uops) {
      for (const luop of ctx.tensor_uops.get(buf_uop)!) ctx.becomes_map.set(luop, buf_uop.view(luop.st!))
    }
  }
  // add kernel children
  const schedule_targets = new Map(prescheduled.flatMap((si) => si.outputs.map((out) => [out, si])))
  const graph = new Map<ScheduleItem, ScheduleItem[]>()
  const in_degree = new Map<ScheduleItem, number>()
  for (const si of prescheduled) {
    // realize outputs before a parent === assigned to
    const parents_assigns = dedup([...si.assign_preloads].map((x) => schedule_targets.get(x.buffer)!).filter((xsi) => xsi && xsi !== si))
    for (const assign of parents_assigns) {
      set_default(graph, si, []).push(assign)
      in_degree.set(assign, set_default(in_degree, assign, 0) + 1)
    }
    // realize outputs after all parents are realized
    const scheduled_parents = dedup(si.inputs.map((x) => schedule_targets.get(x)!).filter((xsi) => xsi !== undefined && !parents_assigns.includes(xsi)))
    for (const x of scheduled_parents) {
      set_default(graph, x, []).push(si)
      in_degree.set(si, set_default(in_degree, si, 0) + 1)
    }
  }

  // do BFS
  const queue = prescheduled.filter((si) => set_default(in_degree, si, 0) === 0)
  const schedule: ScheduleItem[] = []
  while (queue.length) {
    const si = queue.shift()
    schedule.push(si!)
    for (const x of set_default(graph, si!, [])) {
      in_degree.set(x, in_degree.get(x)! - 1)
      if (in_degree.get(x) === 0) queue.push(x)
    }
  }
  // confirm everything was scheduled correctly
  const groups = prescheduled.length
  if (schedule.length !== groups) throw new Error(`cycle detected in graph, grouped ${groups} but only scheduled ${schedule.length}`)
  if (DEBUG >= 1 && schedule.length >= 10) console.log(`scheduled ${schedule.length} kernels`)
  return [schedule, ctx.var_vals, ctx.becomes_map]
}
