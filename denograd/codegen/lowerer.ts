import { dtypes, PtrDType } from '../dtype.ts'
import { all_int, assert, is_eq, isinstance, min, partition, range, zip } from '../helpers.ts'
import { graph_rewrite, identity_element, KernelInfo, Ops, PatternMatcher, prod, sint, sint_to_uop, smax, UOp, UPat } from '../ops.ts'
import { Renderer } from '../renderer/index.ts'

// # returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
export const get_contraction = (old_shape: sint[], new_shape: sint[]): number[][] | undefined => {
  const acc_old = old_shape.reduce((acc, val, i) => [...acc, (val as number) * (acc[i - 1] ?? 1)], [] as number[])
  const acc_new = new_shape.reduce((acc, val, i) => [...acc, (val as number) * (acc[i - 1] ?? 1)], [] as number[])
  try {
    const split = acc_new.map((acc) => acc !== 1 ? acc_old.indexOf(acc) + 1 : 0)
    return zip([0, ...split.slice(0, -1)], [...split.slice(0, -1), old_shape.length]).map(([st, ed]) => range(st, ed))
  } catch {
    return undefined
  }
}
// # ***** indexing *****

export const _limit_dims = (dims: sint[], max_sizes: number[]) => {
  // TODO: symbolic shape
  if (!all_int(dims)) return dims
  while (dims.length > max_sizes.length || zip(dims, max_sizes).some(([d, m]) => (d as number) > m)) {
    let found = false
    for (const [i, m] of max_sizes.entries()) {
      if ((dims[i] as number) * (dims[i + 1] as number) <= m) {
        dims = [...dims.slice(0, i), (dims[i] as number) * (dims[i + 1] as number), ...dims.slice(i + 2)]
        found = true
        break
      }
      if (!found) throw new Error(`can!limit dim dims=${dims}, max_size=${max_sizes}`)
    }
  }
  return dims
}
export const get_grouped_dims = (prefix: any, dims: sint[], max_sizes?: number[], reverse = false): UOp[] => {
  if (reverse) dims = dims.toReversed()
  const limited = max_sizes !== undefined ? _limit_dims(dims, max_sizes) : dims
  const raw_idxs = limited.map((s, i) => new UOp(Ops.SPECIAL, dtypes.int, [], [`${prefix}${i}`, s]))
  let ret = raw_idxs
  if (!is_eq(limited, dims)) {
    ret = []
    const contraction = get_contraction(dims, limited)
    if (contraction === undefined) throw new Error(`get_contraction should !be undefined dims=${dims} limited=${limited}`)
    for (let [idx, contraction_group] of zip(raw_idxs, contraction)) {
      for (const c of contraction_group.slice(0, -1)) {
        ret.push(idx.mod(dims[c]))
        idx = idx.idiv(dims[c])
      }
      ret.push(idx)
    }
  }
  return reverse ? ret.toReversed() : ret
}

export class IndexContext {
  constructor(public idxs: UOp[], public ridxs: UOp[], public acc_num: number = 0) {}
}
export const get_index = (ast: UOp, opts: Renderer): IndexContext => {
  const ki = isinstance(ast.arg, KernelInfo) ? ast.arg : new KernelInfo()
  // NOTE: assumes the shape is <global dims> <local dims> <group_for_reduces> <reduces> <upcasts/unrolls>
  const full_shape = ast.full_shape
  const first_upcasted = full_shape.length - ki.upcasted
  // if there's no reduce, this is first_upcasted. assumes reduces are at the end
  const first_reduce = min([first_upcasted, ...ast.toposort.values().filter((x) => x.op === Ops.REDUCE_AXIS).flatMap((x) => x.axis_arg)])
  const local_loads = ast.toposort.values().filter((x) => x.op === Ops.LOAD && x.src[0].op === Ops.DEFINE_LOCAL)
  // NOTE: sum up the reduced axes looking across all local loads, yields the number of grouped reduces
  const group_for_reduces = range(first_reduce, first_upcasted).filter((i) => local_loads.some((l) => l.st_arg.shape[i] !== ast.src[0].st_arg.shape[i])).length
  const global_dims = first_reduce - ki.local_dims

  let idxs
  if (opts.has_local) {
    if (ki.dont_use_locals) {
      if (ki.local_dims !== 0) throw new Error("can't use locals if there's no local dims")
      idxs = get_grouped_dims('idx', full_shape.slice(0, global_dims), opts.global_max, true)
    } else {
      //       # define indexes for GPU-like execution
      idxs = [
        ...get_grouped_dims('gidx', full_shape.slice(0, global_dims), opts.global_max, true),
        ...get_grouped_dims('lidx', full_shape.slice(global_dims, first_reduce + group_for_reduces), opts.local_max),
      ]
    }
  } else {
    //     # all loops are RANGES
    idxs = full_shape.slice(0, first_reduce).map((g, i) => new UOp(Ops.RANGE, dtypes.int, [sint_to_uop(0), sint_to_uop(g)], i))
  }
  // reduce loops
  idxs = [...idxs, ...full_shape.slice(first_reduce + group_for_reduces, first_upcasted).map((g, i) => new UOp(Ops.RANGE, dtypes.int, [sint_to_uop(0), sint_to_uop(g)], i + first_reduce + group_for_reduces))]

  // upcast loops
  for (const [i, g] of full_shape.slice(first_upcasted).entries()) {
    if (!isinstance(g, Number)) throw new Error('needs to be int to upcast/unroll')
    idxs.push(new UOp(Ops.UNROLL, dtypes.int, [UOp.const(dtypes.int.vec(g as number), range(g as number))], [[i + first_upcasted, g]]))
  }
  // late indexes (group for reduce)
  const ridxs = [...idxs]
  for (const a of range(first_reduce, first_reduce + group_for_reduces)) {
    ridxs[a] = new UOp(Ops.RANGE, dtypes.int, [sint_to_uop(0), sint_to_uop(full_shape[a])], 1000 + a)
  }
  return new IndexContext(idxs, ridxs)
}
// # ***** lowering (given index) *****

export const lower_reduce_axis = (ctx: IndexContext, x: UOp): UOp => {
  // NOTE: always using ridxs is fine here
  const [reduce_range, reduce_expand] = partition(x.axis_arg.map((i) => ctx.ridxs[i]), (y) => y.op === Ops.RANGE)
  if (reduce_expand.some((x) => x.op !== Ops.UNROLL)) throw new Error(`not all UNROLLS in ${reduce_expand} for ${x.axis_arg}`)
  const alu_op: Ops = x.arg[0]
  let ret = x.src[0]
  const contract_axis = reduce_expand.flatMap((x) => x.arg)
  if (contract_axis.length) {
    ret = new UOp(Ops.CONTRACT, x.dtype.vec(prod(contract_axis.map((x) => x[1]))), [ret], contract_axis)
    ret = range(ret.dtype.count).map((i) => ret.gep(i)).reduce((x, y) => x.alu(alu_op, y))
  }
  if (!reduce_range.length) return ret
  // create ACC and assign
  const acc = new UOp(Ops.DEFINE_ACC, x.dtype, [x.const_like(identity_element(alu_op, x.dtype.scalar())), ...reduce_range], [ctx.acc_num + 0])
  ctx = { ...ctx, acc_num: ctx.acc_num + 1 } // not sure, ctx.acc_num += 1 makes tests fail
  return acc.assign(acc.alu(alu_op, ret))
}
export const lower_load_store = (ctx: IndexContext, x: UOp): UOp => {
  let [idx, valid] = x.st_arg.to_indexed_uops(x.op === Ops.LOAD && x.src[0].op === Ops.DEFINE_LOCAL ? ctx.ridxs : ctx.idxs)
  const buf = x.src[0]
  if (x.op === Ops.LOAD) {
    const barrier = x.src[0].op === Ops.DEFINE_LOCAL ? [new UOp(Ops.BARRIER, dtypes.void, [x.src[2]])] : []
    return new UOp(Ops.LOAD, x.dtype, [buf.index(idx, valid), ...barrier])
  }
  // NOTE: only store the local reduceop in the threads that are actually doing the reduce
  let store_back
  if ((x.src[0].dtype as PtrDType).local && x.src[2].op === Ops.ASSIGN) {
    const reduce_input = x.src[2].src[1].src[1] !== x.src[2].src[0] ? x.src[2].src[1].src[1] : x.src[2].src[1].src[0]
    store_back = reduce_input.op === Ops.LOAD && (reduce_input.src[0].dtype as PtrDType).local
  } else store_back = false
  // NOTE: If we're storing the reduced value back into each thread, need to zero-out the reduced axes
  if (store_back) idx = x.st_arg.to_indexed_uops(ctx.idxs.map((u) => x.src[2].src.includes(u) ? u.const_like(0) : u))[0]
  if ((!(x.src[0].dtype as PtrDType).local) || store_back) {
    for (const [oidx, ridx] of zip(ctx.idxs, ctx.ridxs)) {
      if (oidx !== ridx) valid = valid.mul(oidx.eq(0))
    }
  }
  return new UOp(Ops.STORE, dtypes.void, [buf.index(idx, valid), x.src[2]])
}
export const pm_lowerer = new PatternMatcher<Record<string, UOp> & { ctx: IndexContext }, UOp>([
  [new UPat(Ops.REDUCE_AXIS).named('x'), ({ ctx, x }) => lower_reduce_axis(ctx, x)],
  [new UPat(Ops.VALID, undefined, [new UPat(Ops.VIEW)], undefined, 'x'), ({ ctx, x }) => x.st_arg.to_indexed_uops(ctx.idxs)[1]],
  // rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  [new UPat([Ops.LOAD, Ops.STORE], undefined, [new UPat(), new UPat(Ops.VIEW)], undefined, 'x', true), ({ ctx, x }) => lower_load_store(ctx, x)],
  [new UPat(Ops.INDEX, undefined, [UPat.var('b'), UPat.var('idx'), UPat.const(dtypes.bool, true)]), ({ b, idx }) => b.index(idx)],
])

export const rewrite_shapetracker_with_index = (ast: UOp, opts: Renderer) => graph_rewrite(ast, pm_lowerer, get_index(ast, opts))
