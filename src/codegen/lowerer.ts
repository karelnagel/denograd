import { dtypes, PtrDType } from '../dtype.ts'
import { all, assert, isinstance, len, min, partition, prod, range, sum, zip } from '../helpers.ts'
import { graph_rewrite, identity_element, KernelInfo, Ops, PatternMatcher, sint, sint_to_uop, UOp, UPat } from '../ops.ts'
import { Renderer } from '../renderer/index.ts'

// # returns the axes to create new_shape if new_shape can be created by combining axis from old_shape
export const get_contraction = (old_shape: sint[], new_shape: sint[]): number[][] | undefined => {
  throw new Error('not implemented')
}
// # ***** indexing *****

export const _limit_dims = (dims: sint[], max_sizes: number[]) => {
  throw new Error('not implemented')
}
export const get_grouped_dims = (prefix: any, dims: sint[], max_sizes?: number[], reverse = false): UOp[] => {
  throw new Error('not implemented')
}
//   pass
export type IndexContext = {
  idxs: UOp[]
  ridxs: UOp[]
  acc_num: number
}
const get_index = (ast: UOp, opts: Renderer): IndexContext => {
  const ki = isinstance(ast.arg, KernelInfo) ? ast.arg : new KernelInfo()
  // NOTE: assumes the shape is <global dims> <local dims> <group_for_reduces> <reduces> <upcasts/unrolls>
  const full_shape = ast.full_shape
  const first_upcasted = len(full_shape) - ki.upcasted
  // if there's no reduce, this is first_upcasted. assumes reduces are at the end
  const first_reduce = min([first_upcasted, ...ast.toposort.values().filter((x) => x.op === Ops.REDUCE_AXIS).flatMap((x) => x.axis_arg)])
  const local_loads = ast.toposort.values().filter((x) => x.op === Ops.LOAD && x.src[0].op === Ops.DEFINE_LOCAL)
  // NOTE: sum up the reduced axes looking across all local loads, yields the number of grouped reduces
  const group_for_reduces = sum(range(first_reduce, first_upcasted).map((i) => Number(local_loads.some((l) => l.st_arg.shape[i] !== ast.src[0].st_arg.shape[i]))))
  const global_dims = first_reduce - ki.local_dims
  let idxs
  if (opts.has_local) {
    if (ki.dont_use_locals) {
      assert(ki.local_dims === 0, "can't use locals if there's no local dims")
      idxs = get_grouped_dims('idx', full_shape.slice(0, global_dims), opts.global_max, true)
    } else {
      //       # define indexes for GPU-like execution
      idxs = [...get_grouped_dims('gidx', full_shape.slice(0, global_dims), opts.global_max, true), ...get_grouped_dims('lidx', full_shape.slice(global_dims, first_reduce + group_for_reduces), opts.local_max)]
    }
  } else {
    //     # all loops are RANGES
    idxs = full_shape.slice(0, first_reduce).map((g, i) => new UOp({ op: Ops.RANGE, dtype: dtypes.int, src: [sint_to_uop(0), sint_to_uop(g)], arg: i }))
  }
  // reduce loops
  idxs = [...idxs, ...full_shape.slice(first_reduce + group_for_reduces, first_upcasted).map((g, i) => new UOp({ op: Ops.RANGE, dtype: dtypes.int, src: [sint_to_uop(0), sint_to_uop(g)], arg: i + first_reduce + group_for_reduces }))]

  // upcast loops
  for (const [i, g] of full_shape.slice(first_upcasted).entries()) {
    assert(isinstance(g, Number), 'needs to be int to upcast/unroll')
    idxs.push(new UOp({ op: Ops.EXPAND, dtype: dtypes.int, src: [UOp.const(dtypes.int.vec(g as number), range(g as number))], arg: [[i + first_upcasted, g]] })) // TODO: not sure about g, but it is also sint in tiny
  }
  // late indexes (group for reduce)
  const ridxs = [...idxs]
  for (const a of range(first_reduce, first_reduce + group_for_reduces)) {
    ridxs[a] = new UOp({ op: Ops.RANGE, dtype: dtypes.int, src: [sint_to_uop(0), sint_to_uop(full_shape[a])], arg: 1000 + a })
  }
  return { idxs, ridxs, acc_num: 0 }
}
// # ***** lowering (given index) *****

const lower_reduce_axis = (ctx: IndexContext, x: UOp): UOp => {
  // NOTE: always using ridxs is fine here
  const [reduce_range, reduce_expand] = partition(x.axis_arg.map((i) => ctx.ridxs[i]), (y) => y.op === Ops.RANGE)
  assert(all(reduce_expand.map((x) => x.op === Ops.EXPAND)), `not all EXPANDS in ${reduce_expand} for ${x.axis_arg}`)
  const alu_op: Ops = x.arg[0]
  let ret = x.src[0]
  const contract_axis = reduce_expand.flatMap((x) => x.arg)
  if (len(contract_axis)) {
    ret = new UOp({ op: Ops.CONTRACT, dtype: x.dtype.vec(prod(contract_axis.map((x) => x[1]))), src: [ret], arg: contract_axis })
    ret = range(ret.dtype.count).map((i) => ret.gep(i)).reduce((x, y) => x.alu(alu_op, y))
  }
  if (!len(reduce_range)) return ret
  // create ACC and assign
  const acc = new UOp({ op: Ops.DEFINE_ACC, dtype: x.dtype, src: [x.const_like(identity_element(alu_op, x.dtype.scalar())), ...reduce_range], arg: [ctx.acc_num] })
  ctx.acc_num += 1
  return acc.assign(acc.alu(alu_op, ret))
}
const lower_load_store = (ctx: IndexContext, x: UOp): UOp => {
  let [idx, valid] = x.st_arg.to_indexed_uops(x.op === Ops.LOAD && x.src[0].op === Ops.DEFINE_LOCAL ? ctx.ridxs : ctx.idxs)
  // TODO: check has_valid in UPat, not here
  let has_valid = valid.op !== Ops.CONST || valid.arg !== true
  const buf = x.src[0]
  if (x.op === Ops.LOAD) {
    const barrier = x.src[0].op === Ops.DEFINE_LOCAL ? [new UOp({ op: Ops.BARRIER, dtype: dtypes.void, src: [x.src[2]] })] : []
    return new UOp({ op: Ops.LOAD, dtype: x.dtype, src: [buf.index(idx, has_valid ? valid : undefined), ...barrier] })
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
    has_valid = valid.op !== Ops.CONST || valid.arg !== true
  }
  return new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [buf.index(idx, has_valid ? valid : undefined), x.src[2]] })
}
export const pm_lowerer = new PatternMatcher<Record<string, UOp> & { ctx: IndexContext }, UOp>([
  [new UPat({ op: Ops.REDUCE_AXIS, name: 'x' }), ({ ctx, x }) => lower_reduce_axis(ctx, x)],
  [new UPat({ op: Ops.VALID, src: [new UPat({ op: Ops.VIEW })], name: 'x' }), ({ ctx, x }) => x.st_arg.to_indexed_uops(ctx.idxs)[1]],
  // rewrite LOAD/STORE VIEW to LOAD/STORE with indexed
  [new UPat({ op: [Ops.LOAD, Ops.STORE], src: [new UPat({}), new UPat({ op: Ops.VIEW })], allow_any_len: true, name: 'x' }), ({ ctx, x }) => lower_load_store(ctx, x)],
])

export const rewrite_shapetracker_with_index = (ast: UOp, opts: Renderer) => graph_rewrite(ast, pm_lowerer, get_index(ast, opts))
