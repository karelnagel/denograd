import { type DeviceType, uop_realized } from './device.ts'
import type { DType } from './dtype.ts'
import { add, all_same, NotImplemented, range, sum, zip } from './helpers.ts'
import { type ConstLike, MathTrait, type Ops, type sint, UOp } from './ops.ts'

const all_reduce = (bop: Ops, lbs: UOp[]): UOp[] => {
  throw new NotImplemented()
  //   assert all_int(lbs[0].shape), f"does not support symbolic shape {lbs[0].shape}"
  //   assert all_same([lb.shape[0] for lb in lbs]), "allreduce with uneven shards is undefined"
  //   n_lbs, shape, numel = len(lbs), lbs[0].shape, prod(lbs[0].shape)
  //   # ring allreduce doesn't provide a benefit with only 2 nodes or where number of elements is less than 256k (empirically)
  //   # fallback to naive allreduce to save on kernel dispatch, chunking and reassembling chunks.
  //   use_ring = (RING >= 2 or (n_lbs > 2 and numel > getenv("RING_ALLREDUCE_THRESHOLD", 256_000) and RING >= 1))
  //   if DEBUG >= 2: print(f"{'RING ALLREDUCE' if use_ring else 'NAIVE ALLREDUCE'} {n_lbs}x{numel} | {lbs[0].dtype}")
  //   if not use_ring: return [functools.reduce(lambda x,y: x.alu(bop, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]

  //   factor = next((f for f in [32, 16, 8, 4, 2] if numel % f == 0), 1)
  //   base, left = (numel // factor) // n_lbs, (numel // factor) % n_lbs
  //   chunk_sizes = [(base + 1) * factor] * left + [base * factor] * (n_lbs - left)
  //   chunks = list(itertools.pairwise(itertools.accumulate(chunk_sizes, initial=0)))
  //   chunked = [[lb.reshape((numel,)).shrink(((s,e),)) for s,e in chunks] for lb in lbs]

  //   # scatter-reduce
  //   for step in range(n_lbs-1):
  //     for i in range(len(chunks)):
  //       src, dest = (i+step)%n_lbs, (i+step+1)%n_lbs
  //       chunked[dest][i] = chunked[dest][i].alu(bop, chunked[src][i].copy_to_device(chunked[dest][i].device))

  //   # allgather
  //   for step in range(n_lbs-1):
  //     for i in range(len(chunks)):
  //       src, dest = (i+step-1)%n_lbs, (i+step)%n_lbs
  //       chunked[dest][i] = chunked[src][i].copy_to_device(chunked[dest][i].device)

  //   # assemble chunks back
  //   pads = [((s,numel-e),) for s,e in chunks]
  //   return [functools.reduce(operator.add, [c.pad(pad) for pad,c in zip(pads,lb_c)]).reshape(shape) for lb_c in chunked]
}
const to_sharded = (lbs: UOp[], axis: number, bounds: [number, number][]): UOp[] => {
  throw new NotImplemented()
  //   if lbs[0].shape[axis] % len(lbs) != 0: raise RuntimeError(f"multi axis uneven: {lbs[0].shape=} {axis=} {len(lbs)=}, bounds={bounds}")
  //   return [lb.shrink(tuple((0,s) if a != axis else bound for a,s in enumerate(lb.shape))) for i, (bound, lb) in enumerate(zip(bounds, lbs))]
}
export class MultiLazyBuffer extends MathTrait<MultiLazyBuffer> {
  lbs: UOp[]
  axis?: number
  dtype: DType
  device: DeviceType[]
  real: boolean[]
  constructor(lbs: UOp[], axis?: number, real?: boolean[]) {
    super()
    if (!lbs.every((x) => x instanceof UOp) || !lbs.length) throw new Error('all lbs must be LazyBuffers, and we need at least one of them')
    if (!all_same(lbs.map((x) => x.dtype))) throw new Error(`all multilazybuffer needs same dtype, getting ${lbs.map((x) => x.dtype)}`)
    this.lbs = lbs, this.axis = axis, this.dtype = lbs[0].dtype, this.device = lbs.map((x) => x.device), this.real = real?.length ? real : range(lbs.length).map(() => true)
  }
  get shape() {
    return this.real_lbs[0].shape.entries().map(([a, s]) => a === this.axis ? sum(this.real_lbs.map((y) => y.shape[a])) : s)
  }
  get size() {
    return sum(this.real_lbs.map((x) => x.size))
  }
  get real_lbs() {
    return zip(this.lbs, this.real).filter(([lb, r]) => r).map(([lb, r]) => lb)
  }
  get bounds() {
    if (this.axis === undefined) throw new Error('bounds is not defined when axis is None')
    const sizes = this.lbs.map((lb) => lb.shape[this.axis!])
    const acc: sint[] = [0]
    for (const size of sizes) acc.push(add(acc[acc.length - 1], size))
    return acc.slice(1).map((v, i) => [acc[i], v] as [number, number])
  }

  override toString(): string {
    return `<MLB axis=${this.axis} real=${this.real}\n${this.lbs.map((x) => `${x.device} ${x.st}`).join('\n')}>`
  }

  copy_to_device = (device: DeviceType): UOp => {
    throw new NotImplemented()
    //     # if we already have a copy on the device, return that
    //     if self.axis is None: return next((lb for lb in self.real_lbs if lb.device == device), self.real_lbs[0].copy_to_device(device))
    //     # copy lbs to device, pad to final shape, and sum
    //     llbs:list[UOp] = []
    //     for lb,real,(start,end) in zip(self.lbs, self.real, self.bounds):
    //       if not real: continue
    //       pad_arg = tuple((0,0) if a != self.axis else (start, self.bounds[-1][1]-end) for a in range(len(lb.shape)))
    //       llbs.append(lb.copy_to_device(device).pad(pad_arg))
    //     return functools.reduce(operator.add, llbs)
  }

  //   # passthroughs
  get is_realized() {
    return this.real_lbs.every((lb) => uop_realized(lb.base) !== undefined)
  }
  cast = (dtype: DType, bitcast = false) => new MultiLazyBuffer(this.lbs.map((x) => x.cast(dtype, bitcast)), this.axis, this.real)
  override const_like = (b: ConstLike) => new MultiLazyBuffer(this.lbs.map((x) => x.const_like(b)), this.axis, this.real)
  assign = (x: MultiLazyBuffer) => new MultiLazyBuffer(zip(this.lbs, x.lbs).map(([s, d]) => s.assign(d)), this.axis, this.real)
  contiguous = () => new MultiLazyBuffer(this.lbs.map((x) => x.contiguous()), this.axis, this.real)
  clone = () => new MultiLazyBuffer(this.lbs.map((lb) => lb.clone()), this.axis, this.real)
  detach = () => new MultiLazyBuffer(this.lbs.map((lb) => lb.detach()), this.axis, this.real)
  get toposort() {
    return new Set(this.lbs.flatMap((x) => [...x.toposort].map((l) => l)))
  }
  // elementwise is simple
  override alu = (op: Ops, ...in_srcs: MultiLazyBuffer[]): MultiLazyBuffer => {
    throw new NotImplemented()
    //     msrcs = (self,)+in_srcs
    //     assert all(isinstance(x, MultiLazyBuffer) for x in msrcs), f"all buffers must be MultiLazyBuffer {msrcs}"
    //     assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

    //     # NOTE: they all have to share an axis, we always choose [-1]
    //     axis, bounds = axes[-1] if len(axes := dedup([(x.axis, x.bounds) for x in msrcs if x.axis is not None])) else (None, None)
    //     srcs:list[list[UOp]] = []
    //     not_all_real = not all(all(mlb.real) for mlb in msrcs)
    //     new_real = [all(transposed) for transposed in zip(*[mlb.real for mlb in msrcs])] if not_all_real else self.real
    //     assert any(new_real), "output contains no real lb"
    //     for mlb in msrcs:
    //       if (mlb.axis == axis and (mlb.axis is None or mlb.bounds == bounds)) or not_all_real: srcs.append(mlb.lbs)
    //       else:
    //         assert axis is not None and bounds is not None
    //         if mlb.axis is None: srcs.append(to_sharded(mlb.lbs, axis, bounds))
    //         else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.lbs], axis, bounds))
    //     new_real_lbs:dict[int,UOp] = {i:lsrcs[0].alu(op, *lsrcs[1:]) for i,(lsrcs,r) in enumerate(zip(zip(*srcs), new_real)) if r}
    //     # NOTE: const dtype should match real
    //     new_dtype = next(iter(new_real_lbs.values())).dtype
    //     return MultiLazyBuffer([new_real_lbs.get(i, lsrcs[0].const_like(0).cast(new_dtype)) for i,lsrcs in enumerate(zip(*srcs))], axis, new_real)
  }
  r = (op: Ops, axis: number[]): MultiLazyBuffer => {
    throw new NotImplemented()
    //     if self.axis is not None and self.axis in axis:
    //       # all-reduce on sharded axes
    //       reduced_parts = [(x if r else x.const_like(0)).r(op, axis) for x,r in zip(self.lbs, self.real)]
    //       # if all partitions are real, do all_reduce
    //       if all(self.real): return MultiLazyBuffer(all_reduce(op, reduced_parts), None)
    //       # only one partition is real, keep it
    //       return MultiLazyBuffer(reduced_parts, None, self.real)
    //     # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
    //     return MultiLazyBuffer([x.r(op, axis) for x in self.lbs], self.axis, self.real)
  }
  // *** movement ops ***

  _shape_to_single_shard = (shape: sint[], lb: UOp): sint[] => {
    throw new NotImplemented()
    //     return tuple(lb.shape[self.axis] if a == self.axis else s for a,s in enumerate(shape))
  }
  reshape = (arg: sint[]) => {
    throw new NotImplemented()
    //     if self.axis is None: return MultiLazyBuffer([x.reshape(arg) for x in self.lbs], None, self.real)
    //     assert prod(self.shape) == prod(arg), "reshape must maintain prod(shape)"
    //     arg_acc:list[sint] = list(itertools.accumulate(arg, operator.mul, initial=1))
    //     # new_axis is the last one that preserves prod(prior to new_axis) and must not move items between shards
    //     # todo: what to do about shrinking to self.shape[self.axis]==1 len(self.real_lbs)==1?
    //     new_axis = len(arg_acc) - arg_acc[::-1].index(prod(self.shape[:self.axis])) - 1
    //     assert all(prod(lb.shape[self.axis:])%prod(arg[new_axis+1:])==0 for lb in self.lbs), f"reshape cannot move items between shards {self=} {arg=}"
    //     lbs = [x.reshape(tuple(s if a!=new_axis else prod(x.shape[self.axis:])//prod(arg[new_axis+1:]) for a,s in enumerate(arg))) for x in self.lbs]
    //     return MultiLazyBuffer(lbs, new_axis, self.real)
  }
  pad = (arg: [sint, sint][]) => {
    throw new NotImplemented()
    //     assert self.axis is None or arg[self.axis] == (0,0) or not all(self.real), f"padding not supported for {arg=}"
    //     # pad on shard axis -> fill others with zeros and set real to all True
    //     if self.axis is not None and arg[self.axis] != (0,0):
    //       # pad back to whole axis, remove real mask
    //       assert all(arg[i] == (0, 0) for i in range(len(self.shape)) if i != self.axis), "cannot pad sharded and non-sharded axis at the same time"
    //       dim, bound = sum(lb.shape[self.axis] for lb in self.lbs), self.bounds[self.real.index(True)]
    //       assert arg[self.axis] == (bound[0], dim-bound[1]), "can only pad to whole axis"
    //       return MultiLazyBuffer([x if r else x.const_like(0) for x,r in zip(self.lbs, self.real)], self.axis)
    //     return MultiLazyBuffer([x.pad(arg) for x in self.lbs], self.axis, self.real)
  }
  expand = (arg: sint[]) => {
    throw new NotImplemented()
    //     # NOTE: this assert isn't needed, sharded axis can have dim 1
    //     assert self.axis is None or arg[self.axis] == self.shape[self.axis], f"expand not supported on sharded axis {arg=}"
    //     return MultiLazyBuffer([x.expand(self._shape_to_single_shard(arg, x)) for x in self.lbs], self.axis, self.real)
  }
  permute = (arg: number[]) => {
    throw new NotImplemented()
    // all permutes supported!
    //     return MultiLazyBuffer([x.permute(arg) for x in self.lbs], arg.index(self.axis) if self.axis is not None else None, self.real)
  }
  shrink = (arg: [sint, sint][]) => {
    throw new NotImplemented()
    //     assert self.axis is None or arg[self.axis] == (0, self.shape[self.axis]) or arg[self.axis] in self.bounds, f"shrinking not supported for {arg=}"
    //     if self.axis is not None and arg[self.axis] in self.bounds and arg[self.axis] != (0, self.shape[self.axis]):
    //       assert all(arg[i] == (0, s) or i == self.axis for i,s in enumerate(self.shape)), "cannot shrink sharded and non-sharded axis at the same time"
    //       # NOTE: shrink on the shard axis is only allowed when result is a single partition, denoted by the new real
    //       idx = self.bounds.index(arg[self.axis])
    //       # zero out other lbs to not create lb reference
    //       return MultiLazyBuffer([lb if i==idx else lb.const_like(0) for i,lb in enumerate(self.lbs)], self.axis, [i==idx for i in range(len(self.lbs))])
    //     return MultiLazyBuffer([x.shrink(tuple((0, x.shape[self.axis]) if a == self.axis else s for a,s in enumerate(arg))) for x in self.lbs],
    //                            self.axis, self.real)
  }
  stride = (arg: number[]) => {
    throw new NotImplemented()
    //     assert self.axis is None or arg[self.axis] == 1, "flipping not supported on sharded axis"
    //     return MultiLazyBuffer([x.stride(arg) for x in self.lbs], self.axis, self.real)
  }
}
