import { NotImplemented } from './helpers.ts'
import { type Ops, PatternMatcher, type sint, type UOp } from './ops.ts'

export const all_reduce = (bop: Ops, lbs: UOp[]): UOp[] => {
  throw new NotImplemented()
}
export const to_sharded = (lbs: UOp[], axis: number, bounds: [number, number][]): UOp[] => {
  throw new NotImplemented()
}

export const alu_multi = (root: UOp): UOp => {
  throw new NotImplemented()
}

export const reduce_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}
export const _shape_to_single_shard = (axis: number, shape: sint[], lb: UOp): sint[] => {
  throw new NotImplemented()
}

export const reshape_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const expand_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const pad_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const permute_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const shrink_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const stride_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const copy_multi = (multi: UOp, device: UOp): UOp => {
  throw new NotImplemented()
}

export const assign_multi = (dest: UOp, src: UOp): UOp => {
  throw new NotImplemented()
}
export const passthrough_multi = (root: UOp, multi: UOp): UOp => {
  throw new NotImplemented()
}

export const multi_pm = new PatternMatcher([])

export const get_multi_map = (big_sink: UOp): Map<UOp, UOp> => {
  throw new NotImplemented()
}
