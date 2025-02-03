import type { Buffer } from '../device.ts'
import { type ArrayMap, NotImplemented } from '../helpers.ts'
import type { Variable } from '../ops.ts'
import { Estimates } from '../renderer/index.ts'
import { type ExecItem, Runner } from './realize.ts'

export class GraphException extends Error {}

export const apply_graph_to_jit = (jit_cache: ExecItem[], input_rawbuffers: Buffer[], var_vals: Map<Variable, number>, max_batch_size = 0): ExecItem[] => {
  throw new NotImplemented()
}
export const get_input_replace = (jit_cache: ExecItem[], input_rawbuffers: Buffer[]): ArrayMap<[number, number], number> => {
  throw new NotImplemented()
}

export class GraphRunner extends Runner {
  constructor(jit_cache: ExecItem[], input_rawbuffers: Buffer[], var_vals: Map<Variable, number>) {
    super('', 'CLANG', new Estimates())
    throw new NotImplemented()
  }

  updated_vars = (var_vals: Map<Variable, number>) => {
    throw new NotImplemented()
  }
  updated_launch_dims = (var_vals: Map<Variable, number>) => {
    throw new NotImplemented()
  }
  _access_resources = (rawbufs: Buffer[], write: number[], new_dependency: any) => {
    throw new NotImplemented()
  }
}

// a marker for your graph supporting multiple devices of the same type
export class MultiGraphRunner extends GraphRunner {}

export const update_depends = (depends: Set<Buffer | undefined>, jit_cache: ExecItem[]) => {
  throw new NotImplemented()
}
// ReturnType = TypeVar('ReturnType')
export class CapturedJit<ReturnType> {
  constructor() {
    throw new NotImplemented()
  }
}
export const _prepare_jit_inputs = (args: any, kwargs: any) => {
  throw new NotImplemented()
}
export class TinyJit<ReturnType> {
  constructor() {
    throw new NotImplemented()
  }
}
