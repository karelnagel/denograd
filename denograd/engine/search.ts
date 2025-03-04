import type { Kernel } from '../codegen/kernel.ts'
import { Buffer, type Compiler, Device, type Program } from '../device.ts'
import { env } from '../env/index.ts'
import { idiv, isInf, mod, NotImplemented, prod, range, zip } from '../helpers.ts'
import { sym_infer, type UOp, type Variable } from '../ops.ts'
import type { ProgramSpec } from '../renderer/index.ts'
import { Tensor } from '../tensor.ts'
import { CompiledRunner } from './realize.ts'

// TODO: Causes circular import
// export const actions: Opt[] = [
//   ...range(6).flatMap((axis) => [0, 2, 3, 4, 5, 7].map((amt) => new Opt(OptOps.UPCAST, axis, amt))),
//   ...range(5).flatMap((axis) => [0, 4, 7].map((amt) => new Opt(OptOps.UNROLL, axis, amt))),
//   ...range(6).flatMap((axis) => [2, 3, 4, 8, 13, 16, 29].map((amt) => new Opt(OptOps.LOCAL, axis, amt))),
//   ...range(3).flatMap((axis) => [13, 16, 28, 29, 32, 49, 64, 256].map((amt) => new Opt(OptOps.GROUPTOP, axis, amt))),
//   ...range(3).flatMap((axis) => [0, 4, 8, 16].map((amt) => new Opt(OptOps.GROUP, axis, amt))),
//   ...(get_number_env('BEAM_PADTO', 1) ? range(7).flatMap((axis) => [32].map((amt) => new Opt(OptOps.PADTO, axis, amt))) : []),
//   new Opt(OptOps.LOCAL, 0, 32),
//   new Opt(OptOps.LOCAL, 6, 2),
//   new Opt(OptOps.TC, 0, 0),
//   ...range(9).map((axis) => new Opt(OptOps.TC, axis, get_number_env('TC_OPT', 2))), // covers resnet kernels (3 global * 3 reduce)
//   ...range(5).flatMap((axis) => range(axis + 1, 5).map((amt) => new Opt(OptOps.SWAP, axis, amt))),
//   ...(get_env('NOLOCALS') ? [new Opt(OptOps.NOLOCALS)] : []),
// ]

export const _get_test_global_size = (global_size: number[], max_global_size: number, var_vals: Map<UOp, number>): [number[], number] => {
  let [test_global_size, factor] = [global_size.map((sz) => sym_infer(sz, var_vals)), 1]
  while (prod(test_global_size) > max_global_size) {
    for (const j of range(global_size.length - 1, -1, -1)) {
      if (test_global_size[j] > 16) {
        test_global_size[j] = idiv(test_global_size[j], 2)
        factor *= 2
        break
      }
    }
  }
  return [test_global_size, factor]
}

export const _time_program = async (p: ProgramSpec, lib: Uint8Array, var_vals: Map<Variable, number>, rawbufs: Buffer[], early_stop?: number, max_global_size = 65536, clear_l2 = false, cnt = 3, name = 'test'): Promise<number[]> => {
  let factor = 1, global_size, car
  if (p.global_size !== undefined && max_global_size !== undefined) {
    ;[global_size, factor] = _get_test_global_size(p.global_size, max_global_size, var_vals)
    p.global_size = global_size
  }
  try {
    car = await CompiledRunner.init(p, lib)
  } catch {
    return range(cnt).map((x) => Infinity)
  }
  const tms = []
  const input_bufs = car.p.globals.map((i) => rawbufs[i])
  for (const _ of range(cnt)) {
    if (clear_l2) {
      const dev = Device.get(p.device)
      if ('invalidate_caches' in dev) (dev.invalidate_caches as any)()
      else await Tensor.ones([1024, 1024]).contiguous().realize(undefined, false)
      //TODO:  with Context(DEBUG=0, BEAM=0, CAPTURING=0, TRACK_MATCH_STATS=0): Tensor.ones(1024,1024).contiguous().realize(do_update_stats=false)
    }
    tms.push((await car.call(input_bufs, var_vals as Map<UOp, number>, true))! * factor)
    if (early_stop !== undefined && early_stop < Math.min(...tms)) break
  }
  return tms
}

class TimeoutException extends Error {}
export const timeout_handler = (signum: any, frame: any) => {
  throw new Error('Timout handler')
}

export const _try_compile_linearized_w_idx = (x: [number, Kernel], compiler: Compiler): [number, [ProgramSpec, Uint8Array, number | undefined] | undefined] => {
  throw new NotImplemented()
}

export const _ensure_buffer_alloc = (bufs: Buffer[]): Buffer[] => bufs.map((buf) => buf.ensure_allocated())

// // *** external API ***

export const bufs_from_lin = (lin: Kernel, allocate = true): Buffer[] => {
  throw new NotImplemented()
}
// get dictionary of all possible actions
export const get_kernel_actions = (lin: Kernel, include_0 = true): Map<number, Kernel> => {
  throw new NotImplemented()
}

export const [beam_pool, BEAM_DEBUG, CAPTURE_BEAM] = [undefined, env.get('BEAM_DEBUG'), env.get('CAPTURE_BEAM', '')]
export const beam_search = (lin: Kernel, rawbufs: Buffer[], amt: number, allow_test_size = true, disable_cache = env.get('IGNORE_BEAM_CACHE')): Kernel => {
  throw new NotImplemented()
}

export const optimize_local_size = async (_prg: Program, global_size: number[], rawbufs: Buffer[]): Promise<number[]> => {
  const test_rawbuffers = rawbufs.slice(1).includes(rawbufs[0]) ? [new Buffer(rawbufs[0].device, rawbufs[0].size, rawbufs[0].dtype).allocate(), ...rawbufs.slice(1)] : rawbufs
  const MAX_WORKGROUP = 1024
  const local_dims = global_size.map((sz) => [...new Set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP])].filter((x) => x <= sz))
  const local_sizes = [...local_dims.reduce((acc, curr) => acc.flatMap((x) => curr.map((y) => [...x, y])), [[]] as number[][])].filter((x) => prod(x) <= MAX_WORKGROUP).flatMap((x) => [x, x]) // try each valid size twice
  const try_exec = async (local_size: number[]): Promise<number> => {
    try {
      return (await _prg.call(test_rawbuffers.map((x) => x._buf), {
        global_size: zip(global_size, local_size).map(([g, l]) => mod(g, l) === 0 ? idiv(g, l) : g / l),
        local_size,
      }, true))!
    } catch {
      return Infinity
    }
  }
  const ret = await Promise.all(local_sizes.map(async (local_size) => [await try_exec(local_size), local_size] as [number, number[]])) // KAREL: randomise local_sizes, and instead of [0] use min()
  if (isInf(ret[0][0])) throw new Error('all optimize_local_size exec failed')
  return ret[0][1]
}
export const time_linearizer = (lin: Kernel, rawbufs: Buffer[], allow_test_size = true, max_global_size = 65536, cnt = 3, disable_cache = false, clear_l2 = false): number => {
  throw new NotImplemented()
}
