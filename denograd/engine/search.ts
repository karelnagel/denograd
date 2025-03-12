// deno-lint-ignore-file custom-lint-rules/no-floating-promises
import { type Kernel, KernelOptError, Opt, OptOps } from '../codegen/kernel.ts'
import { Buffer, type Compiler, Device, type Program } from '../device.ts'
import { ImageDType, PtrDType } from '../dtype.ts'
import { env, withEnvAsync } from '../env/index.ts'
import { add, colored, type ConstType, DefaultMap, flatten, get_key, idiv, isInf, mod, mul, perf, prod, range, to_function_name, zip } from '../helpers.ts'
import { Ops, type sint, sym_infer, type UOp, type Variable } from '../ops.ts'
import type { ProgramSpec } from '../renderer/index.ts'
import { Tensor } from '../tensor.ts'
import { CompiledRunner } from './realize.ts'

export const actions: Opt[] = [
  ...range(6).flatMap((axis) => [0, 2, 3, 4, 5, 7].map((amt) => new Opt(OptOps.UPCAST, axis, amt))),
  ...range(5).flatMap((axis) => [0, 4, 7].map((amt) => new Opt(OptOps.UNROLL, axis, amt))),
  ...range(6).flatMap((axis) => [2, 3, 4, 8, 13, 16, 29].map((amt) => new Opt(OptOps.LOCAL, axis, amt))),
  ...range(3).flatMap((axis) => [13, 16, 28, 29, 32, 49, 64, 256].map((amt) => new Opt(OptOps.GROUPTOP, axis, amt))),
  ...range(3).flatMap((axis) => [0, 4, 8, 16].map((amt) => new Opt(OptOps.GROUP, axis, amt))),
  ...(env.get_num('BEAM_PADTO', 1) ? range(7).flatMap((axis) => [32].map((amt) => new Opt(OptOps.PADTO, axis, amt))) : []),
  new Opt(OptOps.LOCAL, 0, 32),
  new Opt(OptOps.LOCAL, 6, 2),
  new Opt(OptOps.TC, 0, 0),
  ...range(9).map((axis) => new Opt(OptOps.TC, axis, env.get_num('TC_OPT', 2))), // covers resnet kernels (3 global * 3 reduce)
  ...range(5).flatMap((axis) => range(axis + 1, 5).map((amt) => new Opt(OptOps.SWAP, axis, amt))),
  ...(env.get('NOLOCALS') ? [new Opt(OptOps.NOLOCALS)] : []),
]

export const _get_test_global_size = (global_size: number[], max_global_size: number, var_vals: Map<UOp, ConstType>): [number[], number] => {
  let test_global_size = global_size.map((sz) => sym_infer(sz, var_vals)), factor = 1
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

export const _time_program = async (p: ProgramSpec, lib: Uint8Array, var_vals: Map<Variable, ConstType>, rawbufs: Buffer[], early_stop?: number, max_global_size = 65536, clear_l2 = false, cnt = 3, name = 'test'): Promise<number[]> => {
  let factor = 1, global_size: number[], car: CompiledRunner
  if (p.global_size !== undefined && max_global_size !== undefined) {
    ;[global_size, factor] = _get_test_global_size(p.global_size, max_global_size, var_vals)
    p.global_size = global_size
  }
  try {
    car = await CompiledRunner.init(p, lib)
  } catch {
    return range(cnt).map((x) => Infinity)
  }
  const tms: number[] = []
  const input_bufs = car.p.globals.map((i) => rawbufs[i])
  for (const _ of range(cnt)) {
    if (clear_l2) {
      const dev = Device.get(p.device)
      if ('invalidate_caches' in dev) (dev.invalidate_caches as any)()
      else {
        await withEnvAsync({ DEBUG: 0, BEAM: 0, CAPTURING: 0, TRACK_MATCH_STATS: 0 }, async () => {
          await Tensor.ones([1024, 1024]).contiguous().realize(undefined, false)
        })
      }
    }
    tms.push((await car.call(input_bufs, var_vals as Map<UOp, number>, true))! * factor)
    if (early_stop !== undefined && early_stop < Math.min(...tms)) break
  }
  return tms
}

class TimeoutException extends Error {}
export const timeout_handler = (signum: any, frame: any) => {
  throw new TimeoutException()
}

const timeout = async (seconds: number) => await new Promise((_, reject) => setTimeout(() => reject(new Error(`Timed out after ${seconds}s`)), seconds * 1000))

export const _try_compile_linearized_w_idx = async (x: [number, Kernel], compiler: Compiler): Promise<[number, [ProgramSpec, Uint8Array, number | undefined] | undefined]> => {
  let ret: [ProgramSpec, Uint8Array, number | undefined] | undefined = undefined
  try {
    const compile = async () => {
      const p = x[1].to_program('test')
      if (p.uops === undefined) throw new Error("uop list wasn't generated?")
      if (p.uops.length >= env.get_num('BEAM_UOPS_MAX', 3000) && env.get_num('BEAM_UOPS_MAX', 3000) > 0) throw new Error('too many uops')
      const st = performance.now()
      const prog = await compiler.compile(p.src)
      ret = [p, prog, perf(st)]
    }
    await Promise.race([compile(), timeout(env.get_num('BEAM_TIMEOUT_SEC', 10))])
  } catch (e) {
    if (env.DEBUG >= 4) console.trace()
    if (env.get_num('BEAM_STRICT_MODE')) throw e
  }
  return [x[0], ret]
}

export const _ensure_buffer_alloc = (bufs: Buffer[]): Buffer[] => bufs.map((buf) => buf.ensure_allocated())

// *** external API ***

// get (scrap) buffers for timing the linearizer
export const bufs_from_lin = (lin: Kernel, allocate = true): Buffer[] => {
  const bufsts = new DefaultMap<number, UOp[]>(undefined, () => [])
  for (const x of lin.bufs) {
    if (x.src[0].op === Ops.DEFINE_GLOBAL) bufsts.get(x.src[0].arg).push(x)
  }
  const rawbufs: (Buffer | undefined)[] = range(bufsts.size).map(() => undefined)
  for (const [k, lx] of bufsts.entries()) {
    let dtype = lx[0].src[0].dtype, buf_size = dtype instanceof ImageDType ? prod(dtype.shape) : Math.max(...lx.map((y) => y.st_arg.real_size()))
    if (!(dtype instanceof PtrDType || dtype instanceof ImageDType)) throw new Error()
    if (buf_size === 0) buf_size = 1 // create a size 1 buffer if no cell is accessed in kernel. # TODO: remove from kernel input in this case.
    const buf_dtype = dtype instanceof ImageDType ? dtype : dtype.base
    rawbufs[k] = allocate ? new Buffer(lin.opts.device, buf_size, buf_dtype).allocate() : new Buffer(lin.opts.device, buf_size, buf_dtype)
  }
  if (rawbufs.some((r) => r === undefined)) throw new Error()
  return rawbufs as Buffer[]
}
// get dictionary of all possible actions
export const get_kernel_actions = (lin: Kernel, include_0 = true): Map<number, Kernel> => {
  let acted_lins = new Map<number, Kernel>(include_0 ? [[0, lin]] : []), max_up = env.get_num('BEAM_UPCAST_MAX', 256), max_lcl = env.get_num('BEAM_LOCAL_MAX', 1024)
  for (const [i, a] of actions.entries()) {
    if (a.axis !== undefined && a.op !== OptOps.TC) {
      const ax = a.real_axis(lin)
      if ((ax >= lin.shape_len) || (lin.full_shape[ax] === a.amt && actions.includes(new Opt(a.op, ax, 0)))) continue
    }
    const lin2 = lin.copy()
    try {
      lin2.apply_opt(a)
      let up: sint = 1, lcl: sint = 1, tc = lin2.tensor_core, tc_up = tc ? idiv(prod(tc.dims), tc.threads) : 1
      for (const [s, c] of zip(lin2.full_shape, lin2.colors())) {
        if (['magenta', 'yellow'].includes(c)) up = mul(up, s)
        else if (['cyan', 'green', 'white'].includes(c)) lcl = mul(lcl, s)
      }
      if (idiv(up as number, tc_up) > max_up || lcl as number > max_lcl) continue
      acted_lins.set(i + 1, lin2)
    } catch (e) {
      if (!(e instanceof KernelOptError)) throw e
    }
  }
  return acted_lins
}

export const BEAM_DEBUG = env.get_num('BEAM_DEBUG')
export const beam_search = async (lin: Kernel, rawbufs: Buffer[], amt: number, allow_test_size = true, disable_cache = env.get('IGNORE_BEAM_CACHE')): Promise<Kernel> => {
  const key = JSON.stringify({ 'ast': lin.ast.key, 'amt': amt, 'allow_test_size': allow_test_size, 'device': lin.opts.device, 'suffix': lin.opts.suffix })
  if (!disable_cache && env.CACHELEVEL >= 1) {
    const val = await env.disk_get('beam_search', key)
    if (val !== undefined) {
      const ret = lin.copy()
      const opts = JSON.parse(val).map((o: any) => new Opt(OptOps.values().find((x) => x.name === o.op.name)!, o.axis, o.amt))
      if (BEAM_DEBUG) console.log(`BEAM_CACHE: opts=${opts}`)
      for (const o of opts.slice(lin.applied_opts.length)) ret.apply_opt(o)
      return ret
    }
  }

  let beam: [Kernel, number][] = [[lin, Infinity]]
  const seen_libs = new Set<string>()

  const min_progress = env.get_num('BEAM_MIN_PROGRESS', 0.01) / 1e6
  if (BEAM_DEBUG >= 2) console.log(`BEAM_SEARCH:\n${lin.ast}`)
  if (env.DEBUG >= 2) console.log(`   0.00s:                 from   1 ->   1 actions ${lin.colored_shape()}`)

  try {
    rawbufs = _ensure_buffer_alloc(rawbufs)
    const var_vals = new Map<Variable, ConstType>(lin.ast.variables().map((k) => [k, idiv(add(k.vmax, k.vmin), 2)]))
    let exiting = false, st = performance.now()
    const dev = Device.get(lin.opts.device)
    while (!exiting) {
      const acted_lins = flatten(beam.map(([lin, _]) => [...get_kernel_actions(lin, false).values()]))
      const timed_lins: [Kernel, number][] = []
      const _compile_fn = async (x: [number, Kernel]) => await _try_compile_linearized_w_idx(x, dev.compiler)
      let least_compute_ops = Infinity
      for (const x of acted_lins.entries()) {
        const [i, proc] = await _compile_fn(x)
        if (proc === undefined) continue
        const [p, lib, compile_et] = proc
        if (seen_libs.has(get_key(lib))) continue
        // filter out kernels that use 1000x more compute than the smallest
        const this_compute_ops = sym_infer(p.estimates.ops, var_vals)
        least_compute_ops = Math.min(this_compute_ops, least_compute_ops)
        if (least_compute_ops * 1000 < this_compute_ops) continue
        seen_libs.add(get_key(lib))
        let tms: number[]
        try {
          tms = await _time_program(p, lib, var_vals, rawbufs, beam.length ? beam[0][1] * 3 : 1.0, undefined, 'invalidate_caches' in dev)
        } catch {
          continue
        }
        timed_lins.push([acted_lins[i], Math.min(...tms)])
        if (BEAM_DEBUG > 1) console.log(`${(perf(st)).toFixed(2).padEnd(7)}s: ${i.toString().padEnd(5)} ${p.uops!.length.toString().padEnd(5)} uops ${(compile_et! * 1e6).toFixed(2).padEnd(12)} us compile/${(timed_lins.at(-1)![1] * 1e6).toFixed(2).padEnd(12)} us run       ${timed_lins.length.toString().padEnd(4)}/${acted_lins.length.toString().padEnd(4)}         ${timed_lins.at(-1)![0].colored_shape()}`)
        else if (env.DEBUG >= 2) env.writeStdout(`\r${(perf(st)).toFixed(2).padStart(7)}s: ${(timed_lins.at(-1)![1] * 1e6).toFixed(2).padStart(12)} us       ${timed_lins.length.toString().padEnd(4)}/${acted_lins.length.toString().padEnd(4)}         ${timed_lins.at(-1)![0].colored_shape()}\x1b[K`)
      }
      // done
      const opts = timed_lins.toSorted((a, b) => a[1] - b[1])
      exiting = opts.length === 0 || (opts[0][1] < min_progress) || (beam.length > 0 && ((beam[0][1] - opts[0][1]) < min_progress))
      if (!exiting) beam = opts.slice(0, amt)
      else if (opts.length > 0 && opts[0][1] < beam[0][1]) beam = opts.slice(0, 1)
      if (env.DEBUG >= 2) console.log(`\r${(perf(st)).toFixed(2).padStart(7)}s:`, colored(`${(beam[0][1] * 1e6).toFixed(2).padStart(12)} us`, exiting ? 'green' : undefined), `from ${acted_lins.length.toString().padStart(3)} -> ${opts.length.toString().padStart(3)} actions\x1b[K`, beam[0][0].colored_shape())
    }
  } catch (e) {
    //   if beam_pool is not None: beam_pool.terminate()
    throw e
  }

  if (env.CACHELEVEL >= 1) env.disk_put('beam_search', key, JSON.stringify(beam[0][0].applied_opts))
  if (BEAM_DEBUG) console.log(`BEAM_SEARCH: final tm=${(beam[0][1] * 1e6).toFixed(2)} us, applied_opts=${beam[0][0].applied_opts}`)
  return beam[0][0]
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
  const results: [number, number[]][] = []
  for (const local_size of local_sizes) { // TODO: randomize local sizes
    results.push([await try_exec(local_size), local_size])
  }
  const min = results.toSorted(([a], [b]) => a - b)[0]
  if (isInf(min[0])) throw new Error('all optimize_local_size exec failed')
  return min[1]
}
export const time_linearizer = async (lin: Kernel, rawbufs: Buffer[], allow_test_size = true, max_global_size = 65536, cnt = 3, disable_cache = false, clear_l2 = false): Promise<number> => {
  const key = JSON.stringify({ 'ast': lin.ast.key, 'opts': String(lin.applied_opts), 'allow_test_size': allow_test_size, 'max_global_size': max_global_size, 'clear_l2': clear_l2, 'device': lin.opts.device, 'suffix': lin.opts.suffix })
  if (!disable_cache && env.CACHELEVEL >= 2) {
    const val = await env.disk_get('time_linearizer', key)
    if (val !== undefined) return Math.min(...JSON.parse(val))
  }
  const dev = Device.get(lin.opts.device)
  if (dev.compiler === undefined) throw new Error()

  rawbufs = _ensure_buffer_alloc(rawbufs)
  const var_vals = new Map<Variable, ConstType>(lin.ast.variables().map((k) => [k, idiv(add(k.vmax, k.vmin), 2)]))
  const p = lin.to_program()
  const tms = await _time_program(p, await dev.compiler.compile(p.src), var_vals, rawbufs, undefined, allow_test_size ? max_global_size : undefined, clear_l2, cnt, to_function_name(lin.name))

  if (env.CACHELEVEL >= 2) env.disk_put('time_linearizer', key, JSON.stringify(tms))
  return Math.min(...tms)
}
