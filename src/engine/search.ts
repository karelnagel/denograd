import { Kernel, Opt, OptOps } from '../codegen/kernel.ts'
import { Buffer, Compiler, Device } from '../device.ts'
import { ImageDType, PtrDType } from '../dtype.ts'
import { assert, bytes, CACHELEVEL, DEBUG, diskcache_get, diskcache_put, getEnv, getNumberEnv, isinstance, min, prod, range, to_function_name, zip } from '../helpers.ts'
import { idiv, mul, Ops, sym_infer, UOp, Variable } from '../ops.ts'
import { ProgramSpec } from '../renderer/index.ts'
import { Tensor } from '../tensor.ts'
import { CompiledRunner } from './realize.ts'
import os from 'node:os'

const actions: Opt[] = [
  ...range(6).flatMap((axis) => [0, 2, 3, 4, 5, 7].map((amt) => new Opt(OptOps.UPCAST, axis, amt))),
  ...range(5).flatMap((axis) => [0, 4, 7].map((amt) => new Opt(OptOps.UNROLL, axis, amt))),
  ...range(6).flatMap((axis) => [2, 3, 4, 8, 13, 16, 29].map((amt) => new Opt(OptOps.LOCAL, axis, amt))),
  ...range(3).flatMap((axis) => [13, 16, 28, 29, 32, 49, 64, 256].map((amt) => new Opt(OptOps.GROUPTOP, axis, amt))),
  ...range(3).flatMap((axis) => [0, 4, 8, 16].map((amt) => new Opt(OptOps.GROUP, axis, amt))),
  ...(getNumberEnv('BEAM_PADTO', 1) ? range(7).flatMap((axis) => [32].map((amt) => new Opt(OptOps.PADTO, axis, amt))) : []),
  new Opt(OptOps.LOCAL, 0, 32),
  new Opt(OptOps.LOCAL, 6, 2),
  new Opt(OptOps.UPCASTMID, 1, 4),
  new Opt(OptOps.TC, 0, 0),
  ...range(9).map((axis) => new Opt(OptOps.TC, axis, getNumberEnv('TC_OPT', 2))), // covers resnet kernels (3 global * 3 reduce)
  ...range(5).flatMap((axis) => range(axis + 1, 5).map((amt) => new Opt(OptOps.SWAP, axis, amt))),
  ...(getEnv('NOLOCALS') ? [new Opt(OptOps.NOLOCALS)] : []),
]

export const _get_test_global_size = (global_size: number[], max_global_size: number, var_vals: Map<UOp, number>): [number[], number] => {
  let [test_global_size, factor] = [global_size.map((sz) => sym_infer(sz, var_vals)), 1]
  while (prod(test_global_size) > max_global_size) {
    for (const j of range(global_size.length - 1, -1, -1)) {
      if (test_global_size[j] > 16) {
        test_global_size[j] //= 2
        factor *= 2
        break
      }
    }
  }
  return [test_global_size, factor]
}
export const _time_program = (p: ProgramSpec, lib: bytes, var_vals: Map<Variable, number>, rawbufs: Buffer[], early_stop?: number, max_global_size = 65536, clear_l2 = false, cnt = 3, name = 'test'): number[] => {
  let factor = 1, global_size, car
  if (p.global_size !== undefined && max_global_size !== undefined) {
    ;[global_size, factor] = _get_test_global_size(p.global_size, max_global_size, var_vals)
    p.global_size = global_size
  }
  try {
    car = new CompiledRunner(p, lib)
  } catch {
    return range(cnt).map((x) => Infinity)
  }
  const tms = []
  const input_bufs = car.p.globals.map((i) => rawbufs[i])
  for (const _ of range(cnt)) {
    if (clear_l2) {
      const dev = Device.get(p.device)
      if ('invalidate_caches' in dev) (dev.invalidate_caches as any)()
      else Tensor.ones(1024, 1024).contiguous().realize(false)
      //TODO:  with Context(DEBUG=0, BEAM=0, CAPTURING=0, TRACK_MATCH_STATS=0): Tensor.ones(1024,1024).contiguous().realize(do_update_stats=false)
    }
    tms.push(car.call(input_bufs, var_vals, true)! * factor)
    if (early_stop !== undefined && early_stop < Math.min(...tms)) break
  }
  return tms
}
class TimeoutException extends Error {}
export const timeout_handler = (signum: any, frame: any) => {
  throw new Error('Timout handler')
}

export const _try_compile_linearized_w_idx = (x: [number, Kernel], compiler: Compiler): [number, [ProgramSpec, bytes, number | undefined] | undefined] => {
  // TODO:
  //   if hasattr(signal, "alarm"):
  //     signal.signal(getattr(signal, 'SIGALRM'), timeout_handler)
  //     // set timeout
  //     signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))
  let ret: [ProgramSpec, bytes, number | undefined] | undefined = undefined
  try {
    const p = x[1].to_program('test')
    assert(p.uops !== undefined, "uop list wasn't generated?")
    if (p.uops!.length >= getNumberEnv('BEAM_UOPS_MAX', 3000) && getNumberEnv('BEAM_UOPS_MAX', 3000) > 0) throw new Error('too many uops')
    const st = performance.now()
    const prog = compiler.compile(p.src)
    const et = performance.now() - st
    ret = [p, prog, et]
  } //   except RuntimeError:
  //     if DEBUG >= 4: traceback.print_exc()
  catch (e) {
    if (getEnv('BEAM_STRICT_MODE')) throw e
  } finally {
    //     if hasattr(signal, "alarm"): signal.alarm(0)
  }
  return [x[0], ret]
}
// // workers should ignore ctrl c
// const _init_worker = () => {
//   return signal.signal(signal.SIGINT, signal.SIG_IGN)
// }

const _ensure_buffer_alloc = (bufs: Buffer[]): Buffer[] => bufs.map((buf) => buf.ensure_allocated())

// // *** external API ***

// // get (scrap) buffers for timing the linearizer
export const bufs_from_lin = (lin: Kernel, allocate = true): Buffer[] => {
  const bufsts = new Map<number, UOp[]>()
  for (const x of lin.bufs) if (x.src[0].op === Ops.DEFINE_GLOBAL) bufsts.get(x.src[0].arg)!.push(x)
  const rawbufs: (Buffer | undefined)[] = range(bufsts.size).map((x) => undefined)
  for (const [k, lx] of bufsts.entries()) {
    const dtype = lx[0].src[0].dtype
    let buf_size = isinstance(dtype, ImageDType) ? prod(dtype.shape) : Math.max(...lx.map((y) => y.st_arg.real_size()))
    assert(isinstance(dtype, PtrDType) || isinstance(dtype, ImageDType))
    if (buf_size === 0) buf_size = 1 // create a size 1 buffer if no cell === accessed in kernel. // TODO: remove from kernel input in this case.
    const buf_dtype = isinstance(dtype, ImageDType) ? dtype : dtype.base
    rawbufs[k] = allocate ? new Buffer({ device: lin.opts.device, size: buf_size, dtype: buf_dtype }).allocate() : new Buffer({ device: lin.opts.device, size: buf_size, dtype: buf_dtype })
  }
  assert(rawbufs.every((r) => r !== undefined))
  return rawbufs.map((b) => b!)
}
// // get dictionary of all possible actions
export const get_kernel_actions = (lin: Kernel, include_0 = true): Map<number, Kernel> => {
  const [acted_lins, max_up, max_lcl] = [include_0 ? new Map([[0, lin]]) : new Map<number, Kernel>(), getNumberEnv('BEAM_UPCAST_MAX', 256), getNumberEnv('BEAM_LOCAL_MAX', 1024)]
  for (const [i, a] of actions.entries()) {
    if (a.axis !== undefined && a.op !== OptOps.TC) {
      const ax = a.real_axis(lin)
      if ((ax >= lin.shape_len) || (lin.full_shape[ax] === a.amt && actions.includes(new Opt(a.op, ax, 0)))) continue
    }
    const lin2 = lin //.copy()
    try {
      lin2.apply_opt(a)
      let [up, lcl, tc_up] = [1, 1, lin2.tensor_core ? idiv(prod(lin2.tensor_core.dims), prod(lin2.tensor_core.threads.map((x) => x[1]))) : 1]
      for (const [s, c] of zip(lin2.full_shape as number[], lin2.colors())) {
        if (['magenta', 'yellow'].includes(c)) up = mul(up, s)
        else if (['cyan', 'green', 'white'].includes(c)) lcl = mul(lcl, s)
      }
      if (idiv(up, tc_up) > max_up || lcl > max_lcl) continue
      acted_lins.set(i + 1, lin2)
    } catch {
      // pass
    }
  }
  return acted_lins
}
let [beam_pool, BEAM_DEBUG, CAPTURE_BEAM] = [undefined, getEnv('BEAM_DEBUG'), getEnv('CAPTURE_BEAM', '')]
export const beam_search = (lin: Kernel, rawbufs: Buffer[], amt: number, allow_test_size = true, disable_cache = getEnv('IGNORE_BEAM_CACHE')): Kernel => {
  const key = { 'ast': lin.ast.key, 'amt': amt, 'allow_test_size': allow_test_size, 'device': lin.opts.device, 'suffix': lin.opts.suffix }
  const val = diskcache_get('beam_search', key)
  if (!disable_cache && CACHELEVEL >= 1 && val !== undefined) {
    const ret = lin //.copy()
    for (const o of val.slice(lin.applied_opts.length)) ret.apply_opt(o)
    return ret
  }
  let beam = [[lin, Infinity] as [Kernel, number]]
  const seen_libs = new Set()

  const default_parallel = ['CUDA', 'AMD', 'NV', 'METAL'].includes(lin.opts.device) ? os.cpus().length : 0
  const workers = getNumberEnv('PARALLEL', default_parallel)
  // TODO:
  // if (beam_pool === undefined && workers) beam_pool = multiprocessing.get_context('spawn').Pool(workers, _init_worker, [], getNumberEnv('BEAM_MAX_TASKS_PER_CHILD', 16))

  const min_progress = getNumberEnv('BEAM_MIN_PROGRESS', 0.01) / 1e6
  if (BEAM_DEBUG) console.log(`BEAM_SEARCH:\n${lin.ast}`)
  if (DEBUG >= 2) console.log(`   0.00s:                 from   1 ->   1 actions ${lin.colored_shape()}`)

  try {
    rawbufs = _ensure_buffer_alloc(rawbufs)
    const var_vals = new Map<Variable, number>(lin.ast.variables().map((k) => [k, idiv(k.vmax + k.vmin, 2)]))
    const [exiting, st] = [false, performance.now()]
    const dev = Device.get(lin.opts.device)
    while (!exiting) {
      const acted_lins: Kernel[] = beam.flatMap(([lin, _]) => [...get_kernel_actions(lin, false).values()])
      const timed_lins: [Kernel, number][] = []
      const _compile_fn = (i: [number, Kernel]) => _try_compile_linearized_w_idx(i, dev.compiler)
      let least_compute_ops = Infinity
      for (const [i, proc] of (beam_pool === undefined ? acted_lins.map((k, i) => _compile_fn([i, k])) : acted_lins.map((k, i) => _compile_fn([i, k])))) { //Todo
        if (proc === undefined) continue
        let [p, lib, compile_et] = proc
        if (seen_libs.has(lib)) continue
        //         // filter out kernels that use 1000x more compute than the smallest
        const this_compute_ops = sym_infer(p.op_estimate, var_vals)
        least_compute_ops = Math.min(this_compute_ops, least_compute_ops)
        if (least_compute_ops * 1000 < this_compute_ops) continue
        if (CAPTURE_BEAM.length > 0) {
          //           with open(CAPTURE_BEAM, 'a') as f: f.write(string(acted_lins[i].ast).replace('\n','')+` :: ${acted_lins[i].applied_opts}\n`)
          throw new Error('Unimplemented')
        }
        seen_libs.add(lib)
        let tms
        try {
          tms = _time_program(p, lib, var_vals, rawbufs, beam.length ? beam[0][1] * 3 : 1.0, undefined, 'invalidate_caches' in dev)
        } catch (e) {
          if (CAPTURE_BEAM.length > 0) {
            //             with open(CAPTURE_BEAM, 'a') as f: f.write("// Upper ast finished with an error:" + string(e).replace('\n',' ')+ "\n")
            throw new Error('not implemented')
          }
          continue // for runtime issues
        }
        timed_lins.push([acted_lins[i], Math.min(...tms)])
        //         if BEAM_DEBUG > 1: console.log(`${time.perf_counter() - st:7.2f}s: ${i:5d} ${len(cast(List, p.uops)):5d} uops ${compile_et*1e6:12.2f} us compile/${timed_lins.at(-1)![1]*1e6:12.2f} us run       ${len(timed_lins):4d}/${len(acted_lins):4d}         ${timed_lins.at(-1)![0].colored_shape()}`)  // noqa: E501
        //         elif DEBUG >= 2: console.log(`\r${time.perf_counter() - st:7.2f}s: ${timed_lins.at(-1)![1]*1e6:12.2f} us       ${len(timed_lins):4d}/${len(acted_lins):4d}         ${timed_lins.at(-1)![0].colored_shape()}\033[K`, end="")  // noqa: E501
      }
      //       // done
      const opts = timed_lins.toSorted((a, b) => a[1] - b[1])
      const exiting = opts.length === 0 || (opts[0][1] < min_progress) || (beam.length > 0 && ((beam[0][1] - opts[0][1]) < min_progress))
      if (!exiting) beam = opts.slice(0, amt)
      else if (opts.length > 0 && opts[0][1] < beam[0][1]) beam = opts.slice(0, 1)
      //       if DEBUG >= 2: console.log(`\r${time.perf_counter() - st:7.2f}s:`, colored(`${beam[0][1]*1e6:12.2f} us`, "green" if exiting else undefined), `from ${len(acted_lins):3d} -> ${len(opts):3d} actions\033[K`, beam[0][0].colored_shape())  // noqa: E501
    }
  } catch (e) {
    if (beam_pool !== undefined) beam_pool.terminate()
    throw e
  }
  if (CACHELEVEL >= 1) diskcache_put('beam_search', key, beam[0][0].applied_opts)
  //   if BEAM_DEBUG: console.log(`BEAM_SEARCH: final tm=${beam[0][1]*1e6:0.2f} us, applied_opts=${beam[0][0].applied_opts}`)
  return beam[0][0]
}
export const optimize_local_size = (_prg: () => number, global_size: number[], rawbufs: Buffer[]): number[] => {
  const test_rawbuffers = rawbufs.slice(1).includes(rawbufs[0]) ? [new Buffer({ device: rawbufs[0].device, size: rawbufs[0].size, dtype: rawbufs[0].dtype }).allocate(), ...rawbufs.slice(1)] : rawbufs
  const MAX_WORKGROUP = 1024
  const local_dims = global_size.map((sz) => [...new Set([sz, 1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_WORKGROUP])].filter((x) => x <= sz))
  const local_sizes = [...local_dims.reduce((acc, curr) => acc.flatMap((x) => curr.map((y) => [...x, y])), [[]] as number[][])].filter((x) => prod(x) <= MAX_WORKGROUP).flatMap((x) => [x, x]) // try each valid size twice
  const try_exec = (local_size: number[]): number[] => {
    try {
      return _prg({
        idk: test_rawbuffers.map((x) => x._buf),
        global_size: zip(global_size, local_size).map(([g, l]) => g % l === 0 ? idiv(g, l) : g / l),
        local_size,
        wait: true,
      })
    } catch {
      return Infinity
    }
  }
  const ret = local_sizes.flatMap((local_size) => [try_exec(local_size), local_size]) // TODO: randomise local_sizes
  assert(isFinite(ret[0]), 'all optimize_local_size exec failed')
  return ret[1]
}
export const time_linearizer = (lin: Kernel, rawbufs: Buffer[], allow_test_size = true, max_global_size = 65536, cnt = 3, disable_cache = false, clear_l2 = false): number => { // noqa: E501
  const key = { 'ast': lin.ast.key, 'opts': string(lin.applied_opts), 'allow_test_size': allow_test_size, 'max_global_size': max_global_size, 'clear_l2': clear_l2, 'device': lin.opts.device, 'suffix': lin.opts.suffix }
  if (!disable_cache && CACHELEVEL >= 2) {
    const val = diskcache_get('time_linearizer', key)
    if (val !== undefined) return min(val)
  }
  const dev = Device.get(lin.opts.device)
  assert(dev.compiler !== undefined)

  rawbufs = _ensure_buffer_alloc(rawbufs)
  const var_vals = new Map<Variable, number>(lin.ast.variables().map((k) => [k, idiv(k.vmax + k.vmin, 2)]))
  const p = lin.to_program()
  const tms = _time_program(p, dev.compiler.compile(p.src), var_vals, rawbufs, undefined, allow_test_size ? max_global_size : undefined, clear_l2, cnt, to_function_name(lin.name))

  if (CACHELEVEL >= 2) diskcache_put('time_linearizer', key, tms)
  return Math.min(...tms)
}
