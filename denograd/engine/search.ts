import { type Kernel, Opt, OptOps } from '../codegen/kernel.ts'
import { Buffer, type Compiler, Device, type Program } from '../device.ts'
import { env, withEnvAsync } from '../env/index.ts'
import { idiv, isInf, mod, prod, range, zip } from '../helpers.ts'
import { sym_infer, type UOp, type Variable } from '../ops.ts'
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

export const _get_test_global_size = (global_size: number[], max_global_size: number, var_vals: Map<UOp, number>): [number[], number] => {
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

export const _try_compile_linearized_w_idx = async (x: [number, Kernel], compiler: Compiler): Promise<[number, [ProgramSpec, Uint8Array, number | undefined] | undefined]> => {
  // TODO: add timeout
  let ret: [ProgramSpec, Uint8Array, number | undefined] | undefined = undefined
  try {
    const p = x[1].to_program('test')
    if (p.uops === undefined) throw new Error("uop list wasn't generated?")
    if (p.uops.length >= env.get_num('BEAM_UOPS_MAX', 3000) && env.get_num('BEAM_UOPS_MAX', 3000) > 0) throw new Error('too many uops')
    const st = performance.now()
    const prog = await compiler.compile(p.src)
    const et = performance.now() - st
    ret = [p, prog, et]
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
  // bufsts: defaultdict[int, list[UOp]] = defaultdict(list)
  // for x in lin.bufs:
  //   if x.src[0].op is Ops.DEFINE_GLOBAL: bufsts[x.src[0].arg].append(x)
  // rawbufs: list[Optional[Buffer]] = [None]*len(bufsts)
  // for k,lx in bufsts.items():
  //   buf_size = prod(dtype.shape) if isinstance(dtype:=lx[0].src[0].dtype, ImageDType) else max(y.st_arg.real_size() for y in lx)
  //   assert isinstance(dtype, (PtrDType, ImageDType))
  //   if buf_size == 0: buf_size = 1  # create a size 1 buffer if no cell is accessed in kernel. # TODO: remove from kernel input in this case.
  //   buf_dtype = dtype if isinstance(dtype, ImageDType) else dtype.base
  //   rawbufs[k] = Buffer(lin.opts.device, buf_size, buf_dtype).allocate() if allocate else Buffer(lin.opts.device, buf_size, buf_dtype)
  // assert all(r is not None for r in rawbufs)
  // return cast(list[Buffer], rawbufs)
}
// get dictionary of all possible actions
export const get_kernel_actions = (lin: Kernel, include_0 = true): Map<number, Kernel> => {
  // acted_lins, max_up, max_lcl = {0:lin} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256), getenv("BEAM_LOCAL_MAX", 1024)
  // for i,a in enumerate(actions):
  //   if a.axis is not None and a.op is not OptOps.TC:
  //     if ((ax:=a.real_axis(lin)) >= lin.shape_len) or (lin.full_shape[ax] == a.amt and Opt(a.op, ax, 0) in actions): continue
  //   lin2 = lin.copy()
  //   try:
  //     lin2.apply_opt(a)
  //     up, lcl, tc_up = 1, 1, prod(tc.dims)//tc.threads if (tc:=lin2.tensor_core) else 1
  //     for s,c in zip(lin2.full_shape, lin2.colors()):
  //       if c in {"magenta", "yellow"}: up *= s
  //       elif c in {"cyan", "green", "white"}: lcl *= s
  //     if up//tc_up > max_up or lcl > max_lcl: continue
  //     acted_lins[i+1] = lin2
  //   except KernelOptError: pass
  // return acted_lins
}

export const beam_pool = undefined, BEAM_DEBUG = env.get('BEAM_DEBUG')
export const beam_search = (lin: Kernel, rawbufs: Buffer[], amt: number, allow_test_size = true, disable_cache = env.get('IGNORE_BEAM_CACHE')): Kernel => {
  // global beam_pool
  // key = {"ast": lin.ast.key, "amt": amt, "allow_test_size": allow_test_size, "device": lin.opts.device, "suffix": lin.opts.suffix}
  // if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("beam_search", key)) is not None:
  //   ret = lin.copy()
  //   for o in val[len(lin.applied_opts):]: ret.apply_opt(o)
  //   return ret

  // beam: list[tuple[Kernel, float]] = [(lin, float("inf"))]
  // seen_libs = set()

  // default_parallel = multiprocessing.cpu_count() if lin.opts.device in {"CUDA", "AMD", "NV", "METAL"} else 0
  // if beam_pool is None and (workers := getenv("PARALLEL", default_parallel)):
  //   beam_pool = multiprocessing.get_context("spawn").Pool(workers, _init_worker, (), getenv("BEAM_MAX_TASKS_PER_CHILD", 16))

  // min_progress = getenv("BEAM_MIN_PROGRESS", 0.01)/1e6
  // if BEAM_DEBUG: print(f"BEAM_SEARCH:\n{lin.ast}")
  // if DEBUG >= 2: print(f"   0.00s:                 from   1 ->   1 actions {lin.colored_shape()}")

  // try:
  //   rawbufs = _ensure_buffer_alloc(rawbufs)
  //   var_vals: dict[Variable, int] = {k:int(k.vmax+k.vmin)//2 for k in lin.ast.variables()}
  //   exiting, st = False, time.perf_counter()
  //   dev = Device[lin.opts.device]
  //   while not exiting:
  //     acted_lins: list[Kernel] = flatten([get_kernel_actions(lin, include_0=False).values() for lin,_ in beam])
  //     timed_lins: list[tuple[Kernel, float]] = []
  //     _compile_fn = functools.partial(_try_compile_linearized_w_idx, compiler=dev.compiler)
  //     least_compute_ops = math.inf
  //     for i,proc in (map(_compile_fn, enumerate(acted_lins)) if beam_pool is None else beam_pool.imap_unordered(_compile_fn, enumerate(acted_lins))):
  //       if proc is None: continue
  //       p, lib, compile_et = proc
  //       if lib in seen_libs: continue
  //       # filter out kernels that use 1000x more compute than the smallest
  //       least_compute_ops = min(this_compute_ops:=sym_infer(p.estimates.ops, var_vals), least_compute_ops)
  //       if least_compute_ops*1000 < this_compute_ops: continue
  //       seen_libs.add(lib)
  //       try: tms = _time_program(p, lib, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0, clear_l2=hasattr(dev, 'invalidate_caches'))
  //       except RuntimeError: continue # for runtime issues
  //       timed_lins.append((acted_lins[i], min(tms)))
  //       if BEAM_DEBUG > 1: print(f"{time.perf_counter() - st:7.2f}s: {i:5d} {len(cast(list, p.uops)):5d} uops {compile_et*1e6:12.2f} us compile/{timed_lins[-1][1]*1e6:12.2f} us run       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}")  # noqa: E501
  //       elif DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s: {timed_lins[-1][1]*1e6:12.2f} us       {len(timed_lins):4d}/{len(acted_lins):4d}         {timed_lins[-1][0].colored_shape()}\033[K", end="")  # noqa: E501

  //     # done
  //     opts = sorted(timed_lins, key=lambda x: x[1])
  //     exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (len(beam) > 0 and ((beam[0][1]-opts[0][1]) < min_progress))
  //     if not exiting: beam = opts[:amt]
  //     elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]
  //     if DEBUG >= 2: print(f"\r{time.perf_counter() - st:7.2f}s:", colored(f"{beam[0][1]*1e6:12.2f} us", "green" if exiting else None), f"from {len(acted_lins):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())  # noqa: E501
  // except KeyboardInterrupt as e:
  //   if beam_pool is not None: beam_pool.terminate()
  //   raise e

  // if CACHELEVEL >= 1: diskcache_put("beam_search", key, beam[0][0].applied_opts)
  // if BEAM_DEBUG: print(f"BEAM_SEARCH: final tm={beam[0][1]*1e6:0.2f} us, applied_opts={beam[0][0].applied_opts}")
  // return beam[0][0]
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
  const results = await Promise.all(local_sizes.map(async (local_size) => [await try_exec(local_size), local_size] as [number, number[]])) // KAREL: randomise local_sizes,
  const min = results.toSorted(([a], [b]) => b - a)[0]
  if (isInf(min[0])) throw new Error('all optimize_local_size exec failed')
  return min[1]
}
export const time_linearizer = (lin: Kernel, rawbufs: Buffer[], allow_test_size = true, max_global_size = 65536, cnt = 3, disable_cache = false, clear_l2 = false): number => {
  //   key = {"ast": lin.ast.key, "opts": str(lin.applied_opts), "allow_test_size": allow_test_size,
  //     "max_global_size": max_global_size, "clear_l2": clear_l2, "device": lin.opts.device, "suffix": lin.opts.suffix}
  // if not disable_cache and CACHELEVEL >= 2 and (val:=diskcache_get("time_linearizer", key)) is not None: return min(val)

  // dev = Device[lin.opts.device]
  // assert dev.compiler is not None

  // rawbufs = _ensure_buffer_alloc(rawbufs)
  // var_vals: dict[Variable, int] = {k:int(k.vmax+k.vmin)//2 for k in lin.ast.variables()}
  // p = lin.to_program()
  // tms = _time_program(p, dev.compiler.compile(p.src), var_vals, rawbufs,
  //                  max_global_size=max_global_size if allow_test_size else None, clear_l2=clear_l2, cnt=cnt, name=to_function_name(lin.name))

  // if CACHELEVEL >= 2: diskcache_put("time_linearizer", key, tms)
  // return min(tms)
}
