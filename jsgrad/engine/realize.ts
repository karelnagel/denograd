import { Kernel } from '../codegen/kernel.ts'
import { type Buffer, Device, type Program } from '../device.ts'
import { all_int, all_same, ansilen, colored, get_key, GlobalCounters, idiv, list_str, type Metadata, mod, perf, replace, to_function_name, vars, zip } from '../helpers.ts'
import { Ops, PatternMatcher, sym_infer, type UOp, UPat, type Variable } from '../ops.ts'
import { Estimates, type ProgramSpec, type Renderer } from '../renderer/index.ts'
import type { TinyJit } from './jit.ts'
import type { ScheduleItem } from './schedule.ts'
import { beam_search, bufs_from_lin, optimize_local_size } from './search.ts'

// **************** Program Creation ****************

// const [logkerns, logkerns_level] = [getEnv('LOGKERNS', '') ? open(getEnv('LOGKERNS', ''), 'a') : undefined, getNumberEnv('LOGKERNS_LEVEL', 1)]
export const get_kernel = async (renderer: Renderer, ast: UOp): Promise<Kernel> => {
  if (vars.DEBUG >= 5) console.log(ast)
  let k = new Kernel(ast, renderer).required_optimizations()
  if (!vars.NOOPT) {
    if (!k.apply_tensor_cores(vars.get_num('TC', 1))) k.hand_coded_optimizations()
    if (vars.BEAM >= 1) {
      const kb = new Kernel(ast, renderer).required_optimizations()
      const rawbufs = bufs_from_lin(kb, false)
      k = await beam_search(kb, rawbufs, vars.BEAM, Boolean(vars.get_num('BEAM_ESTIMATE', 1)))
    }
  }
  // if (logkerns !== undefined) logkerns.writelines([`${(k.ast, k.applied_opts)}\n`])
  if (vars.DEBUG >= 5) console.log((k.ast, k.applied_opts)) // print here to show final applied_opts for all kernels instead of just in beam_search
  return k
}
// **************** Runners ****************

export class Runner {
  first_run = true
  constructor(public display_name: string, public device: string, public estimates = new Estimates()) {}

  get dev() {
    return Device.get(this.device)
  }
  exec = async (rawbufs: Buffer[], var_vals?: Map<Variable, number>): Promise<number | undefined> => await this.call(rawbufs, var_vals === undefined ? new Map() : var_vals)
  call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false): Promise<number | undefined> | number | undefined => {
    throw new Error('override this')
  }
}

export class CompiledRunner extends Runner {
  _prg!: Program
  p!: ProgramSpec
  lib?: Uint8Array
  static init = async (p: ProgramSpec, lib?: Uint8Array) => {
    const res = new CompiledRunner(p.name, p.device, p.estimates)
    res.p = p
    res.lib = lib
    const dev = Device.get(p.device)
    if (vars.DEBUG >= 4) console.log(p.src)
    if (!res.lib) res.lib = await dev.compiler.compile_cached(p.src)
    if (vars.DEBUG >= 6) dev.compiler.disassemble(res.lib)
    const Runtime = Device.get(p.device).runtime!
    // KAREL: TODO: should be p.function_name
    res._prg = await Runtime.init(to_function_name(p.name), res.lib)
    return res
  }
  __reduce__ = () => [this.p, this.lib]

  override call = async (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false): Promise<number | undefined> => {
    let [global_size, local_size] = this.p.launch_dims(var_vals)
    if (global_size !== undefined && local_size === undefined && all_int(this.p.global_size!)) {
      local_size = await optimize_local_size(this._prg, global_size, rawbufs)
      global_size = zip(global_size, local_size!).map(([g, l]) => mod(g, l) === 0 ? idiv(g, l) : g / l)
      this.p.global_size = global_size
      this.p.global_size = local_size
    }
    const lra: Record<string, any> = {}
    if (global_size?.length) {
      lra['global_size'] = global_size
      if (global_size.length !== 3) throw new Error('global size must have len 3')
    }
    if (local_size?.length) {
      lra['local_size'] = local_size
      if (local_size.length !== 3) throw new Error('local size must have len 3')
    }
    return await this._prg.call(rawbufs.map((x) => x._buf), { ...lra, vals: this.p.vars?.map((k) => var_vals.get(k)!) }, wait)
  }
}

export class ViewOp extends Runner {
  constructor(buf: Buffer) {
    super(colored(`view ${buf.nbytes.toString().padStart(8)} @ ${buf.offset.toString().padEnd(10)}`, 'yellow'), buf.device)
  }
  override call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false) => {
    if (rawbufs[0]._base === undefined || rawbufs[0]._base !== rawbufs[1].base) throw new Error(`must be base ${rawbufs}`)
    return 0
  }
}
export class BufferCopy extends Runner {
  constructor(total_sz: number, dest_device: string, src_device: string) {
    let name
    if (total_sz >= 1e6) name = `copy ${(total_sz / 1e6).toFixed(2)}M, ${dest_device.slice(0, 7).padStart(7)} <- ${src_device.slice(0, 7).padEnd(7)}`
    else name = `copy ${total_sz.toFixed(0).padStart(8)}, ${dest_device.slice(0, 7).padStart(7)} <- ${src_device.slice(0, 7).padEnd(7)}`
    super(colored(name, 'yellow'), dest_device, new Estimates(undefined, total_sz, total_sz))
  }
  copy = async (dest: Buffer, src: Buffer) => {
    const disk_supports_fast_copyout = src.device.startsWith('DISK') && 'io_uring' in (src.allocator as any).dev && (src.allocator as any).dev.fd !== undefined
    if (src.device.startsWith('DISK') && 'copy_from_disk' in dest.allocator! && disk_supports_fast_copyout && src.nbytes >= 4096) {
      throw new Error('KAREL: implement copy_from_disk')
      // dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    } else if (src.device.startsWith('DISK') && dest.allocator?._as_buffer) {
      // fast(ish) path, uses readinto in diskbuffers
      await src.allocator!._copyout(dest.allocator._as_buffer(dest._buf!), src._buf!)
    } else {
      dest.copyin(await src.as_buffer(true)) // may allocate a CPU buffer depending on allow_zero_copy
    }
  }
  override call = async (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false) => {
    const [dest, src] = rawbufs.slice(0, 2)
    if (dest.size !== src.size || dest.dtype !== src.dtype) throw new Error(`buffer copy mismatch, ${dest.size} !== ${src.size}, ${dest.dtype} !== ${src.dtype}`)
    const st = performance.now()
    await this.copy(dest, src)
    if (wait) {
      Device.get(dest.device).synchronize()
      return perf(st)
    }
    return 0
  }
}
export class BufferXfer extends BufferCopy {
  override copy = async (dest: Buffer, src: Buffer) => {
    if (!dest.allocator?._transfer) throw new Error()
    return dest.allocator!._transfer!(dest._buf, src._buf, dest.nbytes, (src.allocator as any).dev, (dest.allocator as any).dev)
  }
}
// **************** method cache ****************

const method_cache: Record<string, CompiledRunner> = {}
export const get_runner = async (device: string, ast: UOp): Promise<CompiledRunner> => {
  await Device.get(device).init()
  const ckey = get_key(device, ast.key, vars.BEAM, vars.NOOPT, false)
  const cret = method_cache[ckey]
  if (cret) return cret
  const bkey = get_key(device.split(':')[0], ast.key, vars.BEAM, vars.NOOPT, true)
  let ret
  const bret = method_cache[bkey]
  if (bret) {
    ret = await CompiledRunner.init(replace(bret.p, { device: device }), bret.lib)
    method_cache[ckey] = ret
  } else {
    const prg: ProgramSpec = (await get_kernel(Device.get(device).renderer!, ast)).to_program()
    ret = await CompiledRunner.init(replace(prg, { device: device }))
    method_cache[ckey] = ret
    method_cache[bkey] = ret
  }
  return ret
}

// // **************** lowering functions ****************

export class ExecItem {
  constructor(public prg: Runner, public bufs: (Buffer | undefined)[], public metadata?: Metadata[]) {}
  run = async (_var_vals?: Map<Variable, number>, wait = false, jit = false, do_update_stats = true): Promise<number | undefined> => {
    await Device.get(this.prg.device).init()
    const var_vals = _var_vals === undefined ? new Map<UOp, number>() : _var_vals
    const bufs = jit ? this.bufs.map((x) => x!) : this.bufs.map((x) => x!.ensure_allocated())
    const et = await this.prg.call(bufs, var_vals, wait || vars.DEBUG >= 2)
    if (do_update_stats) {
      GlobalCounters.kernel_count += 1
      const op_est = sym_infer(this.prg.estimates.ops, var_vals)
      GlobalCounters.global_ops += op_est
      let mem_est = sym_infer(this.prg.estimates.ops, var_vals)
      GlobalCounters.global_mem += mem_est
      if (et !== undefined) GlobalCounters.time_sum_s += et
      if (vars.DEBUG >= 2) {
        const lds_est = sym_infer(this.prg.estimates.lds, var_vals)
        mem_est = Math.min(mem_est, lds_est) // there can't be more memory accessed than loads/stores. remove this when symbolic is fixed
        const ptm = et === undefined ? '' : et > 0.01 ? colored(`${(et * 1e3).toFixed(2).padStart(9)}ms`, 'yellow') : `${(et * 1e6).toFixed(2).padStart(9)}us`
        console.log(
          colored(`*** ${this.prg.device.slice(0, 7).padEnd(7)} ${GlobalCounters.kernel_count.toString().padStart(4)}`, jit ? 'magenta' : (this.prg.first_run ? 'green' : undefined)) +
            ` ${this.prg.display_name + ' '.repeat(41 - ansilen(this.prg.display_name))} arg ` +
            `${bufs.length.toString().padStart(2)} mem  ${(GlobalCounters.mem_used / 1e9).toFixed(2)} GB ` +
            (et === undefined ? '' : `tm ${ptm}/${(GlobalCounters.time_sum_s * 1e3).toFixed(2).padStart(9)}ms (${(op_est / ((et || 1e-20) * 1e9)).toFixed(2).padStart(9)} GFLOPS ${(mem_est / ((et || 1e-20) * 1e9)).toFixed(1).padStart(6)}|${(lds_est / ((et || 1e-20) * 1e9)).toFixed(1).padEnd(7)} GB/s)` + ` ${this.metadata?.length ? list_str(this.metadata.map((m) => m.name)) : ''}`),
        )
      }
      this.prg.first_run = false
    }
    return et
  }
}
// NOTE: ctx is the buffers
export const si_lowerer = new PatternMatcher<Buffer[], Promise<[Runner, Buffer[]]> | [Runner, Buffer[]]>([
  new UPat(Ops.SINK).named('sink').fn(({ ctx, sink }) => get_runner(ctx[0].device, sink).then((runner) => [runner, runner.p.globals.map((x) => ctx[x])] as [Runner, Buffer[]])),
  new UPat(Ops.BUFFER_VIEW).fn(({ ctx }) => [new ViewOp(ctx[0]), [...ctx]]),
  [
    new UPat(Ops.COPY).named('copy'),
    ({ ctx, copy }) => [Device.get(ctx[0].device)!.allocator!._transfer && all_same(ctx.map((x) => x.device.split(':')[0])) ? new BufferXfer(ctx[0].nbytes, ctx[0].device, ctx[1].device) : new BufferCopy(ctx[0].nbytes, ctx[0].device, ctx[1].device), [...ctx]],
  ],
])
const lower_schedule_item = async (si: ScheduleItem) => new ExecItem(...await si_lowerer.rewrite(si.ast, si.bufs)!, si.metadata)

export const lower_schedule = async function* (schedule: ScheduleItem[]): AsyncGenerator<ExecItem, void, unknown> {
  while (schedule.length) {
    const si = schedule.shift()
    try {
      yield await lower_schedule_item(si!)
    } catch (e) {
      if (vars.DEBUG >= 2) {
        console.log(`error lowering ${si!.ast.op}`)
        console.log('tensor operations:')
        console.log(si!.metadata)
      }
      throw e
    }
  }
}
// // **************** main run function ****************

export const capturing: TinyJit<any, any>[] = [] // put classes with an add method in here

export const run_schedule = async (schedule: ScheduleItem[], var_vals?: Map<Variable, number>, do_update_stats = true) => {
  for await (const ei of lower_schedule(schedule)) {
    if (capturing.length && vars.CAPTURING) capturing[0].add(ei)
    await ei.run(var_vals, undefined, undefined, do_update_stats)
  }
}
