// from typing import List, Dict, Optional, cast, Generator, Tuple
// import time, pprint
// from dataclasses import dataclass, replace
// from tinygrad.helpers import colored, getenv, DEBUG, GlobalCounters, ansilen, BEAM, NOOPT, all_int, CAPTURING, Metadata, Context, TRACEMETA
// from tinygrad.ops import Ops, UOp, Variable, sym_infer, sint
// from tinygrad.dtype import dtypes
// from tinygrad.device import Device, Buffer
// from tinygrad.renderer import Renderer, ProgramSpec
// from tinygrad.codegen.kernel import Kernel
// from tinygrad.engine.schedule import ScheduleItem

import { Kernel } from '../codegen/kernel.ts'
import { Buffer, Device } from '../device.ts'
import { all_int, assert, BEAM, CAPTURING, colored, DEBUG, getEnv, getNumberEnv, GlobalCounters, Metadata, NOOPT, replace, zip } from '../helpers.ts'
import { idiv, Ops, sint, sym_infer, UOp, Variable } from '../ops.ts'
import { ProgramSpec, Renderer } from '../renderer/index.ts'
import { ScheduleItem } from './schedule.ts'
import { optimize_local_size } from './search.ts'

// **************** Program Creation ****************

// const [logkerns, logkerns_level] = [getEnv('LOGKERNS', '') ? open(getEnv('LOGKERNS', ''), 'a') : undefined, getNumberEnv('LOGKERNS_LEVEL', 1)]
export const get_kernel = (renderer: Renderer, ast: UOp): Kernel => {
  if (DEBUG >= 5) console.log(ast)
  const k = new Kernel(ast, renderer).required_optimizations()
  if (!NOOPT) {
    const used_tensor_cores = k.apply_tensor_cores(getNumberEnv('TC', 1))
    if (!used_tensor_cores) k.hand_coded_optimizations()
  }
  // if (logkerns !== undefined) logkerns.writelines([`${(k.ast, k.applied_opts)}\n`])
  if (DEBUG >= 5) console.log((k.ast, k.applied_opts)) // print here to show final applied_opts for all kernels instead of just in beam_search
  return k
}
// **************** Runners ****************

export class Runner {
  first_run = true

  constructor(public display_name: string, public device: string, public op_estimate: sint = 0, public mem_estimate: sint = 0, public lds_estimate: sint = mem_estimate) {}
  get dev() {
    return Device.get(this.device)
  }
  exec = (rawbufs: Buffer[], var_vals?: Map<Variable, number>): number | undefined => this.call(rawbufs, var_vals === undefined ? new Map() : var_vals)
  call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false): number | undefined => {
    throw new Error('override this')
  }
}
export class CompiledRunner extends Runner {
  lib: Uint8Array
  _prg: any
  constructor(public p: ProgramSpec, precompiled?: Uint8Array) {
    super(p.name, p.device, p.op_estimate, p.mem_estimate, p.lds_estimate)
    if (DEBUG >= 4) console.log(p.src)
    this.lib = precompiled !== undefined ? precompiled : Device.get(p.device).compiler.compile_cached(p.src)
    if (DEBUG >= 6) Device.get(p.device).compiler.disassemble(this.lib)
    this._prg = Device.get(p.device).runtime(p.function_name, this.lib)
  }
  __reduce__ = () => [this.p, this.lib]

  override call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false): number | undefined => {
    let [global_size, local_size] = this.p.launch_dims(var_vals)
    if (global_size !== undefined && local_size === undefined && all_int(this.p.global_size!)) {
      // TODO: this === copied from get_program
      // from tinygrad.engine.search import optimize_local_size
      local_size = optimize_local_size(this._prg, global_size, rawbufs)
      global_size = zip(global_size, local_size!).map(([g, l]) => g % l === 0 ? idiv(g, l) : g / l)
      this.p.global_size = global_size
      this.p.global_size = local_size
    }
    const lra: Record<string, any> = {}
    if (global_size) {
      lra['global_size'] = global_size
      assert(global_size.length === 3, 'global size must have len 3')
    }
    if (local_size) {
      lra['local_size'] = local_size
      assert(local_size.length === 3, 'local size must have len 3')
    }
    return this._prg(rawbufs.map((x) => x._buf), { ...lra, vals: this.p.vars?.map((k) => var_vals.get(k)), wait })
  }
}
export class EmptyOp extends Runner {
  constructor(buf: Buffer) {
    super(colored(`empty ${buf.size.toString().padStart(10)} ${buf.dtype}`, 'yellow'), buf.device)
  }
  override call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false): number | undefined => undefined
}
export class ViewOp extends Runner {
  constructor(buf: Buffer) {
    super(colored(`view ${buf.nbytes.toString().padStart(8)} @ ${buf.offset.toString().padEnd(10)}`, 'yellow'), buf.device)
  }
  override call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false): number | undefined => {
    assert(rawbufs[0]._base !== undefined && rawbufs[0]._base === rawbufs[1].base, `must be base ${rawbufs}`)
    return undefined
  }
}
export class BufferCopy extends Runner {
  constructor(total_sz: number, dest_device: string, src_device: string) {
    if (total_sz >= 1e6) name = `copy ${(total_sz / 1e6).toFixed(2)}M, ${dest_device.slice(0, 7).padStart(7)} <- ${src_device.slice(0, 7).padEnd(7)}`
    else name = `copy ${total_sz.toFixed(0).padStart(8)}, ${dest_device.slice(0, 7).padStart(7)} <- ${src_device.slice(0, 7).padEnd(7)}`
    super(colored(name, 'yellow'), dest_device, 0, total_sz)
  }
  copy = (dest: Buffer, src: Buffer) => {
    const disk_supports_fast_copyout = src.device.startsWith('DISK') //TODO: && 'io_uring' in src.allocator!.dev && src.allocator!.dev.fd !== undefined
    if (src.device.startsWith('DISK') && 'copy_from_disk' in dest.allocator! && disk_supports_fast_copyout && src.nbytes >= 4096) {
      throw new Error('TODO implement copy_from_disk')
      // dest.allocator.copy_from_disk(dest._buf, src._buf, src.nbytes)
    } else if (src.device.startsWith('DISK') && '_as_buffer' in dest.allocator!) {
      //       // fast(ish) path, uses readinto in diskbuffers
      src.allocator!._copyout((dest.allocator._as_buffer as any)(dest._buf), src._buf)
    } else {
      dest.copyin(src.as_buffer(true)) // may allocate a CPU buffer depending on allow_zero_copy
    }
  }
  override call = (rawbufs: Buffer[], var_vals: Map<Variable, number>, wait = false) => {
    const [dest, src] = rawbufs.slice(0, 2)
    assert(dest.size === src.size && dest.dtype === src.dtype, `buffer copy mismatch, ${dest.size} !== ${src.size}, ${dest.dtype} !== ${src.dtype}`)
    const st = performance.now()
    this.copy(dest, src)
    if (wait) {
      Device.get(dest.device).synchronize()
      return performance.now() - st
    }
  }
}
class BufferXfer extends BufferCopy {
  override copy = (dest: Buffer, src: Buffer) => {
    throw new Error('TODO implement _transfer')
    // return dest.allocator._transfer(dest._buf, src._buf, dest.nbytes, src_dev = src.allocator.dev, dest_dev = dest.allocator.dev)
  }
}
// // **************** method cache ****************

type Key = [string, string, number, number, boolean]
const method_cache = new Map<Key, CompiledRunner>()
export const get_runner = (device: string, ast: UOp): CompiledRunner => {
  const ckey = [device, ast.key, BEAM, NOOPT, false] as Key
  const cret = method_cache.get(ckey)
  if (cret) return cret
  const bkey = [device.split(':')[0], ast.key, BEAM, NOOPT, true] as Key
  let ret
  const bret = method_cache.get(bkey)
  if (bret) {
    ret = new CompiledRunner(replace(bret.p, { device: device }), bret.lib)
    method_cache.set(ckey, ret)
  } else {
    const prg: ProgramSpec = get_kernel(Device.get(device).renderer, ast).to_program()
    // if (getEnv('FUZZ_UOPS')) {
    //       from test.external.fuzz_uops import UOpsFuzzerRunner
    //       return UOpsFuzzerRunner(replace(prg, device=device))
    // }
    ret = new CompiledRunner(replace(prg, { device: device }))
    method_cache.set(ckey, ret), method_cache.set(bkey, ret)
  }
  return ret
}
// // **************** lowering functions ****************

// @dataclass(frozen=true)
export class ExecItem {
  constructor(public prg: Runner, public bufs: (Buffer | undefined)[], public metadata?: Metadata[]) {}
  run = (_var_vals?: Map<Variable, number>, wait = false, jit = false, do_update_stats = true): number | undefined => {
    const var_vals = _var_vals === undefined ? new Map<UOp, number>() : _var_vals
    const bufs = jit ? this.bufs.map((x) => x!) : this.bufs.map((x) => x!.ensure_allocated())
    const et = this.prg.call(bufs, var_vals, wait = wait || DEBUG >= 2)
    if (do_update_stats) {
      GlobalCounters.kernel_count += 1
      const op_est = sym_infer(this.prg.op_estimate, var_vals)
      GlobalCounters.global_ops += op_est
      let mem_est = sym_infer(this.prg.mem_estimate, var_vals)
      GlobalCounters.global_mem += mem_est
      if (et !== undefined) GlobalCounters.time_sum_s += et
      if (DEBUG >= 2) {
        const lds_est = sym_infer(this.prg.lds_estimate, var_vals)
        mem_est = Math.min(mem_est, lds_est) // there can't be more memory accessed than loads/stores. remove this when symbolic === fixed
        const ptm = et !== undefined ? (et > 0.01 ? colored(`${(et * 1e3).toFixed(2)}ms`, 'yellow') : `${(et * 1e6).toFixed(2)}us`) : ''
        // TODO: Fucking hell, skipping this for now
        //         console.log(`${colored(f'*** {this.prg.device[:7]:7s} ${GlobalCounters.kernel_count:4d}', 'magenta' if (jit else ('green' if this.prg.first_run else undefined))} ${this.prg.display_name+' '*(41-ansithis.prg.display_name.length)} arg ${bufs.length:2d} mem ${GlobalCounters.mem_used/1e9:5.2f} GB ` +  // noqa) E501
        //               (string() if (et === undefined else `tm ${ptm}/${GlobalCounters.time_sum_s*1e3:9.2f}ms (${op_est/((et || 1e-20)*1e9):9.2f} GFLOPS ${mem_est/((et || 1e-20)*1e9):6.1f}|${lds_est/((et || 1e-20)*1e9):<7.1f} GB/s)` +  // noqa) E501
        //                ` ${[repr(m) if TRACEMETA >= 2 else string(m) for m in this.metadata] if this.metadata else ''}`))
      }
      this.prg.first_run = false
    }
    return et
  }
}
export const lower_schedule_item = (si: ScheduleItem): ExecItem => {
  assert(new Set(si.bufs.map((x) => x.device)).size === 1 || si.ast.op === Ops.COPY)
  if (si.ast.op === Ops.SINK) {
    const runner = get_runner(si.outputs[0].device, si.ast)
    return new ExecItem(runner, runner.p.globals.map((x) => si.bufs[x]), si.metadata)
  }
  const [out, arg] = [si.outputs[0], si.ast.arg]
  if (si.ast.op === Ops.COPY) {
    let kernel_type = BufferCopy
    if ('_transfer' in Device.get(out.device).allocator! && out.device.split(':')[0] === si.inputs[0].device.split(':')[0]) {
      kernel_type = BufferXfer
    }
    return new ExecItem(new kernel_type(arg, out.device, si.inputs[0].device), si.bufs)
  }
  if (si.ast.op === Ops.EMPTY) return new ExecItem(new EmptyOp(out), si.bufs)
  if (si.ast.op === Ops.BUFFER_VIEW) return new ExecItem(new ViewOp(out), si.bufs)
  throw new Error(`don't know how to lower ${si.ast}`)
}
export const lower_schedule = function* (schedule: ScheduleItem[]): Generator<ExecItem, void, unknown> {
  while (schedule.length) {
    const si = schedule.shift()
    try {
      yield lower_schedule_item(si!)
    } catch (e) {
      if (DEBUG >= 2) {
        console.log(`error lowering ${si!.ast.op}`)
        console.log('tensor operations:')
        console.log(si!.metadata) //indent=2)
      }
      throw e
    }
  }
}
// // **************** main run function ****************

const capturing: any[] = [] // put classes with an add method in here

export const run_schedule = (schedule: ScheduleItem[], var_vals?: Map<Variable, number>, do_update_stats = true) => {
  for (const ei of lower_schedule(schedule)) {
    if (capturing.length && CAPTURING) capturing[0].add(ei)
    ei.run(var_vals, undefined, undefined, do_update_stats)
  }
}
