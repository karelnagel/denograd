import { Buffer, type Compiled, Device, type DeviceType, uop_realized } from '../device.ts'
import type { DType } from '../dtype.ts'
import { ArrayMap, colored, DEBUG, DefaultMap, flatten, get_number_env, is_eq, JIT, merge_maps, partition, WeakKeyMap } from '../helpers.ts'
import { get_parameters } from '../nn/state.ts'
import { Ops, sym_infer, UOp, type Variable } from '../ops.ts'
import { Estimates } from '../renderer/index.ts'
import type { ShapeTracker } from '../shape/shapetracker.ts'
import { Tensor } from '../tensor.ts'
import { _internal_memory_planner } from './memory.ts'
import { BufferCopy, BufferXfer, CompiledRunner, ExecItem, Runner, ViewOp } from './realize.ts'

export class GraphException extends Error {}

export const apply_graph_to_jit = (jit_cache: ExecItem[], input_rawbuffers: Buffer[], var_vals: Map<Variable, number>, max_batch_size = 0): ExecItem[] => {
  // Split JIT cache into batches for faster graph execution.
  // This allows the accelerator to run some batches while subsequent graphs are still being updated.
  const graphed_jit_cache: ExecItem[] = []
  let current_batch: ExecItem[] = []
  let current_device: undefined | Compiled

  const flush_batch = () => {
    try {
      if (current_batch.length <= 1 || current_device === undefined) throw new GraphException("only one kernel doesn't graph")
      const graph_runner = current_device.graph(current_batch, input_rawbuffers, var_vals)
      // clear jit inputs to allow their memory to be freed/reused
      for (const [j, i] of graph_runner.input_replace.keys()) graph_runner.jit_cache[j].bufs[i] = undefined
      graphed_jit_cache.push(new ExecItem(graph_runner, input_rawbuffers))
      max_batch_size *= 2
      if (DEBUG >= 2) console.log(`JIT GRAPHing batch with ${current_batch.length} kernels on device ${current_device}`)
    } catch (e) {
      if (e instanceof GraphException) {
        graphed_jit_cache.push(...current_batch)
        if (DEBUG >= 2) console.log(`JIT GRAPHing failed batch with ${current_batch.length} kernels on device ${current_device}: ${e}`)
      }
      throw e
    }
    current_batch = []
    current_device = undefined
  }
  for (const ji of jit_cache) {
    if (ji.prg instanceof ViewOp) continue
    let ji_graph_dev: undefined | Compiled // device on which the ji will be graphed. Not graphed if None.
    if (ji.prg instanceof CompiledRunner) ji_graph_dev = ji.prg.dev
    else if (ji.prg instanceof BufferXfer && ji.bufs[0] && ['CUDA', 'NV', 'AMD'].includes(ji.bufs[0].device.split(':', 1)[0])) ji_graph_dev = Device.get(ji.bufs[0].device)

    const graph_class = ji_graph_dev ? (ji_graph_dev.graph instanceof Object ? ji_graph_dev.graph.func : ji_graph_dev.graph) : undefined // KAREL: TODO: Object should be functools.partial
    const can_be_graphed = ji_graph_dev && ji_graph_dev.graph
    const can_share_graph = ji_graph_dev === current_device || (graph_class instanceof MultiGraphRunner) && ji_graph_dev?.constructor.name === current_device?.constructor.name
    const can_extend_graph_batch = can_be_graphed && (max_batch_size === 0 || current_batch.length < max_batch_size) && can_share_graph
    if (!can_extend_graph_batch && current_batch.length > 0) flush_batch()

    if (can_be_graphed) current_batch.push(ji)
    else graphed_jit_cache.push(ji)

    current_device = ji_graph_dev
  }

  if (current_batch.length > 0) flush_batch()
  return graphed_jit_cache
}
export const get_input_replace = (jit_cache: ExecItem[], input_rawbuffers: Buffer[]): ArrayMap<[number, number], number> => {
  const input_replace = new ArrayMap<[number, number], number>()
  for (const [j, ji] of jit_cache.entries()) {
    for (const [i, a] of ji.bufs.entries()) {
      if (input_rawbuffers.includes(a!)) input_replace.set([j, i], input_rawbuffers.indexOf(a!))
    }
  }
  return input_replace
}

export class GraphRunner extends Runner {
  input_replace: ArrayMap<[number, number], number>

  var_vals_replace = new Map<number, number[]>()
  launch_dims_replace = new Map<number, [number?, number?]>()
  launch_dims_base = new Map<number, [number[], number[]]>()
  vars: UOp[]
  symbolic_dims: ArrayMap<(number[] | undefined), number> // value is array index in python

  // used in MultiGraphRunner. the ints are id() of _bufs
  w_dependency_map = new Map<string, any>()
  r_dependency_map = new DefaultMap<string, any[]>(undefined, () => [])
  constructor(public jit_cache: ExecItem[], input_rawbuffers: Buffer[], var_vals: Map<Variable, number>) {
    super(colored(`<batched ${jit_cache.length}>`, 'cyan'), jit_cache[0].prg.device.split(':')[0] as DeviceType, new Estimates())
    this.input_replace = get_input_replace(jit_cache, input_rawbuffers)
    const is_sym_dim = (dim: number[]): boolean => !dim.every((d) => typeof d === 'number')

    this.vars = [...var_vals.keys()].toSorted((a, b) => b.expr - a.expr)
    this.symbolic_dims = new ArrayMap([
      ...jit_cache.filter((ji) => ji.prg instanceof CompiledRunner && ji.prg.p.local_size?.length && is_sym_dim(ji.prg.p.local_size)).map((ji) => (ji.prg as CompiledRunner).p.local_size),
      ...jit_cache.filter((ji) => ji.prg instanceof CompiledRunner && ji.prg.p.global_size?.length && is_sym_dim(ji.prg.p.global_size)).map((ji) => (ji.prg as CompiledRunner).p.global_size),
    ].map((key, i) => [key, i]))
    const find_symbolic_dim = (dim?: number[]) => dim !== undefined && this.symbolic_dims.has(dim) ? this.symbolic_dims.get(dim) : undefined

    for (const [j, ji] of jit_cache.entries()) {
      this.estimates = this.estimates.add(ji.prg.estimates)
      if (ji.prg instanceof CompiledRunner) {
        if (ji.prg.p.vars.length) this.var_vals_replace.set(j, ji.prg.p.vars.map((v) => this.vars.indexOf(v)))

        const global_dim_idx = find_symbolic_dim(ji.prg.p.global_size), local_dim_idx = find_symbolic_dim(ji.prg.p.local_size)
        if (global_dim_idx !== undefined || local_dim_idx !== undefined) {
          this.launch_dims_replace.set(j, [global_dim_idx, local_dim_idx])
          if (ji.prg.p.global_size === undefined || ji.prg.p.local_size === undefined) throw new Error()
          this.launch_dims_base.set(j, [ji.prg.p.global_size, ji.prg.p.local_size])
        }
      }
    }
    this.estimates = this.estimates.simplify()
  }

  *updated_vars(var_vals: Map<Variable, number>) {
    const vals = this.vars.map((v) => var_vals.get(v))
    for (const [j, vidxs] of this.var_vals_replace.entries()) {
      for (const [i, v] of vidxs.entries()) yield [j, i, vals[v]]
    }
  }
  *updated_launch_dims(var_vals: Map<Variable, number>) {
    const dims = this.symbolic_dims.keys().map((dim) => dim!.map((s) => sym_infer(s, var_vals)))
    for (const [j, [gl, lc]] of this.launch_dims_replace.entries()) {
      yield [j, gl !== undefined ? dims[gl] : this.launch_dims_base.get(j)![0], lc !== undefined ? dims[lc] : this.launch_dims_base.get(j)![1]]
    }
  }
  _access_resources = (rawbufs: Buffer[], write: number[], new_dependency: any) => {
    // To synchronize access to resources, we monitor the necessary prerequisites for accessing each resource,
    // whether for write or read operations. A resource can be accessed by either a single writer or multiple readers.
    const wait_nodes = []

    for (const [i, rawbuf] of rawbufs.entries()) {
      if (this.w_dependency_map.has(rawbuf.base.key)) wait_nodes.push(this.w_dependency_map.get(rawbuf.base.key))
      if (write.includes(i)) {
        if (this.r_dependency_map.has(rawbuf.base.key)) {
          wait_nodes.push(...this.r_dependency_map.get(rawbuf.base.key))
          this.r_dependency_map.delete(rawbuf.base.key)
        }
        this.w_dependency_map.set(rawbuf.base.key, new_dependency)
      } else this.r_dependency_map.get(rawbuf.base.key).push(new_dependency)
    }
    // return  list({id(x):x for x in wait_nodes}.values())
    return wait_nodes //TODO: id() not good rn
  }
}

// a marker for your graph supporting multiple devices of the same type
export class MultiGraphRunner extends GraphRunner {}

export const update_depends = (depends: Set<Buffer | undefined>, jit_cache: ExecItem[]) => {
  for (const ei of jit_cache) {
    if (ei.bufs.some((b) => depends.has(b))) {
      if (ei.prg instanceof CompiledRunner) {
        for (const out of ei.prg.p.outs) if (!ei.prg.p.ins.includes(out)) depends.add(ei.bufs[out])
      }
      if (ei.prg instanceof BufferCopy || ei.prg instanceof BufferXfer) depends.add(ei.bufs[0])
    }
  }
}
// ReturnType = TypeVar('ReturnType')
export class CapturedJit<Return extends any> {
  constructor(
    public ret: Return, // includes the Tensors or any other returned object
    public jit_cache: ExecItem[],
    public input_replace: ArrayMap<[number, number], number>,
    public extra_view_inputs: [number, number, string, number, DType][],
    public expected_names: number[],
    public expected_st_vars_dtype_device: [ShapeTracker, Variable[], DType, DeviceType | DeviceType[]][],
  ) {
    this.init()
  }

  _jit_cache!: ExecItem[]
  _input_replace!: ArrayMap<[number, number], number>
  _first_run!: boolean
  init = () => {
    this._jit_cache = this.jit_cache
    this._input_replace = this.input_replace
    this._first_run = true
    this._clear_inputs()
  }
  _clear_inputs = () => {
    for (const [j, i] of this._input_replace.keys()) this._jit_cache[j].bufs[i] = undefined
  }
  free_intermediates = () => {
    const depends = new Set<Buffer | undefined>([undefined])
    update_depends(depends, this.jit_cache)
    for (const b of depends) {
      if (b !== undefined) b.deallocate()
    }
    this.init() // reset the graph state
  }
  // jit exec
  call = (input_buffers: Buffer[], var_vals: Map<Variable, number>): Return => {
    // assign inputs
    for (const [idx, offset, device, size, dtype] of this.extra_view_inputs) {
      input_buffers.push(new Buffer(device as DeviceType, size, dtype, undefined, undefined, undefined, undefined, input_buffers[idx], offset).ensure_allocated())
    }
    for (const [[j, i], input_idx] of this._input_replace.entries()) this._jit_cache[j].bufs[i] = input_buffers[input_idx]

    // Condense the items into a graph executor.
    if (this._first_run) {
      // allocate intermediates if freed
      for (const ji of this.jit_cache) {
        for (const b of ji.bufs) {
          if (b !== undefined) b.ensure_allocated()
        }
      }
      // create graph if needed
      if (JIT < 2) {
        this._jit_cache = apply_graph_to_jit(this.jit_cache, input_buffers, var_vals, get_number_env('JIT_BATCH_SIZE', 32))
        this._input_replace = get_input_replace(this._jit_cache, input_buffers)
      }
      this._first_run = false
    }

    if (DEBUG >= 1 && this._jit_cache.length >= 10) console.log(`jit execs ${this._jit_cache.length} kernels`)
    for (const ei of this._jit_cache) ei.run(var_vals, undefined, true)
    this._clear_inputs()
    return this.ret
  }
}
export const _prepare_jit_inputs = async (...args: any[]) => {
  const input_tensors: [number, Tensor][] = args.filter((t) => t instanceof Tensor).map((t, name) => [name, t as Tensor])
  const names = input_tensors.map(([name]) => name), tensors = input_tensors.map(([_, t]) => t)
  if (tensors.length) await Tensor.realize(tensors)
  // TODO: should we be unpacking multi here?
  const lbs = flatten(tensors.map((t) => t.lazydata.op === Ops.MULTI ? t.lazydata.src : [t.lazydata]))
  const input_buffers = lbs.filter((lb) => uop_realized(lb.base) !== undefined).map((lb) => uop_realized(lb.base))
  if (new Set(input_buffers).size !== input_buffers.length) throw new Error('duplicate inputs to JIT')
  const st_varval_dtype_device = lbs.map((lb) => [...lb.st!.unbind(), lb.dtype, lb.device] as const)
  const var_vals = merge_maps([
    ...st_varval_dtype_device.map((x) => x[1]),
    new Map(args.filter((v) => v instanceof UOp).map((v) => v.unbind())),
  ])
  const st_vars_dtype_device = st_varval_dtype_device.map((x) => [x[0], [...x[1].keys()].toSorted((a, b) => b.expr - a.expr), x[2], x[3]] satisfies [ShapeTracker, UOp[], DType, DeviceType | DeviceType[]])
  return [input_buffers as Buffer[], var_vals, names, st_vars_dtype_device] as const
}

const capturing = new Set<TinyJit<any, any>>()
export class TinyJit<Args extends any[], Return extends any> {
  cnt: number
  _buffer_replace?: WeakKeyMap<Buffer, Buffer>
  _jit_cache?: ExecItem[]
  constructor(public fxn?: (...a: Args) => Promise<Return>, public captured?: CapturedJit<Return>, public prune = false) {
    if (!fxn && !captured) throw new Error('need either a function or a CapturedJit')
    this.cnt = this.fxn === undefined ? 2 : 0
  }
  add_buffer = (b: Buffer): Buffer => {
    if (!this._buffer_replace) throw new Error()
    const found = this._buffer_replace.get(b)
    if (found) return found
    if (b.is_allocated() || b.lb_refcount > 0) return b
    const ret = b._base !== undefined ? new Buffer(b.device, b.size, b.dtype, undefined, undefined, undefined, undefined, this.add_buffer(b._base), b.offset) : new Buffer(b.device, b.size, b.dtype, undefined, b.options)
    this._buffer_replace.set(b, ret)
    return ret
  }
  add = (ei: ExecItem) => {
    this._jit_cache!.push(new ExecItem(ei.prg, ei.bufs.filter((buf) => buf !== undefined).map((buf) => this.add_buffer(buf))))
  }
  reset = () => {
    if (this.fxn === undefined) throw new Error("can't reset without function")
    this.cnt = 0
    this.captured = undefined
  }

  // keep legacy code working
  get jit_cache(): ExecItem[] {
    return this.captured !== undefined ? this.captured._jit_cache : []
  }
  get input_replace(): ArrayMap<[number, number], number> {
    return this.captured !== undefined ? this.captured._input_replace : new ArrayMap()
  }

  // get = (obj, objtype) => () => this.call(obj) // add support for instance methods

  call = async (...args: Args): Promise<Return> => {
    const [input_buffers, var_vals, names, st_vars_dtype_device] = await _prepare_jit_inputs(...args)
    let ret: Return
    if (!JIT || this.cnt === 0) {
      // jit ignore
      if (!this.fxn) throw new Error()
      // TODO:
      //   with Context(BEAM=0 if getenv("IGNORE_JIT_FIRST_BEAM") else BEAM.value):
      ret = await this.fxn(...args)
      const params = get_parameters(ret)
      if (params.length) Tensor.realize([params[0], ...params.slice(1)])
    } else if (this.cnt === 1) {
      // jit capture
      if (!this.fxn) throw new Error()
      // if (capturing) throw new Error(`having TinyJit inside another TinyJit is not supported ${capturing.length} ${capturing}`)
      this._jit_cache = []
      this._buffer_replace = new WeakKeyMap()
      // TODO: should we always disable the memory planner here? it must be off for prune
      // TODO:
      //   with Context(BEAM=getenv("JITBEAM", BEAM.value), NO_MEMORY_PLANNER=int(this.prune)):
      capturing.add(this)
      try {
        ret = await this.fxn(...args)
        const params = get_parameters(ret)
        if (params.length) await Tensor.realize(params)
      } catch (e) {
        throw e
      } finally {
        capturing.clear()
      }
      let jit_cache = this._jit_cache
      this._buffer_replace = undefined, this._jit_cache = undefined
      if (!jit_cache.length) throw new Error("didn't JIT anything!")
      if (DEBUG >= 1) console.log(`JIT captured ${jit_cache.length} kernels with ${input_buffers.length} inputs`)

      // track inputs that are views of buffers
      // TODO: eventually expected_buffers should live in ExecItem
      const extra_view_inputs: [number, number, string, number, DType][] = []
      for (const item of jit_cache) {
        for (const b of item.bufs) {
          if (b !== undefined && b._base !== undefined && input_buffers.includes(b._base)) {
            input_buffers.push(b)
            extra_view_inputs.push([input_buffers.indexOf(b.base), b.offset, b.device, b.size, b.dtype])
          }
        }
      }

      // prune independent kernels (optional)
      if (this.prune) {
        const depends = new Set(input_buffers)
        update_depends(depends, jit_cache)
        const [pruned, onetime] = partition(jit_cache, (ei) => !(ei.prg instanceof CompiledRunner) || ei.prg.p.outs.some((out) => depends.has(ei.bufs[out] as Buffer)))
        if (DEBUG >= 1) console.log(`pruned from ${jit_cache.length} -> ${pruned.length} kernels`)
        // run the onetime kernels here
        for (const ei of onetime) {
          for (const b of ei.bufs) b!.ensure_allocated()
          ei.run(var_vals, undefined, true)
        }
        jit_cache = pruned
      }
      // memory planning (optional)
      // Exclude buffers involved in transfer ops to preserve parallelism.
      const noopt_buffers = jit_cache.filter((ji) => ji.prg instanceof BufferXfer).flatMap((ji) => ji.bufs) as Buffer[]
      const assigned = _internal_memory_planner(jit_cache.map((item) => item.bufs as Buffer[]), noopt_buffers, 'JIT ')
      jit_cache = jit_cache.map((item) => new ExecItem(item.prg, item.bufs.filter((b) => b !== undefined).map((b) => (assigned.get(b) || b)!.ensure_allocated())))

      const input_replace = get_input_replace(jit_cache, input_buffers)
      if (DEBUG >= 1 && new Set(input_replace.values()).size !== input_buffers.length) throw new Error('WARNING: some input tensors not found')

      // set this for next run
      this.captured = new CapturedJit(ret, jit_cache, input_replace, extra_view_inputs, names, st_vars_dtype_device)
    } else {
      // jit exec
      if (this.captured === undefined) throw new Error()
      if (!is_eq(this.captured.expected_names, names)) throw new Error(`args mismatch in JIT: ${this.captured.expected_names} != ${names}`)
      if (!is_eq(this.captured.expected_st_vars_dtype_device, st_vars_dtype_device)) throw new Error(`args mismatch in JIT: ${this.captured.expected_st_vars_dtype_device} != ${st_vars_dtype_device}`)
      ret = this.captured.call(input_buffers, var_vals)
    }
    this.cnt += 1
    return ret
  }
}
