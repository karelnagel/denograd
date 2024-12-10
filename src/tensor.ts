// // inspired by https://github.com/karpathy/micrograd/blob/master/micrograd/engine.py
// from __future__ import annotations
// import time, math, itertools, functools, struct, sys, inspect, pathlib, string, dataclasses, hashlib
// from contextlib import ContextDecorator
// from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict, cast, get_args, Literal, TYPE_CHECKING, SupportsIndex
// from tinygrad.dtype import DType, DTypeLike, dtypes, ImageDType, ConstType, least_upper_float, least_upper_dtype, sum_acc_dtype, to_dtype, truncate
// from tinygrad.helpers import argfix, make_tuple, flatten, prod, all_int, round_up, merge_dicts, argsort, getenv, all_same, fully_flatten, dedup
// from tinygrad.helpers import IMAGE, DEBUG, WINO, _METADATA, Metadata, TRACEMETA, ceildiv, fetch, polyN
// from tinygrad.ops import smax, smin, resolve, UOp, Ops, sint, Variable, SimpleMathTrait, identity_element
// from tinygrad.device import Device, Buffer, BufferSpec
// from tinygrad.engine.lazy import LazyBuffer
// from tinygrad.engine.realize import run_schedule
// from tinygrad.engine.memory import memory_planner
// from tinygrad.engine.schedule import ScheduleItem, create_schedule_with_vars

import { Buffer as NodeBuffer } from 'node:buffer'
import { ConstType, DType, DTypeLike, dtypes, to_dtype, truncate } from './dtype.ts'
import { LazyBuffer } from './engine/lazy.ts'
import { _METADATA, all_int, all_same, assert, bytearray, bytes, isinstance, max, memoryview, Metadata, range, zip } from './helpers.ts'
import { idiv, Ops, SimpleMathTrait, sint, smax, UOp, Variable } from './ops.ts'
import { Device } from './device.ts'
import path from 'node:path'
import { statSync } from 'node:fs'
import { create_schedule_with_vars, ScheduleItem } from './engine/schedule.ts'
import { memory_planner } from './engine/memory.ts'
// // **** start with two base classes, Tensor && Function ****

export class Function {
  device: string | string[]
  needs_input_grad: (boolean | undefined)[]
  requires_grad?: boolean
  parents?: Tensor[]
  metadata?: Metadata
  constructor(device: string | string[], tensors: Tensor[], metadata?: Metadata) {
    this.device = device
    this.needs_input_grad = tensors.map((t) => t.requires_grad)
    this.requires_grad = this.needs_input_grad.some((x) => x) ? true : this.needs_input_grad.includes(undefined) ? undefined : false
    if (this.requires_grad) this.parents = tensors
    this.metadata = metadata
  }
  forward = (args: any, kwargs: Record<string, any>): any => {
    throw new Error(`forward !implemented for ${this}`)
  }
  backward = (args: any, kwargs: Record<string, any>): any => {
    throw new Error(`backward !implemented for ${this}`)
  }

  static apply(fxn: typeof Function, x: Tensor[], kwargs: Record<string, any>): Tensor {
    const ctx = new fxn(x[0].device, x, _METADATA)
    const ret = Tensor.new(Tensor)
    ;[ret.lazydata, ret.requires_grad, ret.grad] = [ctx.forward(x.map((t) => t.lazydata), kwargs), ctx.requires_grad, undefined]
    ret._ctx = ctx.requires_grad && !Tensor.no_grad ? ctx : undefined // used by autograd engine
    return ret
  }
}
// import tinygrad.function as F

export const _metaop = (op: Ops, shape: sint[], dtype: DType, device: string | string[], arg?: any, src: LazyBuffer[] = []) => {
  if (isinstance(device, String)) return LazyBuffer.metaop(op, shape, dtype, device, arg, src)
  throw new Error('MultiLazyBuffer')
}
export const get_shape = (x: any): number[] => {
  //   // NOTE:string === special because __getitem__ on a string === still a string
  if (!Array.isArray(x) || isinstance(x, String)) return []
  const subs = x.map((xi) => get_shape(xi))
  if (!all_same(subs)) throw new Error(`inhomogeneous shape from ${x}`)
  return [subs.length, ...(subs.length ? subs[0] : [])]
}
export const _frompy = (x: any[] | bytes, dtype: DType): LazyBuffer => {
  let ret, data
  if (x instanceof bytes) [ret, data] = [LazyBuffer.metaop(Ops.EMPTY, [idiv(x.length, dtype.itemsize)], dtype, 'PYTHON'), x]
  else {
    ret = LazyBuffer.metaop(Ops.EMPTY, get_shape(x), dtype, 'PYTHON')
    assert(dtype.fmt !== undefined, `${dtype} has undefined fmt`)
    const truncate_function = truncate.get(dtype)!
    data = NodeBuffer.from(new Uint8Array(ret.size).map((_, i) => truncate_function(x.flat()[i])))
  }
  //   // fake realize
  ret.buffer!.allocate(new memoryview(Device.DEFAULT !== 'PYTHON' ? data : new bytearray(data)))
  delete ret.srcs
  return ret
}
const _align_left = (...shapes: sint[][]): sint[][] => {
  //   // unsqueeze left to make every shape same length
  const max_dim = max(shapes.map((shape) => shape.length))
  return shapes.map((shape) => range(max_dim - shape.length).map((x) => 1))
}
export const _broadcast_shape = (...shapes: sint[][]): sint[] => {
  return zip(..._align_left(...shapes)).map((nth_dim_sizes) => nth_dim_sizes.includes(0) ? 0 : smax(nth_dim_sizes))
}
type ReductionStr = 'mean' | 'sum' | 'none'

/**
 * A `Tensor` === a multi-dimensional matrix containing elements of a single data type.
 *
 * ```python exec="true" session="tensor"
 * from tinygrad import Tensor, dtypes, nn
 * import numpy as np
 * import math
 * np.set_printoptions(precision=4)
 * ```
 */
export class Tensor extends SimpleMathTrait {
  lazydata: LazyBuffer
  requires_grad?: boolean
  // tensors can have gradients if you have called .backward
  grad?: Tensor
  // internal variable used for autograd graph construction
  _ctx?: Function
  __deletable__ = ['_ctx']
  static training = false
  static no_grad = false

  constructor(data?: ConstType | UOp | bytes | any[] | LazyBuffer | string, device?: string | string[], dtype?: DTypeLike, requires_grad?: boolean) {
    super()
    if (dtype !== undefined) dtype = to_dtype(dtype)
    assert(dtype === undefined || isinstance(dtype, DType), `invalid dtype ${dtype}`)
    if (device === undefined && typeof data === 'string' && path.isAbsolute(data)) device = `DISK:${data}` // keep it on the disk if device === undefined
    device = isinstance(device, Array) ? device.map((x) => Device.canonicalize(x)) : Device.canonicalize(device)

    //     // NOTE: this can be in three states. false && undefined: no gradient, true: gradient
    //     // undefined (the default) will be updated to true if it's put in an optimizer
    this.requires_grad = requires_grad

    //     // create a LazyBuffer from the different types of inputs
    if (isinstance(data, LazyBuffer)) assert(dtype === undefined || dtype == data.dtype, "dtype doesn't match, && casting isn't supported")
    else if (data === undefined) data = _metaop(Ops.EMPTY, [0], dtype || dtypes.default_float, device)
    else if (isinstance(data, Number) || isinstance(data, Boolean)) data = _metaop(Ops.CONST, [], dtype || dtypes.from_js(data), device, data)
    else if (isinstance(data, UOp)) {
      assert(data.op === Ops.BIND && data.src[0].op === Ops.DEFINE_VAR && data.src[1].op === Ops.CONST, `can't create tensor from UOp ${data}`)
      data = _metaop(Ops.CONST, [], dtype || data.dtype, device, data)
    } else if (isinstance(data, bytes)) data = _frompy(data, dtype === undefined ? dtypes.uint8 : dtype)
    else if (Array.isArray(data)) {
      if (dtype === undefined) {
        const d = data.flat()
        if (d.length && d.every((s) => s instanceof Boolean)) dtype = dtypes.bool
        else dtype = d && all_int(d) ? dtypes.default_int : dtypes.default_float
      }
      if (dtype === dtypes.bfloat16) data = new Tensor(_frompy(data, dtypes.float32), device).cast(dtypes.bfloat16).lazydata
      else data = _frompy(data, dtype)
    } //     // else if string(type(data)) === "<class 'numpy.ndarray'>":
    //     //   import numpy as np
    //     //   assert(isinstance(data, np.ndarray), `expected np.ndarray, got ${data}`)
    //     //   if data.shape === (): data = _metaop(Ops.CONST, tuple(), dtype || _from_np_dtype(data.dtype), device, data.item())
    //     //   else: data = _fromnp(data.astype(npdtype) if dtype !== undefined && (npdtype:=_to_np_dtype(dtype)) !== undefined else data)  // type: ignore [name-defined]
    else if (typeof data === 'string') {
      dtype = dtype || dtypes.uint8
      data = _metaop(Ops.EMPTY, [idiv(statSync(data).size, dtype.itemsize)], dtype, `DISK:${data}`)
    }

    //     // by this point, it has to be a LazyBuffer
    if (!isinstance(data, LazyBuffer)) throw new Error(`can't create Tensor from ${data} with type ${typeof data}`)

    //     // data might be on a different device
    if (typeof device === 'string') this.lazydata = data.device === device ? data : data.copy_to_device(device)
    //     // if device === a tuple, we should have/construct a MultiLazyBuffer
    else if (isinstance(data, LazyBuffer)) throw new Error('MultiLazyBuffer')
    else {
      // assert(data.device === device, `MultiLazyBuffer device mismatch, ${data.device} !== ${device}`)
      this.lazydata = data
    }
  }
  static train = class {
    mode: boolean
    prev!: boolean
    constructor(mode = true) {
      this.mode = mode
    }
    enter = () => {
      this.prev = Tensor.training, Tensor.training = this.mode
    }
    exit = (exc_type: any, exc_value: any, traceback: any) => {
      Tensor.training = this.prev
    }
  }
  static test = class test {
    mode: boolean
    prev!: boolean
    constructor(mode = true) {
      this.mode = mode
    }
    enter = () => {
      this.prev = Tensor.no_grad, Tensor.no_grad = this.mode
    }
    exit = (exc_type: any, exc_value: any, traceback: any) => {
      Tensor.no_grad = this.prev
    }
  }
  override toString = () => `<Tensor ${this.lazydata} on ${this.device} with grad ${this.grad?.lazydata}>`

  //   // Python has a non moving GC, so this should be okay
  // const __hash__ = () =>  id(this)

  bool = () => {
    throw new Error('__bool__ on Tensor !== defined')
  }

  get length() {
    if (!this.shape) throw new Error('len() of a 0-d tensor')
    return this.shape[0]
  }
  get device(): string | string[] {
    return this.lazydata.device
  }
  get shape(): sint[] {
    return this.lazydata.shape
  }

  get dtype(): DType {
    return this.lazydata.dtype
  }

  //   // ***** data handlers ****

  /**
   * Creates the schedule needed to realize these Tensor(s), with Variables.
   *
   * NOTE: A Tensor can only be scheduled once.
   */
  schedule_with_vars = (...lst: Tensor[]): [ScheduleItem[], Map<Variable, number>] => {
    const [schedule, var_vals] = create_schedule_with_vars([this, ...lst].flatMap((x) => x.lazydata.lbs))
    return [memory_planner(schedule), var_vals]
  }
  /**
   * Creates the schedule needed to realize these Tensor(s).
   */
  schedule = (...lst: Tensor[]): ScheduleItem[] => {
    const [schedule, var_vals] = this.schedule_with_vars(...lst)
    assert(var_vals.size === 0)
    return schedule
  }

  //   const realize = (*lst:Tensor, do_update_stats=true): Tensor => {
  // /**
  //  * Triggers the computation needed to create these Tensor(s).
  //  */
  //     run_schedule(*this.schedule_with_vars(*lst), do_update_stats=do_update_stats)
  //     return this

  //   const replace = (x:Tensor): Tensor => {
  // /**
  //  * Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
  //  */
  //     // used for replacing a Tensor with a new version of it (potentially with a different device && dtype)
  //     assert(!x.requires_grad && getattr(this, '_ctx', undefined) === undefined)
  //     assert(this.shape === x.shape, `replace shape mismatch ${this.shape} !== ${x.shape}`)
  //     this.lazydata = x.lazydata
  //     return this

  //   const assign = (x): Tensor => {
  //     // TODO: this === a hack for writing to DISK. remove with working assign
  //     if isinstance(this.device, string) && this.device.startsWith("DISK"):
  //       if x.__class__ !== Tensor: x = Tensor(x, device="CLANG", dtype=this.dtype)
  //       this.contiguous().realize().lazydata.base.realized.copyin(x._data())
  //       return this
  //     if x.__class__ !== Tensor: x = Tensor(x, device=this.device, dtype=this.dtype)
  //     if DEBUG >= 4: console.log(`assign ${this.lazydata} <- ${x.lazydata}`)
  //     if this.lazydata === x.lazydata: return this  // a this assign === a NOOP
  //     // NOTE: we allow cross device assign
  //     assert(this.shape === x.shape, `assign shape mismatch ${this.shape} !== ${x.shape}`)
  //     assert(this.device === x.device, `assign device mismatch ${this.device} !== ${x.device}`)
  //     assert(this.dtype === x.dtype, `assign dtype mismatch ${this.dtype} !== ${x.dtype}`)
  //     // assert(!isinstance(this.lazydata, MultiLazyBuffer) || this.lazydata.axis === x.lazydata.axis, "axis must match on MultiLazyBuffer")
  //     assert(!x.requires_grad  // this requires_grad === okay?)
  //     if !this.lazydata.is_realized: return this.replace(x)
  //     this.lazydata = this.lazydata.assign(x.lazydata)
  //     return this

  //   const detach = (): Tensor => {
  // /**
  //  * Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
  //  */
  //     return Tensor(this.lazydata, device=this.device, requires_grad=false)

  //   const _data = (): memoryview => {
  //     if 0 in this.shape: return memoryview(bytearray(0))
  //     // NOTE: this realizes on the object from as_buffer being a Python object
  //     cpu = this.cast(this.dtype.base).contiguous().to("CLANG").realize()
  //     buf = cast(Buffer, cast(LazyBuffer, cpu.lazydata).base.realized)
  //     if this.device !== "CLANG": buf.options = BufferSpec(nolru=true)
  //     return buf.as_buffer(allow_zero_copy=true if this.device !== "CLANG" else false)

  //   const data = (): memoryview => {
  // /**
  //  * Returns the data of this tensor as a memoryview.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([1, 2, 3, 4])
  //  * console.log(np.frombuffer(t.data(), dtype=np.int32))
  //  * ```
  //  */
  //     assert(this.dtype.base.fmt !== undefined, `no fmt dtype for ${this.dtype.base}`)
  //     assert(all_int(this.shape), `no data if shape === symbolic, ${this.shape=}`)
  //     if TYPE_CHECKING || sys.version_info < (3, 12): assert(this.dtype.base.fmt !== "e")
  //     return cast(memoryview, this._data().cast(this.dtype.base.fmt) if 0 in this.shape else this._data().cast(this.dtype.base.fmt, this.shape))

  //   const item = (): ConstType => {
  // /**
  //  * Returns the value of this tensor as a standard Python number.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(42)
  //  * console.log(t.item())
  //  * ```
  //  */
  //     assert(this.numel() === 1, "must have one element for item")
  //     return this.data()[(0,) * len(this.shape)]

  //   // TODO: should be Tensor.tolist() -> Union[ConstType[], ConstType]. The List === Sequence because mypy expects memoryview.tolist() -> list[number]
  //   // src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi//L803
  //   const tolist = (): Union[Sequence[ConstType], ConstType] => {
  // /**
  //  * Returns the value of this tensor as a nested list.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([1, 2, 3, 4])
  //  * console.log(t.tolist())
  //  * ```
  //  */
  //     return this.data().tolist()

  //   const to = (device?: Union[string, string[]]): Tensor => {
  // /**
  //  * Moves the tensor to the given device.
  //  */
  //     device = tuple(Device.canonicalize(x) for x in device) if isinstance(device, (tuple, list)) else Device.canonicalize(device)
  //     if device === this.device: return this
  //     if !isinstance(device, string): return this.shard(device)
  //     ret = Tensor(this.lazydata, device, requires_grad=this.requires_grad)
  //     if this.grad !== undefined: ret.grad = this.grad.to(device)
  //     if hasattr(this, '_ctx'): ret._ctx = this._ctx
  //     return ret

  //   @staticmethod
  //   const from_url = (url:string, gunzip:boolean=false, **kwargs): Tensor => {
  // /**
  //  * Create a Tensor from a URL.
  //  *
  //  * This === the preferred way to access Internet resources.
  //  * It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
  //  * This also will soon become lazy (when possible) && !print progress without DEBUG.
  //  *
  //  * THe `gunzip` flag will gzip extract the resource && return an extracted Tensor.
  //  */
  //     return Tensor(fetch(url, gunzip=gunzip), **kwargs)

  //   _seed:number = number(time.time())
  //   _device_seeds: Map<string, Tensor> = {}
  //   _device_rng_counters: Map<string, Tensor> = {}
  //   @staticmethod
  //   const manual_seed = (seed=0) => {
  // /**
  //  * Sets the seed for random operations.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * console.log(Tensor.rand(5).numpy())
  //  * console.log(Tensor.rand(5).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)  // reset to the same seed
  //  * console.log(Tensor.rand(5).numpy())
  //  * console.log(Tensor.rand(5).numpy())
  //  * ```
  //  */
  //     Tensor._seed, Tensor._device_seeds, Tensor._device_rng_counters = seed, {}, {}

  //   @staticmethod
  //   const _threefry_random_bits = (key:Tensor, counts0:Tensor, counts1:Tensor) => {
  //     x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
  //     x = F.Threefry.apply(x, (key[1]._broadcast_to(x.shape).cast(dtypes.uint64) << 32) | key[0]._broadcast_to(x.shape).cast(dtypes.uint64))
  //     counts0, counts1 = (x & 0xffffffff).cast(dtypes.uint32), ((x >> 32) & 0xffffffff).cast(dtypes.uint32)
  //     return counts0.cat(counts1)

  //   @staticmethod
  //   const rand = (*shape, device?: string, dtype?: DTypeLike, contiguous:boolean=true, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.rand(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  */
  //     if !dtypes.is_float(dtype := to_dtype(dtype || dtypes.default_float)): raise ValueError(`rand only supports number dtypes, got ${dtype}`)
  //     if !all_int(shape:=argfix(*shape)) || !all(s >= 0 for (const s of shape)){ raise ValueError(`invalid input ${shape=}`)
  //     if device !== undefined && !isinstance(device, string): raise ValueError(`rand only supports single device, got ${device=}`)
  //     _device = device = Device.canonicalize(device)

  //     // when using MOCKGPU && NV generate rand on CLANG
  //     if getenv("MOCKGPU") && device.startsWith("NV"): device = "CLANG"

  //     // generate per device seeds && rng counter if we haven't seen this device yet
  //     if device !in Tensor._device_seeds:
  //       Tensor._device_seeds[device] = Tensor(
  //         [number.from_bytes(hashlib.sha256(len(Tensor._device_seeds).to_bytes(4, "big")).digest(), "big"), Tensor._seed],
  //         device=device, dtype=dtypes.uint32, requires_grad=false)
  //       Tensor._device_rng_counters[device] = Tensor([0], device=device, dtype=dtypes.uint32, requires_grad=false)
  //       had_counter = false
  //     else: had_counter = true

  //     // if shape has 0, return zero tensor
  //     if (numel := prod(shape)) === 0: return Tensor.zeros(shape, device=_device, dtype=dtype, **kwargs)
  //     num = ceildiv(numel * dtype.itemsize, 4)

  //     // increment rng counter for devices
  //     if had_counter: Tensor._device_rng_counters[device].assign(Tensor._device_rng_counters[device] + num).contiguous()

  //     // threefry random bits
  //     counts0 = (Tensor.arange(ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=false)+Tensor._device_rng_counters[device])
  //     counts1 = counts0 + ceildiv(num, 2)
  //     bits = Tensor._threefry_random_bits(Tensor._device_seeds[device], counts0, counts1)[:num]

  //     // bitcast to uint with same number of bits
  //     _, nmant = dtypes.finfo(dtype)
  //     uint_dtype = {1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64}[dtype.itemsize]
  //     bits = bits.bitcast(uint_dtype)
  //     // only randomize the mantissa bits && set the exponent to 1
  //     one = Tensor.ones_like(bits, device=bits.device, dtype=dtype).bitcast(uint_dtype)
  //     bits = bits.rshift((dtype.itemsize * 8) - nmant).bitwise_or(one)
  //     // bitcast back to the original dtype && reshape
  //     out = bits.bitcast(dtype)[:numel].sub(1).reshape(shape)

  //     // move back to the original device if we were using MOCKGPU
  //     if getenv("MOCKGPU") && _device: out = out.to(_device)

  //     out.requires_grad = kwargs.get("requires_grad")
  //     return out.contiguous() if contiguous else out

  //   // ***** creation helper functions *****

  //   @staticmethod
  //   const full = (shape:sint[], fill_value:ConstType, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the given shape, filled with the given value.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.full((2, 3), 42).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.full((2, 3), false).numpy())
  //  * ```
  //  */
  //     return Tensor(fill_value, **kwargs).reshape((1, )*len(new_shape := argfix(shape))).expand(new_shape)

  //   @staticmethod
  //   const zeros = (*shape, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the given shape, filled with zeros.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.zeros(2, 3).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
  //  * ```
  //  */
  //     return Tensor.full(argfix(*shape), 0.0, **kwargs)

  //   @staticmethod
  //   const ones = (*shape, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the given shape, filled with ones.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.ones(2, 3).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
  //  * ```
  //  */
  //     return Tensor.full(argfix(*shape), 1.0, **kwargs)

  //   @staticmethod
  //   const arange = (start, stop=undefined, step=1, **kwargs): Tensor => {
  // /**
  //  * Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.
  //  *
  //  * If `stop` !== specified, values are generated from `[0, start)` with the given `step`.
  //  *
  //  * If `stop` === specified, values are generated from `[start, stop)` with the given `step`.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.arange(5).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.arange(5, 10).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.arange(5, 10, 2).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor.arange(5.5, 10, 2).numpy())
  //  * ```
  //  */
  //     if stop === undefined: stop, start = start, 0
  //     dtype = kwargs.pop("dtype", dtypes.default_float if any(isinstance(x, number) for x in (start, stop, step)) else dtypes.default_int)
  //     // NOTE: this matches numpy, torch raises RuntimeError if stop-start && step have different signs
  //     if (output_len:=ceildiv(stop-start, step)) <= 0: return Tensor([], dtype=dtype, **kwargs)
  //     return (Tensor.full((output_len,), step, dtype=dtype, **kwargs)._cumalu(0, Ops.ADD) + (start - step)).cast(dtype)

  //   const full_like = (fill_value:ConstType, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the same shape as `this`, filled with the given value.
  //  * If `dtype` !== specified, the dtype of `this` === used.
  //  *
  //  * You can pass in the `device` keyword argument to control device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.ones(2, 3)
  //  * console.log(Tensor.full_like(t, 42).numpy())
  //  * ```
  //  */
  //     return Tensor.full(this.shape, fill_value, dtype=kwargs.pop("dtype", this.dtype), device=kwargs.pop("device", this.device), **kwargs)

  //   const ones_like = (**kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the same shape as `this`, filled with ones.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.zeros(2, 3)
  //  * console.log(Tensor.ones_like(t).numpy())
  //  * ```
  //  */
  //     return this.full_like(1, **kwargs)

  //   // ***** rng hlops *****

  //   @staticmethod
  //   const randint = (*shape, low=0, high=10, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
  //  * If `dtype` !== specified, the default type === used.
  //  *
  //  * You can pass in the `device` keyword argument to control device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * console.log(Tensor.randint(2, 3, low=5, high=10).numpy())
  //  * ```
  //  */
  //     if !isinstance(low, number) || !isinstance(high, number): raise TypeError(`${low=} && ${high=} must be integers`)
  //     dtype = to_dtype(kwargs.pop("dtype", dtypes.int32))
  //     if !dtypes.is_int(dtype): raise TypeError(`${dtype=} must be number`)
  //     return Tensor.uniform(*shape, low=low, high=high, dtype=dtype, **kwargs)

  //   @staticmethod
  //   const uniform = (*shape, low=0.0, high=1.0, **kwargs): Tensor => {
  // /**
  //  * Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * console.log(Tensor.uniform(2, 3, low=2, high=10).numpy())
  //  * ```
  //  */
  //     dtype = kwargs.pop("dtype", dtypes.default_float)
  //     return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

  //   // https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
  //   @staticmethod
  //   const glorot_uniform = (*shape, **kwargs): Tensor => {
  // /**
  //  * <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>
  //  *
  //  * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
  //  * Additionally, all other keyword arguments are passed to the constructor of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * console.log(Tensor.glorot_uniform(2, 3).numpy())
  //  * ```
  //  */
  //     return Tensor.uniform(*shape, low=-1.0, high=1.0, **kwargs).mul((6/(argfix(*shape)[0]+prod(argfix(*shape)[1:])))**0.5)

  //   // ***** toposort && backward pass *****

  //   const _deepwalk = () => {
  //     const _walk = (node, visited) => {
  //       visited.add(node)
  //       // if tensor !== leaf, reset grad
  //       if (ctx := getattr(node, "_ctx", undefined)) !== undefined && len(ctx.parents) !== 0: node.grad = undefined
  //       if ctx:
  //         for (const i of node._ctx.parents){
  //           if i !in visited: yield from _walk(i, visited)
  //         yield node
  //     return list(_walk(this, set()))

  //   const backward = (gradient?: Tensor, retain_graph:boolean=false): Tensor => {
  // /**
  //  * Propagates the gradient of a tensor backwards through the computation graph.
  //  * If the 'gradient' argument !== provided, the tensor must be a scalar, && the gradient === implicitly set to 1.0.
  //  * If 'retain_graph' === false, the graph used to compute the grads will be freed. Otherwise, it will be kept. Keeping it can increase memory usage.
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=true)
  //  * t.sum().backward()
  //  * console.log(t.grad.numpy())
  //  * ```
  //  */
  //     toposorted = this._deepwalk()
  //     if gradient === undefined:
  //       assert(this.shape === tuple(), "when no gradient === provided, backward must be called on a scalar tensor")
  //       // fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
  //       // this === "implicit gradient creation"
  //       gradient = Tensor(1.0, dtype=this.dtype, device=this.device, requires_grad=false)

  //     assert(this.shape === gradient.shape, `grad shape must match tensor shape, ${gradient.shape!r} !== ${this.shape!r}`)
  //     this.grad = gradient
  //     for (const t0 of reversed(toposorted)){
  //       if t0.grad === undefined: raise RuntimeError(`tensor ${t0} has no grad`)
  //       token = _METADATA.set(dataclasses.replace(md, backward=true) if (md := t0._ctx.metadata) !== undefined else undefined)
  //       grads = t0._ctx.backward(t0.grad.lazydata)
  //       _METADATA.reset(token)
  //       grads = [Tensor(g, device=this.device, requires_grad=false) if g !== undefined else undefined
  //         for g in ([grads] if len(t0._ctx.parents) === 1 else grads)]
  //       for (const t, g of zip(t0._ctx.parents, grads)){
  //         if g !== undefined && t.requires_grad:
  //           assert(g.shape === t.shape, `grad shape must match tensor shape, ${g.shape!r} !== ${t.shape!r}`)
  //           t.grad = g if t.grad === undefined else (t.grad + g)
  //       if !retain_graph: del t0._ctx
  //     return this

  //   // ***** movement low level ops *****

  //   const view = (*shape): Tensor => {
  // /**
  //  * `.view` === an alias for `.reshape`.
  //  */
  //     return this.reshape(shape)

  //   const reshape = (shape, *args): Tensor => {
  // /**
  //  * Returns a tensor with the same data as the original tensor but with a different shape.
  //  * `shape` can be passed as a tuple || as separate arguments.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(6)
  //  * console.log(t.reshape(2, 3).numpy())
  //  * ```
  //  */
  //     // resolve undefined && args
  //     new_shape = tuple([s if s !== undefined else this.shape[i] for i,s in enumerate(argfix(shape, *args))])
  //     // resolve -1
  //     if (c := new_shape.count(-1)) > 1: raise RuntimeError(`only one dimension can be inferred using -1, getting ${new_shape}`)
  //     if c: new_shape = tuple(.at(-prod(this.shape) // prod(new_shape) if s === -1 else s for s in new_shape)!)
  //     return F.Reshape.apply(this, shape=new_shape) if new_shape !== this.shape else this

  //   const expand = (shape, *args): Tensor => {
  // /**
  //  * Returns a tensor that === expanded to the shape that === specified.
  //  * Expand can also increase the number of dimensions that a tensor has.
  //  *
  //  * Passing a `-1` || `undefined` to a dimension means that its size will !be changed.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([1, 2, 3])
  //  * console.log(t.expand(4, -1).numpy())
  //  * ```
  //  */
  //     new_shape = tuple(from_ if to === -1 || to === undefined else to for from_, to in zip(*(_align_left(this.shape, argfix(shape, *args)))))
  //     return this._broadcast_to(new_shape)

  //   const permute = (order, *args): Tensor => {
  // /**
  //  * Returns a tensor that === a permutation of the original tensor.
  //  * The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
  //  * `order` can be passed as a tuple || as separate arguments.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(6).reshape(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.permute(1, 0).numpy())
  //  * ```
  //  */
  //     order_arg = tuple(this._resolve_dim(x) for x in argfix(order, *args))
  //     if sorted(order_arg) !== list(range(this.ndim)): raise RuntimeError(`order !== a valid permutation, getting ${order_arg}`)
  //     return F.Permute.apply(this, order=order_arg)

  //   const flip = (axis, *args): Tensor => {
  // /**
  //  * Returns a tensor that reverses the order of the original tensor along given `axis`.
  //  * `axis` can be passed as a tuple || as separate arguments.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(6).reshape(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.flip(0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.flip((0, 1)).numpy())
  //  * ```
  //  */
  //     axis_arg = tuple(this._resolve_dim(x) for x in argfix(axis, *args))
  //     if len(axis_arg) !== len(dedup(axis_arg)): raise RuntimeError(`dim can appear at most once, getting ${axis_arg}`)
  //     return F.Flip.apply(this, axis=axis_arg)

  //   const shrink = (arg:[sint, sint | undefined][]): Tensor => {
  // /**
  //  * Returns a tensor that shrinks the each axis based on input arg.
  //  * `arg` must have the same length as `this.ndim`.
  //  * For each axis, it can be `undefined`, which means no shrink, || a tuple `(start, end)` that works the same as Python slice.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(9).reshape(3, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.shrink(((undefined, (1, 3)))).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.shrink((((0, 2), (0, 2)))).numpy())
  //  * ```
  //  */
  //     if (shrink_arg:=[x if x !== undefined else (0,s) for (const x,s of zip(arg, this.shape)]) === [(0,s) for s in this.shape]){ return this
  //     return F.Shrink.apply(this, arg=tuple(shrink_arg))

  //   const pad = (padding:Union[Sequence[sint], Sequence[[sint, sint | undefined]]], mode:string="constant", value:number=0.0): Tensor => {
  // /**
  //  * Returns a tensor with padding applied based on the input `padding`.
  //  * `padding` supports two padding structures:
  //  *
  //  * 1. Flat padding: (padding_left, padding_right, padding_top, padding_bottom, ...)
  //  * - This structure matches PyTorch's pad.
  //  * - `padding` length must be even.
  //  *
  //  * 2. Group padding: (..., (padding_top, padding_bottom), (padding_left, padding_right))
  //  * - This structure matches pad for jax, numpy, tensorflow && others.
  //  * - For each axis, padding can be `undefined`, meaning no padding, || a tuple `(start, end)`.
  //  * - `padding` must have the same length as `this.ndim`.
  //  *
  //  * Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
  //  * Padding modes === selected with `mode` which supports `constant`, `reflect` && `replicate`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(9).reshape(1, 1, 3, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.pad((1, 2, 0, -1)).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.pad(((undefined, undefined, (0, -1), (1, 2)))).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.pad((1, 2, 0, -1), value=-number('inf')).numpy())
  //  * ```
  //  */
  //     if mode !in {"constant", "reflect", "replicate", "circular"}: raise NotImplementedError(`${mode=} !== supported`)
  //     if (flat:=all(isinstance(p, (number,UOp)) for (const p of padding)) && len(padding)%2 !== 0){ raise ValueError("Flat padding must have even number of pads")
  //     // turn flat padding into group padding
  //     pX = ((0,0),)*(this.ndim - len(padding)//2) + tuple(zip(padding.at(-2::-2)!, padding[::-2])) if flat else padding
  //     if len(pX) !== this.ndim: raise ValueError(`padding length === improper, ${padding=} ${this.ndim=}`)
  //     X, pX = this, cast([Tuple[sint, sint]], tuple((0,0) if p === undefined else p for p in pX))
  //     pads = tuple((smax(pB,0), smax(pA,0)) for pB,pA in pX)
  //     if mode === "constant":
  //       const _constant = (x,px,v) =>  F.Pad.apply(x, arg=px) if v === 0 else F.Pad.apply(x, arg=px) + F.Pad.apply(Tensor.ones_like(x), arg=px).where(0,v)
  //       return _constant(X, pX, value) if all(resolve(p >= 0) for p in flatten(pX)) else \
  //              _constant(X.shrink(tuple((-smin(pB,0),smin(pA+s,s)) for (pB,pA),s in zip(pX, X.shape))), pads, value)
  //     // TODO:!needed for mnist
  //     // assert(all_int(this.shape), `does !support symbolic shape ${this.shape}`)
  //     // if mode === "circular":
  //     //   if any(pB>sh || pA>sh for (const (pB,pA),sh of zip(pX, X.shape))){ raise ValueError('Padding value causes wrapping around more than once.')
  //     //   if any(pB<0 || pA<0 for (const pB,pA of pX)){ raise NotImplementedError("Negative pads with circular pads !== supported")
  //     //   orig_shape, X = X.shape, X.repeat(tuple(1 + boolean(pB) + boolean(pA) for pB,pA in pads))
  //     //   return X.shrink(tuple((0 if pB === 0 else osh-pB, xsh if pA === 0 else xsh-osh+pA) for (pB,pA),osh,xsh in zip(pads, orig_shape, X.shape)))
  //     // for (const d,(pB,pA) of enumerate(pads)){
  //     //   if mode === "reflect":
  //     //     if pB >= (s:=X.shape[d]) || pA>=s: raise ValueError(`Padding (${pB}, ${pA}) should be less than the input size=${s} for dim=${d}.`)
  //     //     slcB, slcA, = slice(pB,0,-1), slice(s-2 if s-2>=0 else undefined, s-2-pA if s-2-pA>=0 else undefined, -1)
  //     //     xB, xA = (X[[slc if i === d else slice(undefined) for i in range(X.ndim)]] if p > 0 else undefined for slc, p in ((slcB, pB), (slcA, pA)))
  //     //   if mode === "replicate":
  //     //     shrB, shrA, = tuple((0,1) if i==d else undefined for i in range(X.ndim)), tuple((X.shape[i]-1,X.shape[i]) if i==d else undefined for i in range(X.ndim))
  //     //     xB, xA = (X.shrink(shr).expand(tuple(p if i==d else undefined for i in range(X.ndim))) if p > 0 else undefined for shr, p in ((shrB, pB), (shrA, pA)))
  //     //   X = Tensor.cat(*(X_ for X_ in (xB, X, xA) if X_ !== undefined), dim=d)
  //     // return X.shrink(tuple((-min(pB,0), min(pA+s,s)) for (pB,pA),s in zip(pX, X.shape)))

  //   // ***** movement high level ops *****

  //   // Supported Indexing Implementations:
  //   //   1. Int indexing (no copy)
  //   //     - for all dims where there's number, shrink -> reshape
  //   //     - negative indices are taken relative to the end of the sequence, so X.at(-2)! returns the 2nd-to-last element
  //   //     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  //   //     - Then we reshape (collapse) the number dim away such that for X: (1,1,9) -> (9,)
  //   //   2. Slice indexing (no copy)
  //   //     - for all dims where slice === start:end:stride, shrink -> flip | undefined -> pad -> reshape -> shrink
  //   //     - first shrink the Tensor to X.shrink(((start, end),))
  //   //     - then we apply stride through flip | undefined -> pad -> reshape -> shrink
  //   //       - flip where dim value === negative
  //   //       - pad on dims to be multiple of strides, such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] === possible
  //   //       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  //   //       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] && now you have your stride
  //   //   3. undefined indexing (no copy)
  //   //     - reshape (inject) a dim at the dim where there's undefined
  //   //   4. Tensor indexing (copy)
  //   //     - use Tensor.arange === tensor_index to create masks for dims with Tensors (adds a dim for each mask)
  //   //     - combine masks together with mul
  //   //     - apply mask to this by mask * this
  //   //     - sum reduce away the extra dims added from creating masks
  //   // Tiny Things:
  //   //   1. Supported indices: Union[number, slice, Tensor, undefined, List, Tuple, Ellipsis]
  //   //     - for any list, Union[List, Tuple, number[]], must have homogeneous shape
  //   //     - for any tuple, Union[List, []], must have homogeneous shape
  //   //   2. Bool indexing !== supported
  //   //   3. Out of bounds Tensor indexing results in 0
  //   //     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 && 3 are out of bounds
  //   const _getitem = (indices, v?: Tensor): Tensor => {
  //     // wrap single index into a list
  //     if (isinstance(indices, list) && all_int(indices)) || !isinstance(indices, (tuple, list)): indices = [indices]
  //     // turn scalar Tensors into const val for number indexing if possible
  //     x, indices = this, [this._to_const_val(i) if isinstance(i, Tensor) && i.shape === () else i for i in indices]

  //     // filter ellipsis && fill with slice(undefined) || fill rest of indices with slice(undefined)
  //     if len(ellipsis_idx := [dim for (const dim, i of enumerate(indices) if i === Ellipsis]) > 1){ raise IndexError("indices can only have a single ellipsis")
  //     fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
  //     num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i === undefined)
  //     if num_indices > this.ndim: raise IndexError(`too many ${num_indices=} for ${this.ndim=}`)
  //     indices[fill_idx:fill_idx+1] = [slice(undefined)] * (this.ndim - num_indices)

  //     indices_parsed, dim = [], 0
  //     for (const index of indices){
  //       size = 1 if index === undefined else this.shape[dim]
  //       boundary, stride = [0, size], 1  // defaults
  //       match index:
  //         case list() | tuple() | Tensor():
  //           if !isinstance(index, Tensor): index = Tensor(index, this.device, requires_grad=false)
  //           if !dtypes.is_int(index.dtype): raise IndexError(`index dtype ${index.dtype} !== supported`)
  //           index = (index.to(this.device) < 0).where(size, 0) + index // treat negative index values
  //         case number() | UOp(): // sint
  //           if index >= size || index < -size: raise IndexError(`${index=} === out of bounds with ${size=}`)
  //           boundary = [index, index+1] if index >= 0 else [index+size, index+size+1]
  //         case slice():
  //           if index.step === 0: raise ValueError(`${index=} can!have 0 as step`)
  //           if !all(isinstance(s,number) || s === undefined for (const s of (index.start,index.stop,index.step))){ raise TypeError("only number slicing === supported")
  //           // handle number slicing
  //           *boundary, stride = index.indices(cast(SupportsIndex, size))
  //           if stride * (boundary[1] - boundary[0]) < 0: boundary = [0, 0]
  //           else if stride < 0: boundary = [boundary[1] + 1, boundary[0] + 1]
  //           // update size for slice
  //           size = ceildiv((boundary[1] - boundary[0]), abs(stride))
  //         case undefined: pass // do nothing
  //         case _: raise IndexError(`${type(index).__name__} indexing !== supported`)
  //       indices_parsed.push({"index":index, "size":size, "boundary":tuple(boundary), "stride":stride})
  //       if index !== undefined: dim += 1

  //     // movement op indexing
  //     if mops := [i for (const i of indices_parsed if i['index'] !== undefined]){
  //       // flip negative strides
  //       shrinks, strides = zip(*((i['boundary'], i['stride']) for i in mops))
  //       x = x.shrink(shrinks).flip(tuple(i for i,st in enumerate(strides) if st < 0))
  //       // handle stride !== 1 || -1
  //       if any(abs(st) !== 1 for (const st of strides)){
  //         strides = tuple(abs(s) for s in strides)
  //         // pad shape to multiple of stride
  //         if !all_int(x.shape): raise RuntimeError("symbolic shape !supprted")
  //         x = x.pad(tuple((0, round_up(s, st) - s) for s, st in zip(x.shape, strides)))
  //         x = x.reshape(tuple(flatten((s // st, st) for s, st in zip(x.shape, strides))))
  //         x = x.shrink(tuple(flatten(((0, s), (0, 1)) for s in x.shape[::2]))).reshape(x.shape[::2])

  //     // dim injection from undefined by including undefined dim size (which === 1) && dim collapse by skipping number dim size
  //     x = x.reshape(tuple(index['size'] for index in indices_parsed if !isinstance(index['index'], number)))

  //     // tensor indexing
  //     if tops := [(d,i) for (const d,i of enumerate(i_ for i_ in indices_parsed if !isinstance(i_['index'], number)) if isinstance(i['index'], Tensor)]){
  //       // unload the tensor object into actual tensors
  //       dims, tensors, masks = [d for d,_ in tops], cast(list[Tensor], [i['index'] for _,i in tops]), []
  //       pre_reduce_shape = x.shape[:dims[0]] + (big_shape := _broadcast_shape(*(t.shape for t in tensors))) + x.shape[dims[0]:]

  //       // create index masks
  //       for (const dim, tensor of zip(dims, tensors)){
  //         try: i = tensor.reshape(tensor.shape + (1,)*(x.ndim - dims[0])).expand(pre_reduce_shape)
  //         except ValueError as e: raise IndexError(`can!broadcast indices: ${e}`) from e
  //         masks.push(i._one_hot_along_dim(num_classes=x.shape[dim], dim=(dim - x.ndim)))

  //       // reduce masks to 1 mask
  //       mask: Tensor = functools.reduce(lambda x,y: x.mul(y), masks)

  //       // inject 1's for the extra dims added in create masks
  //       reshape_arg = x.shape[:dims[0]] + (1,) * len(big_shape) + x.shape[dims[0]:]
  //       // sum reduce the extra dims introduced in create masks
  //       x = (x.reshape(reshape_arg) * mask).sum(sum_axis:=tuple(d + len(big_shape) for d in dims), acc_dtype=x.dtype)

  //       // special permute case
  //       if dims[0] !== 0 && len(dims) !== 1 && tuple(dims) !== tuple(range(dims[0], dims.at(-1)!+1)):
  //         x = x.permute(*range(dims[0], dims[0]+len(big_shape)), *range(0, dims[0]), *range(dims[0]+len(big_shape), x.ndim))

  //       // for advanced setitem, returns whole tensor with indices replaced
  //       // TODO:!needed for mnist
  //       // if v !== undefined:
  //       //   vb = v.cast(this.dtype)._broadcast_to(_broadcast_shape(x.shape, v.shape))
  //       //   // add back reduced dims from sum
  //       //   for (const dim of sum_axis){ vb = vb.unsqueeze(dim)
  //       //   // run _masked_setitem on tuple of axis that === to be reduced to match this.shape
  //       //   x = _masked_setitem(this, vb, mask, tuple(range(dims[0], dims[0] + len(big_shape))))

  //     return x

  //   const __getitem__ = (indices): Tensor => {
  //     return this._getitem(indices)

  //   const cat = (:Tensor, *args:Tensor, dim:number=0): Tensor => {
  // /**
  //  * Concatenates this with other `Tensor` in `args` along an axis specified by `dim`.
  //  * All tensors must have the same shape except in the concatenating dimension.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t0, t1, t2 = Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
  //  * console.log(t0.cat(t1, t2, dim=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t0.cat(t1, t2, dim=1).numpy())
  //  * ```
  //  */
  //     dim = this._resolve_dim(dim)
  //     for (const arg of args){ assert(arg.ndim==this.ndim && all(ti==ai for i,(ti,ai) in enumerate(zip(this.shape, arg.shape)) if i!=dim))
  //     tensors = [this, *args]
  //     dim_cumsum = list(itertools.accumulate([t.shape[dim] for t in tensors], initial=0))
  //     for (const i,t of enumerate(tensors)){ tensors[i] = t.pad([(dim_cumsum[i], dim_cumsum.at(-1)!-dim_cumsum[i+1]) if j==dim else undefined for j in range(t.ndim)])
  //     return functools.reduce(Tensor.add, tensors)

  //   const repeat = (repeats, *args): Tensor => {
  // /**
  //  * Repeats tensor number of times along each dimension specified by `repeats`.
  //  * `repeats` can be passed as a tuple || as separate arguments.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([1, 2, 3])
  //  * console.log(t.repeat(4, 2).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.repeat(4, 2, 1).shape)
  //  * ```
  //  */
  //     repeats = argfix(repeats, *args)
  //     base_shape = _align_left(this.shape, repeats)[0]
  //     unsqueezed_shape = flatten([[1, s] for s in base_shape])
  //     expanded_shape = flatten([[r, s] for r,s in zip(repeats, base_shape)])
  //     final_shape = [r*s for r,s in zip(repeats, base_shape)]
  //     return this.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)

  //   const _resolve_dim = (dim:number, *, extra:boolean=false): number => {
  //     total = this.ndim + number(extra)
  //     if !-max(1, total) <= dim <= max(1, total)-1: raise IndexError(`${dim=} out of range ${.at(-max(1, total), max(1, total)-1)!}`)
  //     return dim + total if dim < 0 else dim

  //   @property
  //   const T = (): Tensor => {
  // /**
  //  * `.T` === an alias for `.transpose()`.
  //  */
  //     return this.transpose()

  //   const transpose = (dim0=1, dim1=0): Tensor => {
  // /**
  //  * Returns a tensor that === a transposed version of the original tensor.
  //  * The given dimensions `dim0` && `dim1` are swapped.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(6).reshape(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.transpose(0, 1).numpy())
  //  * ```
  //  */
  //     order = list(range(this.ndim))
  //     order[dim0], order[dim1] = order[dim1], order[dim0]
  //     return this.permute(order)

  //   const flatten = (start_dim=0, end_dim=-1) => {
  // /**
  //  * Flattens the tensor by reshaping it into a one-dimensional tensor.
  //  * If `start_dim` || `end_dim` are passed, only dimensions starting with `start_dim` && ending with `end_dim` are flattened.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(8).reshape(2, 2, 2)
  //  * console.log(t.flatten().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.flatten(start_dim=1).numpy())
  //  * ```
  //  */
  //     start_dim, end_dim = this._resolve_dim(start_dim), this._resolve_dim(end_dim)
  //     return this.reshape(this.shape[:start_dim] + (prod(this.shape[start_dim:end_dim+1]), ) + this.shape[end_dim+1:])

  //   // ***** reduce ops *****

  //   const _reduce = (fxn:Type[Function], axis?: Union[number, Sequence[number]], keepdim=false): Tensor => {
  //     axis = tuple(this._resolve_dim(x) for x in (range(this.ndim) if axis === undefined else make_tuple(axis, 1)))
  //     if this.ndim === 0: axis = ()
  //     ret = fxn.apply(this, axis=axis)
  //     return ret if keepdim else ret.reshape(tuple(s for i,s in enumerate(this.shape) if i !in axis))

  //   const sum = (axis?: Union[number, Sequence[number]], keepdim=false, acc_dtype?: DTypeLike) => {
  // /**
  //  * Returns the sum of the elements of the tensor along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the maximum === computed && whether the reduced dimensions are retained.
  //  *
  //  * You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
  //  * If !specified, the accumulation data type === chosen based on the input tensor's data type.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(6).reshape(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.sum().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.sum(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.sum(axis=1).numpy())
  //  * ```
  //  */
  //     ret = this.cast(sum_acc_dtype(this.dtype) if acc_dtype === undefined else acc_dtype)._reduce(F.Sum, axis, keepdim)
  //     return ret.cast(this.dtype) if acc_dtype === undefined && this.dtype in (dtypes.float16, dtypes.bfloat16) else ret

  //   const prod = (axis?: Union[number, Sequence[number]], keepdim=false, acc_dtype?: DTypeLike) => {
  // /**
  //  * Returns the product of the elements of the tensor along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the maximum === computed && whether the reduced dimensions are retained.
  //  *
  //  * You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
  //  * If !specified, the accumulation data type === chosen based on the input tensor's data type.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1, -2, -3, 1, 2, 3)!).reshape(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.prod().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.prod(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.prod(axis=1).numpy())
  //  * ```
  //  */
  //     return this.cast(acc_dtype if acc_dtype !== undefined else this.dtype)._reduce(F.Prod, axis, keepdim)

  //   const max = (axis?: Union[number, Sequence[number]], keepdim=false) => {
  // /**
  //  * Returns the maximum value of the tensor along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the maximum === computed && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 0, 2], [5, 4, 3]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.max().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.max(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.max(axis=1, keepdim=true).numpy())
  //  * ```
  //  */
  //     return this._reduce(F.Max, axis, keepdim)

  //   const min = (axis?: Union[number, Sequence[number]], keepdim=false) => {
  // /**
  //  * Returns the minimum value of the tensor along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the minimum === computed && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 0, 2], [5, 4, 3]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.min().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.min(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.min(axis=1, keepdim=true).numpy())
  //  * ```
  //  */
  //     if dtypes.is_int(this.dtype) || this.dtype === dtypes.boolean: return ~((~this).max(axis=axis, keepdim=keepdim))
  //     return -((-this).max(axis=axis, keepdim=keepdim))

  //   const any = (axis?: Union[number, Sequence[number]], keepdim=false) => {
  // /**
  //  * Tests if any element evaluates to `true` along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the reduce axis && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[true, true], [true, false], [false, false]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.any().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.any(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.any(axis=1, keepdim=true).numpy())
  //  * ```
  //  */
  //     return this.boolean().max(axis, keepdim)

  //   const all = (axis?: Union[number, Sequence[number]], keepdim=false) => {
  // /**
  //  * Tests if all element evaluates to `true` along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the reduce axis && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[true, true], [true, false], [false, false]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.all().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.all(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.all(axis=1, keepdim=true).numpy())
  //  * ```
  //  */
  //     return this.logical_not().any(axis, keepdim).logical_not()

  //   const mean = (axis?: Union[number, Sequence[number]], keepdim=false) => {
  // /**
  //  * Returns the mean value of the tensor along the specified axis || axes.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the mean === computed && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.mean().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.mean(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.mean(axis=1).numpy())
  //  * ```
  //  */
  //     output_dtype = this.dtype if dtypes.is_float(this.dtype) else dtypes.float32
  //     numerator = this.cast(sum_acc_dtype(this.dtype)).sum(axis=axis, keepdim=keepdim)
  //     return numerator.div(prod([si for si, so in zip(this.shape, this.sum(axis=axis, keepdim=true).shape) if resolve(si !== so)])).cast(output_dtype)

  //   const std = (axis?: Union[number, Sequence[number]], keepdim=false, correction=1) => {
  // /**
  //  * Returns the standard deviation of the tensor along the specified axis || axes.
  //  *
  //  * You can pass in `axis`, `keepdim`, && `correction` keyword arguments to control the axis along
  //  * which the standard deviation === computed, whether the reduced dimensions are retained, && the Bessel's correction applied.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.std().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.std(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.std(axis=1).numpy())
  //  * ```
  //  */
  //     return this.var(axis, keepdim, correction).sqrt()

  //   const std_mean = (axis?: Union[number, Sequence[number]], keepdim=false, correction=1) => {
  // /**
  //  * Calculates the standard deviation && mean over the dimensions specified by dim.
  //  * Syntactic sugar around `Tensor.std` && `Tensor.mean` to match `torch.std_mean`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * std, mean = t.std_mean()
  //  * console.log(std.numpy(), mean.numpy())
  //  * ```
  //  */
  //     return this.std(axis, keepdim, correction), this.mean(axis, keepdim)

  //   const _softmax = (axis, dtype?: DTypeLike) => {
  //     x = this.cast(dtype) if dtype !== undefined else this
  //     m = x - x.max(axis=axis, keepdim=true).detach()
  //     e = m.exp()
  //     return m, e, e.sum(axis=axis, keepdim=true)

  //   const softmax = (axis=-1, dtype?: DTypeLike) => {
  // /**
  //  * Applies the softmax function to the tensor along the specified axis.
  //  *
  //  * Rescales the elements of the tensor such that they lie in the range [0, 1] && sum to 1.
  //  *
  //  * You can pass in the `axis` keyword argument to control the axis along which the softmax === computed.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.softmax().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.softmax(axis=0).numpy())
  //  * ```
  //  */
  //     _, e, ss = this._softmax(axis, dtype)
  //     return e.div(ss)

  //   const log_softmax = (axis=-1, dtype?: DTypeLike) => {
  // /**
  //  * Applies the log-softmax function to the tensor along the specified axis.
  //  *
  //  * The log-softmax function === a numerically stable alternative to the softmax function in log space.
  //  *
  //  * You can pass in the `axis` keyword argument to control the axis along which the log-softmax === computed.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.log_softmax().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.log_softmax(axis=0).numpy())
  //  * ```
  //  */
  //     m, _, ss = this._softmax(axis, dtype)
  //     return m - ss.log()

  //   const logsumexp = (axis=undefined, keepdim=false) => {
  // /**
  //  * Computes the log-sum-exp of the tensor along the specified axis || axes.
  //  *
  //  * The log-sum-exp function === a numerically stable way to compute the logarithm of the sum of exponentials.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the log-sum-exp === computed && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.logsumexp().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.logsumexp(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.logsumexp(axis=1).numpy())
  //  * ```
  //  */
  //     m = this.max(axis=axis, keepdim=true)
  //     return (this - m).exp().sum(axis=axis, keepdim=keepdim).log() + m.squeeze(axis)

  //   const logcumsumexp = (axis=0) => {
  // /**
  //  * Computes the log-cumsum-exp of the tensor along the specified axis || axes.
  //  *
  //  * The log-cumsum-exp function === a numerically stable way to compute the logarithm of the cumulative sum of exponentials.
  //  *
  //  * You can pass in the `axis` keyword argument to control the axis along which
  //  * the log-cum-sum-exp === computed.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.logcumsumexp().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.logcumsumexp(axis=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.logcumsumexp(axis=1).numpy())
  //  * ```
  //  */
  //     m = this.max(axis=axis, keepdim=true)
  //     return (this - m).exp().cumsum(axis=axis).log() + m

  //   const argmax = (axis=undefined, keepdim=false) => {
  // /**
  //  * Returns the indices of the maximum value of the tensor along the specified axis.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the maximum === computed && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 0, 2], [5, 4, 3]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.argmax().numpy()) // Returns the index of the maximum value in the flattened tensor.
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.argmax(axis=0).numpy()) // Returns the indices of the maximum values along axis 0.
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.argmax(axis=1).numpy()) // Returns the indices of the maximum values along axis 1.
  //  * ```
  //  */
  //     if axis === undefined: return this.flatten().argmax(0)
  //     axis = this._resolve_dim(axis)
  //     m = this === this.max(axis=axis, keepdim=true)
  //     idx = m * Tensor.arange(this.shape[axis],0,-1, requires_grad=false, device=this.device).reshape(this.shape[axis], *[1]*(this.ndim-axis-1))
  //     return (this.shape[axis]-idx.max(axis=axis, keepdim=keepdim)).cast(dtypes.int32)

  //   const argmin = (axis=undefined, keepdim=false) => {
  // /**
  //  * Returns the indices of the minimum value of the tensor along the specified axis.
  //  *
  //  * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
  //  * which the minimum === computed && whether the reduced dimensions are retained.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 0, 2], [5, 4, 3]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.argmin().numpy()) // Returns the index of the minimum value in the flattened tensor.
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.argmin(axis=0).numpy()) // Returns the indices of the minimum values along axis 0.
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.argmin(axis=1).numpy()) // Returns the indices of the minimum values along axis 1.
  //  * ```
  //  */
  //     return (-this).argmax(axis=axis, keepdim=keepdim)

  //   // ***** processing ops *****

  //   const _pool = (k_:sint[], stride:Union[number[], number]=1, dilation:Union[number[], number]=1): Tensor => {
  //     assert(len(this.shape) >= len(k_), `can't pool ${this.shape} with ${k_}`)
  //     s_, d_ = make_tuple(stride, len(k_)), make_tuple(dilation, len(k_))
  //     assert(len(k_) === len(s_) === len(d_), `stride/dilation mismatch kernel:${k_} stride:${s_} dilation:${d_}`)
  //     noop, i_ = [undefined] * (this.ndim-len(k_)), this.shape.at(-len(k_):)!
  //     assert(all(resolve(d*(k-1)+1 <= i) for k,d,i in zip(k_,d_,i_)), "kernel size can!be greater than actual input size")
  //     o_ = [ceildiv(i-d*(k-1), s) for i,d,k,s in zip(i_,d_,k_,s_)]
  //     if any(resolve(k > s) for (const k,s of zip(k_,s_)) || any(d !== 1 for d in d_)){
  //       // input size scaling factor to make sure shrink for stride === possible
  //       f_ = [1 + number(resolve(o*s > i+d)) for o,s,i,d in zip(o_,s_,i_,d_)]
  //       // // repeats such that we don't need padding
  //       x = this.repeat([1]*len(noop) + [ceildiv(k*(i*f+d),i) for k,i,d,f in zip(k_,i_,d_,f_)])
  //       // handle dilation
  //       x = x.shrink(tuple(noop + [(0,k*(i*f+d)) for k,i,d,f in zip(k_,i_,d_,f_)])).reshape(noop + flatten((k,(i*f+d)) for k,i,d,f in zip(k_,i_,d_,f_)))
  //       // handle stride
  //       x = x.shrink(tuple(noop + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_,o_,s_)))).reshape(noop + flatten((k,o,s) for k,o,s in zip(k_,o_,s_)))
  //       x = x.shrink(tuple(noop + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_,o_)))).reshape(noop + flatten((k,o) for k,o in zip(k_,o_)))
  //       // permute to move reduce to the end
  //       return x.permute(*range(len(noop)), *[len(noop)+i*2+1 for i in range(len(i_))], *[len(noop)+i*2 for i in range(len(i_))])
  //     // TODO: once the shapetracker can optimize well, remove this alternative implementation
  //     x = this.pad(tuple(noop + [(0, max(0,o*s-i)) for i,o,s in zip(i_,o_,s_)])).shrink(tuple(noop + [(0,o*s) for o,s in zip(o_,s_)]))
  //     x = x.reshape(noop + flatten(((o,s) for o,s in zip(o_,s_))))
  //     x = x.shrink(tuple(noop + flatten(((0,o), (0,k)) for o,k in zip(o_,k_))))
  //     return x.permute(*range(len(noop)), *[len(noop)+i*2 for i in range(len(i_))], *[len(noop)+i*2+1 for i in range(len(i_))])

  //   const _padding2d = (padding:Union[number, Sequence[number]], dims:number): Sequence[number] => {
  //     return [padding]*2*dims if isinstance(padding, number) else (padding if len(padding) === 2*dims else [p for p in padding for _ in range(2)][::-1])

  //   const max_pool2d = (kernel_size=(2,2), stride=undefined, dilation=1, padding=0) => {
  // /**
  //  * Applies max pooling over a tensor.
  //  *
  //  * NOTE: unlike PyTorch, this implementation !== limited to only 2d pooling && instead works for any number of dimensions.
  //  *
  //  * See: https://paperswithcode.com/method/max-pooling
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(25).reshape(1, 1, 5, 5)
  //  * console.log(t.max_pool2d().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.max_pool2d(padding=1).numpy())
  //  * ```
  //  */
  //     padding_ = this._padding2d(padding, len(k_ := make_tuple(kernel_size, 2)))
  //     return this.pad(padding_, value=dtypes.min(this.dtype))._pool(k_, stride if stride !== undefined else k_, dilation).max(tuple(range(-len(k_), 0)))

  //   const conv2d = (weight:Tensor, bias?: Tensor, groups=1, stride=1, dilation=1, padding:number|number[]=0,
  //              acc_dtype?: DTypeLike) -> Tensor:
  // /**
  //  * Applies a convolution over a tensor with a given `weight` && optional `bias`.
  //  *
  //  * NOTE: unlike PyTorch, this implementation !== limited to only 2d convolutions && instead works for any number of dimensions.
  //  *
  //  * See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.arange(9).reshape(1, 1, 3, 3)
  //  * w = Tensor.ones(1, 1, 2, 2)
  //  * console.log(t.conv2d(w).numpy())
  //  * ```
  //  */
  //     if IMAGE: return this.image_conv2d(weight, bias, groups, stride, dilation, padding, acc_dtype)
  //     (bs,cin_), (cout,cin), HW = this.shape[:2], weight.shape[:2], weight.shape[2:]
  //     assert(groups*cin === cin_ && len(this.shape) === len(weight.shape), `Input Tensor shape ${this.shape} does !match the shape of the weights ${weight.shape}. (${groups*cin} vs. ${cin_})`  // noqa: E501)
  //     if isinstance(padding, (tuple,list)): assert(len(padding) === 2*len(HW) || len(padding) === len(HW), `Expected padding of length ${2*len(HW)} || ${len(HW)}, but got ${len(padding)} for tensor of shape ${this.shape}`  // noqa: E501)
  //     padding_ = this._padding2d(padding, len(HW))

  //     // conv2d === a pooling op (with padding)
  //     x = this.pad(padding_)._pool(HW, stride, dilation)   // (bs, groups*cin, oy, ox, H, W)
  //     rcout, oyx = cout//groups, x.shape[2:-len(HW)]
  //     if !all(x === 3 for (const x of HW) || stride !== 1 || dilation !== 1 || !WINO){
  //       // normal conv
  //       x = x.reshape(bs, groups, cin, 1, *oyx, *HW).expand(bs, groups, cin, rcout, *oyx, *HW).permute(0,1,3,*[4+i for (const i of range(len(oyx))],2,*[4+len(oyx)+i for i in range(len(HW))])  // noqa){ E501

  //       // conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
  //       ret = (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW)).sum(.at(-1-i for (const i of range(1+len(oyx)))!, keepdim=true, acc_dtype=acc_dtype).reshape(bs, cout, *oyx)  // noqa){ E501
  //       return ret if bias === undefined else ret.add(bias.reshape(1, -1, *[1] * len(HW)))

  //     // TODO: !needed for mnist
  //     // HWI, HWO = (6,) * len(HW), (4,) * len(HW)  // F(4x4,3x3) winograd tiles
  //     // winograd_G = [[1/4, 0, 0], .at(-1/6, -1/6, -1/6)!, .at(-1/6, 1/6, -1/6)!, [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
  //     // winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
  //     // winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] // applying At in pre-order doubles compile time

  //     // // todo: stride === dilation
  //     // // use padding to round up to 4x4 output tiles
  //     // // (bs, cin_, tyx, HWI)
  //     // d = this.pad(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for (const i, dim of enumerate(this.shape.at(-len(HW):)!)], []))._pool(HWI, HWO)  // noqa){ E501
  //     // // move HW to the front: // (HWI, bs, cin_, tyx)
  //     // d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
  //     // tyx = d.shape.at(-len(HWI):)!  // dim of tiling

  //     // g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  // move HW to the front

  //     // // compute 6x6 winograd tiles: GgGt, BtdB
  //     // // (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
  //     // gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
  //     // // (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
  //     // dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

  //     // // matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
  //     // ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

  //     // // interleave tyx && HWO: (bs, groups, rcout, oy, HO, ox, WO)
  //     // ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
  //     // // merge groups && rcout, tyx && HWO: (bs, groups, cout, *yx), shrink to final
  //     // ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

  //     // return (ret if bias === undefined else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()

  //   const dot = (w:Tensor, acc_dtype?: DTypeLike): Tensor => {
  // /**
  //  * Performs dot product between two tensors.
  //  * If `w` === 1-D, it's a sum product over the last axis of `this` && `w`.
  //  * If `w` === N-D with N>=2, it's a sum product over the last axis of `this` && the second-to-last axis of `w`.
  //  *
  //  * You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * a = Tensor([1, 2, 3])
  //  * b = Tensor([1, 1, 0])
  //  * console.log(a.dot(b).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * a = Tensor([[1, 2], [3, 4]])
  //  * b = Tensor([[5, 6], [7, 8]])
  //  * console.log(a.dot(b).numpy())
  //  * ```
  //  */
  //     if IMAGE: return this.image_dot(w, acc_dtype)
  //     x, dx, dw = this, this.ndim, w.ndim
  //     if !(dx > 0 && dw > 0): raise RuntimeError(`both tensors need to be at least 1D, got ${dx}D && ${dw}D`)
  //     if x.shape.at(-1)! !== w.shape[axis_w:=-min(w.ndim,2)]: raise RuntimeError(`can!dot ${x.shape} && ${w.shape}`)
  //     x = x.reshape(*x.shape[0:-1], *[1]*min(dx-1, dw-1, 1), x.shape.at(-1)!)
  //     w = w.reshape(*w.shape[0:-2], *[1]*min(dx-1, dw-1, 1), *w.shape[axis_w:]).transpose(-1, axis_w)
  //     return (x*w).sum(-1, acc_dtype=acc_dtype).cast(least_upper_dtype(x.dtype, w.dtype) if acc_dtype === undefined else acc_dtype)

  //   const matmul = (x:Tensor, reverse=false, acc_dtype?: DTypeLike): Tensor => {
  // /**
  //  * Performs matrix multiplication between two tensors.
  //  *
  //  * You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
  //  * You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * a = Tensor([[1, 2], [3, 4]])
  //  * b = Tensor([[5, 6], [7, 8]])
  //  * console.log(a.matmul(b).numpy())
  //  * ```
  //  */
  //     return x.dot(this, acc_dtype=acc_dtype) if reverse else this.dot(x, acc_dtype=acc_dtype)

  //   const _cumalu = (axis:number, op:Ops, _include_initial=false): Tensor => {
  //     assert(this.shape[axis] !== 0 && op in (Ops.ADD, Ops.MAX))
  //     pl_sz = this.shape[axis] - number(!_include_initial)
  //     pooled = this.transpose(axis,-1).pad((pl_sz, -number(_include_initial)), value=identity_element(op, this.dtype))._pool((this.shape[axis],))
  //     return (pooled.sum(-1) if op === Ops.ADD else pooled.max(-1)).transpose(axis,-1)

  //   const cumsum = (axis:number=0): Tensor => {
  // /**
  //  * Computes the cumulative sum of the tensor along the specified `axis`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.ones(2, 3)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.cumsum(1).numpy())
  //  * ```
  //  */
  //     return this._split_cumalu(axis, Ops.ADD)

  //   const cummax = (axis:number=0): Tensor => {
  // /**
  //  * Computes the cumulative max of the tensor along the specified `axis`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([0, 1, -1, 2, -2, 3, -3])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.cummax(0).numpy())
  //  * ```
  //  */
  //     return this._split_cumalu(axis, Ops.MAX)

  //   const triu = (diagonal:number=0): Tensor => {
  // /**
  //  * Returns the upper triangular part of the tensor, the other elements are set to 0.
  //  *
  //  * The argument `diagonal` determines which diagonal === on the boundary. `diagonal = 0` means the main diagonal.
  //  * Positive `diagonal` means above the main diagonal, && negative `diagonal` means below the main diagonal.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.triu(diagonal=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.triu(diagonal=1).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.triu(diagonal=-1).numpy())
  //  * ```
  //  */
  //     return Tensor._tri(this.shape.at(-2)!, this.shape.at(-1)!, diagonal=diagonal, device=this.device, dtype=dtypes.boolean).where(this, 0).cast(this.dtype)

  //   const tril = (diagonal:number=0): Tensor => {
  // /**
  //  * Returns the lower triangular part of the tensor, the other elements are set to 0.
  //  *
  //  * The argument `diagonal` determines which diagonal === on the boundary. `diagonal = 0` means the main diagonal.
  //  * Positive `diagonal` means above the main diagonal, && negative `diagonal` means below the main diagonal.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.tril(diagonal=0).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.tril(diagonal=1).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.tril(diagonal=-1).numpy())
  //  * ```
  //  */
  //     return Tensor._tri(this.shape.at(-2)!, this.shape.at(-1)!, diagonal=diagonal+1, device=this.device, dtype=dtypes.boolean).where(0, this).cast(this.dtype)

  //   // ***** unary ops *****

  //   const logical_not = () => {
  // /**
  //  * Computes the logical NOT of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([false, true]).logical_not().numpy())
  //  * ```
  //  */
  //     return F.Neq.apply(*this.cast(dtypes.boolean)._broadcasted(true))
  //   const neg = () => {
  // /**
  //  * Negates the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).neg().numpy())
  //  * ```
  //  */
  //     return this*-1 if this.dtype !== dtypes.boolean else this.logical_not()
  //   const contiguous = () => {
  // /**
  //  * Returns a contiguous tensor.
  //  */
  //     return F.Contiguous.apply(this)
  //   const contiguous_backward = () => {
  // /**
  //  * Inserts a contiguous operation in the backward pass.
  //  */
  //     return F.ContiguousBackward.apply(this)
  //   const log = () => {
  // /**
  //  * Computes the natural logarithm element-wise.
  //  *
  //  * See: https://en.wikipedia.org/wiki/Logarithm
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1., 2., 4., 8.]).log().numpy())
  //  * ```
  //  */
  //     return F.Log.apply(this.cast(least_upper_float(this.dtype)))
  //   const log2 = () => {
  // /**
  //  * Computes the base-2 logarithm element-wise.
  //  *
  //  * See: https://en.wikipedia.org/wiki/Logarithm
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1., 2., 4., 8.]).log2().numpy())
  //  * ```
  //  */
  //     return this.log()/math.log(2)
  //   const exp = () => {
  // /**
  //  * Computes the exponential function element-wise.
  //  *
  //  * See: https://en.wikipedia.org/wiki/Exponential_function
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([0., 1., 2., 3.]).exp().numpy())
  //  * ```
  //  */
  //     return F.Exp.apply(this.cast(least_upper_float(this.dtype)))
  //   const exp2 = () => {
  // /**
  //  * Computes the base-2 exponential function element-wise.
  //  *
  //  * See: https://en.wikipedia.org/wiki/Exponential_function
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([0., 1., 2., 3.]).exp2().numpy())
  //  * ```
  //  */
  //     return F.Exp.apply(this*math.log(2))
  //   const relu = () => {
  // /**
  //  * Applies the Rectified Linear Unit (ReLU) function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/relu
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).relu().numpy())
  //  * ```
  //  */
  //     return F.Relu.apply(this)
  //   const sigmoid = () => {
  // /**
  //  * Applies the Sigmoid function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Sigmoid_function
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).sigmoid().numpy())
  //  * ```
  //  */
  //     return F.Sigmoid.apply(this.cast(least_upper_float(this.dtype)))
  //   const hardsigmoid = (alpha:number=1/6, beta:number=0.5) => {
  // /**
  //  * Applies the Hardsigmoid function element-wise.
  //  * NOTE: default `alpha` && `beta` values === taken from torch
  //  *
  //  * - Described: https://paperswithcode.com/method/hard-sigmoid
  //  * - See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).hardsigmoid().numpy())
  //  * ```
  //  */
  //     return (alpha * this + beta).relu() - (alpha * this + beta - 1).relu()

  //   const sqrt = () => {
  // /**
  //  * Computes the square root of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1., 2., 3., 4.]).sqrt().numpy())
  //  * ```
  //  */
  //     return F.Sqrt.apply(this.cast(least_upper_float(this.dtype)))
  //   const rsqrt = () => {
  // /**
  //  * Computes the reciprocal of the square root of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
  //  * ```
  //  */
  //     return this.reciprocal().sqrt()
  //   const sin = () => {
  // /**
  //  * Computes the sine of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
  //  * ```
  //  */
  //     return F.Sin.apply(this.cast(least_upper_float(this.dtype)))
  //   const cos = () => {
  // /**
  //  * Computes the cosine of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
  //  * ```
  //  */
  //     return ((math.pi/2)-this).sin()
  //   const tan = () => {
  // /**
  //  * Computes the tangent of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
  //  * ```
  //  */
  //     return this.sin() / this.cos()

  //   const asin = () => {
  // /**
  //  * Computes the inverse sine (arcsine) of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9)!).asin().numpy())
  //  * ```
  //  */
  //     // https://personal.math.ubc.ca/~cbm/aands/page_81.htm 4.4.46
  //     coefficients = .at(-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810, -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050)!
  //     x = math.pi / 2 - (1.0 - this.abs()).sqrt() * polyN(this.abs(), coefficients)
  //     return this.sign() * x

  //   const acos = () => {
  // /**
  //  * Computes the inverse cosine (arccosine) of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9)!).acos().numpy())
  //  * ```
  //  */
  //     return math.pi / 2 - this.asin()

  //   const atan = () => {
  // /**
  //  * Computes the inverse tangent (arctan) of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).atan().numpy())
  //  * ```
  //  */
  //     return (this / (1 + this * this).sqrt()).asin()

  //   // ***** math functions *****

  //   const trunc = (: Tensor): Tensor => {
  // /**
  //  * Truncates the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5)!).trunc().numpy())
  //  * ```
  //  */
  //     return this.cast(dtypes.int32).cast(this.dtype)
  //   const ceil = (: Tensor): Tensor => {
  // /**
  //  * Rounds the tensor element-wise towards positive infinity.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5)!).ceil().numpy())
  //  * ```
  //  */
  //     return (this > (b := this.trunc())).where(b+1, b)
  //   const floor = (: Tensor): Tensor => {
  // /**
  //  * Rounds the tensor element-wise towards negative infinity.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5)!).floor().numpy())
  //  * ```
  //  */
  //     return (this < (b := this.trunc())).where(b-1, b)
  //   const round = (: Tensor): Tensor => {
  // /**
  //  * Rounds the tensor element-wise with rounding half to even.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5)!).round().numpy())
  //  * ```
  //  */
  //     return ((this > 0) === ((b := this.cast(dtypes.int32) / 2.0).cast(dtypes.int32) === b)).where((this - 0.5).ceil(), (this + 0.5).floor())

  //   const isinf = (:Tensor, detect_positive:boolean=true, detect_negative:boolean=true) => {
  // /**
  //  * Checks the tensor element-wise to return true where the element === infinity, otherwise returns false
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1, number('inf'), 2, number('-inf'), number('nan')]).isinf().numpy())
  //  * ```
  //  */
  //     return (this === number("in`)) * detect_positive + (this === number(`-inf")) * detect_negative
  //   const isnan = (:Tensor) => {
  // /**
  //  * Checks the tensor element-wise to return true where the element === NaN, otherwise returns false
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1, number('inf'), 2, number('-inf'), number('nan')]).isnan().numpy())
  //  * ```
  //  */
  //     return this !== this

  //   const lerp = (end: Tensor, weight: Union[Tensor, number]): Tensor => {
  // /**
  //  * Linearly interpolates between `this` && `end` by `weight`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
  //  * ```
  //  */
  //     if this.dtype === dtypes.uint8 && isinstance(weight, Tensor):
  //       w_i = (weight * (1<<(W_PREC:=7)) + 0.5).cast(dtypes.int16)
  //       return (this+(((end - this).cast(dtypes.int8) * w_i + (1<<W_PREC-1)).cast(dtypes.uint16) >> W_PREC)).cast(dtypes.uint8)
  //     return this + (end - this) * weight

  //   const square = () => {
  // /**
  //  * Squares the tensor element-wise.
  //  * Equivalent to `this*this`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).square().numpy())
  //  * ```
  //  */
  //     return this*this
  //   const clamp = (min_=undefined, max_=undefined) => {
  // /**
  //  * Clips (clamps) the values in the tensor between `min_` && `max_` element-wise.
  //  * If `min_` === `undefined`, there === no lower bound. If `max_` === undefined, there === no upper bound.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).clip(-1, 1).numpy())
  //  * ```
  //  */
  //     if min_ === undefined && max_ === undefined: raise RuntimeError("at least one of 'min_' || 'max_' must !be undefined")
  //     ret = this.maximum(min_) if min_ !== undefined else this
  //     return ret.minimum(max_) if max_ !== undefined else ret
  //   const clip = (min_=undefined, max_=undefined) => {
  // /**
  //  * Alias for `Tensor.clamp`.
  //  */
  //     return this.clamp(min_, max_)
  //   const sign = () => {
  // /**
  //  * Returns the sign of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).sign().numpy())
  //  * ```
  //  */
  //     return F.Sign.apply(this)
  //   const abs = () => {
  // /**
  //  * Computes the absolute value of the tensor element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).abs().numpy())
  //  * ```
  //  */
  //     return this * this.sign()
  //   const reciprocal = () => {
  // /**
  //  * Compute `1/x` element-wise.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
  //  * ```
  //  */
  //     return F.Reciprocal.apply(this.cast(least_upper_float(this.dtype)))

  //   // ***** activation functions *****

  //   const elu = (alpha=1.0) => {
  // /**
  //  * Applies the Exponential Linear Unit (ELU) function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/elu
  //  * - Paper: https://arxiv.org/abs/1511.07289v5
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).elu().numpy())
  //  * ```
  //  */
  //     return this.relu() - alpha*(1-this.exp()).relu()

  //   const celu = (alpha=1.0) => {
  // /**
  //  * Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/celu
  //  * - Paper: https://arxiv.org/abs/1704.07483
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).celu().numpy())
  //  * ```
  //  */
  //     return this.maximum(0) + (alpha * ((this / alpha).exp() - 1)).minimum(0)

  //   const selu = (alpha=1.67326, gamma=1.0507) => {
  // /**
  //  * Applies the Scaled Exponential Linear Unit (SELU) function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/selu
  //  * - Paper: https://arxiv.org/abs/1706.02515v5
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).selu().numpy())
  //  * ```
  //  */
  //     return gamma * (this >= 0).detach().where(this, alpha * (this.exp() - 1))

  //   const swish = () => {
  // /**
  //  * See `.silu()`
  //  *
  //  * - Paper: https://arxiv.org/abs/1710.05941v1
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).swish().numpy())
  //  * ```
  //  */
  //     return this * this.sigmoid()

  //   const silu = () => {
  // /**
  //  * Applies the Sigmoid Linear Unit (SiLU) function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/silu
  //  * - Paper: https://arxiv.org/abs/1606.08415
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).silu().numpy())
  //  * ```
  //  */
  //     return this.swish()   // The SiLU function === also known as the swish function.

  //   const relu6 = () => {
  // /**
  //  * Applies the ReLU6 function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/relu6
  //  * - Paper: https://arxiv.org/abs/1704.04861v1
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-9., -6., -3., 0., 3., 6., 9.)!).relu6().numpy())
  //  * ```
  //  */
  //     return this.relu() - (this-6).relu()

  //   const hardswish = () => {
  // /**
  //  * Applies the Hardswish function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/hard-swish
  //  * - Paper: https://arxiv.org/abs/1905.02244v5
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).hardswish().numpy())
  //  * ```
  //  */
  //     return this * (this+3).relu6() * (1/6)

  //   const tanh = () => {
  // /**
  //  * Applies the Hyperbolic Tangent (tanh) function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions//Tanh
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).tanh().numpy())
  //  * ```
  //  */
  //     return 2.0 * ((2.0 * this).sigmoid()) - 1.0

  //   const sinh = () => {
  // /**
  //  * Applies the Hyperbolic Sine (sinh) function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions//Sinh
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).sinh().numpy())
  //  * ```
  //  */
  //     return (this.exp() - this.neg().exp()) / 2

  //   const cosh = () => {
  // /**
  //  * Applies the Hyperbolic Cosine (cosh) function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions//Cosh
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).cosh().numpy())
  //  * ```
  //  */
  //     return (this.exp() + this.neg().exp()) / 2

  //   const atanh = () => {
  // /**
  //  * Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions//atanh
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9)!).atanh().numpy())
  //  * ```
  //  */
  //     return ((1 + this)/(1 - this)).log() / 2

  //   const asinh = () => {
  // /**
  //  * Applies the Inverse Hyperbolic Sine (asinh) function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions//asinh
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).asinh().numpy())
  //  * ```
  //  */
  //     return (this + (this.square() + 1).sqrt()).log()

  //   const acosh = () => {
  // /**
  //  * Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions//acosh
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).acosh().numpy())
  //  * ```
  //  */
  //     return (this + (this.square() - 1).sqrt()).log()

  //   const hardtanh = (min_val=-1, max_val=1) => {
  // /**
  //  * Applies the Hardtanh function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/hardtanh-activation
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5)!).hardtanh().numpy())
  //  * ```
  //  */
  //     return this.clip(min_val, max_val)

  //   const erf = () => {
  // /**
  //  * Applies error function element-wise.
  //  *
  //  * - Described: https://en.wikipedia.org/wiki/Error_function
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5)!).erf().numpy())
  //  * ```
  //  */
  //     // https://personal.math.ubc.ca/~cbm/aands/page_299.htm 7.1.26
  //     t = 1.0 / (1.0 + 0.3275911 * this.abs())
  //     return this.sign() * (1.0 - t * polyN(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592]) * (-this.square()).exp())

  //   const gelu = () => {
  // /**
  //  * Applies the Gaussian Error Linear Unit (GELU) function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/gelu
  //  * - Paper: https://arxiv.org/abs/1606.08415v5
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).gelu().numpy())
  //  * ```
  //  */
  //     return 0.5 * this * (1 + (math.sqrt(2 / math.pi) * (this + 0.044715 * this ** 3)).tanh())

  //   const quick_gelu = () => {
  // /**
  //  * Applies the Sigmoid GELU approximation element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/gelu
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).quick_gelu().numpy())
  //  * ```
  //  */
  //     return this * (this * 1.702).sigmoid()

  //   const leakyrelu = (neg_slope=0.01) => {
  // /**
  //  * Applies the Leaky ReLU function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/leaky-relu
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).leakyrelu().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).leakyrelu(neg_slope=0.42).numpy())
  //  * ```
  //  */
  //     return this.relu() - (-neg_slope*this).relu()

  //   const mish = () => {
  // /**
  //  * Applies the Mish function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/mish
  //  * - Paper: https://arxiv.org/abs/1908.08681v3
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).mish().numpy())
  //  * ```
  //  */
  //     return this * this.softplus().tanh()

  //   const softplus = (beta=1) => {
  // /**
  //  * Applies the Softplus function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/softplus
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).softplus().numpy())
  //  * ```
  //  */
  //     return (1/beta) * (1 + (this*beta).exp()).log()

  //   const softsign = () => {
  // /**
  //  * Applies the Softsign function element-wise.
  //  *
  //  * - Described: https://paperswithcode.com/method/softsign
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-3., -2., -1., 0., 1., 2., 3.)!).softsign().numpy())
  //  * ```
  //  */
  //     return this / (1 + this.abs())

  //   // ***** broadcasted elementwise ops *****
  //   const _broadcast_to = (new_shape:sint[]): Tensor => {
  //     if this.shape === new_shape: return this
  //     if this.ndim > len(new_shape): raise ValueError(`can!broadcast tensor to fewer dimensions. shape=${this.shape} to ${new_shape=}`)
  //     // first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
  //     shape, _ = _align_left(this.shape, new_shape)
  //     // for each dimension, check either dim === 1, || it does !change
  //     if !all(resolve(s === ns) || resolve(s === 1) for (const s,ns of zip(shape, new_shape))){
  //       raise ValueError(`can!broadcast ${this.shape} to ${new_shape=}`)
  //     return F.Expand.apply(this.reshape(shape), shape=new_shape)

  //   const _broadcasted = (y:Union[Tensor, UOp, ConstType], reverse:boolean=false, match_dtype:boolean=true): [Tensor, Tensor] => {
  //     x: Tensor = this
  //     if !isinstance(y, Tensor):
  //       // make y a Tensor
  //       assert(isinstance(y, (*get_args(ConstType), UOp)), `${type(y)=}, ${y=}`)
  //       if isinstance(x.dtype, ImageDType) || dtypes.is_float(x.dtype) || (dtypes.is_int(x.dtype) && isinstance(y, number)): y_dtype = x.dtype
  //       else if !isinstance(y, UOp): y_dtype = dtypes.from_py(y)
  //       if isinstance(y, UOp): y = Tensor.from_uop(y, device=x.device)
  //       else: y = Tensor(dtypes.as_const(y, y_dtype), x.device, y_dtype, requires_grad=false)

  //     if match_dtype && x.dtype !== y.dtype:
  //       output_dtype = least_upper_dtype(x.dtype, y.dtype)
  //       x, y = x.cast(output_dtype), y.cast(output_dtype)

  //     if reverse: x, y = y, x

  //     // broadcast
  //     return x._broadcast_to(out_shape:=_broadcast_shape(x.shape, y.shape)), y._broadcast_to(out_shape)

  //   const _to_const_val = (x:Union[Tensor, ConstType]): Union[Tensor, ConstType] => {
  //     return x.lazydata.base.arg if isinstance(x, Tensor) && isinstance(x.lazydata, LazyBuffer) && x.lazydata.is_unrealized_unmasked_const() \
  //       && !x.requires_grad && this._broadcasted(x)[0].shape === this.shape else x

  //   const add = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Adds `this` && `x`.
  //  * Equivalent to `this + x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(4)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.add(20).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.add(Tensor([[2.0], [3.5]])).numpy())
  //  * ```
  //  */
  //     return F.Add.apply(*this._broadcasted(x, reverse))

  //   const sub = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Subtracts `x` from `this`.
  //  * Equivalent to `this - x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(4)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.sub(20).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.sub(Tensor([[2.0], [3.5]])).numpy())
  //  * ```
  //  */
  //     a, b = this._broadcasted(x, reverse)
  //     return a + (-b)

  //   const mul = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Multiplies `this` && `x`.
  //  * Equivalent to `this * x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(4)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.mul(3).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.mul(Tensor([.at(-1.0)!, [2.0]])).numpy())
  //  * ```
  //  */
  //     return F.Mul.apply(*this._broadcasted(x, reverse))

  //   const idiv = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Divides `this` by `x`.
  //  * Equivalent to `this // x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer inputs.
  //  * `idiv` performs integer division.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1, 4, 10]).idiv(Tensor([2, 3, 4])).numpy())
  //  * ```
  //  */
  //     return F.IDiv.apply(*this._broadcasted(x, reverse))

  //   const div = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Divides `this` by `x`.
  //  * Equivalent to `this / x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
  //  * `div` performs true division.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(4)
  //  * console.log(t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.div(3).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
  //  * ```
  //  */
  //     numerator, denominator = this._broadcasted(x, reverse)
  //     return numerator.cast(least_upper_float(numerator.dtype)) * denominator.cast(least_upper_float(denominator.dtype)).reciprocal()

  //   const xor = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Computes bitwise xor of `this` && `x`.
  //  * Equivalent to `this ^ x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, boolean inputs.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1, -2, 3)!).xor(Tensor([1, 0, 3])).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([true, true, false, false]).xor(Tensor([true, false, true, false])).numpy())
  //  * ```
  //  */
  //     if this.dtype !== dtypes.boolean && !dtypes.is_int(this.dtype): raise RuntimeError(`${this.dtype} !== supported`)
  //     return F.Xor.apply(*this._broadcasted(x, reverse))

  //   const bitwise_and = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Compute the bit-wise AND of `this` && `x`.
  //  * Equivalent to `this & x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, boolean inputs.
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([true, true, false, false]).bitwise_and(Tensor([true, false, true, false])).numpy())
  //  * ```
  //  */
  //     if this.dtype !== dtypes.boolean && !dtypes.is_int(this.dtype): raise RuntimeError(`${this.dtype} !== supported`)
  //     return F.BitwiseAnd.apply(*this._broadcasted(x, reverse))

  //   const bitwise_or = (x:Union[Tensor, ConstType], reverse=false): Tensor => {
  // /**
  //  * Compute the bit-wise OR of `this` && `x`.
  //  * Equivalent to `this | x`.
  //  * Supports broadcasting to a common shape, type promotion, && integer, boolean inputs.
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([true, true, false, false]).bitwise_or(Tensor([true, false, true, false])).numpy())
  //  * ```
  //  */
  //     if this.dtype !== dtypes.boolean && !dtypes.is_int(this.dtype): raise RuntimeError(`${this.dtype} !== supported`)
  //     return F.BitwiseOr.apply(*this._broadcasted(x, reverse))

  //   const bitwise_not = (): Tensor => {
  // /**
  //  * Compute the bit-wise NOT of `this`.
  //  * Equivalent to `~this`.
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([true, false]).bitwise_not().numpy())
  //  * ```
  //  */
  //     if this.dtype !== dtypes.boolean && !dtypes.is_int(this.dtype): raise RuntimeError(`${this.dtype} !== supported`)
  //     return this.logical_not() if this.dtype === dtypes.boolean else this ^ ((1<<8*this.dtype.itemsize)-1)

  //   const lshift = (x:number) => {
  // /**
  //  * Computes left arithmetic shift of `this` by `x` bits. `this` must have unsigned dtype.
  //  * Equivalent to `this << x`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
  //  * ```
  //  */
  //     assert(dtypes.is_unsigned(this.dtype) && isinstance(x, number) && x >= 0, `!supported ${this.dtype=} ${x=}`)
  //     return this.mul(2 ** x)

  //   const rshift = (x:number) => {
  // /**
  //  * Computes right arithmetic shift of `this` by `x` bits. `this` must have unsigned dtype.
  //  * Equivalent to `this >> x`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
  //  * ```
  //  */
  //     assert(dtypes.is_unsigned(this.dtype) && isinstance(x, number) && x >= 0, `!supported ${this.dtype=} ${x=}`)
  //     return this.idiv(2 ** x)

  //   const maximum = (x:Union[Tensor, ConstType]): Tensor => {
  // /**
  //  * Computes element-wise maximum of `this` && `x`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1, 2, 3)!).maximum(1).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1, 2, 3)!).maximum(Tensor(.at(-4, -2, 9)!)).numpy())
  //  * ```
  //  */
  //     return (this<x).detach().where(x, (this==x).detach().where(((this * 0.5 + x * 0.5).cast(this.dtype)), this))

  //   const minimum = (x:Union[Tensor, ConstType]): Tensor => {
  // /**
  //  * Computes element-wise minimum of `this` && `x`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1, 2, 3)!).minimum(1).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(Tensor(.at(-1, 2, 3)!).minimum(Tensor(.at(-4, -2, 9)!)).numpy())
  //  * ```
  //  */
  //     return -((-this).maximum(-x))

  //   const where = (:Tensor, x:Union[Tensor, ConstType, sint], y:Union[Tensor, ConstType, sint]) => {
  // /**
  //  * Return a tensor of elements selected from either `x` || `y`, depending on `this`.
  //  * `output_i = x_i if this_i else y_i`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * cond = Tensor([[true, true, false], [true, false, false]])
  //  * console.log(cond.where(1, 3).numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * cond = Tensor.randn(2, 3)
  //  * console.log(cond.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log((cond > 0).where(cond, -number("inf")).numpy())
  //  * ```
  //  */
  //     if isinstance(x, Tensor): x, y = x._broadcasted(y)
  //     else if isinstance(y, Tensor): y, x = y._broadcasted(x)
  //     cond, x = this._broadcasted(x, match_dtype=false)
  //     cond, y = cond._broadcasted(y, match_dtype=false)
  //     return F.Where.apply(cond.cast(dtypes.boolean), *x._broadcasted(y))

  //   const masked_fill = (:Tensor, mask:Tensor, value:Union[Tensor, ConstType]) =>  mask.where(value, this)

  //   // ***** op wrappers *****

  //   const __invert__ = (): Tensor =>  this.bitwise_not()

  //   const __lshift__ = (x): Tensor =>  this.lshift(x)
  //   const __rshift__ = (x): Tensor =>  this.rshift(x)

  //   const __pow__ = (x): Tensor =>  this.pow(x)
  //   const __matmul__ = (x): Tensor =>  this.matmul(x)

  //   const __rpow__ = (x): Tensor =>  this.pow(x, true)
  //   const __rmatmul__ = (x): Tensor =>  this.matmul(x, true)

  //   const __iadd__ = (x): Tensor =>  this.assign(this.add(x))
  //   const __isub__ = (x): Tensor =>  this.assign(this.sub(x))
  //   const __imul__ = (x): Tensor =>  this.assign(this.mul(x))
  //   const __ipow__ = (x): Tensor =>  this.assign(this.pow(x))
  //   const __itruediv__ = (x): Tensor =>  this.assign(this.div(x))
  //   const __ifloordiv__ = (x): Tensor =>  this.assign(this.idiv(x))
  //   const __imatmul__ = (x): Tensor =>  this.assign(this.matmul(x))
  //   const __iand__ = (x): Tensor =>  this.assign(this.bitwise_and(x))
  //   const __ior__ = (x): Tensor =>  this.assign(this.bitwise_or(x))
  //   const __ixor__ = (x): Tensor =>  this.assign(this.xor(x))
  //   const __ilshift__ = (x): Tensor =>  this.assign(this.lshift(x))
  //   const __irshift__ = (x): Tensor =>  this.assign(this.rshift(x))

  //   const __lt__ = (x): Tensor =>  F.Less.apply(*this._broadcasted(x, false))
  //   const __gt__ = (x): Tensor =>  F.Less.apply(*this._broadcasted(x, true))
  //   const ne = (x): Tensor =>  F.Neq.apply(*this._broadcasted(x))

  //   const __eq__ = (x): Tensor =>  this.eq(x)                      // type: ignore[override]

  //   // ***** functional nn ops *****

  //   const linear = (weight:Tensor, bias?: Tensor) => {
  // /**
  //  * Applies a linear transformation to `this` using `weight` && `bias`.
  //  *
  //  * See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 2], [3, 4]])
  //  * weight = Tensor([[1, 2], [3, 4]])
  //  * bias = Tensor([1, 2])
  //  * console.log(t.linear(weight, bias).numpy())
  //  * ```
  //  */
  //     x = this.mul(weight) if len(weight.shape) === 1 else this.dot(weight)
  //     return x.add(bias) if bias !== undefined else x

  //   const sequential = (ll:Callable[[Tensor[], Tensor]]) => {
  // /**
  //  * Applies a sequence of functions to `this` chaining the output of each function to the input of the next.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([1, 2, 3])
  //  * console.log(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
  //  * ```
  //  */
  //     return functools.reduce(lambda x,f: f(x), ll, this)

  //   const layernorm = (axis:Union[number,number[]]=-1, eps:number=1e-5): Tensor => {
  // /**
  //  * Applies Layer Normalization over a mini-batch of inputs.
  //  *
  //  * - Described: https://paperswithcode.com/method/layer-normalization
  //  * - Paper: https://arxiv.org/abs/1607.06450v1
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.randn(8, 10, 16) * 2 + 8
  //  * console.log(t.mean().item(), t.std().item())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.layernorm()
  //  * console.log(t.mean().item(), t.std().item())
  //  * ```
  //  */
  //     y = (this - this.mean(axis, keepdim=true))
  //     return y.mul((y*y).mean(axis, keepdim=true).add(eps).rsqrt())

  //   const batchnorm = (weight?: Tensor, bias?: Tensor, mean:Tensor, invstd:Tensor, axis:Union[number,number[]]=1): Tensor => {
  // /**
  //  * Applies Batch Normalization over a mini-batch of inputs.
  //  *
  //  * - Described: https://paperswithcode.com/method/batch-normalization
  //  * - Paper: https://arxiv.org/abs/1502.03167
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor.randn(8, 4, 16, 16) * 2 + 8
  //  * console.log(t.mean().item(), t.std().item())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.batchnorm(undefined, undefined, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
  //  * console.log(t.mean().item(), t.std().item())
  //  * ```
  //  */
  //     axis_ = argfix(axis)
  //     shape = tuple(s if ax in axis_ else 1 for ax, s in enumerate(this.shape))
  //     x = this - mean.reshape(shape)
  //     if weight !== undefined: x = x * weight.reshape(shape)
  //     ret = x.mul(invstd.reshape(shape) if len(invstd.shape) === len(axis_) else invstd)
  //     return (ret + bias.reshape(shape)) if bias !== undefined else ret

  //   const dropout = (p=0.5): Tensor => {
  // /**
  //  * Applies dropout to `this`.
  //  *
  //  * NOTE: dropout === only applied when `Tensor.training` === `true`.
  //  *
  //  * - Described: https://paperswithcode.com/method/dropout
  //  * - Paper: https://jmlr.org/papers/v15/srivastava14a.html
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * Tensor.manual_seed(42)
  //  * t = Tensor.randn(2, 2)
  //  * with Tensor.train():
  //  * console.log(t.dropout().numpy())
  //  * ```
  //  */
  //     if !Tensor.training || p === 0: return this
  //     return (Tensor.rand_like(this, requires_grad=false, dtype=dtypes.default_float, contiguous=false) >= p).contiguous().where(this, 0) / (1.0 - p)

  //   // helper function commonly used for indexing
  //   const _one_hot_along_dim = (:Tensor, num_classes:sint, dim:number=-1) => {
  //     offset = this.ndim - this._resolve_dim(dim) - 1
  //     return this === Tensor.arange(num_classes, device=this.device, requires_grad=false).reshape((num_classes,) + (1,) * offset)

  //   const one_hot = (num_classes:number=-1): Tensor => {
  // /**
  //  * Converts `this` to a one-hot tensor.
  //  *
  //  * `num_classes` defaults to -1, which means num_classes will be inferred as max(this) + 1.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([0, 1, 3, 3, 4])
  //  * console.log(t.one_hot(5).numpy())
  //  * ```
  //  */
  //     if num_classes === -1: num_classes = (this.max()+1).item()
  //     return this[..., undefined]._one_hot_along_dim(num_classes).where(1, 0)

  //   const sparse_categorical_crossentropy = (Y:Tensor, ignore_index:number=-1, label_smoothing=0.0, reduction:ReductionStr="mean"): Tensor => {
  // /**
  //  * Computes the sparse categorical cross-entropy loss between `this` && `Y`.
  //  *
  //  * NOTE: `this` === logits && `Y` === the target labels.
  //  * NOTE: unlike PyTorch, this function expects the class axis to be -1
  //  *
  //  * See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([.at(-1, 2, -3)!, [1, -2, 3]])
  //  * Y = Tensor([1, 2])
  //  * console.log(t.sparse_categorical_crossentropy(Y).item())
  //  * ```
  //  */
  //     assert(0.0 <= label_smoothing <= 1.0, "label_smoothing must be in [0.0, 1.0]")
  //     assert(reduction in ("mean", "sum", "none"), "reduction must be one of ['mean', 'sum', 'none']")
  //     log_probs, loss_mask = this.log_softmax(), (Y !== ignore_index) if ignore_index !== -1 else Y.ones_like(dtype=dtypes.boolean)
  //     y_counted = Y.to(this.device).flatten().reshape(-1, 1)._one_hot_along_dim(this.shape.at(-1)!)
  //     y = (y_counted * loss_mask.reshape(-1, 1)).reshape(*Y.shape, this.shape.at(-1)!)
  //     smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
  //     unreduced = ((1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing)
  //     // NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
  //     return -(unreduced.sum() / loss_mask.sum() if reduction === "mean" else (unreduced.sum() if reduction === "sum" else unreduced))

  //   // ***** Tensor Properties *****

  //   @property
  //   const ndim = (): number => {
  // /**
  //  * Returns the number of dimensions in the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[1, 2], [3, 4]])
  //  * console.log(t.ndim)
  //  * ```
  //  */
  //     return len(this.shape)

  //   const numel = (): sint => {
  // /**
  //  * Returns the total number of elements in the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
  //  * console.log(t.numel())
  //  * ```
  //  */
  //     return prod(this.shape)

  //   const element_size = (): number => {
  // /**
  //  * Returns the size in bytes of an individual element in the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([5], dtype=dtypes.int16)
  //  * console.log(t.element_size())
  //  * ```
  //  */
  //     return this.dtype.itemsize

  //   const nbytes = (): number => {
  // /**
  //  * Returns the total number of bytes of all elements in the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([8, 9], dtype=dtypes.number)
  //  * console.log(t.nbytes())
  //  * ```
  //  */
  //     return this.numel() * this.element_size()

  //   const is_floating_point = (): boolean => {
  // /**
  //  * Returns `true` if the tensor contains floating point types, i.e. === one of `dtype.float64`, `dtype.float32`,
  //  * `dtype.float16`, `dtype.bfloat16`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([8, 9], dtype=dtypes.float32)
  //  * console.log(t.is_floating_point())
  //  * ```
  //  */
  //     return dtypes.is_float(this.dtype)

  //   const size = (dim?: number): Union[sint, sint[]] => {
  // /**
  //  * Return the size of the tensor. If `dim` === specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor([[4, 5, 6], [7, 8, 9]])
  //  * console.log(t.size())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * console.log(t.size(dim=1))
  //  * ```
  //  */
  //     return this.shape if dim === undefined else this.shape[dim]

  //   // ***** cast ops *****

  //   const llvm_bf16_cast = (dtype:DTypeLike) => {
  //     // hack for devices that don't support bfloat16
  //     assert(this.dtype === dtypes.bfloat16)
  //     return this.to("LLVM").bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1<<16).bitcast(dtypes.float32).cast(dtype)

  //   const cast = (dtype:DTypeLike): Tensor => {
  // /**
  //  * Casts `this` to the given `dtype`.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1, 2.5, 3)!, dtype=dtypes.number)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.cast(dtypes.int32)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  */
  //     return this if this.dtype === (dt:=to_dtype(dtype)) else F.Cast.apply(this, dtype=dt)

  //   const bitcast = (dtype:DTypeLike): Tensor => {
  // /**
  //  * Bitcasts `this` to the given `dtype` of the same itemsize.
  //  *
  //  * `this` must !require a gradient.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1, 2, 3)!, dtype=dtypes.int32)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.bitcast(dtypes.uint32)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  */
  //     if this.requires_grad: raise RuntimeError("can't backprop through bitcast")
  //     dt = to_dtype(dtype)
  //     if (!isinstance(this.device, string) || !this.device.startsWith("DISK")) && (ns:=dt.itemsize) !== (os:=this.dtype.itemsize):
  //       if (this.shape.at(-1)!*os) % ns !== 0: raise RuntimeError("unsupported size in bitcast")
  //       new_uint, old_uint = to_dtype(`uint${8*ns}`), to_dtype(`uint${8*os}`)
  //       tmp = this.bitcast(old_uint)
  //       if ns > os: return functools.reduce(Tensor.add, (tmp[..., i::ns//os].cast(new_uint) << 8*i*os for i in range(ns//os))).bitcast(dtype)
  //       return Tensor.stack(*(tmp>>8*i*ns for i in range(os//ns)), dim=-1).flatten(-2).cast(new_uint).bitcast(dtype)
  //     return F.Cast.apply(this, dtype=dt, bitcast=true) if this.dtype !== dt else this

  //   const number = (): Tensor => {
  // /**
  //  * Convenience method to cast `this` to a `float32` Tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1, 2, 3)!, dtype=dtypes.int32)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.number()
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  */
  //     return this.cast(dtypes.float32)

  //   const half = (): Tensor => {
  // /**
  //  * Convenience method to cast `this` to a `float16` Tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1, 2, 3)!, dtype=dtypes.int32)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.half()
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  */
  //     return this.cast(dtypes.float16)

  //   const number = (): Tensor => {
  // /**
  //  * Convenience method to cast `this` to a `int32` Tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1.5, -0.5, 0.0, 0.5, 1.5)!)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.number()
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  */
  //     return this.cast(dtypes.int32)

  //   const boolean = (): Tensor => {
  // /**
  //  * Convenience method to cast `this` to a `boolean` Tensor.
  //  *
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = Tensor(.at(-1, 0, 1)!)
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  * ```python exec="true" source="above" session="tensor" result="python"
  //  * t = t.boolean()
  //  * console.log(t.dtype, t.numpy())
  //  * ```
  //  */
  //     return this.cast(dtypes.boolean)
}
//   // *** image Tensor function replacements ***

// const _metadata_wrapper = (fn) => {
//   const _wrapper = (*args, **kwargs) => {
//     if _METADATA.get() !== undefined: return fn(*args, **kwargs)

//     caller = ""

//     token = _METADATA.set(Metadata(name=fn.__name__, caller=caller))
//     ret = fn(*args, **kwargs)
//     _METADATA.reset(token)
//     return ret
//   return _wrapper

// if TRACEMETA >= 1:
//   for (const name, fn of inspect.getmembers(Tensor, inspect.isfunction)){
//     if name in ["__class__", "__init__", "__new__", "__repr__", "backward", "sequential"]: continue
//     setattr(Tensor, name, functools.wraps(fn)(_metadata_wrapper(fn)))
