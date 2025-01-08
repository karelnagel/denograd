// deno-lint-ignore-file no-this-alias
import { ConstType, DType, DTypeLike, dtypes, ImageDType, least_upper_dtype, least_upper_float, sum_acc_dtype, to_dtype } from './dtype.ts'
import { LazyBuffer } from './engine/lazy.ts'
import { _METADATA, all_int, all_same, argfix, assert, bytesToBigInt, DEBUG, dedup, fully_flatten, get_env, IMAGE, intToBytes, isEq, isinstance, listStr, max, Metadata, range, sha256, Slice, slice, WINO, zip } from './helpers.ts'
import { add, ceildiv, ge, gt, identity_element, idiv, le, MathTrait, mul, ne, neg, Ops, polyN, prod, resolve, sint, smax, smin, sub, UOp, Variable } from './ops.ts'
import { Buffer, BufferSpec, Device, DeviceType } from './device.ts'
import path from 'node:path'
import { create_schedule_with_vars, ScheduleContext, ScheduleItem, to_uop } from './engine/schedule.ts'
import { memory_planner } from './engine/memory.ts'
import { run_schedule } from './engine/realize.ts'
// // **** start with two base classes, Tensor && Function ****
import { gunzip } from 'node:zlib'
import { promisify } from 'node:util'
const gunzipAsync = promisify(gunzip)
import { make_tuple, round_up } from './helpers.ts'
import { argsort } from './helpers.ts'
import { MemoryView } from './memoryview.ts'

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
  forward = (..._args: any[]): any => {
    throw new Error(`forward !implemented for ${this}`)
  }
  backward = (_grad_output: LazyBuffer): any => {
    throw new Error(`backward !implemented for ${this}`)
  }

  static apply(...args: any[]): Tensor {
    assert(args.length > 0, 'No args')

    const x = args.filter((x) => x instanceof Tensor)
    const ctx = new this(x[0].device, x, _METADATA.value)

    const ret = new Tensor(undefined, undefined, true)
    ret.lazydata = ctx.forward(...args.map((v) => v instanceof Tensor ? v.lazydata : v))
    ret.requires_grad = ctx.requires_grad
    ret.grad = undefined

    ret._ctx = ctx.requires_grad && !Tensor.no_grad ? ctx : undefined // used by autograd engine
    return ret
  }
}
// ************* function.py start *************
/**
 * This === where the forwards && backwards passes live.
 */

export class Contiguous extends Function {
  override forward = (x: LazyBuffer): LazyBuffer => x.contiguous()
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output
}

export class ContiguousBackward extends Function {
  override forward = (x: LazyBuffer): LazyBuffer => x
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.contiguous()
}

export class Cast extends Function {
  input_dtype!: DType
  bitcast?: boolean
  override forward = (x: LazyBuffer, dtype: DType, bitcast?: boolean): LazyBuffer => {
    this.input_dtype = x.dtype, this.bitcast = bitcast
    return this.bitcast ? x.bitcast(dtype) : x.cast(dtype)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => {
    if (this.bitcast) throw new Error('bitcast can!backward')
    return grad_output.cast(this.input_dtype!)
  }
}

// // ************* unary ops *************

export class Reciprocal extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = x.reciprocal()
    return this.ret
  }

  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.neg().mul(this.ret).mul(this.ret)
}
export class Sin extends Function {
  x!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.x = x
    return x.sin()
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => (this.x.sub(Math.PI / 2, true)).sin().mul(grad_output)
}
export class Relu extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = x.maximum(0)
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => (this.ret.gt(0)).cast(grad_output.dtype).mul(grad_output)
}
export class Log extends Function {
  x!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.x = x
    return x.log2().mul(Math.log(2))
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.div(this.x)
}
export class Exp extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = (x.mul(1 / Math.log(2))).exp2()
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => this.ret.mul(grad_output)
}
export class Sqrt extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = x.sqrt()
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.div(this.ret.mul(2))
}
// // NOTE: the implicit derivative of sigmoid !== stable
// // https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
// // TODO: have the backend automatically find this
export class Sigmoid extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer) => {
    this.ret = ((x.mul(-1 / Math.log(2))).exp2().add(1, true)).reciprocal()
    return this.ret
  }
  override backward = (grad_output: LazyBuffer) => {
    return (this.ret.mul(this.ret.sub(1, true))).mul(grad_output)
  }
}
export class Sign extends Function {
  override forward = (x: LazyBuffer): LazyBuffer => x.ne(0).where((x.lt(0)).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  //   // backward always return 0 to match torch
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.const_like(0)
}
// // ************* binary ops *************

export class Less extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.lt(y)
  override backward = (_grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [undefined, undefined]
}
export class Neq extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.ne(y)
  override backward = (_grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [undefined, undefined]
}
export class Xor extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.xor(y)
}
export class BitwiseAnd extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.bitwise_and(y)
}
export class BitwiseOr extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.bitwise_or(y)
}
export class Threefry extends Function {
  override forward = (x: LazyBuffer, seed: LazyBuffer): LazyBuffer => x.threefry(seed)
}

export class Add extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.add(y)

  override backward = (grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [this.needs_input_grad[0] ? grad_output : undefined, this.needs_input_grad[1] ? grad_output : undefined]
}
export class Mul extends Function {
  x!: LazyBuffer
  y!: LazyBuffer
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => {
    this.x, this.y = x, y
    return x.mul(y)
  }
  override backward = (grad_output: LazyBuffer): [LazyBuffer?, LazyBuffer?] => [
    this.needs_input_grad[0] ? (this.y.mul(grad_output)) : undefined,
    this.needs_input_grad[1] ? (this.x.mul(grad_output)) : undefined,
  ]
}
export class IDiv extends Function {
  override forward = (x: LazyBuffer, y: LazyBuffer): LazyBuffer => x.idiv(y)
}
// // ************* ternary ops *************

export class Where extends Function {
  x!: LazyBuffer
  override forward = (x: LazyBuffer, y: LazyBuffer, z: LazyBuffer): LazyBuffer => {
    this.x = x
    return this.x.where(y, z)
  }
  override backward = (grad_output: LazyBuffer): [undefined, LazyBuffer | undefined, LazyBuffer | undefined] => [
    undefined,
    this.needs_input_grad[1] ? this.x.where(grad_output, grad_output.const_like(0)) : undefined,
    this.needs_input_grad[2] ? this.x.where(grad_output.const_like(0), grad_output) : undefined,
  ]
}
// // ************* reduce ops *************

export class Sum extends Function {
  input_shape!: sint[]
  override forward = (x: LazyBuffer, axis: number[]): LazyBuffer => {
    this.input_shape = x.shape
    return x.r(Ops.ADD, axis)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.expand(this.input_shape)
}
export class Prod extends Function {
}
export class Max extends Function {
  x!: LazyBuffer
  ret!: LazyBuffer
  axis!: number[]
  override forward = (x: LazyBuffer, axis: number[]): LazyBuffer => {
    this.x = x, this.ret = x.r(Ops.MAX, axis), this.axis = axis
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => {
    // 1s in locations where the max was chosen (can be two locations)
    const max_is_1s = this.x.ne(this.ret.expand(this.x.shape)).ne(this.x.const_like(1).cast(dtypes.bool)).cast(grad_output.dtype)
    const div = max_is_1s.r(Ops.ADD, this.axis).expand(this.x.shape)
    return (max_is_1s.div(div)).mul(grad_output.expand(this.x.shape))
  }
}
// // ************* movement ops *************

// // NOTE: this === sum in reverse
export class Expand extends Function {
  expanded_axis!: number[]
  override forward = (x: LazyBuffer, shape: number[]): LazyBuffer => {
    this.expanded_axis = zip(x.shape, shape).filter(([si, so]) => resolve(ne(si, so))).map((_, i) => i)
    return x.expand(shape)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => {
    return grad_output.cast(sum_acc_dtype(grad_output.dtype)).r(Ops.ADD, this.expanded_axis).cast(grad_output.dtype)
  }
}

export class Reshape extends Function {
  input_shape!: sint[]
  override forward = (x: LazyBuffer, shape: number[]): LazyBuffer => {
    this.input_shape = x.shape
    return x.reshape(shape)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.reshape(this.input_shape)
}
export class Permute extends Function {
  input_order!: number[]
  override forward = (x: LazyBuffer, order: number[]): LazyBuffer => {
    this.input_order = order
    return x.permute(order)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.permute(argsort(this.input_order))
}
export class Pad extends Function {
  narg!: [sint, sint][]
  override forward = (x: LazyBuffer, arg: [number, number][]): LazyBuffer => {
    this.narg = zip(x.shape, arg).map(([s, p]) => [p[0], add(s, p[0])])
    return x.pad(arg)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.shrink(this.narg)
}
export class Shrink extends Function {
  narg!: [sint, sint][]
  override forward = (x: LazyBuffer, arg: [sint, sint][]): LazyBuffer => {
    this.narg = zip(x.shape, arg).map(([s, p]) => [p[0], sub(s, p[1])])
    return x.shrink(arg)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.pad(this.narg)
}
export class Flip extends Function {
  arg!: number[]
  override forward = (x: LazyBuffer, axis: number[]): LazyBuffer => {
    this.arg = range(x.shape.length).map((i) => axis.includes(i) ? -1 : 1)
    return x.stride(this.arg)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.stride(this.arg)
}

// ************* function.py end *************

export const _metaop = (op: Ops, shape: sint[], dtype: DType, device: DeviceType | DeviceType[], arg?: any, src: LazyBuffer[] = []) => {
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
export const _frompy = (x: any[] | Uint8Array, dtype: DType): LazyBuffer => {
  let ret, data
  if (x instanceof Uint8Array) [ret, data] = [LazyBuffer.metaop(Ops.EMPTY, [idiv(x.length, dtype.itemsize)], dtype, 'PYTHON'), x]
  else {
    ret = LazyBuffer.metaop(Ops.EMPTY, get_shape(x), dtype, 'PYTHON')
    assert(dtype.fmt !== undefined, `${dtype} has undefined fmt`)
    data = new MemoryView(fully_flatten(x), { fmt: dtype.fmt }) //KAREL: not that sure
  }
  //   // fake realize
  ret.buffer!.allocate(new MemoryView(data as Uint8Array, { fmt: 'B' }))
  ret.srcs?.forEach((x) => x.__del__())
  delete ret.srcs
  return ret
}
const _align_left = (...shapes: sint[][]): sint[][] => {
  //   // unsqueeze left to make every shape same length
  const max_dim = max(shapes.map((shape) => shape.length))
  return shapes.map((shape) => [...range(max_dim - shape.length).map(() => 1), ...shape])
}
export const _broadcast_shape = (shapes: sint[][]): sint[] => {
  return zip(..._align_left(...shapes)).map((nth_dim_sizes) => nth_dim_sizes.includes(0) ? 0 : smax(nth_dim_sizes))
}
type ReductionStr = 'mean' | 'sum' | 'none'

export type TensorOptions = { device?: DeviceType | DeviceType[]; dtype?: DType; requires_grad?: boolean }
const sliceGetIndices = (index: Slice, size: number): [number, number, number] => {
  let start = index.start ?? 0
  let stop = index.stop ?? size
  const step = index.step ?? 1
  if (start < 0) start += size
  if (stop < 0) stop += size
  start = Math.max(step < 0 ? -1 : 0, Math.min(size, start))
  stop = Math.max(step < 0 ? -1 : 0, Math.min(size, stop))
  return [start, stop, step]
}
export type TensorIndice = number | boolean | Tensor | UOp | undefined | '...' | Slice | (number | boolean | UOp | Tensor | undefined | '...' | Slice)[]
export type Layer = ((x: Tensor) => Tensor) | { call: (x: Tensor) => Tensor }
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
export class Tensor extends MathTrait<Tensor> {
  lazydata!: LazyBuffer
  requires_grad?: boolean
  // tensors can have gradients if you have called .backward
  grad?: Tensor
  // internal variable used for autograd graph construction
  _ctx?: Function
  __deletable__ = ['_ctx']
  static training = false
  static no_grad = false

  static new = (data?: ConstType | UOp | Uint8Array | any[] | LazyBuffer | Tensor | string, opts?: TensorOptions) => new Tensor(data, opts)
  constructor(data?: ConstType | UOp | Uint8Array | any[] | LazyBuffer | Tensor | string, { device, dtype, requires_grad }: TensorOptions = {}, skip_constructor = false) {
    super()
    if (skip_constructor) return
    if (dtype !== undefined) dtype = to_dtype(dtype)
    assert(dtype === undefined || dtype instanceof DType, `invalid dtype ${dtype}`)
    if (device === undefined && typeof data === 'string' && path.isAbsolute(data)) device = `DISK:${data}` // keep it on the disk if device === undefined
    device = Array.isArray(device) ? device.map((x) => Device.canonicalize(x)) : Device.canonicalize(device)

    //     // NOTE: this can be in three states. false && undefined: no gradient, true: gradient
    //     // undefined (the default) will be updated to true if it's put in an optimizer
    this.requires_grad = requires_grad

    //     // create a LazyBuffer from the different types of inputs
    if (data instanceof LazyBuffer) {
      assert(dtype === undefined || dtype === data.dtype, "dtype doesn't match, && casting isn't supported")
    } else if (data === undefined) {
      data = _metaop(Ops.EMPTY, [0], dtype || dtypes.default_float, device)
    } else if (typeof data === 'number' || typeof data === 'boolean' || typeof data === 'bigint') {
      data = _metaop(Ops.CONST, [], dtype || dtypes.from_js(data), device, data)
    } else if (data instanceof UOp) {
      assert(data.op === Ops.BIND && data.src[0].op === Ops.DEFINE_VAR && data.src[1].op === Ops.CONST, `can't create tensor from UOp ${data}`)
      data = _metaop(Ops.CONST, [], dtype || data.dtype, device, data)
    } else if (data instanceof Uint8Array) {
      data = _frompy(data, dtype || dtypes.uint8)
    } else if (Array.isArray(data)) {
      if (dtype === undefined) {
        const d = fully_flatten(data)
        if (d.length && d.every((s) => typeof s === 'boolean')) dtype = dtypes.bool
        else dtype = (d.length && all_int(d)) ? dtypes.default_int : dtypes.default_float
      }
      if (dtype === dtypes.bfloat16) data = new Tensor(_frompy(data, dtypes.float32), { device }).cast(dtypes.bfloat16).lazydata
      else data = _frompy(data, dtype)
    } //     // else if string(type(data)) === "<class 'numpy.ndarray'>":
    //     //   import numpy as np
    //     //   assert(isinstance(data, np.ndarray), `expected np.ndarray, got ${data}`)
    //     //   if data.shape === (): data = _metaop(Ops.CONST, tuple(), dtype || _from_np_dtype(data.dtype), device, data.item())
    //     //   else: data = _fromnp(data.astype(npdtype) if dtype !== undefined && (npdtype:=_to_np_dtype(dtype)) !== undefined else data)  // type: ignore [name-defined]
    // Using string as path
    else if (typeof data === 'string') {
      dtype = dtype || dtypes.uint8
      data = _metaop(Ops.EMPTY, [idiv(Deno.statSync(data).size, dtype.itemsize)], dtype, `DISK:${data}`)
    }

    //     // by this point, it has to be a LazyBuffer
    if (!isinstance(data, LazyBuffer)) throw new Error(`can't create Tensor from ${data} with type ${typeof data}`)

    //     // data might be on a different device
    if (typeof device === 'string') this.lazydata = data.device === device ? data : data.copy_to_device(device)
    //     // if device === a tuple, we should have/construct a MultiLazyBuffer
    else throw new Error('TODO: multi')
    // else if (isinstance(data, LazyBuffer)) throw new Error('MultiLazyBuffer')
    // else {
    //   // assert(data.device === device, `MultiLazyBuffer device mismatch, ${data.device} !== ${device}`)
    //   this.lazydata = data
    // }
  }
  override toString = () => `<Tensor ${this.lazydata} on ${this.device} with grad ${this.grad?.lazydata}>`;
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }

  //   // Python has a non moving GC, so this should be okay
  // const __hash__ = () =>  id(this)

  bool = () => {
    throw new Error('__bool__ on Tensor !== defined')
  }
  override mod = (x: ConstType<Tensor>, reverse?: boolean) => {
    throw new Error("KAREL: Tensor doesn't have mod, this is only here because I needed to extend MathTrait and not SimpleMathTrait")
  }
  get length() {
    if (!this.shape.length) throw new Error('len() of a 0-d tensor')
    return this.shape[0]
  }
  get device(): DeviceType | DeviceType[] {
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
  schedule_with_vars = (lst: Tensor[] = []): [ScheduleItem[], Map<Variable, number>] => {
    const [schedule, var_vals] = create_schedule_with_vars([this, ...lst].flatMap((x) => x.lazydata.lbs))
    return [memory_planner(schedule), var_vals]
  }
  _debug_ast = () => {
    const [schedule, vars] = create_schedule_with_vars(this.cast(this.dtype.base).contiguous().to('CLANG').lazydata.lbs)
    return schedule.map((s) => s.ast)
  }
  _debug = () => {
    const ctx = new ScheduleContext()
    const cache = new Map<LazyBuffer, UOp>()
    const buffers = new Map<UOp, Buffer>()
    const uop = to_uop(this.lazydata, ctx, buffers, cache)
    return uop
  }
  /**
   * Creates the schedule needed to realize these Tensor(s).
   */
  schedule = (...lst: Tensor[]): ScheduleItem[] => {
    const [schedule, var_vals] = this.schedule_with_vars(lst)
    assert(var_vals.size === 0)
    return schedule
  }

  //   /**
  //    * Triggers the computation needed to create these Tensor(s).
  //    */
  realize = async (lst?: Tensor[], do_update_stats = true): Promise<Tensor> => {
    await run_schedule(...this.schedule_with_vars(lst || []), do_update_stats)
    return this
  }
  static realize = (lst: Tensor[], do_update_stats = true) => lst[0].realize(lst.slice(1), do_update_stats)
  /**
   * Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
   */
  replace = (x: Tensor): Tensor => {
    // used for replacing a Tensor with a new version of it (potentially with a different device && dtype)
    assert(!x.requires_grad && this._ctx === undefined)
    assert(isEq(this.shape, x.shape), `replace shape mismatch ${this.shape} !== ${x.shape}`)
    this.lazydata = x.lazydata
    return this
  }
  assign_disk = async (x: Tensor | number[] | string | Uint8Array): Promise<Tensor> => {
    if (!(x instanceof Tensor)) x = new Tensor(x, { device: this.device, dtype: this.dtype })
    if (typeof this.device === 'string' && !this.device.startsWith('DISK')) throw new Error('This can be only used with DISK device')
    ;(await this.contiguous().realize()).lazydata.base.realized!.copyin(await x._data())
    return this
  }
  assign = (x: Tensor | number[] | string | Uint8Array): Tensor => {
    if (!(x instanceof Tensor)) x = new Tensor(x, { device: this.device, dtype: this.dtype })
    //   // TODO: this is a hack for writing to DISK. remove with working assign
    if (typeof this.device === 'string' && this.device.startsWith('DISK')) throw new Error("Use async assign_disk instead, until disk get's good assign")
    if (DEBUG >= 4) console.log(`assign ${this.lazydata} <- ${x.lazydata}`)
    if (this.lazydata === x.lazydata) return this // a this assign === a NOOP
    // NOTE: we allow cross device assign
    assert(isEq(this.shape, x.shape), `assign shape mismatch ${this.shape} !== ${x.shape}`)
    assert(this.device === x.device, `assign device mismatch ${this.device} !== ${x.device}`)
    assert(this.dtype === x.dtype, `assign dtype mismatch ${this.dtype} !== ${x.dtype}`)
    // assert(!isinstance(this.lazydata, MultiLazyBuffer) || this.lazydata.axis === x.lazydata.axis, "axis must match on MultiLazyBuffer")
    assert(!x.requires_grad) // this requires_grad === okay?
    if (!this.lazydata.is_realized) return this.replace(x)
    this.lazydata = this.lazydata.assign(x.lazydata)
    return this
  }
  /**
   * Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
   */
  detach = (): Tensor => {
    return new Tensor(this.lazydata, { device: this.device, requires_grad: false })
  }
  _data = async (): Promise<MemoryView> => {
    if (this.shape.includes(0)) return new MemoryView(new Uint8Array(0))
    // NOTE: this realizes on the object from as_buffer being a Python object
    const cpu = await this.cast(this.dtype.base).contiguous().to('CLANG').realize()
    const buf = cpu.lazydata!.base.realized
    if (this.device !== 'CLANG') buf!.options = new BufferSpec(undefined, undefined, undefined, undefined, true)
    return buf!.as_buffer(this.device !== 'CLANG' ? true : false)
  }
  /**
   * Returns the data of this tensor as a memoryview.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([1, 2, 3, 4])
   * console.log(np.frombuffer(t.data(), dtype=np.int32))
   * ```
   */
  data = async (): Promise<MemoryView<any>> => {
    assert(this.dtype.base.fmt !== undefined, `no fmt dtype for ${this.dtype.base}`)
    assert(all_int(this.shape), `no data if shape === symbolic, ${this.shape}`)
    // if (TYPE_CHECKING ) assert(this.dtype.base.fmt !== "e")
    return await this._data().then((x) => x.cast(this.dtype.base.fmt!, this.shape.includes(0) ? undefined : this.shape as number[]))
  }
  /**
   * Returns the value of this tensor as a standard Python number.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor(42)
   * console.log(t.item())
   * ```
   */
  item = async (): Promise<ConstType> => {
    assert(this.numel() === 1, 'must have one element for item')
    return await this.data().then((x) => x.getValue(...range(this.shape.length || 1).map(() => 0))) as number
  }
  // TODO: should be Tensor.tolist() -> Union[ConstType[], ConstType]. The List === Sequence because mypy expects memoryview.tolist() -> list[number]
  // src: https://github.com/python/mypy/blob/release-1.6/mypy/typeshed/stdlib/builtins.pyi//L803
  /**
   * Returns the value of this tensor as a nested list.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([1, 2, 3, 4])
   * console.log(t.tolist())
   * ```
   */
  tolist = async <T = any>(): Promise<T> => {
    return await this.data().then((x) => x.toList()) as T
  }
  /**
   * Creates a clone of this tensor allocating a separate buffer for the data.
   */
  clone = (): Tensor => {
    const ret = new Tensor(this.lazydata.clone(), { device: this.device, requires_grad: this.requires_grad })
    if (this.grad !== undefined) ret.grad = this.grad.clone()
    if (this._ctx) ret._ctx = this._ctx
    return ret
  }
  /**
   * Moves the tensor to the given device.
   */
  to = (device?: DeviceType | DeviceType[]): Tensor => {
    device = Array.isArray(device) ? device.map((x) => Device.canonicalize(x)) : Device.canonicalize(device)
    if (device === this.device) return this
    if (typeof device !== 'string') {
      throw new Error('KAREL: implement shard()')
      // return this.shard(device)
    }
    const ret = new Tensor(this.lazydata, { device, requires_grad: this.requires_grad })
    if (this.grad !== undefined) ret.grad = this.grad.to(device)
    if (this._ctx !== undefined) ret._ctx = this._ctx
    return ret
  }

  static from_uop = (y: UOp, opts?: TensorOptions): Tensor => {
    if (y.op === Ops.BIND) return new Tensor(y, { ...opts, requires_grad: false }) // this is the only UOp allowed in Tensor
    if (y.op === Ops.CONST) return new Tensor(y.arg, { ...opts, requires_grad: false })
    if (y.op === Ops.MUL) return Tensor.from_uop(y.src[0]).mul(Tensor.from_uop(y.src[1]))
    if (y.op === Ops.ADD) return Tensor.from_uop(y.src[0]).add(Tensor.from_uop(y.src[1]))
    if (y.op === Ops.MAX) return Tensor.from_uop(y.src[0]).maximum(Tensor.from_uop(y.src[1]))
    throw new Error(`unhandled UOp {y}`)
  }

  static _metaop = (op: any, shape: number[], opts: TensorOptions = {}, arg?: any) => {
    const dtype = opts.dtype !== undefined ? to_dtype(opts.dtype) : dtypes.default_float
    return new Tensor(LazyBuffer.metaop(op, shape, dtype, Device.canonicalize(opts.device as DeviceType), arg), { ...opts, dtype })
  }

  /**
   * Creates an empty tensor with the given shape.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.empty(2, 3)
   * print(t.shape)
   * ```
   * """
   */
  static empty = (shape: number[], opts: TensorOptions = {}) => Tensor._metaop(Ops.EMPTY, argfix(shape), opts)

  /**
   * Create a Tensor from a URL.
   *
   * This === the preferred way to access Internet resources.
   * It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
   * This also will soon become lazy (when possible) && !print progress without DEBUG.
   *
   * THe `gunzip` flag will gzip extract the resource && return an extracted Tensor.
   */

  static from_url = async (url: string, gunzip = false, opts?: TensorOptions): Promise<Tensor> => {
    let data = await fetch(url).then((data) => data.arrayBuffer())
    if (gunzip) data = await gunzipAsync(data)
    return new Tensor(new Uint8Array(data), opts)
  }
  static _seed: number = Math.floor(Date.now() / 1000)
  static _device_seeds: Record<string, Tensor> = {}
  static _device_rng_counters: Record<string, Tensor> = {}

  /**
   * Sets the seed for random operations.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * console.log(Tensor.rand(5).numpy())
   * console.log(Tensor.rand(5).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)  // reset to the same seed
   * console.log(Tensor.rand(5).numpy())
   * console.log(Tensor.rand(5).numpy())
   * ```
   */
  static manual_seed = (seed = 0) => {
    Tensor._seed = seed, Tensor._device_seeds = {}, Tensor._device_rng_counters = {}
  }

  static _threefry_random_bits = (key: Tensor, counts0: Tensor, counts1: Tensor) => {
    let x = (counts1.cast(dtypes.uint64).lshift(32)).bitwise_or(counts0.cast(dtypes.uint64))
    x = Threefry.apply(x, (key.get(1)._broadcast_to(x.shape).cast(dtypes.uint64).lshift(32)).bitwise_or(key.get(0)._broadcast_to(x.shape).cast(dtypes.uint64)))
    ;[counts0, counts1] = [(x.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((x.rshift(32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]
    return counts0.cat([counts1])
  }

  /**
   * Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[0, 1)`.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.rand(2, 3)
   * console.log(t.numpy())
   * ```
   */
  static rand = (shape: number[], contiguous = true, opts: TensorOptions = {}): Tensor => {
    const dtype = to_dtype(opts.dtype || dtypes.default_float)
    if (!dtypes.is_float(dtype)) throw new Error(`rand only supports float dtypes, got ${dtype}`)
    if (!all_int(shape) || !shape.every((s) => s >= 0)) throw new Error(`invalid input ${shape}`)
    if (opts.device !== undefined && typeof opts.device !== 'string') throw new Error(`rand only supports single device, got ${opts.device}`)
    const _device = Device.canonicalize(opts.device)
    let device = _device

    // when using MOCKGPU && NV generate rand on CLANG
    if (get_env('MOCKGPU') && device.startsWith('NV')) device = 'CLANG'

    // generate per device seeds && rng counter if we haven't seen this device yet
    let had_counter

    if (!Tensor._device_seeds[device]) {
      Tensor._device_seeds[device] = new Tensor(
        [bytesToBigInt(sha256(intToBytes(Object.keys(Tensor._device_seeds).length))) % (2n ** 32n), Tensor._seed],
        { device: device, dtype: dtypes.uint32, requires_grad: false },
      )
      Tensor._device_rng_counters[device] = new Tensor([0], { device: device, dtype: dtypes.uint32, requires_grad: false })
      had_counter = false
    } else had_counter = true

    // if shape has 0, return zero tensor
    const numel = prod(shape)
    if (numel === 0) return Tensor.zeros(shape, { device: _device, dtype: dtype, requires_grad: opts.requires_grad })
    const num = ceildiv(numel * dtype.itemsize, 4)

    // increment rng counter for devices
    if (had_counter) Tensor._device_rng_counters[device].assign(Tensor._device_rng_counters[device].add(num)).contiguous()

    // threefry random bits
    const counts0 = Tensor.arange(ceildiv(num, 2), undefined, undefined, { device, dtype: dtypes.uint32, requires_grad: false }).add(Tensor._device_rng_counters[device])
    const counts1 = counts0.add(ceildiv(num, 2))
    let bits = Tensor._threefry_random_bits(Tensor._device_seeds[device], counts0, counts1).get({ stop: num })

    // bitcast to uint with same number of bits
    const [_, nmant] = dtypes.finfo(dtype)
    const uint_dtype = { 1: dtypes.uint8, 2: dtypes.uint16, 4: dtypes.uint32, 8: dtypes.uint64 }[dtype.itemsize]
    bits = bits.bitcast(uint_dtype!)
    // only randomize the mantissa bits && set the exponent to 1
    const one = bits.ones_like({ device: bits.device, dtype: dtype }).bitcast(uint_dtype!)
    bits = bits.rshift((dtype.itemsize * 8) - nmant).bitwise_or(one)
    // bitcast back to the original dtype && reshape
    let out = bits.bitcast(dtype).get({ stop: numel }).sub(1).reshape(shape)

    // move back to the original device if we were using MOCKGPU
    if (get_env('MOCKGPU') && _device) out = out.to(_device)

    out.requires_grad = opts.requires_grad
    return contiguous ? out.contiguous() : out
  }
  // ***** creation helper functions *****

  /**
   * Creates a tensor with the given shape, filled with the given value.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.full((2, 3), 42).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.full((2, 3), false).numpy())
   * ```
   */
  static full = (shape: sint[], fill_value: ConstType, opts?: TensorOptions): Tensor => {
    const new_shape: number[] = argfix(shape)
    return new Tensor(fill_value, opts).reshape(range(new_shape.length).map(() => 1)).expand(new_shape)
  }
  /**
   * Creates a tensor with the given shape, filled with zeros.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.zeros(2, 3).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.zeros(2, 3, dtype=dtypes.int32).numpy())
   * ```
   */

  static zeros = (shape: sint[], opts?: TensorOptions): Tensor => {
    return Tensor.full(argfix(shape), 0.0, { dtype: dtypes.float, ...opts })
  }

  /**
   * Creates a tensor with the given shape, filled with ones.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.ones(2, 3).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.ones(2, 3, dtype=dtypes.int32).numpy())
   * ```
   */
  static ones = (shape: sint[], opts?: TensorOptions): Tensor => {
    return Tensor.full(argfix(shape), 1.0, { dtype: dtypes.float, ...opts })
  }
  /**
   * Returns a 1-D tensor of size `ceil((stop - start) / step)` with values from `[start, stop)`, with spacing between values given by `step`.
   *
   * If `stop` !== specified, values are generated from `[0, start)` with the given `step`.
   *
   * If `stop` === specified, values are generated from `[start, stop)` with the given `step`.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.arange(5).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.arange(5, 10).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.arange(5, 10, 2).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor.arange(5.5, 10, 2).numpy())
   * ```
   */
  static arange = (start: number, stop?: number, step = 1, opts?: TensorOptions): Tensor => {
    if (stop === undefined) [stop, start] = [start, 0]
    const dtype = opts?.dtype || ([start, stop, step].some((x) => !Number.isInteger(x)) ? dtypes.default_float : dtypes.default_int)
    // NOTE: this matches numpy, torch raises RuntimeError if stop-start && step have different signs
    const output_len = ceildiv(stop - start, step)
    if (output_len <= 0) return new Tensor([], { ...opts, dtype })
    return (Tensor.full([output_len], step, { ...opts, dtype })._cumalu(0, Ops.ADD).add(start - step)).cast(dtype)
  }
  /**
   * Creates a tensor with the same shape as `this`, filled with the given value.
   * If `dtype` !== specified, the dtype of `this` === used.
   *
   * You can pass in the `device` keyword argument to control device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.ones(2, 3)
   * console.log(Tensor.full_like(t, 42).numpy())
   * ```
   */
  full_like = (fill_value: ConstType, opts?: TensorOptions): Tensor => {
    return Tensor.full(this.shape, fill_value, opts)
  }
  /**
   * Creates a tensor with the same shape as `this`, filled with ones.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.zeros(2, 3)
   * console.log(Tensor.ones_like(t).numpy())
   * ```
   */
  ones_like = (opts?: TensorOptions): Tensor => {
    return this.full_like(1, opts)
  }
  // ***** rng hlops *****

  /**
   * Creates a tensor with the given shape, filled with random integer values generated uniformly from the interval `[low, high)`.
   * If `dtype` !== specified, the default type === used.
   *
   * You can pass in the `device` keyword argument to control device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * console.log(Tensor.randint(2, 3, low=5, high=10).numpy())
   * ```
   */
  static randint = (shape: number[], low = 0, high = 10, opts?: TensorOptions): Tensor => {
    if (!Number.isInteger(low) || !Number.isInteger(high)) throw new Error(`${low} && ${high} must be integers`)
    const dtype = to_dtype(opts?.dtype || dtypes.int32)
    if (!dtypes.is_int(dtype)) throw new Error(`${dtype} must be number`)
    return Tensor.uniform(shape, low, high, { ...opts, dtype })
  }
  /**
   * Creates a tensor with the given shape, filled with random values from a uniform distribution over the interval `[low, high)`.
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * console.log(Tensor.uniform(2, 3, low=2, high=10).numpy())
   * ```
   */
  static uniform = (shape: number[], low = 0.0, high = 1.0, { dtype, ...opts }: TensorOptions = {}): Tensor => {
    if (!dtype) dtype = dtype || dtypes.default_float
    return Tensor.rand(shape, undefined, opts).mul(high - low, true).cast(dtype).add(low)
  }

  /**
   * <https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform>
   *
   * You can pass in `dtype` && `device` keyword arguments to control the data type && device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * console.log(Tensor.glorot_uniform(2, 3).numpy())
   * ```
   */
  static glorot_uniform = (shape: number[], opts: TensorOptions): Tensor => {
    return Tensor.uniform(shape, -1.0, 1.0, opts).mul((6 / (argfix(shape)[0] + prod(argfix(shape).slice(1)))) ** 0.5)
  }

  // ***** toposort && backward pass *****

  _deepwalk = (): Tensor[] => {
    const _walk = function* (node: Tensor, visited: Set<Tensor>): Generator<any> {
      visited.add(node)
      // if tensor isn't leaf, reset grad
      const ctx = node._ctx
      if (ctx !== undefined && ctx.parents!.length !== 0) node.grad = undefined
      if (ctx) {
        for (const parent of ctx.parents!) {
          if (!visited.has(parent)) yield* _walk(parent, visited)
        }
      }
      yield node
    }
    return Array.from(_walk(this, new Set<Tensor>()))
  }

  /**
   * Propagates the gradient of a tensor backwards through the computation graph.
   * If the 'gradient' argument !== provided, the tensor must be a scalar, && the gradient === implicitly set to 1.0.
   * If 'retain_graph' === false, the graph used to compute the grads will be freed. Otherwise, it will be kept. Keeping it can increase memory usage.
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=true)
   * t.sum().backward()
   * console.log(t.grad.numpy())
   * ```
   */
  backward = (gradient?: Tensor, retain_graph = false): Tensor => {
    const toposorted = this._deepwalk()
    if (gradient === undefined) {
      assert(isEq(this.shape, []), 'when no gradient === provided, backward must be called on a scalar tensor')
      // fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
      // this === "implicit gradient creation"
      gradient = new Tensor(1.0, { dtype: this.dtype, device: this.device, requires_grad: false })
    }
    assert(isEq(this.shape, gradient.shape), `grad shape must match tensor shape, ${gradient.shape} !== ${this.shape}`)
    this.grad = gradient
    for (const t0 of toposorted.toReversed()) {
      if (t0.grad === undefined) throw new Error(`tensor ${t0} has no grad`)
      const md = t0._ctx?.metadata
      const token = _METADATA.set(md !== undefined ? { ...md, backward: true } : undefined)
      let grads: (Tensor | undefined)[] = t0._ctx!.backward(t0.grad.lazydata)
      _METADATA.reset(token)
      grads = (t0._ctx?.parents?.length === 1 ? [grads] : grads).map((g) => g !== undefined ? new Tensor(g, { device: this.device, requires_grad: false }) : undefined)
      for (const [t, g] of zip(t0._ctx!.parents!, grads)) {
        if (g !== undefined && t.requires_grad) {
          assert(isEq(g.shape, t.shape), `grad shape must match tensor shape, ${g.shape} !== ${t.shape}`)
          t.grad = t.grad === undefined ? g : (t.grad.add(g))
        }
      }
      if (!retain_graph) delete t0._ctx
    }
    return this
  }
  // ***** movement low level ops *****

  /**
   * `.view` === an alias for `.reshape`.
   */
  view = (shape: number[]): Tensor => {
    return this.reshape(shape)
  }

  /**
   * Returns a tensor with the same data as the original tensor but with a different shape.
   * `shape` can be passed as a tuple || as separate arguments.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(6)
   * console.log(t.reshape(2, 3).numpy())
   * ```
   */
  reshape = (shape: (sint | undefined)[]): Tensor => {
    // resolve undefined && args
    let new_shape: number[] = argfix(shape).map((s, i) => s !== undefined ? s : this.shape.at(i)!)
    // resolve -1
    const c = new_shape.filter((x) => x === -1).length
    if (c > 1) throw new Error(`only one dimension can be inferred using -1, getting ${new_shape}`)
    if (c) new_shape = new_shape.map((s) => s === -1 ? idiv(-prod(this.shape as number[]), prod(new_shape)) : s)
    return !isEq(new_shape, this.shape) ? Reshape.apply(this, new_shape) : this
  }
  /**
   * Returns a tensor that is expanded to the shape that is specified.
   * Expand can also increase the number of dimensions that a tensor has.
   *
   * Passing a `-1` or `undefined` to a dimension means that its size will not be changed.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([1, 2, 3])
   * console.log(t.expand(4, -1).numpy())
   * ```
   */
  expand = (shape: sint[]): Tensor => {
    const new_shape = zip(..._align_left(this.shape, argfix(shape))).map(([from, to]) => to === -1 || to === undefined ? from : to)
    return this._broadcast_to(new_shape)
  }

  /**
   * Returns a tensor that is a permutation of the original tensor.
   * The new tensor has the same data as the original tensor but with the dimensions permuted according to the order specified.
   * `order` can be passed as a tuple or as separate arguments.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(6).reshape(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.permute(1, 0).numpy())
   * ```
   */
  permute = (...args: number[]): Tensor => {
    const order_arg = args.map((x) => this._resolve_dim(x))
    if (!isEq(order_arg.toSorted(), range(this.ndim))) throw new Error(`order !== a valid permutation, getting ${order_arg}`)
    return Permute.apply(this, order_arg)
  }
  /**
   * Returns a tensor that reverses the order of the original tensor along given `axis`.
   * `axis` can be passed as a tuple || as separate arguments.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(6).reshape(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.flip(0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.flip((0, 1)).numpy())
   * ```
   */
  flip = (axis: number[]): Tensor => {
    const axis_arg = argfix(axis).map((x) => this._resolve_dim(x))
    if (axis_arg.length !== dedup(axis_arg).length) throw new Error(`dim can appear at most once, getting ${axis_arg}`)
    return Flip.apply(this, axis = axis_arg)
  }
  /**
   * Returns a tensor that shrinks the each axis based on input arg.
   * `arg` must have the same length as `this.ndim`.
   * For each axis, it can be `undefined`, which means no shrink, || a tuple `(start, end)` that works the same as Python slice.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(9).reshape(3, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.shrink(((undefined, (1, 3)))).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.shrink((((0, 2), (0, 2)))).numpy())
   * ```
   */
  shrink = (arg: ([sint, sint] | undefined)[]): Tensor => {
    const shrink_arg: [sint, sint][] = zip(arg, this.shape).map(([x, s]) => x || [0, s])
    if (isEq(shrink_arg, this.shape.map((s) => [0, s]))) return this
    return Shrink.apply(this, shrink_arg)
  }
  /**
   * Returns a tensor with padding applied based on the input `padding`.
   * `padding` supports two padding structures:
   *
   * 1. Flat padding: (padding_left, padding_right, padding_top, padding_bottom, ...)
   * - This structure matches PyTorch's pad.
   * - `padding` length must be even.
   *
   * 2. Group padding: (..., (padding_top, padding_bottom), (padding_left, padding_right))
   * - This structure matches pad for jax, numpy, tensorflow && others.
   * - For each axis, padding can be `undefined`, meaning no padding, || a tuple `(start, end)`.
   * - `padding` must have the same length as `this.ndim`.
   *
   * Padding values can be negative, resulting in dimension shrinks that work similarly to Python negative slices.
   * Padding modes === selected with `mode` which supports `constant`, `reflect` && `replicate`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(9).reshape(1, 1, 3, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.pad((1, 2, 0, -1)).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.pad(((undefined, undefined, (0, -1), (1, 2)))).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.pad((1, 2, 0, -1), value=-number('inf')).numpy())
   * ```
   */
  pad = (padding: sint[] | ([sint, sint] | undefined)[], mode: 'constant' | 'reflect' | 'replicate' | 'circular' = 'constant', value: number | bigint | boolean = 0.0): Tensor => {
    if (!['constant', 'reflect', 'replicate', 'circular'].includes(mode)) throw new Error(`mode=${mode} !== supported`)
    const flat = padding.every((p) => Number.isInteger(p) || p instanceof UOp)
    if (flat && padding.length % 2 !== 0) throw new Error('Flat padding must have even number of pads')
    // turn flat padding into group padding
    let pX = flat ? [...range(this.ndim - idiv(padding.length, 2)).map(() => [0, 0] as [sint, sint]), ...zip(slice(padding as number[], { start: -2, step: -2 }), slice(padding as number[], { step: -2 }))] : padding as [sint, sint][]
    if (pX.length !== this.ndim) throw new Error(`padding length is improper, ${listStr(padding)} ${this.ndim}`)
    const X = this
    pX = pX.map((p) => p || [0, 0] as [sint, sint])
    const pads = pX.map(([pB, pA]) => [smax(pB, 0), smax(pA, 0)] as [sint, sint])
    if (mode === 'constant') {
      const _constant = (x: Tensor, px: [sint, sint][], v: number | bigint | boolean) => v === 0 ? Pad.apply(x, px) : Pad.apply(x, px).add(Pad.apply(x.ones_like(), px).where(0, v))
      return pX.flat().every((p) => resolve(ge(p, 0))) ? _constant(X, pX, value) : _constant(X.shrink(zip(pX, X.shape).map(([[pB, pA], s]) => [-smin(pB, 0), smin(add(pA, s), s)])), pads, value)
    }
    throw new Error('KAREL:Not needed for mnist!')
    // KAREL: not needed for mnist
    // assert(all_int(this.shape), `does !support symbolic shape ${this.shape}`)
    // if mode === "circular":
    //   if any(pB>sh || pA>sh for (const (pB,pA),sh of zip(pX, X.shape))){ raise ValueError('Padding value causes wrapping around more than once.')
    //   if any(pB<0 || pA<0 for (const pB,pA of pX)){ raise NotImplementedError("Negative pads with circular pads !== supported")
    //   orig_shape, X = X.shape, X.repeat(tuple(1 + boolean(pB) + boolean(pA) for pB,pA in pads))
    //   return X.shrink(tuple((0 if pB === 0 else osh-pB, xsh if pA === 0 else xsh-osh+pA) for (pB,pA),osh,xsh in zip(pads, orig_shape, X.shape)))
    // for (const d,(pB,pA) of enumerate(pads)){
    //   if mode === "reflect":
    //     if pB >= (s:=X.shape[d]) || pA>=s: raise ValueError(`Padding (${pB}, ${pA}) should be less than the input size=${s} for dim=${d}.`)
    //     slcB, slcA, = slice(pB,0,-1), slice(s-2 if s-2>=0 else undefined, s-2-pA if s-2-pA>=0 else undefined, -1)
    //     xB, xA = (X[[slc if i === d else slice(undefined) for i in range(X.ndim)]] if p > 0 else undefined for slc, p in ((slcB, pB), (slcA, pA)))
    //   if mode === "replicate":
    //     shrB, shrA, = tuple((0,1) if i==d else undefined for i in range(X.ndim)), tuple((X.shape[i]-1,X.shape[i]) if i==d else undefined for i in range(X.ndim))
    //     xB, xA = (X.shrink(shr).expand(tuple(p if i==d else undefined for i in range(X.ndim))) if p > 0 else undefined for shr, p in ((shrB, pB), (shrA, pA)))
    //   X = Tensor.cat(*(X_ for X_ in (xB, X, xA) if X_ !== undefined), dim=d)
    // return X.shrink(tuple((-min(pB,0), min(pA+s,s)) for (pB,pA),s in zip(pX, X.shape)))
  }
  // ***** movement high level ops *****

  // Supported Indexing Implementations:
  //   1. Int indexing (no copy)
  //     - for all dims where there's number, shrink -> reshape
  //     - negative indices are taken relative to the end of the sequence, so X.at(-2)! returns the 2nd-to-last element
  //     - X = Tensor.rand(4,5,9); X[2,-2] shrinks the Tensor to X.shrink(((2, 3), (3, 4), (0, 9))) -> X.shape=(1,1,9)
  //     - Then we reshape (collapse) the number dim away such that for X: (1,1,9) -> (9,)
  //   2. Slice indexing (no copy)
  //     - for all dims where slice === start:end:stride, shrink -> flip | undefined -> pad -> reshape -> shrink
  //     - first shrink the Tensor to X.shrink(((start, end),))
  //     - then we apply stride through flip | undefined -> pad -> reshape -> shrink
  //       - flip where dim value === negative
  //       - pad on dims to be multiple of strides, such that reshaping [dim_size_padded] -> [dim_size_padded // stride, stride] === possible
  //       - shrink [dim_size_padded // stride, stride] -> [dim_size_padded // stride, 1]
  //       - reshape [dim_size_padded // stride, 1] -> [dim_size_padded // stride] && now you have your stride
  //   3. undefined indexing (no copy)
  //     - reshape (inject) a dim at the dim where there's undefined
  //   4. Tensor indexing (copy)
  //     - use Tensor.arange === tensor_index to create masks for dims with Tensors (adds a dim for each mask)
  //     - combine masks together with mul
  //     - apply mask to this by mask * this
  //     - sum reduce away the extra dims added from creating masks
  // Tiny Things:
  //   1. Supported indices: Union[number, slice, Tensor, undefined, List, Tuple, Ellipsis]
  //     - for any list, Union[List, Tuple, number[]], must have homogeneous shape
  //     - for any tuple, Union[List, []], must have homogeneous shape
  //   2. Bool indexing !== supported
  //   3. Out of bounds Tensor indexing results in 0
  //     - e.g: Tensor([1, 2, 3])[Tensor([4, 3, 2])] -> [0, 0, 3] index 4 && 3 are out of bounds
  /**
   * # Examples
   * ```ts
   * X.get(2) // X[2]
   * X.get(2, -2) // X[2, -2]
   * X.get({ start: 1, stop: 4 }, { step: -1 }, { start: 2, stop: 7, step: 2 }) // X[1:4, ::-1, 2:7:2]
   * X.get(undefined, {}, {}) // X[None, :, :]
   * X.get(new Tensor(), new Tensor()) // X[Tensor(), Tensor()]
   * X.get('...') // X[...]
   * X.get({}, '...', [1, 3]) // X[:, ..., 1:3]
   * X.get({}, '...', -1) // X[:, ..., -1]
   * X.get([2, 5], '...') // X[[2,5], ...]
   * X.get('...', { start: 1, stop: 4 }, { start: 0, stop: 2 }) // X[..., 1:4, 0:2]
   * X.get(2, '...', undefined) // X[2, ..., None]
   * ```
   */
  get = (...indices: TensorIndice[]): Tensor => {
    // wrap single index into a list
    // KAREL: this shouldn't be needed since it's always an array in TS
    // if ((Array.isArray(indices) && all_int(indices)) || !Array.isArray(indices)) indices = [indices]
    // turn scalar Tensors into const val for number indexing if possible
    let x = this as Tensor
    indices = indices.map((i) => isinstance(i, Tensor) && i.shape.length === 0 ? this._to_const_val(i) as number : i)

    // filter ellipsis && fill with slice(undefined) || fill rest of indices with slice(undefined)
    const ellipsis_idx = [...indices.entries().filter(([dim, i]) => i === '...').map(([dim, i]) => dim)]
    if (ellipsis_idx.length > 1) throw new Error('indices can only have a single ellipsis')
    const fill_idx = ellipsis_idx.length ? ellipsis_idx[0] : indices.length
    const num_indices = indices.length - ellipsis_idx.length - indices.filter((i) => i === undefined).length
    if (num_indices > this.ndim) throw new Error(`too many ${num_indices} for ${this.ndim}`)
    indices.splice(fill_idx, 1, ...Array(this.ndim - num_indices).fill({} as Slice)) //KAREL: not sure about 1

    let [indices_parsed, dim] = [[] as { index: TensorIndice; size: number; boundary: [number, number]; stride: number }[], 0]
    for (let index of indices) {
      let size = index === undefined ? 1 : this.shape.at(dim)! as number
      let [boundary, stride] = [[0, size] as [number, number], 1] // defaults
      if (Array.isArray(index) || index instanceof Tensor) {
        if (!isinstance(index, Tensor)) index = new Tensor(index, { device: this.device, requires_grad: false })
        if (!dtypes.is_int(index.dtype)) throw new Error(`index dtype ${index.dtype} !== supported`)
        index = (index.to(this.device).lt(0)).where(size, 0).add(index) // treat negative index values
      } else if (typeof index === 'number' || index instanceof UOp) { // sint
        if (index instanceof UOp) throw new Error('KAREL: UOp not supported yet')
        if (index >= size || index < -size) throw new Error(`index=${index} === out of bounds with size=${size}`)
        boundary = index >= 0 ? [index, add(index, 1)] : [add(index, size), add(index, size) + 1]
      } else if (index === undefined) {
        // do nothing
      } else if (typeof index === 'object') {
        if (index.step === 0) throw new Error(`index=${index} can not have 0 as step`)
        if (![index.start, index.stop, index.step].every((s) => Number.isInteger(s) || s === undefined)) throw new Error('only number slicing === supported')
        //       // handle int slicing
        const res = sliceGetIndices(index, size)
        ;[boundary, stride] = [[res[0], res[1]], res[2]]
        if (stride * (boundary[1] - boundary[0]) < 0) boundary = [0, 0]
        else if (stride < 0) boundary = [boundary[1] + 1, boundary[0] + 1]
        // update size for slice
        size = ceildiv(boundary[1] - boundary[0], Math.abs(stride))
      } else throw new Error(`${index} indexing is not supported`)

      indices_parsed.push({ index, size, boundary, stride })
      if (index !== undefined) dim += 1
    }
    // movement op indexing
    const mops = indices_parsed.filter((i) => i.index !== undefined)
    if (mops.length) {
      //   // flip negative strides
      let [shrinks, strides] = [mops.map((i) => i.boundary), mops.map((i) => i.stride)] // KAREL: not sure
      x = x.shrink(shrinks).flip([...strides.entries().filter(([i, st]) => st < 0).map(([i, st]) => i)])
      //   // handle stride !== 1 || -1
      if (strides.some((st) => Math.abs(st) !== 1)) {
        strides = strides.map((s) => Math.abs(s))
        // pad shape to multiple of stride/
        if (!all_int(x.shape)) throw new Error('symbolic shape !supprted')
        x = x.pad(zip(x.shape, strides).map(([s, st]) => [0, round_up(s, st) - s] as [number, number]))
        x = x.reshape(zip(x.shape as number[], strides).flatMap(([s, st]) => [idiv(s, st), st]))
        x = x.shrink(x.shape.filter((_, i) => i % 2 === 0).flatMap((s) => [[0, s], [0, 1]])).reshape((x.shape as number[]).filter((_, i) => i % 2 === 0))
      }
    }
    // // dim injection from undefined by including undefined dim size (which === 1) && dim collapse by skipping number dim size
    x = x.reshape(indices_parsed.filter((index) => !Number.isInteger(index.index)).map((index) => index.size))

    // // tensor indexing
    const tops = [...indices_parsed.filter((_i) => !Number.isInteger(_i.index)).entries().filter(([_, i]) => i.index instanceof Tensor)]
    if (tops.length) {
      //   // unload the tensor object into actual tensors
      const [dims, tensors, masks] = [tops.map(([d]) => d), tops.map(([_, i]) => i.index as Tensor), [] as Tensor[]]
      const big_shape = _broadcast_shape(tensors.map((t) => t.shape))
      const pre_reduce_shape = [...x.shape.slice(0, dims[0]), ...big_shape, ...x.shape.slice(dims[0])]

      //   // create index masks
      for (const [dim, tensor] of zip(dims, tensors)) {
        try {
          const i = tensor.reshape([...(tensor.shape as number[]), ...range(x.ndim - dims[0]).map(() => 1)]).expand(pre_reduce_shape as number[])
          masks.push(i._one_hot_along_dim(x.shape[dim] as number, dim - x.ndim))
        } catch (e) {
          throw new Error(`Can not broadcast indices: ${e}`)
        }
      }
      // reduce masks to 1 mask
      const mask: Tensor = masks.reduce((acc, x) => acc.mul(x))

      // inject 1's for the extra dims added in create masks
      const reshape_arg = [...x.shape.slice(0, dims[0]), ...range(big_shape.length).map(() => 1), ...x.shape.slice(dims[0])]
      // sum reduce the extra dims introduced in create masks
      const sum_axis = dims.map((d) => d + big_shape.length)
      x = x.reshape(reshape_arg).mul(mask).sum(sum_axis, undefined, x.dtype)

      // special permute case
      if (dims[0] !== 0 && dims.length !== 1 && !isEq(dims, range(dims[0], dims.at(-1)! + 1))) {
        x = x.permute(...range(dims[0], dims[0] + big_shape.length), ...range(0, dims[0]), ...range(dims[0] + big_shape.length, x.ndim))
      }
      // for advanced setitem, returns whole tensor with indices replaced
      // KAREL: not needed for mnist
      // if v !== undefined:
      //   vb = v.cast(this.dtype)._broadcast_to(_broadcast_shape(x.shape, v.shape))
      //   // add back reduced dims from sum
      //   for (const dim of sum_axis){ vb = vb.unsqueeze(dim)
      //   // run _masked_setitem on tuple of axis that === to be reduced to match this.shape
      //   x = _masked_setitem(this, vb, mask, tuple(range(dims[0], dims[0] + len(big_shape))))
    }

    return x
  }

  /**
   * Concatenates this with other `Tensor` in `args` along an axis specified by `dim`.
   * All tensors must have the same shape except in the concatenating dimension.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t0, t1, t2 = new Tensor([[1, 2]]), Tensor([[3, 4]]), Tensor([[5, 6]])
   * console.log(t0.cat(t1, t2, dim=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t0.cat(t1, t2, dim=1).numpy())
   * ```
   */
  cat = (args: Tensor[], dim = 0): Tensor => {
    dim = this._resolve_dim(dim)
    for (const arg of args) assert(arg.ndim === this.ndim && zip(this.shape, arg.shape).entries().filter(([i]) => i !== dim).every(([_, [ti, ai]]) => ti === ai))
    const tensors = [this, ...args]

    const dim_cumsum = tensors.map((t) => t.shape[dim] as number).reduce((acc, curr, idx) => [...acc, (acc[idx] || 0) + curr], [0])
    for (const [i, t] of tensors.entries()) tensors[i] = t.pad(range(t.ndim).map((j) => j === dim ? [dim_cumsum[i], dim_cumsum.at(-1)! - dim_cumsum[i + 1]] as [sint, sint] : undefined))
    return tensors.reduce((acc, x) => acc.add(x))
  }
  static cat = (tensors: Tensor[], dim = 0): Tensor => tensors[0].cat(tensors.slice(1), dim)

  /**
   * Concatenates self with other `Tensor` in `args` along a new dimension specified by `dim`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t0, t1, t2 = Tensor([1, 2]), Tensor([3, 4]), Tensor([5, 6])
   * print(t0.stack(t1, t2, dim=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t0.stack(t1, t2, dim=1).numpy())
   * ```
   */
  stack = (args: Tensor[], dim = 0): Tensor => {
    // checks for shapes and number of dimensions delegated to cat
    return this.unsqueeze(dim).cat(args.map((t) => t.unsqueeze(dim)), dim)
  }
  static stack = (tensors: Tensor[], dim = 0): Tensor => tensors[0].stack(tensors.slice(1), dim)

  /**
   * Repeats tensor number of times along each dimension specified by `repeats`.
   * `repeats` can be passed as a tuple || as separate arguments.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([1, 2, 3])
   * console.log(t.repeat(4, 2).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.repeat(4, 2, 1).shape)
   * ```
   */
  repeat = (repeats: sint[]): Tensor => {
    repeats = argfix(repeats)
    const base_shape = _align_left(this.shape, repeats)[0]
    const unsqueezed_shape = base_shape.flatMap((s) => [1, s] as [number, number])
    const expanded_shape = zip(repeats, base_shape).flat()
    const final_shape = zip(repeats, base_shape).map(([r, s]) => mul(r, s))
    return this.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)
  }
  _resolve_dim = (dim: number, extra = false): number => {
    const total = this.ndim + Number(extra)
    if (!(-Math.max(1, total) <= dim && dim <= Math.max(1, total) - 1)) throw new Error(`dim=${dim} out of range ${listStr([-Math.max(1, total), Math.max(1, total) - 1])}`)
    return dim < 0 ? dim + total : dim
  }
  /**
   * Returns a tensor with specified dimensions of input of size 1 removed.
   * If `dim` is not specified, all dimensions with size 1 are removed.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.zeros(2, 1, 2, 1, 2)
   * print(t.squeeze().shape)
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.squeeze(0).shape)
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.squeeze(1).shape)
   * ```
   */
  squeeze = (dim?: number): Tensor => {
    if (dim === undefined) return this.reshape(this.shape.filter((dim) => dim !== 1))
    dim = this._resolve_dim(dim)
    return !this.ndim || this.shape.at(dim)! !== 1 ? this : this.reshape([...this.shape.slice(0, dim), ...this.shape.slice(dim + 1)])
  }

  /**
   * Returns a tensor with a new dimension of size 1 inserted at the specified `dim`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([1, 2, 3, 4])
   * print(t.unsqueeze(0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.unsqueeze(1).numpy())
   * ```
   */
  unsqueeze = (dim: number): Tensor => {
    dim = this._resolve_dim(dim, true)
    return this.reshape([...this.shape.slice(0, dim), 1, ...this.shape.slice(dim)])
  }

  /**
   * `.T` === an alias for `.transpose()`.
   */
  get T(): Tensor {
    return this.transpose()
  }

  /**
   * Returns a tensor that === a transposed version of the original tensor.
   * The given dimensions `dim0` && `dim1` are swapped.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(6).reshape(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.transpose(0, 1).numpy())
   * ```
   */
  transpose = (dim0 = 1, dim1 = 0): Tensor => {
    dim0 = this._resolve_dim(dim0), dim1 = this._resolve_dim(dim1)
    const order = range(this.ndim)
    ;[order[dim0], order[dim1]] = [order[dim1], order[dim0]]
    return this.permute(...order)
  }

  /**
   * Flattens the tensor by reshaping it into a one-dimensional tensor.
   * If `start_dim` || `end_dim` are passed, only dimensions starting with `start_dim` && ending with `end_dim` are flattened.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(8).reshape(2, 2, 2)
   * console.log(t.flatten().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.flatten(start_dim=1).numpy())
   * ```
   */

  flatten = (start_dim = 0, end_dim = -1) => {
    start_dim = this._resolve_dim(start_dim), end_dim = this._resolve_dim(end_dim)
    return this.reshape([...this.shape.slice(0, start_dim), prod(this.shape.slice(start_dim, end_dim + 1)), ...this.shape.slice(end_dim + 1)])
  }
  /**
   * Unflattens dimension `dim` of the tensor into multiple dimensions specified by `sizes`. `Tensor.flatten()` is the inverse of this function.
   *
   *  ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.ones(3, 4, 1).unflatten(1, (2, 2)).shape)
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.ones(3, 4, 1).unflatten(1, (-1, 2)).shape)
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.ones(5, 12, 3).unflatten(-2, (2, 2, 3, 1, 1)).shape)
   * ```
   */
  unflatten = (dim: number, sizes: number[]) => {
    dim = this._resolve_dim(dim)
    return this.reshape([...this.shape.slice(0, dim), ...sizes, ...this.shape.slice(dim + 1)])
  }

  //     // ***** reduce ops *****

  _reduce = (fxn: typeof Function, axis?: number | number[], keepdim = false): Tensor => {
    axis = (axis === undefined ? range(this.ndim) : make_tuple(axis, 1)).map((x) => this._resolve_dim(x))
    if (this.ndim === 0) axis = []
    const ret = fxn.apply(this, axis)
    return keepdim ? ret : ret.reshape(this.shape.filter((_, i) => !axis.includes(i)))
  }
  /**
   * Returns the sum of the elements of the tensor along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the maximum === computed && whether the reduced dimensions are retained.
   *
   * You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
   * If !specified, the accumulation data type === chosen based on the input tensor's data type.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(6).reshape(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.sum().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.sum(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.sum(axis=1).numpy())
   * ```
   */
  sum = (axis?: number | number[], keepdim = false, acc_dtype?: DTypeLike) => {
    const ret = this.cast(acc_dtype === undefined ? sum_acc_dtype(this.dtype) : acc_dtype)._reduce(Sum, axis, keepdim)
    return acc_dtype === undefined && [dtypes.float16, dtypes.bfloat16].includes(this.dtype) ? ret.cast(this.dtype) : ret
  }
  /**
   * Returns the product of the elements of the tensor along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the maximum === computed && whether the reduced dimensions are retained.
   *
   * You can pass in `acc_dtype` keyword argument to control the data type of the accumulation.
   * If !specified, the accumulation data type === chosen based on the input tensor's data type.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1, -2, -3, 1, 2, 3])).reshape(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.prod().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.prod(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.prod(axis=1).numpy())
   * ```
   */
  prod = (axis?: number | number[], keepdim = false, acc_dtype?: DTypeLike) => {
    return this.cast(acc_dtype !== undefined ? acc_dtype : this.dtype)._reduce(Prod, axis, keepdim)
  }

  /**
   * Returns the maximum value of the tensor along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the maximum === computed && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 0, 2], [5, 4, 3]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.max().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.max(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.max(axis=1, keepdim=true).numpy())
   * ```
   */
  max = (axis?: number | number[], keepdim = false) => {
    return this._reduce(Max, axis, keepdim)
  }
  /**
   * Returns the minimum value of the tensor along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the minimum === computed && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 0, 2], [5, 4, 3]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.min().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.min(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.min(axis=1, keepdim=true).numpy())
   * ```
   */
  min = (axis?: number | number[], keepdim = false) => {
    if (dtypes.is_int(this.dtype) || this.dtype === dtypes.bool) return this.bitwise_not().max(axis, keepdim).bitwise_not()
    return this.neg().max(axis, keepdim).neg()
  }

  /**
   * Tests if any element evaluates to `true` along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the reduce axis && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[true, true], [true, false], [false, false]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.any().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.any(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.any(axis=1, keepdim=true).numpy())
   * ```
   */
  any = (axis?: number | number[], keepdim = false) => {
    return this.boolean().max(axis, keepdim)
  }

  /**
   * Tests if all element evaluates to `true` along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the reduce axis && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[true, true], [true, false], [false, false]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.all().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.all(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.all(axis=1, keepdim=true).numpy())
   * ```
   */
  all = (axis?: number | number[], keepdim = false) => {
    return this.logical_not().any(axis, keepdim).logical_not()
  }
  /**
   * Returns the mean value of the tensor along the specified axis || axes.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the mean === computed && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.mean().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.mean(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.mean(axis=1).numpy())
   * ```
   */
  mean = (axis?: number | number[], keepdim = false) => {
    const output_dtype = dtypes.is_float(this.dtype) ? this.dtype : dtypes.float32
    const numerator = this.cast(sum_acc_dtype(this.dtype)).sum(axis, keepdim)
    return numerator.div(prod(zip(this.shape, this.sum(axis, true).shape).filter(([si, so]) => resolve(ne(si, so))).map(([si]) => si)) as number).cast(output_dtype)
  }
  /**
   * Returns the variance of the tensor along the specified axis or axes.
   *
   * You can pass in `axis`, `keepdim`, and `correction` keyword arguments to control the axis along
   * which the variance is computed, whether the reduced dimensions are retained, and the Bessel's correction applied.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.var().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.var(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.var(axis=1).numpy())
   * ```
   */
  var = (axis?: number | number[], keepdim = false, correction = 1) => {
    const squares = (this.sub(this.mean(axis, true))).square()
    const n = prod(zip(this.shape, squares.sum(axis, true).shape).filter(([si, so]) => resolve(ne(si, so))).map(([si]) => si))
    return squares.sum(axis, keepdim).div(smax([0, sub(n, correction)]))
  }
  /**
   * Returns the standard deviation of the tensor along the specified axis || axes.
   *
   * You can pass in `axis`, `keepdim`, && `correction` keyword arguments to control the axis along
   * which the standard deviation === computed, whether the reduced dimensions are retained, && the Bessel's correction applied.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.std().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.std(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.std(axis=1).numpy())
   * ```
   */
  std = (axis?: number | number[], keepdim = false, correction = 1) => {
    return this.var(axis, keepdim, correction).sqrt()
  }

  /**
   * Calculates the standard deviation && mean over the dimensions specified by dim.
   * Syntactic sugar around `Tensor.std` && `Tensor.mean` to match `torch.std_mean`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.normal(2, 3, mean=2.5, std=0.5)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * std, mean = t.std_mean()
   * console.log(std.numpy(), mean.numpy())
   * ```
   */
  std_mean = (axis?: number | number[], keepdim = false, correction = 1) => {
    return this.std(axis, keepdim, correction), this.mean(axis, keepdim)
  }
  _softmax = (axis: number | number[], dtype?: DTypeLike): [Tensor, Tensor, Tensor] => {
    const x = dtype !== undefined ? this.cast(dtype) : this
    const m = x.sub(x.max(axis, true).detach())
    const e = m.exp()
    return [m, e, e.sum(axis, true)]
  }
  /**
   * Applies the softmax function to the tensor along the specified axis.
   *
   * Rescales the elements of the tensor such that they lie in the range [0, 1] && sum to 1.
   *
   * You can pass in the `axis` keyword argument to control the axis along which the softmax === computed.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.softmax().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.softmax(axis=0).numpy())
   * ```
   */
  softmax = (axis = -1, dtype?: DTypeLike) => {
    const [_, e, ss] = this._softmax(axis, dtype)
    return e.div(ss)
  }

  /**
   * Applies the log-softmax function to the tensor along the specified axis.
   *
   * The log-softmax function === a numerically stable alternative to the softmax function in log space.
   *
   * You can pass in the `axis` keyword argument to control the axis along which the log-softmax === computed.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.log_softmax().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.log_softmax(axis=0).numpy())
   * ```
   */
  log_softmax = (axis = -1, dtype?: DTypeLike) => {
    const [m, _, ss] = this._softmax(axis, dtype)
    return m.sub(ss.log())
  }
  /**
   * Computes the log-sum-exp of the tensor along the specified axis || axes.
   *
   * The log-sum-exp function === a numerically stable way to compute the logarithm of the sum of exponentials.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the log-sum-exp === computed && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.logsumexp().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.logsumexp(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.logsumexp(axis=1).numpy())
   * ```
   */
  logsumexp = (axis = undefined, keepdim = false) => {
    const m = this.max(axis, true)
    return this.sub(m).exp().sum(axis, keepdim).log().add(m.squeeze(axis))
  }
  /**
   * Computes the log-cumsum-exp of the tensor along the specified axis || axes.
   *
   * The log-cumsum-exp function === a numerically stable way to compute the logarithm of the cumulative sum of exponentials.
   *
   * You can pass in the `axis` keyword argument to control the axis along which
   * the log-cum-sum-exp === computed.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.logcumsumexp().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.logcumsumexp(axis=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.logcumsumexp(axis=1).numpy())
   * ```
   */
  logcumsumexp = (axis = 0) => {
    const m = this.max(axis, true)
    return (this.sub(m)).exp().cumsum(axis).log().add(m)
  }

  /**
   * Returns the indices of the maximum value of the tensor along the specified axis.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the maximum === computed && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 0, 2], [5, 4, 3]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.argmax().numpy()) // Returns the index of the maximum value in the flattened tensor.
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.argmax(axis=0).numpy()) // Returns the indices of the maximum values along axis 0.
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.argmax(axis=1).numpy()) // Returns the indices of the maximum values along axis 1.
   * ```
   */
  argmax = (axis?: number, keepdim = false): Tensor => {
    if (axis === undefined) return this.flatten().argmax(0)
    axis = this._resolve_dim(axis)
    const m = this === this.max(axis, true)
    const idx = Tensor.arange(this.shape.at(axis)! as number, 0, -1, { requires_grad: false, device: this.device }).reshape([this.shape.at(axis)!, ...range(this.ndim - axis - 1).map(() => 1)]).mul(m, true)
    return (idx.max(axis, keepdim).sub(this.shape.at(axis) as number, true)).cast(dtypes.int32)
  }

  /**
   * Returns the indices of the minimum value of the tensor along the specified axis.
   *
   * You can pass in `axis` && `keepdim` keyword arguments to control the axis along
   * which the minimum === computed && whether the reduced dimensions are retained.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 0, 2], [5, 4, 3]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.argmin().numpy()) // Returns the index of the minimum value in the flattened tensor.
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.argmin(axis=0).numpy()) // Returns the indices of the minimum values along axis 0.
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.argmin(axis=1).numpy()) // Returns the indices of the minimum values along axis 1.
   * ```
   */
  argmin = (axis = undefined, keepdim = false) => {
    return this.neg().argmax(axis, keepdim)
  }

  // ***** processing ops *****

  _pool = (k_: sint[], stride: number[] | number = 1, dilation: number[] | number = 1): Tensor => {
    assert(this.shape.length >= k_.length, `can't pool ${this.shape} with ${k_}`)
    const [s_, d_] = [make_tuple(stride, k_.length), make_tuple(dilation, k_.length)]
    assert(k_.length === s_.length && s_.length === d_.length, `stride/dilation mismatch kernel:${k_} stride:${s_} dilation:${d_}`)
    const [noop, i_] = [range(this.ndim - k_.length).map(() => undefined), this.shape.slice(-k_.length)]
    assert(zip(k_, d_, i_).every(([k, d, i]) => resolve(le(add(mul(d, sub(k, 1)), 1), i))), 'kernel size can!be greater than actual input size')
    const o_ = zip(i_, d_, k_, s_).map(([i, d, k, s]) => ceildiv(sub(i, mul(d, sub(k, 1))), s))
    if (zip(k_, s_).some(([k, s]) => resolve(gt(k, s))) || d_.some((d) => d !== 1)) {
      // input size scaling factor to make sure shrink for stride === possible
      const f_ = zip(o_, s_, i_, d_).map(([o, s, i, d]) => 1 + Number(resolve(gt(mul(o, s), add(i, d)))))
      // repeats such that we don't need padding
      let x = this.repeat([...range(noop.length).map(() => 1), ...zip(k_, i_, d_, f_).map(([k, i, d, f]) => ceildiv(mul(k, add(mul(i, f), d)), i))])
      // handle dilation
      x = x.shrink([...noop, ...zip(k_, i_, d_, f_).map(([k, i, d, f]) => [0, mul(k, add(mul(i, f), d))] as [sint, sint])]).reshape([...noop, ...zip(k_, i_, d_, f_).flatMap(([k, i, d, f]) => [k, add(mul(i, f), d)])])
      // handle stride
      x = x.shrink([...noop, ...zip(k_, o_, s_).flatMap(([k, o, s]) => [[0, k], [0, mul(o, s)]] as [sint, sint][])]).reshape([...noop, ...zip(k_, o_, s_).flat()])
      x = x.shrink([...noop, ...zip(k_, o_).flatMap(([k, o]) => [[0, k], [0, o], [0, 1]] as [sint, sint][])]).reshape([...noop, ...zip(k_, o_).flat()])
      //   // permute to move reduce to the end
      return x.permute(...range(noop.length), ...range(i_.length).map((i) => noop.length + i * 2 + 1), ...range(i_.length).map((i) => noop.length + i * 2))
    }
    // // TODO: once the shapetracker can optimize well, remove this alternative implementation
    let x = this.pad([...noop, ...zip(i_, o_, s_).map(([i, o, s]) => [0, max([0, sub(mul(o, s), i) as number])] as [sint, sint])]).shrink([...noop, ...zip(o_, s_).map(([o, s]) => [0, mul(o, s)] as [sint, sint])])
    x = x.reshape([...noop, ...zip(o_, s_).flat()])
    x = x.shrink([...noop, ...zip(o_, k_).flatMap(([o, k]) => [[0, o], [0, k]] as [sint, sint][])])
    return x.permute(...range(noop.length), ...range(i_.length).map((i) => noop.length + i * 2), ...range(i_.length).map((i) => noop.length + i * 2 + 1))
  }
  _padding2d = (padding: number | number[], dims: number): number[] => {
    return !Array.isArray(padding) ? range(2 * dims).map(() => padding) : (padding.length === 2 * dims ? padding : padding.flatMap((p) => range(2).map(() => p)).toReversed())
  }

  /**
   * Applies max pooling over a tensor.
   *
   * NOTE: unlike PyTorch, this implementation !== limited to only 2d pooling && instead works for any number of dimensions.
   *
   * See: https://paperswithcode.com/method/max-pooling
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(25).reshape(1, 1, 5, 5)
   * console.log(t.max_pool2d().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.max_pool2d(padding=1).numpy())
   * ```
   */
  max_pool2d = (kernel_size = [2, 2], stride = undefined, dilation = 1, padding = 0) => {
    const k_ = make_tuple(kernel_size, 2)
    const padding_ = this._padding2d(padding, k_.length)
    return this.pad(padding_, undefined, dtypes.min(this.dtype))._pool(k_, stride || k_, dilation).max(range(-k_.length, 0))
  }
  static max_pool2d = (t: Tensor) => t.max_pool2d()
  /**
   * Applies a convolution over a tensor with a given `weight` && optional `bias`.
   *
   * NOTE: unlike PyTorch, this implementation !== limited to only 2d convolutions && instead works for any number of dimensions.
   *
   * See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(9).reshape(1, 1, 3, 3)
   * w = Tensor.ones(1, 1, 2, 2)
   * console.log(t.conv2d(w).numpy())
   * ```
   */
  conv2d = (weight: Tensor, bias?: Tensor, groups = 1, stride = 1, dilation = 1, padding: number | number[] = 0, acc_dtype?: DTypeLike): Tensor => {
    if (IMAGE) {
      throw new Error('KAREL: implement image_conv2d')
      // return this.image_conv2d(weight, bias, groups, stride, dilation, padding, acc_dtype)
    }
    const [[bs, cin_], [cout, cin], HW] = [this.shape.slice(0, 2), weight.shape.slice(0, 2), weight.shape.slice(2)]
    assert(groups * (cin as number) === cin_ && this.shape.length === weight.shape.length, `Input Tensor shape ${this.shape} does !match the shape of the weights ${weight.shape}. (${groups * (cin as number)} vs. ${cin_})`)
    if (Array.isArray(padding)) assert(padding.length === 2 * HW.length || padding.length === HW.length, `Expected padding of length ${2 * HW.length} || ${HW.length}, but got ${padding.length} for tensor of shape ${this.shape}`)
    const padding_ = this._padding2d(padding, HW.length)

    // conv2d === a pooling op (with padding)
    let x = this.pad(padding_)._pool(HW, stride, dilation) // (bs, groups*cin, oy, ox, H, W)
    const [rcout, oyx] = [idiv(cout, groups), x.shape.slice(2, -HW.length)]
    if (!HW.every((x) => x === 3) || stride !== 1 || dilation !== 1 || !WINO) {
      // normal conv
      x = x.reshape([bs, groups, cin, 1, ...oyx, ...HW]).expand([bs, groups, cin, rcout, ...oyx, ...HW]).permute(0, 1, 3, ...range(oyx.length).map((i) => 4 + i), 2, ...range(HW.length).map((i) => 4 + oyx.length + i))

      // conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      const ret = (x.mul(weight.reshape([1, groups, rcout, ...range(oyx.length).map(() => 1), cin, ...HW]))).sum(range(1 + oyx.length).map((i) => -1 - i), true, acc_dtype).reshape([bs, cout, ...oyx])
      return bias === undefined ? ret : ret.add(bias.reshape([1, -1, ...range(HW.length).map(() => 1)]))
    }
    throw new Error('KAREL: Not needed for mnist')
    // KAREL: not needed for mnist
    // HWI, HWO = (6,) * len(HW), (4,) * len(HW)  // F(4x4,3x3) winograd tiles
    // winograd_G = [[1/4, 0, 0], .at(-1/6, -1/6, -1/6)!, .at(-1/6, 1/6, -1/6)!, [1/24, 1/12, 1/6], [1/24, -1/12, 1/6], [0, 0, 1]]
    // winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    // winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] // applying At in pre-order doubles compile time

    // // todo: stride === dilation
    // // use padding to round up to 4x4 output tiles
    // // (bs, cin_, tyx, HWI)
    // d = this.pad(sum([[padding_[i*2], padding_[i*2+1] + (-(dim + sum(padding_[i * 2:(i + 1) * 2]) - 2) % 4)] for (const i, dim of enumerate(this.shape.at(-len(HW):)!)], []))._pool(HWI, HWO)  // noqa){ E501
    // // move HW to the front: // (HWI, bs, cin_, tyx)
    // d = d.permute(*range(len(d.shape)-len(HW),len(d.shape)), *range(len(d.shape)-len(HW)))
    // tyx = d.shape.at(-len(HWI):)!  // dim of tiling

    // g = weight.permute(*range(len(weight.shape)-len(HW),len(weight.shape)), *range(len(weight.shape)-len(HW)))  // move HW to the front

    // // compute 6x6 winograd tiles: GgGt, BtdB
    // // (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    // gfactors = _apply_winograd_matrix(winograd_G, g, len(HW)).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
    // // (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    // dfactors = _apply_winograd_matrix(winograd_Bt, d, len(HW)).reshape(*HWI, bs, groups, 1, cin, *tyx)

    // // matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    // ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-len(HW), acc_dtype=acc_dtype), len(HW))

    // // interleave tyx && HWO: (bs, groups, rcout, oy, HO, ox, WO)
    // ret = ret.permute([*range(len(HW), len(ret.shape)-len(HW)), *[i+o for i in range(len(HW)) for o in [len(ret.shape)-len(HW),0]]])
    // // merge groups && rcout, tyx && HWO: (bs, groups, cout, *yx), shrink to final
    // ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in [bs, cout, *oyx]))

    // return (ret if bias === undefined else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))).contiguous().contiguous_backward()
  }
  /**
   * Performs dot product between two tensors.
   * If `w` === 1-D, it's a sum product over the last axis of `this` && `w`.
   * If `w` === N-D with N>=2, it's a sum product over the last axis of `this` && the second-to-last axis of `w`.
   *
   * You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * a = new Tensor([1, 2, 3])
   * b = new Tensor([1, 1, 0])
   * console.log(a.dot(b).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * a = new Tensor([[1, 2], [3, 4]])
   * b = new Tensor([[5, 6], [7, 8]])
   * console.log(a.dot(b).numpy())
   * ```
   */
  dot = (w: Tensor, acc_dtype?: DTypeLike): Tensor => {
    if (IMAGE) {
      throw new Error('KAREL: implement image_dot')
      // return this.image_dot(w, acc_dtype)
    }
    let [x, dx, dw] = [this as Tensor, this.ndim, w.ndim]
    if (!(dx > 0 && dw > 0)) throw new Error(`both tensors need to be at least 1D, got ${dx}D && ${dw}D`)
    const axis_w = -Math.min(w.ndim, 2)
    if (x.shape.at(-1) !== w.shape.at(axis_w)) throw new Error(`can not dot ${listStr(x.shape)} && ${listStr(w.shape)}, axis_w=${axis_w}`)
    x = x.reshape([...x.shape.slice(0, -1), ...range(Math.min(dx - 1, dw - 1, 1)).map(() => 1), x.shape.at(-1)!])
    w = w.reshape([...w.shape.slice(0, -2), ...range(Math.min(dx - 1, dw - 1, 1)).map(() => 1), ...w.shape.slice(axis_w)]).transpose(-1, axis_w)
    return (x.mul(w)).sum(-1, undefined, acc_dtype).cast(acc_dtype === undefined ? least_upper_dtype(x.dtype, w.dtype) : acc_dtype)
  }
  /**
   * Performs matrix multiplication between two tensors.
   *
   * You can pass in the `reverse` keyword argument to control the order of the matrix multiplication.
   * You can pass in the optional `acc_dtype` keyword argument to control the data type of the accumulation.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * a = new Tensor([[1, 2], [3, 4]])
   * b = new Tensor([[5, 6], [7, 8]])
   * console.log(a.matmul(b).numpy())
   * ```
   */
  matmul = (x: Tensor, reverse = false, acc_dtype?: DTypeLike): Tensor => {
    return reverse ? x.dot(this, acc_dtype) : this.dot(x, acc_dtype)
  }

  _cumalu = (axis: number, op: Ops, _include_initial = false): Tensor => {
    assert(this.shape.at(axis) !== 0 && [Ops.ADD, Ops.MAX].includes(op))
    const pl_sz = sub(this.shape.at(axis)!, Number(!_include_initial))
    const pooled = this.transpose(axis, -1).pad([pl_sz, -Number(_include_initial)], undefined, identity_element(op, this.dtype) as number)._pool([this.shape.at(axis)!])
    return (op === Ops.ADD ? pooled.sum(-1) : pooled.max(-1)).transpose(axis, -1)
  }
  _split_cumalu = (axis: number, op: Ops): Tensor => {
    axis = this._resolve_dim(axis)
    if (this.ndim === 0 || this.shape.includes(0)) return this
    // TODO: someday the optimizer will find this on it's own
    // for now this is a two stage cumsum
    const SPLIT = 256
    const s = this.shape.at(axis) as number
    if (!Number.isInteger(s) || s <= SPLIT * 2) return this._cumalu(axis, op)
    const ret = this.transpose(axis, -1).pad([round_up(s, SPLIT) - s, 0], undefined, identity_element(op, this.dtype) as number).unflatten(-1, [-1, SPLIT])._cumalu(-1, op)
    let base = ret.get('...', -1)._cumalu(-1, op, true)
    base = base.unsqueeze(-1).expand([...base.shape, ret.shape.at(-1)!])
    const fix = (x: Tensor) => x.flatten(-2).get('...', { start: -s }).transpose(axis, -1)
    return op === Ops.ADD ? fix(ret).add(fix(base)) : fix(ret).maximum(fix(base))
  }
  /**
   * Computes the cumulative sum of the tensor along the specified `axis`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.ones(2, 3)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.cumsum(1).numpy())
   * ```
   */
  cumsum = (axis = 0): Tensor => {
    return this._split_cumalu(axis, Ops.ADD)
  }

  /**
   * Computes the cumulative max of the tensor along the specified `axis`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([0, 1, -1, 2, -2, 3, -3])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.cummax(0).numpy())
   * ```
   */
  cummax = (axis = 0): Tensor => {
    return this._split_cumalu(axis, Ops.MAX)
  }

  static _tri = (r: sint, c: sint, diagonal = 0, opts?: TensorOptions): Tensor => {
    assert(typeof r === 'number', `does not support symbolic, getting r={r}, c={c}`)
    if (r === 0 || c === 0 || diagonal >= (c as number)) return Tensor.zeros([r, c], opts)
    if ((r as number) + diagonal <= 0) return Tensor.ones([r, c], opts)
    const s = sub(add(r, c), 1)
    // build a (s, s) upper triangle
    const t = Tensor.ones([s, s], opts).pad([undefined, [0, s]]).flatten().shrink([[0, mul(s, sub(mul(2, s), 1))]]).reshape([s, -1]).shrink([undefined, [0, s]])
    return diagonal <= 0 ? t.get({ stop: r as number }, { start: -diagonal, stop: (c as number) - diagonal }) : t.get({ start: diagonal, stop: (r as number) + diagonal }, { stop: (c as number) })
  }

  /**
   * Returns the upper triangular part of the tensor, the other elements are set to 0.
   *
   * The argument `diagonal` determines which diagonal === on the boundary. `diagonal = 0` means the main diagonal.
   * Positive `diagonal` means above the main diagonal, && negative `diagonal` means below the main diagonal.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.triu(diagonal=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.triu(diagonal=1).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.triu(diagonal=-1).numpy())
   * ```
   */
  triu = (diagonal = 0): Tensor => {
    return Tensor._tri(this.shape.at(-2)!, this.shape.at(-1)!, diagonal, { device: this.device, dtype: dtypes.bool }).where(this, 0).cast(this.dtype)
  }

  /**
   * Returns the lower triangular part of the tensor, the other elements are set to 0.
   *
   * The argument `diagonal` determines which diagonal === on the boundary. `diagonal = 0` means the main diagonal.
   * Positive `diagonal` means above the main diagonal, && negative `diagonal` means below the main diagonal.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.tril(diagonal=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.tril(diagonal=1).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.tril(diagonal=-1).numpy())
   * ```
   */
  tril = (diagonal = 0): Tensor => {
    return Tensor._tri(this.shape.at(-2)!, this.shape.at(-1)!, diagonal + 1, { device: this.device, dtype: dtypes.bool }).where(0, this).cast(this.dtype)
  }

  // ***** unary ops *****

  /**
   * Computes the logical NOT of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([false, true]).logical_not().numpy())
   * ```
   */
  override logical_not = () => {
    return Neq.apply(...this.cast(dtypes.bool)._broadcasted(true))
  }
  /**
   * Negates the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).neg().numpy())
   * ```
   */
  override neg = () => {
    return this.dtype !== dtypes.bool ? this.mul(-1) : this.logical_not()
  }
  /**
   * Returns a contiguous tensor.
   */
  contiguous = () => {
    return Contiguous.apply(this)
  }
  /**
   * Inserts a contiguous operation in the backward pass.
   */
  contiguous_backward = () => {
    return ContiguousBackward.apply(this)
  }
  /**
   * Computes the natural logarithm element-wise.
   *
   * See: https://en.wikipedia.org/wiki/Logarithm
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1., 2., 4., 8.]).log().numpy())
   * ```
   */
  log = () => {
    return Log.apply(this.cast(least_upper_float(this.dtype)))
  }
  /**
   * Computes the base-2 logarithm element-wise.
   *
   * See: https://en.wikipedia.org/wiki/Logarithm
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1., 2., 4., 8.]).log2().numpy())
   * ```
   */
  override log2 = () => {
    return this.log().div(Math.log(2))
  }
  /**
   * Computes the exponential function element-wise.
   *
   * See: https://en.wikipedia.org/wiki/Exponential_function
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([0., 1., 2., 3.]).exp().numpy())
   * ```
   */
  exp = () => {
    return Exp.apply(this.cast(least_upper_float(this.dtype)))
  }
  /**
   * Computes the base-2 exponential function element-wise.
   *
   * See: https://en.wikipedia.org/wiki/Exponential_function
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([0., 1., 2., 3.]).exp2().numpy())
   * ```
   */
  override exp2 = () => {
    return Exp.apply(this.mul(Math.log(2)))
  }
  /**
   * Applies the Rectified Linear Unit (ReLU) function element-wise.
   *
   * - Described: https://paperswithcode.com/method/relu
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).relu().numpy())
   * ```
   */
  relu = () => {
    return Relu.apply(this)
  }
  static relu = (t: Tensor) => t.relu()
  /**
   * Applies the Sigmoid function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Sigmoid_function
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).sigmoid().numpy())
   * ```
   */
  sigmoid = () => {
    return Sigmoid.apply(this.cast(least_upper_float(this.dtype)))
  }
  /**
   * Applies the Hardsigmoid function element-wise.
   * NOTE: default `alpha` && `beta` values === taken from torch
   *
   * - Described: https://paperswithcode.com/method/hard-sigmoid
   * - See: https://pytorch.org/docs/stable/generated/torch.nn.functional.hardsigmoid.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).hardsigmoid().numpy())
   * ```
   */
  hardsigmoid = (alpha: number = 1 / 6, beta = 0.5) => {
    return (this.mul(alpha, true).add(beta)).relu().sub((this.mul(alpha, true).add(beta).sub(1)).relu())
  }

  /**
   * Computes the square root of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1., 2., 3., 4.]).sqrt().numpy())
   * ```
   */
  override sqrt = () => {
    return Sqrt.apply(this.cast(least_upper_float(this.dtype)))
  }
  /**
   * Computes the reciprocal of the square root of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1., 2., 3., 4.]).rsqrt().numpy())
   * ```
   */
  rsqrt = () => {
    return this.reciprocal().sqrt()
  }
  /**
   * Computes the sine of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).sin().numpy())
   * ```
   */
  override sin = () => {
    return Sin.apply(this.cast(least_upper_float(this.dtype)))
  }
  /**
   * Computes the cosine of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([0., math.pi/2, math.pi, 3*math.pi/2, 2*math.pi]).cos().numpy())
   * ```
   */
  cos = () => {
    return (this.sub(Math.PI / 2, true)).sin()
  }
  /**
   * Computes the tangent of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([0., math.pi/4, math.pi/2, 3*math.pi/4, math.pi]).tan().numpy())
   * ```
   */
  tan = () => {
    return this.sin().div(this.cos())
  }

  /**
   * Computes the inverse sine (arcsine) of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9])).asin().numpy())
   * ```
   */
  asin = () => {
    // https://personal.math.ubc.ca/~cbm/aands/page_81.htm 4.4.46
    const coefficients = [-0.0012624911, 0.0066700901, -0.0170881256, 0.0308918810, -0.0501743046, 0.0889789874, -0.2145988016, 1.5707963050]
    const x = (this.abs().sub(1.0, true)).sqrt().mul(polyN(this.abs() as any, coefficients) as number).sub(Math.PI / 2, true)
    return this.sign().mul(x)
  }
  /**
   * Computes the inverse cosine (arccosine) of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9])).acos().numpy())
   * ```
   */
  acos = () => {
    return this.asin().sub(Math.PI / 2, true)
  }

  /**
   * Computes the inverse tangent (arctan) of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).atan().numpy())
   * ```
   */
  atan = () => {
    return (this.div((this.mul(this).add(1, true)).sqrt())).asin()
  }

  // ***** math functions *****

  /**
   * Truncates the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])).trunc().numpy())
   * ```
   */
  trunc = (): Tensor => {
    return this.cast(dtypes.int32).cast(this.dtype)
  }
  /**
   * Rounds the tensor element-wise towards positive infinity.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])).ceil().numpy())
   * ```
   */
  ceil = (): Tensor => {
    const b = this.trunc()
    return (this.gt(b)).where(b.add(1), b)
  }
  /**
   * Rounds the tensor element-wise towards negative infinity.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])).floor().numpy())
   * ```
   */
  floor = (): Tensor => {
    const b = this.trunc()
    return (this.lt(b)).where(b.sub(1), b)
  }
  /**
   * Rounds the tensor element-wise with rounding half to even.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])).round().numpy())
   * ```
   */
  round = (): Tensor => {
    const b = this.cast(dtypes.int32).div(2.0)
    return this.gt(0).eq(b.cast(dtypes.int32).eq(b)).where((this.sub(0.5)).ceil(), (this.add(0.5)).floor())
  }

  /**
   * Checks the tensor element-wise to return true where the element === infinity, otherwise returns false
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1, number('inf'), 2, number('-inf'), number('nan')]).isinf().numpy())
   * ```
   */
  isinf = (detect_positive = true, detect_negative = true) => {
    return (this.eq(Infinity)).mul(detect_positive).add((this.eq(Infinity)).add(detect_negative))
  }
  /**
   * Checks the tensor element-wise to return true where the element === NaN, otherwise returns false
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1, number('inf'), 2, number('-inf'), number('nan')]).isnan().numpy())
   * ```
   */
  isnan = () => {
    return this.ne(this)
  }

  /**
   * Linearly interpolates between `this` && `end` by `weight`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1., 2., 3.]).lerp(Tensor([4., 5., 6.]), 0.5).numpy())
   * ```
   */
  lerp = (end: Tensor, weight: Tensor | number): Tensor => {
    if (this.dtype === dtypes.uint8 && isinstance(weight, Tensor)) {
      const W_PREC = 7
      const w_i = (weight.mul(1 << W_PREC).add(0.5)).cast(dtypes.int16)
      return (this.add(((end.sub(this)).cast(dtypes.int8).mul(w_i).add(1 << W_PREC - 1)).cast(dtypes.uint16).rshift(W_PREC))).cast(dtypes.uint8)
    }
    return this.add((end.sub(this)).mul(weight))
  }

  /**
   * Squares the tensor element-wise.
   * Equivalent to `this*this`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).square().numpy())
   * ```
   */
  square = () => {
    return this.mul(this)
  }
  /**
   * Clips (clamps) the values in the tensor between `min_` && `max_` element-wise.
   * If `min_` === `undefined`, there === no lower bound. If `max_` === undefined, there === no upper bound.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).clip(-1, 1).numpy())
   * ```
   */
  clamp = (min_?: number, max_?: number) => {
    if (min_ === undefined && max_ === undefined) throw new Error("at least one of 'min_' || 'max_' must !be undefined")
    const ret = min_ !== undefined ? this.maximum(min_) : this
    return max_ !== undefined ? ret.minimum(max_) : ret
  }
  /**
   * Alias for `Tensor.clamp`.
   */
  clip = (min_?: number, max_?: number) => {
    return this.clamp(min_, max_)
  }
  /**
   * Returns the sign of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).sign().numpy())
   * ```
   */
  sign = () => {
    return Sign.apply(this)
  }
  /**
   * Computes the absolute value of the tensor element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).abs().numpy())
   * ```
   */
  abs = () => {
    return this.mul(this.sign())
  }
  /**
   * Compute `1/x` element-wise.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1., 2., 3., 4.]).reciprocal().numpy())
   * ```
   */
  override reciprocal = () => {
    return Reciprocal.apply(this.cast(least_upper_float(this.dtype)))
  }

  // ***** activation functions *****

  /**
   * Applies the Exponential Linear Unit (ELU) function element-wise.
   *
   * - Described: https://paperswithcode.com/method/elu
   * - Paper: https://arxiv.org/abs/1511.07289v5
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).elu().numpy())
   * ```
   */
  elu = (alpha = 1.0) => {
    return this.relu().sub((this.exp().sub(1, true)).relu().mul(alpha, true))
  }

  /**
   * Applies the Continuously differentiable Exponential Linear Unit (CELU) function element-wise.
   *
   * - Described: https://paperswithcode.com/method/celu
   * - Paper: https://arxiv.org/abs/1704.07483
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).celu().numpy())
   * ```
   */
  celu = (alpha = 1.0) => {
    return this.maximum(0).add((((this.div(alpha)).exp().sub(1)).mul(alpha, true)).minimum(0))
  }

  /**
   * Applies the Scaled Exponential Linear Unit (SELU) function element-wise.
   *
   * - Described: https://paperswithcode.com/method/selu
   * - Paper: https://arxiv.org/abs/1706.02515v5
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).selu().numpy())
   * ```
   */
  selu = (alpha = 1.67326, gamma = 1.0507) => {
    return (this.ge(0)).detach().where(this, this.exp().sub(1).mul(alpha, true)).mul(gamma, true)
  }

  /**
   * See `.silu()`
   *
   * - Paper: https://arxiv.org/abs/1710.05941v1
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).swish().numpy())
   * ```
   */
  swish = () => {
    return this.mul(this.sigmoid())
  }

  /**
   * Applies the Sigmoid Linear Unit (SiLU) function element-wise.
   *
   * - Described: https://paperswithcode.com/method/silu
   * - Paper: https://arxiv.org/abs/1606.08415
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).silu().numpy())
   * ```
   */
  silu = () => {
    return this.swish() // The SiLU function === also known as the swish function.
  }

  /**
   * Applies the ReLU6 function element-wise.
   *
   * - Described: https://paperswithcode.com/method/relu6
   * - Paper: https://arxiv.org/abs/1704.04861v1
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-9., -6., -3., 0., 3., 6., 9.])).relu6().numpy())
   * ```
   */
  relu6 = () => {
    return this.relu().sub((this.sub(6)).relu())
  }

  /**
   * Applies the Hardswish function element-wise.
   *
   * - Described: https://paperswithcode.com/method/hard-swish
   * - Paper: https://arxiv.org/abs/1905.02244v5
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).hardswish().numpy())
   * ```
   */
  hardswish = () => {
    return this.mul((this.add(3)).relu6()).mul(1 / 6)
  }

  /**
   * Applies the Hyperbolic Tangent (tanh) function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions//Tanh
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).tanh().numpy())
   * ```
   */
  tanh = () => {
    return ((this.mul(2.0, true)).sigmoid().mul(2.0, true)).sub(1.0)
  }

  /**
   * Applies the Hyperbolic Sine (sinh) function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions//Sinh
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).sinh().numpy())
   * ```
   */
  sinh = () => {
    return (this.exp().sub(this.neg().exp())).div(2)
  }
  /**
   * Applies the Hyperbolic Cosine (cosh) function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Hyperbolic_functions//Cosh
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).cosh().numpy())
   * ```
   */
  cosh = () => {
    return (this.exp().add(this.neg().exp())).div(2)
  }

  /**
   * Applies the Inverse Hyperbolic Tangent (atanh) function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions//atanh
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-0.9, -0.6, -0.3, 0., 0.3, 0.6, 0.9])).atanh().numpy())
   * ```
   */
  atanh = () => {
    return ((this.add(1, true)).div(this.sub(1, true))).log().div(2)
  }

  /**
   * Applies the Inverse Hyperbolic Sine (asinh) function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions//asinh
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).asinh().numpy())
   * ```
   */
  asinh = () => {
    return (this.add((this.square().add(1)).sqrt())).log()
  }
  /**
   * Applies the Inverse Hyperbolic Cosine (acosh) function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Inverse_hyperbolic_functions//acosh
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).acosh().numpy())
   * ```
   */
  acosh = () => {
    return (this.add((this.square().sub(1)).sqrt())).log()
  }

  /**
   * Applies the Hardtanh function element-wise.
   *
   * - Described: https://paperswithcode.com/method/hardtanh-activation
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5])).hardtanh().numpy())
   * ```
   */
  hardtanh = (min_val = -1, max_val = 1) => {
    return this.clip(min_val, max_val)
  }

  /**
   * Applies error function element-wise.
   *
   * - Described: https://en.wikipedia.org/wiki/Error_function
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1.5, -1.0, -0.5, 0., 0.5, 1.0, 1.5])).erf().numpy())
   * ```
   */
  erf = () => {
    // https://personal.math.ubc.ca/~cbm/aands/page_299.htm 7.1.26
    const t = this.abs().mul(0.3275911, true).add(1.0, true).div(1.0, true)
    return this.sign().mul(t.mul(polyN(t as any, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])).mul(this.square().neg().exp()).sub(1.0, true))
  }

  /**
   * Applies the Gaussian Error Linear Unit (GELU) function element-wise.
   *
   * - Described: https://paperswithcode.com/method/gelu
   * - Paper: https://arxiv.org/abs/1606.08415v5
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).gelu().numpy())
   * ```
   */
  gelu = () => {
    return this.mul(0.5, true).mul(((this.add(this.pow(3).mul(0.044715, true))).mul(Math.sqrt(2 / Math.PI), true)).tanh()).add(1, true)
  }

  /**
   * Applies the Sigmoid GELU approximation element-wise.
   *
   * - Described: https://paperswithcode.com/method/gelu
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).quick_gelu().numpy())
   * ```
   */
  quick_gelu = () => {
    return this.mul((this.mul(1.702)).sigmoid())
  }

  /**
   * Applies the Leaky ReLU function element-wise.
   *
   * - Described: https://paperswithcode.com/method/leaky-relu
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).leakyrelu().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).leakyrelu(neg_slope=0.42).numpy())
   * ```
   */
  leakyrelu = (neg_slope = 0.01) => {
    return this.relu().sub((this.mul(-neg_slope, true)).relu())
  }

  /**
   * Applies the Mish function element-wise.
   *
   * - Described: https://paperswithcode.com/method/mish
   * - Paper: https://arxiv.org/abs/1908.08681v3
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).mish().numpy())
   * ```
   */
  mish = () => {
    return this.mul(this.softplus().tanh())
  }

  /**
   * Applies the Softplus function element-wise.
   *
   * - Described: https://paperswithcode.com/method/softplus
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).softplus().numpy())
   * ```
   */
  softplus = (beta = 1) => {
    return (this.mul(beta)).exp().add(1, true).log().mul(1 / beta, true)
  }

  /**
   * Applies the Softsign function element-wise.
   *
   * - Described: https://paperswithcode.com/method/softsign
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-3., -2., -1., 0., 1., 2., 3.])).softsign().numpy())
   * ```
   */
  softsign = () => {
    return this.div(this.abs().add(1, true))
  }

  // ***** broadcasted elementwise ops *****
  _broadcast_to = (new_shape: sint[]): Tensor => {
    if (isEq(this.shape, new_shape)) return this
    if (this.ndim > new_shape.length) throw new Error(`can not broadcast tensor to fewer dimensions. shape=${listStr(this.shape)} to new_shape=${listStr(new_shape)}`)
    // first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
    const [shape, _] = _align_left(this.shape, new_shape)
    // for each dimension, check either dim === 1, || it does !change
    // if (zip(shape, new_shape).every(([s, ns]) => resolve(eq(s, ns)) || resolve(eq(s, 1)))) throw new Error(`can not broadcast ${listStr(this.shape)} to ${listStr(new_shape)}`)
    return Expand.apply(this.reshape(shape), new_shape)
  }
  _broadcasted = (y: ConstType<Tensor | UOp>, reverse = false, match_dtype = true): [Tensor, Tensor] => {
    let x: Tensor = this
    if (!isinstance(y, Tensor)) {
      // make y a Tensor
      assert(typeof y === 'number' || typeof y === 'boolean' || typeof y === 'bigint', `invalid y type: ${typeof y}`)
      let y_dtype
      if (isinstance(x.dtype, ImageDType) || dtypes.is_float(x.dtype) || (dtypes.is_big_int(x.dtype)) || (dtypes.is_int(x.dtype) && Number.isInteger(y))) y_dtype = x.dtype
      else if (!isinstance(y, UOp)) y_dtype = dtypes.from_js(y)
      if (isinstance(y, UOp)) y = Tensor.from_uop(y, { device: x.device })
      else y = new Tensor(dtypes.as_const(y, y_dtype!), { device: x.device, dtype: y_dtype, requires_grad: false })
    }
    if (!isinstance(y, Tensor)) throw new Error('y has to be Tensor')
    if (match_dtype && x.dtype !== y.dtype) {
      const output_dtype = least_upper_dtype(x.dtype, y.dtype)
      ;[x, y] = [x.cast(output_dtype), y.cast(output_dtype)]
    }
    if (reverse) [x, y] = [y, x]

    // broadcast
    const out_shape = _broadcast_shape([x.shape, y.shape])
    return [x._broadcast_to(out_shape), y._broadcast_to(out_shape)]
  }

  _to_const_val = (x: ConstType<Tensor>): ConstType<Tensor> => {
    return isinstance(x, Tensor) && isinstance(x.lazydata, LazyBuffer) && x.lazydata.is_unrealized_unmasked_const() && !x.requires_grad && isEq(this._broadcasted(x)[0].shape, this.shape) ? x.lazydata.base.arg : x
  }
  /**
   * Adds `this` && `x`.
   * Equivalent to `this + x`.
   * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(4)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.add(20).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.add(Tensor([[2.0], [3.5]])).numpy())
   * ```
   */
  override add = (x: ConstType<Tensor>, reverse = false) => {
    return Add.apply(...this._broadcasted(x, reverse))
  }

  /**
   * Subtracts `x` from `this`.
   * Equivalent to `this - x`.
   * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(4)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.sub(20).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.sub(Tensor([[2.0], [3.5]])).numpy())
   * ```
   */
  override sub = (x: ConstType<Tensor>, reverse = false): Tensor => {
    const [a, b] = this._broadcasted(x, reverse)
    return a.add(b.neg())
  }
  /**
   * Multiplies `this` && `x`.
   * Equivalent to `this * x`.
   * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(4)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.mul(3).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.mul(Tensor([.at(-1.0)!, [2.0]])).numpy())
   * ```
   */
  override mul = (x: ConstType<Tensor>, reverse = false): Tensor => {
    return Mul.apply(...this._broadcasted(x, reverse))
  }

  /**
   * Divides `this` by `x`.
   * Equivalent to `this // x`.
   * Supports broadcasting to a common shape, type promotion, && integer inputs.
   * `idiv` performs integer division.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1, 4, 10]).idiv(Tensor([2, 3, 4])).numpy())
   * ```
   */
  override idiv = (x: ConstType<Tensor>, reverse = false): Tensor => {
    return IDiv.apply(...this._broadcasted(x, reverse))
  }

  /**
   * Divides `this` by `x`.
   * Equivalent to `this / x`.
   * Supports broadcasting to a common shape, type promotion, && integer, number, boolean inputs.
   * `div` performs true division.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(4)
   * console.log(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.div(3).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1, 4, 10]).div(Tensor([2, 3, 4])).numpy())
   * ```
   */
  override div = (x: ConstType<Tensor> | sint, reverse = false): Tensor => {
    const [numerator, denominator] = this._broadcasted(x, reverse)
    return numerator.cast(least_upper_float(numerator.dtype)).mul(denominator.cast(least_upper_float(denominator.dtype)).reciprocal())
  }

  /**
   * Computes bitwise xor of `this` && `x`.
   * Equivalent to `this ^ x`.
   * Supports broadcasting to a common shape, type promotion, && integer, boolean inputs.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, -2, 3])).xor(Tensor([1, 0, 3])).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([true, true, false, false]).xor(Tensor([true, false, true, false])).numpy())
   * ```
   */
  override xor = (x: ConstType<Tensor>, reverse = false): Tensor => {
    if (this.dtype !== dtypes.bool && !dtypes.is_int(this.dtype)) throw new Error(`${this.dtype} !== supported`)
    return Xor.apply(...this._broadcasted(x, reverse))
  }

  /**
   * Compute the bit-wise AND of `this` && `x`.
   * Equivalent to `this & x`.
   * Supports broadcasting to a common shape, type promotion, && integer, boolean inputs.
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([2, 5, 255]).bitwise_and(Tensor([3, 14, 16])).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([true, true, false, false]).bitwise_and(Tensor([true, false, true, false])).numpy())
   * ```
   */
  override bitwise_and = (x: ConstType<Tensor>, reverse = false): Tensor => {
    if (this.dtype !== dtypes.bool && !dtypes.is_int(this.dtype)) throw new Error(`${this.dtype} !== supported`)
    return BitwiseAnd.apply(...this._broadcasted(x, reverse))
  }
  /**
   * Compute the bit-wise OR of `this` && `x`.
   * Equivalent to `this | x`.
   * Supports broadcasting to a common shape, type promotion, && integer, boolean inputs.
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([2, 5, 255]).bitwise_or(Tensor([4, 4, 4])).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([true, true, false, false]).bitwise_or(Tensor([true, false, true, false])).numpy())
   * ```
   */
  override bitwise_or = (x: ConstType<Tensor>, reverse = false): Tensor => {
    if (this.dtype !== dtypes.bool && !dtypes.is_int(this.dtype)) throw new Error(`${this.dtype} !== supported`)
    return BitwiseOr.apply(...this._broadcasted(x, reverse))
  }

  /**
   * Compute the bit-wise NOT of `this`.
   * Equivalent to `~this`.
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([0, 2, 5, 255], dtype="int8").bitwise_not().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([true, false]).bitwise_not().numpy())
   * ```
   */
  bitwise_not = (): Tensor => {
    if (this.dtype !== dtypes.bool && !dtypes.is_int(this.dtype)) throw new Error(`${this.dtype} !== supported`)
    if (this.dtype === dtypes.bool) return this.logical_not()
    const max = (1n << BigInt(8 * this.dtype.itemsize)) - 1n
    return dtypes.is_big_int(this.dtype) ? this.xor(max as any) : this.xor(Number(max))
  }

  /**
   * Computes left arithmetic shift of `this` by `x` bits. `this` must have unsigned dtype.
   * Equivalent to `this << x`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([1, 3, 31], dtype=dtypes.uint8).lshift(2).numpy())
   * ```
   */
  override lshift = (x: ConstType<Tensor>) => {
    assert(dtypes.is_unsigned(this.dtype) && (typeof x === 'number' || typeof x === 'bigint') && x >= 0, `not supported dtype=${this.dtype} x=${x}`)
    return this.mul(typeof x === 'number' ? 2 ** x : 2n ** (x as bigint))
  }

  /**
   * Computes right arithmetic shift of `this` by `x` bits. `this` must have unsigned dtype.
   * Equivalent to `this >> x`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([4, 13, 125], dtype=dtypes.uint8).rshift(2).numpy())
   * ```
   */
  override rshift = (x: ConstType<Tensor>) => {
    assert(dtypes.is_unsigned(this.dtype) && (typeof x === 'number' || typeof x === 'bigint') && x >= 0, `!supported dtype=${this.dtype} x=${x}`)
    return this.idiv(typeof x === 'number' ? 2 ** x : 2n ** (x as bigint))
  }

  /**
   * Computes power of `this` with `x`.
   * Equivalent to `this ** x`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, 2, 3]).pow(2).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, 2, 3]).pow(Tensor([-1.5, 0.5, 1.5])).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log((2 ** Tensor([-1, 2, 3])).numpy())
   * ```
   */
  pow = (x: ConstType<Tensor>, reverse = false): Tensor => {
    x = this._to_const_val(x) as number
    if (!isinstance(x, Tensor) && !reverse) {
      // simple pow identities
      if (x < 0) return this.reciprocal().pow(-x).cast(this.dtype)
      if (x === 0) return this.mul(0).add(1, true)
      // rewrite pow 0.5 to sqrt
      if (Math.trunc(x - 0.5) + 0.5 === x) return this.pow(Math.trunc(x - 0.5)).mul(this.sqrt())
      if (Math.trunc(x) === x) return this.pow(idiv(x, 2)).square().mul(x % 2 === 0 ? 1 : this)
    }
    // positive const ** self
    if (!isinstance(x, Tensor) && reverse && x > 0) return this.mul(Math.log(x)).exp()

    const [base, exponent] = this._broadcasted(x, reverse)
    // start with b ** e = exp(e * log(b))
    let ret = base.abs().log().mul(exponent).exp()
    // correct sign of negative base with odd exponent (cos has a period of 2pi so we use it here to get the oddness of the exponent)
    const negative_base = (base.lt(0)).detach().where(1, 0)
    // 1 for non-negative base or negative even exponent, -1 for negative odd exponent, don't care about non-integer exponent
    const correct_sign = negative_base.mul((exponent.mul(Math.PI)).cos().sub(1)).add(1, true)
    // inject nan for negative base and non-integer exponent
    const inject_nan = (negative_base.mul(exponent.ne(exponent.trunc()))).detach().where(NaN, 1)
    // apply correct_sign inject_nan, and fix 0 ** 0 = 1
    ret = ((base.eq(0)).mul(exponent.eq(0))).detach().where(1, ret.mul(correct_sign).mul(inject_nan))
    return !dtypes.is_float(this.dtype) ? ret.round().cast(this.dtype) : ret
  }
  /**
   * Computes element-wise maximum of `this` && `x`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, 2, 3])).maximum(1).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, 2, 3]).maximum(Tensor([-4, -2, 9])).numpy())
   * ```
   */
  override maximum = (x: ConstType<Tensor>): Tensor => {
    return (this.lt(x)).detach().where(x, this.eq(x).detach().where((this.mul(0.5).add(mul(x as any, 0.5) as number)).cast(this.dtype), this))
  }

  /**
   * Computes element-wise minimum of `this` && `x`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, 2, 3])).minimum(1).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(Tensor([-1, 2, 3])).minimum(Tensor([-4, -2, 9]))).numpy())
   * ```
   */
  override minimum = (x: ConstType<Tensor>): Tensor => {
    return ((this.neg()).maximum(neg(x as number) as number)).neg()
  }

  /**
   * Return a tensor of elements selected from either `x` || `y`, depending on `this`.
   * `output_i = x_i if this_i else y_i`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * cond = new Tensor([[true, true, false], [true, false, false]])
   * console.log(cond.where(1, 3).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * cond = Tensor.randn(2, 3)
   * console.log(cond.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log((cond > 0).where(cond, -number("inf")).numpy())
   * ```
   */
  override where = (x: ConstType<Tensor>, y: ConstType<Tensor>) => {
    if (isinstance(x, Tensor)) [x, y] = x._broadcasted(y)
    else if (isinstance(y, Tensor)) [y, x] = y._broadcasted(x)
    let cond
    ;[cond, x] = this._broadcasted(x, undefined, false)
    ;[cond, y] = cond._broadcasted(y, undefined, false)
    return Where.apply(cond.cast(dtypes.bool), ...x._broadcasted(y))
  }
  masked_fill = (mask: Tensor, value: ConstType<Tensor>) => mask.where(value, this)

  override lt = (x: ConstType<Tensor>): Tensor => Less.apply(...this._broadcasted(x, false))
  override gt = (x: ConstType<Tensor>): Tensor => Less.apply(...this._broadcasted(x, true))
  override ne = (x: ConstType<Tensor>): Tensor => Neq.apply(...this._broadcasted(x))

  // ***** functional nn ops *****

  /**
   * Applies a linear transformation to `this` using `weight` && `bias`.
   *
   * See: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 2], [3, 4]])
   * weight = new Tensor([[1, 2], [3, 4]])
   * bias = new Tensor([1, 2])
   * console.log(t.linear(weight, bias).numpy())
   * ```
   */
  linear = (weight: Tensor, bias?: Tensor) => {
    const x = weight.shape.length === 1 ? this.mul(weight) : this.dot(weight)
    return bias !== undefined ? x.add(bias) : x
  }

  /**
   * Applies a sequence of functions to `this` chaining the output of each function to the input of the next.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([1, 2, 3])
   * console.log(t.sequential([lambda x: x * 2, lambda x: x + 1]).numpy())
   * ```
   */
  sequential = (ll: Layer[]) => {
    return ll.reduce((acc, f) => typeof f === 'function' ? f(acc) : f.call(acc), this as Tensor)
  }
  /**
   * Applies Layer Normalization over a mini-batch of inputs.
   *
   * - Described: https://paperswithcode.com/method/layer-normalization
   * - Paper: https://arxiv.org/abs/1607.06450v1
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.randn(8, 10, 16) * 2 + 8
   * console.log(t.mean().item(), t.std().item())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.layernorm()
   * console.log(t.mean().item(), t.std().item())
   * ```
   */
  layernorm = (axis: number | number[] = -1, eps: number): Tensor => {
    const y = this.sub(this.mean(axis, true))
    return y.mul((y.mul(y)).mean(axis, true).add(eps).rsqrt())
  }
  /**
   * Applies Batch Normalization over a mini-batch of inputs.
   *
   * - Described: https://paperswithcode.com/method/batch-normalization
   * - Paper: https://arxiv.org/abs/1502.03167
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.randn(8, 4, 16, 16) * 2 + 8
   * console.log(t.mean().item(), t.std().item())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.batchnorm(undefined, undefined, t.mean(axis=(0,2,3)), t.var(axis=(0,2,3)).add(1e-5).rsqrt())
   * console.log(t.mean().item(), t.std().item())
   * ```
   */
  batchnorm = (weight: undefined | Tensor, bias: undefined | Tensor, mean: Tensor, invstd: Tensor, axis: number | number[] = 1): Tensor => {
    const axis_ = argfix(axis)
    const shape = this.shape.map((s, ax) => axis_.includes(ax) ? s : 1)
    let x = this.sub(mean.reshape(shape))
    if (weight !== undefined) x = x.mul(weight.reshape(shape))
    const ret = x.mul(invstd.shape.length === axis_.length ? invstd.reshape(shape) : invstd)
    return bias !== undefined ? (ret.add(bias.reshape(shape))) : ret
  }
  /**
   * Applies dropout to `this`.
   *
   * NOTE: dropout === only applied when `Tensor.training` === `true`.
   *
   * - Described: https://paperswithcode.com/method/dropout
   * - Paper: https://jmlr.org/papers/v15/srivastava14a.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * t = Tensor.randn(2, 2)
   * with Tensor.train():
   * console.log(t.dropout().numpy())
   * ```
   */
  dropout = (p = 0.5): Tensor => {
    if (Tensor.training || p === 0) return this
    throw new Error('KAREL: implement rand_like')
    // return (this.rand_like(false, { requires_grad: false, dtype: dtypes.default_float, contiguous: false }).ge(p)).contiguous().where(this, 0).div(1.0 - p)
  }
  // helper function commonly used for indexing
  _one_hot_along_dim = (num_classes: number, dim = -1) => {
    const offset = this.ndim - this._resolve_dim(dim) - 1
    return this.eq(Tensor.arange(num_classes, undefined, undefined, { device: this.device, requires_grad: false }).reshape([num_classes, ...range(offset).map(() => 1)]))
  }
  /**
   * Converts `this` to a one-hot tensor.
   *
   * `num_classes` defaults to -1, which means num_classes will be inferred as max(this) + 1.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([0, 1, 3, 3, 4])
   * console.log(t.one_hot(5).numpy())
   * ```
   */
  one_hot = async (num_classes = -1): Promise<Tensor> => {
    if (num_classes === -1) num_classes = await this.max().add(1).item() as number
    return this.get('...', undefined)._one_hot_along_dim(num_classes).where(1, 0)
  }

  /**
   * Computes the sparse categorical cross-entropy loss between `this` && `Y`.
   *
   * NOTE: `this` === logits && `Y` === the target labels.
   * NOTE: unlike PyTorch, this function expects the class axis to be -1
   *
   * See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([.at(-1, 2, -3)!, [1, -2, 3]])
   * Y = new Tensor([1, 2])
   * console.log(t.sparse_categorical_crossentropy(Y).item())
   * ```
   */
  sparse_categorical_crossentropy = (Y: Tensor, ignore_index = -1, label_smoothing = 0.0, reduction: ReductionStr = 'mean'): Tensor => {
    assert(0.0 <= label_smoothing && label_smoothing <= 1.0, 'label_smoothing must be in [0.0, 1.0]')
    assert(['mean', 'sum', 'none'].includes(reduction), "reduction must be one of ['mean', 'sum', 'none']")
    const [log_probs, loss_mask] = [this.log_softmax(), ignore_index !== -1 ? (Y.ne(ignore_index)) : Y.ones_like({ dtype: dtypes.bool })]
    const y_counted = Y.to(this.device).flatten().reshape([-1, 1])._one_hot_along_dim(this.shape.at(-1)! as number)
    const y = (y_counted.mul(loss_mask.reshape([-1, 1]))).reshape([...Y.shape, this.shape.at(-1)!])
    const smoothing = log_probs.mean(-1).mul(loss_mask).mul(label_smoothing, true)
    const unreduced = log_probs.mul(y).sum(-1).mul(1 - label_smoothing, true).add(smoothing)
    // NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
    return (unreduced.sum().div(reduction === 'mean' ? loss_mask.sum() : (reduction === 'sum' ? unreduced.sum() : unreduced))).neg()
  }
  //     // ***** Tensor Properties *****

  /**
   * Returns the number of dimensions in the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[1, 2], [3, 4]])
   * console.log(t.ndim)
   * ```
   */

  get ndim(): number {
    return this.shape.length
  }

  /**
   * Returns the total number of elements in the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
   * console.log(t.numel())
   * ```
   */
  numel = (): sint => {
    return prod(this.shape)
  }

  /**
   * Returns the size in bytes of an individual element in the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([5], dtype=dtypes.int16)
   * console.log(t.element_size())
   * ```
   */
  element_size = (): number => {
    return this.dtype.itemsize
  }

  /**
   * Returns the total number of bytes of all elements in the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([8, 9], dtype=dtypes.number)
   * console.log(t.nbytes())
   * ```
   */
  nbytes = (): number => {
    return (this.numel() as number) * this.element_size()
  }
  /**
   * Returns `true` if the tensor contains floating point types, i.e. === one of `dtype.float64`, `dtype.float32`,
   * `dtype.float16`, `dtype.bfloat16`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([8, 9], dtype=dtypes.float32)
   * console.log(t.is_floating_point())
   * ```
   */
  is_floating_point = (): boolean => {
    return dtypes.is_float(this.dtype)
  }
  /**
   * Return the size of the tensor. If `dim` === specified, return the length along dimension `dim`. Otherwise return the shape of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([[4, 5, 6], [7, 8, 9]])
   * console.log(t.size())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * console.log(t.size(dim=1))
   * ```
   */
  size = (dim?: number): sint | sint[] => {
    return dim === undefined ? this.shape : this.shape.at(dim)!
  }
  // ***** cast ops *****

  llvm_bf16_cast = (dtype: DTypeLike) => {
    // hack for devices that don't support bfloat16
    assert(this.dtype === dtypes.bfloat16)
    return this.to('LLVM' as any).bitcast(dtypes.uint16).cast(dtypes.uint32).mul(1 << 16).bitcast(dtypes.float32).cast(dtype)
  }

  /**
   * Casts `this` to the given `dtype`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1, 2.5, 3]), dtype=dtypes.number)
   * console.log(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.cast(dtypes.int32)
   * console.log(t.dtype, t.numpy())
   * ```
   */
  cast = (dtype: DTypeLike): Tensor => {
    const dt = to_dtype(dtype)
    return this.dtype === dt ? this : Cast.apply(this, dt)
  }
  /**
   * Bitcasts `this` to the given `dtype` of the same itemsize.
   *
   * `this` must !require a gradient.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1, 2, 3]), dtype=dtypes.int32)
   * console.log(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.bitcast(dtypes.uint32)
   * console.log(t.dtype, t.numpy())
   * ```
   */
  bitcast = (dtype: DTypeLike): Tensor => {
    if (this.requires_grad) throw new Error("can't backprop through bitcast")
    const dt = to_dtype(dtype)
    const ns = dt.itemsize, os = this.dtype.itemsize
    if ((typeof this.device !== 'string' || !this.device.startsWith('DISK')) && ns !== os) {
      if (((this.shape.at(-1)! as number) * os) % ns !== 0) throw new Error('unsupported size in bitcast')
      const [new_uint, old_uint] = [to_dtype(`uint${8 * ns}`), to_dtype(`uint${8 * os}`)]
      const tmp = this.bitcast(old_uint)
      if (ns > os) return range(idiv(ns, os)).map((i) => tmp.get('...', { start: i, step: idiv(ns, os) }).cast(new_uint).lshift(8 * i * os)).reduce((acc, x) => acc.add(x)).bitcast(dtype)
      return Tensor.stack(range(idiv(os, ns)).map((i) => tmp.rshift(8).mul(i).mul(ns)), -1).flatten(-2).cast(new_uint).bitcast(dtype)
    }
    return this.dtype !== dt ? Cast.apply(this, dt, true) : this
  }
  /**
   * Convenience method to cast `this` to a `float32` Tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1, 2, 3]), dtype=dtypes.int32)
   * console.log(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.number()
   * console.log(t.dtype, t.numpy())
   * ```
   */
  float = (): Tensor => {
    return this.cast(dtypes.float32)
  }
  /**
   * Convenience method to cast `this` to a `float16` Tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1, 2, 3]), dtype=dtypes.int32)
   * console.log(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.half()
   * console.log(t.dtype, t.numpy())
   * ```
   */
  half = (): Tensor => {
    return this.cast(dtypes.float16)
  }
  /**
   * Convenience method to cast `this` to a `int32` Tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1.5, -0.5, 0.0, 0.5, 1.5]))
   * console.log(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.number()
   * console.log(t.dtype, t.numpy())
   * ```
   */
  int = (): Tensor => {
    return this.cast(dtypes.int32)
  }
  /**
   * Convenience method to cast `this` to a `boolean` Tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor([-1, 0, 1]))
   * console.log(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.boolean()
   * console.log(t.dtype, t.numpy())
   * ```
   */
  boolean = (): Tensor => {
    return this.cast(dtypes.bool)
  }
}
