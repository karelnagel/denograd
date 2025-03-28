// deno-lint-ignore-file no-this-alias
import { type ConstType, DType, type DTypeLike, dtypes, ImageDType, least_upper_dtype, least_upper_float, sum_acc_dtype, to_dtype } from './dtype.ts'
import { _METADATA, all_int, all_same, assert, bytes_to_bigint, dedup, div, flatten, fully_flatten, int_to_bytes, is_eq, isConst, list_str, max, Metadata, min, mod, NotImplemented, num, product, random_id, range, type Slice, slice, sorted, vars, WeakValueMap, zip } from './helpers/helpers.ts'
import { identity_element, MathTrait, Ops, resolve, type sint, smax, smin, UOp, type Variable } from './ops.ts'
import { add, ceildiv, ge, gt, idiv, le, mul, ne, polyN, prod, sub, sum } from './helpers/helpers.ts'
import { BufferSpec, Device, uop_buffer, uop_is_realized, uop_realized } from './device.ts'
import { create_schedule_with_vars, type ScheduleItem } from './engine/schedule.ts'
import { memory_planner } from './engine/memory.ts'
import { run_schedule } from './engine/realize.ts'
// // **** start with two base classes, Tensor && Function ****
import { isInt, make_tuple, round_up } from './helpers/helpers.ts'
import { argsort } from './helpers/helpers.ts'
import { MemoryView } from './helpers/memoryview.ts'
import { env } from './env/index.ts'
import { compute_gradient } from './gradient.ts'
import { get_multi_map } from './multi.ts'

export const all_tensors = new WeakValueMap<string, Tensor>()

const _apply_map_to_tensors = (applied_map: Map<UOp, UOp>): undefined => {
  // get all children of keys in applied_map
  const all_uops = new Set<UOp>()
  let search_uops = [...applied_map.keys()]
  while (search_uops.length) {
    let x = search_uops.shift()!
    if (all_uops.has(x)) continue
    all_uops.add(x)
    search_uops = [...search_uops, ...[...x.children.values()].map((c) => c).filter((u) => u !== undefined)]
  }

  // link the found UOps back to Tensors. exit early if there's no Tensors to realize
  // NOTE: this uses all_tensors, but it's fast
  const fixed_tensors = [...all_tensors.values()].map((tref) => tref).filter((t) => t !== undefined && all_uops.has(t.lazydata))

  if (fixed_tensors.length) {
    // potentially rewrite all the discovered Tensors
    const sink = UOp.sink(...fixed_tensors.map((t) => t.lazydata))
    const new_sink = sink.substitute(applied_map)

    // set the relevant lazydata to the realized UOps
    for (const [t, s, ns] of zip(fixed_tensors, sink.src, new_sink.src)) {
      if (s === ns) continue
      t.lazydata = ns
    }
  }
}

type ReplaceUOpsWithTensor<Args extends any[]> = { [K in keyof Args]: Args[K] extends UOp ? Tensor : Args[K] }
// Has to be a function cause you can't use generics for static functions
function CreateFunction<Args extends any[] = [UOp]>() {
  return class Function {
    needs_input_grad: (boolean | undefined)[]
    requires_grad?: boolean
    parents?: Tensor[]
    constructor(public device: string | string[], tensors: Tensor[], public metadata?: Metadata) {
      this.needs_input_grad = tensors.map((t) => t.requires_grad)
      this.requires_grad = this.needs_input_grad.some((x) => x) ? true : this.needs_input_grad.includes(undefined) ? undefined : false
      if (this.requires_grad) this.parents = tensors
    }
    forward = (..._args: Args): UOp => {
      throw new Error(`forward !implemented for ${this}`)
    }
    backward = (_grad_output: UOp): UOp | (UOp | undefined)[] => {
      throw new Error(`backward !implemented for ${this}`)
    }

    static apply(...args: ReplaceUOpsWithTensor<Args>): Tensor {
      if (!args.length) throw new Error('No args')

      const x = args.filter((x) => x instanceof Tensor)
      const ctx = new this(x[0].device, x, _METADATA.get())

      const ret = new Tensor(undefined, undefined, true)
      // @ts-expect-error it doesn't like the spread arguments
      ret.lazydata = ctx.forward(...args.map((v) => v instanceof Tensor ? v.lazydata : v))
      ret.requires_grad = ctx.requires_grad
      ret.grad = undefined

      ret._ctx = ctx.requires_grad && !Tensor.no_grad ? ctx : undefined // used by autograd engine
      return ret
    }
  }
}
// ************* function.py start *************
/**
 * This === where the forwards && backwards passes live.
 */

export class Contiguous extends CreateFunction() {
  override forward = (x: UOp): UOp => x.contiguous()
  override backward = (grad_output: UOp): UOp => grad_output
}

export class ContiguousBackward extends CreateFunction() {
  override forward = (x: UOp): UOp => x.contiguous_backward()
  override backward = (grad_output: UOp): UOp => grad_output.contiguous()
}

export class Cast extends CreateFunction<[UOp, DType, boolean?]>() {
  input_dtype!: DType
  bitcast?: boolean
  override forward = (x: UOp, dtype: DType, bitcast?: boolean): UOp => {
    this.input_dtype = x.dtype, this.bitcast = bitcast
    return this.bitcast ? x.bitcast(dtype) : x.cast(dtype)
  }
  override backward = (grad_output: UOp): UOp => {
    if (this.bitcast) throw new Error('bitcast can not backward')
    return grad_output.cast(this.input_dtype!)
  }
}

// // ************* unary ops *************

export class Reciprocal extends CreateFunction() {
  ret!: UOp
  override forward = (x: UOp): UOp => {
    this.ret = x.reciprocal()
    return this.ret
  }

  override backward = (grad_output: UOp): UOp => grad_output.neg().mul(this.ret).mul(this.ret)
}
export class Sin extends CreateFunction() {
  x!: UOp
  override forward = (x: UOp): UOp => {
    this.x = x
    return x.sin()
  }
  override backward = (grad_output: UOp): UOp => (this.x.sub(Math.PI / 2, true)).sin().mul(grad_output)
}
export class Relu extends CreateFunction() {
  ret!: UOp
  override forward = (x: UOp): UOp => {
    this.ret = (x.gt(0)).where(x, 0)
    return this.ret
  }
  override backward = (grad_output: UOp): UOp => (this.ret.gt(0)).cast(grad_output.dtype).mul(grad_output)
}
export class Log extends CreateFunction() {
  x!: UOp
  override forward = (x: UOp): UOp => {
    this.x = x
    return x.log2().mul(Math.log(2))
  }
  override backward = (grad_output: UOp): UOp => grad_output.div(this.x)
}
export class Exp extends CreateFunction() {
  ret!: UOp
  override forward = (x: UOp): UOp => {
    this.ret = (x.mul(1 / Math.log(2))).exp2()
    return this.ret
  }
  override backward = (grad_output: UOp): UOp => this.ret.mul(grad_output)
}
export class Sqrt extends CreateFunction() {
  ret!: UOp
  override forward = (x: UOp): UOp => {
    this.ret = x.sqrt()
    return this.ret
  }
  override backward = (grad_output: UOp): UOp => grad_output.div(this.ret.mul(2))
}

export class Sign extends CreateFunction() {
  // NOTE: the x*0 is to match torch behavior without function.py
  override forward = (x: UOp): UOp => x.ne(0).where((x.lt(0)).where(x.const_like(-1), x.const_like(1)), x.const_like(0)).add(x.mul(0))
  //   // backward always return 0 to match torch
  override backward = (grad_output: UOp): UOp => grad_output.const_like(0)
}
// // ************* binary ops *************

export class Less extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.lt(y)
  override backward = (_grad_output: UOp): [UOp | undefined, UOp | undefined] => [undefined, undefined]
}
export class Neq extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.ne(y)
  override backward = (_grad_output: UOp): [UOp | undefined, UOp | undefined] => [undefined, undefined]
}
export class Xor extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.xor(y)
}
export class BitwiseAnd extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.bitwise_and(y)
}
export class BitwiseOr extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.bitwise_or(y)
}
export class Threefry extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, seed: UOp): UOp => x.threefry(seed)
}

export class Add extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.add(y)

  override backward = (grad_output: UOp): [UOp | undefined, UOp | undefined] => [this.needs_input_grad[0] ? grad_output : undefined, this.needs_input_grad[1] ? grad_output : undefined]
}
export class Mul extends CreateFunction<[UOp, UOp]>() {
  x!: UOp
  y!: UOp
  override forward = (x: UOp, y: UOp): UOp => {
    this.x = x, this.y = y
    return x.mul(y)
  }
  override backward = (grad_output: UOp): [UOp?, UOp?] => [
    this.needs_input_grad[0] ? (this.y.mul(grad_output)) : undefined,
    this.needs_input_grad[1] ? (this.x.mul(grad_output)) : undefined,
  ]
}
export class IDiv extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.idiv(y)
}

export class Mod extends CreateFunction<[UOp, UOp]>() {
  override forward = (x: UOp, y: UOp): UOp => x.mod(y)
}
// // ************* ternary ops *************

export class Where extends CreateFunction<[UOp, UOp, UOp]>() {
  x!: UOp
  override forward = (x: UOp, y: UOp, z: UOp): UOp => {
    this.x = x
    return this.x.where(y, z)
  }
  override backward = (grad_output: UOp): [undefined, UOp | undefined, UOp | undefined] => [
    undefined,
    this.needs_input_grad[1] ? this.x.where(grad_output, grad_output.const_like(0)) : undefined,
    this.needs_input_grad[2] ? this.x.where(grad_output.const_like(0), grad_output) : undefined,
  ]
}
// // ************* reduce ops *************

export class Sum extends CreateFunction<[UOp, number[]]>() {
  input_shape!: sint[]
  override forward = (x: UOp, axis: number[]): UOp => {
    this.input_shape = x.shape
    return x.r(Ops.ADD, axis)
  }
  override backward = (grad_output: UOp): UOp => grad_output.expand(this.input_shape)
}
export class Prod extends CreateFunction<[UOp, number[]]>() {
  x!: UOp
  ret!: UOp
  override forward = (x: UOp, axis: number[]): UOp => {
    this.x = x, this.ret = x.r(Ops.MUL, axis)
    return this.ret
  }
  override backward = (grad_output: UOp): UOp => {
    return (grad_output.mul(this.ret)).expand(this.x.shape).div(this.x)
  }
}
export class Max extends CreateFunction<[UOp, number[]]>() {
  x!: UOp
  ret!: UOp
  axis!: number[]
  override forward = (x: UOp, axis: number[]): UOp => {
    this.x = x, this.ret = x.r(Ops.MAX, axis), this.axis = axis
    return this.ret
  }
  override backward = (grad_output: UOp): UOp => {
    // 1s in locations where the max was chosen (can be two locations)
    const max_is_1s = this.x.ne(this.ret.expand(this.x.shape)).ne(this.x.const_like(1).cast(dtypes.bool)).cast(grad_output.dtype)
    const div = max_is_1s.r(Ops.ADD, this.axis).expand(this.x.shape)
    return (max_is_1s.div(div)).mul(grad_output.expand(this.x.shape))
  }
}
// // ************* movement ops *************

// // NOTE: this === sum in reverse
export class Expand extends CreateFunction<[UOp, sint[]]>() {
  expanded_axis!: number[]
  override forward = (x: UOp, shape: sint[]): UOp => {
    this.expanded_axis = [...zip(x.shape, shape).entries()].filter(([i, [si, so]]) => resolve(ne(si, so))).map(([i]) => i)
    return x.expand(shape)
  }
  override backward = (grad_output: UOp): UOp => {
    return grad_output.cast(sum_acc_dtype(grad_output.dtype)).r(Ops.ADD, this.expanded_axis).cast(grad_output.dtype)
  }
}

export class Reshape extends CreateFunction<[UOp, sint[]]>() {
  input_shape!: sint[]
  override forward = (x: UOp, shape: sint[]): UOp => {
    this.input_shape = x.shape
    return x.reshape(shape)
  }
  override backward = (grad_output: UOp): UOp => grad_output.reshape(this.input_shape)
}
export class Permute extends CreateFunction<[UOp, number[]]>() {
  input_order!: number[]
  override forward = (x: UOp, order: number[]): UOp => {
    this.input_order = order
    return x.permute(order)
  }
  override backward = (grad_output: UOp): UOp => grad_output.permute(argsort(this.input_order))
}
export class Pad extends CreateFunction<[UOp, [number, number][]]>() {
  narg!: [sint, sint][]
  override forward = (x: UOp, arg: [number, number][]): UOp => {
    this.narg = zip(x.shape, arg).map(([s, p]) => [p[0], add(s, p[0])])
    return x.pad(arg)
  }
  override backward = (grad_output: UOp): UOp => grad_output.shrink(this.narg)
}
export class Shrink extends CreateFunction<[UOp, [sint, sint][]]>() {
  narg!: [sint, sint][]
  override forward = (x: UOp, arg: [sint, sint][]): UOp => {
    this.narg = zip(x.shape, arg).map(([s, p]) => [p[0], sub(s, p[1])])
    return x.shrink(arg)
  }
  override backward = (grad_output: UOp): UOp => grad_output.pad(this.narg)
}
export class Flip extends CreateFunction<[UOp, number[]]>() {
  arg!: number[]
  override forward = (x: UOp, axis: number[]): UOp => {
    this.arg = range(x.shape.length).map((i) => axis.includes(i) ? -1 : 1)
    return x.stride(this.arg)
  }
  override backward = (grad_output: UOp): UOp => grad_output.stride(this.arg)
}

// ************* function.py end *************

export const _metaop = (op: Ops, shape: sint[], dtype: DType, device: string | string[], arg?: any) => {
  if (typeof device === 'string') return UOp.metaop(op, shape, dtype, device, arg)
  return UOp.multi(device.map((d) => UOp.metaop(op, shape, dtype, d, arg)), undefined)
}
export const get_shape = (x: any): number[] => {
  //   // NOTE:string === special because __getitem__ on a string === still a string
  if (!Array.isArray(x) || typeof x === 'string') return []
  const subs = x.map((xi) => get_shape(xi))
  if (!all_same(subs)) throw new Error(`inhomogeneous shape from ${x}`)
  return [subs.length, ...(subs.length ? subs[0] : [])]
}
export const _frompy = (x: ConstType[] | Uint8Array, dtype: DType): UOp => {
  let ret, data
  if (x instanceof Uint8Array) [ret, data] = [UOp.metaop(Ops.EMPTY, [idiv(x.length, dtype.itemsize)], dtype, 'JS'), x]
  else {
    ret = UOp.metaop(Ops.EMPTY, get_shape(x), dtype, 'JS')
    if (dtype.fmt === undefined) throw new Error(`${dtype} has undefined fmt`)
    data = MemoryView.fromArray(fully_flatten(x), dtype.fmt)
  }
  //   // fake realize
  uop_buffer(ret)!.allocate(data instanceof Uint8Array ? new MemoryView(data) : data)
  return ret.buf_uop_view()
}

const _get_winograd_matcols = (mat: number[][], dims: number, shp: sint[], device: string | string[], dtype: DType): Tensor[][] => {
  return range(mat[0].length).map((k) => range(dims).map((dim) => Tensor.cat(mat.map((m) => Tensor.full([...shp.slice(0, dim), 1, ...shp.slice(dim + 1)], Number(m[k]), { device: device, dtype: dtype }), dim))))
}
// winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
const _apply_winograd_matrix = (mat: number[][], t: Tensor, dims: number): Tensor => {
  // multiply mat_1 @ mat_2 @ t with foldable constants, where mat_i acts on vector t along dimension i; roughly kron(mat, mat) @ t
  // due to realize-before-expand rule in lazy.py, we must operate in this order: reshape -> expand -> arithmetic
  const t_ = t.reshape([...t.shape.slice(0, dims), ...range(dims).map((x) => 1), ...t.shape.slice(dims)]).expand([...t.shape.slice(0, dims), ...range(dims).map((x) => mat.length), ...t.shape.slice(dims)]) // add output dims
  // precalculate mat columns for each dim; prod(itertools.product(matcols)) gives the columns of kron(mat, mat, ...)
  const matcols = _get_winograd_matcols(mat, dims, t_.shape.slice(dims), t_.device, t_.dtype)
  // multiply each element of t_ by the corresponding stacked column of kron(mat, mat), producing only one view for each element of t
  const ret = sum(product(range(mat[0].length), dims).map((mat_is) => prod(zip(matcols, mat_is).map(([col, idx]) => col[idx])).mul(t.get(mat_is))))
  if (!(ret instanceof Tensor)) throw new Error("sum didn't return a Tensor")
  return ret
}
const _align_left = (...shapes: sint[][]): sint[][] => {
  //   // unsqueeze left to make every shape same length
  const max_dim = max(shapes.map((shape) => shape.length))
  return shapes.map((shape) => [...range(max_dim - shape.length).map(() => 1), ...shape])
}
export const _broadcast_shape = (shapes: sint[][]): sint[] => {
  return zip(..._align_left(...shapes)).map((nth_dim_sizes) => nth_dim_sizes.includes(0) ? 0 : smax(...nth_dim_sizes))
}
const _masked_setitem = (target: Tensor, values: Tensor, mask: Tensor, axes: number[]) => {
  // apply mask to values (already broadcasted) and reduce such that if mask contains repeated indices the last one remains
  values = values.mul(mask)
  for (const dim of axes) [mask, values] = zip(mask.split(1, dim), values.split(1, dim)).reduce(([x, y]) => [x.get(0).bitwise_or(y.get(0)), y.get(0).where(y.get(1), x.get(1))])
  // remove extra dims from reduce
  for (const dim of axes.toReversed()) [mask, values] = [mask.squeeze(dim), values.squeeze(dim)]
  // select from values for each True element in mask else select from self
  return mask.where(values, target)
}
//  `(padding_left, padding_right, padding_top, padding_bottom, ...)` ->  `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
const _flat_to_grouped = (padding: sint[]): [sint, sint][] => zip(slice(padding, { start: -2, step: -2 }), slice(padding, { step: -2 }))

const ReductionStr = ['mean', 'sum', 'none']
type ReductionStr = typeof ReductionStr[number]

export type TensorOptions = { device?: string | string[]; dtype?: DType; requires_grad?: boolean }
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
export type LayerAsync = ((x: Tensor) => Tensor) | { call: (x: Tensor) => Tensor } | ((x: Tensor) => Promise<Tensor>) | { call: (x: Tensor) => Promise<Tensor> }
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
  static registry = new FinalizationRegistry((key: string) => {
    all_tensors.delete(key)
  })
  lazydata!: UOp
  requires_grad?: boolean
  // tensors can have gradients if you have called .backward
  grad?: Tensor
  // internal variable used for autograd graph construction
  _ctx?: InstanceType<ReturnType<typeof CreateFunction>>
  static training = false
  static no_grad = false
  key: string
  // KAREL: TODO: this probably won't work correctly
  constructor(data?: ConstType | UOp | Uint8Array | any[] | UOp | Tensor | string, { device, dtype, requires_grad }: TensorOptions = {}, skip_constructor = false) {
    super()
    this.key = random_id()
    all_tensors.set(this.key, this)
    Tensor.registry.register(this, this.key)
    if (skip_constructor) return
    if (dtype !== undefined) dtype = to_dtype(dtype)
    if (dtype !== undefined && !(dtype instanceof DType)) throw new Error(`invalid dtype ${dtype}`)
    if (device === undefined && typeof data === 'string') {
      device = `DISK:${data}`
    } // keep it on the disk if device === undefined
    device = Array.isArray(device) ? device.map((x) => Device.canonicalize(x)) : Device.canonicalize(device)

    //     // NOTE: this can be in three states. false && undefined: no gradient, true: gradient
    //     // undefined (the default) will be updated to true if it's put in an optimizer
    this.requires_grad = requires_grad

    //     // create a LazyBuffer from the different types of inputs
    if (data instanceof UOp) {
      if (dtype !== undefined && dtype !== data.dtype) throw new Error("dtype doesn't match, && casting isn't supported")
      // NOTE: this is here because LazyBuffer = UOp
      if (data instanceof UOp && data.op === Ops.BIND) data = _metaop(Ops.BIND, [], dtype || data.dtype, device, data)
    } else if (data === undefined) {
      data = _metaop(Ops.EMPTY, [0], dtype || dtypes.default_float, device)
    } else if (isConst(data)) {
      data = _metaop(Ops.CONST, [], dtype || dtypes.from_js(data), device, data)
    } else if (data instanceof Uint8Array) {
      data = _frompy(data, dtype || dtypes.uint8)
    } else if (Array.isArray(data)) {
      if (data.some((x) => typeof x === 'string')) throw new Error(`There's a string in data: ${list_str(data)}`)
      if (dtype === undefined) {
        const d = fully_flatten(data)
        if (d.length && d.every((s) => typeof s === 'boolean')) dtype = dtypes.bool
        else dtype = (d.length && all_int(d)) ? dtypes.default_int : dtypes.default_float // NOTE: this works because all_int([True, False]) is True
      }
      if (dtype === dtypes.bfloat16) data = new Tensor(_frompy(data, dtypes.float32), { device }).cast(dtypes.bfloat16).lazydata
      else data = _frompy(data, dtype)
    } else if (typeof data === 'string') {
      dtype = dtype || dtypes.uint8
      data = _metaop(Ops.EMPTY, [idiv(env.statSync(data).size, dtype.itemsize)], dtype, `DISK:${data}`)
    }

    // by this point, it has to be a LazyBuffer
    if (!(data instanceof UOp)) throw new Error(`can't create Tensor from ${data} with type ${typeof data}`)

    // data might be on a different device
    if (typeof device === 'string') this.lazydata = data.device === device ? data : data.copy_to_device(device)
    else if (data instanceof UOp && typeof data.device === 'string') this.lazydata = new Tensor(data).shard(device).lazydata
    else {
      if (data.device !== device) throw new Error(`MultiLazyBuffer device mismatch, ${data.device} != ${device}`)
      this.lazydata = data
    }
  }
  requires_grad_ = (requires_grad: boolean | undefined): Tensor => {
    this.requires_grad = requires_grad
    return this
  }
  override toString = () => {
    const ld = this.lazydata
    const ld_repr = `<UOp ${ld.device} ${list_str(ld.shape)} ${ld.dtype.toString().slice(7)} ${ld.base !== ld ? ld.st : list_str([ld.op, uop_realized(ld)])}>`
    return `<Tensor ${ld_repr} on ${this.device} with grad ${this.grad?.lazydata}>`
  };
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }

  //   // Python has a non moving GC, so this should be okay
  // const __hash__ = () =>  id(this)
  get length() {
    if (!this.shape.length) throw new Error('len() of a 0-d tensor')
    return this.shape[0]
  }
  get device(): string | string[] {
    return this.lazydata.device
  }
  get shape(): sint[] {
    return this.lazydata.shape
  }
  get shape_num(): number[] {
    if (this.lazydata.shape.some((x) => typeof x !== 'number')) throw new Error(`Shape has UOps`)
    return this.lazydata.shape as number[]
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
    let big_sink = UOp.sink(...[this, ...lst].map((x) => x.lazydata))

    // TODO: move this to scheduler tensor_map pass
    if ([...big_sink.toposort].some((x) => x.op === Ops.MULTI)) {
      // multi fixup
      _apply_map_to_tensors(get_multi_map(big_sink))
      big_sink = UOp.sink(...flatten([this, ...lst].map((x) => x.lazydata.op === Ops.MULTI ? x.lazydata.src : [x.lazydata])))
    }
    const [schedule, var_vals, becomes_map] = create_schedule_with_vars(big_sink)
    _apply_map_to_tensors(becomes_map)
    return [memory_planner(schedule), var_vals]
  }

  _debug_ast = () => {
    const [schedule] = this.schedule_with_vars()
    return schedule.map((x) => x.ast)
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
  realize = async (lst: Tensor[] = [], do_update_stats = true): Promise<Tensor> => {
    await run_schedule(...this.schedule_with_vars(lst), do_update_stats)
    return this
  }
  static realize = async (lst: Tensor[], do_update_stats = true) => await lst[0].realize(lst.slice(1), do_update_stats)
  /**
   * Replaces the data of this tensor with the data of another tensor. Only the shape of the tensors must match.
   */
  replace = (x: Tensor): Tensor => {
    // used for replacing a Tensor with a new version of it (potentially with a different device && dtype)
    assert(this._ctx === undefined)
    if (!is_eq(this.shape, x.shape)) throw new Error(`replace shape mismatch ${this.shape} !== ${x.shape}`)
    this.lazydata = x.lazydata
    return this
  }
  assign_disk = async (x: Tensor | number[] | string | Uint8Array): Promise<Tensor> => {
    if (!(x instanceof Tensor)) x = new Tensor(x, { device: this.device, dtype: this.dtype })
    if (typeof this.device === 'string' && !this.device.startsWith('DISK')) throw new Error('This can be only used with DISK device')
    const realized = await this.contiguous().realize()
    uop_realized((realized.lazydata as UOp).base)!.copyin(await x._data())
    return this
  }
  assign = (x: Tensor | number[] | number | string | Uint8Array): Tensor => {
    if (!(x instanceof Tensor)) x = new Tensor(x, { device: this.device, dtype: this.dtype })
    //   // TODO: this is a hack for writing to DISK. remove with working assign
    if (typeof this.device === 'string' && this.device.startsWith('DISK')) throw new Error("Use async assign_disk instead, until disk get's good assign")
    if (vars.DEBUG >= 4) console.log(`assign ${this.lazydata} <- ${x.lazydata}`)
    if (this.lazydata === x.lazydata) return this // a this assign === a NOOP
    // NOTE: we allow cross device assign
    if (!is_eq(this.shape, x.shape)) throw new Error(`assign shape mismatch ${this.shape} !== ${x.shape}`)
    if (this.device !== x.device) throw new Error(`assign device mismatch ${this.device} !== ${x.device}`)
    if (this.dtype !== x.dtype) throw new Error(`assign dtype mismatch ${this.dtype} !== ${x.dtype}`)
    if (x.requires_grad) throw new Error("assign can't have grad") // this requires_grad === okay?
    if (!uop_is_realized(this.lazydata)) return this.replace(x)
    this.lazydata = this.lazydata.assign(x.lazydata as any)
    return this
  }
  /**
   * Returns a new tensor with the same data as this tensor, but detached from the autograd graph.
   */
  detach = (): Tensor => {
    return new Tensor(this.lazydata.detach(), { device: this.device, requires_grad: false })
  }
  _data = async (): Promise<MemoryView> => {
    if (this.shape.includes(0)) return new MemoryView(0)
    // NOTE: this realizes on the object from as_buffer being a Python object
    const cpu = await this.cast(this.dtype.base).contiguous().to(env.CPU_DEVICE).realize()
    const buf = uop_realized((cpu.lazydata as UOp).base)
    if (buf === undefined) throw new Error(`${cpu.lazydata.base} was not realized`)
    if (this.device !== env.CPU_DEVICE) buf!.options = new BufferSpec(undefined, undefined, undefined, undefined, true)
    return await buf!.as_buffer(this.device !== env.CPU_DEVICE ? true : false)
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
    if (this.dtype.base.fmt === undefined) throw new Error(`no fmt dtype for ${this.dtype.base}`)
    if (!all_int(this.shape)) throw new Error(`no data if shape === symbolic, ${this.shape}`)
    // if (TYPE_CHECKING ) assert(this.dtype.base.fmt !== "e")
    return (await this._data()).cast(this.dtype.base.fmt!, this.shape.includes(0) ? undefined : this.shape_num)
  }
  /**
   * Returns the value of this tensor as a standard Python number.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = new Tensor(42)
   * console.log(t.item())
   * ```
   */
  item = async <T = number>(): Promise<T> => {
    if (this.numel() !== 1) throw new Error('must have one element for item')
    return (await this.data()).getValue(...range(this.shape.length || 1).map(() => 0)) as T
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
    return (await this.data()).toList() as T
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
  to = (device?: string | string[]): Tensor => {
    device = Array.isArray(device) ? device.map((x) => Device.canonicalize(x)) : Device.canonicalize(device)
    if (device === this.device) return this
    if (typeof device !== 'string') return this.shard(device)
    const ret = new Tensor(this.lazydata, { device, requires_grad: this.requires_grad })
    if (this.grad !== undefined) ret.grad = this.grad.to(device)
    if (this._ctx !== undefined) ret._ctx = this._ctx
    return ret
  }
  /**
   * Moves the tensor to the given device in place.
   */
  to_ = (device?: string | string[]) => {
    const real = this.to(device)
    if (this.grad !== undefined && real.grad !== undefined) this.grad.replace(real.grad)
    return this.replace(real)
  }

  /**
   * Shards the tensor across the given devices. Optionally specify which axis to shard on.
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.empty(2, 4)
   * print(t.shard((t.device, t.device), axis=1).lazydata)
   * ```
   */
  shard = (devices: string[], axis?: number): Tensor => {
    if (Array.isArray(this.device)) throw new Error("can't shard a MultiLazyBuffer")
    devices = devices.map((x) => Device.canonicalize(x))
    const mlb = this.lazydata.shard(devices, axis !== undefined ? this._resolve_dim(axis) : undefined)
    return new Tensor(mlb, { device: devices, requires_grad: this.requires_grad })
  }
  /**
   * Shards the tensor across the given devices in place.
   */
  shard_ = (devices: string[], axis?: number) => {
    return this.replace(this.shard(devices, axis))
  }
  static from_uop = (y: UOp, opts: TensorOptions = {}): Tensor => {
    if (y.op === Ops.BIND) return new Tensor(y, { ...opts, requires_grad: false }) // this is the only UOp allowed in Tensor
    if (y.op === Ops.CONST) return new Tensor(y.arg, { ...opts, requires_grad: false })
    if (y.op === Ops.MUL) return Tensor.from_uop(y.src[0]).mul(Tensor.from_uop(y.src[1]))
    if (y.op === Ops.ADD) return Tensor.from_uop(y.src[0]).add(Tensor.from_uop(y.src[1]))
    throw new Error(`unhandled UOp ${y}`)
  }
  // ***** creation entrypoint *****

  static _metaop = (op: Ops, shape: sint[], { dtype, device, ...opts }: TensorOptions, arg?: any) => {
    dtype = dtype !== undefined ? to_dtype(dtype) : dtypes.default_float
    if (Array.isArray(device)) {
      return new Tensor(UOp.multi(device.map((d) => UOp.metaop(op, shape, dtype, Device.canonicalize(d), arg)), undefined), { device, dtype, ...opts })
    }
    return new Tensor(UOp.metaop(op, shape, dtype, Device.canonicalize(device), arg), { device, dtype, ...opts })
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
  static empty = (shape: number[], opts: TensorOptions = {}) => Tensor._metaop(Ops.EMPTY, shape, opts)
  /**
   * Exposes the pointer as a Tensor without taking ownership of the original data.
   * The pointer must remain valid for the entire lifetime of the created Tensor.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   */
  static from_blob = (ptr: bigint, shape: number[], opts: TensorOptions): Tensor => {
    const r = Tensor._metaop(Ops.EMPTY, shape, opts)
    uop_buffer(r.lazydata as UOp).allocate(undefined, ptr)
    ;(r.lazydata as UOp).buf_uop_view()
    return r
  }
  /**
   * Create a Tensor from a URL.
   *
   * This === the preferred way to access Internet resources.
   * It currently returns a DISK Tensor, but in the future it may return an HTTP Tensor.
   * This also will soon become lazy (when possible) && !print progress without DEBUG.
   *
   * THe `gunzip` flag will gzip extract the resource && return an extracted Tensor.
   */

  static from_url = async (url: string, opts?: TensorOptions): Promise<Tensor> => {
    let res = await fetch(url)
    if (!res.ok) throw new Error(`Failed to get ${url}`)
    let data = await res.clone().arrayBuffer()
    // checking if it is gzipped, using this instead of a flag cause, sometimes fetch automatically ungzips
    const preview = new Uint8Array(data.slice(0, 2))
    if (preview.length === 2 && preview[0] === 0x1f && preview[1] === 0x8b) {
      data = await env.gunzip(res)
    }
    return new Tensor(new Uint8Array(data), opts)
  }
  static from_file = async (path: string, opts?: TensorOptions): Promise<Tensor> => {
    let data = await env.readFile(path)
    return new Tensor(data, opts)
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
    counts0 = (x.bitwise_and(0xffffffff)).cast(dtypes.uint32), counts1 = ((x.rshift(32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)
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
  static rand = (shape: number[], contiguous = true, { device, dtype, ...opts }: TensorOptions = {}): Tensor => {
    dtype = to_dtype(dtype || dtypes.default_float)
    if (!dtypes.is_float(dtype)) throw new Error(`rand only supports float dtypes, got ${dtype}`)
    if (!all_int(shape) || !shape.every((s) => s >= 0)) throw new Error(`invalid input ${shape}`)
    if (device !== undefined && typeof device !== 'string') throw new Error(`rand only supports single device, got ${device}`)
    device = Device.canonicalize(device)
    const _device = device

    // if shape has 0, return zero tensor
    const numel = prod(shape)
    if (numel === 0) return Tensor.zeros(shape, { device: _device, dtype: dtype, ...opts })
    const num = ceildiv(numel * dtype.itemsize, 4)

    // when using MOCKGPU && NV generate rand on CLANG
    if (vars.get('MOCKGPU') && device.startsWith('NV')) device = env.CPU_DEVICE

    // generate per device seeds && rng counter if we haven't seen this device yet
    if (!Tensor._device_seeds[device]) {
      Tensor._device_seeds[device] = new Tensor(
        [mod(bytes_to_bigint(env.sha256(int_to_bytes(Object.keys(Tensor._device_seeds).length))), 2n ** 32n), Tensor._seed],
        { device: device, dtype: dtypes.uint32, requires_grad: false },
      )
      Tensor._device_rng_counters[device] = new Tensor([0], { device: device, dtype: dtypes.uint32, requires_grad: false })
    } // increment rng counter for devices
    else Tensor._device_rng_counters[device].assign(Tensor._device_rng_counters[device].add(num)).contiguous()

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
    // TODO: adding contiguous() after bits. fixes it for wasm
    let out = bits.bitcast(dtype).get({ stop: numel }).sub(1).reshape(shape)

    // move back to the original device if we were using MOCKGPU
    if (vars.get('MOCKGPU') && _device) out = out.to(_device)

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
    return new Tensor(fill_value, opts).reshape(range(shape.length).map(() => 1)).expand(shape)
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
    return Tensor.full(shape, 0.0, { dtype: dtypes.float, ...opts })
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
    return Tensor.full(shape, 1.0, { dtype: dtypes.float, ...opts })
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
   * Returns a 1-D tensor of `steps` evenly spaced values from `start` to `stop`, inclusive.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.linspace(0, 10, 5).numpy())
   * ```
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.linspace(-1, 1, 5).numpy())
   * ```
   */
  static linspace = (start: Tensor | number, stop: Tensor | number, steps: number, { dtype = dtypes.default_float, ...opts }: TensorOptions = {}): Tensor => {
    if (steps < 0) throw new Error('number of steps must be non-negative')
    dtype = to_dtype(dtype)
    if (dtype === dtypes.bool) throw new Error('linspace with bool dtype is not supported')
    if (steps === 1) return new Tensor([start], { dtype: dtype, ...opts })
    return Tensor.arange(steps, undefined, undefined, opts).mul(div(sub(stop, start), sub(steps, 1))).add(start, true).cast(dtype)
  }

  /**
   * Returns a 2-D tensor with `n` rows and `m` columns, with ones on the diagonal and zeros elsewhere.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.eye(3).numpy())
   * ```
   *  ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.eye(2, 4).numpy())
   * ```
   */
  static eye = (n: number, m?: number, opts: TensorOptions = {}): Tensor => {
    if (n < 0 || (m !== undefined && m < 0)) throw new Error(`cannot have negative n=${n}, m=${m}`)
    const x = Tensor.ones([n, 1], opts).pad([undefined, [0, n]]).flatten().shrink([[0, n * n]]).reshape([n, n])
    return m === undefined ? x : m > n ? x.pad([undefined, [0, m - n]]) : x.shrink([undefined, [0, m]])
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
   * Creates a tensor with the same shape as `self`, filled with zeros.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.ones(2, 3)
   * print(Tensor.zeros_like(t).numpy())
   * ```
   */
  zeros_like = (opts: TensorOptions): Tensor => {
    return this.full_like(0, opts)
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
  /**
   * Creates a tensor with the same shape and sharding as `self`, filled with random values from a uniform distribution over the interval `[0, 1)`.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.ones(2, 3)
   * print(Tensor.rand_like(t).numpy())
   * ```
   */
  rand_like = ({ dtype = this.dtype, contiguous = true, ...opts }: TensorOptions & { contiguous?: boolean } = {}): Tensor => {
    if (Array.isArray(this.device)) {
      if (opts.device !== undefined) throw new Error('cannot specify `device` on `rand_like` of a multi device tensor')
      if (this.lazydata.axis === undefined) return Tensor.rand(this.shape_num, undefined, { dtype: dtype, ...opts }).shard(this.device)

      const rands = this.lazydata.src.map((lb) => {
        if (lb.shape.some((x) => typeof x !== 'number')) throw new Error(`${lb.shape}`)
        return Tensor.rand(lb.shape as number[], contiguous, { device: lb.device, dtype: dtype, ...opts }).lazydata
      })

      return new Tensor(UOp.multi(rands, this.lazydata.axis), { device: this.device, dtype: dtype, ...opts })
    }
    return Tensor.rand(this.shape_num, undefined, { device: this.device, dtype: dtype, ...opts })
  }

  // ***** rng hlops *****
  /**
   * Creates a tensor with the given shape, filled with random values from a normal distribution with mean `0` and standard deviation `1`.
   * If `dtype` is not specified, the default type is used.
   *
   * You can pass in the `device` keyword argument to control device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * print(Tensor.randn(2, 3).numpy())
   * ```
   */
  static randn = (shape: number[], { dtype, requires_grad, ...opts }: TensorOptions = {}): Tensor => {
    // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    const src = Tensor.rand([2, ...shape], undefined, { ...opts, dtype: dtype || dtypes.float32 })
    return (src.get(0).mul(2 * Math.PI).cos().mul(src.get(1).sub(1, true).log().mul(-2).sqrt()).cast(dtype || dtypes.default_float)).requires_grad_(requires_grad)
  }

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
  static randint = (shape: number[], low = 0, high = 10, { dtype = dtypes.int32, ...opts }: TensorOptions = {}): Tensor => {
    if (!Number.isInteger(low) || !Number.isInteger(high)) throw new Error(`${low} && ${high} must be integers`)
    dtype = to_dtype(dtype)
    if (!dtypes.is_int(dtype)) throw new Error(`${dtype} must be int`)
    return Tensor.uniform(shape, low, high, { ...opts, dtype })
  }
  /**
   * Creates a tensor with the given shape, filled with random values from a normal distribution with the given `mean` and standard deviation `std`.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * print(Tensor.normal(2, 3, mean=10, std=2).numpy())
   * ```
   */
  static normal = (shape: number[], mean = 0.0, std = 1.0, { requires_grad, ...opts }: TensorOptions): Tensor => {
    return (Tensor.randn(shape, opts).mul(std, true)).add(mean).requires_grad_(requires_grad)
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
  static uniform = (shape: number[], low = 0.0, high = 1.0, { dtype, requires_grad, ...opts }: TensorOptions = {}): Tensor => {
    return Tensor.rand(shape, undefined, opts).mul(high - low, true).cast(dtype || dtypes.default_float).add(low).requires_grad_(requires_grad)
  }
  /**
   * Creates a tensor with the given shape, filled with random values from a uniform distribution
   * over the interval `[-prod(shape)**-0.5, prod(shape)**-0.5)`.
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * print(Tensor.scaled_uniform(2, 3).numpy())
   * ```
   */
  static scaled_uniform = (shape: number[], opts: TensorOptions): Tensor => {
    return Tensor.uniform(shape, -1.0, 1.0, opts).mul(prod(shape) ** -0.5)
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
  static glorot_uniform = (shape: number[], opts: TensorOptions = {}): Tensor => {
    return Tensor.uniform(shape, -1.0, 1.0, opts).mul((6 / (shape[0] + prod(shape.slice(1)))) ** 0.5)
  }

  // https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_uniform_
  /**
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * print(Tensor.kaiming_uniform(2, 3).numpy())
   * ```
   */
  static kaiming_uniform = (shape: number[], a = 0.01, opts: TensorOptions): Tensor => {
    const bound = Math.sqrt(3.0) * Math.sqrt(2.0 / (1 + a ** 2)) / Math.sqrt(prod(shape.slice(1)))
    return Tensor.uniform(shape, -bound, bound, opts)
  }
  // https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_
  /**
   * <https://pytorch.org/docs/stable/_modules/torch/nn/init.html#kaiming_normal_>
   *
   * You can pass in `dtype` and `device` keyword arguments to control the data type and device of the tensor.
   * Additionally, all other keyword arguments are passed to the constructor of the tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * Tensor.manual_seed(42)
   * print(Tensor.kaiming_normal(2, 3).numpy())
   * ```
   */
  static kaiming_normal = (shape: number[], a = 0.01, opts: TensorOptions): Tensor => {
    const std = Math.sqrt(2.0 / (1 + a ** 2)) / Math.sqrt(prod(shape.slice(1)))
    return Tensor.normal(shape, 0.0, std, opts)
  }

  multinomial = (num_samples = 1, replacement = false): Tensor => {
    if (!(1 <= this.ndim && this.ndim <= 2 && num_samples > 0)) throw new Error(`ndim=${this.ndim} must be 1 or 2 dim, num_samples=${num_samples} must be positive`)
    if (!replacement && num_samples !== 1) throw new Error('no replacement only supports num_samples = 1')
    const weight = this.ndim === 1 ? this.unsqueeze(0) : this
    const cw = weight.cumsum(1).float(), cdf = cw.div(cw.get({}, -1).unsqueeze(1))
    const unif_samples = Tensor.rand([num_samples, cdf.shape_num[0], 1]).to(this.device)
    const indices = unif_samples.expand([-1, -1, cdf.shape[1]]).ge(cdf).sum(2).permute(1, 0)
    return (this.ndim === 1 ? indices.squeeze(0) : indices).cast(dtypes.int32)
  }

  // // ***** toposort and backward pass *****

  /**
   * Compute the gradient of the targets with respect to self.
   * ```python exec="true" source="above" session="tensor" result="python"
   * x = Tensor.eye(3)
   * y = Tensor([[2.0,0,-2.0]])
   * z = y.matmul(x).sum()
   * dx, dy = z.gradient(x, y)
   *
   * print(dx.tolist())  // dz/dx
   * print(dy.tolist())  // dz/dy
   * ```
   */
  gradient = (targets: Tensor[], gradient?: Tensor): Tensor[] => {
    if (gradient === undefined && this.shape.length !== 0) 'when no gradient is provided, backward must be called on a scalar tensor'
    if (gradient === undefined) gradient = new Tensor(1.0, { dtype: this.dtype, device: this.device, requires_grad: false })
    let rets: UOp[][] = []
    const target_uops = targets.map((x) => x.lazydata)
    const grads = compute_gradient(this.lazydata, gradient.lazydata, new Set(target_uops))
    const ret: UOp[] = []
    for (const x of target_uops) {
      const y = grads.get(x)
      if (y === undefined) throw new Error(`${x}\n\nnot found in\n\n${this.lazydata}`)
      ret.push(y)
    }
    rets.push(ret)
    // create returned Tensors
    return zip(targets, rets[0]).map(([t, u]) => new Tensor(u, { device: t.device }))
  }
  _deepwalk = (): Tensor[] => {
    const _walk = (node: Tensor, visited: Set<Tensor>): Tensor[] => {
      let res: Tensor[] = []
      visited.add(node)
      // if tensor isn't leaf, reset grad
      const ctx = node._ctx
      if (ctx !== undefined && ctx.parents!.length !== 0) node.grad = undefined
      if (ctx) {
        for (const i of node._ctx!.parents!) {
          if (!visited.has(i)) res = [...res, ..._walk(i, visited)]
        }
        res.push(node)
      }
      return res
    }
    return _walk(this, new Set<Tensor>())
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
      if (!is_eq(this.shape, [])) throw new Error('when no gradient is provided, backward must be called on a scalar tensor')
      // fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
      // this === "implicit gradient creation"
      gradient = new Tensor(1.0, { dtype: this.dtype, device: this.device, requires_grad: false })
    }
    const toposort_uop = this.lazydata.toposort
    if (!is_eq(this.shape, gradient.shape)) throw new Error(`grad shape must match tensor shape, ${gradient.shape} !== ${this.shape}`)
    this.grad = gradient
    for (const t0 of toposorted.toReversed()) {
      if (t0.grad === undefined) throw new Error(`tensor ${t0} has no grad`)
      const ctx = t0._ctx!, md = ctx.metadata, token = _METADATA.set(md !== undefined ? new Metadata(md.name, md.caller, true) : undefined)
      const lazys = ctx.backward(t0.grad.lazydata as UOp)
      _METADATA.reset(token)
      const grads = (!Array.isArray(lazys) ? [lazys] : lazys).map((g) => g !== undefined ? new Tensor(g, { device: this.device, requires_grad: false }) : undefined)
      for (const [t, g] of zip(ctx.parents!, grads)) {
        if (g !== undefined && t.requires_grad) {
          if (!is_eq(g.shape, t.shape)) throw new Error(`grad shape must match tensor shape, ${list_str(g.shape)} !== ${list_str(t.shape)}`)
          if (!toposort_uop.has(t.lazydata)) throw new Error(`grad uop must have a path from self\ngrad uop: ${t.lazydata}`)
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
    let new_shape = shape.map((s, i) => s !== undefined ? s : this.shape.at(i)!)
    // resolve -1
    const c = new_shape.filter((x) => x === -1).length
    if (c > 1) throw new Error(`only one dimension can be inferred using -1, getting ${new_shape}`)
    if (c) new_shape = new_shape.map((s) => s === -1 ? idiv(-prod(this.shape_num), prod(new_shape)) : s)
    return !is_eq(new_shape, this.shape) ? Reshape.apply(this, new_shape) : this
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
    const new_shape = zip(..._align_left(this.shape, shape)).map(([from, to]) => to === -1 || to === undefined ? from : to)
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
    if (!is_eq(sorted(order_arg), range(this.ndim))) throw new Error(`order !== a valid permutation, getting ${order_arg}`)
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
  flip = (...axis: number[]): Tensor => {
    const axis_arg = axis.map((x) => this._resolve_dim(x))
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
    if (is_eq(shrink_arg, this.shape.map((s) => [0, s]))) return this
    return Shrink.apply(this, shrink_arg)
  }
  /**
   * Returns a tensor with padding applied based on the input `padding`.
   *
   * `padding` supports two padding structures:
   *
   * 1. Flat padding: `(padding_left, padding_right, padding_top, padding_bottom, ...)`
   * - This structure matches PyTorch's pad.
   * - `padding` length must be even.
   *
   * 2. Group padding: `(..., (padding_top, padding_bottom), (padding_left, padding_right))`
   * - This structure matches pad for JAX, NumPy, TensorFlow and others.
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
    // flat padding
    let pX: [sint, sint][]
    if (padding.every((p) => isInt(p) || p instanceof UOp)) {
      if (mod(padding.length, 2) !== 0) throw new Error('Flat padding must have even number of pads')
      pX = _flat_to_grouped([...padding, ...range(this.ndim - idiv(padding.length, 2)).flatMap((x) => [0, 0])])
    } // group padding
    else pX = padding.map((p) => p === undefined ? [0, 0] : p)
    if (pX.length !== this.ndim) throw new Error(`padding length is improper, padding=${padding} ndim=${this.ndim}`)
    let X: Tensor = this, pads = pX.map(([pB, pA]) => [smax(pB, 0), smax(pA, 0)] as [sint, sint])
    if (mode === 'constant') {
      const _constant = (x: Tensor, px: [number, number][], v: number | bigint | boolean) => v === 0 ? Pad.apply(x, px) : Pad.apply(x, px).add(Pad.apply(x.ones_like(), px).where(0, v))
      return pX.flat().every((p) => resolve(ge(p, 0))) ? _constant(X, pX as [number, number][], value) : _constant(X.shrink(zip(pX, X.shape).map(([[pB, pA], s]) => [-smin(pB, 0), smin(add(pA, s), s)])), pads as [number, number][], value)
    }
    if (!all_int(this.shape)) throw new Error(`does not support symbolic shape ${this.shape}`)
    if (mode === 'circular') {
      if (zip(pX, X.shape_num).some(([[pB, pA], sh]) => num(pB) > sh || num(pA) > sh)) throw new Error('Padding value causes wrapping around more than once.')
      if (pX.some(([pB, pA]) => num(pB) < 0 || num(pA) < 0)) throw new Error('Negative pads with circular pads is not supported')
      const orig_shape = X.shape_num
      X = X.repeat(pads.map(([pB, pA]) => (1 + Number(Boolean(pB)) + Number(Boolean(pA)))))
      return X.shrink(zip(pads, orig_shape, X.shape_num).map(([[pB, pA], osh, xsh]) => [pB === 0 ? 0 : osh - num(pB), pA === 0 ? xsh : xsh - osh + num(pA)]))
    }
    for (const [d, [pB, pA]] of pads.entries()) {
      let xB: Tensor | undefined, xA: Tensor | undefined
      if (mode === 'reflect') {
        const s = X.shape_num[d]
        if (num(pB) >= s || num(pA) >= s) throw new Error(`Padding (${pB}, ${pA}) should be less than the input size=${s} for dim=${d}.`)
        const slcB = { start: pB, stop: 0, step: -1 }, slcA = { start: s - 2 >= 0 ? s - 2 : undefined, stop: s - 2 - num(pA) >= 0 ? s - 2 - num(pA) : undefined, step: -1 }
        ;[xB, xA] = [[slcB, pB], [slcA, pA]].map(([slc, p]) => num(p) > 0 ? X.get(...range(X.ndim).map((i) => i === d ? slc : {})) : undefined)
      }
      if (mode === 'replicate') {
        const shrB = range(X.ndim).map((i) => i === d ? [0, 1] as [number, number] : undefined), shrA = range(X.ndim).map((i) => i === d ? [X.shape_num[i] - 1, X.shape_num[i]] as [number, number] : undefined)
        ;[xB, xA] = ([[shrB, pB], [shrA, pA]] as const).map(([shr, p]) => num(p) > 0 ? X.shrink(shr).expand(range(X.ndim).map((i) => i === d ? p : undefined) as sint[]) : undefined)
      } else throw new Error(`invalid mode ${mode}`)
      X = Tensor.cat([xB, X, xA].filter((X_) => X_ !== undefined), d)
    }
    return X.shrink(zip(pX, X.shape_num).map(([[pB, pA], s]) => [-min([num(pB), 0]), min([num(pA) + s, s])] as [sint, sint]))
  }
  // ***** movement high level ops *****

  _getitem = (indices: TensorIndice[], v?: Tensor): Tensor => {
    // turn scalar Tensors into const val for number indexing if possible
    let x = this as Tensor
    indices = indices.map((i) => (i instanceof Tensor) && i.shape.length === 0 ? num(this._to_const_val(i)) : i)

    // filter ellipsis && fill with slice(undefined) || fill rest of indices with slice(undefined)
    const ellipsis_idx = [...indices.entries().filter(([dim, i]) => i === '...').map(([dim, i]) => dim)]
    if (ellipsis_idx.length > 1) throw new Error('indices can only have a single ellipsis')
    const fill_idx = ellipsis_idx.length ? ellipsis_idx[0] : indices.length
    const num_indices = indices.length - ellipsis_idx.length - indices.filter((i) => i === undefined).length
    if (num_indices > this.ndim) throw new Error(`too many ${num_indices} for ${this.ndim}`)
    indices.splice(fill_idx, 1, ...Array(this.ndim - num_indices).fill({} as Slice))

    let [indices_parsed, dim] = [[] as { index: TensorIndice; size: number; boundary: [number, number]; stride: number }[], 0]
    for (let index of indices) {
      let size = index === undefined ? 1 : num(this.shape.at(dim))
      let [boundary, stride] = [[0, size] as [number, number], 1] // defaults
      if (Array.isArray(index) || index instanceof Tensor) {
        if (!(index instanceof Tensor)) index = new Tensor(index, { device: this.device, requires_grad: false })
        if (!dtypes.is_int(index.dtype)) throw new Error(`index dtype ${index.dtype} is not supported`)
        index = (index.to(this.device).lt(0)).where(size, 0).add(index) // treat negative index values
      } else if (typeof index === 'number' || index instanceof UOp) { // sint
        if (index instanceof UOp) throw new Error('KAREL: UOp not supported yet')
        if (index >= size || index < -size) throw new Error(`index=${index} is out of bounds with size=${size}`)
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
      let [shrinks, strides] = [mops.map((i) => i.boundary), mops.map((i) => i.stride)]
      x = x.shrink(shrinks).flip(...strides.entries().filter(([i, st]) => st < 0).map(([i, st]) => i))
      //   // handle stride !== 1 || -1
      if (strides.some((st) => Math.abs(st) !== 1)) {
        strides = strides.map((s) => Math.abs(s))
        // pad shape to multiple of stride/
        if (!all_int(x.shape)) throw new Error('symbolic shape not supported')
        x = x.pad(zip(x.shape, strides).map(([s, st]) => [0, round_up(s, st) - s] as [number, number]))
        x = x.reshape(zip(x.shape, strides).flatMap(([s, st]) => [idiv(s, st), st]))
        x = x.shrink(x.shape.filter((_, i) => mod(i, 2) === 0).flatMap((s) => [[0, s], [0, 1]])).reshape(x.shape.filter((_, i) => mod(i, 2) === 0))
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
          const i = tensor.reshape([...(tensor.shape_num), ...range(x.ndim - dims[0]).map(() => 1)]).expand(pre_reduce_shape)
          masks.push(i._one_hot_along_dim(x.shape_num[dim], dim - x.ndim))
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
      if (dims[0] !== 0 && dims.length !== 1 && !is_eq(dims, range(dims[0], dims.at(-1)! + 1))) {
        x = x.permute(...range(dims[0], dims[0] + big_shape.length), ...range(0, dims[0]), ...range(dims[0] + big_shape.length, x.ndim))
      }
      // for advanced setitem, returns whole tensor with indices replaced
      if (v !== undefined) {
        let vb = v.cast(this.dtype)._broadcast_to(_broadcast_shape([x.shape, v.shape]))
        // add back reduced dims from sum
        for (const dim of sum_axis) vb = vb.unsqueeze(dim)
        // run _masked_setitem on tuple of axis that is to be reduced to match self.shape
        x = _masked_setitem(this, vb, mask, range(dims[0], dims[0] + big_shape.length))
      }
    }

    return x
  }
  /**
   * Retrieve a sub-tensor using indexing.
   *
   * Supported Index Types: `int | slice | Tensor | None | List | Tuple | Ellipsis`
   *
   * Examples:
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(12).reshape(3, 4)
   * print(t.numpy())
   * ```
   *
   * - Int Indexing: Select an element or sub-tensor using integers for each dimension.
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t[1, 2].numpy())
   * ```
   *
   * - Slice Indexing: Select a range of elements using slice notation (`start:end:stride`).
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t[0:2, ::2].numpy())
   * ```
   *
   * - Tensor Indexing: Use another tensor as indices for advanced indexing. Using `tuple` or `list` here also works.
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t[Tensor([2, 0, 1]), Tensor([1, 2, 3])].numpy())
   * ```
   *
   * - `None` Indexing: Add a new dimension to the tensor.
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t[:, None].shape)
   * ```
   *
   * NOTE: Out-of-bounds indexing results in a value of `0`.
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([1, 2, 3])
   * print(t[Tensor([4, 3, 2])].numpy())
   * ```
   */
  get = (...indices: TensorIndice[]) => this._getitem(indices)

  set = async (indices: TensorIndice[], v: Tensor | number) => {
    if (typeof this.device === 'string' && this.device.startsWith('DISK')) {
      this._getitem(indices).assign(v)
      return
    }
    // NOTE: check that setitem target is valid first
    if (!this.lazydata.st!.contiguous) throw new Error('setitem target needs to be contiguous')
    if (!(v instanceof Tensor || isConst(v))) throw new Error(`can't set a ${v} to a Tensor`)
    if (!(v instanceof Tensor)) v = new Tensor(v, { device: this.device, dtype: this.dtype })
    if (this.requires_grad || v.requires_grad) throw new Error('setitem with requires_grad is not supported')

    const res = (await this.realize())._getitem(indices, v)
    // if shapes match and data is not shared it's a copy and we assign to self
    if (res.shape === this.shape && res.lazydata !== this.lazydata) await this.assign(res).realize()
    else {
      v = v.cast(res.dtype)._broadcast_to(_broadcast_shape([res.shape, v.shape])).contiguous()
      await res.assign(v).realize()
    }
  }
  /**
   * Gathers values along an axis specified by `dim`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([[1, 2], [3, 4]])
   * print(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.gather(1, Tensor([[0, 0], [1, 0]])).numpy())
   * ```
   * """
   */
  gather = (dim: number, index: Tensor): Tensor => {
    if (index.ndim !== this.ndim) throw new Error(`self.ndim must equal index.ndim, this.ndim=${this.ndim}, index.ndim=${index.ndim}`)
    dim = this._resolve_dim(dim)
    if (!zip(this.shape, index.shape).entries().filter(([d]) => d !== dim).every(([d, [s, i]]) => s >= i)) throw new Error('requires self.shape[d] >= index.shape[d] for all d != dim')
    index = index.to(this.device)
    const x = this.shrink([...index.shape.entries().map(([d, i]) => d !== dim ? [0, i] as [sint, sint] : undefined)]).unsqueeze(-1).transpose(-1, dim)
    return (x.mul(index.unsqueeze(-1)._one_hot_along_dim(this.shape_num[dim]))).sum(-1, undefined, this.dtype)
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

    const dim_cumsum = tensors.map((t) => t.shape[dim]).reduce((acc, curr, idx) => [...acc, add(acc[idx] ?? 0, curr)], [0] as sint[])
    for (const [i, t] of tensors.entries()) tensors[i] = t.pad(range(t.ndim).map((j) => j === dim ? [dim_cumsum[i], sub(dim_cumsum.at(-1)!, dim_cumsum[i + 1])] as [sint, sint] : undefined))
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
  static stack = (args: Tensor[], dim = 0): Tensor => {
    // checks for shapes and number of dimensions delegated to cat
    return Tensor.cat(args.map((t) => t.unsqueeze(dim)), dim)
  }
  stack = (args: Tensor[], dim = 0) => Tensor.stack([this, ...args], dim)

  /**
   * Repeat elements of a tensor.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([1, 2, 3])
   * print(t.repeat_interleave(2).numpy())
   * ```
   */
  repeat_interleave = (repeats: number, dim?: number): Tensor => {
    const x = dim === undefined ? this.flatten() : this
    dim = dim === undefined ? 0 : this._resolve_dim(dim)
    const shp = x.shape
    return x.reshape([...shp.slice(0, dim + 1), 1, ...shp.slice(dim + 1)]).expand([...shp.slice(0, dim + 1), repeats, ...shp.slice(dim + 1)]).reshape([...shp.slice(0, dim), ...range(repeats).map(() => shp[dim]), ...shp.slice(dim + 1)])
  }

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
    const base_shape = _align_left(this.shape, repeats)[0]
    const unsqueezed_shape = base_shape.flatMap((s) => [1, s] as [number, number])
    const expanded_shape = zip(repeats, base_shape).flat()
    const final_shape = zip(repeats, base_shape).map(([r, s]) => mul(r, s))
    return this.reshape(unsqueezed_shape).expand(expanded_shape).reshape(final_shape)
  }
  _resolve_dim = (dim: number, extra = false): number => {
    const total = this.ndim + Number(extra)
    if (!(-Math.max(1, total) <= dim && dim <= Math.max(1, total) - 1)) throw new Error(`dim=${dim} out of range ${list_str([-Math.max(1, total), Math.max(1, total) - 1])}`)
    return dim < 0 ? dim + total : dim
  }
  /**
   * Splits the tensor into chunks along the dimension specified by `dim`.
   * If `sizes` is an integer, it splits into equally sized chunks if possible, otherwise the last chunk will be smaller.
   * If `sizes` is a list, it splits into `len(sizes)` chunks with size in `dim` according to `size`.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(10).reshape(5, 2)
   * print(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * split = t.split(2)
   * print("\\n".join([repr(x.numpy()) for x in split]))
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * split = t.split([1, 4])
   * print("\\n".join([repr(x.numpy()) for x in split]))
   * ```
   */
  split = (sizes: number | number[], dim = 0): Tensor[] => {
    if (!all_int(this.shape)) throw new Error(`does not support symbolic shape ${this.shape}`)
    dim = this._resolve_dim(dim)
    if (typeof sizes === 'number') sizes = range(0, max([1, this.shape[dim]]), max([1, sizes])).map((i) => min([num(sizes), this.shape_num[dim] - i]))
    if (sum(sizes) !== this.shape[dim]) throw new Error(`expect sizes to sum exactly to {self.shape[dim]}, but got {sum(sizes)}`)
    return range(sizes.length).map((i) => [...range(dim).map(() => ({})), { start: sum(sizes.slice(0, i)), stop: sum(sizes.slice(0, i + 1)) }]).map((sl) => this.get(sl))
  }

  /**
   * Splits the tensor into `chunks` number of chunks along the dimension `dim`.
   * If the tensor size along `dim` is not divisible by `chunks`, all returned chunks will be the same size except the last one.
   * The function may return fewer than the specified number of chunks.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * chunked = Tensor.arange(11).chunk(6)
   * print("\\n".join([repr(x.numpy()) for x in chunked]))
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * chunked = Tensor.arange(12).chunk(6)
   * print("\\n".join([repr(x.numpy()) for x in chunked]))
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * chunked = Tensor.arange(13).chunk(6)
   * print("\\n".join([repr(x.numpy()) for x in chunked]))
   * ```
   */
  chunk = (chunks: number, dim = 0): Tensor[] => {
    if (!all_int(this.shape)) throw new Error(`does not support symbolic shape ${this.shape}`)
    if (chunks <= 0) throw new Error(`expect chunks to be greater than 0, got: ${chunks}`)
    dim = this._resolve_dim(dim)
    return this.split(this.shape[dim] ? ceildiv(this.shape[dim], chunks) : range(chunks).map(() => 0), dim)
  }

  /**
   * Generates coordinate matrices from coordinate vectors.
   * Input tensors can be scalars or 1D tensors.
   *
   * `indexing` determines how the output grids are aligned.
   * `ij` indexing follows matrix-style indexing and `xy` indexing follows Cartesian-style indexing.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * x, y = Tensor([1, 2, 3]), Tensor([4, 5, 6])
   * grid_x, grid_y = x.meshgrid(y)
   * print(grid_x.numpy())
   * print(grid_y.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * grid_x, grid_y = x.meshgrid(y, indexing="xy")
   * print(grid_x.numpy())
   * print(grid_y.numpy())
   * ```
   */
  meshgrid = (args: Tensor[], indexing: 'ij' | 'xy' = 'ij'): Tensor[] => {
    if (!['ij', 'xy'].includes(indexing)) throw new Error(`indexing must be in ("ij", "xy"), got ${indexing}`)
    let tensors = [this, ...args]
    if (tensors.length === 1) return tensors
    const basis = indexing === 'ij' ? range(tensors.length) : [1, 0, ...range(2, tensors.length)]
    tensors = zip(basis, tensors).map(([i, t]) => t.reshape([-1, ...range(args.length - 1).map(() => 1)]))
    const output_shape = _broadcast_shape(tensors.map((t) => t.shape))
    return tensors.map((t) => t._broadcast_to(output_shape))
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
    return !this.ndim || this.shape[dim] !== 1 ? this : this.reshape([...this.shape.slice(0, dim), ...this.shape.slice(dim + 1)])
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
   * ```python exec="true" source="above" session="tensor" result="python"
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

  /**
   * Rolls the tensor along specified dimension(s).
   * The rolling operation is circular, meaning that elements that go beyond the edge are wrapped around to the beginning of the dimension.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(4)
   * print(t.roll(shifts=1, dims=0).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.roll(shifts=-1, dims=0).numpy())
   * ```
   */
  roll = (shifts: number | number[], dims: number | number[]): Tensor => {
    dims = make_tuple(dims, 1).map((d) => this._resolve_dim(d))
    let rolled: Tensor = this
    for (let [dim, shift] of zip(dims, make_tuple(shifts, 1))) {
      shift = mod(shift, this.shape_num[dim])
      rolled = Tensor.cat([rolled.get(...range(rolled.ndim).map((i) => i !== dim ? {} : { start: -shift })), rolled.get(...range(rolled.ndim).map((i) => i !== dim ? {} : { stop: -shift }))], dim)
    }
    return rolled
  }
  /**
   * Rearranges input according to formula
   *
   * See: https://einops.rocks/api/rearrange/
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * x = Tensor([[1, 2], [3, 4]])
   * print(Tensor.rearrange(x, "batch channel -> (batch channel)").numpy())
   * ```
   */
  rearrange = (formula: string, sizes: any): Tensor => {
    const parse_formula = (formula: string): [string[], [number, number][]] => {
      const tokens = ` ${formula} `.replace('…', '...').replace('(', ' ( ').replace(')', ' ) ').replace(' ', '  ').replace(' 1 ', ' ( ) ').split(/\s+/).filter(Boolean)
      const [lparens, rparens] = ['(', ')'].map((x) => tokens.map((ch, i) => ch === x ? i : -1).filter((i) => i !== -1))
      const pairs = zip(lparens, rparens)
      if (lparens.length !== rparens.length || !is_eq(flatten(pairs), flatten(pairs))) throw new Error('bracket mismatch')
      return [tokens.filter((name) => !['(', ')'].includes(name)), pairs.map(([s, e], i) => [s - 2 * i, e - 1 - 2 * i])]
    }
    if (formula.split('->').length !== 2) throw new Error('need exactly one "->" in formula')

    let [[lhs, unflatten_dims], [rhs, flatten_dims]] = formula.split('->').map(parse_formula)

    for (const name of sizes) if (lhs.includes(name)) throw new Error(`axis ${name} is not used in transform`)
    if (!is_eq(lhs.toSorted(), rhs.toSorted()) || lhs.length !== new Set(lhs).size) throw new Error(`name mismatch in ${formula}`)
    for (const name of flatten([lhs, rhs])) if (name !== '...' && name.includes(' ')) throw new Error(`invalid axis name ${name}`)
    if (flatten(unflatten_dims.map(([s, e]) => lhs.slice(s, e))).includes('...')) throw new Error(`cannot have collapsed ellipsis (...) in lhs of ${formula}`)
    if (lhs.filter((x) => x === '...').length > 1) throw new Error(`too many ellipses in ${formula}`)

    // resolve ellipsis
    let ell_len: number
    if (lhs.includes('...')) ell_len = this.shape.length - lhs.length + 1 + sum(unflatten_dims.map(([s, e]) => e - s - 1))
    ;[lhs, rhs] = [lhs, rhs].map((l) => [...l.slice(0, l.indexOf('...')), ...range(ell_len).map((j) => `...${j}`), ...(l.includes('...') ? l.slice(l.indexOf('...') + 1) : l)])
    unflatten_dims = unflatten_dims.map(([s, e]) => [s + (lhs.slice(0, s).includes('...0') ? ell_len - 1 : 0), e + (lhs.slice(0, e).includes('...0') ? ell_len - 1 : 0)])
    flatten_dims = flatten_dims.map(([s, e]) => [s + (rhs.slice(0, s).includes('...0') ? ell_len - 1 : 0), e + (rhs.slice(0, e).includes('...0') ? ell_len - 1 : 0)])

    // apply movement ops in order unflatten -> permute -> flatten/unsqueeze
    let t = unflatten_dims.reduce((x, dims) => x.unflatten(dims[0], range(...dims).map((d) => sizes.get(lhs[d], -1))), this as Tensor)
    for (const [i, name] of lhs.entries()) if (sizes.includes(name) && sizes[name] !== t.shape[i]) throw new Error(`size provided for dimension ${name} incorrect`)
    t = t.permute(...rhs.map((name) => lhs.indexOf(name)))
    return flatten_dims.toReversed().reduce((x, dims) => dims[0] < dims[1] ? x.flatten(dims[0], dims[1] - 1) : x.unsqueeze(dims[0]), t)
  }
  //     // ***** reduce ops *****

  _reduce = (fxn: ReturnType<typeof CreateFunction>, axis?: number | number[], keepdim = false): Tensor => {
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
  _inverse = (): Tensor => this.is_floating_point() ? this.neg() : dtypes.is_int(this.dtype) ? this.bitwise_not() : this.logical_not()

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
    return this._inverse().max(axis, keepdim)._inverse()
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
    return this.bool().max(axis, keepdim)
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
  all = (axis?: number | number[], keepdim = false): Tensor => {
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
    return numerator.div(num(prod(zip(this.shape, this.sum(axis, true).shape).filter(([si, so]) => resolve(ne(si, so))).map(([si]) => si)))).cast(output_dtype)
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
    return squares.sum(axis, keepdim).div(smax(0, sub(n, correction)))
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
    let m = this.sub(this.max(axis, true).detach())
    if (dtype !== undefined) m = m.cast(dtype)
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
    const m = this.eq(this.max(axis, true))
    const idx = Tensor.arange(this.shape_num.at(axis)!, 0, -1, { requires_grad: false, device: this.device }).reshape([this.shape.at(axis)!, ...range(this.ndim - axis - 1).map(() => 1)]).mul(m, true)
    return (idx.max(axis, keepdim).sub(this.shape_num.at(axis)!, true)).cast(dtypes.int32)
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
  argmin = (axis?: number, keepdim = false) => {
    return this._inverse().argmax(axis, keepdim)
  }

  /**
   * Sums the product of the elements of the input tensors according to a formula based on the Einstein summation convention.
   *
   * See: https://pytorch.org/docs/stable/generated/torch.einsum.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * x = Tensor([[1, 2], [3, 4]])
   * y = Tensor([[5, 6], [7, 8]])
   * print(Tensor.einsum("ij,ij->", x, y).numpy())
   * ```
   */
  static einsum = (formula: string, operands: Tensor | Tensor[], acc_dtype?: DTypeLike): Tensor => {
    throw new NotImplemented()
  }

  // ***** processing ops *****

  _pool = (k_: sint[], stride: number[] | number = 1, dilation: number[] | number = 1): Tensor => {
    if (this.shape.length < k_.length) throw new Error(`can't pool ${this.shape} with ${k_}`)
    const [s_, d_] = [make_tuple(stride, k_.length), make_tuple(dilation, k_.length)]
    if (k_.length !== s_.length || s_.length !== d_.length) throw new Error(`stride/dilation mismatch kernel:${k_} stride:${s_} dilation:${d_}`)
    const [noop, i_] = [range(this.ndim - k_.length).map(() => undefined), this.shape.slice(-k_.length)]
    if (!zip(k_, d_, i_).every(([k, d, i]) => resolve(le(add(mul(d, sub(k, 1)), 1), i)))) throw new Error('kernel size can not be greater than actual input size')
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
    let x = this.pad([...noop, ...zip(i_, o_, s_).map(([i, o, s]) => [0, max([0, num(sub(mul(o, s), i))])] as [sint, sint])]).shrink([...noop, ...zip(o_, s_).map(([o, s]) => [0, mul(o, s)] as [sint, sint])])
    x = x.reshape([...noop, ...zip(o_, s_).flat()])
    x = x.shrink([...noop, ...zip(o_, k_).flatMap(([o, k]) => [[0, o], [0, k]] as [sint, sint][])])
    return x.permute(...range(noop.length), ...range(i_.length).map((i) => noop.length + i * 2), ...range(i_.length).map((i) => noop.length + i * 2 + 1))
  }
  _resolve_pool_pads = (padding: number | number[], dims: number): number[] => {
    if (Array.isArray(padding) && !(padding.length === 2 * dims || padding.length === dims)) throw new Error(`Padding must be an int or a sequence of length ${dims} or ${2 * dims}, but got padding=${padding} for shape=${this.shape} with dims=${dims}.`)
    return !Array.isArray(padding) ? range(2 * dims).map(() => padding) : padding.length === 2 * dims ? padding : range(2).flatMap(() => padding).toReversed()
  }
  _apply_ceil_mode = (pads: number[], k_: sint[], s_: number[] | number, d_: number | number[]): number[] => {
    ;[d_, s_] = [d_, s_].map((x) => make_tuple(x, k_.length))
    const i_ = this.shape_num.slice(-k_.length)
    pads = [...pads]
    const grouped_pads = _flat_to_grouped(pads)
    // https://arxiv.org/pdf/1603.07285 section 5.1, relationship 15.
    const o_ = zip(i_, d_, k_, s_, grouped_pads).map(([i, d, k, s, [pB, pA]]) => ceildiv(i + num(pB) + num(pA) - (d * (num(k) - 1) + 1), s) + 1)
    for (const [dim, [o, i, s, k, d, [pB, pA]]] of zip(o_, i_, s_, k_, d_, grouped_pads).entries()) {
      // we have to do additional padding before `_pool` so that `o_` in `_pool` is calculated correctly
      // `s*(o-1) + (d*(k-1)+1) - (i+pB+pA)` -> last_sliding_window_start + full_kernel_size - padded_input_shape
      // we decrease padding in the case that a sliding window starts in the end padded region, thereby decreasing `o_` in `_pool`
      // `smax(s*(o-1) - (pB+i-1), 0)` -> last_sliding_window_start - (pad_before + input_size - zero_offset)
      pads[-1 - dim * 2] += s * (o - 1) + (d * (num(k) - 1) + 1) - (i + num(pB) + num(pA)) - num(smax(s * (o - 1) - (num(pB) + i - 1), 0))
    }
    return pads
  }
  // NOTE: these work for more than 2D
  avg_pool2d = (kernel_size = [2, 2], stride?: number, dilation = 1, padding = 0, ceil_mode = false, count_include_pad = true) => {
    const k_ = make_tuple(kernel_size, 2), axis = range(-k_.length, 0)
    const pool = (x: Tensor, padding_: number[]): Tensor => x.pad(padding_)._pool(k_, stride !== undefined ? stride : k_, dilation)
    const reg_pads = this._resolve_pool_pads(padding, k_.length)
    const ceil_pads = this._apply_ceil_mode(reg_pads, k_, stride !== undefined ? stride : k_, dilation)
    if (!count_include_pad) {
      const pads = ceil_mode ? ceil_pads : reg_pads
      return pool(this, pads).sum(axis).div(pool(this.ones_like(), pads).sum(axis))
    }
    if (!ceil_mode) return pool(this, reg_pads).mean(axis)
    return pool(this, ceil_pads).sum(axis).div(pool(this.pad(reg_pads).ones_like(), zip(ceil_pads, reg_pads).map(([cp, rp]) => cp - rp)).sum(axis))
  }
  /**
   * Applies average pooling over a tensor.
   *
   * This function supports three different types of `padding`:
   *
   * 1. `int` (single value):
   *   Applies the same padding value uniformly to all spatial dimensions.
   *
   * 2. `Tuple[int, ...]` (length = number of spatial dimensions):
   *   Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.
   *
   * 3. `Tuple[int, ...]` (length = 2 * number of spatial dimensions):
   *   Specifies explicit padding for each side of each spatial dimension in the form
   *   `(padding_left, padding_right, padding_top, padding_bottom, ...)`.
   *
   * When `ceil_mode` is set to `true`, output shape will be determined using ceil division.
   * When `count_include_pad` is set to `false`, zero padding will not be included in the averaging calculation.
   *
   * NOTE: unlike PyTorch, this implementation is not limited to only 2d pooling and instead works for any number of dimensions.
   *
   * See: https://paperswithcode.com/method/average-pooling
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(25).reshape(1, 1, 5, 5)
   * print(t.avg_pool2d().numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.avg_pool2d(ceil_mode=true).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.avg_pool2d(padding=1).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.avg_pool2d(padding=1, count_include_pad=false).numpy())
   * ```
   */
  max_pool2d = (kernel_size = [2, 2], stride?: number, dilation = 1, padding = 0, ceil_mode = false) => {
    let k_ = make_tuple(kernel_size, 2), pads = this._resolve_pool_pads(padding, k_.length)
    if (ceil_mode) pads = this._apply_ceil_mode(pads, k_, stride !== undefined ? stride : k_, dilation)
    return this.pad(pads, undefined, dtypes.min(this.dtype))._pool(k_, stride !== undefined ? stride : k_, dilation).max(range(-k_.length, 0))
  }
  static max_pool2d = (t: Tensor) => t.max_pool2d()
  /**
   * Applies a convolution over a tensor with a given `weight` && optional `bias`.
   *
   * 1. `int` (single value):
   *   Applies the same padding value uniformly to all spatial dimensions.
   *
   * 2. `Tuple[int, ...]` (length = number of spatial dimensions):
   *   Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.
   *
   * 3. `Tuple[int, ...]` (length = 2 * number of spatial dimensions):
   *   Specifies explicit padding for each side of each spatial dimension in the form
   *   `(padding_left, padding_right, padding_top, padding_bottom, ...)`.
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
  conv2d = (weight: Tensor, bias?: Tensor, groups = 1, stride = 1, dilation: number | number[] = 1, padding: number | number[] = 0, acc_dtype?: DTypeLike): Tensor => {
    if (vars.IMAGE) return this.image_conv2d(weight, bias, groups, stride, dilation, padding, acc_dtype as DType)
    const [[bs, cin_], [cout, cin], HW] = [this.shape_num.slice(0, 2), weight.shape_num.slice(0, 2), weight.shape_num.slice(2)]
    const padding_ = this._resolve_pool_pads(padding, HW.length)
    if (!(groups * cin === cin_ && this.shape.length === weight.shape.length)) throw new Error(`Input Tensor shape ${this.shape} does !match the shape of the weights ${weight.shape}. (${groups * cin} vs. ${cin_})`)

    // conv2d === a pooling op (with padding)
    let x = this.pad(padding_)._pool(HW, stride, dilation) // (bs, groups*cin, oy, ox, H, W)
    const [rcout, oyx] = [idiv(cout, groups), x.shape.slice(2, -HW.length)]
    if (!HW.every((x) => x === 3) || stride !== 1 || dilation !== 1 || !vars.WINO) {
      // normal conv
      x = x.reshape([bs, groups, cin, 1, ...oyx, ...HW]).expand([bs, groups, cin, rcout, ...oyx, ...HW]).permute(0, 1, 3, ...range(oyx.length).map((i) => 4 + i), 2, ...range(HW.length).map((i) => 4 + oyx.length + i))

      // conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
      const ret = (x.mul(weight.reshape([1, groups, rcout, ...range(oyx.length).map(() => 1), cin, ...HW]))).sum(range(1 + oyx.length).map((i) => -1 - i), true, acc_dtype).reshape([bs, cout, ...oyx])
      return bias === undefined ? ret : ret.add(bias.reshape([1, -1, ...range(HW.length).map(() => 1)]))
    }
    const HWI = range(HW.length).map(() => 6), HWO = range(HW.length).map(() => 4) // F(4x4,3x3) winograd tiles
    const winograd_G = [[1 / 4, 0, 0], [-1 / 6, -1 / 6, -1 / 6], [-1 / 6, 1 / 6, -1 / 6], [1 / 24, 1 / 12, 1 / 6], [1 / 24, -1 / 12, 1 / 6], [0, 0, 1]]
    const winograd_Bt = [[4, 0, -5, 0, 1, 0], [0, -4, -4, 1, 1, 0], [0, 4, -4, -1, 1, 0], [0, -2, -1, 2, 1, 0], [0, 2, -1, -2, 1, 0], [0, 4, 0, -5, 0, 1]]
    const winograd_At = [[1, 1, 1, 1, 1, 0], [0, 1, -1, 2, -2, 0], [0, 1, 1, 4, 4, 0], [0, 1, -1, 8, -8, 1]] // applying At in pre-order doubles compile time

    // todo: stride == dilation
    // use padding to round up to 4x4 output tiles
    // (bs, cin_, tyx, HWI)
    let d = this.pad(this.shape_num.slice(-HW.length).flatMap((dim, i) => [padding_[i * 2], padding_[i * 2 + 1] + mod(-(dim + sum(padding_.slice(i * 2, (i + 1) * 2)) - 2), 4)]))._pool(HWI, HWO)
    // move HW to the front: // (HWI, bs, cin_, tyx)
    d = d.permute(...range(d.shape.length - HW.length, d.shape.length), ...range(d.shape.length - HW.length))
    const tyx = d.shape_num.slice(-HWI.length) // dim of tiling

    const g = weight.permute(...range(weight.shape.length - HW.length, weight.shape.length), ...range(weight.shape.length - HW.length)) // move HW to the front

    // compute 6x6 winograd tiles: GgGt, BtdB
    // (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    const gfactors = _apply_winograd_matrix(winograd_G, g, HW.length).reshape([...HWI, 1, groups, rcout, cin, ...range(tyx.length).map(() => 1)])
    // (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)
    const dfactors = _apply_winograd_matrix(winograd_Bt, d, HW.length).reshape([...HWI, bs, groups, 1, cin, ...tyx])

    // matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)
    let ret = _apply_winograd_matrix(winograd_At, (gfactors.mul(dfactors)).sum(-1 - HW.length, undefined, acc_dtype), HW.length)

    // interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.permute(...range(HW.length, ret.shape.length - HW.length), ...range(HW.length).flatMap((i) => [ret.shape.length - HW.length, 0].map((o) => i + o)))
    // merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final
    ret = ret.reshape([bs, cout, ...tyx.map((c, i) => c * HWO[i])]).shrink([bs, cout, ...oyx].map((s) => [0, s]))

    return (bias === undefined ? ret : ret.add(bias.reshape([1, -1, ...range(HW.length).map(() => 1)]))).contiguous().contiguous_backward()
  }
  /**
   * Applies a transposed convolution over a tensor with a given `weight` and optional `bias`.
   *
   * This function supports three different types of `padding`
   *
   * 1. `int` (single value):
   *   Applies the same padding value uniformly to all spatial dimensions.
   *
   * 2. `Tuple[int, ...]` (length = number of spatial dimensions):
   *   Specifies a distinct padding value for each spatial dimension in the form `(padding_height, padding_width, ...)`.
   *
   * 3. `Tuple[int, ...]` (length = 2 * number of spatial dimensions):
   *   Specifies explicit padding for each side of each spatial dimension in the form
   *   `(padding_left, padding_right, padding_top, padding_bottom, ...)`.
   *
   * NOTE: unlike PyTorch, this implementation is not limited to only 2d transposed convolutions and instead works for any number of dimensions.
   *
   * See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor.arange(9).reshape(1, 1, 3, 3)
   * w = Tensor.ones(1, 1, 2, 2)
   * print(t.conv_transpose2d(w).numpy())
   * ```
   */
  conv_transpose2d = (weight: Tensor, bias?: Tensor, groups = 1, stride_ = 1, dilation_ = 1, padding_: number | number[] = 0, output_padding_ = 0): Tensor => {
    let x: Tensor = this, w = weight.unflatten(0, [groups, -1]).transpose(1, 2).flip(...range(3, weight.shape.length + 1))
    const HW = weight.shape_num.slice(2)
    let padding = _flat_to_grouped(this._resolve_pool_pads(padding_, HW.length))
    const [stride, dilation, output_padding] = [stride_, dilation_, output_padding_].map((x) => make_tuple(x, HW.length))
    if (stride.some((s) => s > 1)) {
      // handle strides: (k) -> reshape -> (k,1) -> pad -> (k,s) -> reshape -> (k*s) -> shrink (k-(s-1))
      x = x.reshape([undefined, undefined, ...flatten(x.shape.slice(2).map((k) => [k, 1]))])
      x = x.pad([undefined, undefined, ...flatten(stride.map((s) => [undefined, [0, s - 1] as [sint, sint]]))])
      x = x.reshape([undefined, undefined, ...zip(slice(x.shape_num, { start: 2, step: 2 }), stride).map(([k, s]) => k * s)])
      x = x.shrink([undefined, undefined, ...zip(x.shape_num.slice(2), stride).map(([k, s]) => [0, k - (s - 1)] as [sint, sint])])
    }
    const new_padding = flatten(zip(HW, dilation, padding, output_padding).toReversed().map(([k, d, [pB, pA], op]) => [(k - 1) * d - num(pB), (k - 1) * d - num(pA) + op]))
    return x.conv2d(w.flatten(undefined, 1), bias, groups, undefined, dilation, new_padding)
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
    if (vars.IMAGE) {
      throw new Error('KAREL: implement image_dot')
      // return this.image_dot(w, acc_dtype)
    }
    let [x, dx, dw] = [this as Tensor, this.ndim, w.ndim]
    if (!(dx > 0 && dw > 0)) throw new Error(`both tensors need to be at least 1D, got ${dx}D && ${dw}D`)
    const axis_w = -Math.min(w.ndim, 2)
    if (x.shape.at(-1) !== w.shape.at(axis_w)) throw new Error(`can not dot ${list_str(x.shape)} && ${list_str(w.shape)}, axis_w=${axis_w}`)
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
    const pooled = this.transpose(axis, -1).pad([pl_sz, -Number(_include_initial)], undefined, num(identity_element(op, this.dtype)))._pool([this.shape.at(axis)!])
    return (op === Ops.ADD ? pooled.sum(-1) : pooled.max(-1)).transpose(axis, -1)
  }

  _split_cumalu = (axis: number, op: Ops): Tensor => {
    axis = this._resolve_dim(axis)
    if (this.ndim === 0 || this.shape.includes(0)) return this
    // TODO: someday the optimizer will find this on it's own
    // for now this is a two stage cumsum
    const SPLIT = 256
    const s = num(this.shape.at(axis))
    if (!Number.isInteger(s) || s <= SPLIT * 2) return this._cumalu(axis, op)
    const ret = this.transpose(axis, -1).pad([round_up(s, SPLIT) - s, 0], undefined, num(identity_element(op, this.dtype))).unflatten(-1, [-1, SPLIT])._cumalu(-1, op)
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
    if (!isInt(r) || !isInt(c)) throw new Error(`does not support symbolic, getting r=${r}, c=${c}`)
    if (r === 0 || c === 0 || diagonal >= num(c)) return Tensor.zeros([r, c], opts)
    if (num(r) + diagonal <= 0) return Tensor.ones([r, c], opts)
    const s = sub(add(r, c), 1)
    // build a (s, s) upper triangle
    const t = Tensor.ones([s, s], opts).pad([undefined, [0, s]]).flatten().shrink([[0, mul(s, sub(mul(2, s), 1))]]).reshape([s, -1]).shrink([undefined, [0, s]])
    return diagonal <= 0 ? t.get({ stop: num(r) }, { start: -diagonal, stop: num(c) - diagonal }) : t.get({ start: diagonal, stop: num(r) + diagonal }, { stop: num(c) })
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

  /**
   * Downsamples or Upsamples to the input `size`, accepts 0 to N batch dimensions.
   *
   * The interpolation algorithm is selected with `mode` which currently only supports `linear`, `nearest` and `nearest-exact`.
   * To run `bilinear` or `trilinear`, pass in a 2D or 3D size.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([[1, 2, 3, 4], [21, 22, 23, 24], [41, 42, 43, 44]])
   * print(t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(t.interpolate(size=(2,3), mode="linear").numpy())
   * ```
   */
  interpolate = (size: number[], mode: 'linear' | 'nearest' | 'nearest-exact' = 'linear', align_corners = false): Tensor => {
    if (!(Array.isArray(size) && all_int(size) && 0 < size.length && size.length <= this.ndim)) throw new Error(`invalid size=${size}`)
    if (!['linear', 'nearest', 'nearest-exact'].includes(mode)) throw new Error('only supports linear, nearest or nearest-exact interpolate')
    if (align_corners && mode !== 'linear') throw new Error('align_corners option can only be set with the interpolating mode linear')
    let x: Tensor = this, expand = [...this.shape]
    for (const i of range(-1, -size.length - 1, -1)) {
      const scale = (this.shape_num[i] - Number(align_corners)) / (size[i] - Number(align_corners))
      const arr = Tensor.arange(size[i], undefined, undefined, { dtype: dtypes.float32, device: this.device }), reshape = range(this.ndim).map((x) => 1)
      reshape[i] = expand[i] = size[i]
      if (mode === 'linear') {
        const index = (align_corners ? arr.mul(scale, true) : (arr.add(0.5).mul(scale, true)).sub(0.5)).clip(0, this.shape_num[i] - 1)
        const [low, high, perc] = [index.floor(), index.ceil(), index.sub(index.floor())].map((y) => y.reshape(reshape).expand(expand))
        x = x.gather(i, low).lerp(x.gather(i, high), perc)
      } else {
        const index = (mode === 'nearest-exact' ? (arr.add(0.5).mul(scale, true)) : arr.mul(scale, true)).cast(dtypes.int32).reshape(reshape).expand(expand)
        x = x.gather(i, index)
      }
    }
    return x.cast(this.dtype)
  }
  /**
   *
   * Scatters `src` values along an axis specified by `dim`.
   * Apply `add` or `multiply` reduction operation with `reduce`.

   * ```python exec="true" source="above" session="tensor" result="python"
   * src = Tensor.arange(1, 11).reshape(2, 5)
   * print(src.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * index = Tensor([[0, 1, 2, 0]])
   * print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(0, index, src).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * index = Tensor([[0, 1, 2], [0, 1, 4]])
   * print(Tensor.zeros(3, 5, dtype=src.dtype).scatter(1, index, src).numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='multiply').numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor.full((2, 4), 2.0).scatter(1, Tensor([[2], [3]]), 1.23, reduce='add').numpy())
   * ```
   */
  scatter = (dim: number, index: Tensor, src: Tensor | ConstType, reduce?: 'multiply' | 'add'): Tensor => {
    if (!['add', 'multiply', undefined].includes(reduce)) throw new Error(`reduce=${reduce} must be one of None, 'multiply', or 'add'`)
    index = index.to(this.device), dim = this._resolve_dim(dim)
    src = src instanceof Tensor ? src.cast(this.dtype) : new Tensor(src, { device: this.device, dtype: this.dtype })._broadcast_to(index.shape)
    if (index.ndim !== this.ndim || this.ndim !== src.ndim) throw new Error(`self.ndim, index.ndim and src.dim must all equal, ndim=${this.ndim} index.ndim=${index.ndim} src.ndim=${src.ndim}`)
    if (!zip(this.shape, index.shape, src.shape).every(([self_, index_, src_], d) => (d === dim || self_ >= index_) && src_ >= index_)) throw new Error(`All dimensions of ${index.shape} should be <= to all dimensions of ${src.shape} and all dimensions except dimension ${dim} of ${this.shape}`)
    // shrink src to index shape to shrink away the unused values
    src = src.shrink(index.shape.map((s) => [0, s]))
    // prepare src and mask for reduce with respect to dim
    src = src.unsqueeze(-1).expand([...src.shape, this.shape[dim]]).transpose(-1, dim)
    let mask = index.unsqueeze(-1)._one_hot_along_dim(this.shape_num[dim]).transpose(-1, dim) // pad src and mask to self.shape so that reduce can be done with padded values as no-ops
    ;[src, mask] = [src, mask].map((x) => x.pad([...range(this.ndim).map((i) => i !== dim ? [0, this.shape_num[i] - x.shape_num[i]] as [sint, sint] : undefined), undefined]))
    if (reduce === 'add') return mask.where(src, 0).sum(-1, undefined, this.dtype).add(this)
    if (reduce === 'multiply') return mask.where(src, 1).prod(-1, undefined, this.dtype).mul(this)
    return _masked_setitem(this, src, mask, [-1])
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
    return this.mul(-1 / Math.log(2)).exp2().add(1, true).reciprocal()
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
    const x = (this.abs().sub(1.0, true)).sqrt().mul(polyN(this.abs(), coefficients)).sub(Math.PI / 2, true)
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
    if (this.dtype === dtypes.uint8 && (weight instanceof Tensor)) {
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
    return this.sign().mul(t.mul(polyN(t, [1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592])).mul(this.square().neg().exp()).sub(1.0, true))
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
    return this.mul(0.5, true).mul(this.pow(3).mul(0.044715, true).add(this, true).mul(Math.sqrt(2 / Math.PI), true).tanh().add(1, true))
  }
  static gelu = (x: Tensor) => x.gelu()

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
    if (is_eq(this.shape, new_shape)) return this
    if (this.ndim > new_shape.length) throw new Error(`can not broadcast tensor to fewer dimensions. shape=${list_str(this.shape)} to new_shape=${list_str(new_shape)}`)
    // first unsqueeze left with 1s https://data-apis.org/array-api/latest/API_specification/broadcasting.html
    const [shape, _] = _align_left(this.shape, new_shape)
    // for each dimension, check either dim === 1, || it does !change
    // if (zip(shape, new_shape).every(([s, ns]) => resolve(eq(s, ns)) || resolve(eq(s, 1)))) throw new Error(`can not broadcast ${listStr(this.shape)} to ${listStr(new_shape)}`)
    return Expand.apply(this.reshape(shape), new_shape)
  }
  _broadcasted = (y: ConstType<Tensor | UOp>, reverse = false, match_dtype = true): [Tensor, Tensor] => {
    let x: Tensor = this
    if (!(y instanceof Tensor)) {
      // make y a Tensor
      if (!isConst(y)) throw new Error(`invalid y type: ${typeof y}`)
      let y_dtype
      if (x.dtype instanceof ImageDType || dtypes.is_float(x.dtype) || (dtypes.is_big_int(x.dtype)) || (dtypes.is_int(x.dtype) && Number.isInteger(y))) y_dtype = x.dtype
      else if (!(y as any instanceof UOp)) y_dtype = dtypes.from_js(y)
      if (y as any instanceof UOp) y = Tensor.from_uop(y as any, { device: x.device })
      else y = new Tensor(dtypes.as_const(y, y_dtype!), { device: x.device, dtype: y_dtype, requires_grad: false })
    }
    if (!(y instanceof Tensor)) throw new Error('y has to be Tensor')
    if (match_dtype && x.dtype !== y.dtype) {
      const output_dtype = least_upper_dtype(x.dtype, y.dtype)
      ;[x, y] = [x.cast(output_dtype), y.cast(output_dtype)]
    }
    if (reverse) [x, y] = [y, x]

    // broadcast
    const out_shape = _broadcast_shape([x.shape, y.shape])
    return [x._broadcast_to(out_shape), y._broadcast_to(out_shape)]
  }

  // TODO: tensor should stop checking if things are const
  _to_const_val = (x: ConstType<Tensor>): ConstType<Tensor> => {
    return x instanceof Tensor && x.lazydata instanceof UOp && x.lazydata.base.op === Ops.CONST && x.lazydata.st!.views[0].mask !== undefined && !x.requires_grad && is_eq(this._broadcasted(x)[0].shape, this.shape) ? x.lazydata.const_arg : x
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
   * `idiv` performs integer division (truncate towards zero).
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor([-4, 7, 5, 4, -7, 8]).idiv(Tensor([2, -3, 8, -2, 3, 5])).numpy())
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
   * Mod `self` by `x`.
   * Equivalent to `self % x`.
   * Supports broadcasting to a common shape, type promotion, and integer inputs.
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * print(Tensor([-4, 7, 5, 4, -7, 8]).mod(Tensor([2, -3, 8, -2, 3, 5])).numpy())
   * ```
   */
  override mod = (x: ConstType<Tensor>, reverse = false): Tensor => {
    const [a, b] = this._broadcasted(x, reverse)
    const r = Mod.apply(a, b)
    return r.add(b.mul(((r.lt(0)).bitwise_and(b.gt(0))).bitwise_or((r.gt(0)).bitwise_and(b.lt(0)))))
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
    return dtypes.is_big_int(this.dtype) ? this.xor(-1n) : this.xor(-1)
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
    if (!(dtypes.is_unsigned(this.dtype) && (typeof x === 'number' || typeof x === 'bigint') && x >= 0)) throw new Error(`not supported dtype=${this.dtype} x=${x}`)
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
    if (!(dtypes.is_unsigned(this.dtype) && (typeof x === 'number' || typeof x === 'bigint') && x >= 0)) throw new Error(`!supported dtype=${this.dtype} x=${x}`)
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
    x = num(this._to_const_val(x))
    if (!(x as any instanceof Tensor) && !reverse) {
      // simple pow identities
      if (x < 0) return this.reciprocal().pow(-x).cast(this.dtype)
      if (x === 0) return this.mul(0).add(1, true)
      // rewrite pow 0.5 to sqrt
      if (Math.trunc(x - 0.5) + 0.5 === x) return this.pow(Math.trunc(x - 0.5)).mul(this.sqrt())
      if (Math.trunc(x) === x) return this.pow(idiv(x, 2)).square().mul(mod(x, 2) === 0 ? 1 : this)
    }
    // positive const ** self
    if (!(x as any instanceof Tensor) && reverse && x > 0) return this.mul(Math.log(x)).exp()

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
    // NOTE: the mid-point is for backward, revisit after new gradient API
    if (this.is_floating_point()) return (this.lt(x)).detach().where(x, (this.eq(x)).detach().where((this.mul(0.5).add(mul(x, 0.5))).cast(this.dtype), this))
    return (this.lt(x)).detach().where(x, this)
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
    let [t, x2] = this._broadcasted(x)
    return t._inverse().maximum(x2._inverse())._inverse()
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
    if ((x instanceof Tensor)) [x, y] = x._broadcasted(y)
    else if (y instanceof Tensor) [y, x] = y._broadcasted(x)
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
  sequentialAsync = async (ll: LayerAsync[]) => {
    let x: Tensor = this
    for (const f of ll) {
      x = typeof f === 'function' ? await f(x) : await f.call(x)
    }
    return x
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
    const axis_ = Array.isArray(axis) ? axis : [axis]
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
    if (num_classes === -1) num_classes = num(await this.max().add(1).item())
    return this.get('...', undefined)._one_hot_along_dim(num_classes).where(1, 0)
  }

  /**
   *
   * Computes scaled dot-product attention.
   * `self` is the query tensor, `key` is the key tensor, and `value` is the value tensor.
   *
   * - Described: https://paperswithcode.com/method/scaled
   * - Paper: https://arxiv.org/abs/1706.03762v7

   * ```python exec="true" source="above" session="tensor" result="python"
   * q = Tensor.randn(2, 4, 8)
   * k = Tensor.randn(2, 4, 8)
   * v = Tensor.randn(2, 4, 8)
   * print(q.scaled_dot_product_attention(k, v).numpy())
   * ```
   */
  scaled_dot_product_attention = (key: Tensor, value: Tensor, attn_mask?: Tensor, dropout_p = 0.0, is_causal = false): Tensor => {
    // NOTE: it also works when `key` and `value` have symbolic shape.
    if (!all_int(this.shape)) throw new Error(`does not support symbolic shape ${this.shape}`)
    let qk = this.matmul(key.transpose(-2, -1), undefined, least_upper_dtype(this.dtype, key.dtype, dtypes.float32)).div(Math.sqrt(this.shape.at(-1)!))
    // handle attention mask
    if (is_causal) {
      if (attn_mask !== undefined) throw new Error('cannot set attn_mask when is_causal=True')
      attn_mask = qk.ones_like({ requires_grad: false, device: this.device, dtype: dtypes.bool }).tril()
    }
    if (attn_mask !== undefined) {
      if (attn_mask.dtype === dtypes.bool) attn_mask = attn_mask.where(0, -Infinity)
      qk = qk.add(attn_mask)
    }
    return qk.softmax(-1).cast(this.dtype).dropout(dropout_p).matmul(value)
  }
  _do_reduction = (reduction: ReductionStr = 'mean'): Tensor => {
    if (!ReductionStr.includes(reduction)) throw new Error(`reduction=${reduction} must be one of ${ReductionStr}`)
    const reductions: Record<ReductionStr, (x: Tensor) => Tensor> = { 'mean': (x) => x.mean(), 'sum': (x) => x.sum(), 'none': (x) => x }
    return reductions[reduction](this)
  }
  /**
   * Computes the binary cross-entropy loss between `self` and `Y`.
   *
   * See: https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([0.1, 0.9, 0.2])
   * Y = Tensor([0, 1, 0])
   * print(t.binary_crossentropy(Y).item())
   * ```
   */
  binary_crossentropy = (Y: Tensor, reduction: ReductionStr = 'mean'): Tensor => {
    return (Y.neg().mul(this.log()).sub(Y.sub(1, true).mul(this.sub(1, true).log())))._do_reduction(reduction)
  }
  /**
   *  Computes the binary cross-entropy loss between `self` and `Y` where `self` is logits.
   *
   *  See: https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
   *
   *  ```python exec="true" source="above" session="tensor" result="python"
   *  t = Tensor([-1, 2, -3])
   *  Y = Tensor([0, 1, 0])
   *  print(t.binary_crossentropy_logits(Y).item())
   *  ```
   */
  binary_crossentropy_logits = (Y: Tensor, reduction: ReductionStr = 'mean'): Tensor => {
    return (this.maximum(0).sub(Y.mul(this)).add((this.abs().neg().exp().add(1, true)).log()))._do_reduction(reduction)
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
    if (!(0.0 <= label_smoothing && label_smoothing <= 1.0)) throw new Error('label_smoothing must be in [0.0, 1.0]')
    if (!['mean', 'sum', 'none'].includes(reduction)) throw new Error("reduction must be one of ['mean', 'sum', 'none']")
    const [log_probs, loss_mask] = [this.log_softmax(), ignore_index !== -1 ? (Y.ne(ignore_index)) : Y.ones_like({ dtype: dtypes.bool })]
    const y_counted = Y.to(this.device).flatten().reshape([-1, 1])._one_hot_along_dim(num(this.shape.at(-1)))
    const y = (y_counted.mul(loss_mask.reshape([-1, 1]))).reshape([...Y.shape, this.shape.at(-1)!])
    const smoothing = log_probs.mean(-1).mul(loss_mask).mul(label_smoothing, true)
    const unreduced = log_probs.mul(y).sum(-1).mul(1 - label_smoothing, true).add(smoothing)
    // NOTE: because of ignore_index, we can't use Tensor.mean (so can't use `_do_reduction` here)
    return (unreduced.sum().div(reduction === 'mean' ? loss_mask.sum() : (reduction === 'sum' ? unreduced.sum() : unreduced))).neg()
  }
  /**
   * Compute the cross entropy loss between input logits and target.
   *
   * NOTE: `self` are logits and `Y` are the target labels or class probabilities.
   * See: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([[-1, 2, -3], [1, -2, 3]])
   * Y = Tensor([1, 2])
   * print(t.cross_entropy(Y).item())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([[-1, 2, -3], [1, -2, 3]])
   * Y = Tensor([1, 2])
   *  ```
   */
  cross_entropy = (Y: Tensor, reduction: ReductionStr = 'mean', label_smoothing = 0.0): Tensor => {
    if (!(0.0 <= label_smoothing && label_smoothing <= 1.0)) throw new Error('label_smoothing must be in [0.0, 1.0]')
    if (Y.ndim < 2) throw new Error('not implemented cause Y.one_hot is async')
    Y = Y.mul(1 - label_smoothing, true).add(label_smoothing / Y.shape_num[1])
    let ret = this.log_softmax(1).mul(Y).sum(1).neg()
    return ret._do_reduction(reduction)
  }

  /**
   * Compute the negative log likelihood loss between log-probabilities and target labels.
   *
   * NOTE: `self` is log-probabilities and `Y` is the Y labels or class probabilities.
   *
   * See: https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html
   *
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([[-1, 2, -3], [1, -2, 3]])
   * Y = Tensor([1, 2])
   * print(t.log_softmax().nll_loss(Y).item())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = Tensor([[-1, 2, -3], [1, -2, 3]])
   * Y = Tensor([1, 2])
   * print(t.log_softmax().nll_loss(Y, reduction='none').numpy())
   * ```
   */
  nll_loss = (Y: Tensor, weight?: Tensor, ignore_index?: number, reduction: ReductionStr = 'mean'): Tensor => {
    weight = weight === undefined ? Y.ones_like({ requires_grad: false }) : weight.get(Y)
    const masked_weight = ignore_index === undefined ? weight : weight.mul(Y.ne(ignore_index))
    const nll = this.gather(1, Y.unsqueeze(1)).squeeze(1).neg().mul(masked_weight)
    return reduction === 'mean' ? nll.sum().div(masked_weight.sum()) : nll._do_reduction(reduction)
  }
  // ***** Tensor Properties *****

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
    return num(this.numel()) * this.element_size()
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
    return this.to('LLVM' as any).cast(dtype)
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
   * print(t.dtype, t.numpy())
   * ```
   * ```python exec="true" source="above" session="tensor" result="python"
   * t = t.cast(dtypes.uint8)
   * console.log(t.dtype, t.numpy())
   * ```
   */
  cast = (dtype: DTypeLike): Tensor => {
    const dt = to_dtype(dtype)
    if ([dtypes.uint8, dtypes.uint16].includes(dt) && dtypes.is_float(this.dtype)) {
      // NOTE: values within the int32 range and outside the unsigned dtype range will cause values to wrap around
      return Cast.apply(Cast.apply(this, dtypes.int32), dt)
    }
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
    if (ns !== os && mod(this.shape_num.at(-1)! * os, ns) !== 0) throw new Error('unsupported size in bitcast')
    if ((Array.isArray(this.device) || !this.device.startsWith('DISK')) && ns !== os) {
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
  bool = (): Tensor => {
    return this.cast(dtypes.bool)
  }

  // *** image Tensor function replacements ***

  image_dot = (w: Tensor, acc_dtype?: DType): Tensor => {
    // NOTE: we use a 1x1 conv2d to do the matmul. mxk @ kxn = (1,k,m,1).conv2d(n,k,1,1)
    let x: Tensor = this, dx = this.ndim, dw = w.ndim
    if (!(dx > 0 && dw > 0)) throw new Error(`both tensors need to be at least 1D, got ${dx}D and ${dw}D`)
    if (x.shape.at(-1) !== w.shape.at(-min([w.ndim, 2]))) throw new Error(`cannot image_dot ${x.shape} and ${w.shape}`)

    let bs = prod(this.shape.slice(0, -2)), groups = prod(w.shape_num.slice(0, -2)), cin = w.shape_num.at(-2)!, cout = w.shape_num.at(-1)!
    const out_shape_t = [...this.shape.slice(0, -2), ...(this.shape.length > 1 ? [cout, -1] : [cout])]

    // NOTE: with NHWC we can remove the transposes
    // bs x groups*cin x H x W
    const cx = this.transpose(this.ndim - 1, this.ndim - 2).reshape([idiv(bs, groups), groups * cin, -1, 1])
    // groups*cout x cin x H, W
    const cw = w.transpose(w.ndim - 1, w.ndim - 2).reshape([groups * cout, cin, 1, 1])
    return cx.image_conv2d(cw, undefined, groups, undefined, undefined, undefined, acc_dtype).reshape(out_shape_t).transpose(this.ndim - 1, this.ndim - 2)
  }
  image_conv2d = (weight: Tensor, bias?: Tensor, groups = 1, stride = 1, dilation: number | number[] = 1, padding: number | number[] = 0, acc_dtype?: DType): Tensor => {
    const base_image_type = vars.get_num('FLOAT16', 0) ? dtypes.imageh : dtypes.imagef

    let [bs, _, iy, ix] = this.shape_num, [cout, cin, H, W] = weight.shape_num
    let x: Tensor = this, rcout = idiv(cout, groups), w = weight.reshape([groups, rcout, cin, H, W])

    // hack for non multiples of 4 on cin
    if (mod(cin, 4) !== 0 && !(cin === 1 && mod(groups, 4) === 0)) {
      x = x.reshape([bs, groups, cin, iy, ix]) // do this always?
      const added_input_channels = 4 - mod(cin, 4)
      w = w.pad(range(w.ndim).map((i) => i === 2 ? [0, added_input_channels] as [sint, sint] : undefined))
      x = x.pad(range(x.ndim).map((i) => i === 2 ? [0, added_input_channels] as [sint, sint] : undefined))
      cin = cin + added_input_channels
      x = x.reshape([bs, groups * cin, iy, ix])
    }
    // hack for non multiples of 4 on rcout
    let added_output_channels = 0
    if (mod(rcout, 4) !== 0 && !(rcout === 1 && mod(groups, 4) === 0)) {
      added_output_channels = 4 - mod(rcout, 4)
      rcout += added_output_channels
      cout = groups * rcout
      w = w.pad(range(w.ndim).map((i) => i === 1 ? [0, added_output_channels] as [sint, sint] : undefined))
    }

    // packed (note: flipping bs and iy would make the auto-padding work)
    x = x.permute(0, 2, 3, 1)
    const cin_last = iy === 1 && ix === 1
    if (cin === 1) w = w.reshape([idiv(cout, 4), 4, H, W]).permute(0, 2, 3, 1)
    else if (cin_last) w = w.reshape([idiv(cout, 4), 4, idiv(cin, 4), 4, H, W]).permute(0, 4, 2, 5, 1, 3)
    else w = w.reshape([idiv(cout, 4), 4, idiv(cin, 4), 4, H, W]).permute(0, 4, 2, 5, 3, 1)

    // contiguous creates the image, and early realize static weights (TODO: test for the static weight)
    if (vars.IMAGE >= 2) x = x.cast(base_image_type(bs * iy, idiv(ix * groups * cin, 4), 4)), w = w.cast(base_image_type(idiv(cout, 4), H * W * cin, 4))
    x = x.contiguous(), w = w.contiguous()

    // expand out
    const rcin_hi = cin >= 4 ? idiv(cin, 4) : 1, rcin_lo = cin >= 4 ? 4 : 1
    const cout_expand = [cin === 1 ? idiv(groups, 4) : groups, cin === 1 ? 4 : 1, rcout >= 4 ? idiv(rcout, 4) : 1, rcout >= 4 ? 4 : 1]
    x = x.reshape([bs, iy, ix, groups, rcin_hi, rcin_lo])
    if (cin_last) w = w.reshape([idiv(cout, 4), H, rcin_hi, W, 4, rcin_lo])
    else w = w.reshape([idiv(cout, 4), H, rcin_hi, W, rcin_lo, 4]).permute(0, 1, 2, 3, 5, 4)

    // prepare input
    x = x.permute(0, 3, 4, 5, 1, 2).pad(this._resolve_pool_pads(padding, 2))._pool([H, W], stride, dilation) // -> (bs, groups, rcin_hi, rcin_lo, oy, ox, H, W)
    const oy = x.shape_num[4], ox = x.shape_num[5]
    x = x.permute(0, 4, 5, 1, 2, 3, 6, 7).reshape([bs, oy, ox, ...cout_expand.slice(0, 2), 1, 1, rcin_hi, rcin_lo, H, W])

    // prepare weights
    w = w.permute(0, 4, 2, 5, 1, 3).reshape([1, 1, 1, ...cout_expand, rcin_hi, rcin_lo, H, W])

    // the conv!
    let ret = (x.mul(w)).cast(vars.IMAGE >= 2 ? base_image_type(bs * oy, idiv(ox * cout, 4), 4) : dtypes.float32).sum([-4, -3, -2, -1], undefined, acc_dtype)

    // undo hack for non multiples of 4 on C.rcout
    if (added_output_channels !== 0) {
      ret = ret.reshape([bs, oy, ox, groups, rcout]).get({}, {}, {}, {}, { stop: -added_output_channels })
      cout = groups * (rcout - added_output_channels)
    }
    // NCHW output
    ret = ret.reshape([bs, oy, ox, cout]).permute(0, 3, 1, 2)
    return bias === undefined ? ret : ret.add(bias.reshape([1, -1, 1, 1]))
  }
}

// Metadata wrapper function
if (vars.TRACEMETA >= 1) {
  const wrapper = (fn: any, name: string, isArrow: boolean = false) => {
    return function (this: any, ...args: any[]) {
      if (_METADATA.get()) {
        return isArrow ? fn(...args) : fn.apply(this, args)
      }
      let caller: string
      if (vars.TRACEMETA >= 2) {
        throw new NotImplemented()
      } else {
        caller = ''
      }
      const token = _METADATA.set(new Metadata(name, caller))
      const result = isArrow ? fn(...args) : fn.apply(this, args)
      _METADATA.reset(token)
      return result
    }
  }

  const IGNORED = ['constructor', 'backward', 'toString', 'sequential']

  const staticFuncs = Object.entries(Tensor).filter(([k, v]) => typeof v === 'function' && !IGNORED.includes(k))
  for (const [name, func] of staticFuncs) (Tensor as any)[name] = wrapper(func, name)

  const instanceFuncs = Object.getOwnPropertyNames(Tensor.prototype).map((k) => [k, Object.getOwnPropertyDescriptor(Tensor.prototype, k)?.value]).filter(([k, v]) => typeof v === 'function' && !IGNORED.includes(k))
  for (const [name, func] of instanceFuncs) (Tensor.prototype as any)[name] = wrapper(func, name)

  // @ts-ignore reassigning Tensor
  // deno-lint-ignore no-class-assign
  Tensor = new Proxy(Tensor, {
    construct(target, args) {
      const instance = new target(...args)

      const instanceProps = Object.getOwnPropertyDescriptors(instance)
      for (const [name, descriptor] of Object.entries(instanceProps)) {
        const func = descriptor.value
        if (
          typeof func === 'function' &&
          !IGNORED.includes(name) &&
          // Check if it's an arrow function (no prototype property is a heuristic)
          !Object.prototype.hasOwnProperty.call(func, 'prototype')
        ) {
          Object.defineProperty(instance, name, { ...descriptor, value: wrapper(func, name, true) })
        }
      }
      return instance
    },
  })
}
