import { Buffer, DeviceType } from '../device.ts'
import { ConstType, DType, DTypeLike, dtypes, ImageDType, to_dtype } from '../dtype.ts'
import { _METADATA, all_int, all_same, assert, DEBUG, get_number_env, isEq, isinstance, LAZYCACHE, listStr, Metadata, prod, range, SPLIT_REDUCEOP } from '../helpers.ts'
import { exec_alu, GroupOp, identity_element, mod, ne, python_alu, resolve, sint_prod, sub } from '../ops.ts'
import { ConstLike } from '../ops.ts'
import { idiv, MathTrait, Ops, sint, UOp } from '../ops.ts'
import { ShapeTracker } from '../shape/shapetracker.ts'

const lazycache = new Map<any, LazyBuffer>()
export const create_lazybuffer = (device: DeviceType, st: ShapeTracker, dtype: DType, op?: Ops, arg?: any, srcs: LazyBuffer[] = [], base?: LazyBuffer, enable_cache = Boolean(LAZYCACHE)) => {
  dtype = to_dtype(dtype)
  if (op === Ops.CONST) [arg, enable_cache] = [!isinstance(arg, UOp) ? dtypes.as_const(arg, dtype) : arg, true]

  const cache_key = base === undefined ? [device, st, dtype, op, arg, srcs.map((x) => new WeakRef(x))] : [st, new WeakRef(base)]
  if (enable_cache) {
    const rret = lazycache.get(cache_key)
    if (rret !== undefined) return rret
  }

  const ret = new LazyBuffer(device, st, dtype, op, arg, srcs, base, _METADATA.value)
  if (enable_cache) lazycache.set(cache_key, ret)
  return ret
}
export const view_supported_devices = ['LLVM', 'CLANG', 'CUDA', 'NV', 'AMD', 'METAL', 'QCOM', 'DSP', 'DISK']
export class LazyBuffer extends MathTrait {
  shape: sint[]
  size: number

  op?: Ops
  arg?: any
  srcs?: LazyBuffer[]
  dtype: DType

  buffer?: Buffer
  contiguous_child?: [WeakRef<LazyBuffer>, ShapeTracker]
  forced_realize?: boolean
  _base?: LazyBuffer
  constructor(
    public device: DeviceType,
    public st: ShapeTracker,
    dtype: DTypeLike,
    op?: Ops,
    arg?: any,
    srcs: LazyBuffer[] = [],
    base?: LazyBuffer,
    public metadata?: Metadata,
  ) {
    super()
    this.shape = st.shape, this.size = st.size, this.dtype = to_dtype(dtype)
    if (base === undefined) {
      //       // properties on base
      this.op = op, this.arg = arg, this.srcs = srcs // this === a UOp, except the src === LazyBuffers && !UOps
      assert(this.op !== Ops.ASSIGN || srcs[0].base.realized !== undefined, 'assign target must be realized')
      assert(all_same(this.srcs.map((x) => x.st.shape)), `src shape mismatch! ${this.srcs}`)
      //         // some LazyBuffers can be processed with only a view, no AST required
      if (this.op === Ops.BUFFER_VIEW) this.buffer = srcs[0].base.buffer?.view(st.size, this.dtype, (srcs[0].st.views[0].offset as number) * srcs[0].dtype.itemsize)
      else this.buffer = this.op === Ops.ASSIGN ? srcs[0].base.buffer : new Buffer(device, this.size, this.dtype)
      this.forced_realize = false
    } else {
      //       // properties on view
      assert(base.base === base, 'base must be a base itself')
      this._base = base
    }
  }
  __del__ = () => {
    if (this.buffer) this.buffer.ref(-1)
  }
  override toString = (): string => {
    return `<LB ${this.device} ${listStr(this.shape)} ${this.dtype.toString().slice(7)} ${this.base !== this ? this.st.toString() : `(${this.op}, ${this.realized})`}>`
  }
  get realized(): undefined | Buffer {
    // NOTE: we check for a lack of srcs instead of an allocated buffer to make unrealized assigns return undefined here
    return this._base === undefined && !this.srcs ? this.buffer : undefined
  }

  //   // NOTE: this has to be a function to prevent this reference
  get base(): LazyBuffer {
    return this._base !== undefined ? this._base : this
  }

  //   // same API as multi
  get lbs(): LazyBuffer[] {
    return [this]
  }

  static metaop = (op: Ops, shape: sint[], dtype: DType, device: DeviceType, arg?: any, src: LazyBuffer[] = [], enable_cache = false): LazyBuffer => {
    assert(isinstance(src, Array))
    return create_lazybuffer(device, ShapeTracker.from_shape(shape), dtype, op, arg, src, undefined, enable_cache)
  }
  override const_like = (b: ConstLike): typeof this => this.const_with_shape(b as ConstType, this.shape) as typeof this
  const_with_shape = (val: ConstType, shape: sint[]): LazyBuffer => {
    assert(isinstance(val, Number) || isinstance(val, Boolean), `val=${val} has ${typeof val}, !a ConstType`)
    return LazyBuffer.metaop(Ops.CONST, [], this.dtype, this.device, val).reshape(range(shape.length).map((x) => 1)).expand(shape)
  }
  //   @property
  get is_realized(): boolean {
    return this.base.realized !== undefined
  }

  assign = (x: LazyBuffer): LazyBuffer => {
    assert(x.size === this.size, `assign target must have same size ${this.size} !== ${x.size}`)
    assert(this.is_realized, `assign target must be realized ${this}`)
    return LazyBuffer.metaop(Ops.ASSIGN, this.shape, this.dtype, this.device, this.st.contiguous ? undefined : this.st, [this, x], true) // NOTE: assign to VIEW === fine
  }
  can_view = () => {
    return (this.st.consecutive && !this.is_unrealized_const() && !isinstance(this.dtype, ImageDType) && view_supported_devices.includes(this.device.split(':')[0]))
  }
  contiguous = (allow_buffer_view = true) => {
    if (!this.st.contiguous || this.size !== this.base.size || this.is_unrealized_const()) {
      const ret = allow_buffer_view && this.can_view() ? this.alu(Ops.BUFFER_VIEW) : this.alu(Ops.CONTIGUOUS)
      const sti = this.st.invert(this.base.shape)
      if (sti !== undefined) this.base.contiguous_child = [new WeakRef(ret), sti]
      return ret
    }
    this.base.forced_realize = true
    return this
  }

  bitcast = (dtype: DType): LazyBuffer => this.cast(dtype, true)
  cast = (dtype: DType, bitcast = false, allow_buffer_view = true): LazyBuffer => {
    if (this.dtype === dtype) return this
    if (this.device.startsWith('DISK') && !bitcast) throw new Error('attempted to cast disk buffer (bitcast only)')
    if (this.is_unrealized_unmasked_const() && !bitcast) return create_lazybuffer(this.device, this.st, dtype, Ops.CONST, dtypes.as_const(this.base.arg, dtype))
    let new_shape = this.shape
    if (bitcast && this.dtype.itemsize !== dtype.itemsize) {
      if (!this.device.startsWith('DISK')) throw new Error('shape changing bitcast only supported on DISK right now')
      if (!all_int(new_shape)) throw new Error("shape changing bitcast with symbolic shape isn't supported yet")
      //       // https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
      if ((new_shape.at(-1)! * this.dtype.itemsize) % dtype.itemsize !== 0) throw new Error('unsupported size in bitcast')
      new_shape = [...new_shape.slice(0, -1), idiv(new_shape.at(-1)! * this.dtype.itemsize, dtype.itemsize)]
    } else if (get_number_env('CAST_BEFORE_VIEW', 1) && dtype.itemsize <= this.dtype.itemsize && this !== this.base) {
      //       // TODO: applying this makes gpt2 slower
      return this.base.cast(dtype, bitcast)._view(this.st)
    }
    const cast_op: Ops = bitcast ? (this.can_view() && allow_buffer_view ? Ops.BUFFER_VIEW : Ops.BITCAST) : Ops.CAST
    return create_lazybuffer(this.device, ShapeTracker.from_shape(new_shape), dtype, cast_op, undefined, [this])
  }
  is_unrealized_const = () => this.base.realized === undefined && this.base.op === Ops.CONST && !isinstance(this.base.arg, UOp)
  is_unrealized_unmasked_const = () => this.is_unrealized_const() && this.st.views.every((v) => v.mask === undefined)

  _copy = (device: DeviceType): LazyBuffer => {
    assert(!!this.st.contiguous && this.size === this.base.size, `can only copy contig ${this} ${this.base}`)
    return create_lazybuffer(device, ShapeTracker.from_shape(this.shape), this.dtype, Ops.COPY, this.buffer?.nbytes, [this], undefined, false)
  }
  copy_to_device = (device: DeviceType, force = false, clone = false): LazyBuffer => {
    //     // no COPY
    if (this.device === device && !clone) return this

    //     // double COPY = one COPY
    if (!force && this.st.contiguous && this.size === this.base.size && !this.base.realized && this.base.op === Ops.COPY) {
      return this.base.srcs![0].copy_to_device(device).reshape(this.st.shape)
    }

    //     // const doesn't have to be copied (issues with disk tensor)
    if (this.is_unrealized_const()) {
      return LazyBuffer.metaop(Ops.CONST, [], this.dtype, device, this.base.arg)._view(this.st)
    }

    //     // if it's a shrink, do the shrink before the copy with CONTIGUOUS
    if (prod(this.st.shape as number[]) < prod(this.base.st.shape as number[])) return this.contiguous()._copy(device)

    //     // copy the base && apply the shapetracker on the new device
    return this.base._copy(device)._view(this.st)
  }
  clone = (): LazyBuffer => this.copy_to_device(this.device, undefined, true)

  override alu = (op: Ops, ...in_srcs: typeof this[]): typeof this => {
    const srcs: LazyBuffer[] = []
    for (const s of [this, ...in_srcs]) {
      if (isEq(s, s.base) && s.base.contiguous_child) { // KAREL: idk maybe it's never true
        const root = s.base.contiguous_child[0].deref()
        if (root !== undefined) srcs.push(root._view(s.base.contiguous_child[1]))
      } else {
        srcs.push(s)
      }
    }
    const dts = (op === Ops.WHERE ? srcs.slice(1) : srcs).map((x) => x.dtype.base)
    if (!all_same(dts)) throw new Error(`all dtypes must match ${dts} on ${op}`)
    assert(all_same(srcs.map((x) => x.shape)), `all shapes must be the same ${srcs.map((x) => x.shape)}`)
    if (op === Ops.WHERE) assert(srcs[0].dtype === dtypes.bool, 'Ops.WHERE must have the first arg be bool')

    const out_dtype = [Ops.CMPLT, Ops.CMPNE].includes(op) ? dtypes.bool : srcs.at(-1)!.dtype

    //     // const folding
    if (python_alu.has(op) && srcs.every((s) => s.is_unrealized_unmasked_const())) return this.cast(out_dtype).const_like(exec_alu(op, out_dtype, srcs.map((s) => s.base.arg))) as typeof this
    if (GroupOp.Binary.includes(op)) {
      const [x, y] = [this, in_srcs[0]]
      if (op === Ops.ADD) {
        if (y.is_unrealized_unmasked_const() && y.base.arg === 0) return x
        if (x.is_unrealized_unmasked_const() && x.base.arg === 0) return y
      }
      if (op === Ops.MUL) {
        if (x.is_unrealized_unmasked_const() && [1, 0].includes(x.base.arg)) return x.base.arg === 1 ? y : y.const_like(0)
        if (y.is_unrealized_unmasked_const() && [1, 0].includes(y.base.arg)) return y.base.arg === 1 ? x : x.const_like(0)
      }
      if (op === Ops.IDIV && y.is_unrealized_unmasked_const() && y.base.arg === 1) return x
    }
    return create_lazybuffer(this.device, ShapeTracker.from_shape(this.shape), out_dtype, op, undefined, srcs) as typeof this
  }
  //   // *** reduce ops ***

  _reduce_op = (op: Ops, axis: number[]): LazyBuffer => {
    assert(axis.every((x) => 0 <= x && x < this.shape.length), `axis args ${axis} out of range for shape ${this.shape}`)
    axis = axis.filter((x) => resolve(ne(this.shape[x], 1))).toSorted()
    if (axis.length === 0) return this
    return create_lazybuffer(this.device, ShapeTracker.from_shape(this.st.reduce(axis)), this.dtype, Ops.REDUCE_AXIS, [op, axis], [this])
  }
  r = (op: Ops, axis: number[]): LazyBuffer => {
    const new_shape = this.st.reduce(axis)
    //     // TODO: this logic should move to the scheduler
    if (this.shape.includes(0) && !new_shape.includes(0)) return this.const_with_shape(identity_element(op, this.dtype) as number, new_shape)

    //     // const folding
    //     // TODO: fold this for symbolic?
    if (this.is_unrealized_unmasked_const() && all_int(this.shape)) {
      if (op === Ops.ADD) return this.const_with_shape(this.base.arg * (sint_prod(axis.map((i) => this.shape[i])) as number), new_shape)
      if (op === Ops.MUL) return this.const_with_shape(this.base.arg ** (sint_prod(axis.map((i) => this.shape[i])) as number), new_shape)
      if (op === Ops.MAX) return this.const_with_shape(this.base.arg, new_shape)
    }
    //     // TODO: can we split symbolic shape if the reduce axis !== symbolic?
    if (!SPLIT_REDUCEOP || !all_int(this.shape) || (this.shape.includes(0)) || idiv(prod(this.shape), prod(new_shape as number[])) < get_number_env('REDUCEOP_SPLIT_THRESHOLD', 32768)) return this._reduce_op(op, axis)

    //     // if there are few globals, make some reduces into globals by splitting into two kernels
    //     // cap output buffer to 2**22: heuristic number of global outputs to achieve max occupancy with enough locals+upcasts for gemm
    //     //   ~2**10 should be enough if GROUP === used
    //     // 256 split maximum should be "negligible reduce" for low prod(new_shape), 8 split minimum.
    //     // split === moved to the end to provide maximum locality for the second phase reduce.
    const this_real_strides = this.st.real_strides(true)
    const split_candidates = range(Math.min(256, idiv(2 ** get_number_env('REDUCEOP_SPLIT_SIZE', 22), prod(new_shape as number[]))), 8 - 1, -1).flatMap((x) => axis.map((i) => [i, x]))
      .filter(([i, x]) => mod(this.shape[i], x) as number === 0 && this_real_strides[i] !== 0)
    if (!split_candidates) return this._reduce_op(op, axis)
    const [dim_to_split, divisor] = split_candidates[0]
    const splitted_shape = [...this.shape.slice(0, dim_to_split), divisor, idiv(this.shape[dim_to_split], divisor), ...this.shape.slice(dim_to_split + 1)]
    const splitted = this.reshape(splitted_shape).permute([...range(splitted_shape.length).filter((x) => x !== dim_to_split), dim_to_split])
    if (DEBUG >= 3) console.log(`split ${divisor}: ${this.shape} -> ${splitted.shape} -> ${new_shape}`)
    return splitted._reduce_op(op, axis)._reduce_op(op, [new_shape.length]).reshape(new_shape) // reduce original axes, then split
  }
  //   // *** movement ops ***

  _view = (new_st: ShapeTracker): LazyBuffer => {
    if (this.st.size === 0 || (new_st.views.at(-1)!.mask !== undefined && new_st.views.at(-1)!.mask!.some((x) => (x[1] as number) - (x[0] as number) === 0))) {
      return this.const_with_shape(0, new_st.shape)
    }
    if (new_st.contiguous && isEq(this.base.shape, new_st.shape)) return this.base
    return create_lazybuffer(this.device, new_st, this.dtype, undefined, undefined, undefined, this.base)
  }
  reshape = (arg: sint[]) => this._view(this.st.reshape(arg))
  pad = (arg: [sint, sint][]) => this._view(this.st.pad(arg))
  expand = (arg: sint[]) => this._view(this.st.expand(arg))
  permute = (arg: number[]) => this._view(this.st.permute(arg))
  shrink = (arg: [sint, sint][]) => this._view(this.st.shrink(arg))
  stride = (arg: number[]) => this._view(this.st.stride(arg))
}
