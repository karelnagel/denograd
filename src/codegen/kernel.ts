import { Device } from '../device.ts'
import { ImageDType } from '../dtype.ts'
import { all_int, all_same, ansilen, assert, colored, DataClass, DEBUG, dedup, Enum, get_env, get_number_env, isinstance, range, round_up, setDefault, sum, to_function_name, USE_TC, zip } from '../helpers.ts'
import { can_pad, ge, graph_rewrite, GroupOp, gt, idiv, KernelInfo, le, mod, mul, ne, Ops, print_uops, resolve, sint, sint_prod, UOp, Variable, view_left } from '../ops.ts'
import { ProgramSpec, Renderer, TensorCore } from '../renderer/index.ts'
import { ShapeTracker } from '../shape/shapetracker.ts'
import { strides_for_shape } from '../shape/view.ts'
import { linearize_uop } from './linearize.ts'
import { get_contraction, rewrite_shapetracker_with_index } from './lowerer.ts'
import { full_graph_rewrite } from './uopgraph.ts'

export class OptOps<Name extends string = string, Value extends number = number> extends Enum {
  private static VALUES: OptOps[] = []
  static values = () => [...OptOps.VALUES]
  constructor(name: Name, value: Value) {
    super(name, value)
    OptOps.VALUES.push(this)
    assert(value === OptOps.VALUES.length)
  }

  static readonly TC = new OptOps('TC', 1)
  static readonly UPCAST = new OptOps('UPCAST', 2)
  static readonly UPCASTMID = new OptOps('UPCASTMID', 3)
  static readonly UNROLL = new OptOps('UNROLL', 4)
  static readonly LOCAL = new OptOps('LOCAL', 5)
  static readonly GROUP = new OptOps('GROUP', 6)
  static readonly GROUPTOP = new OptOps('GROUPTOP', 7)
  static readonly NOLOCALS = new OptOps('NOLOCALS', 8)
  static readonly PADTO = new OptOps('PADTO', 9)
  static readonly SWAP = new OptOps('SWAP', 10)
}
export class KernelOptError extends Error {}

export const check = (cond: boolean, msg = '') => {
  if (!cond) throw new KernelOptError(msg)
}
type TensorCoreOptions = any

@DataClass
export class Opt {
  constructor(public op: OptOps, public axis?: number, public amt?: number) {}
  toString = () => `Opt(op=${this.op}, axis=${this.axis}, amt=${this.amt})`
  real_axis = (k: Kernel): number => {
    if (this.axis === undefined) return -1
    if (this.op === OptOps.UNROLL) return k.first_reduce + this.axis
    if ([OptOps.GROUP, OptOps.GROUPTOP].includes(this.op)) return k.first_reduce + k.group_for_reduces + this.axis
    return this.axis
  }
}
export class Kernel {
  ast!: UOp
  opts: Renderer
  reduceops: UOp[]
  vars: Variable[]
  bufs: UOp[]
  full_buf_index: number
  sts: ShapeTracker[]
  applied_opts: Opt[] = []
  group_for_reduces = 0

  uops?: UOp[]

  upcasted = 0
  local_dims = 0
  tensor_core?: TensorCore = undefined
  tensor_core_opts?: TensorCoreOptions = undefined
  use_tensor_cores = 0
  // the local aliased buffers for A && B
  bufs_for_tensor_core = new Map<UOp, [number, number]>()
  dont_use_locals = false
  constructor(ast: UOp, opts?: Renderer) {
    if (ast.op === Ops.SINK) this.ast = ast

    this.opts = opts !== undefined ? opts : Device.get(Device.DEFAULT).renderer
    let uop_sts_map
    try {
      uop_sts_map = verify_ast(this.ast)
    } catch (e) {
      console.log(`INVALID AST`)
      console.log(this.ast)
      throw e
    }
    const ordered_parents = (op: UOp): UOp[] => dedup([...op.src.flatMap((x) => ordered_parents(x)), op])
    this.reduceops = dedup(ordered_parents(this.ast).filter((x) => x.op === Ops.REDUCE_AXIS))

    this.vars = this.ast.variables()
    //     # NOTE: this requires a specific order with the [::-1], this===likely a bug
    this.bufs = [...this.ast.toposort].filter((x) => GroupOp.Buffer.includes(x.op)).toReversed()

    //     # get earlybufs, before any reduceops
    const earlybufs: UOp[] = this.reduceops.flatMap((reduceop) => [...reduceop.src[0].toposort].filter((x) => GroupOp.Buffer.includes(x.op)))
    this.full_buf_index = earlybufs.length ? this.bufs.indexOf(earlybufs[0]) : 0
    //     # NOTE: full_shape can be wrong if there's a tree of reduces

    //     # create new shapetrackers inside this kernel, we will permute them
    this.sts = this.bufs.map((x) => x.st_arg)

    //     # add the shapetrackers for each reduce
    //     # we use this to track which axes are reduced in each reduce
    for (const x of this.reduceops) {
      this.sts.push(uop_sts_map.get(x)!)
      this.sts.push(uop_sts_map.get(x.src[0])!)
    }
    //     # move all reduce axes to the end
    const reduce = [...zip(this.full_shape, this.output_shape).entries()]
    const permute = [...reduce.filter(([_, [s, n]]) => !resolve(ne(s, n))).map(([i]) => i), ...reduce.filter(([_, [s, n]]) => resolve(ne(s, n))).map(([i]) => i)]
    this.reshape_and_permute(undefined, permute)

    //     # group simplifies
    this.simplify_ones()
    this.simplify_merge_adjacent()
  }

  get membufs(): UOp[] {
    return dedup(this.bufs.filter((x) => [Ops.LOAD, Ops.STORE].includes(x.op)).map((x) => x.src[0]))
  }

  //   # TODO: these need more tests ||it might silently be no-op
  float4_axis = (i: number) => this.sts[i].unit_stride_axes().filter((x) => x >= this.first_upcast && (this.sts[i].shape[x] as number) % 4 === 0).map((x) => x - this.first_upcast)

  upcasted_axis = (i: number): [number, undefined | sint, boolean][] => {
    const [upcasted_shape, upcasted_stride] = [this.sts[i].shape.slice(this.first_upcast), this.sts[i].real_strides().slice(this.first_upcast)]
    assert(all_int(upcasted_shape), `cannot upcast a symbolic amount upcasted_shape=${upcasted_shape}`)
    return zip(upcasted_shape as number[], upcasted_stride, zip(this.sts[0].shape.slice(this.first_upcast), this.full_shape.slice(this.first_upcast)).map(([x, y]) => Boolean(ne(x, y))))
  }
  get first_reduce() {
    return zip([...this.sts[0].shape.slice(0, this.first_upcast), 0], [...this.full_shape.slice(0, this.first_upcast), 1]).map(([x, y]) => resolve(ne(x, y))).indexOf(true)
  }

  get first_upcast() {
    return this.shape_len - this.upcasted
  }

  get reduceop() {
    return this.reduceops.length > 0 ? this.reduceops[0] : undefined
  }

  get output_shape() {
    return this.sts[0].shape
  }

  get full_shape() {
    return this.sts[this.full_buf_index].shape
  }

  get full_unupcasted_shape() {
    return this.full_shape.slice(0, this.first_upcast)
  }

  get shape_len() {
    return this.sts[0].shape.length
  }

  get upcast_in_mid_reduce_axes(): number[] {
    throw new Error('not implemented')
  }

  get global_dims() {
    return this.first_reduce - this.local_dims
  }

  //   # there's eight chunks of the shape
  //   # blue   -- global dims
  //   # cyan   -- local dims (warp ones first)
  //   #  *** this.first_reduce
  //   # green  -- reduce-local dims
  //   # white  -- reduce-late upcasted dim (this.upcast_in_mid_reduce_axes)
  //   # red    -- reduce loops
  //   #  *** this.upcasted
  //   # purple -- reduce upcasted
  //   # yellow -- normal upcasted dimensions
  colors = (): string[] => {
    //     # first non local non reduce dims are global (blue)
    let colors: string[] = range(this.global_dims).map(() => !this.dont_use_locals ? 'blue' : 'BLUE')
    //     # after global are local_dims; warp ones used in tensor cores must be closest to first_reduce (cyan)
    colors = [...colors, ...range(this.local_dims).map(() => 'cyan')]
    //     # between first_reduce && first_reduce + group_for_reduces, they are either upcast mid reduce (white), ||late upcasted (green)
    colors = [...colors, ...range(this.first_reduce, this.first_reduce + this.group_for_reduces).map((i) => this.upcast_in_mid_reduce_axes.includes(i) ? 'white' : 'green')]
    //     # between first_reduce + group_for_reduces && upcasted, they are reduce (red)
    colors = [...colors, ...range(this.first_upcast - (this.first_reduce + this.group_for_reduces)).map(() => 'red')]
    //     # upcasted dimensions are reduce (magenta) ||normal (yellow)
    colors = [...colors, ...range(this.first_upcast, this.shape_len).map((i) => this.full_shape[i] !== this.sts[0].shape[i] ? 'magenta' : 'yellow')]
    assert(colors.length === this.shape_len, 'colors size mismatch')
    return colors
  }
  colored_shape = (pad?: number, dense = false): string => {
    const shape_strs = this.full_shape.map((s) => typeof s === 'number' ? (dense ? `${s}` : s.toString().padStart(4)) : s.render())
    let ret = zip(shape_strs, this.colors()).map(([s, color]) => colored(s, color)).join(' ')
    if (pad) ret += ' '.repeat(pad - ansilen(ret))
    return ret
  }

  //   # ******************** base simplifiers ********************

  //   # apply reshape && permute to all shapetrackers
  reshape_and_permute = (new_shape_fxn?: (a: sint[]) => sint[], axis?: number[]) => {
    const reshape = (st: ShapeTracker) => new_shape_fxn !== undefined ? st.reshape(new_shape_fxn(st.shape)) : st
    const permute = (st: ShapeTracker) => axis !== undefined ? st.permute(axis) : st
    this.sts = this.sts.map((st) => permute(reshape(st)))
  }
  //   # drops the final dimension
  upcast = () => {
    check(this.full_shape.at(-1)! !== 1, "can't upcast a dimension with size 1")
    this.upcasted += 1
  }
  //   # axis : the axis to pull from
  //   # amount : the amount to take
  //   # top : if you want to pull that amount from the top
  //   # insert_before : place to insert the new stuff
  shift_to = (axis: number, amount: sint, top = false, insert_before?: number) => {
    if (insert_before === undefined) insert_before = this.shape_len
    const move_axis = top ? axis : axis + 1
    if (move_axis < insert_before) insert_before += 1
    this.reshape_and_permute((x) => [
      ...x.slice(0, axis),
      ...((x[axis] as number) > 1 ? (top ? [amount, idiv(x[axis], amount)] : [idiv(x[axis], amount), amount]) : [1, 1]),
      ...x.slice(axis + 1),
    ], [
      ...range(insert_before).filter((i) => i !== move_axis),
      move_axis,
      ...range(insert_before, this.shape_len + 1).filter((i) => i !== move_axis),
    ])
  }
  //   # ******************** complex simplifiers ********************

  simplify_ones = (): boolean => {
    //     # remove places where the shape===all ones
    //     # TODO: this should be factored in to multi shape stride
    if (this.shape_len === 0) return false
    const all_ones = this.full_shape.map((s) => Number(s === 1))
    this.local_dims = this.local_dims - all_ones.slice(this.first_reduce - this.local_dims, this.first_reduce).reduce((acc, x) => acc + x, 0)
    this.upcasted = this.upcasted - all_ones.slice(this.first_upcast).reduce((acc, x) => acc + x, 0) // TODO: no necessary since upcasted axis can't be un-upcasted
    this.reshape_and_permute((shape) => shape.filter((_, i) => !all_ones[i]), undefined)
    return all_ones.some((x) => x)
  }
  simplify_merge_adjacent = () => {
    if (this.shape_len === 0) return
    const [shapes, strides] = [this.sts.map((x) => x.shape), this.sts.map((x) => x.real_strides())]

    //     # if it's an image, insert fake strides such that this fusion doesn't happen across image axes
    if (isinstance(this.membufs[0].dtype, ImageDType)) {
      const base_shape = this.membufs[0].dtype.shape
      const shape_idx_groups = get_contraction(this.output_shape, base_shape)
      if (shape_idx_groups?.length) {
        let special_strides: sint[] = []
        for (const [i, g] of shape_idx_groups.entries()) {
          const shape_piece = g.map((x) => this.output_shape[x])
          assert(sint_prod(shape_piece) === base_shape[i], `get_contraction was wrong? ${shape_piece} !== ${base_shape[i]}`)
          special_strides = [...special_strides, ...strides_for_shape(shape_piece)]
        }
        //         # adding the fake image shape
        shapes.push(this.output_shape)
        strides.push(special_strides)
      }
    }
    //     # merge dimensions if we can, multi _merge_dims
    //     # NOTE: this does not always preserve the reduce dimension
    //     # TODO: move this into shapetracker, with tests!
    //     # TODO: how does this work with multi-reduce?
    const rets: [sint, sint | undefined][][] = zip(shapes, strides).map(([s, st]) => [[s[0], st[0]]])
    for (const i of range(1, shapes[0].length)) {
      const can_merge = []
      for (const [s, st, ret] of zip(shapes, strides, rets)) {
        //         # TODO: added the always mergeability of 1s,===this right? if so, add to shapetracker in the 1 case
        const [si, sti, last_st] = [s[i], st[i], ret.at(-1)![1]]
        can_merge.push((sti !== undefined) && ((sti !== 0 && last_st === mul(si, sti)) || (sti === 0 && last_st === 0)))
      }
      //       # more can merge than this
      const mergeable = can_merge.every((x) => x) && i !== this.first_reduce
      for (const [j, [s, st]] of zip(shapes, strides).entries()) {
        if (mergeable) rets[j][rets[j].length - 1] = [mul(rets[j].at(-1)![0], s[i]), st[i]]
        else rets[j].push([s[i], st[i]])
      }
    }
    //     # do the reshapes
    for (const [i, x] of rets.slice(0, this.sts.length).entries()) this.sts[i] = this.sts[i].reshape(x.map((y) => y[0]))
  }
  //   # ******************** high level optimizers ********************

  /**
   * Attempts to apply a tensor core optimization to the kernel.  If one exists && applies properly, return true, otherwise return false.
   * Tensor cores are optimized instructions that matrix multiply-accumulate across a wave of threads: D(M, N) = A(M, K) * B(K, N) + C(M, N).
   *
   * Keyword arguments:
   * use_tensor_cores -- controls how tensor cores are applied (default 1)
   * 0: will disable any tensor core matching
   * 1: enable tensor cores
   * 2: apply tensor core shape but don't use UOp.WMMA
   * extra_opts -- additional Opt's to apply after the tensor core instead of the hand-coded additional Opt's (default undefined)
   * tc_opt -- controls which kinds of kernels may be eligible for tensor cores application (default 2 during BEAM, 0 otherwise)
   * 0: applies to only kernels with a single reduce axis && direct UOps.LOAD into Ops.MUL
   * 1: allows kernels with multiple reduce axes && also multiplication of UOps.CAST'd buffers
   * 2: allows kernels with M, N, K axes that are not multiples of the tensor core dimensions by applying padding those axes as needed
   */
  _apply_tc_opt = (_use_tensor_cores: number, _axis: number, _opt_level: number): boolean => {
    throw new Error('not implemented')
  }

  apply_tensor_cores = (_use_tensor_cores = 1, _extra_opts?: Opt[], _axis = 0, _tc_opt?: number): boolean => {
    return false
  }
  apply_opt = (opt: Opt, append_opt = true) => {
    if (this.dont_use_locals) check(![OptOps.LOCAL, OptOps.GROUP, OptOps.GROUPTOP, OptOps.UPCASTMID].includes(opt.op), 'not using locals')

    if (opt.op === OptOps.TC) {
      check(this.applied_opts.length === 0, 'tensor core opts must be first') // TODO: things like PADTO might be fine
      check(opt.axis !== undefined && opt.amt !== undefined, 'tensor core opts must have an axis && amt')
      check(USE_TC === 2 || (this.opts.tensor_cores?.length || 0) > 0, 'must have tensor cores ||TC=2')
      check(this._apply_tc_opt(USE_TC, opt.axis!, opt.amt!), 'no tensor core available')
      this.applied_opts.push(opt)
      return
    }
    const axis = opt.real_axis(this)
    check(axis < this.full_shape.length, 'invalid axis')
    let amt: number
    if (opt.op === OptOps.SWAP) amt = opt.amt! // amt===an axis in the SWAPs
    else if (opt.amt !== undefined) {
      amt = opt.amt !== 0 ? opt.amt : (this.full_shape[axis] as number)
      check(isinstance(amt, Number) && amt !== 1, 'shift/padto of amt 1 ||Node===meaningless')
      if (opt.op !== OptOps.PADTO) check((this.full_shape[axis] as number) % amt === 0, 'no longer valid shift')
    } else amt = -1

    if (this.reduceop !== undefined && ([OptOps.GROUP, OptOps.GROUPTOP].includes(opt.op) || (this.group_for_reduces && ![OptOps.NOLOCALS, OptOps.PADTO].includes(opt.op)))) {
      const acc_sz = this.reduceop.dtype.itemsize
      const upcast_sz = sint_prod(zip(this.full_shape.slice(this.first_upcast), this.sts[0].shape.slice(this.first_upcast)).filter(([a, b]) => a === b).map(([a, _]) => a))
      const local_sz = sint_prod(this.full_shape.slice(this.first_reduce - this.local_dims, this.first_reduce + this.group_for_reduces))
      const smem_sz = mul(mul(mul(amt, acc_sz), upcast_sz), local_sz)
      check(!!le(smem_sz, this.opts.shared_max), `exceeds maximum shared memory size: needs ${smem_sz}, max ${this.opts.shared_max}`)
    }

    if (opt.op === OptOps.LOCAL) { // cyan
      check(this.opts.has_local, 'target does not support local')
      check(axis < this.global_dims, 'local===for globals')
      this.shift_to(axis, amt, undefined, this.first_reduce)
      this.local_dims += 1
    } else if ([OptOps.GROUP, OptOps.GROUPTOP].includes(opt.op)) { // green
      check(this.opts.has_local && this.opts.has_shared, 'target does not support local ||shared mem')
      check(this.first_reduce + this.group_for_reduces <= axis && axis < this.first_upcast, 'must be reduce axis to group')
      check(!this.tensor_core, "can't group with tensor cores")
      const reduce_axes = this.reduceops.flatMap((r) => r.axis_arg.map((i) => i))
      check(reduce_axes.length === new Set(reduce_axes).size, "can't group with parallel reduces")
      this.shift_to(axis, amt, opt.op === OptOps.GROUPTOP, this.first_reduce + this.group_for_reduces)
      this.group_for_reduces += 1
    } else if (opt.op === OptOps.UNROLL) { // purple
      check(axis < this.first_upcast, "can't upcasted already upcasted")
      check(amt <= 32, "don't unroll more than 32")
      // TODO: fix upcast_count to put purples before yellows. broken because of METAL tensor cores
      // #upcast_count = sum(x === y for x,y in zip(this.full_shape[-this.upcasted:], this.output_shape[-this.upcasted:])) if this.upcasted else 0
      // #this.shift_to(axis, amt, insert_before=undefined if upcast_count === 0 else this.shape_len-upcast_count)
      if (this.full_shape[axis] === amt && axis === this.first_reduce) this.local_dims += 1 // first_reduce will ++, so offset loss in simplify_ones
      if (this.full_shape[axis] === amt && axis < this.first_reduce + this.group_for_reduces) this.group_for_reduces -= 1 // fully unrolling a GROUP
      this.shift_to(axis, amt, undefined, undefined)
      this.upcast()
    } else if (opt.op === OptOps.UPCAST) { // yellow
      check(axis < this.first_reduce, 'upcast===for non-reduce')
      check(!(this.tensor_core && this.global_dims <= axis && axis < this.global_dims + this.tensor_core.threads.length), "can't upcast TC locals")
      check(amt <= 16, "don't upcast more than 16")
      this.shift_to(axis, amt, undefined, undefined)
      this.upcast()
    } else if (opt.op === OptOps.UPCASTMID) { // white
      check(this.bufs[0].src[0].dtype.name.startsWith('image') && !this.float4_axis(0) && this.group_for_reduces !== 0 && this.first_reduce <= 2 && (sint_prod(this.sts[0].shape) as number) > 1, 'invalid upcast mid reduce')
      const axes = this.sts[0].unit_stride_axes()
      check(axes.length === 1, `wrong number of stride 1 axis : ${axes}`)
      check(axes[0] === axis, 'wrong axis')
      check(amt === 4, "don't upcast mid anything but 4")
      this.shift_to(axis, amt, undefined, this.first_reduce + this.group_for_reduces)
      this.group_for_reduces += 1
    } else if (opt.op === OptOps.NOLOCALS) {
      check(this.opts.has_local && !this.dont_use_locals, 'NOLOCALS===meaningless if target does not support local ||already not using locals')
      check(this.local_dims === 0 && this.group_for_reduces === 0, "can't have no locals with locals")
      this.dont_use_locals = true
    } else if (opt.op === OptOps.SWAP) {
      check(axis < amt && amt < this.global_dims, `swap===only for globals with axis < amt, getting amt=${amt}, axis=${axis}, this.global_dims=${this.global_dims}`)
      const permute = range(this.shape_len)
      ;[permute[axis], permute[amt]] = [permute[amt], permute[axis]]
      this.reshape_and_permute(undefined, permute)
    } else if (opt.op === OptOps.PADTO) {
      check(!this.vars, 'does not work with symbolic shape')
      check(axis < this.first_upcast, 'cannot pad upcasted')
      //       # ok to pad SUM if all parent ALU ops have f(0) = 0
      const r = this.reduceop
      if (r !== undefined && this.first_reduce <= axis) check(r.arg[0] === Ops.ADD && can_pad(r, new Map(), new Set([])), `cannot pad ${r}`)
      let padded = false
      for (const [i, st] of this.sts.entries()) {
        const s = st.shape[axis] as number
        if (s === 1) continue // reduced
        check(s > idiv(amt, 4), `pad adds more than quadruple the work ${st.shape[axis]} > ${idiv(amt, 4)}`)
        const ru = round_up(s, amt) - s
        if (ru) {
          //           # pad right seems to be faster
          this.sts[i] = st.pad([...range(axis).map(() => [0, 0] as [number, number]), [0, ru], ...range(st.shape.length - axis - 1).map((x) => [0, 0] as [number, number])])
          padded = true
        }
      }
      check(padded, 'nothing was padded')
    }

    if (append_opt) this.applied_opts.push(opt)
    if (this.simplify_ones() && this.tensor_core_opts) {
      this.tensor_core_opts.fix_axes(axis) // fix up axes in TC opts if required after simplify_ones()
    }
  }
  required_optimizations = (): Kernel => {
    if (isinstance(this.membufs[0].dtype, ImageDType)) {
      const unit_stride_axes_mul_4 = this.sts[0].unit_stride_axes(true).filter((i) => (this.sts[0].shape[i] as number) % 4 === 0)
      assert(!!unit_stride_axes_mul_4.length, `needs a unit stride axis in ${this.bufs[0]}`)
      if (unit_stride_axes_mul_4.every((x) => x < this.first_upcast) && !this.upcast_in_mid_reduce_axes.includes(unit_stride_axes_mul_4[0])) {
        this.apply_opt(new Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
      }
    }
    return this
  }
  get mulop() {
    return this.reduceop!.src[0]
  }
  hand_coded_optimizations = (): Kernel => {
    this.required_optimizations()

    //     # should use matvec - TODO: adjust/tune based on the wide vs tall/large vs small mat
    const [MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD] = [get_number_env('MV_BLOCKSIZE', 4), get_number_env('MV_THREADS_PER_ROW', 8), get_number_env('MV_ROWS_PER_THREAD', 4)]
    if (
      this.opts.has_local && get_number_env('MV', 1) !== 0 && (MV_BLOCKSIZE > 1 || MV_THREADS_PER_ROW > 1 || MV_ROWS_PER_THREAD > 1) && this.reduceop !== undefined && this.reduceop.arg[0] === Ops.ADD && this.full_shape.length >= 2 &&
      this.opts.has_shared && this.mulop.op === Ops.MUL && this.mulop.src[0].op === Ops.LOAD && this.mulop.src[1].op === Ops.LOAD
    ) {
      const [st0, st1] = [this.sts[this.bufs.indexOf(this.mulop.src[0])], this.sts[this.bufs.indexOf(this.mulop.src[1])]]
      const [strides0, strides1] = [st0.real_strides(), st1.real_strides()]
      const has_expanded_axis = (shape: sint[], strides: sint[]) => zip(shape, strides).some(([s, st]) => resolve(gt(s, 1)) && !resolve(ne(st, 0)))
      if (strides0[this.first_reduce] === 1 && !(has_expanded_axis(st0.shape, strides0 as sint[]) && has_expanded_axis(st1.shape, strides1 as sint[]))) {
        for (const global_idx of range(this.global_dims)) {
          if ((this.full_shape[this.first_reduce] as number) % MV_THREADS_PER_ROW === 0 && (this.full_shape[global_idx] as number) % (MV_BLOCKSIZE * MV_ROWS_PER_THREAD) === 0) {
            if (DEBUG >= 3) console.log(`MATVEC: ${{ full_shape: this.full_shape, first_reduce: this.first_reduce, strides0, MV_BLOCKSIZE, MV_THREADS_PER_ROW, MV_ROWS_PER_THREAD }}`)
            if (MV_THREADS_PER_ROW > 1) this.apply_opt(new Opt(OptOps.GROUP, 0, MV_THREADS_PER_ROW))
            if (MV_BLOCKSIZE > 1) this.apply_opt(new Opt(OptOps.LOCAL, global_idx, MV_BLOCKSIZE))
            if (MV_ROWS_PER_THREAD > 1) this.apply_opt(new Opt(OptOps.UPCAST, global_idx, MV_ROWS_PER_THREAD))
            return this
          }
        }
      }
    }
    if (this.opts.has_local && this.opts.has_shared && all_int(this.sts[0].shape.slice(0, this.first_reduce))) {
      //       # are we grouping? (requires local shape support)
      if (!this.float4_axis(0) && this.first_reduce <= 2 && this.first_reduce + 1 <= this.shape_len && sint_prod(this.sts[0].shape.slice(0, this.first_reduce)) as number <= 2048) {
        //         # TODO: use 1024 if it's allowed in a smarter way
        for (const sz of sint_prod(this.sts[0].shape.slice(0, this.first_reduce)) as number <= 32 ? [256, 16] : [16]) {
          if (this.sts.every((st) => st.shape[this.first_reduce] as number % sz === 0 || st.shape[this.first_reduce] === 1)) {
            try { // may fail due to excessive smem usage
              this.apply_opt(new Opt(OptOps.GROUPTOP, 0, sz))
              break
            } catch {
              // do nothing
            }
          }
        }
      }
      //       # are we upcasting in mid reduce? (only for images)
      if (this.bufs[0].src[0].dtype.name.startsWith('image') && !this.float4_axis(0) && this.group_for_reduces && this.first_reduce <= 2 && sint_prod(this.sts[0].shape) as number > 1) {
        const axes = this.sts[0].unit_stride_axes()
        assert(axes.length === 1, `wrong number of stride 1 axis : ${axes}`)
        if (this.sts[0].shape[axes[0]] as number % 4 === 0) {
          this.apply_opt(new Opt(OptOps.UPCASTMID, axes[0], 4))
        }
      }
    }
    //     # upcast float4 images
    for (const [buf_index, buf] of this.bufs.entries()) {
      const unit_stride_axes_mul_4 = this.sts[buf_index].unit_stride_axes(true).filter((i) => this.sts[buf_index].shape[i] as number % 4 === 0)
      if (isinstance(buf.src[0].dtype, ImageDType)) {
        //         #assert len(unit_stride_axes_mul_4) >= 1, f"needs a unit stride axis in {this.bufs[buf_index]}"
        if (unit_stride_axes_mul_4.length && unit_stride_axes_mul_4.every((x) => x < this.first_upcast) && !this.upcast_in_mid_reduce_axes.includes(unit_stride_axes_mul_4[0])) {
          if (unit_stride_axes_mul_4[0] < this.first_reduce) {
            this.apply_opt(new Opt(OptOps.UPCAST, unit_stride_axes_mul_4[0], 4))
          } else {
            this.apply_opt(new Opt(OptOps.UNROLL, unit_stride_axes_mul_4[0] - this.first_reduce, 4))
          }
        }
      }
    }
    //     # no more opt if we are grouping
    if (this.group_for_reduces) return this

    //     # **** below this line need to be optional && benchmarked ****

    //     # TODO: doing extra upcasts with images doesn't work for some reason (maybe has to do with to_image_idx)
    //     # to trigger the above bug, remove prod(this.full_shape[this.first_upcast:]) from the below
    //     # expression && run test/test_ops.py with IMAGE=2
    //     # if there are small dims with lots of valid masks, upcast them (they might be from Tensor.stack)
    //     # this can be made much smarter
    const to_upcast: number[] = []
    //     # upcast leading axes first (hack-ish for winograd; we actually want to upcast masked axes with low stride first)
    for (const axis of range(this.first_reduce)) {
      //       # we might want to be able to split axes that are masked, ||refuse to merge them in simplify_merge_adjacent
      //       # for now skip upcasting here if there===a symbolic axis
      if (
        isinstance(this.full_shape[axis], Number) && this.full_shape[axis] <= 7 && this.sts.every((st) => st.axis_is_masked(axis)) &&
        (sint_prod(this.full_shape.slice(this.first_upcast)) as number) * (sint_prod(to_upcast.map((j) => this.full_shape[j])) as number) * this.full_shape[axis] <= 7 * 7
      ) {
        if (DEBUG >= 4) console.log(`upcasting masked axis : ${axis}`)
        to_upcast.push(axis)
      }
    }
    for (const axis of to_upcast.toReversed()) {
      this.apply_opt(new Opt(OptOps.UPCAST, axis, 0))
    }

    //     # potentially do more upcasts of non reduce axes based on a heuristic
    const upcasted_axis = new Set<number>()
    while (resolve(ge(sint_prod(this.sts[0].shape.slice(0, this.first_reduce)), 1024))) {
      let xb_choices: [number, number, number, number][] = []

      for (const [axis, upcast_amount] of range(this.first_reduce).flatMap((a) => [3, 4].map((b) => [a, b]))) { // consider all the non reduce axes, && a 3 ||4 reduce
        //         # if we haven't upcasted it, it's not symbolic, it mods, && buffer has stride 0 on axis while having no stride 0 in the upcasted axis already
        if (
          !upcasted_axis.has(axis) && typeof this.full_shape[axis] === 'number' && this.full_shape[axis] % upcast_amount === 0 &&
          this.sts.entries().some(([buf_index, st]) => st.views.at(-1)!.strides[axis] === 0 && !this.upcasted_axis(buf_index).some((x) => x[1] === 0))
        ) {
          xb_choices.push([
            sum(this.sts.map((st) => Number(st.views.at(-1)!.strides[axis] as number > 0))),
            sum(this.sts.map((st) => st.views.at(-1)!.strides[axis] as number)),
            axis,
            upcast_amount,
          ])
        }
      }
      if (xb_choices.length) {
        xb_choices = xb_choices.toSorted()
        if (DEBUG >= 4) console.log(`float4 merging axis : {xb_choices}`)
        this.apply_opt(new Opt(OptOps.UPCAST, xb_choices[0][2], xb_choices[0][3]))
        upcasted_axis.add(xb_choices[0][2])
      } else break
    }
    //     # if last dim===small(ish) && it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast.
    if (
      this.first_reduce < this.first_upcast && (sint_prod(this.full_shape.slice(this.first_upcast)) as number <= 4 || !this.upcasted_axis(this.full_buf_index).some(([_, _1, r]) => r)) &&
      (this.upcasted === 0 || sint_prod(this.full_shape.slice(-this.upcasted)) as number < 64)
    ) {
      const s = this.full_unupcasted_shape.at(-1)!
      if (isinstance(s, Number) && s <= 32) { // NOTE: cannot loop unroll symbolic axis
        this.apply_opt(new Opt(OptOps.UNROLL, this.full_unupcasted_shape.length - 1 - this.first_reduce, 0))
        //         # if it's small, upcast a second reduce dimension too
        const s2 = this.full_unupcasted_shape.at(-1)!
        if (this.first_reduce < this.first_upcast && s <= 3 && isinstance(s2, Number) && s2 <= 3) {
          this.apply_opt(new Opt(OptOps.UNROLL, this.full_unupcasted_shape.length - 1 - this.first_reduce, 0))
        }
      } else {
        for (const splits of [4]) {
          if ((this.full_unupcasted_shape.at(-1)! as number) % splits === 0) {
            this.apply_opt(new Opt(OptOps.UNROLL, this.full_unupcasted_shape.length - 1 - this.first_reduce, splits))
            break
          }
        }
      }
    }
    //     # if nothing at all===upcasted && it's easy to, do an upcast
    //     # TODO: this===breaking the tests
    for (const splits of [4]) {
      if (this.upcasted === 0 && this.full_unupcasted_shape && (this.full_unupcasted_shape.at(-1)! as number) % splits === 0) {
        this.apply_opt(new Opt(OptOps.UPCAST, this.full_unupcasted_shape.length - 1, splits))
      }
    }
    //     # **** local groups ****

    if (this.opts.has_local) {
      if (get_env('NOLOCALS') && this.local_dims === 0 && !this.group_for_reduces) {
        this.apply_opt(new Opt(OptOps.NOLOCALS))
      } else {
        //         # prioritize making expand axes local
        const local_axis_ranking = [range(this.full_shape.slice(0, this.first_reduce).length).map((axis) => [range(this.sts.length).some((buf_index) => this.sts[buf_index].views.at(-1)!.strides[axis] === 0), axis] as [boolean, number])]
        const to_local: [number, number][] = []
        for (const [_, axis] of local_axis_ranking[0].toSorted((a, b) => (b[0] === a[0] ? b[1] - a[1] : Number(b[0]) - Number(a[0])))) {
          const local_size = sint_prod(to_local.map(([_, sz]) => sz))
          const local_sz: number | undefined = [...(axis === 0 ? [32] : []), 16, 8, 4, 3, 2].filter((x) => (this.full_shape[axis] as number) % x === 0 && (local_size as number) * x <= 128)[0]
          if (local_sz !== undefined) to_local.push([axis, local_sz])
        }
        let deleted_shape = 0
        for (let [axis, local_sz] of to_local.slice(0, 3).toSorted()) {
          axis = axis - deleted_shape
          const will_delete_shape = local_sz === this.full_shape[axis]
          this.apply_opt(new Opt(OptOps.LOCAL, axis, local_sz))
          if (will_delete_shape) deleted_shape += 1
        }
      }
    }
    return this
  }

  //   # **** kernel outputs ****

  static kernel_cnt: Record<string, number> = {}
  //   @functools.cached_property
  get name(): string {
    //     # kernel name (before late upcast)
    const kernel_type = this.reduceop !== undefined ? 'r' : ([...this.ast.toposort].every((x) => x.op === Ops.SINK || GroupOp.Buffer.includes(x.op)) ? 'C' : 'E')
    const suffix = zip(this.full_shape, this.colors()).map(([x, c]) => colored(isinstance(x, UOp) ? x.render() : x.toString(), c)).join(colored('_', 'BLACK'))
    const name = kernel_type + (this.ast.src.length > 1 ? `${this.ast.src.length}` : '') + '_' + suffix

    //     # name the function something unique
    const function_name = to_function_name(name)
    Kernel.kernel_cnt[function_name] = (Kernel.kernel_cnt[function_name] || 0) + 1
    const num = Kernel.kernel_cnt[function_name] > 1 ? `n${Kernel.kernel_cnt[function_name] - 1}` : ''
    return name + colored(num, 'BLACK')
  }
  get_optimized_ast = (): UOp => {
    //     @functools.lru_cache(undefined)
    const fixup_ast = (op: UOp): UOp => {
      let ret = op.replace({ src: op.src.map((x) => fixup_ast(x)) })
      if (GroupOp.Buffer.includes(op.op) && this.bufs.includes(op)) {
        const st_uop = this.sts[this.bufs.indexOf(op)].to_uop()
        return op.op === Ops.VALID ? ret.replace({ src: [st_uop] }) : ret.replace({ src: [ret.src[0], st_uop, ...ret.src.slice(2)] })
      }
      if (op.op === Ops.SINK) return ret.replace({ arg: new KernelInfo(this.local_dims, this.upcasted, this.dont_use_locals) })
      if (op.op === Ops.REDUCE_AXIS) {
        const reduce_idx = this.bufs.length + this.reduceops.indexOf(op) * 2

        const reduced_axes = (start: number, stop: number) => range(start, stop).filter((i) => resolve(ne(this.sts[reduce_idx].shape[i], this.sts[reduce_idx + 1].shape[i])))
        const axes = reduced_axes(this.first_reduce + this.group_for_reduces, this.shape_len)
        const grouped_axes = reduced_axes(this.first_reduce, this.first_reduce + this.group_for_reduces)
        //       # KAREL: not needed for mnist
        //         # if (tc := this.tensor_core) && (this.use_tensor_cores === 1 ||this.use_tensor_cores === 3):
        //         #   def fix_st(st: ShapeTracker, wd_pattern, tcd_pattern):
        //         #     st = ShapeTracker.from_shape(st.shape) # st needs to be contiguous
        //         #     wd, warp_dims = this.global_dims,  tuple(sz for _, sz in tc.threads)
        //         #     tcd, tcd_dims = this.first_upcast, tuple(sz for _, sz in tc.reduce_axes + tc.early_upcast_axes)

        //         #     assert st.shape[wd:wd+len(warp_dims)] === warp_dims, f"warp dims wrong: {st.shape[wd:wd+len(warp_dims)]=} !== {warp_dims=}"
        //         #     assert st.shape[tcd:tcd+len(tcd_dims)] === tcd_dims, f"tcd dims wrong: {st.shape[tcd:tcd+len(tcd_dims)]=} !== {tcd_dims=}"
        //         #     assert tc.expanded_shape!==undefined

        //         #     new_shape = st.shape[:tcd] + tc.expanded_shape + st.shape[tcd+len(tcd_dims):]  # expand the tcd
        //         #     permaxis = list(range(wd)) + [y + (wd if x === 0 else tcd) for x,y in wd_pattern]  + list(range(wd+len(warp_dims),tcd)) + \
        //         #                                  [y + (wd if x === 0 else tcd) for x,y in tcd_pattern] + list(range(tcd+len(tc.expanded_shape),len(new_shape)))
        //         #     return st.reshape(new_shape).permute(tuple(permaxis)).reshape(st.shape).simplify()

        //         #   srcs = list((ret.src[0] if ret.src[0].op!==Ops.CAST else ret.src[0].src[0]).src)
        //         #   for i, tc_pattern in enumerate([tc.st1_pattern, tc.st2_pattern]):
        //         #     if tc_pattern: srcs[i] = srcs[i].view(fix_st(srcs[i].st_arg if srcs[i].op===Ops.LOAD else srcs[i].src[0].st_arg, *tc_pattern))

        //         #     if this.use_tensor_cores === 3:  # for TC=3, emulate the warp addressing with locals
        //         #       local_shape = tuple(1 if i >= this.first_reduce && i < this.first_upcast else s for i, s in enumerate(this.full_shape))
        //         #       st = store_st = ShapeTracker.from_shape(local_shape)
        //         #       local_buffer = UOp(Ops.DEFINE_LOCAL, tc.dtype_in.ptr(local=true), (), (f"temp{i + 1}", st.real_size()))
        //         #       if tc_pattern: store_st = fix_st(store_st, *tc_pattern)
        //         #       local_store = UOp.store(local_buffer, store_st.to_uop(), srcs[i])
        //         #       srcs[i] = UOp(Ops.LOAD, tc.dtype_in, (local_buffer, st.to_uop(), local_store))

        //         #   tc_reduce_axes = tuple(this.first_upcast + ax for ax, _ in tc.reduce_axes)
        //         #   if this.use_tensor_cores === 1: # real WMMA, use CONTRACT/EXPAND to get the vectorization right
        //         #     upcast_axes = tuple(tuple((this.first_upcast + ax, sz) for ax, sz in up) for up in tc.upcast_axes)
        //         #     wmma_arg = (str(tc), tc.dims, tc.dtype_in, tc.dtype_out, this.opts.device, prod(sz for _, sz in tc.threads), upcast_axes, tc_reduce_axes)
        //         #     wmma_sz = [prod(x[1] for x in l) for l in upcast_axes]
        //         #     wmma = UOp(Ops.WMMA, dtype=tc.dtype_out.vec(wmma_sz[2]), src=(
        //         #       UOp(Ops.CONTRACT, dtype=srcs[0].dtype.vec(wmma_sz[0]), src=(srcs[0],), arg=upcast_axes[0]),
        //         #       UOp(Ops.CONTRACT, dtype=srcs[1].dtype.vec(wmma_sz[1]), src=(srcs[1],), arg=upcast_axes[1]),
        //         #       UOp.const(tc.dtype_out.vec(wmma_sz[2]), 0.0)), arg=wmma_arg)
        //         #     tc_uop = UOp(Ops.EXPAND, tc.dtype_out, (wmma,), arg=upcast_axes[2])

        //         #   else: # for TC=3 MUL/SUM instead of WMMA
        //         #     tc_uop = UOp(Ops.REDUCE_AXIS, tc.dtype_out, ((srcs[0] * srcs[1]).cast(tc.dtype_out),), (Ops.ADD, tc_reduce_axes))

        //         #   new_reduce_axes = tuple(i for i in axes if i not in tc_reduce_axes)
        //         #   return ret.replace(src=(tc_uop,), arg=(Ops.ADD, new_reduce_axes)) if new_reduce_axes else tc_uop

        ret = ret.replace({ arg: [op.arg[0], axes] })
        if (this.group_for_reduces && grouped_axes) {
          const local_shape = [
            ...range(this.global_dims).map(() => 1),
            ...this.full_shape.slice(this.global_dims, this.global_dims + this.local_dims),
            ...range(this.first_reduce, this.first_reduce + this.group_for_reduces).map((i) => this.sts[reduce_idx].shape[i] !== this.sts[reduce_idx + 1].shape[i] ? this.full_shape[i] : 1),
            ...range(this.shape_len - this.upcasted - this.group_for_reduces - this.first_reduce).map(() => 1),
            ...this.upcasted_axis(0).map((x) => x[0]),
          ]
          let st_uop = ShapeTracker.from_shape(local_shape).to_uop()
          const local_buffer = new UOp(Ops.DEFINE_LOCAL, op.dtype.ptr(true), [], [`temp${this.reduceops.indexOf(op) + 1}`, st_uop.arg.real_size()])
          const local_load = new UOp(Ops.LOAD, op.dtype, [local_buffer, st_uop, local_buffer.store([st_uop, ret])])
          const grouped_reduce = new UOp(Ops.REDUCE_AXIS, op.dtype, [local_load], [op.arg[0], grouped_axes])
          if (op === this.reduceops.at(-1)) return grouped_reduce
          st_uop = ShapeTracker.from_shape(local_shape.map((a, i) => grouped_axes.includes(i) ? 1 : a)).to_uop()
          return new UOp(Ops.LOAD, op.dtype, [local_buffer, st_uop, local_buffer.store([st_uop, grouped_reduce])])
        }
      }
      return ret
    }
    return graph_rewrite(fixup_ast(this.ast), view_left)
  }
  //   # **** this===the lowerer ****

  //   @track_rewrites()
  linearize = (): Kernel => {
    const modified_ast = this.get_optimized_ast()
    if (DEBUG >= 3) {
      console.log(this.name)
      if (get_env('RAWAST')) console.log(this.ast)
      console.log(modified_ast)
      console.log(this.applied_opts)
      verify_ast(modified_ast)
    }
    this.uops = linearize_uop(full_graph_rewrite(rewrite_shapetracker_with_index(modified_ast, this.opts), this.opts))
    if (DEBUG >= 5) print_uops(this.uops)
    return this
  }
  to_program = (name_override?: string): ProgramSpec => {
    this.linearize()
    const ansiname = name_override !== undefined ? name_override : this.name
    const name = to_function_name(ansiname)
    const src = this.opts.render(name, this.uops!)
    // KAREL: not needed for mnist
    //     if getenv("RUN_PROCESS_REPLAY"):
    //       from test.external.process_replay.helpers import get_process_replay_ctx
    //       diskcache_put("kernel_process_replay", str(id(this)), (this.ast, this.opts, this.applied_opts, name, *get_process_replay_ctx(), src))

    //     # group non-local bufs by the op type (LOAD ||STORE) && the buffer arg. take the max access of that buffer in bytes
    //     # TODO: these max && min don't work on symbolic, && results are very wrong.
    const groups = new Map<[Ops, any], UOp[]>()
    for (const x of [...this.ast.toposort].filter((x) => GroupOp.Buffer.includes(x.op) && x.src[0].op === Ops.DEFINE_GLOBAL)) setDefault(groups, [x.op, x.src[0].arg], []).push(x)
    const mem_bytes = [...groups.values()].flatMap((group) => Math.max(...group.map((x) => x.src[0].dtype.itemsize * x.st_arg.real_size()))).reduce((acc, x) => acc + x, 0)
    return new ProgramSpec(ansiname, src, this.opts.device, this.uops, mem_bytes, this.opts.has_local ? [1, 1, 1] : undefined, this.opts.has_local ? [1, 1, 1] : undefined)
  }
}
// # the living definition of intermediate UOps

export const _assert_valid_uop = (uop: UOp, st: ShapeTracker, sts: Map<UOp, ShapeTracker>): undefined => {
  if (!uop.has_st || sts.has(uop)) return
  // restore globals from the two stage reduce
  if (uop.op === Ops.LOAD && uop.src[0].op === Ops.DEFINE_LOCAL) {
    const local_reduce = uop.src[2].src[2]
    _assert_valid_uop(local_reduce, uop.st_arg, sts)
    sts.set(uop, sts.get(local_reduce)!)
    return
  }
  for (const x of uop.src) _assert_valid_uop(x, st, sts)
  // only reduceuop===allowed to change shape, limited to turning n to 1
  if ([Ops.REDUCE_AXIS, Ops.WMMA].includes(uop.op)) st = ShapeTracker.from_shape(sts.get(uop.src[0])!.reduce(uop.axis_arg))
  // movementops are pushed to VIEW
  else if (uop.op === Ops.VIEW) {
    assert(uop.src.length === 0, `can't swizzle in kernel yet ${uop}`)
    st = uop.arg
  } // everything else inherits shape
  else {
    const src_sts = uop.src.filter((x) => x.has_st).map((x) => sts.get(x)!)
    st = src_sts[0]
    const shapes = src_sts.map((x) => x.shape)
    if (!all_same(shapes)) {
      const sizes = shapes.map((x) => sint_prod(x))
      if (all_same(sizes)) throw new Error(`found implicit reshape ${shapes}`)
      throw new Error(`found implicit expand ${sizes} ${shapes}`)
    }
  }
  sts.set(uop, st)
}
export const verify_ast = (ast: UOp): Map<UOp, ShapeTracker> => {
  assert(ast.op === Ops.SINK && ast.src.every((x) => x.op === Ops.STORE), 'must be SINK')
  assert(all_same(ast.src.map((x) => x.st_arg.size)), 'outputs must be exactly the same size')
  const sts = new Map<UOp, ShapeTracker>()
  for (const out of ast.src) _assert_valid_uop(out, out.st_arg, sts)
  const shape_dims = zip(...sts.values().map((x) => x.shape)).map((dims) => dedup(dims).toSorted())
  assert(shape_dims.every((x) => x.length === 1 || (x.length === 2 && x[0] === 1)), `shapes must have either 1 ||n in each dimension, ${shape_dims}`)
  return sts
}
