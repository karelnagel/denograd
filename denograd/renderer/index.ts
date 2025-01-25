import type { DeviceType } from '../device.ts'
import type { DType } from '../dtype.ts'
import { add, assert, cache, dedup, get_key, idiv, mul, prod, range, replace, WeakValueMap } from '../helpers.ts'
import { GroupOp, Ops, type sint, ssimplify, sym_infer, type UOp, type Variable } from '../ops.ts'

export type TC = [number, number]

type TensorCoreArgs = {
  dims: [number, number, number]
  threads: number
  elements_per_thread: [number, number, number]
  dtype_in: DType
  dtype_out: DType
  opts: string[]
  swizzle: [[number[], number[]] | undefined, [number[], number[]] | undefined]
}
export class TensorCore { // D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  key: string
  static cache = new WeakValueMap<string,TensorCore>()

  dims: [number, number, number]
  threads: number
  elements_per_thread: [number, number, number]
  dtype_in: DType
  dtype_out: DType
  opts: string[]
  swizzle: [[number[], number[]] | undefined, [number[], number[]] | undefined] = [undefined, undefined]
  constructor(args: TensorCoreArgs) {
    this.dims = args.dims, this.threads = args.threads, this.elements_per_thread = args.elements_per_thread, this.dtype_in = args.dtype_in, this.dtype_out = args.dtype_out, this.opts = args.opts, this.swizzle = args.swizzle
    this.key = get_key(args)
    if (TensorCore.cache.has(this.key)) return TensorCore.cache.get(this.key)!
    const local_axes = this.get_local_axes().length, upcast_axes = this.get_upcast_axes().length, reduce_axes = this.get_reduce_axes().length
    if (this.dims[0] * this.dims[1] !== 2 ** (local_axes + upcast_axes)) throw new Error(`N(${this.dims[0]}) x M(${this.dims[1]}) != local(${2 ** local_axes}) x upcast(${2 ** upcast_axes}) with opts(${this.opts})`)
    if (2 ** local_axes !== this.threads) throw new Error(`${this.threads} threads construct the warp but found ${2 ** local_axes} in ${this.opts}`)
    if (2 ** upcast_axes !== this.elements_per_thread[2]) throw new Error(`${this.elements_per_thread[2]} elements from C are processed per thread but found ${2 ** upcast_axes} in ${this.opts}`)
    if (!this.swizzle.filter((perm) => perm?.length).every((perm) => perm![0].length === local_axes && perm![1].length === reduce_axes + upcast_axes)) throw new Error(`swizzle perm should be of len ((${local_axes})(${reduce_axes + upcast_axes}))`)
  }
  get_reduce_axes = () => range(Math.trunc(Math.log2(this.dims[2]))).map((i) => [i, 2])
  get_upcast_axes = () => this.opts.filter((opt) => opt[0] === 'u')
  get_local_axes = () => this.opts.filter((opt) => opt[0] === 'l')
  toString = () => ['WMMA', ...this.dims.map((d) => d.toString()), this.dtype_in.name, this.dtype_out.name].join('_')
}

export class Estimates {
  constructor(
    // number of FLOPS used in the Kernel
    public ops: sint = 0,
    // bytes accessed in loads and stores
    public lds: sint = 0,
    // total bytes accessed, counting only once for bytes that are accessed multiple times
    public mem: sint = 0,
  ) {}
  add = (o: Estimates) => new Estimates(add(this.ops, o.ops), add(this.lds, o.lds), add(this.mem, o.mem))
  simplify = () => new Estimates(ssimplify(this.ops as UOp), ssimplify(this.lds as UOp), ssimplify(this.mem as UOp))
  static from_uops = (uops: UOp[], ignore_indexing = false): Estimates => {
    let flops: sint = 0, lds: sint = 0, mults: sint = 1, mult_stack: sint[] = [], dont_count = new Set<UOp>()
    if (ignore_indexing) {
      for (const u of uops) {
        if ([Ops.LOAD, Ops.STORE].includes(u.op)) {
          dont_count = dont_count.union(u.src[0].toposort)
          if (u.src.length > 2) dont_count = dont_count.union(u.src[2].toposort)
        } else if (u.op === Ops.IF) dont_count = dont_count.union(u.src[0].toposort)
      }
    }
    for (const u of uops) {
      if (u.op === Ops.RANGE) {
        mult_stack.push(mults)
        mults = mul(mults, (u.src[1].sub(u.src[0])).ssimplify())
      } else if (u.op === Ops.ENDRANGE) mults = mult_stack.pop()!
      else if (u.op === Ops.SPECIAL) mults = mul(mults, u.arg[1]) // NOTE: we don't push to the mult_stack here, you can't end these
      else if (u.op === Ops.LOAD) lds = add(lds, mul(u.dtype.itemsize, mults))
      else if (u.op === Ops.STORE) lds = add(lds, mul(u.src[1].dtype.itemsize, mults))
      else if (GroupOp.ALU.includes(u.op) && !dont_count.has(u)) flops = add(flops, mul(mul(mults, u.op === Ops.MULACC ? 2 : 1), u.dtype.count))
      else if (u.op === Ops.WMMA && !dont_count.has(u)) flops = add(flops, mul(idiv(mul(2, prod(u.arg[1])), u.arg[5]), mults))
    }
    return new Estimates(flops, lds, lds) // TODO: properly track memory, lds is always a high estimate
  }
}

export class ProgramSpec {
  constructor(
    public name: string,
    public src: string,
    public device: DeviceType,
    public uops?: UOp[],
    public mem_estimate = 0,
    public global_size?: number[],
    public local_size?: number[],
    public vars: Variable[] = [],
    public globals: number[] = [],
    public outs: number[] = [],
    public ins: number[] = [],
    public _ran_post_init = false, // NOTE: this is needed if you call replace on the Program
  ) {
    if (!this._ran_post_init && this.uops !== undefined) {
      // single pass through the uops
      for (const u of this.uops) {
        if (u.op === Ops.DEFINE_VAR) this.vars?.push(u)
        if (u.op === Ops.DEFINE_GLOBAL) this.globals.push(u.arg)
        if (u.op === Ops.STORE) this.outs = [...this.outs, ...[...u.src[0].toposort].filter((x) => x.op === Ops.DEFINE_GLOBAL).map((x) => x.arg)]
        if (u.op === Ops.LOAD) this.ins = [...this.ins, ...[...u.src[0].toposort].filter((x) => x.op === Ops.DEFINE_GLOBAL).map((x) => x.arg)]
        if (u.op === Ops.SPECIAL) {
          // NOTE: you have to set local_size and global_size to the base [1,1,1] outside this
          if (u.arg[0][0] === 'i') this.local_size = undefined
          const specialSize = (u.arg[0][0] === 'l' ? this.local_size : this.global_size) || []
          assert(specialSize !== undefined)
          specialSize[Number(u.arg[0].at(-1)!)] = u.arg[1]
        }
      }
      this.vars = this.vars?.toSorted((a, b) => b.arg - a.arg)
      this.outs = dedup(this.outs).toSorted()
      this.ins = dedup(this.ins).toSorted()
      this._ran_post_init = true
    }
  }
  @cache
  get estimates() {
    return replace(this.uops === undefined ? new Estimates() : Estimates.from_uops(this.uops, true), { mem: this.mem_estimate })
  }

  // KAREL: TODO: for some reason gives invalid out
  // @cache
  // get function_name() {
  //   return to_function_name(this.name)
  // }

  launch_dims = (varVals: Map<Variable, number>) => {
    const globalSize = this.global_size?.map((sz) => sym_infer(sz, varVals))
    const localSize = this.local_size?.map((sz) => sym_infer(sz, varVals))
    return [globalSize, localSize]
  }
}

export class Renderer {
  device!: DeviceType
  suffix = ''
  // TODO: make this generic with a list of supported types
  supports_float4 = true
  has_local = true
  has_shared = true
  // NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max?: [number, number, number] = [0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF] // TODO: UOps.SPECIAL int32 indexes right now
  local_max = [0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF] // TODO: UOps.SPECIAL int32 indexes right now
  shared_max = 32768
  tensor_cores: TensorCore[] | undefined = []
  extra_matcher?: any
  code_for_op = new Map<Ops, (...a: string[]) => string>()

  render = (name: string, uops: UOp[]): string => {
    throw new Error('needs a renderer')
  }
}
