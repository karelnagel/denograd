import { DeviceType } from '../device.ts'
import type { DType } from '../dtype.ts'
import { assert, dataclass, isNone, isNotNone, raise, range, to_function_name } from '../helpers.ts'
import { flops_mem, idiv, Ops, prod, type sint, sym_infer, type UOp, type Variable } from '../ops.ts'

export type TC = [number, number]

@dataclass
export class TensorCore { // D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  constructor(
    public dims: [number, number, number],
    public dtype_in: DType,
    public dtype_out: DType,
    public threads: TC[],
    public reduce_axes: TC[],
    public upcast_axes: [TC[], TC[], TC[]],
  ) {}
  early_upcast_axes = (): TC[] => { // list of (TC dim,amt) that upcasts the threads remainders of dims [0,1]
    return range(2).map((dim) => [dim, prod(this.threads.filter(([d]) => d === dim).map(([d, sz]) => sz))]).filter(([d, sz]) => this.dims[d] > sz).map(([d, sz]) => [d, idiv(this.dims[d], sz)])
  }
  st1_pattern?: TC[][] | TC[] // pattern to fix shapetracker for A
  st2_pattern?: TC[][] | TC[] // pattern to fix shapetracker for B
  expanded_shape?: number[]
  opts_seq: [string, string] = ['UP', 'LC'] // upcast input, local the thread pattern
  toString = () => ['WMMA', ...this.dims.map((d) => d.toString()), this.dtype_in.name, this.dtype_out.name].join('_')
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
    public _ran_post_init = false, // NOTE: this is needed if you call replace on the Program
  ) {
    if (!this._ran_post_init && isNotNone(this.uops)) {
      // single pass through the uops
      for (const u of this.uops) {
        if (u.op === Ops.DEFINE_VAR) this.vars?.push(u)
        if (u.op === Ops.DEFINE_GLOBAL) this.globals.push(u.arg)
        if (u.op === Ops.STORE) this.outs = [...this.outs, ...[...u.src[0].toposort].filter((x) => x.op === Ops.DEFINE_GLOBAL).map((x) => x.arg)]
        if (u.op === Ops.SPECIAL) {
          // NOTE: you have to set local_size and global_size to the base [1,1,1] outside this
          if (u.arg[0][0] === 'i') this.local_size = undefined
          const specialSize = (u.arg[0][0] === 'l' ? this.local_size : this.global_size) || []
          assert(isNotNone(specialSize))
          specialSize[Number(u.arg[0].at(-1)!)] = u.arg[1]
        }
      }
      this.vars = this.vars?.toSorted((a, b) => b.arg - a.arg)
      this.outs = [...new Set(this.outs)].toSorted()
      this._ran_post_init = true
    }
  }
  get op_estimate() {
    return this._ops_lds()[0]
  }
  get lds_estimate() {
    return this._ops_lds()[1]
  }
  _ops_lds = (): [sint, sint] => isNone(this.uops) ? [0, 0] : flops_mem(this.uops, true)

  get function_name() {
    return to_function_name(this.name)
  }

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

  render = (name: string, uops: UOp[]): string => raise('needs a renderer')
}
