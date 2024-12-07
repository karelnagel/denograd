import type { DType } from '../dtype.ts'
import { raise, toFunctionName } from '../helpers.ts'
import { isNone } from '../helpers.ts'
import { assert, isNotNone, prod, range } from '../helpers.ts'
import { flopsMem, Ops, type sint, symInfer, type UOp, type Variable } from '../ops.ts'

export type TC = [number, number]
export class TensorCore { // D = A * B + C, A is (M x K), B is (K x N), C and D are (M x N)
  dims: [number, number, number] // N, M, K
  dtypeIn: DType // dtype for A and B
  dtypeOut: DType // dtype for C and D
  threads: TC[] // list of (TC dim,amt) that construct the warp thread structure
  reduceAxes: TC[] // list of (TC dim,amt) that constructs the shape of the reduce dim

  // deno-fmt-ignore
  constructor(p:{dims: [number, number, number], dtypeIn: DType, dtypeOut: DType, threads: TC[], reduceAxes: TC[], upcastAxes:[TC[],TC[],TC[]]}) {
    this.dims=p.dims; this.dtypeIn=p.dtypeIn; this.dtypeOut=p.dtypeOut; this.threads=p.threads; this.reduceAxes=p.reduceAxes; this.upcastAxes = p.upcastAxes
  }
  earlyUpcastAxes = (): TC[] => { // list of (TC dim,amt) that upcasts the threads remainders of dims [0,1]
    return range(2).map((dim) => [dim, prod(this.threads.filter(([d]) => d === dim).map(([d, sz]) => sz))]).filter(([d, sz]) => this.dims[d] > sz).map(([d, sz]) => [d, Math.floor(this.dims[d] / sz)])
  }
  upcastAxes: [TC[], TC[], TC[]] // list of (TC dim,amt) that upcast A, B and C
  st1Pattern?: TC[][] | TC[] // pattern to fix shapetracker for A
  st2Pattern?: TC[][] | TC[] // pattern to fix shapetracker for B
  expandedShape?: number[]
  optsSeq: [string, string] = ['UP', 'LC'] // upcast input, local the thread pattern
  toString = () => ['WMMA', ...this.dims.map((d) => d.toString()), this.dtypeIn.name, this.dtypeOut.name].join('_')
}

export class ProgramSpec {
  name: string
  src: string
  device: string
  uops?: UOp[]
  memEstimate: sint = 0 // TODO: get this from the load/store uops once min/max are good
  // deno-fmt-ignore
  constructor(name: string, src: string, device: string, uops?: UOp[], memEstimate: sint = 0) {
    this.name = name; this.src = src; this.device = device; this.uops = uops; this.memEstimate = memEstimate
  }

  // filled in from uops (if we have uops)
  globalSize?: number[]
  localSize?: number[]
  vars?: Variable[] = []
  globals: number[] = []
  outs: number[] = []
  _ranPostInit = false // NOTE: this is needed if you call replace on the Program

  __postInit__ = () => {
    if (!this._ranPostInit && isNotNone(this.uops)) {
      // single pass through the uops
      for (const u of this.uops) {
        if (u.op === Ops.DEFINE_VAR) this.vars?.push(u)
        if (u.op === Ops.DEFINE_GLOBAL) this.globals.push(u.arg)
        if (u.op === Ops.STORE) this.outs = [...this.outs, ...[...u.src[0].toposort].filter((x) => x.op === Ops.DEFINE_GLOBAL).map((x) => x.arg)]
        if (u.op === Ops.SPECIAL) {
          // NOTE: you have to set local_size and global_size to the base [1,1,1] outside this
          if (u.arg[0][0] === 'i') this.localSize = undefined
          const specialSize = (u.arg[0][0] === 'l' ? this.localSize : this.globalSize) || []
          assert(isNotNone(specialSize))
          specialSize[Number(u.arg[0].at(-1)!)] = u.arg[1]
        }
      }
      this.vars = this.vars?.toSorted((a, b) => b.arg - a.arg)
      this.outs = [...new Set(this.outs)].toSorted()
      this._ranPostInit = true
    }
  }
  opEstimate = () => this._opsLds()[0]
  ldsEstimate = () => this._opsLds()[1]
  _opsLds = (): [sint, sint] => isNone(this.uops) ? [0, 0] : flopsMem(this.uops, true)

  functionName = () => toFunctionName(this.name)

  launchDims = (varVals: Map<Variable, number>) => {
    const globalSize = this.globalSize?.map((sz) => symInfer(sz, varVals))
    const localSize = this.localSize?.map((sz) => symInfer(sz, varVals))
    return [globalSize, localSize]
  }
}

export class Renderer {
  device = ''
  suffix = ''
  // TODO: make this generic with a list of supported types
  supports_float_4 = true
  has_local = true
  has_hared = true
  // NOTE: these two should be in (x,y,z) order to match the max_sizes argument in get_grouped_dims
  global_max?: [number, number, number] = [0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF] // TODO: UOps.SPECIAL int32 indexes right now
  local_max = [0x8FFFFFFF, 0x8FFFFFFF, 0x8FFFFFFF] // TODO: UOps.SPECIAL int32 indexes right now
  shared_max = 32768
  tensor_cores: TensorCore[] | undefined = []
  extra_matcher?: any
  code_for_op: { [key in Ops]?: (...a: string[]) => string } = {}

  render = (name: string, uops: UOp[]): string => raise('needs a renderer')
}
