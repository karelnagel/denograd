// deno-lint-ignore-file require-await
import { all_same, assert, bytes_to_string, cpu_time_execution, flatten, get_env, isinstance, product, range, string_to_bytes, zip } from '../helpers.ts'
import { exec_alu, GroupOp, idiv, Ops, sum, UOp } from '../ops.ts'
import { Renderer } from '../renderer/index.ts'
import { Allocator, BufferSpec, Compiled, Compiler, Program } from './allocator.ts'
import { bitcast, DType, dtypes, ImageDType, PtrDType, truncate } from '../dtype.ts'
import type { DeviceType, ProgramCallInput } from '../device.ts'
import { MemoryView } from '../memoryview.ts'

const _load = (m: MemoryView, i?: number) => {
  if (i === undefined) return 0.0
  if (i < 0 || i >= m.length) throw new Error(`load out of bounds, size === ${m.length} && access === ${i}`)
  return m.getValue(i)
}
type Inp = [[MemoryView, number][], number[][]] | [[MemoryView, number][], number[][], boolean[]]
const load = (inp: Inp, j = 0) => {
  if (inp.length === 3) return zip(...inp).map(([[m, x], def, gate]) => gate ? _load(m, x !== undefined ? x + j : undefined) : def)
  return inp[0].map(([m, x]) => _load(m, x !== undefined ? x + j : undefined))
}
const _store = (m: MemoryView, i: number, v: any) => {
  if (i < 0 || i >= m.length) throw new Error(`store out of bounds, size === ${m.length}, access === ${i}, value === ${v}`)
  m.setValue(v, i)
}

type PyUOp = [Ops, DType | undefined, number[], any]
const jsonReplace = (key: string, value: unknown) => {
  if (Array.isArray(value)) return value
  if (value === undefined) return 'undefined'
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') return value
  if (typeof value === 'number') {
    if (Number.isNaN(value)) return '__NaN__'
    if (value === Infinity) return '__Infinity__'
    if (value === -Infinity) return '__-Infinity__'
    return value
  }
  if (value instanceof Ops) return value.toString()
  if (value instanceof DType) return value.toString()
  throw new Error(`Can't serialize ${value}`)
}
const jsonRevive = (key: string, value: unknown) => {
  if (Array.isArray(value)) return value
  if (typeof value === 'number') return value
  if (typeof value === 'boolean') return value
  if (typeof value === 'string') {
    if (value === '__NaN__') return NaN
    if (value === '__Infinity__') return Infinity
    if (value === '__-Infinity__') return -Infinity
    if (value === 'undefined') return undefined
    if (value.startsWith('Ops.')) return Ops.values().find((o) => o.toString() === value)
    if (value.startsWith('dtypes.')) return eval(value)
    return value
  }
  throw new Error(`Can't deserialize ${value}`)
}

const serialize = (data: PyUOp[]): Uint8Array => string_to_bytes(JSON.stringify(data, jsonReplace))
const deserialize = (data: Uint8Array): PyUOp[] => JSON.parse(bytes_to_string(data), jsonRevive)

export class PythonProgram extends Program {
  uops: PyUOp[]
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    this.uops = deserialize(lib)
  }
  // KAREL: TODO: use Web workers maybe?
  override call = cpu_time_execution(async (bufs: MemoryView[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallInput, wait = false) => {
    const warp = product(...local_size.toReversed().map((x) => range(x)))
    const warp_size = warp.length
    for (const idxs of product(...global_size.toReversed().map((x) => range(x)))) {
      const ul: Record<number, any[]> = {}
      const dl: Record<number, DType> = {}
      const pbufs = [...bufs]
      const pvals = [...vals]
      let i = 0
      const loop_ends: Record<number, number> = {}
      while (i < this.uops.length) {
        let [uop, dtype, idp, arg] = this.uops[i]
        const void_ops = [Ops.STORE, Ops.ENDRANGE, Ops.BARRIER, Ops.IF, Ops.ENDIF]
        if (uop === Ops.DEFINE_ACC) idp = [idp[0]]
        const inp = idp.filter((v) => !void_ops.includes(this.uops[v][0])).map((v) => ul[v])
        const dtp = idp.filter((v) => !void_ops.includes(this.uops[v][0])).map((v) => dl[v])
        if (get_env('TRACE')) console.log(i, uop, dtype, arg, inp, dtp)
        if (uop === Ops.STORE) {
          if (inp.length === 2) inp.push(range(inp[0].length).map(() => true)) // set the gate to true
          if (dtp[1]!.count > 1) {
            for (const [j, val] of inp[1].entries()) {
              for (const [[m, o], v, g] of zip(inp[0], val, inp[2])) {
                if (g) _store(m, o + j, v)
              }
            }
          } else {
            for (const [[m, o], v, g] of zip(...inp)) {
              if (g) _store(m, o, v)
            }
          }
          i += 1
          continue
        }
        if (uop === Ops.ENDRANGE) {
          loop_ends[idp[0]] = i
          i = idp[0]
          continue
        }
        if ([Ops.BARRIER, Ops.IF, Ops.ENDIF].includes(uop)) {
          // in the python emulator, the warp === always in sync
          i += 1
          continue
        }
        if (dtype === undefined) throw new Error(`${uop} === missing a dtype`)
        dl[i] = dtype
        if ([Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL].includes(uop)) {
          assert(dtype.fmt !== undefined)
          //   if (TYPE_CHECKING) assert(dtype.fmt !== "e")
          const buf = uop === Ops.DEFINE_LOCAL ? new MemoryView(new Uint8Array(arg[1] * dtype.itemsize)) : pbufs.shift()!
          ul[i] = range(warp_size).map(() => buf.cast(dtype.fmt!))
        } else if (uop === Ops.DEFINE_VAR) {
          ul[i] = range(warp_size).map(() => pvals.shift())
        } else if (uop === Ops.SPECIAL) {
          if (arg[0][0] === 'g') ul[i] = range(warp_size).map(() => idxs[2 - Number(arg[0].at(-1)!)])
          else if (arg[0][0] === 'l') ul[i] = warp.map((x) => x[2 - Number(arg[0].at(-1)!)])
        } else if (uop === Ops.CONST) {
          ul[i] = range(warp_size).map(() => arg)
        } else if (uop === Ops.DEFINE_ACC) {
          ul[i] = dtype.count > 1 ? range(dtype.count).flatMap((_) => range(warp_size).map(() => inp[0][0][0])) : range(warp_size).map(() => inp[0][0])
        } else if (uop === Ops.INDEX) {
          const ret = []
          if (isinstance(dtp[0], ImageDType)) {
            for (const [m, ox, oy] of zip<number[]>(inp[0], inp[1][0], inp[1][1])) {
              if (ox < 0 || ox >= dtp[0].shape[1] || oy < 0 || oy >= dtp[0].shape[0]) ret.push([m, undefined])
              else ret.push([m, ox * 4 + oy * dtp[0].shape[1] * 4])
            }
          } else {
            for (const [m, o] of zip(inp[0], inp[1])) ret.push([m, o])
          }
          ul[i] = ret
        } else if (uop === Ops.CAST && isinstance(dtype, PtrDType)) {
          ul[i] = inp[0]
        } else if (uop === Ops.RANGE) {
          if (ul[i] === undefined) ul[i] = range(warp_size).map(() => inp[0][0])
          else {
            for (const j of range(ul[i].length)) ul[i][j] += 1
            if (ul[i][0] === inp[1][0]) {
              delete ul[i]
              i = loop_ends[i] + 1
              continue
            }
          }
        } else if (uop === Ops.VECTORIZE) {
          ul[i] = inp
        } else if ([Ops.CAST, Ops.BITCAST].includes(uop)) {
          assert(!!dtp[0].fmt && !!dtype.fmt)
          if (uop === Ops.BITCAST) {
            ul[i] = bitcast(inp[0], dtp[0].fmt!, dtype.fmt!)
          } else ul[i] = inp[0].map((x) => (truncate.get(dtype) || ((dt: any) => dt))(dtypes.as_const(x, dtype)))
        } else if (uop === Ops.LOAD) {
          if (dtype.count > 1) {
            ul[i] = range(dtype.count).map((j) => load(range(inp.length).map((i) => i !== 0 && dtp[i].count > 1 ? inp[i][j] : inp[i]) as any, j))
          } else ul[i] = load(inp as any)
        } else if (uop === Ops.ASSIGN) {
          for (const j of range(inp[0].length)) inp[0][j] = inp[1][j]
          ul[i] = inp[0]
        } else if (uop === Ops.GEP) {
          assert(arg.length === 1)
          ul[i] = inp[0][arg[0]]
        } else if (uop === Ops.WMMA) {
          // here are the models for the WMMA instruction on the different hardware
          type Fn = (...a: [any[], number, number, number]) => number
          type Fn2 = (a: number, b: number) => [number, number]
          const wmma_helper = (WARP_THREADS: number, K: number, NUM_A: number, NUM_B: number, NUM_C: number, a_elem: Fn, b_elem: Fn, c_map: Fn2) => {
            for (const [cc, tinp, num] of zip(['A', 'B', 'C'], inp, [NUM_A, NUM_B, NUM_C])) {
              if (tinp.length !== num) throw new Error(`${cc} must have ${num} elements per thread, it has ${tinp.length}`)
              if (flatten(tinp).length !== num * warp_size) throw new Error(`WMMA must have ${num * warp_size} total elements for ${cc} in WMMA`)
            }
            if (warp_size <= 0 || warp_size % WARP_THREADS !== 0) throw new Error(`must have multiples of ${WARP_THREADS} warp threads`)
            const out = range(NUM_C).map((elem_idx) => [...inp[2][elem_idx]])
            for (const goff of range(0, warp_size, WARP_THREADS)) {
              for (const lane_id of range(WARP_THREADS)) {
                for (const elem_idx of range(NUM_C)) { // calculate new muls && add to acc
                  const [c_i, c_j] = c_map(lane_id, elem_idx)
                  out[elem_idx][goff + lane_id] += sum(range(K).map((_k) => a_elem(inp[0], _k, c_j, goff) * b_elem(inp[1], c_i, _k, goff)))
                }
              }
            }
            return out
          }

          //   // TODO: refactor these to a shared TensorCoreLayout in kernel.py
          if (arg[4] === 'METAL') {
            // A (2 elements on 32 threads): row major
            const a_b_elem: Fn = (x, i, j, goff) => x[i % 2][goff + idiv(i, 2) % 2 + (j % 4) * 2 + idiv(i, 4) * 8 + idiv(j, 4) * 16]
            // (i, j), C, D (2 elements on 32 threads): row major same as A/B
            const c_map: Fn2 = (lane, elem) => [elem + ((lane % 2) * 2) + (idiv(lane, 8) % 2) * 4, (idiv(lane, 2) % 4) + idiv(lane, 16) * 4]
            ul[i] = wmma_helper(32, 8, 2, 2, 2, a_b_elem, a_b_elem, c_map)
          } else if (arg[4] === 'AMD') {
            // A (16 elements on 32 threads): col major, lane 16-32 === lane 0-15
            const a_elem: Fn = (x, i, j, goff) => {
              if (x[i][goff + j] !== x[i][goff + j + 16]) throw new Error('warp elements !duplicated properly across lanes')
              return x[i][goff + j]
            }
            // B (16 elements on 32 threads): row major, lane 16-32 === lane 0-15
            const b_elem: Fn = (x, i, j, goff) => a_elem(x, j, i, goff)
            const c_map: Fn2 = (lane, elem) => [lane % 16, idiv(lane, 16) + elem * 2] // (i, j), C, D (8 elements on 32 threads): row major
            ul[i] = wmma_helper(32, 16, 16, 16, 8, a_elem, b_elem, c_map)
          } else if (arg[4] === 'CUDA') {
            // A (8 elements on 32 threads)
            const a_elem: Fn = (x, i, j, goff) => x[(i % 2) + idiv(j, 8) * 2 + idiv(i, 8) * 4][goff + (idiv(i, 2) % 4) + (j % 8) * 4]
            // B (4 elements on 32 threads)
            const b_elem: Fn = (x, i, j, goff) => x[(j % 2) + idiv(j, 8) * 2][goff + idiv(j, 2) % 4 + i * 4]
            // (i, j), C, D (4 elements on 32 threads)
            const c_map: Fn2 = (lane, elem) => [(elem % 2) + (lane % 4) * 2, idiv(lane, 4) + idiv(elem, 2) * 8]
            ul[i] = wmma_helper(32, 16, 8, 4, 4, a_elem, b_elem, c_map)
          } else if (arg[4] === 'INTEL') {
            // A (16 elements on 8 threads)
            const a_elem: Fn = (x, i, j, goff) => x[i % 2 + j * 2][goff + idiv(i, 2)]
            // B (16 elements on 8 threads)
            const b_elem: Fn = (x, i, j, goff) => x[j][goff + i]
            // C, D (8 elements on 8 threads)
            const c_map: Fn2 = (lane, elem) => [lane, elem]
            ul[i] = wmma_helper(8, 16, 16, 16, 8, a_elem, b_elem, c_map)
          } else if (arg[4] === 'CLANG') {
            const elem: Fn = (x, i, j, _) => x[i + j][0]
            const c_map: Fn2 = (_, elem) => [elem % 16, idiv(elem, 16)]
            ul[i] = wmma_helper(1, 1, 16, 16, 256, elem, elem, c_map)
          } else throw new Error(`unimplemented tensor core ${arg}`)
        } else if (GroupOp.ALU.includes(uop)) {
          if (!all_same(inp.map((x) => x.length))) throw new Error(`${inp.map((x) => x.length)} doesn't match on ${uop}`)
          if (!all_same([dtype, ...dtp]) && ![Ops.CMPNE, Ops.CMPLT, Ops.WHERE].includes(uop)) throw new Error(`dtype mismatch on ${uop}`)
          ul[i] = zip(...inp).map((p) => exec_alu(uop, dtype, p))
        }
        if (!ul[i]) throw new Error(`${uop}, ${dtype}, ${idp}, ${arg}`)
        i += 1
      }
    }
  })
}

export class PythonRenderer extends Renderer {
  override device: DeviceType = 'PYTHON'
  constructor() {
    //     // if getenv("EMULATE_METAL"): this.device, this.tensor_cores = "METAL", MetalRenderer.tensor_cores
    //     // if getenv("EMULATE_AMD"): this.device, this.tensor_cores = "AMD", AMDRenderer.tensor_cores
    //     // if getenv("EMULATE_CUDA"): this.device, this.tensor_cores = "CUDA", CUDARenderer.tensor_cores
    //     // if getenv("EMULATE_INTEL"): this.device, this.suffix, this.tensor_cores = "INTEL", "INTEL", IntelRenderer.tensor_cores
    //     // if getenv("EMULATE_AMX"): this.device, this.tensor_cores = "CLANG", ClangRenderer.tensor_cores
    super()
  }
  override render = (name: string, uops: UOp[]): string => {
    const lops = uops.map((u) => [u.op, u.dtype, u.src.map((v) => uops.indexOf(v)), u.arg] as PyUOp)
    return btoa(bytes_to_string(serialize(lops)))
  }
}

export class PythonCompiler extends Compiler {
  override compile = (src: string): Uint8Array => string_to_bytes(atob(src))
}

export class PythonAllocator extends Allocator {
  _alloc = (size: number, options: BufferSpec): MemoryView => {
    return new MemoryView(new Uint8Array(size))
  }
  _copyin = (dest: MemoryView, src: MemoryView): MemoryView => {
    return dest.set(src)
  }
  _copyout = (dest: MemoryView, src: MemoryView): MemoryView => {
    return dest.set(src)
  }
  _free = (opaque: MemoryView, options: BufferSpec) => {
    throw new Error("PYTHON doesn't have _free")
  }
}

export class PythonDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, new PythonAllocator(), new PythonRenderer(), new PythonCompiler(), PythonProgram)
  }
}