import { randomUUID } from 'node:crypto'
import { exec } from 'node:child_process'
import { DType, ImageDType, PtrDType } from '../src/dtype.ts'
import { getEnumString, isNotNone, Metadata } from '../src/helpers.ts'
import { expect } from 'expect'
import process from 'node:process'
import { KernelInfo, Ops, UOp, UPat } from '../src/ops.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'
import { View } from '../src/shape/view.ts'
import { writeFileSync } from 'node:fs'
import { IndexContext } from '../src/codegen/lowerer.ts'
import { Kernel, Opt, OptOps } from '../src/codegen/kernel.ts'
import { ClangRenderer } from '../src/renderer/cstyle.ts'
import { BasicBlock } from '../src/codegen/linearize.ts'
import { TensorCore } from '../src/renderer/index.ts'
import { ProgramSpec } from '../src/renderer/index.ts'
import { LazyBuffer } from '../src/engine/lazy.ts'
import { CompiledRunner, ExecItem, Runner } from '../src/engine/realize.ts'
import { ScheduleContext, ScheduleItem, ScheduleItemContext } from '../src/engine/schedule.ts'
import { _Device, _MallocAllocator, Allocator, Buffer, BufferSpec, Compiler, LRUAllocator } from '../src/device.ts'
import { PythonRenderer } from '../src/runtime/ops_python.ts'
import { MemoryView } from '../src/memoryview.ts'
import { Tensor } from '../src/tensor.ts'

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))

export const asdict = (o: any): any => {
  if (!o) return o
  if (o instanceof Set) return [...o.values().map((v) => asdict(v))]
  if (o instanceof DataView) return [...new Uint8Array(o.buffer)].map((v) => asdict(v))
  if (o instanceof MemoryView) return o.toList()
  if (Array.isArray(o)) return o.map(asdict)
  if (o instanceof Map) {
    const res = [...o.entries().map(([k, v]) => [asdict(k), asdict(v)])]
    return typeof res.at(0)?.at(0) === 'object' ? res : Object.fromEntries(res) // If it's Map<string,...> then return object, otherwise array
  }
  if (typeof o === 'object') return Object.fromEntries(Object.entries(o).filter((o) => typeof o[1] !== 'function').map(([k, v]) => [k, asdict(v)]))
  return o
}
export const tryCatch = <Args extends any[], Return>(fn: (...a: Args) => Return): (...a: Args) => Return | string => {
  return (...args) => {
    try {
      return fn(...args)
    } catch (e) {
      if (process.env.FAIL) throw e
      if (e instanceof Error) return e.message
      else return 'error'
    }
  }
}

class SkipFormatting {
  constructor(public value: string) {}
}
class PyEnum<T extends object> {
  constructor(public en: T, public op?: T[keyof T], public prefix = '') {}
  toString = () => this.op ? `${this.prefix}${getEnumString(this.en, this.op)}` : pyStr(undefined)
}
const OpsEnum = (op?: Ops) => new PyEnum(Ops, op, `tiny.ops.Ops.`)
const OptOpsEnum = (op: OptOps) => new PyEnum(OptOps, op, `tiny.codegen.kernel.OptOps.`)
export const pyStr = (o: any, useList = false): string => {
  const t = (strings: TemplateStringsArray, ...values: any[]) => {
    let result = strings[0]
    for (let i = 0; i < values.length; i++) result += pyStr(values[i]) + strings[i + 1]
    return result
  }
  if (o instanceof SkipFormatting) return o.value
  if (o instanceof PyEnum) return o.toString()

  if (Array.isArray(o)) return o.length ? (useList ? `[${o.map((x) => pyStr(x)).join(', ')}]` : `(${o.map((x) => pyStr(x)).join(', ')},)`) : '()'
  if (o === null || typeof o === 'undefined') return 'None'
  if (typeof o === 'boolean') return o ? 'True' : 'False'
  if (typeof o === 'number' || typeof o === 'bigint') return o === Infinity ? 'math.inf' : o === -Infinity ? '-math.inf' : Number.isNaN(o) ? 'math.nan' : o.toString()
  if (typeof o === 'string') return `"${o.replaceAll('\n', '\\\n')}"`
  if (o instanceof Map) return `{${[...o.entries()].map(([k, v]) => `${pyStr(k)}:${pyStr(v)}`).join(',')}}`
  if (o instanceof Set) return `set([${[...o].map((o) => pyStr(o)).join(', ')}])`

  // ************ TENSOR ************
  if (o instanceof Tensor) return t`tiny.tensor.Tensor(${o.lazydata}, requires_grad=${o.requires_grad}, dtype=${o.dtype}, device=${o.device})`

  // ************ ENGINE ************
  if (o instanceof LazyBuffer) return t`tiny.engine.lazy.LazyBuffer(${o.device}, ${o.st}, ${o.dtype}, ${OpsEnum(o.op)}, ${o.arg}, ${o.srcs || []}, ${o._base}, ${o.metadata})`

  if (o instanceof CompiledRunner) return t`tiny.engine.realize.CompiledRunner(${o.p}, ${o.lib})`
  if (o instanceof Runner) return t`tiny.engine.realize.Runner(${o.display_name}, ${o.device}, ${o.op_estimate}, ${o.mem_estimate}, ${o.lds_estimate})`
  if (o instanceof ExecItem) return t`tiny.engine.realize.ExecItem(${o.prg}, ${o.bufs}, ${o.metadata})`

  if (o instanceof ScheduleItem) return t`tiny.engine.schedule.ScheduleItem(${o.ast}, ${o.bufs}, ${o.metadata}, ${o.assign_preloads})`
  if (o instanceof ScheduleContext) return t`tiny.engine.schedule.ScheduleContext(${o.lazybufs}, ${o.var_vals}, ${o.assigns}, ${o.realizes}, ${o.allbufs}, ${o.ops_metadata}, ${o.children})`
  if (o instanceof ScheduleItemContext) return t`tiny.engine.schedule.ScheduleItemContext(${o.lazybufs}, ${o.ops_metadata}, ${o.assigns}, ${o.var_vals}, ${o.sinked}, ${o.sts}, ${o.bufs}, ${o.metadata}, ${o.assign_adj})`

  // ************ DEVICE ************
  if (o instanceof _Device) return t`tiny.device._Device()`
  if (o instanceof BufferSpec) return t`tiny.device.BufferSpec(${o.image}, ${o.uncached}, ${o.cpu_access}, ${o.host}, ${o.nolru}, ${o.external_ptr})`
  if (o instanceof Buffer) {
    return t`tiny.device.Buffer(device=${o.device}, size=${o.size}, dtype=${o.dtype}, opaque=${o.in_opaque}, options=${o.options}, initial_value=${o.in_initial_value}, lb_refcount=${o._lb_refcount}, base=${o._base}, offset=${o.offset}, preallocate=${o.in_preallocate})`
  }
  if (o instanceof _MallocAllocator) return t`tiny.device._MallocAllocator()`
  if (o instanceof LRUAllocator) return t`tiny.device.LRUAllocator()`
  if (o instanceof Allocator) return t`tiny.device.Allocator()`
  if (o instanceof Compiler) return t`tiny.device.Compiler(${o.cachekey})`

  // ************ CODEGEN ************
  if (o instanceof IndexContext) return t`tiny.codegen.lowerer.IndexContext(${o.idxs}, ${o.ridxs}, ${o.acc_num})`
  if (o instanceof Kernel) return t`tiny.codegen.kernel.Kernel(${o.ast}, ${o.opts})`
  if (o instanceof BasicBlock) return t`tiny.codegen.linearize.BasicBlock(${o.ctx}, ${o.lst}, ${o.end})`
  if (o instanceof Opt) return t`tiny.codegen.kernel.Opt(${OptOpsEnum(o.op)}, ${o.axis}, ${o.amt})`

  // ************ RENDERER ************
  if (o instanceof ClangRenderer) return t`tiny.renderer.cstyle.ClangRenderer()`
  if (o instanceof PythonRenderer) return t`PythonRenderer()`
  if (o instanceof TensorCore) return t`tiny.renderer.TensorCore(dims=${o.dims}, threads=${o.threads}, reduce_axes=${o.reduce_axes}, upcast_axes=${o.upcast_axes}, dtype_in=${o.dtype_in}, dtype_out=${o.dtype_out})`
  if (o instanceof ProgramSpec) {
    return t`tiny.renderer.ProgramSpec(name=${o.name},src=${o.src},device=${o.device},uops=${o.uops},mem_estimate=${o.mem_estimate},global_size=${o.global_size},local_size=${o.local_size},vars=${o.vars},globals=${new SkipFormatting(pyStr(o.globals, true))},outs=${new SkipFormatting(pyStr(o.outs, true))}, _ran_post_init=${o._ran_post_init})`
  }

  // ************ SHAPE ************
  if (o instanceof View) return t`tiny.shape.view.View(shape=${o.shape}, strides=${o.strides}, offset=${o.offset}, mask=${o.mask}, contiguous=${o.contiguous})`
  if (o instanceof ShapeTracker) return t`tiny.shape.shapetracker.ShapeTracker(views=${o.views})`

  // ************ DTYPE ************
  if (o instanceof ImageDType) return t`tiny.dtype.ImageDType(${o.priority}, ${o.itemsize}, ${o.name}, ${o.fmt}, ${o.count}, ${o._scalar}, ${o._base}, ${o.local}, ${o.v}, ${o.shape})`
  if (o instanceof PtrDType) return t`tiny.dtype.PtrDType(${o.priority}, ${o.itemsize}, ${o.name}, ${o.fmt}, ${o.count}, ${o._scalar}, ${o._base}, ${o.local}, ${o.v})`
  if (o instanceof DType) return t`tiny.dtype.DType(${o.priority}, ${o.itemsize}, ${o.name}, ${o.fmt}, ${o.count}, ${o._scalar})`

  // ************ OPS ************
  if (o instanceof UPat) {
    // if src is UPat[][] we use list, if UPat[] then tuple
    const src = Array.isArray(o._in_src) ? (Array.isArray(o._in_src.at(0)) ? new SkipFormatting(pyStr(o._in_src.at(0), true)) : o._in_src) : o._in_src
    return t`tiny.ops.UPat(op=${o.op?.map((op) => OpsEnum(op))}, dtype=${o.dtype}, src=${src}, arg=${o.arg}, name=${o.name}, allow_any_len=${o.allowed_len === -1}, location=${o.location}, custom_early_reject=${o.custom_early_reject})`
  }
  if (o instanceof UOp) return t`tiny.ops.UOp(op=${OpsEnum(o.op)}, dtype=${o.dtype}, src=${o.src}, arg=${o.arg})`
  if (o instanceof KernelInfo) return t`tiny.ops.KernelInfo(${o.local_dims}, ${o.upcasted}, ${o.dont_use_locals})`

  // ************ HELPERS ************
  if (o instanceof Metadata) return t`tiny.helpers.Metadata(${o.name}, ${o.caller}, ${o.backward})`

  if (o instanceof Uint8Array) return t`bytes(${Array.from(o)})`
  if (o instanceof MemoryView) return t`memoryview(bytes(${Array.from(o.toBytes())}))`

  if (typeof o === 'function') return 'lambda x: x'
  if (o?.constructor?.name === 'Object') return `{${Object.entries(o).map((entry) => t`${entry[0]}:${entry[1]}`).join(',')}}`
  throw new Error(`Invalid value: ${o.constructor.name} ${JSON.stringify(o)}`)
}
export const python = async <T = any>(code: string, data?: any): Promise<T> => {
  code = `
import tinygrad as tiny
import math
import json
from dataclasses import asdict
import itertools
from tinygrad.renderer import cstyle
from tinygrad.ops import Ops
from tinygrad.to_ts import to_ts
from tinygrad.runtime.ops_python import PythonRenderer

def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)

${isNotNone(data) ? `data = ${pyStr(data)}` : ''}
def out(o):
    print("<<<<<"+to_ts(o)+">>>>>")

${code}
`
  const file = `/tmp/tiny_${randomUUID()}.py`
  console.log(file)
  await execAsync(`echo ${JSON.stringify(code.trim())} > ${file}`)
  const res = await execAsync(`PYTHONPATH=./tinygrad python3 ${file}`)
  console.log(res.split('<<<<<')[0])
  const ts = res.split('<<<<<')[1]?.split('>>>>>')[0].trim()
  try {
    return eval(ts)
  } catch (e) {
    writeFileSync(`invalidcode-${randomUUID()}.ts`, ts)
    throw new Error(`eval failed, code:"${ts}" error: ${e}`)
  }
}

function calculateSimilarity(str1: string, str2: string): number {
  str1 = str1.replaceAll('(', '[').replaceAll(')', ']').replaceAll('new ', '')
  str2 = str2.replaceAll('(', '[').replaceAll(')', ']').replaceAll('new ', '')
  const len1 = str1.length
  const len2 = str2.length
  const dp: number[][] = Array.from({ length: len1 + 1 }, () => Array(len2 + 1).fill(0))

  for (let i = 0; i <= len1; i++) dp[i][0] = i
  for (let j = 0; j <= len2; j++) dp[0][j] = j

  for (let i = 1; i <= len1; i++) {
    for (let j = 1; j <= len2; j++) {
      if (str1[i - 1] === str2[j - 1]) dp[i][j] = dp[i - 1][j - 1]
      else dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    }
  }
  const dist = dp[len1][len2]
  const maxLength = Math.max(len1, len2)

  return 1 - dist / maxLength
}

export const compare = <T extends any[]>(inputs: T[], fn: (...args: T) => any, code: string | string[], options: {
  ignore?: number[]
  ignoreKeys?: string[]
  stringSimilarity?: number
} = {}) => {
  return async (t: Deno.TestContext) => {
    for (const [i, input] of inputs.entries()) {
      await t.step({
        name: i.toString(),
        ignore: options.ignore?.includes(i),
        fn: async () => {
          const ts = fn(...input)
          if (Array.isArray(code)) code = code.join('\n')
          const py = await python(code, input)

          if (typeof ts === 'string' && typeof py === 'string') {
            const similarity = calculateSimilarity(ts, py)
            if (similarity < (options?.stringSimilarity || 1)) {
              expect(`${ts}\n\nsimilarity:${similarity}`).toEqual(`${py}\n\nsimilarity:${similarity}`)
            }
          } else {
            expect(removeKeys(asdict(ts), options.ignoreKeys)).toEqual(removeKeys(asdict(py), options.ignoreKeys))
          }
        },
      })
    }
  }
}

export const removeKeys = (obj: any, keys?: string[]): any => {
  if (!keys?.length) return obj
  if (!obj || typeof obj !== 'object') return obj
  if (Array.isArray(obj)) return obj.map((x) => removeKeys(x, keys))
  const ret = { ...obj }
  for (const key of keys) delete ret[key]
  return Object.fromEntries(Object.entries(ret).map(([k, v]) => [k, removeKeys(v, keys)]))
}
