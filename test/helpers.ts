// deno-lint-ignore-file custom-lint-rules/no-null
import '../denograd/env/server.ts'
import { DType, dtypes, ImageDType, INVERSE_DTYPES_DICT, PtrDType } from '../denograd/dtype.ts'
import { ArrayMap, bytes_to_string, Enum, Metadata, random_id } from '../denograd/helpers.ts'
import { expect } from '@std/expect'
import process from 'node:process'
import { KernelInfo, Ops, UOp, UPat } from '../denograd/ops.ts'
import { ShapeTracker } from '../denograd/shape/shapetracker.ts'
import { View } from '../denograd/shape/view.ts'
import { IndexContext } from '../denograd/codegen/lowerer.ts'
import { Kernel, Opt, OptOps } from '../denograd/codegen/kernel.ts'
import { ClangRenderer } from '../denograd/renderer/cstyle.ts'
import { BasicBlock } from '../denograd/codegen/linearize.ts'
import { Estimates, TensorCore } from '../denograd/renderer/index.ts'
import { ProgramSpec } from '../denograd/renderer/index.ts'
import { CompiledRunner, ExecItem, Runner } from '../denograd/engine/realize.ts'
import { ScheduleContext, ScheduleItem, ScheduleItemContext } from '../denograd/engine/schedule.ts'
import { _Device, _MallocAllocator, Allocator, Buffer, BufferSpec, Compiler, LRUAllocator } from '../denograd/device.ts'
import { PythonRenderer } from '../denograd/runtime/ops_js.ts'
import { MemoryView } from '../denograd/memoryview.ts'
import { Tensor } from '../denograd/tensor.ts'
import { WGSLRenderer } from '../denograd/renderer/wgsl.ts'

export const test = (name: string, fn: (t: Deno.TestContext) => void | Promise<void>, opts: { ignore?: boolean } = {}) => Deno.test({ name, fn, sanitizeResources: false, ...opts })

export const asdict = async (o: any): Promise<any> => {
  if (typeof o === 'function') return undefined
  if (typeof o === 'number' && Number.isFinite(o)) return (Math.round(o * 10000) / 10000).toString()
  if (Array.isArray(o)) return await Promise.all(o.map(async (x) => await asdict(x)))
  if (o === undefined || o === null) return o
  if (typeof o === 'bigint') return o.toString()
  if (o instanceof Enum) return o.toString()
  if (o instanceof Set) return await Promise.all([...o.values().map(async (v) => await asdict(v))])
  if (o instanceof DType) return o.toString()
  if (o instanceof MemoryView) return o.toString()
  if (o instanceof UOp) return o.toString()
  if (o instanceof Tensor) return { dtype: o.dtype.toString(), shape: o.shape, data: await asdict(await o.tolist()) }

  if (o instanceof Map || o instanceof ArrayMap) {
    if (o instanceof ArrayMap) o = new Map(o.entries())
    const res = await Promise.all([...o.entries().map(async ([k, v]: any) => [await asdict(k), await asdict(v)])])
    return typeof res.at(0)?.at(0) === 'object' ? res : Object.fromEntries(res) // If it's Map<string,...> then return object, otherwise array
  }
  if (typeof o === 'object') return Object.fromEntries(await Promise.all(Object.entries(o).filter((o) => typeof o[1] !== 'function').map(async ([k, v]) => [k, await asdict(v)])))
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

// I'm not proud of this async mess here
const pyStr = async (o: any, useList = false): Promise<string> => {
  const t = async (strings: TemplateStringsArray, ...values: any[]) => {
    let result = strings[0]
    for (let i = 0; i < values.length; i++) result += await pyStr(values[i]) + strings[i + 1]
    return result
  }
  if (o instanceof SkipFormatting) return o.value
  if (o instanceof Ops) return `tiny.ops.${o.toString()}`
  if (o instanceof OptOps) return `tiny.codegen.kernel.${o.toString()}`

  if (Array.isArray(o)) return o.length ? (useList ? `[${await Promise.all(o.map(async (x) => await pyStr(x)).join(', '))}]` : `(${(await Promise.all(o.map(async (x) => await pyStr(x)))).join(', ')},)`) : '()'
  if (o === null || typeof o === 'undefined') return 'None'
  if (typeof o === 'boolean') return o ? 'True' : 'False'
  if (typeof o === 'bigint') return o.toString()
  if (typeof o === 'number') {
    if (o === Infinity) return 'math.inf'
    if (o === -Infinity) return '-math.inf'
    if (Number.isNaN(o)) return 'math.nan'
    return o.toString()
  }
  if (typeof o === 'string') return `"${o.replaceAll('\n', '\\\n')}"`
  if (o instanceof Map) return `{${(await Promise.all([...o.entries()].map(async ([k, v]) => `${await pyStr(k)}:${await pyStr(v)}`))).join(',')}}`
  if (o instanceof Set) return `set([${(await Promise.all([...o].map(async (o) => await pyStr(o)))).join(', ')}])`

  // ************ TENSOR ************
  if (o instanceof Tensor) {
    return t`tiny.tensor.Tensor(${await o.clone().tolist()}, requires_grad=${o.requires_grad}, dtype=${o.dtype})`
  }

  // ************ ENGINE ************
  if (o instanceof CompiledRunner) return t`tiny.engine.realize.CompiledRunner(${o.p}, ${o.lib})`
  if (o instanceof Estimates) return t`tiny.renderer.Estimates(${o.ops}, ${o.lds}, ${o.mem})`
  if (o instanceof Runner) return t`tiny.engine.realize.Runner(${o.display_name}, ${o.device}, ${o.estimates})`
  if (o instanceof ExecItem) return t`tiny.engine.realize.ExecItem(${o.prg}, ${o.bufs}, ${o.metadata})`

  if (o instanceof ScheduleItem) return t`tiny.engine.schedule.ScheduleItem(${o.ast}, ${o.bufs}, ${o.metadata})`
  if (o instanceof ScheduleContext) return t`tiny.engine.schedule.ScheduleContext(${o.tensor_uops}, ${o.var_vals}, ${o.assigns}, ${o.realizes}, ${o.allbufs}, ${o.ops_metadata}, ${o.contiguous}, ${o.children}, ${o.becomes_map})`
  if (o instanceof ScheduleItemContext) return t`tiny.engine.schedule.ScheduleItemContext(${o.var_vals}, ${o.sts}, ${o.bufs})`

  // ************ DEVICE ************
  if (o instanceof Buffer) return t`tiny.device.Buffer(${o.device}, ${o.size}, ${o.dtype}, None, ${o.options}, None, 0, ${o._base}, ${o.offset})`
  if (o instanceof _Device) return t`tiny.device._Device()`
  if (o instanceof BufferSpec) return t`tiny.device.BufferSpec(${o.image}, ${o.uncached}, ${o.cpu_access}, ${o.host}, ${o.nolru}, ${o.external_ptr})`
  if (o instanceof _MallocAllocator) return t`tiny.device._MallocAllocator()`
  if (o instanceof LRUAllocator) return t`tiny.device.LRUAllocator()`
  if (o instanceof Allocator) return t`tiny.device.Allocator()`
  if (o instanceof Compiler) return t`tiny.device.Compiler(${o.cachekey})`

  // ************ CODEGEN ************
  if (o instanceof IndexContext) return t`tiny.codegen.lowerer.IndexContext(${o.idxs}, ${o.ridxs}, ${o.acc_num})`
  if (o instanceof Kernel) return t`tiny.codegen.kernel.Kernel(${o.ast}, ${o.opts})`
  if (o instanceof BasicBlock) return t`tiny.codegen.linearize.BasicBlock(${o.ctx}, ${o.lst}, ${o.end})`
  if (o instanceof Opt) return t`tiny.codegen.kernel.Opt(${o.op}, ${o.axis}, ${o.amt})`

  // ************ RENDERER ************
  if (o instanceof ClangRenderer) return t`tiny.renderer.cstyle.ClangRenderer()`
  if (o instanceof WGSLRenderer) return t`tiny.renderer.wgsl.WGSLRenderer()`
  if (o instanceof PythonRenderer) return t`PythonRenderer()`
  if (o instanceof TensorCore) return t`tiny.renderer.TensorCore(dims=${o.dims}, threads=${o.threads}, elements_per_thread=${o.elements_per_thread}, dtype_in=${o.dtype_in}, dtype_out=${o.dtype_out}, opts=${o.opts}, swizzle=${o.swizzle})`
  if (o instanceof ProgramSpec) {
    return t`tiny.renderer.ProgramSpec(${o.name}, ${o.src}, ${o.device}, ${o.uops}, ${o.mem_estimate}, ${o.global_size}, ${o.local_size}, ${o.vars}, list(${o.globals}), list(${o.outs}), list(${o.ins}), ${o._ran_post_init})`
  }

  // ************ SHAPE ************
  if (o instanceof View) return t`tiny.shape.view.View(shape=${o.shape}, strides=${o.strides}, offset=${o.offset}, mask=${o.mask}, contiguous=${o.contiguous})`
  if (o instanceof ShapeTracker) return t`tiny.shape.shapetracker.ShapeTracker(views=${o.views})`

  // ************ DTYPE ************
  if (o instanceof ImageDType) return `tiny.dtype.dtypes.${o.name}(${await pyStr(o.shape)})${o.v !== 1 ? `.vec(${o.v})` : ''}`
  if (o instanceof PtrDType) return `${await pyStr(o.base)}.ptr(${o.size}${o.local ? ', local=True' : ''})${o.v !== 1 ? `.vec(${o.v})` : ''}`
  if (o instanceof DType) return `tiny.dtype.dtypes.${INVERSE_DTYPES_DICT[o.scalar().name]}${o.count > 1 ? `.vec(${o.count})` : ''}`

  // ************ OPS ************
  if (o instanceof UPat) {
    // if src is UPat[][] we use list, if UPat[] then tuple
    const src = Array.isArray(o._in_src) ? (Array.isArray(o._in_src.at(0)) ? new SkipFormatting(await pyStr(o._in_src.at(0), true)) : o._in_src) : o._in_src
    return t`tiny.ops.UPat(op=${o.op}, dtype=${o.dtype}, src=${src}, arg=${o.arg}, name=${o.name}, allow_any_len=${o.allowed_len === -1}, location=${o.location}, custom_early_reject=${o.custom_early_reject})`
  }
  if (o instanceof UOp) {
    const arg = o.op === Ops.CONST && o.dtype === dtypes.float ? `float(${await pyStr(o.arg)})` : await pyStr(o.arg)
    return `tiny.ops.UOp(op=${await pyStr(o.op)}, dtype=${await pyStr(o.dtype)}, src=${await pyStr(o.src)}, arg=${arg})`
  }
  if (o instanceof KernelInfo) return t`tiny.ops.KernelInfo(${o.local_dims}, ${o.upcasted}, ${o.dont_use_locals})`

  // ************ HELPERS ************
  if (o instanceof Metadata) return t`tiny.helpers.Metadata(${o.name}, ${o.caller}, ${o.backward})`

  if (o instanceof Uint8Array) return t`bytes(${Array.from(o)})`
  if (o instanceof MemoryView) return t`memoryview(bytes(${Array.from(o.bytes)}))`

  if (typeof o === 'function') return 'lambda x: x'
  if (o?.constructor?.name === 'Object') return `{${(await Promise.all(Object.entries(o).map(async (entry) => await t`${entry[0]}:${entry[1]}`))).join(',')}}`
  throw new Error(`Invalid value: ${o.constructor.name} ${JSON.stringify(o)}`)
}
export const py_bench = async (b: Deno.BenchContext, code: string | string[]) => {
  if (Array.isArray(code)) code = code.join('\n')
  const file = `/tmp/tiny_${random_id()}.py`
  await Deno.writeTextFile(file, code)
  b.start()
  const out = await new Deno.Command(`python3`, { env: { PYTHONPATH: './tinygrad' }, args: [file] }).output()
  if (out.stdout.length) console.log(bytes_to_string(out.stdout))
  if (!out.success) throw new Error(bytes_to_string(out.stderr))
  b.end()
}

export const python = async <T = any>(code: string | string[], data?: any): Promise<T> => {
  if (Array.isArray(code)) code = code.join('\n')
  code = `
import tinygrad as tiny
import math
from tinygrad.renderer import cstyle
from tinygrad.renderer import wgsl
from tinygrad.ops import Ops
from to_ts import to_ts
from tinygrad.runtime.ops_python import PythonRenderer

def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)

${data !== undefined ? `data = ${(await pyStr(data)).replaceAll('JS', 'PYTHON')}` : ''}
def out(o):
    print("<<<<<"+to_ts(o)+">>>>>")

${code}
`
  const file = `/tmp/tiny_${random_id()}.py`
  // console.log(file)
  await Deno.writeTextFile(file, code.trim())
  const envs = Object.entries(process.env).filter(([k, v]) => k.startsWith('TINY_')).map(([k, v]) => [k.replace('TINY_', ''), v])
  const out = await new Deno.Command(`python3`, { args: [file], clearEnv: true, env: { PATH: process.env.PATH, 'PYTHONPATH': './test:./tinygrad', ...Object.fromEntries(envs) } }).output()
  if (!out.success) throw new Error(bytes_to_string(out.stderr))
  const [stdout, ts] = bytes_to_string(out.stdout).replace('>>>>>', '').trim().split('<<<<<')
  if (stdout) console.log(stdout)
  try {
    return eval(ts)
  } catch (e) {
    Deno.writeTextFileSync(`invalidcode-${random_id()}.ts`, ts)
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

export const compare = <T extends any[] = any[]>(inputs: T[] | (() => T[]), fn: (...args: T) => any, code: string | string[], options: {
  ignore?: number[]
  ignoreKeys?: string[]
  stringSimilarity?: number
} = {}) => {
  return async (t: Deno.TestContext) => {
    if (typeof inputs === 'function') inputs = inputs()
    for (const [i, input] of inputs.entries()) {
      await t.step({
        name: i.toString(),
        ignore: options.ignore?.includes(i),
        fn: async () => {
          const py = await python(code, input)
          const ts = await fn(...input)

          if (typeof ts === 'string' && typeof py === 'string') {
            const similarity = calculateSimilarity(ts, py)
            if (similarity < (options?.stringSimilarity || 1)) {
              expect(`${ts}\n\nsimilarity:${similarity}`).toEqual(`${py}\n\nsimilarity:${similarity}`)
            }
          } else {
            expect(removeKeys(await asdict(ts), options.ignoreKeys)).toEqual(removeKeys(await asdict(py), options.ignoreKeys))
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
