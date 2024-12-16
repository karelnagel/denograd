import { randomUUID } from 'node:crypto'
import { exec } from 'node:child_process'
import { DType, ImageDType, PtrDType } from '../src/dtype.ts'
import { getEnumString, isNotNone } from '../src/helpers.ts'
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

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))

export const asdict = (o: any): object => {
  if (!o) return o
  if (Array.isArray(o)) return o.map(asdict)
  if (o instanceof Map) return Object.fromEntries([...o.entries()].map(([k, v]) => [k, asdict(v)]))
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

const getPyOpsStr = (op: Ops) => `tiny.ops.Ops.${getEnumString(Ops, op)}`
const getPyOptOps = (op: OptOps) => `tiny.codegen.kernel.OptOps.${getEnumString(OptOps, op)}`
export const pyStr = (o: any, useList = false): string => {
  if (Array.isArray(o)) return o.length ? (useList ? `[${o.map((x) => pyStr(x)).join(', ')}]` : `(${o.map((x) => pyStr(x)).join(', ')},)`) : '()'
  if (o === null || typeof o === 'undefined') return 'None'
  if (typeof o === 'boolean') return o ? 'True' : 'False'
  if (typeof o === 'number' || typeof o === 'bigint') return o === Infinity ? 'math.inf' : o === -Infinity ? '-math.inf' : Number.isNaN(o) ? 'math.nan' : o.toString()
  if (typeof o === 'string') return `"${o}"`
  if (o instanceof Map) return `{${[...o.entries()].map(([k, v]) => `${pyStr(k)}:${pyStr(v)}`).join(',')}}`

  if (o instanceof UPat) {
    // if src is UPat[][] we use list, if UPat[] then tuple
    const src = Array.isArray(o._inSrc) ? (Array.isArray(o._inSrc.at(0)) ? pyStr(o._inSrc.at(0), true) : pyStr(o._inSrc)) : pyStr(o._inSrc)
    return `tiny.ops.UPat(op=${o.op ? `(${o.op?.map(getPyOpsStr)},)` : 'None'}, dtype=${pyStr(o.dtype)}, src=${src}, arg=${pyStr(o.arg)}, name=${pyStr(o.name)}, allow_any_len=${pyStr(o.allowed_len === -1)}, location=${
      pyStr(o.location)
    }, custom_early_reject=${pyStr(o.custom_early_reject)})`
  }
  if (o instanceof UOp) return `tiny.ops.UOp(op=${getPyOpsStr(o.op)}, dtype=${pyStr(o.dtype)}, src=${pyStr(o.src)}, arg=${pyStr(o.arg)})`
  if (o instanceof ImageDType) {
    return `tiny.dtype.ImageDType(${pyStr(o.priority)}, ${pyStr(o.itemsize)}, ${pyStr(o.name)}, ${pyStr(o.fmt)}, ${pyStr(o.count)}, ${pyStr(o._scalar)}, ${pyStr(o._base)}, ${pyStr(o.local)}, ${pyStr(o.v)}, ${pyStr(o.shape)})`
  }
  if (o instanceof PtrDType) return `tiny.dtype.PtrDType(${pyStr(o.priority)}, ${pyStr(o.itemsize)}, ${pyStr(o.name)}, ${pyStr(o.fmt)}, ${pyStr(o.count)}, ${pyStr(o._scalar)}, ${pyStr(o._base)}, ${pyStr(o.local)}, ${pyStr(o.v)})`
  if (o instanceof DType) return `tiny.dtype.DType(${pyStr(o.priority)}, ${pyStr(o.itemsize)}, ${pyStr(o.name)}, ${pyStr(o.fmt)}, ${pyStr(o.count)}, ${pyStr(o._scalar)})`
  if (o instanceof View) return `tiny.shape.view.View(shape=${pyStr(o.shape)}, strides=${pyStr(o.strides)}, offset=${pyStr(o.offset)}, mask=${pyStr(o.mask)}, contiguous=${pyStr(o.contiguous)})`
  if (o instanceof ShapeTracker) return `tiny.shape.shapetracker.ShapeTracker(views=${pyStr(o.views)})`
  if (o instanceof IndexContext) return `tiny.codegen.lowerer.IndexContext(${pyStr(o.idxs)}, ${pyStr(o.ridxs)}, ${pyStr(o.acc_num)})`
  if (o instanceof ClangRenderer) return `tiny.renderer.cstyle.ClangRenderer()`
  if (o instanceof Opt) return `tiny.codegen.kernel.Opt(${getPyOptOps(o.op)}, ${pyStr(o.axis)}, ${pyStr(o.amt)})`
  if (o instanceof Kernel) return `tiny.codegen.kernel.Kernel(${pyStr(o.ast)}, ${pyStr(o.opts)})`
  if (o instanceof KernelInfo) return `tiny.ops.KernelInfo(${pyStr(o.local_dims)}, ${pyStr(o.upcasted)}, ${pyStr(o.dont_use_locals)})`
  if (o instanceof BasicBlock) return `tiny.codegen.linearize.BasicBlock(${pyStr(o.ctx)}, ${pyStr(o.lst)}, ${pyStr(o.end)})`

  if (typeof o === 'function') return 'lambda x: x'
  if (typeof o === 'object') return `{${Object.entries(o).map((entry) => `"${entry[0]}":${pyStr(entry[1])}`).join(',')}}`
  throw new Error(`Invalid value: ${o}`)
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

export const compare = <T extends any[]>(inputs: T[], fn: (...args: T) => any, code: string, options?: { ignore?: number[]; stringSimilarity?: number }) => {
  return async (t: Deno.TestContext) => {
    for (const [i, input] of inputs.entries()) {
      await t.step({
        name: i.toString(),
        ignore: options?.ignore?.includes(i),
        fn: async () => {
          const ts = fn(...input)
          const py = await python(code, input)

          if (typeof ts === 'string' && typeof py === 'string') {
            const similarity = calculateSimilarity(ts, py)
            if (similarity < (options?.stringSimilarity || 1)) {
              expect(`${ts}\n\nsimilarity:${similarity}`).toEqual(`${py}\n\nsimilarity:${similarity}`)
            }
          } else {
            expect(asdict(ts)).toEqual(asdict(py))
          }
        },
      })
    }
  }
}

export const removeKeys = (obj: any, keys: string[]): any => {
  if (!obj || typeof obj !== 'object') return obj
  if (Array.isArray(obj)) return obj.map((x) => removeKeys(x, keys))
  const ret = { ...obj }
  for (const key of keys) delete ret[key]
  return Object.fromEntries(Object.entries(ret).map(([k, v]) => [k, removeKeys(v, keys)]))
}
