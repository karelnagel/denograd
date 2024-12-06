import { randomUUID } from 'node:crypto'
import { exec } from 'node:child_process'
import { DType, ImageDType, PtrDType } from '../src/dtype.ts'
import { isNotNone } from '../src/helpers.ts'
import { expect } from 'expect'
import process from 'node:process'
import { type Ops, opsString, UOp, UPat } from '../src/ops.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'
import { View } from '../src/shape/view.ts'

const getPyOpsStr = (op: Ops) => `tiny.ops.${opsString(op)}`
export const pyStr = (v: any, useList = false): string => {
  if (Array.isArray(v)) return v.length ? (useList ? `[${v.map((x) => pyStr(x)).join(', ')}]` : `(${v.map((x) => pyStr(x)).join(', ')},)`) : '()'
  if (v === null || typeof v === 'undefined') return 'None'
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') return v === Infinity ? 'inf' : v === -Infinity ? '-inf' : Number.isNaN(v) ? 'math.nan' : v.toString()
  if (typeof v === 'string') return `"${v}"`

  if (v instanceof UPat) {
    // if src is UPat[][] we use list, if UPat[] then tuple
    const src = Array.isArray(v._inSrc) ? (Array.isArray(v._inSrc.at(0)) ? pyStr(v._inSrc.at(0), true) : pyStr(v._inSrc)) : pyStr(v._inSrc)
    return `tiny.ops.UPat(op=${v.op ? `(${v.op?.map(getPyOpsStr)},)` : 'None'}, dtype=${pyStr(v.dtype)}, src=${src}, arg=${pyStr(v.arg)}, name=${pyStr(v.name)}, allow_any_len=${pyStr(v.allowedLen === -1)}, location=${
      pyStr(v.location)
    }, custom_early_reject=${pyStr(v.customEarlyReject)})`
  }
  if (v instanceof UOp) return `tiny.ops.UOp(op=${getPyOpsStr(v.op)}, dtype=${pyStr(v.dtype)}, src=${pyStr(v.src)}, arg=${pyStr(v.arg)})`
  if (v instanceof ImageDType) {
    return `tiny.dtype.ImageDType(${pyStr(v.priority)}, ${pyStr(v.itemsize)}, ${pyStr(v.name)}, ${pyStr(v.fmt)}, ${pyStr(v.count)}, ${pyStr(v._scalar)}, ${pyStr(v._base)}, ${pyStr(v.local)}, ${pyStr(v.v)}, ${pyStr(v.shape)})`
  }
  if (v instanceof PtrDType) return `tiny.dtype.PtrDType(${pyStr(v.priority)}, ${pyStr(v.itemsize)}, ${pyStr(v.name)}, ${pyStr(v.fmt)}, ${pyStr(v.count)}, ${pyStr(v._scalar)}, ${pyStr(v._base)}, ${pyStr(v.local)}, ${pyStr(v.v)})`
  if (v instanceof DType) return `tiny.dtype.DType(${pyStr(v.priority)}, ${pyStr(v.itemsize)}, ${pyStr(v.name)}, ${pyStr(v.fmt)}, ${pyStr(v.count)}, ${pyStr(v._scalar)})`
  if (v instanceof View) return `tiny.shape.view.View(shape=${pyStr(v.shape)}, strides=${pyStr(v.strides)}, offset=${pyStr(v.offset)}, mask=${pyStr(v.mask)}, contiguous=${pyStr(v.contiguous)})`
  if (v instanceof ShapeTracker) return `tiny.shape.shapetracker.ShapeTracker(views=${pyStr(v.views)})`

  if (typeof v === 'function') return 'lambda x: x'
  if (typeof v === 'object') return `{${Object.entries(v).map((entry) => `"${entry[0]}":${pyStr(entry[1])}`).join(',')}}`
  throw new Error(`Invalid value: ${v}`)
}

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

export const deserialize = (data: string): any => {
  data = data.replaceAll('Infinity', '696969696')
  const obj = JSON.parse(data)

  const de = (i: any): any => {
    if (i === null) return undefined
    if (i === 696969696) return Infinity
    if (i === -696969696) return -Infinity
    if (!i || typeof i !== 'object') return i
    if (Array.isArray(i)) return i.map(de)

    const type = i.__type
    if (type === 'UPat') return new UPat({ op: de(i.op), dtype: de(i.dtype), src: de(i.src), arg: de(i.arg), name: de(i.name), allowAnyLen: de(i.allow_any_len), location: de(i.location), customEarlyReject: de(i.custom_early_reject) })
    if (type === 'UOp') return new UOp({ op: de(i.op), dtype: de(i.dtype), src: de(i.src), arg: de(i.arg) })
    if (type === 'ImageDType') {
      return new ImageDType({ priority: de(i.priority), itemsize: de(i.itemsize), name: de(i.name), fmt: de(i.fmt), count: de(i.count), _scalar: de(i._scalar), shape: de(i.shape), _base: de(i.base), local: de(i.local), v: de(i.v) })
    }
    if (type === 'PtrDType') return new PtrDType({ priority: de(i.priority), itemsize: de(i.itemsize), name: de(i.name), fmt: de(i.fmt), count: de(i.count), _scalar: de(i._scalar), _base: de(i.base), local: de(i.local), v: de(i.v) })
    if (type === 'DType') return new DType({ priority: de(i.priority), itemsize: de(i.itemsize), name: de(i.name), fmt: de(i.fmt), count: de(i.count), _scalar: de(i._scalar) })
    if (type === 'View') return new View({ shape: de(i.shape), strides: de(i.strides), offset: de(i.offset), mask: de(i.mask), contiguous: de(i.contiguous) })
    if (type === 'ShapeTracker') return new ShapeTracker(de(i.views))
    return Object.fromEntries(Object.entries(i).map(([k, v]) => [k, de(v)]))
  }

  return de(obj)
}

export const python = async <T = any>(code: string, data?: any): Promise<T> => {
  code = `
import tinygrad as tiny
import math
import json
from dataclasses import asdict
import itertools
from tinygrad.renderer import cstyle

def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)

def serialize(data):
    class CustomEncoder(json.JSONEncoder):
      def default(self, o):
          if isinstance(o,tiny.ops.UPat): return {"__type":"UPat","op":o.op,"dtype":o.dtype,"src":[o._in_src] if isinstance(o._in_src,list) else o._in_src,"arg":o.arg,"name":o.name,"allow_any_len":o.allowed_len == -1,"location":o.location,"custom_early_reject":o.custom_early_reject}
          if isinstance(o, tiny.ops.UOp): return {"__type": "UOp",'op': o.op, "dtype": o.dtype,"src":o.src,"arg":o.arg} 
          if isinstance(o, tiny.dtype.ImageDType): return {"__type": "ImageDType", "priority": o.priority,"itemsize":o.itemsize,"name":o.name,"fmt":o.fmt,"count":o.count,"_scalar":o._scalar,"shape":o.shape,"base":o.base,"local":o.local,"v":o.v}
          if isinstance(o, tiny.dtype.PtrDType): return {"__type": "PtrDType", "priority": o.priority,"itemsize":o.itemsize,"name":o.name,"fmt":o.fmt,"count":o.count,"_scalar":o._scalar,"base":o.base,"local":o.local,"v":o.v}
          if isinstance(o, tiny.dtype.DType): return {"__type": "DType", "priority": o.priority,"itemsize":o.itemsize,"name":o.name,"fmt":o.fmt,"count":o.count,"_scalar":o._scalar}
          if isinstance(o, tiny.shape.view.View): return {"__type": "View", "shape": o.shape, "strides": o.strides, "offset": o.offset, "mask": o.mask, "contiguous": o.contiguous}
          if isinstance(o, tiny.shape.shapetracker.ShapeTracker): return {"__type": "ShapeTracker", "views": o.views}
          if isinstance(o, itertools.repeat): return CustomEncoder.default(o)
          if callable(o): return None
          if isinstance(o, set): return list(o)
          return super().default(o)
    return json.dumps(data, cls=CustomEncoder)

${isNotNone(data) ? `data = ${pyStr(data)}` : ''}
def out(o):
    print("<<<<<"+serialize(o)+">>>>>")

${code}
`
  const file = `/tmp/tiny_${randomUUID()}.py`
  console.log(file)
  await execAsync(`echo ${JSON.stringify(code.trim())} > ${file}`)
  const res = await execAsync(`PYTHONPATH=./tinygrad python3 ${file}`)
  try {
    console.log(res.split('<<<<<')[0])
    const json = res.split('<<<<<')[1]?.split('>>>>>')[0].trim()
    return deserialize(json)
  } catch (e) {
    if (e instanceof SyntaxError) throw new Error(`Parsing "${res.trim()}" failed: ${e}`)
    throw e
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
  return async (t: any) => {
    for (const [i, input] of inputs.entries()) {
      await t.step({
        name: i.toString(),
        ignore: options?.ignore?.map((i) => i.toString()),
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
