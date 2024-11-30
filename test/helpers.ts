import { randomUUID } from 'node:crypto'
import { exec } from 'node:child_process'
import { getEnumString, type Ops, UPat } from '../src/ops.ts'
import { UOp } from '../src/ops.ts'
import { DType } from '../src/dtype.ts'
import { isNotNone } from '../src/helpers.ts'
import { expect } from 'expect'

const getPyOpsStr = (op: Ops) => `tiny.ops.Ops.${getEnumString(op)}`
export const pyStr = (v: any): string => {
  if (Array.isArray(v)) return v.length ? `(${v.map((x) => pyStr(x)).join(", ")},)`:"()"
  if (v === null || typeof v === 'undefined') return 'None'
  if (typeof v === 'boolean') return v ? 'True' : 'False'
  if (typeof v === 'number') return v === Infinity ? 'inf' : v === -Infinity ? '-inf' : Number.isNaN(v) ? 'math.nan' : v.toString()
  if (typeof v === 'string') return `"${v}"`

  if (v instanceof UPat) {
    return `tiny.ops.UPat(op=${v.op ? `(${v.op?.map(getPyOpsStr)},)` : 'None'}, dtype=${pyStr(v.dtype)}, src=${pyStr(v._inSrc)}, arg=${pyStr(v.arg)}, name=${pyStr(v.name)}, allow_any_len=${pyStr(v.allowedLen === -1)}, location=${pyStr(v.location)}, custom_early_reject=${pyStr(v.customEarlyReject)})`
  }
  if (v instanceof UOp) return `tiny.ops.UOp(op=${getPyOpsStr(v.op)}, dtype=${pyStr(v.dtype)}, src=${pyStr(v.src)}, arg=${pyStr(v.arg)})`
  if (v instanceof DType) return `tiny.ops.DType(${pyStr(v.priority)}, ${pyStr(v.itemsize)}, ${pyStr(v.name)}, ${pyStr(v.fmt)}, ${pyStr(v.count)}, ${pyStr(v._scalar)})`

  if (typeof v === 'function') return 'lambda x: x'
  if (typeof v === 'object') return `{${Object.entries(v).map((entry) => `"${entry[0]}":${pyStr(entry[1])}`).join(',')}}`
  throw new Error(`Invalid value: ${v}`)
}

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))

export const asdict = (o: any): object => {
  if (!o) return o
  if (Array.isArray(o)) return o.map(asdict)
  if (typeof o === 'object') return Object.fromEntries(Object.entries(o).filter((o) => typeof o[1] !== 'function').map(([k, v]) => [k, asdict(v)]))
  return o
}
export const trycatch = <T>(fn: () => T): T | string => {
  try {
    return fn()
  } catch (e) {
    if (e instanceof Error) return e.message
    else return 'error'
  }
}

export const deserialize = (data: string): any => {
  const obj = JSON.parse(data)

  const de = (i: any): any => {
    if (i === null) return undefined
    if (!i || typeof i !== 'object') return i
    if (Array.isArray(i)) return i.map(de)

    const type = i.__type
    if (type === 'UPat') return new UPat({ op: de(i.op), dtype: de(i.dtype), src: de(i.src), arg: de(i.arg), name: de(i.name), allowAnyLen: de(i.allow_any_len), location: de(i.location), customEarlyReject: de(i.custom_early_reject) })
    if (type === 'UOp') return new UOp({ op: de(i.op), dtype: de(i.dtype), src: de(i.src), arg: de(i.arg) })
    if (type === 'DType') return new DType({ priority: de(i.priority), itemsize: de(i.itemsize), name: de(i.name), fmt: de(i.fmt), count: de(i.count), _scalar: de(i._scalar) })
    return Object.fromEntries(Object.entries(i).map(([k, v]) => [k, de(v)]))
  }

  return de(obj)
}

export const python = async (code: string, data?: any) => {
  code = `
import tinygrad as tiny
import math
import json
from dataclasses import asdict
import itertools

def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)

def serialize(data):
    class CustomEncoder(json.JSONEncoder):
      def default(self, o):
          if isinstance(o,tiny.ops.UPat): return {"__type":"UPat","op":o.op,"dtype":o.dtype,"src":o._in_src,"arg":o.arg,"name":o.name,"allow_any_len":o.allowed_len == -1,"location":o.location,"custom_early_reject":o.custom_early_reject}
          if isinstance(o, tiny.ops.UOp): return {"__type": "UOp",'op': o.op, "dtype": o.dtype,"src":o.src,"arg":o.arg} 
          if isinstance(o, tiny.ops.DType): return {"__type": "DType", "priority": o.priority,"itemsize":o.itemsize,"name":o.name,"fmt":o.fmt,"count":o.count,"_scalar":o._scalar}
          if isinstance(o, itertools.repeat): return CustomEncoder.default(o)
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
  const res = await execAsync(`PYTHONPATH=./tinygrad python ${file}`)
  try {
    const json = res.split('<<<<<')[1]?.split('>>>>>')[0].trim()
    return deserialize(json)
  } catch (e) {
    if (e instanceof SyntaxError) throw new Error(`Parsing "${res.trim()}" failed.`)
    throw e
  }
}

export const test = <T extends any[]>(inputs: T[], fn: (...args: T) => any, code: string) => {
  return async (t: any) => {
    for (const input of inputs) {
      await t.step(JSON.stringify(input).slice(0, 120), async () => {
        const ts = fn(...input)
        const py = await python(code, input)
        expect(asdict(ts)).toEqual(asdict(py))
      })
    }
  }
}
