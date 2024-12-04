import { randomUUID } from 'node:crypto'
import { exec } from 'node:child_process'
import { UPat } from '../src/ops.ts'
import { UOp } from '../src/ops.ts'
import { DType } from '../src/dtype.ts'
import { isNotNone } from '../src/helpers.ts'
import { expect } from 'expect'
import { pyStr } from '../src/str.ts'
import process from 'node:process'
import { View } from '../src/shape/view.ts'

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
    if (type === 'DType') return new DType({ priority: de(i.priority), itemsize: de(i.itemsize), name: de(i.name), fmt: de(i.fmt), count: de(i.count), _scalar: de(i._scalar) })
    if (type === 'View') return new View({ shape: de(i.shape), strides: de(i.strides), offset: de(i.offset), mask: de(i.mask), contiguous: de(i.contiguous) })
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

def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)

def serialize(data):
    class CustomEncoder(json.JSONEncoder):
      def default(self, o):
          if isinstance(o,tiny.ops.UPat): return {"__type":"UPat","op":o.op,"dtype":o.dtype,"src":[o._in_src] if isinstance(o._in_src,list) else o._in_src,"arg":o.arg,"name":o.name,"allow_any_len":o.allowed_len == -1,"location":o.location,"custom_early_reject":o.custom_early_reject}
          if isinstance(o, tiny.ops.UOp): return {"__type": "UOp",'op': o.op, "dtype": o.dtype,"src":o.src,"arg":o.arg} 
          if isinstance(o, tiny.ops.DType): return {"__type": "DType", "priority": o.priority,"itemsize":o.itemsize,"name":o.name,"fmt":o.fmt,"count":o.count,"_scalar":o._scalar}
          if isinstance(o, tiny.shape.view.View): return {"__type": "View", "shape": o.shape, "strides": o.strides, "offset": o.offset, "mask": o.mask, "contiguous": o.contiguous}
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

export const compare = <T extends any[]>(inputs: T[], fn: (...args: T) => any, code: string, ignore?: number[]) => {
  return async (t: any) => {
    for (const [i, input] of inputs.entries()) {
      await t.step({
        name: i.toString(),
        ignore: ignore?.map((i) => i.toString()),
        fn: async () => {
          const ts = fn(...input)
          const py = await python(code, input)
          expect(asdict(ts)).toEqual(asdict(py))
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
