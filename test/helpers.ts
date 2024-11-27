import { equal } from 'assert'
import { randomUUID } from 'node:crypto'
import { exec } from 'node:child_process'
import { UPat } from '../src/ops.ts'
import { UOp } from '../src/ops.ts'
import { DType } from '../src/dtype.ts'

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))

export const toPython = (val: any): string => {
  if (Array.isArray(val)) return `[${val.map((x) => toPython(x))}]`
  if (val === null || typeof val === 'undefined') return 'None'
  if (typeof val === 'boolean') return val ? 'True' : 'False'
  if (typeof val === 'number') return val === Infinity ? 'inf' : val === -Infinity ? '-inf' : Number.isNaN(val) ? 'math.nan' : val.toString()
  if (typeof val === 'string') return `"${val}"`
  if (typeof val === 'object') return `{${Object.entries(val).map((entry) => `"${entry[0]}":${toPython(entry[1])}`).join(',')}}`
  throw new Error('invalid value')
}
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
export const serialize = (data: any): string => {
  const customReplacer = (k: string, v: any) => {
    if (!v || typeof v !== 'object') return v
    if (v instanceof UPat) return { __type: 'UPat', op: v.op, dtype: v.dtype, src: v._inSrc, arg: v.arg, name: v.name, allow_any_len: v.allowedLen === -1, location: v.location, custom_early_reject: v.customEarlyReject }
    if (v instanceof UOp) return { __type: 'UOp', op: v.op, dtype: v.dtype, src: v.src, arg: v.arg }
    if (v instanceof DType) return { __type: 'DType', priority: v.priority, itemsize: v.itemsize, name: v.name, fmt: v.fmt, count: v.count, _scalar: v._scalar }
    return v
  }
  return JSON.stringify(data, customReplacer)
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
    if (type === 'DType') return new DType({ priority: de(i.priority), itemsize: de(i.itemsize), name: de(i.name), fmt: de(i.fmt), count: de(i.count), scalar: de(i._scalar) })
    return Object.fromEntries(Object.entries(i).map(([k, v]) => [k, de(v)]))
  }

  return de(obj)
}

export const runPython = async (code: string, data?: any) => {
  code = `
import tinygrad as tiny
import math
import json
from dataclasses import asdict

def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)

def serialize(data):
    class CustomEncoder(json.JSONEncoder):
      def default(self, o):
          if isinstance(o,tiny.ops.UPat): return {"__type":"UPat","op":o.op,"dtype":o.dtype,"src":o.src,"arg":o.arg,"name":o.name,"allow_any_len":o.allowed_len == -1,"location":o.location,"custom_early_reject":o.custom_early_reject}
          if isinstance(o, tiny.ops.UOp): return {"__type": "UOp",'op': o.op, "dtype": o.dtype,"src":o.src,"arg":o.arg} 
          if isinstance(o, tiny.ops.DType): return {"__type": "DType", "priority": o.priority,"itemsize":o.itemsize,"name":o.name,"fmt":o.fmt,"count":o.count,"_scalar":o._scalar}
          return super().default(o)
    return json.dumps(data, cls=CustomEncoder)

def deserialize(data):
    obj = json.loads(data)
    
    def de(item):
        if isinstance(item, dict):
            type = item.get("__type")
            def get(key,should_tuple=False): 
                res = de(item.get(key))
                return tuple(res) if should_tuple and res and isinstance(res,list) else res
            if type=="UPat": return tiny.ops.UPat(get("op",True),get("dtype",True),get("src"),get("arg"),get("name"),get("allow_any_len"),get("location"),get("custom_early_reject"))
            if type == "UOp": return tiny.ops.UOp(get('op',True),get('dtype',True),get('src',True),get('arg',True)) 
            elif type == "DType": return tiny.dtype.DType(get('priority'),get("itemsize"),get('name'),get('fmt'),get("count"),get("_scalar"))
            return {key: de(value) for key, value in item.items()}
        elif isinstance(item, list): return [de(element) for element in item]
        return item 

    return de(obj)

data = deserialize('${serialize(data)}')
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
export const tiny = async (strings: TemplateStringsArray, ...values: any[]): Promise<any> => {
  const code = String.raw({ raw: strings }, ...values.map((x) => toPython(x)))
  return await runPython(code)
}

export const tinyTest = <T extends any[]>(name: string, inputs: T[], fn: (...args: T) => any, python: (...args: T) => Promise<string>) => {
  Deno.test(name, async () => {
    for (const input of inputs) {
      equal(fn(...input), await python(...input))
    }
  })
}
