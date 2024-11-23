import { equal } from 'assert'
import { exec } from 'node:child_process'
import process from 'node:process'

export const execAsync = (cmd: string, opt?: any) => new Promise<string>((res, rej) => exec(cmd, opt, (error, stdout, stderr) => error || stderr ? rej(error) : res(stdout as any as string)))
export const python = async (code: string) => JSON.parse((await execAsync(`cd ${process.cwd()}/tinygrad && python3 -c '${code}'`)).trim())

/**
 * ```ts
 * import {expect} from "expect"
 * expect(toPython("hello")).toBe('"hello"')
 * expect(toPython({a:{b:true,c:3,x:[4,3,2]}})).toBe('{"a":{"b":True,"c":3,"x":[4,3,2]}}')
 * ```
 */
export const toPython = (val: any): string => {
  if (Array.isArray(val)) return `[${val.map((x) => toPython(x))}]`
  if (val === null || typeof val === 'undefined') return 'None'
  if (typeof val === 'boolean') return val ? 'True' : 'False'
  if (typeof val === 'number') return val === Infinity ? 'math.inf' : Number.isNaN(val) ? 'math.nan' : val.toString()
  if (typeof val === 'string') return `"${val}"`
  if (typeof val === 'object') return `{${Object.entries(val).map((entry) => `"${entry[0]}":${toPython(entry[1])}`).join(',')}}`
  throw new Error('invalid value')
}
export const asdict = (o: any): any => !o ?o:Object.fromEntries(Object.entries(o).filter((o) => typeof o[1] !== 'function').map(([k, v]) => typeof v === 'object' ? [k, asdict(v)] : [k, v]))

export const trycatch = <T>(fn: () => T): T | string => {
  try {
    return fn()
  } catch (e) {
    if (e instanceof Error) return e.message
    else return 'error'
  }
}
export const tiny = async (strings: TemplateStringsArray, ...values: any[]): Promise<any> => {
  const code = `
import tinygrad as tiny
import math
import json
def trycatch(fn):
  try: return fn()
  except Exception as e: return str(e)
def out(o):
    print(json.dumps(o))

${String.raw({ raw: strings }, ...values.map((x) => toPython(x)))}
`
  return await python(code)
}

export const tinyTest = <T extends any[]>(name: string, inputs: T[], fn: (...args: T) => any, python: (...args: T) => Promise<string>) => {
  Deno.test(name, async () => {
    for (const input of inputs) {
      equal(fn(...input), await python(...input))
    }
  })
}
