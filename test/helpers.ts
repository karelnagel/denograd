import { assertEquals } from 'jsr:@std/assert@^1.0.8'
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
  if (typeof val === 'undefined') return 'None'
  if (typeof val === 'boolean') return val ? 'True' : 'False'
  if (typeof val === 'number') return val === Infinity ? 'math.inf' : Number.isNaN(val) ? 'math.nan' : val.toString()
  if (typeof val === 'string') return `"${val}"`
  if (typeof val === 'object') return `{${Object.entries(val).map((entry) => `"${entry[0]}":${toPython(entry[1])}`).join(',')}}`
  throw new Error('invalid value')
}
export const tiny = async (strings: TemplateStringsArray, ...values: any[]): Promise<any> => {
  const code = `
import tinygrad as tiny
import math
import json
def out(o):
    print(json.dumps(o))

${String.raw({ raw: strings }, ...values.map((x) => toPython(x)))}
`
  return await python(code)
}

export const tinyTest = <T extends any[]>(name: string, inputs: T[], fn: (...args: T) => any, python: (...args: T) => Promise<string>) => {
  Deno.test(name, async () => {
    for (const input of inputs) {
      assertEquals(fn(...input), await python(...input))
    }
  })
}
