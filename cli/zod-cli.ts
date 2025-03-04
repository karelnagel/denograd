import type { z } from 'zod'
import { colored } from '../denograd/helpers.ts'
import { env } from '../denograd/mod.ts'

const getZodTypeString = (zod: any): string => {
  switch (zod._def.typeName) {
    case 'ZodString':
      return 'string'
    case 'ZodNumber':
      return 'number'
    case 'ZodBoolean':
      return 'boolean'
    case 'ZodEnum':
      return zod._def.values.map((v: string) => v).join(' | ')
    default:
      return getZodTypeString(zod._def.innerType)
  }
}

const help = (schema: z.ZodObject<z.ZodRawShape>): string => {
  let res = `${schema.description || 'help:'}\n\noptions:\n`
  const lines: string[][] = []
  for (const [k, v] of Object.entries(schema.shape)) {
    lines.push([colored(`--${k}`, 'blue'), colored(getZodTypeString(v), 'green'), `${v.description || ''}${v._def.defaultValue ? colored(` (default: ${v._def.defaultValue()})`, 'yellow') : ''}`])
  }
  const maxLengths = lines[0].map((_, i) => Math.max(...lines.map((line) => line[i].length)))
  res += lines.map((line) => line.map((line, i) => line.padEnd(maxLengths[i] + 2)).join('')).join('\n') + '\n'
  return res
}

export const parseArgs = <T extends z.ZodRawShape>(schema: z.ZodObject<T>): z.infer<z.ZodObject<T>> => {
  const args = env.args().join(' ').split('--').filter(Boolean)
  const obj: Record<string, unknown> = {}
  for (const arg of args) {
    const [key, value] = arg.split(/[ |=]/)
    if (value === 'true' || value === '' || value === undefined) obj[key] = true
    else if (value === 'false') obj[key] = false
    else if (!isNaN(Number(value))) obj[key] = Number(value)
    else obj[key] = value
  }
  if (obj.help) {
    console.log(help(schema))
    throw new Error()
  }
  const res = schema.strict().safeParse(obj)
  if (res.success) return res.data
  console.log(res.error.issues.map((x) => colored(`Error with '${x.path.join('.')}': ${x.message}`, 'red')).join('\n') + '\n\n' + help(schema))
  throw new Error()
}
