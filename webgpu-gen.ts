type Struct = { name: string; args: { type: string; name: string }[] }
type Function = { name: string; args: { type: string; name: string }[]; out: string }
type Enum = { name: string; args: { name: string; value: string }[] }
type Const = { name: string; value: string }
type Data = { structs: Struct[]; functions: Function[]; enums: Enum[]; consts: Const[] }

const parseHeaderFile = (content: string): Data => {
  const data: Data = { structs: [], functions: [], enums: [], consts: [] }

  const lines = content.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()

    if (line.startsWith('typedef struct')) {
      const match = line.match(/typedef struct (\w+) \{/)
      if (!match) continue
      let def: Struct | undefined = { name: match[1], args: [] }
      while (i + 1 < lines.length) {
        i++
        if (lines[i].includes('}')) break
        const splits = lines[i].replace(';', '').trim().split(' ')
        def.args.push({ type: splits.slice(0, -1).join(' '), name: splits.at(-1)! })
      }
      data.structs.push(def)
    } else if (line.startsWith('WGPU_EXPORT')) {
      const match = line.match(/WGPU_EXPORT\s+([\w*\s]+)\s+(\w+)\((.*)\)/)
      if (!match) continue
      const [, returnType, name, argsStr] = match
      const args = []
      if (argsStr && argsStr.trim() !== 'void') {
        const argParts = argsStr.split(',').map((part) => part.trim())
        for (const part of argParts) {
          const splits = part.split(' ')
          args.push({ type: splits.slice(0, -1).join(' '), name: splits.at(-1)! })
        }
      }
      data.functions.push({ name, args, out: returnType })
    } else if (line.startsWith('typedef enum')) {
      const match = line.match(/typedef enum (\w+)/)
      if (!match) continue
      const name = match[1]
      const def: Enum = { name, args: [] }
      i++
      while (i < lines.length && !lines[i].includes('}')) {
        const enumLine = lines[i].trim()
        const valueMatch = enumLine.match(/(\w+)\s*=\s*([^,]+),?/)
        if (valueMatch) {
          def.args.push({ name: valueMatch[1].replace(`${name}_`, ''), value: valueMatch[2].trim() })
        }
        i++
      }
      data.enums.push(def)
    } else if (line.startsWith('static const')) {
      const match = line.match(/static const\s+([\w\s]+)\s+(\w+)\s*=\s*([^;]+);/)
      if (!match) continue
      data.consts.push({ name: match[2], value: match[3].trim() })
    }
  }

  return data
}

const getType = (type: string) => {
  const suffix = type.includes('*') || type.includes('WGPU_NULLABLE') ? ' | null' : ''
  type = type.replaceAll('*', '').replaceAll(' const', '').replaceAll('WGPU_NULLABLE', '').trim()
  if (['size_t', 'uint32_t', 'float', 'int', 'double', 'uint16_t', 'int32_t', 'char'].includes(type)) return 'number' + suffix
  if (['uint64_t', 'int64_t'].includes(type)) return 'bigint' + suffix
  if (type === 'void') return 'null'
  if (type === 'const char') return 'string' + suffix
  if (type === 'WGPUBool') return 'boolean' + suffix
  return type + suffix
}

const typeMap: Record<string, string> = {
  'void': 'void',
  'uint8_t': 'u8',
  'uint32_t': 'u32',
  'uint64_t': 'u64',
  'int32_t': 'i32',
  'int64_t': 'i64',
  'usize_t': 'usize',
  'size_t': 'isize',
}
const getDlType = (type: string) => {
  if (type in typeMap) return `'${typeMap[type]}'`

  return "'pointer'"
}

const generateCode = (path: string, { consts, enums, functions, structs }: Data) => {
  const CONSTS = consts.map((x) => `export const ${x.name} = ${x.value}`).join('\n')
  const ENUMS = enums.map((x) => `export enum ${x.name} {\n  ${x.args.map((a) => `'${a.name}' = ${a.value}`).join(',\n  ')}\n}`).join('\n')
  const STRUCTS = structs.map((x) => `export type ${x.name} = {\n  ${x.args.map((a) => `${a.name}: ${getType(a.type)}`).join('\n  ')}\n}`).join('\n')
  const DL = `export const webgpu = Deno.dlopen('${path}', {
  ${functions.map((x) => `${x.name}: { parameters: [${x.args.map((x) => getDlType(x.type)).join(', ')}], result: ${getDlType(x.out)}  }`).join(',\n  ')}
})`
  const FUNCTIONS = functions.map((x) => `export function ${x.name}(${x.args.map((x) => `${x.name}:${getType(x.type)}`).join(', ')}): ${getType(x.out)} {\n return webgpu.symbols.${x.name}(${x.args.map(x=>x.name).join(", ")}) \n}`).join('\n')

  return Object.entries({ DL, CONSTS, ENUMS, STRUCTS, FUNCTIONS }).map(([k, v]) => `// ${k}\n${v}`).join('\n\n')
}
const content = await Deno.readTextFile('webgpu.h')
const data = parseHeaderFile(content)
const path = 'path'
const code = generateCode(path, data)
await Deno.writeTextFile('webgpu.ts', code)
