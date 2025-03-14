import data from './webgpu.json' with { type: 'json' }

let content = `import * as c from './mod.ts'\n\n`

const libTypeMap = {
  ':void': 'void',
  ':pointer': 'pointer',
  'size_t': 'usize',
  'uint8_t': 'u8',
  'uint32_t': 'u32',
  'uint64_t': 'u64',
  'int32_t': 'i32',
  'int64_t': 'i64',
  ':float': 'f32',
  ':double': 'f64',
  ':struct': 'buffer',
  ':enum': 'u32',
  ':function-pointer': 'function',
}
const getLibType = (type: Type): string => {
  if (type.tag in libTypeMap) return `'${libTypeMap[type.tag as keyof typeof libTypeMap]}'`

  const lines = data.filter((x) => x.name === type.tag && x.tag === 'typedef')
  if (lines.length !== 1) throw new Error(`Can't find ${type.tag}`)
  return getLibType(lines[0].type)
}

content += `const lib = Deno.dlopen('/opt/homebrew/Cellar/dawn/0.1.6/lib/libwebgpu_dawn.dylib', {
  ${data.filter((x) => x.tag === 'function').map((x) => `${x.name}: { parameters: [${x.parameters.map((x: any) => getLibType(x.type)).join(', ')}], result: ${getLibType(x['return-type'])} },`).join('\n  ')}
})\n\n`

// TODO: all enums are consts are handled as uint32
const consts: string[] = []
for (const line of data) {
  if (line.tag !== 'const') continue
  consts.push(`export const ${line.name} = new c.U32(${line.value})`)
}

const enums: string[] = []
for (const line of data) {
  if (line.tag !== 'enum') continue
  enums.push(`export class ${line.name} extends c.U32 {
  ${line.fields.map((x: any) => `static '${x.name.replace(`${line.name}_`, '')}' = new ${line.name}(${x.value})`).join('\n  ')}
}`)
}

type Type = { tag: string; type?: Type; name?: string }
const typeMap = {
  'uint8_t': 'c.U8',
  'uint16_t': 'c.U16',
  'uint32_t': 'c.U32',
  'uint64_t': 'c.U64',
  'int32_t': 'c.I32',
  'int64_t': 'c.I64',
  'size_t': 'c.Size',
  ':int': 'c.I32',
  ':char': 'c.U8',
  ':float': 'c.F32',
  ':double': 'c.F64',
  ':void': 'c.Void',
}
const getType = (type: Type): string => {
  if (type.tag === ':pointer' && type.type) return `c.Pointer<${getType(type.type)}>`
  else if (type.tag in typeMap) return typeMap[type.tag as keyof typeof typeMap]
  else if ([':struct', 'struct'].includes(type.tag) && type.name) return type.name
  else if (type.tag === ':function-pointer') return `c.Function`
  if (type.tag === 'struct') console.log(type)
  if (type.tag.startsWith('WGPU')) return type.tag
  throw new Error(`Invalid type ${JSON.stringify(type)}`)
}
const structs: Record<string, string> = {}
for (const line of data) {
  if (line.tag !== 'struct') continue
  structs[line.name] = `export class ${line.name} extends c.Struct<[${line.fields.map((x: any) => `${x.name}: ${getType(x.type)}`).join(', ')}]> {}`
}

const types: string[] = []
for (const line of data) {
  if (line.tag !== 'typedef' || !line.name?.startsWith('WGPU')) continue
  if ([':struct', ':enum'].includes(line.type.tag)) continue
  types.push(`export class ${line.name} extends ${getType(line.type)} {}`)
}

const replaceName = (name: string) => name[4].toLowerCase() + name.slice(5)
const functions: string[] = []
for (const line of data) {
  if (line.tag !== 'function') continue
  const ret = line['return-type']
  functions.push(`export const ${replaceName(line.name)} = (${line.parameters.map((x: any) => `${x.name}: ${getType(x.type)}`).join(', ')}): ${getType(ret)} => new ${getType(ret)}(lib.symbols.${line.name}(${line.parameters.map((x: any) => `${x.name}.value`).join(', ')}))`)
}

content += Object.entries({ consts, enums, structs: Object.values(structs), types, functions })
  .map(([k, v]) => `// ${k}\n${v.join('\n')}`).join('\n\n')
content += '\n'

Deno.writeTextFile('ctypes/dawn.ts', content)
