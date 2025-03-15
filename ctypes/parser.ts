import data from './webgpu.json' with { type: 'json' }

let content = `import * as c from './mod.ts'\nexport * from './mod.ts'\n\n`

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

const rename = (name: string) => {
  if (name.startsWith('WGPU')) return name.slice(4)
  if (name.startsWith('wgpu')) return name[4].toLowerCase() + name.slice(5)
  return name
}
// TODO: all enums are consts are handled as uint32
const consts: string[] = []
for (const line of data) {
  if (line.tag !== 'const') continue
  consts.push(`export const ${rename(line.name)} = c.U32.new(${line.value})`)
}

const enums: string[] = []
for (const line of data) {
  if (line.tag !== 'enum') continue
  enums.push(`export class ${rename(line.name)} extends c.U32 {
  ${line.fields.map((x: any) => `static '${x.name.replace(`${line.name}_`, '')}' = ${rename(line.name)}.new(${x.value})`).join('\n  ')}
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
  if (type.tag === ':pointer') return type.type ? `c.Pointer<${getType(type.type)}>` : `c.Pointer<any>`
  else if (type.tag in typeMap) return typeMap[type.tag as keyof typeof typeMap]
  else if ([':struct', 'struct'].includes(type.tag) && type.name) return rename(type.name)
  if (type.tag === 'struct') console.log(type)
  if (type.tag.startsWith('WGPU')) return rename(type.tag)
  throw new Error(`Invalid type ${JSON.stringify(type)}`)
}
const structs: Record<string, string> = {}
for (const line of data) {
  if (line.tag !== 'struct') continue
  const byteLength = line['bit-size'] / 8
  const alignment = line['bit-alignment'] / 8
  const type = line.fields.length ? `{ ${line.fields.map((x: any) => `${x.name}: ${getType(x.type)}`).join('; ')} }` : `{}`
  const fields = line.fields.map((x: any) => `get $${rename(x.name)}(){ return new ${getType(x.type)}(this.buffer, this.offset + ${x['bit-offset'] / 8}) }`).join('\n  ')
  const _valueFn = !line.fields.length ? undefined : `protected override _value = () => ({${line.fields.map((x: any) => `${rename(x.name)}: this.$${rename(x.name)}`).join(', ')}})`
  const newFn = `static new = (val: Partial<${type}>) => new ${rename(line.name)}().set(val)`
  structs[line.name] = `export class ${rename(line.name)} extends c.Struct<${type}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ${byteLength}, ${alignment})
  }
  ${[fields, _valueFn, newFn].filter(Boolean).join('\n  ')}
}`
}

const types: string[] = []
for (const line of data) {
  if (line.tag !== 'typedef' || !line.name?.startsWith('WGPU')) continue
  if ([':struct', ':enum', ':function-pointer'].includes(line.type.tag)) continue
  types.push(`export class ${rename(line.name)} extends ${getType(line.type)} {}`)
}

const callbacks: string[] = []
const header = await Deno.readTextFile('ctypes/webgpu.h')
const getCallbackParameters = (fnName: string): { name: string; type: Type }[] => {
  const txt = header.split(`(*${fnName})(`)[1].split(`) WGPU_FUNCTION_ATTRIBUTE;`)[0]
  const out: { name: string; type: Type }[] = []
  for (let arg of txt.split(', ').map((x) => x.trim())) {
    const splits = arg.split(' ')
    if (splits.length === 1) continue

    const name = splits.at(-1)!
    const typeStr = splits.slice(0, -1).join(' ').replace('WGPU_NULLABLE ', '').replace('struct ', '')
    let type: Type
    if (!typeStr || typeStr.includes('void')) type = { tag: ':pointer' }
    else if (typeStr.endsWith('const *')) type = { tag: ':pointer', type: { tag: typeStr.replace('const *', '').trim() } }
    else if (typeStr.endsWith(' *')) type = { tag: ':pointer', type: { tag: typeStr.replace(' *', '').trim() } }
    else if (['float'].includes(typeStr)) type = { tag: `:${typeStr}` }
    else type = { tag: typeStr }
    out.push({ name, type })
  }
  return out
}
for (const line of data) {
  if (line.tag !== 'typedef' || line.type.tag !== ':function-pointer') continue
  const parameters = getCallbackParameters(line.name)
  const types = parameters.map((x) => `${x.name}: ${getType(x.type)}`).join(', ')
  callbacks.push(`export class ${rename(line.name)} extends c.Function<[${types}]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, [${parameters.map((x) => getLibType(x.type)).join(', ')}])
  }
  protected override _fn = (fn: (${types}) => void) => (${parameters.map((x) => `${x.name}: any`).join(', ')}) => void fn(${parameters.map((x) => `new ${getType(x.type)}().setNative(${x.name})`).join(', ')})
  static new = (fn: (${types}) => void) => new ${rename(line.name)}().set(fn)
}`)
}

const functions: string[] = []
for (const line of data) {
  if (line.tag !== 'function') continue
  const ret = line['return-type']
  functions.push(`export const ${rename(line.name)} = (${line.parameters.map((x: any) => `${x.name}: ${getType(x.type)}`).join(', ')}): ${getType(ret)} => new ${getType(ret)}().setNative(lib.symbols.${line.name}(${line.parameters.map((x: any) => `${x.name}.native`).join(', ')}))`)
}

content += Object.entries({ consts, enums, structs: Object.values(structs), types, callbacks, functions })
  .map(([k, v]) => `// ${k}\n${v.join('\n')}`).join('\n\n')
content += '\n'

Deno.writeTextFile('ctypes/dawn.ts', content)
