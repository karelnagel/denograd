import { walk } from 'jsr:@std/fs/walk'

const replace = (text: string, regex: RegExp, keys: string[], fn: (x: string) => string) => {
  return text.replace(regex, (_, args: string) => {
    const out: Record<string, string | undefined> = {}
    const getIndex = (key: string) => Math.max(args.indexOf(` ${key}:`), args.indexOf(` ${key},`))
    for (const key of keys.filter((x) => getIndex(x) !== -1).toSorted((a, b) => getIndex(b) - getIndex(a))) {
      ;[args, out[key]] = args.split(` ${key}:`)
      if (!out[key]) {
        args = args.split(` ${key},`)[0]
        out[key] = key
      }
    }
    if (args.trim() !== '') throw new Error(`Args is not empty: ${args}, ${keys}, ${text}`)
    const trim = (txt?: string) => txt?.trim()?.replace(/^,/, '').replace(/,$/, '') || 'undefined'
    let res = keys.map((key) => out[key])

    let lastIndex = res.length
    for (const item of res.toReversed()) {
      if (item !== undefined) break
      lastIndex--
    }

    res = res.slice(0, lastIndex)
    return fn(res.map((x) => trim(x)).join(', '))
  })
}

const run = (file: string) => {
  console.log(file)
  const data = Deno.readTextFileSync(file)
  let text = ''

  const chunks = []
  let current = ''
  for (const char of data.split('')) {
    current += char
    if (current.length === 5) {
      chunks.push(current)
      current = ''
    }
  }
  chunks.push(current)

  for (const chunk of chunks.toReversed()) {
    text = chunk + text

    text = replace(
      text,
      /new DType\(\{([\s\S]*?)\}\)/g,
      ['priority', 'itemsize', 'name', 'fmt', 'count', '_scalar'],
      (x) => `new DType(${x})`,
    )
    text = replace(
      text,
      /new PtrDType\(\{([\s\S]*?)\}\)/g,
      ['priority', 'itemsize', 'name', 'fmt', 'count', '_scalar', '_base', 'local', 'v'],
      (x) => `new PtrDType(${x})`,
    )
    text = replace(
      text,
      /new ImageDType\(\{([\s\S]*?)\}\)/g,
      ['priority', 'itemsize', 'name', 'fmt', 'count', '_scalar:', '_base', 'local', 'v', 'shape'],
      (x) => `new ImageDType(${x})`,
    )

    text = replace(
      text,
      /new UOp\(\{([\s\S]*?)\}\)/g,
      ['op', 'dtype', 'src', 'arg'],
      (x) => `new UOp(${x})`,
    )

    text = replace(
      text,
      /new UPat\(\{([\s\S]*?)\}\)/g,
      ['op', 'dtype', 'src', 'arg', 'name', 'allow_any_len', 'location', 'custom_early_reject'],
      (x) => `new UPat(${x})`,
    )
  }
  Deno.writeTextFileSync(file, text)
}

for await (const dirEntry of walk('.')) {
  if (dirEntry.isFile && dirEntry.path.endsWith(".ts")) run(dirEntry.path)
}
