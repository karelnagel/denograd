import { walk } from 'jsr:@std/fs/walk'
import { Ops } from './src/ops.ts'

const run = (path: string) => {
  let data = Deno.readTextFileSync(path)
  for (const op of Ops.values()) {
    console.log(op)
    data = data.replaceAll(`new UOp(${op.value},`, `new UOp(${op.toString()},`)
    data = data.replaceAll(`new UPat(${op.value},`, `new UPat(${op.toString()},`)
  }
  Deno.writeTextFileSync(path, data)
}
console.log('eh')
for await (const dirEntry of walk('.')) {
  if (dirEntry.isFile && dirEntry.path.endsWith('.ts')) run(dirEntry.path)
}
