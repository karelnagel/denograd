import { walk } from 'jsr:@std/fs/walk'
import { Ops } from './src/ops.ts'
import { DType, dtypes, PtrDType } from './src/dtype.ts'
import { range } from './src/helpers.ts'

DType
dtypes
PtrDType

const run = (path: string) => {
  let data = Deno.readTextFileSync(path)

  const dts = [
    'new DType(-1, 0, `void`, undefined, 1, undefined)',
    'new DType(0, 1, `bool`, `?`, 1, undefined)',
    'new DType(1, 1, `signed char`, `b`, 1, undefined)',
    'new DType(2, 1, `unsigned char`, `B`, 1, undefined)',
    'new DType(3, 2, `short`, `h`, 1, undefined)',
    'new DType(4, 2, `unsigned short`, `H`, 1, undefined)',
    'new DType(5, 4, `int`, `i`, 1, undefined)',
    'new DType(6, 4, `unsigned int`, `I`, 1, undefined)',
    'new DType(7, 8, `long`, `q`, 1, undefined)',
    'new DType(8, 8, `unsigned long`, `Q`, 1, undefined)',
    'new DType(9, 2, `half`, `e`, 1, undefined)',
    'new DType(10, 2, `__bf16`, undefined, 1, undefined)',
    'new DType(11, 4, `float`, `f`, 1, undefined)',
    'new DType(12, 8, `double`, `d`, 1, undefined)',
  ]

  for (const dt of dts) {
    data = data.replaceAll(dt, eval(dt).toString())
  }

  const vecs = range(100).flatMap((i) => [
    `new DType(5, ${4 * i}, \`int${i}\`, undefined, ${i}, dtypes.int)`,
    `new DType(11, ${4 * i}, \`float${i}\`, undefined, ${i}, dtypes.float)`,
  ])

  for (const dt of vecs) {
    data = data.replaceAll(dt, eval(dt).toString())
  }

  const ptrs = data.matchAll(/new PtrDType\((.*?)1\)/g)
  for (const ptr of ptrs) {
    const dt = ptr[0]
    data = data.replaceAll(dt, eval(dt).toString())
  }
  // const ptrs = [
  //   "new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1)",
  //   "new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, true, 1)",
  //   "new PtrDType(5, 4, `int`, `i`, 1, undefined, dtypes.int, false, 1)",
  //   "new PtrDType(5, 4, `int`, `i`, 1, undefined, dtypes.int, true, 1)",
  //   "new PtrDType(6, 4, `unsigned int`, `I`, 1, undefined, dtypes.uint, false, 1)",
  //   "new PtrDType(6, 4, `unsigned int`, `I`, 1, undefined, dtypes.uint, true, 1)",
  //   "new PtrDType(0, 1, `bool`, `?`, 1, undefined, dtypes.bool, false, 1)",
  //   "new PtrDType(0, 1, `bool`, `?`, 1, undefined, dtypes.bool, true, 1)",
  //   "new PtrDType(-1, 0, `void`, undefined, 1, undefined, dtypes.void, false, 1)",
  //   "new PtrDType(-1, 0, `void`, undefined, 1, undefined, dtypes.void, true, 1)"
  // ]

  // for (const dt of ptrs) {
  //   data = data.replaceAll(dt, eval(dt).toString())
  // }

  // data = data.replaceAll('new DType(-1, 0, `void`, undefined, 1, undefined)', 'dtypes.void')
  // data = data.replaceAll('new DType(11, 4, `float`, `f`, 1, undefined)', 'dtypes.float')
  // data = data.replaceAll('new DType(0, 1, `bool`, `?`, 1, undefined)', 'dtypes.bool')
  // data = data.replaceAll('new DType(5, 4, `int`, `i`, 1, undefined)', 'dtypes.int')
  // data = data.replaceAll('new DType(6, 4, `unsigned int`, `I`, 1, undefined)', 'dtypes.uint')
  // data = data.replaceAll('new DType(11, 16, `float4`, undefined, 4, dtypes.float)', 'dtypes.half')
  // 'new DType(2, 1, `unsigned char`, `B`, 1, undefined)'

  // // data = data.replaceAll("new DType(5, 128, `int32`, undefined, 32, dtypes.int)","dtypes.int")
  // data = data.replaceAll('new DType(5, 16, `int4`, undefined, 4, dtypes.int)', 'dtypes.int.vec(4)')
  // data = data.replaceAll('new DType(5, 12, `int3`, undefined, 3, dtypes.int)', 'dtypes.int.vec(3)')

  Deno.writeTextFileSync(path, data)
}
for await (const dirEntry of walk('.')) {
  if (dirEntry.isFile && dirEntry.path.endsWith('.ts') && !dirEntry.path.includes('replace-enums')) run(dirEntry.path)
}
