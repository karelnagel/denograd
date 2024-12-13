import * as dt from '../src/dtype.ts'
import { dtypes, promoLattice } from '../src/dtype.ts'
import { sorted } from '../src/helpers.ts'
import { compare, tryCatch } from './helpers.ts'
// TODO check if only created once

Deno.test(
  'DType.init',
  compare(
    [
      [{ priority: 1, itemsize: 1, name: 'bool', fmt: 'f', count: 1, _scalar: undefined }],
      [{ priority: 2, itemsize: 1, name: 'void', fmt: 'e', count: 1, _scalar: undefined }],
      [{ priority: 2, itemsize: 1, name: 'unsigned short', fmt: 'e', count: 2, _scalar: new dt.DType({ count: 1, itemsize: 4, name: 'void', priority: 4 }) }],
    ],
    (args: dt.DTypeArgs) => new dt.DType(args),
    'out(tiny.dtype.DType(*(data[0][key] for key in data[0])))',
  ),
)
Deno.test(
  'DType.vec',
  compare(
    [
      [dtypes.int, 4],
      [dtypes.float, 1],
      [dtypes.float, 6],
      [dtypes.bool, 6],
    ],
    (d: dt.DType, sz: number) => d.vec(sz),
    'out(data[0].vec(data[1]))',
  ),
)
Deno.test(
  'DType.ptr',
  compare(
    [
      [dtypes.int, true],
      [dtypes.float, false],
      [dtypes.float, true],
      [dtypes.bool, false],
    ],
    (d: dt.DType, local: boolean) => d.ptr(local),
    'out(data[0].ptr(data[1]))',
  ),
)

Deno.test(
  'dtypes.finfo',
  compare(
    [
      [dtypes.int],
      [dtypes.float],
      [dtypes.float16],
      [dtypes.bool],
      [dtypes.half],
      [dtypes.imagef(2, 2)],
      [dtypes.imageh(552, 0)],
    ],
    tryCatch(dtypes.finfo),
    'out(trycatch(lambda: tiny.dtype.dtypes.finfo(*data)))',
    { ignore: [5, 6] },
  ),
)
Deno.test(
  'dtypes.min',
  compare(
    [
      [dtypes.int],
      [dtypes.float],
      [dtypes.float16],
      [dtypes.bool],
      [dtypes.half],
      [dtypes.imagef(2, 2)],
    ],
    dtypes.min,
    'out(tiny.dtype.dtypes.min(*data))',
  ),
)
Deno.test(
  'dtypes.max',
  compare(
    [
      [dtypes.int],
      [dtypes.float],
      [dtypes.float16],
      [dtypes.bool],
      [dtypes.half],
      [dtypes.imagef(2, 2)],
    ],
    dtypes.max,
    'out(tiny.dtype.dtypes.max(*data))',
  ),
)

Deno.test(
  'fromJS',
  compare(
    [
      [4],
      [true],
      [4.4],
      [[4, 4, 2, 5, 3]],
      [[true, false, true, 4, 5.5]],
      [[4, 5.4]],
      [[false, 4]],
      [[3.3, 5]],
    ],
    dt.dtypes.from_js,
    'out(tiny.dtype.dtypes.from_py(*data))',
  ),
)

Deno.test(
  'asConst',
  compare(
    [
      [4, dtypes.int],
      [4.1, dtypes.int],
      [4.9, dtypes.int],
      [4, dtypes.float],
      [4, dtypes.bool],
      [[true, 4, 4.4], dtypes.float.vec(3)],
      [[true], dtypes.float.vec(3)],
    ],
    tryCatch(dt.dtypes.as_const),
    'out(trycatch(lambda:tiny.dtype.dtypes.as_const(*data)))',
  ),
)
Deno.test('promoLattice', compare([[]], () => [...promoLattice.entries()], 'out([[key,tiny.dtype.promo_lattice[key]] for key in tiny.dtype.promo_lattice])'))

Deno.test(
  '_getRecursiveParents',
  compare(
    [['float64'], ['float32'], ['float16'], ['half'], ['bool'], ['int'], ['uint']] as const,
    (type) => sorted(dt._getRecursiveParents(dtypes[type])),
    'out(sorted(tiny.dtype._get_recursive_parents(tiny.dtype.DTYPES_DICT[data[0]])))',
  ),
)

Deno.test(
  'leastUpperDType',
  compare(
    [
      ['int', 'int'],
      ['int', 'uint', 'long'],
      ['float64', 'float32'],
      ['float64', 'half'],
      ['bool', 'half'],
      ['int', 'int32', 'int64'],
      ['bool', 'float64', 'int64'],
      ['bool'],
      ['bool', 'uint'],
      ['bool', 'int'],
      ['bool', 'float'],
      ['half', 'uint'],
      ['half', 'int'],
      ['half', 'float'],
    ] as const,
    (...inputs) => dt.least_upper_dtype(...inputs.map((i) => dtypes[i])),
    'out(tiny.dtype.least_upper_dtype(*[tiny.dtype.DTYPES_DICT[key] for key in data]))',
  ),
)

Deno.test(
  'sumAccDType',
  compare(
    [
      ['float64'],
      ['float32'],
      ['half'],
      ['bool'],
      ['int'],
      ['uint'],
    ] as const,
    (input) => dt.sum_acc_dtype(dtypes[input]),
    'out(tiny.dtype.sum_acc_dtype(tiny.dtype.DTYPES_DICT[data[0]]))',
  ),
)

Deno.test(
  'truncate',
  compare(
    [
      [dtypes.bool, true],
      [dtypes.bool, false],
      [dtypes.float16, 4.4],
      [dtypes.float32, 4.4],
      [dtypes.float64, 4.4],
      [dtypes.uint8, 4],
      [dtypes.uint16, 4],
      [dtypes.uint32, 4],
      [dtypes.uint64, 4n],
      [dtypes.int8, 4],
      [dtypes.int16, 4],
      [dtypes.int32, 4],
      [dtypes.int64, 4n],
    ],
    (d: dt.DType, x: any) => dt.truncate.get(d)!(x),
    'out(tiny.dtype.truncate[data[0]](data[1]))',
    { ignore: [8, 12] },
  ),
)
