import { expect } from 'expect/expect'
import { _get_recursive_parents, DType, dtypes, least_upper_dtype, promoLattice, sum_acc_dtype, truncate } from '../denograd/dtype.ts'
import { compare, test, tryCatch } from './helpers.ts'
import { PtrDType } from '../denograd/dtype.ts'

// test(
//   'DType.init',
//   compare(
//     [
//       [1, 1, 'bool', 'f', 1, undefined],
//       [2, 1, 'void', 'e', 1, undefined],
//       [2, 1, 'unsigned short', 'e', 2, new DType(4, 4, 'void', undefined, 1)],
//     ],
//     (priority: number, itemsize: number, name: string, fmt: undefined | FmtStr, count: number, _scalar?: DType) => new DType(priority, itemsize, name, fmt, count, _scalar),
//     'out(tiny.dtype.DType(*data))',
//   ),
// )
test(
  'DType.vec',
  compare(
    [
      [dtypes.int, 4],
      [dtypes.float, 1],
      [dtypes.float, 6],
      [dtypes.bool, 6],
    ],
    (d: DType, sz: number) => d.vec(sz),
    'out(data[0].vec(data[1]))',
  ),
)
test(
  'DType.ptr',
  compare(
    [
      [dtypes.int, 4, true],
      [dtypes.float, 3, false],
      [dtypes.float, 4, true],
      [dtypes.bool, 1, false],
    ],
    (d: DType, size: number, local: boolean) => d.ptr(size, local),
    'out(data[0].ptr(data[1],data[2]))',
  ),
)

test(
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
test(
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
test(
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

test(
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
    dtypes.from_js,
    'out(tiny.dtype.dtypes.from_py(*data))',
  ),
)

test(
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
    tryCatch(dtypes.as_const),
    'out(trycatch(lambda:tiny.dtype.dtypes.as_const(*data)))',
  ),
)
test('promoLattice', compare([[]], () => [...promoLattice.entries()], 'out([[key,tiny.dtype.promo_lattice[key]] for key in tiny.dtype.promo_lattice])'))

test(
  '_getRecursiveParents',
  compare(
    [['float64'], ['float32'], ['float16'], ['half'], ['bool'], ['int'], ['uint']] as const,
    (type) => _get_recursive_parents(dtypes[type]).map((x) => x.toString()).toSorted(),
    'out(sorted([str(x) for x in tiny.dtype._get_recursive_parents(tiny.dtype.DTYPES_DICT[data[0]])]))',
  ),
)

test(
  'least_upper_dtype',
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
    (...inputs) => least_upper_dtype(...inputs.map((i) => dtypes[i])),
    'out(tiny.dtype.least_upper_dtype(*[tiny.dtype.DTYPES_DICT[key] for key in data]))',
  ),
)

test(
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
    (input) => sum_acc_dtype(dtypes[input]),
    'out(tiny.dtype.sum_acc_dtype(tiny.dtype.DTYPES_DICT[data[0]]))',
  ),
)

test(
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
    (d: DType, x: any) => truncate.get(d)!(x),
    'out(tiny.dtype.truncate[data[0]](data[1]))',
    { ignore: [8, 12] },
  ),
)

test('DType.dataclass', () => {
  expect(new DType(3, 3, 'sdf', '?', 3, undefined)).toBe(new DType(3, 3, 'sdf', '?', 3, undefined))
  expect(dtypes.int16).toBe(DType.new(3, 2, 'short', 'h'))
  expect(dtypes.int16.ptr(4, true)).toBe(new PtrDType(3, 2, 'short', 'h', 1, undefined, new DType(3, 2, 'short', 'h', 1), true, 1, 4))
})
