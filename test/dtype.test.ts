import * as dt from '../src/dtype.ts'
import { dtypes, promoLattice } from '../src/dtype.ts'
import { sorted } from '../src/helpers.ts'
import { asdict, compare, python, tryCatch } from './helpers.ts'
import { expect } from 'expect'
// TODO check if only created once

Deno.test('create DType', async () => {
  const inputs: dt.DTypeArgs[] = [
    { priority: 1, itemsize: 1, name: 'bool', fmt: 'f', count: 1, _scalar: undefined },
    { priority: 2, itemsize: 1, name: 'void', fmt: 'e', count: 1, _scalar: undefined },
    { priority: 2, itemsize: 1, name: 'unsigned short', fmt: 'e', count: 2, _scalar: new dt.DType({ count: 1, itemsize: 4, name: 'void', priority: 4 }) },
  ]
  for (const args of inputs) {
    const ts = new dt.DType(args)
    const py = await python(
      `
dtype = tiny.dtype.DType(*(data[key] for key in data))
out({
    "dtype": asdict(dtype),
    "repr": dtype.__repr__(),
    "reduce": dtype.__reduce__()[1],
    "vcount": dtype.vcount,
    "base": asdict(dtype.base),
    "vec1": trycatch(lambda: asdict(dtype.vec(1))),
    "vec2": trycatch(lambda: asdict(dtype.vec(2))),
    "vec11": trycatch(lambda: asdict(dtype.vec(11))),
    "scalar": asdict(dtype.scalar()),
    "ptr": [{
        "asdict":asdict(ptr),
        "vcount":ptr.vcount,
        "repr":ptr.__repr__(),
        "vec1":trycatch(lambda: asdict(ptr.vec(1))),
        "vec2":trycatch(lambda: asdict(ptr.vec(2))),
        } for ptr in [dtype.ptr(False),dtype.ptr(True)]] ,
})`,
      args,
    )
    expect(asdict(ts)).toEqual(py.dtype)
    // expect(ts.reduce()[1]).toEqual(py.reduce)
    expect(ts.vcount).toEqual(py.vcount)
    expect(asdict(ts.base)).toEqual(py.base)
    expect(asdict(tryCatch(() => ts.vec(1))())).toEqual(py.vec1)
    expect(asdict(tryCatch(() => ts.vec(2))())).toEqual(py.vec2)
    expect(asdict(tryCatch(() => ts.vec(11))())).toEqual(py.vec11)
    expect(asdict(ts.scalar())).toEqual(py.scalar)

    for (const [i, ptr] of [ts.ptr(false), ts.ptr(true)].entries()) {
      expect(asdict(ptr)).toEqual(py.ptr[i].asdict)
      expect(ptr.vcount).toEqual(py.ptr[i].vcount)
      expect(ptr.toString()).toEqual(py.ptr[i].repr)
      expect(asdict(ptr.vec(1))).toEqual(py.ptr[i].vec1)
      expect(asdict(ptr.vec(2))).toEqual(py.ptr[i].vec2)
    }
  }
})

Deno.test('dtypes', async () => {
  const shouldBeFloats = [dtypes.float, dtypes.float32, dtypes.float64, dtypes.double, dtypes.half, dtypes.imagef(2, 3), dtypes.imageh(4, 4, 4)]

  shouldBeFloats.forEach((float) => expect(dtypes.isFloat(float)).toBe(true))
  ;[dtypes.int, dtypes.bool, dtypes.void, dtypes.int8].forEach((int) => expect(dtypes.isFloat(int)).toBe(false))
  ;[dtypes.uint8, dtypes.uchar, dtypes.ulong].forEach((uint) => expect(dtypes.isUnsigned(uint)).toBe(true))
  ;[dtypes.int16, dtypes.long, dtypes.float, dtypes.imagef(4, 4)].forEach((uint) => expect(dtypes.isUnsigned(uint)).toBe(false))
  ;[dtypes.int, dtypes.defaultInt, dtypes.uint, dtypes.uchar].forEach((int) => expect(dtypes.isInt(int)).toBe(true))
  ;[dtypes.float, dtypes.imagef(4, 4)].forEach((int) => expect(dtypes.isInt(int)).toBe(false))

  expect(asdict(dtypes.fields())).toEqual(await python(`out({k:asdict(v) for k,v in tiny.dtype.dtypes.fields().items()})`))
  expect(dt.INVERSE_DTYPES_DICT).toEqual(await python(`out(tiny.dtype.INVERSE_DTYPES_DICT)`))
})

Deno.test('dtypes.finfo', async () => {
  expect(dtypes.finfo(dtypes.float)).toEqual(await python(`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float))`))
  expect(dtypes.finfo(dtypes.float16)).toEqual(await python(`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float16))`))
  expect(dtypes.finfo(dtypes.half)).toEqual(await python(`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.half))`))
  expect(dtypes.finfo(dtypes.float32)).toEqual(await python(`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float32))`))
  expect(dtypes.finfo(dtypes.float64)).toEqual(await python(`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float64))`))

  expect(tryCatch(dtypes.finfo)(dtypes.int16)).toEqual(await python(`out(trycatch(lambda: tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.int16)))`))
  expect(tryCatch(dtypes.finfo)(dtypes.imagef(2, 2))).toEqual(`Invalid dtype ${await python(`out(trycatch(lambda: tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.imagef((2,2)))))`)} for finfo`)
})
Deno.test('dtypes.min/max', async () => {
  // Int
  expect(dtypes.min(dtypes.int)).toEqual(await python(`out(tiny.dtype.dtypes.min(tiny.dtype.dtypes.int))`))
  expect(dtypes.max(dtypes.int)).toEqual(await python(`out(tiny.dtype.dtypes.max(tiny.dtype.dtypes.int))`))

  // Uint
  expect(dtypes.min(dtypes.uint)).toEqual(await python(`out(tiny.dtype.dtypes.min(tiny.dtype.dtypes.uint))`))
  expect(dtypes.max(dtypes.uint)).toEqual(await python(`out(tiny.dtype.dtypes.max(tiny.dtype.dtypes.uint))`))

  // Float
  expect(dtypes.min(dtypes.float)).toEqual(-Infinity)
  expect(dtypes.max(dtypes.float)).toEqual(Infinity)

  // Bool
  expect(dtypes.min(dtypes.bool)).toEqual(await python(`out(tiny.dtype.dtypes.min(tiny.dtype.dtypes.bool))`))
  expect(dtypes.max(dtypes.bool)).toEqual(await python(`out(tiny.dtype.dtypes.max(tiny.dtype.dtypes.bool))`))
})

Deno.test(
  'fromJS',
  compare(
    [[4], [true], [4.4], [[4, 4, 2, 5, 3]], [[true, false, true, 4, 5.5]], [[4, 5.4]], [[false, 4]], [[3.3, 5]]],
    dt.dtypes.fromJS,
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
    tryCatch(dt.dtypes.asConst),
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
    (...inputs) => dt.leastUpperDType(...inputs.map((i) => dtypes[i])),
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
    (input) => dt.sumAccDType(dtypes[input]),
    'out(tiny.dtype.sum_acc_dtype(tiny.dtype.DTYPES_DICT[data[0]]))',
  ),
)
Deno.test('truncate', async (t) => {
  const truncate = async (dtype: string, val1: any, val2: any) => {
    await t.step(dtype, async () => {
      expect(dt.truncate(dtypes[dtype as keyof dtypes])(val1)).toEqual(await python(`out(tiny.dtype.truncate[tiny.dtype.dtypes.${dtype}](${val2}))`))
    })
  }
  await truncate('bool', true, 'True')
  await truncate('float16', 4.4, '4.4')
  await truncate('float32', 4.4, '4.4')
  await truncate('float64', 4.4, '4.4')
  await truncate('uint8', 4, '4')
  await truncate('uint16', 4, '4')
  await truncate('uint16', 4, '4')
  await truncate('uint32', 4, '4')
  // TODO fix these
  // await truncate('uint64', 4n, '4n')
  await truncate('int8', 4, '4')
  await truncate('int16', 4, '4')
  await truncate('int32', 4, '4')
  // await truncate('int64', 4n, '4n')
})
