import * as dt from '../src/dtype.ts'
import { dtypes } from '../src/dtype.ts'
import { asdict, runPython, tiny, trycatch } from './helpers.ts'
import { expect } from 'expect'
// TODO check if only created once

Deno.test('create DType', async () => {
    const inputs: dt.DTypeArgs[] = [
        { count: 1, fmt: 'f', itemsize: 1, name: 'bool', priority: 1, scalar: null },
        { count: 1, fmt: 'e', itemsize: 1, name: 'void', priority: 2, scalar: null },
        { count: 2, fmt: 'e', itemsize: 1, name: 'unsigned short', priority: 2, scalar: null },
    ]
    for (const args of inputs) {
        const dtype = new dt.DType(args)
        const out = await tiny`
dtype = tiny.dtype.DType(${args.priority},${args.itemsize},${args.name},${args.fmt},${args.count},${args.scalar})
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
})`
        expect(asdict(dtype)).toEqual(out.dtype)
        expect(dtype.reduce()[1]).toEqual(out.reduce)
        expect(dtype.vcount).toEqual(out.vcount)
        expect(asdict(dtype.base)).toEqual(out.base)
        expect(trycatch(() => asdict(dtype.vec(1)))).toEqual(out.vec1)
        expect(trycatch(() => asdict(dtype.vec(2)))).toEqual(out.vec2)
        expect(trycatch(() => asdict(dtype.vec(11)))).toEqual(out.vec11)
        expect(asdict(dtype.scalar())).toEqual(out.scalar)

        for (const [i, ptr] of [dtype.ptr(false), dtype.ptr(true)].entries()) {
            expect(asdict(ptr)).toEqual(out.ptr[i].asdict)
            expect(ptr.vcount).toEqual(out.ptr[i].vcount)
            expect(ptr.toString()).toEqual(out.ptr[i].repr)
            expect(trycatch(() => asdict(ptr.vec(1)))).toEqual(out.ptr[i].vec1)
            expect(trycatch(() => asdict(ptr.vec(2)))).toEqual(out.ptr[i].vec2)
        }
    }
})

Deno.test('dtypes', async () => {
    const shouldBeFloats = [dtypes.float, dtypes.float32, dtypes.float64, dtypes.double, dtypes.half, dtypes.imagef(2, 3), dtypes.imageh(4, 4, 4)]

    shouldBeFloats.forEach((float) => expect(dtypes.isFloat(float)).toBe(true))
    ;[dtypes.int, dtypes.bool, dtypes.void, dtypes.int8].forEach((int) => expect(dtypes.isFloat(int)).toBe(false))
    ;[dtypes.uint8, dtypes.uchar, dtypes.ulong].forEach((uint) => expect(dtypes.isUnsigned(uint)).toBe(true))
    ;[dtypes.int16, dtypes.long, dtypes.float, dtypes.imagef(4, 4)].forEach((uint) => expect(dtypes.isUnsigned(uint)).toBe(false))
    ;[dtypes.int, dtypes.default_int, dtypes.uint, dtypes.uchar].forEach((int) => expect(dtypes.isInt(int)).toBe(true))
    ;[dtypes.float, dtypes.imagef(4, 4)].forEach((int) => expect(dtypes.isInt(int)).toBe(false))

    expect(asdict(dtypes.fields())).toEqual(await tiny`out({k:asdict(v) for k,v in tiny.dtype.dtypes.fields().items()})`)
    expect(dt.INVERSE_DTYPES_DICT).toEqual(await tiny`out(tiny.dtype.INVERSE_DTYPES_DICT)`)
})

Deno.test('dtypes.finfo', async () => {
    expect(dtypes.finfo(dtypes.float)).toEqual(await tiny`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float))`)
    expect(dtypes.finfo(dtypes.float16)).toEqual(await tiny`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float16))`)
    expect(dtypes.finfo(dtypes.half)).toEqual(await tiny`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.half))`)
    expect(dtypes.finfo(dtypes.float32)).toEqual(await tiny`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float32))`)
    expect(dtypes.finfo(dtypes.float64)).toEqual(await tiny`out(tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.float64))`)

    expect(trycatch(() => dtypes.finfo(dtypes.int16))).toEqual(await tiny`out(trycatch(lambda: tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.int16)))`)
    expect(trycatch(() => dtypes.finfo(dtypes.imagef(2, 2)))).toEqual(`Invalid dtype ${await tiny`out(trycatch(lambda: tiny.dtype.dtypes.finfo(tiny.dtype.dtypes.imagef((2,2)))))`} for finfo`)
})
Deno.test('dtypes.min/max', async () => {
    // Int
    expect(dtypes.min(dtypes.int)).toEqual(await tiny`out(tiny.dtype.dtypes.min(tiny.dtype.dtypes.int))`)
    expect(dtypes.max(dtypes.int)).toEqual(await tiny`out(tiny.dtype.dtypes.max(tiny.dtype.dtypes.int))`)

    // Uint
    expect(dtypes.min(dtypes.uint)).toEqual(await tiny`out(tiny.dtype.dtypes.min(tiny.dtype.dtypes.uint))`)
    expect(dtypes.max(dtypes.uint)).toEqual(await tiny`out(tiny.dtype.dtypes.max(tiny.dtype.dtypes.uint))`)

    // Float
    expect(dtypes.min(dtypes.float)).toEqual(-Infinity)
    expect(dtypes.max(dtypes.float)).toEqual(Infinity)

    // Bool
    expect(dtypes.min(dtypes.bool)).toEqual(await tiny`out(tiny.dtype.dtypes.min(tiny.dtype.dtypes.bool))`)
    expect(dtypes.max(dtypes.bool)).toEqual(await tiny`out(tiny.dtype.dtypes.max(tiny.dtype.dtypes.bool))`)
})

// TODO test dtypes.fromJS and asConst

Deno.test('_getRecursiveParents', async (t) => {
    const inputs = ['float64', 'float32', 'float16', 'half',"int"] as const
    for (const input of inputs) {
        await t.step(input, async () => {
            const res = await runPython(`out([asdict(v) for v in tiny.dtype._get_recursive_parents(tiny.dtype.dtypes.${input}) ])`)
            const res2 = dt._getRecursiveParents(dtypes[input])

            expect(res2.map((x) => `${x.name} ${x.itemsize}`)).toEqual(res.map((x: any) => `${x.name} ${x.itemsize}`))
        })
    }
})
// Deno.test('leastUpper', async (t) => {
//     const inputs = [
//         ['int', 'int'],
//         ['int', 'uint', 'long'],
//         ['float64', 'float32'],
//         ['float64', 'half'],
//         ['bool', 'half'],
//         ['int', 'int32', 'int64'],
//     ] as const
//     for (const input of inputs) {
//         await t.step(input.toString(), async () => {
//             const res = await runPython(`out(asdict(tiny.dtype.least_upper_dtype(${input.map((x) => `tiny.dtype.dtypes.${x}`)})))`)
//             // expect(asdict(dt.leastUpperDType(...input.map((i) => dtypes[i])))).toEqual(res)
//         })
//     }
// })
Deno.test('sumAccDType', async () => {})
Deno.test('truncate', async () => {})
