import * as dt from '../src/dtype.ts'
import { asdict, tiny, trycatch } from './helpers.ts'
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
from dataclasses import asdict
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
