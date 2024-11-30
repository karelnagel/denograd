import { Ops, spec, symbolicFlat, UOp, UPat } from '../src/ops.ts'
import { test } from './helpers.ts'
import { _merge_dims, _reshape_mask, canonicalize_strides, strides_for_shape } from '../src/shape/view.ts'
import { dtypes } from '../src/dtype.ts'

Deno.test(
    'serialization',
    test(
        [
            [Ops.ADD],
            [Ops.ASSIGN],
            [new UOp({ op: Ops.BARRIER, dtype: dtypes.float, arg: 5445 })],
            [new UPat({ op: Ops.ASSIGN, dtype: dtypes.floats, arg: 555, name: 'sdf' })],
            [new UPat({ op: Ops.ASSIGN })],
            [dtypes.floats],
            [dtypes.defaultFloat],
            ...spec.patterns.map((p) => [p[0]] as any),
            ...symbolicFlat.patterns.map((p) => [p[0]]),
        ],
        (x) => x,
        'out(*data)',
    ),
)

Deno.test(
    'canonicalize_strides',
    test(
        [[[UOp.int(200), UOp.int(100), UOp.int(30)], [UOp.int(20), UOp.int(2100), UOp.int(20000)]]],
        canonicalize_strides,
        'out(tiny.shape.view.canonicalize_strides(*data))',
    ),
)

Deno.test(
    'strides_for_shape',
    test(
        [
            [[UOp.int(200), UOp.int(500), UOp.int(500), UOp.int(500)]],
            [[UOp.int(2), UOp.float(4.4), UOp.int(743)]],
        ],
        strides_for_shape,
        `
res = tiny.shape.view.strides_for_shape(*data)
# chaning the last element to UOp
out([*res[:-1],tiny.ops.UOp.const(tiny.dtype.dtypes.int,res[-1])])`,
    ),
)

Deno.test(
    'merge_dims',
    test(
        [
            [[4, 34, 534], [3, 4, 6], [[23, 32], [43, 43], [43, 43]]],
            [[7, 21, 123], [8, 2, 9], [[12, 45], [31, 55.4], [67, 89]]],
            [[4, 34, 534], [3, 4, 6]],
        ],
        _merge_dims,
        `out(tiny.shape.view._merge_dims(*data))`,
    ),
)

Deno.test(
    '_reshape_mask',
    test(
        [
            [[[UOp.int(2), UOp.int(3)]], [UOp.int(44), UOp.int(44)], [UOp.int(444), UOp.int(44)]],
            [[[UOp.int(2), UOp.int(3)], [UOp.int(5), UOp.int(5)]], [UOp.int(44), UOp.int(44)], [UOp.int(444), UOp.int(44)]],
        ],
        (...args) => {
            const res = _reshape_mask(args[0] as any, args[1] as any, args[2] as any)
            console.log(res)
            return res
        },
        `res =tiny.shape.view._reshape_mask(*data)\nout(res)`,
    ),
)
