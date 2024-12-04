import { Ops, spec, symbolicFlat, UOp, UPat } from '../src/ops.ts'
import { compare, tryCatch } from './helpers.ts'
import { _merge_dims, _reshape_mask, canonicalize_strides, strides_for_shape, un1d, View } from '../src/shape/view.ts'
import { dtypes } from '../src/dtype.ts'

Deno.test(
  'serialization',
  compare(
    [
      [Ops.ADD],
      [Ops.ASSIGN],
      [new UOp({ op: Ops.BARRIER, dtype: dtypes.float, arg: 5445 })],
      [new UPat({ op: Ops.ASSIGN, dtype: dtypes.floats, arg: 555, name: 'sdf' })],
      [new UPat({ op: Ops.IF, name: 'conditional_op', dtype: dtypes.bool, src: [new UPat({ op: Ops.CMPLT, name: 'cmp_op', dtype: dtypes.bool }), new UPat({ name: 'true_case' }), new UPat({ name: 'false_case' })] })],
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
  compare(
    [
      [[UOp.int(200), UOp.int(100), UOp.int(30)], [UOp.int(20), UOp.int(2100), UOp.int(20000)]],
      [[44444, UOp.int(100), 4.4], [UOp.int(20), 5555, -5.5]],
      [[44444, 543, 4.4], [555, 5555, -5.5]],
    ],
    canonicalize_strides,
    'out(tiny.shape.view.canonicalize_strides(*data))',
  ),
)

Deno.test(
  'strides_for_shape',
  compare(
    [
      [[4, 23, 545, 32443, 44, -1]],
      [[4, 23, 545, 32443, 44, -1, UOp.int(444), UOp.bool(true), UOp.float(4.44)]],
      [[UOp.int(200), UOp.int(500), UOp.int(500), UOp.int(500)]],
      [[UOp.int(2), UOp.float(4.4), UOp.int(743)]],
    ],
    strides_for_shape,
    `out(tiny.shape.view.strides_for_shape(*data))`,
  ),
)

Deno.test(
  'merge_dims',
  compare(
    [
      [[UOp.int(4), UOp.int(9), UOp.int(4)], [UOp.int(1), UOp.int(9), UOp.int(4)], [[UOp.int(23), UOp.int(44)], [UOp.int(323), UOp.int(23)], [UOp.int(43), UOp.int(43)]]],
      [[UOp.int(4), UOp.int(9), 44], [4, 9, 4], [[UOp.int(23), UOp.int(44)], [UOp.int(323), UOp.int(23)], [UOp.int(43), UOp.int(43)]]],
      [[UOp.int(4), UOp.int(9), 44], [4, 9, 4, 44], [[UOp.int(23), UOp.int(44)], [44, UOp.int(23)], [UOp.int(43), UOp.int(43)]]],
      [[4, 34, 534], [3, 4, 6], [[23, 32], [43, 43], [43, 43]]],
      [[7, 21, 123], [8, 2, 9], [[12, 45], [31, 55.4], [67, 89]]],
      [[4, 34, 534], [3, 4, 6]],
    ],
    tryCatch(_merge_dims),
    `out(trycatch(lambda: tiny.shape.view._merge_dims(*data)))`,
  ),
)

Deno.test(
  '_reshape_mask',
  compare(
    [
      [[[UOp.int(2), UOp.int(3)]], [UOp.int(44), UOp.int(44)], [UOp.int(444), UOp.int(44)]],
      [[[UOp.int(2), UOp.int(3)], [UOp.int(5), UOp.int(5)]], [UOp.int(44), UOp.int(44)], [UOp.int(444), UOp.int(44)]],
      [[[2, 3], [UOp.int(5), 44444]], [555, UOp.int(44)], [UOp.int(444), UOp.float(44)]],
    ],
    _reshape_mask,
    `out(tiny.shape.view._reshape_mask(*data))`,
  ),
)

Deno.test(
  'un1d',
  compare(
    [
      [[4, 23, 545], UOp.int(44)],
      [[2, 3, 4], UOp.int(15)],
      [[3, 5], UOp.int(7)],
      [[4], UOp.int(2)],
      [[UOp.int(2), UOp.int(3)], UOp.int(4)],
      [[UOp.int(4), UOp.int(5), UOp.int(6)], UOp.int(47)],
    ],
    un1d,
    `out(tiny.shape.view.un1d(*data))`,
  ),
)

Deno.test(
  'View.create',
  compare(
    [
      [[4, 34, 534]], // basic shape
      [[4, 34, 534], [3, 4, 6]], // shape and strides
      [[4, 34, 534], [3, 4, 6], 4, [[23, 32], [43, 43], [43, 43]]], // shape, strides, offset, mask
      [[UOp.int(4), UOp.int(9), UOp.int(4)]], // symbolic shape
      [[UOp.int(4), UOp.int(9), UOp.int(4)], [UOp.int(1), UOp.int(9), UOp.int(4)]], // symbolic shape and strides
      [[UOp.int(4), UOp.int(9), UOp.int(4)], [UOp.int(1), UOp.int(9), UOp.int(4)], 0, [[UOp.int(23), UOp.int(44)], [UOp.int(323), UOp.int(23)], [UOp.int(43), UOp.int(43)]]], // symbolic everything
      [[0, 34, 534]], // zero in shape
      [[4, 0, 534]], // zero in middle of shape
      [[4, 34, 0]], // zero at end of shape
      [[4, 34, 534], undefined, UOp.int(5)], // offset with no strides
      [[4, 34], [36, 1], 0, [[0, 4], [0, 34]]], // mask matching shape
      [[4, 34], [36, 1], 0, [[1, 3], [2, 33]]], // partial mask
      [[4, 34], [36, 1], 0, [[2, 2], [3, 4]]], // single element mask
      [[4, 34], [36, 1], 0, [[3, 2], [4, 3]]], // invalid mask (start > end)
      [[UOp.int(4), UOp.int(9)], [UOp.int(9), UOp.int(1)], UOp.int(3), [[UOp.int(2), UOp.int(3)], [UOp.int(4), UOp.int(8)]]], // symbolic mask
      [[UOp.int(4), UOp.int(9)], [UOp.int(9), UOp.int(1)], UOp.int(3), [[UOp.int(3), UOp.int(2)], [UOp.int(8), UOp.int(4)]]], // invalid symbolic mask
    ],
    View.create,
    'out(tiny.shape.view.View.create(*data))',
  ),
)
