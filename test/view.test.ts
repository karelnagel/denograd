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
      [undefined, [4, 4], [8, 1, 1, 2, 1, 1, 1]],
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

const view1 = View.create([4, 4], [4, 1], 0, undefined)
const view2 = View.create([UOp.int(4), UOp.int(6)], [UOp.int(6), UOp.int(1)], UOp.int(8), [[UOp.int(1), UOp.int(3)], [UOp.int(2), UOp.int(5)]])
const view3 = View.create([4, 4], [-4, -1], 0, undefined)
const view4 = View.create([0, 3, 2], [6, -2, 1], 0, [[1, 3], [0, 4]])

Deno.test(
  'View.t',
  compare(
    [[view1], [view2], [view3], [view4]],
    (view: View) => view.t,
    'out(data[0].t)',
  ),
)

const testView = <T extends any[]>(fn: (view: View) => (...a: T) => any) => tryCatch((view: View, args: T) => (fn(view)(...args)))
Deno.test(
  'View.to_indexed_uops',
  compare(
    [
      [view1, []], // default case with no indices
      [view1, [[UOp.int(2), UOp.int(3)]]], // basic indices
      [view2, []], // symbolic shape with mask
      [view2, [[UOp.int(1), UOp.int(4)]]], // symbolic shape, mask and custom indices
      [view3, []], // 3D view with mask
      [view3, [[UOp.int(1), UOp.int(2), UOp.int(1)]]], // 3D view with mask and indices
      [view4, []], // 4D view with mask
      [view4, [[UOp.int(0), UOp.int(2), UOp.int(1), UOp.int(0)]]], // 4D view with mask and indices
    ],
    testView((v) => v.to_indexed_uops),
    'out(trycatch(lambda:data[0].to_indexed_uops(*data[1])))',
  ),
)

Deno.test(
  'View.size',
  compare(
    [
      [view1, []],
      [view2, []],
      [view3, []],
      [view4, []],
    ],
    testView((v) => v.size),
    'out(trycatch(lambda:data[0].size(*data[1])))',
  ),
)

Deno.test(
  'View.vars',
  compare(
    [
      [view1, []],
      [view2, []],
      [view3, []],
      [view4, []],
    ],
    testView((v) => v.vars),
    'out(trycatch(lambda:data[0].vars(*data[1])))',
  ),
)

Deno.test(
  'View.unbind',
  compare(
    [
      [view1, []],
      [view2, []],
      [view3, []],
      [view4, []],
    ],
    testView((v) => v.unbind),
    'out(trycatch(lambda:data[0].unbind(*data[1])))',
  ),
)

Deno.test(
  'View.__add__',
  compare(
    [
      [view1, [view1]],
      [view2, [view1]],
      [view1, [view2]],
      [view3, [view1]],
      [view3, [view4]],
    ],
    testView((v) => v.__add__),
    'out(trycatch(lambda:data[0].__add__(*data[1])))',
  ),
)

Deno.test(
  'View.invert',
  compare(
    [
      [view1, [[3, UOp.int(4)]]],
      [view2, [[UOp.int(4), UOp.int(6)]]],
      [view3, [[UOp.int(3), UOp.int(5), UOp.int(2)]]],
      [view4, [[UOp.int(2), UOp.int(4), UOp.int(3), UOp.int(2)]]],
      [view4, [[4, 4]]],
      [view4, [[3, 3, 3]]],
      [view1, [[4, 4]]],
    ],
    testView((v) => v.invert),
    'out(trycatch(lambda:data[0].invert(*data[1])))',
  ),
)

Deno.test(
  'View.minify',
  compare(
    [
      [view1, []],
      [view2, []],
      [view3, []],
      [view4, []],
    ],
    testView((v) => v.minify),
    'out(trycatch(lambda:data[0].minify(*data[1])))',
  ),
)

Deno.test(
  'View.__unsafe_resize',
  compare(
    [
      [view1, [[[0, 2], [1, 3]]]],
      [view1, [[[1, 3], [0, 2]], [[0, 1], [1, 2]]]],
      [view2, [[[0, 2], [2, 4]]]],
      [view3, [[[1, 3], [1, 3]]]],
      [view4, [[[0, 2], [1, 2], [0, 1]]]],
      [view4, [[[1, 2], [0, 2], [0, 1]], [[0, 1], [1, 2], [0, 1]]]],
      [view1, [[[UOp.int(0), UOp.int(2)], [UOp.int(1), UOp.int(3)]]]],
      [view2, [[[UOp.int(1), UOp.int(3)], [UOp.int(2), UOp.int(4)]]]],
      [view2, [[[UOp.int(0), UOp.int(2)], [UOp.int(1), UOp.int(4)]], [[UOp.int(1), UOp.int(2)], [UOp.int(2), UOp.int(3)]]]],
      [view3, [[[0, UOp.int(2)], [1, UOp.int(3)]]]],
      [view4, [[[UOp.int(0), 2], [1, UOp.int(3)], [0, UOp.int(1)]]]],
    ],
    testView((v) => v.__unsafe_resize),
    'out(trycatch(lambda:data[0].__unsafe_resize(*data[1])))',
  ),
)
Deno.test(
  'View.pad',
  compare(
    [
      [view1, [[[0, 0], [0, 0]]]],
      [view1, [[[1, 1], [2, 2]]]],
      [view2, [[[0, 1], [1, 0]]]],
      [view3, [[[1, 0], [0, 1], [2, 1]]]],
      [view4, [[[0, 0], [1, 1], [2, 2], [0, 1]]]],
      [view1, [[[UOp.int(1), UOp.int(2)], [UOp.int(0), UOp.int(1)]]]],
      [view2, [[[UOp.int(2), UOp.int(0)], [UOp.int(1), UOp.int(2)]]]],
      [view3, [[[0, UOp.int(1)], [UOp.int(1), 0], [1, UOp.int(2)]]]],
    ],
    testView((v) => v.pad),
    'out(trycatch(lambda:data[0].pad(*data[1])))',
  ),
)

Deno.test(
  'View.shrink',
  compare(
    [
      [view1, [[[0, 2], [1, 3]]]],
      [view2, [[[0, 2], [2, 4]]]],
      [view3, [[[1, 3], [1, 3]]]],
      [view4, [[[0, 2], [1, 2], [0, 1]]]],
      [view1, [[[UOp.int(0), UOp.int(2)], [UOp.int(1), UOp.int(3)]]]],
      [view2, [[[UOp.int(1), UOp.int(3)], [UOp.int(2), UOp.int(4)]]]],
      [view3, [[[0, UOp.int(2)], [1, UOp.int(3)]]]],
      [view4, [[[UOp.int(0), 2], [1, UOp.int(3)], [0, UOp.int(1)]]]],
    ],
    testView((v) => v.shrink),
    'out(trycatch(lambda:data[0].shrink(*data[1])))',
  ),
)
Deno.test(
  'View.expand',
  compare(
    [
      [view1, [[2, 3]]],
      [view1, [[1, 4]]],
      [view2, [[2, 4]]],
      [view3, [[3, 3, 3]]],
      [view4, [[2, 2, 1, 2]]],
      [view1, [[UOp.int(2), UOp.int(3)]]],
      [view2, [[UOp.int(2), UOp.int(4)]]],
      [view3, [[UOp.int(3), 3, UOp.int(3)]]],
      [view4, [[2, UOp.int(2), 1, UOp.int(2)]]],
    ],
    testView((v) => v.expand),
    'out(trycatch(lambda:data[0].expand(*data[1])))',
  ),
)

Deno.test(
  'View.permute',
  compare(
    [
      [view1, [[0, 1]]],
      [view1, [[1, 0]]],
      [view2, [[1, 0]]],
      [view3, [[0, 2, 1]]],
      [view3, [[1, 0, 2]]],
      [view4, [[3, 1, 2, 0]]],
      [view4, [[0, 2, 1, 3]]],
    ],
    testView((v) => v.permute),
    'out(trycatch(lambda:data[0].permute(*data[1])))',
  ),
)

Deno.test(
  'View.stride',
  compare(
    [
      [view1, [[2]]],
      [view1, [[-2]]],
      [view2, [[2, 3]]],
      [view2, [[-2, -3]]],
      [view3, [[2, -2, 3]]],
      [view4, [[2, -1, 3, -2]]],
    ],
    testView((v) => v.stride),
    'out(trycatch(lambda:data[0].stride(*data[1])))',
  ),
)

Deno.test(
  'View.reshape',
  compare(
    [
      [view1, [[1, 16]]],
      [view1, [[4, 4]]],
      [view2, [[3, 8]]],
      [view2, [[24]]],
      [view2, [[UOp.int(24)]]],
      [view2, [[12, 1, 1, 2, 1, 1, 1]]],
      [view2, [[UOp.int(12), 1, 1, 2, 1, 1, 1]]],
      [view3, [[8, 1, 1, 2, 1, 1, 1]]],
    ],
    testView((v) => v.reshape),
    'out(trycatch(lambda:data[0].reshape(*data[1])))',
    { stringSimilarity: 0.6 },
  ),
)
