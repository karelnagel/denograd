import { UOp } from '../../src/ops.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { compare, tryCatch } from '../helpers.ts'

Deno.test(
  'ShapeTracker.from_shape',
  compare(
    [
      [[5, 5]],
      [[UOp.int(44), UOp.int(-44)]],
      [[UOp.int(44), UOp.int(44)]],
      [[4, UOp.int(44344)]],
    ],
    tryCatch(ShapeTracker.from_shape),
    'out(trycatch(lambda: tiny.shape.shapetracker.ShapeTracker.from_shape(*data)))',
    { stringSimilarity: 0.82 },
  ),
)

const st1 = ShapeTracker.from_shape([5, 5])
const st2 = ShapeTracker.from_shape([4, UOp.int(55)])
const st3 = ShapeTracker.from_shape([UOp.float(8), UOp.int(110), UOp.int(33)])

// GETTERS
Deno.test(
  'ShapeTracker.contiguous',
  compare(
    [[st1], [st1], [st2]],
    (shape: ShapeTracker) => shape.contiguous,
    'out(trycatch(lambda: data[0].contiguous))',
  ),
)
Deno.test(
  'ShapeTracker.consecutive',
  compare(
    [[st1], [st1], [st2]],
    (shape: ShapeTracker) => shape.consecutive,
    'out(trycatch(lambda: data[0].consecutive))',
  ),
)
Deno.test(
  'ShapeTracker.shape',
  compare(
    [[st1], [st1], [st2]],
    (shape: ShapeTracker) => shape.shape,
    'out(trycatch(lambda: data[0].shape))',
  ),
)
Deno.test(
  'ShapeTracker.size',
  compare(
    [[st1], [st1], [st2]],
    (shape: ShapeTracker) => shape.size,
    'out(trycatch(lambda: data[0].size))',
  ),
)
Deno.test(
  'ShapeTracker.var_vals',
  compare(
    [[st1], [st1], [st2]],
    (shape: ShapeTracker) => shape.var_vals,
    'out(trycatch(lambda: data[0].var_vals))',
  ),
)

const testShape = <T extends any[]>(fn: (shape: ShapeTracker) => (...a: T) => any) => tryCatch((shape: ShapeTracker, args: T) => (fn(shape)(...args)))

Deno.test(
  'ShapeTracker.__add__',
  compare(
    [
      [st1, [st2]],
      [st1, [st3]],
      [st2, [st3]],
    ],
    testShape((shape) => shape.__add__),
    'out(trycatch(lambda: data[0].__add__(*data[1])))',
  ),
)
Deno.test(
  'ShapeTracker.reduce',
  compare(
    [
      [st1, [[0]]],
      [st1, [[1]]],
      [st1, [[0, 1]]],
      [st2, [[0]]],
      [st2, [[1]]],
      [st2, [[0, 1]]],
    ],
    testShape((shape) => shape.reduce),
    'out(trycatch(lambda: data[0].reduce(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.to_uop',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.to_uop),
    'out(trycatch(lambda: data[0].to_uop(*data[1])))',
  ),
)
Deno.test(
  'ShapeTracker.to_indexed_uops',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.to_indexed_uops),
    'out(trycatch(lambda: data[0].to_indexed_uops(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.real_size',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.real_size),
    'out(trycatch(lambda: data[0].real_size(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.vars',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.vars),
    'out(trycatch(lambda: data[0].vars(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.unbind',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.unbind),
    'out(trycatch(lambda: data[0].unbind(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.real_strides',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.real_strides),
    'out(trycatch(lambda: data[0].real_strides(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.unit_stride_axes',
  compare(
    [
      [st1, []],
      [st1, []],
      [st2, []],
    ],
    testShape((shape) => shape.unit_stride_axes),
    'out(trycatch(lambda: data[0].unit_stride_axes(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.axis_is_masked',
  compare(
    [
      [st1, [0]],
      [st1, [1]],
      [st2, [0]],
    ],
    testShape((shape) => shape.axis_is_masked),
    'out(trycatch(lambda: data[0].axis_is_masked(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.simplify',
  compare(
    [
      [st1, []],
      [st2, []],
      [st3, []],
    ],
    testShape((shape) => shape.simplify),
    'out(trycatch(lambda: data[0].simplify(*data[1])))',
  ),
)

Deno.test(
  'ShapeTracker.reshape',
  compare(
    [
      [st1, [[UOp.int(33), UOp.int(333)]]],
      [st1, [[3, 4]]],
      [st1, [[12]]],
      [st1, [[2, 2, 3]]],
      [st2, [[UOp.int(10), UOp.int(20)]]],
      [st2, [[5, 4]]],
      [st2, [[20]]],
    ],
    testShape((shape) => shape.reshape),
    'out(trycatch(lambda: data[0].reshape(*data[1])))',
    { stringSimilarity: 0.81 },
  ),
)
