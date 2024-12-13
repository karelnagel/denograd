import { get_index, lower_load_store, lower_reduce_axis, rewrite_shapetracker_with_index } from '../../src/codegen/lowerer.ts'
import { dtypes } from '../../src/dtype.ts'
import { Ops, UOp } from '../../src/ops.ts'
import { ClangRenderer } from '../../src/renderer/cstyle.ts'
import { compare } from '../helpers.ts'

Deno.test(
  'get_index',
  compare(
    [
      // Test case for basic shape with no reduces/upcasts
      [new UOp({ op: Ops.DEFINE_GLOBAL, dtype: dtypes.float32, src: [], arg: ['buf', 10] }),new ClangRenderer()],

      // Test case with reduces
      [new UOp({ op: Ops.REDUCE_AXIS, dtype: dtypes.float32, src: [UOp.int(0)], arg: [Ops.ADD, [1]] }),new ClangRenderer()],

      // Test case with upcasts
      [new UOp({ op: Ops.DEFINE_GLOBAL, dtype: dtypes.float32, src: [], arg: ['buf', 10], }),new ClangRenderer()],

      // Test case with local dims
      [new UOp({ op: Ops.DEFINE_LOCAL, dtype: dtypes.float32, src: [], arg: ['local', 16] }),new ClangRenderer()],

      // Test case with grouped reduces
      [new UOp({ op: Ops.REDUCE_AXIS, dtype: dtypes.float32, src: [UOp.int(0)], arg: [Ops.ADD, [1,2]],  }),new ClangRenderer()]
    ],
     get_index,
    'out(tiny.codegen.lowerer.get_index(*data))',
  ),
)

Deno.test(
  'lower_reduce_axis',
  compare(
    [],
    lower_reduce_axis,
    'out(tiny.codegen.lowerer.lower_reduce_axis(*data))',
  ),
)

Deno.test(
  'lower_load_store',
  compare(
    [],
    lower_load_store,
    'out(tiny.codegen.lowerer.lower_load_store(*data))',
  ),
)

Deno.test(
  'rewrite_shapetracker_with_index',
  compare(
    [],
    rewrite_shapetracker_with_index,
    'out(tiny.codegen.lowerer.rewrite_shapetracker_with_index(*data))',
  ),
)
