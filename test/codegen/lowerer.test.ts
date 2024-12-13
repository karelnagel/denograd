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
      [
        new UOp({
          op: Ops.SINK,
          dtype: dtypes.void,
          src: [
            new UOp({
              op: Ops.STORE,
              dtype: dtypes.void,
              src: [
                new UOp({
                  op: Ops.DEFINE_GLOBAL,
                  dtype: dtypes.float32,
                  src: [],
                  arg: ['buf', [10, 20]], // 2D shape
                }),
                UOp.const(dtypes.float32, 1.0),
              ],
              arg: undefined,
            }),
          ],
          arg: undefined,
        }),
        new ClangRenderer(),
      ],

      // Test case with reduces
      [
        new UOp({
          op: Ops.REDUCE_AXIS,
          dtype: dtypes.float32,
          src: [
            new UOp({
              op: Ops.LOAD,
              dtype: dtypes.float32,
              src: [
                new UOp({
                  op: Ops.DEFINE_GLOBAL,
                  dtype: dtypes.float32,
                  src: [],
                  arg: ['buf', [10, 5]], // Shape with reduce axis
                }),
              ],
              arg: undefined,
            }),
          ],
          arg: [Ops.ADD, [1]], // Reduce along axis 1
        }),
        new ClangRenderer(),
      ],

      // Test case with upcasts and local dims
      [
        new UOp({
          op: Ops.SINK,
          dtype: dtypes.void,
          src: [
            new UOp({
              op: Ops.STORE,
              dtype: dtypes.void,
              src: [
                new UOp({
                  op: Ops.DEFINE_LOCAL,
                  dtype: dtypes.float32,
                  src: [],
                  arg: ['local', [16, 4]], // Local dims
                }),
                UOp.const(dtypes.float32, 1.0),
              ],
              arg: undefined,
            }),
          ],
          arg: undefined, // 2 upcast dimensions
        }),
        new ClangRenderer(), // Enable local dims
      ],

      // Test case with grouped reduces
      [
        new UOp({
          op: Ops.REDUCE_AXIS,
          dtype: dtypes.float32,
          src: [
            new UOp({
              op: Ops.LOAD,
              dtype: dtypes.float32,
              src: [
                new UOp({
                  op: Ops.DEFINE_LOCAL,
                  dtype: dtypes.float32,
                  src: [],
                  arg: ['local', [8, 4, 2]], // 3D shape for grouped reduces
                }),
              ],
              arg: undefined,
            }),
          ],
          arg: [Ops.ADD, [1, 2]], // Reduce along axes 1 and 2
        }),
        new ClangRenderer(),
      ],
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
