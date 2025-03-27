import type { Kernel } from '../../jsgrad/codegen/kernel.ts'
import { dtypes } from '../../jsgrad/dtype.ts'
import { Ops, UOp } from '../../jsgrad/ops.ts'
import { ClangRenderer } from '../../jsgrad/renderer/cstyle.ts'
import type { Renderer } from '../../jsgrad/renderer/index.ts'
import { ShapeTracker } from '../../jsgrad/shape/shapetracker.ts'
import { View } from '../../jsgrad/shape/view.ts'

export const kernelKeys = [
  'ast',
  'opts',
  'vars',
  'bufs',
  'applied_opts',
  'group_for_reduces',
  'upcasted',
  'local_dims',
  'tensor_core',
  'tensor_core_opts',
  'use_tensor_cores',
  'dont_use_locals',
  'sts',
  'reduceops',
  'full_buf_index',
  'uops',
] as const
export const tsKernel = (k: Kernel) => kernelKeys.map((key) => k[key])
export const pyKernel = `out([getattr(k,key,None) for key in [${kernelKeys.map((k) => `"${k}"`)}]])`

export const kernelInputs = (): [Renderer, UOp][] => [
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([1], [0], 0, undefined, true)]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([1], [0], 0, undefined, true)]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 1.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(10), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([10], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([10], [0], 0, undefined, false)]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(64), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([64], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([64], [0], 0, undefined, false)]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(32), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([32], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([32], [0], 0, undefined, false)]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(5760), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([
                new View([10, 576], [0, 0], 0, undefined, false),
              ]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(800), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([
            new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true),
          ]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([
                new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false),
              ]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(36864), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([
            new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true),
          ]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([
                new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false),
              ]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(18432), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([
            new View([64, 32, 3, 3], [288, 9, 3, 1], 0, undefined, true),
          ]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([
                new View([64, 32, 3, 3], [0, 0, 0, 0], 0, undefined, false),
              ]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(25600), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([
            new View([32, 32, 5, 5], [800, 25, 5, 1], 0, undefined, true),
          ]),
        ),
        new UOp(Ops.WHERE, dtypes.float, [
          new UOp(Ops.VALID, dtypes.bool, [
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([
                new View([32, 32, 5, 5], [0, 0, 0, 0], 0, undefined, false),
              ]),
            ),
          ], undefined),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
          new UOp(Ops.CONST, dtypes.float, [], 0.1),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.long.ptr(1), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([1], [0], 0, undefined, true)]),
        ),
        new UOp(Ops.ADD, dtypes.long, [
          new UOp(Ops.LOAD, dtypes.long, [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.long.ptr(1), [], 0),
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([1], [0], 0, undefined, true)]),
            ),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.long, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([1], [0], 0, undefined, true)]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.long, [], 1),
            new UOp(Ops.CONST, dtypes.long, [], 0),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(1), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([1], [0], 0, undefined, true)]),
        ),
        new UOp(Ops.ADD, dtypes.uint, [
          new UOp(Ops.LOAD, dtypes.uint, [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(1), [], 1),
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([1], [0], 0, undefined, true)]),
            ),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.uint, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([1], [0], 0, undefined, true)]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.uint, [], 32),
            new UOp(Ops.CONST, dtypes.uint, [], 0),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(1), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([1], [0], 0, undefined, true)]),
        ),
        new UOp(Ops.ADD, dtypes.uint, [
          new UOp(Ops.LOAD, dtypes.uint, [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(1), [], 1),
            new UOp(
              Ops.VIEW,
              dtypes.void,
              [],
              new ShapeTracker([new View([1], [0], 0, undefined, true)]),
            ),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.uint, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([1], [0], 0, undefined, true)]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.uint, [], 64),
            new UOp(Ops.CONST, dtypes.uint, [], 0),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],

  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(4), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([4], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.CMPNE, dtypes.bool, [
          new UOp(Ops.CMPNE, dtypes.bool, [
            new UOp(Ops.LOAD, dtypes.float, [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 1),
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([4], [1], 0, undefined, true)]),
              ),
            ], undefined),
            new UOp(Ops.WHERE, dtypes.float, [
              new UOp(Ops.VALID, dtypes.bool, [
                new UOp(
                  Ops.VIEW,
                  dtypes.void,
                  [],
                  new ShapeTracker([new View([4], [0], 0, undefined, false)]),
                ),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], Infinity),
              new UOp(Ops.CONST, dtypes.float, [], 0.1),
            ], undefined),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.bool, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([4], [0], 0, undefined, false)]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.bool, [], true),
            new UOp(Ops.CONST, dtypes.bool, [], false),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(4), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([4], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.CMPNE, dtypes.bool, [
          new UOp(Ops.CMPNE, dtypes.bool, [
            new UOp(Ops.LOAD, dtypes.float, [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 1),
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([4], [1], 0, undefined, true)]),
              ),
            ], undefined),
            new UOp(Ops.WHERE, dtypes.float, [
              new UOp(Ops.VALID, dtypes.bool, [
                new UOp(
                  Ops.VIEW,
                  dtypes.void,
                  [],
                  new ShapeTracker([new View([4], [0], 0, undefined, false)]),
                ),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], Infinity),
              new UOp(Ops.CONST, dtypes.float, [], 0.1),
            ], undefined),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.bool, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([4], [0], 0, undefined, false)]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.bool, [], true),
            new UOp(Ops.CONST, dtypes.bool, [], false),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([4], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 1),
          new UOp(
            Ops.VIEW,
            dtypes.void,
            [],
            new ShapeTracker([new View([4], [1], 0, undefined, true)]),
          ),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.bool.ptr(4), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([4], [1], 0, undefined, true)]),
        ),
        new UOp(Ops.CMPNE, dtypes.bool, [
          new UOp(Ops.CMPNE, dtypes.bool, [
            new UOp(Ops.LOAD, dtypes.float, [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 1),
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([4], [1], 0, undefined, true)]),
              ),
            ], undefined),
            new UOp(Ops.WHERE, dtypes.float, [
              new UOp(Ops.VALID, dtypes.bool, [
                new UOp(
                  Ops.VIEW,
                  dtypes.void,
                  [],
                  new ShapeTracker([new View([4], [0], 0, undefined, false)]),
                ),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.1),
              new UOp(Ops.CONST, dtypes.float, [], 0.1),
            ], undefined),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.bool, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([new View([4], [0], 0, undefined, false)]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.bool, [], true),
            new UOp(Ops.CONST, dtypes.bool, [], false),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  [
    new ClangRenderer(),
    new UOp(Ops.SINK, dtypes.void, [
      new UOp(Ops.STORE, dtypes.void, [
        new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
        new UOp(
          Ops.VIEW,
          dtypes.void,
          [],
          new ShapeTracker([new View([60000, 1], [1, 0], 0, undefined, true)]),
        ),
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.REDUCE_AXIS, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int, [
              new UOp(Ops.VALID, dtypes.bool, [
                new UOp(
                  Ops.VIEW,
                  dtypes.void,
                  [],
                  new ShapeTracker([
                    new View([60001, 119999], [0, 0], 0, [[0, 60001], [
                      59999,
                      119999,
                    ]], false),
                    new View([60000, 60000], [1, 120000], 0, undefined, false),
                  ]),
                ),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 1),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ], [Ops.ADD, [1]]),
          new UOp(Ops.WHERE, dtypes.int, [
            new UOp(Ops.VALID, dtypes.bool, [
              new UOp(
                Ops.VIEW,
                dtypes.void,
                [],
                new ShapeTracker([
                  new View([60000, 1], [0, 0], 0, undefined, false),
                ]),
              ),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int, [], -1),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ], undefined),
      ], undefined),
    ], undefined),
  ],
  // For some reason fails in vitest but not in Deno.test
  // [new ClangRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6553600), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1, 1, 32, 1, 1, 32, 5, 5, 256], [0, 0, 204800, 0, 0, 6400, 1280, 256, 1], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.MUL, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(9437184), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2, 1, 32, 20, 20, 32, 5, 5, 256], [18432, 0, 0, 24, 1, 576, 24, 1, 36864], 0, undefined, false)]))], undefined), new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6553600), [], 2), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2, 1, 32, 20, 20, 32, 5, 5, 256], [12800, 0, 400, 20, 1, 0, 0, 0, 25600], 0, undefined, false)]))], undefined)], undefined)], [Ops.ADD, [0, 3, 4]])], undefined)], undefined)],
]
