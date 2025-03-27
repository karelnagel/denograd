import { _choices_from_args, _expand_arg_to_idx, _swizzle_args, create_gate, delete_redundant_gates, devectorize, do_contract, do_expand, fix_unfoldable_image_load, fold_expanded, full_graph_rewrite, loop_collapse, move_mask, no_vectorized_acc, no_vectorized_alu, no_vectorized_load_store, reduce_collapse, simplify_valid_load, sym, threefry2x32 } from '../../jsgrad/ops.ts'
import { dtypes } from '../../jsgrad/dtype.ts'
import { KernelInfo, Ops, UOp } from '../../jsgrad/ops.ts'
import { compare, tryCatch } from '../helpers.ts'
import { ClangRenderer } from '../../jsgrad/renderer/cstyle.ts'
import { WGSLRenderer } from '../../jsgrad/renderer/wgsl.ts'
import { describe } from 'vitest'

describe(
  'fold_expanded',
  compare(
    [
      [
        new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [])]),
        UOp.variable('buf'),
      ],
      [
        new UOp(Ops.STORE, undefined, [
          new UOp(Ops.INDEX, undefined, [UOp.variable('buf')]),
        ]),
        UOp.variable('buf'),
      ],
      [
        new UOp(Ops.VECTORIZE, undefined, [
          new UOp(Ops.LOAD, undefined, [
            new UOp(Ops.INDEX, undefined, [UOp.variable('buf')], 0),
          ]),
        ]),
        UOp.variable('buf'),
      ],
      [
        new UOp(Ops.VECTORIZE, undefined, [
          new UOp(Ops.LOAD, undefined, [
            new UOp(Ops.ADD, undefined, [UOp.variable('idx'), UOp.int(1)]),
          ]),
        ]),
        UOp.variable('buf'),
      ],
      [
        new UOp(Ops.VECTORIZE, undefined, [
          new UOp(Ops.LOAD, undefined, [
            new UOp(Ops.INDEX, undefined, [
              UOp.variable('buf'),
              UOp.variable('gate'),
            ]),
          ]),
        ]),
        UOp.variable('buf'),
      ],
      [
        new UOp(Ops.VECTORIZE, dtypes.void, [
          new UOp(Ops.LOAD, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.void, [
              new UOp(Ops.DEFINE_VAR, dtypes.bool, [], [`buf`, false, true]),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.DEFINE_VAR, dtypes.bool, [], [`buf`, false, true]),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 2]),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 5]),
                  new UOp(Ops.CONST, dtypes.int, [], 2),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
        ], new KernelInfo(1, 0, false)),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 8]),
                new UOp(Ops.CONST, dtypes.int, [], 4),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 8]),
                  new UOp(Ops.CONST, dtypes.int, [], 4),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 8]),
                  new UOp(Ops.CONST, dtypes.int, [], 4),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 8]),
                  new UOp(Ops.CONST, dtypes.int, [], 4),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 16]),
                new UOp(Ops.CONST, dtypes.int, [], 4),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 16]),
                  new UOp(Ops.CONST, dtypes.int, [], 4),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 16]),
                  new UOp(Ops.CONST, dtypes.int, [], 4),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 16]),
                  new UOp(Ops.CONST, dtypes.int, [], 4),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 60]),
                  new UOp(Ops.CONST, dtypes.int, [], 96),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 32]),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 60]),
                    new UOp(Ops.CONST, dtypes.int, [], 96),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 32]),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 60]),
                    new UOp(Ops.CONST, dtypes.int, [], 96),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 32]),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 60]),
                  new UOp(Ops.CONST, dtypes.int, [], 96),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 32]),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 60]),
                    new UOp(Ops.CONST, dtypes.int, [], 96),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 32]),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 60]),
                    new UOp(Ops.CONST, dtypes.int, [], 96),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 32]),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0),
          ], undefined),
        ], new KernelInfo(1, 1, false)),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 1),
            ], undefined),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 2),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 1),
            ], undefined),
          ], undefined),
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 2),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 2, false)),
        new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
      ],
    ],
    fold_expanded,
    'out(tiny.codegen.rewriter.fold_expanded(*data))',
  ),
)

describe(
  'fix_unfoldable_image_load',
  compare(
    [
      [
        new UOp(Ops.LOAD, undefined, [
          new UOp(Ops.INDEX, undefined, [
            UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)),
            UOp.int(5),
          ]),
        ]),
        UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)),
      ],
      [
        new UOp(Ops.LOAD, undefined, [
          new UOp(Ops.INDEX, undefined, [
            UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)),
            UOp.variable('idx'),
          ]),
        ]),
        UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)),
      ],
      [
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 11),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 11),
          new UOp(Ops.CONST, dtypes.int, [], 0),
        ], undefined),
      ],
      [
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 10),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 10),
          new UOp(Ops.CONST, dtypes.int, [], 0),
        ], undefined),
      ],
      [
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 12),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 12),
          new UOp(Ops.CONST, dtypes.int, [], 0),
        ], undefined),
      ],
      [
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 4),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 32),
            ], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 4),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 32),
          ], 0),
        ], undefined),
      ],
      [
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 3),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 32),
            ], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 3),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 32),
          ], 0),
        ], undefined),
      ],
      [
        new UOp(Ops.LOAD, dtypes.float, [
          new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 1),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 32),
            ], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.INDEX, dtypes.float.ptr(-1), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(-1), [], 1),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 32),
          ], 0),
        ], undefined),
      ],
    ],
    fix_unfoldable_image_load,
    'out(tiny.codegen.rewriter.fix_unfoldable_image_load(*data))',
  ),
)

describe(
  'simplify_valid_load',
  compare(
    [
      [UOp.variable('buf'), UOp.int(5), UOp.int(0)], // idx is None case
      [UOp.variable('buf'), UOp.int(5), UOp.int(1)], // non-image buffer, same idx
      [UOp.variable('buf'), UOp.variable('idx'), UOp.variable('valid')], // non-image buffer, different idx
    ],
    simplify_valid_load,
    'out(tiny.codegen.rewriter.simplify_valid_load(*data))',
  ),
)

describe(
  'threefry2x32',
  compare(
    [
      [
        new UOp(Ops.OR, dtypes.ulong, [
          new UOp(Ops.CAST, dtypes.ulong, [
            new UOp(Ops.LOAD, dtypes.uint, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 2),
                new UOp(Ops.SPECIAL, dtypes.int, [], ['gidx0', 8]),
              ], undefined),
            ], undefined),
          ], undefined),
          new UOp(Ops.MUL, dtypes.ulong, [
            new UOp(Ops.CAST, dtypes.ulong, [
              new UOp(Ops.LOAD, dtypes.uint, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 1),
                  new UOp(Ops.SPECIAL, dtypes.int, [], ['gidx0', 8]),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.ulong, [], 4294967296),
          ], undefined),
        ], undefined),
        new UOp(Ops.OR, dtypes.ulong, [
          new UOp(Ops.CAST, dtypes.ulong, [
            new UOp(Ops.LOAD, dtypes.uint, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 3),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
            ], undefined),
          ], undefined),
          new UOp(Ops.MUL, dtypes.ulong, [
            new UOp(Ops.CAST, dtypes.ulong, [
              new UOp(Ops.LOAD, dtypes.uint, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 3),
                  new UOp(Ops.CONST, dtypes.int, [], 1),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.ulong, [], 4294967296),
          ], undefined),
        ], undefined),
      ],
    ],
    (x: UOp, key: UOp) => {
      threefry2x32(x, key)
    },
    'tiny.codegen.rewriter.threefry2x32(*data)',
  ),
)

describe(
  'loop_collapse',
  compare(
    [
      // Basic test with positive mul
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
        undefined,
        undefined,
        undefined,
        undefined,
        UOp.int(1),
      ],

      // Test with negative mul
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
        undefined,
        undefined,
        undefined,
        undefined,
        undefined,
        UOp.int(0),
        UOp.int(-1),
      ],

      // Test with idx2 and idx3
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
        UOp.int(2),
        UOp.int(3),
      ],

      // Test with vectorization
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
        undefined,
        undefined,
        undefined,
        UOp.variable('vec', undefined, undefined, dtypes.float.vec(4)),
      ],

      // Test with extra accumulation
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
        undefined,
        undefined,
        UOp.int(5),
      ],

      // Test with non-zero loop start (should return undefined)
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
      ],

      // Test with disabled loop collapse (should return undefined)
      [
        UOp.int(10),
        UOp.int(1),
        new UOp(Ops.RANGE, undefined, undefined, [0, 5]),
        UOp.variable('acc'),
      ],
      [
        new UOp(Ops.VCONST, dtypes.int.vec(3), [], [59999, 59998, 59997]),
        new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
        new UOp(Ops.RANGE, dtypes.int, [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.CONST, dtypes.int, [], 15000),
        ], 1),
        new UOp(Ops.DEFINE_ACC, dtypes.int.vec(3), [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 15000),
          ], 1),
        ], [0]),
        undefined,
        undefined,
        undefined,
        new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
        new UOp(Ops.MUL, dtypes.int, [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20000),
          ], 0),
          new UOp(Ops.CONST, dtypes.int, [], 3),
        ], undefined),
        new UOp(Ops.CONST, dtypes.int, [], 4),
      ],
      [
        new UOp(Ops.VCONST, dtypes.int.vec(3), [], [59998, 59997, 59996]),
        new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
        new UOp(Ops.RANGE, dtypes.int, [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.CONST, dtypes.int, [], 15000),
        ], 1),
        new UOp(Ops.DEFINE_ACC, dtypes.int.vec(3), [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 15000),
          ], 1),
        ], [0]),
        undefined,
        undefined,
        new UOp(Ops.WHERE, dtypes.int.vec(3), [
          new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
            new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
              new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
              ], undefined),
              new UOp(Ops.VCONST, dtypes.int.vec(3), [], [59999, 59998, 59997]),
            ], undefined),
            new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
          ], undefined),
          new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
          new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
        ], undefined),
        new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
        new UOp(Ops.MUL, dtypes.int, [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20000),
          ], 0),
          new UOp(Ops.CONST, dtypes.int, [], 3),
        ], undefined),
        new UOp(Ops.CONST, dtypes.int, [], 4),
      ],
      [
        new UOp(Ops.VCONST, dtypes.int.vec(3), [], [59997, 59996, 59995]),
        new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
        new UOp(Ops.RANGE, dtypes.int, [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.CONST, dtypes.int, [], 15000),
        ], 1),
        new UOp(Ops.DEFINE_ACC, dtypes.int.vec(3), [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 15000),
          ], 1),
        ], [0]),
        undefined,
        undefined,
        new UOp(Ops.ADD, dtypes.int.vec(3), [
          new UOp(Ops.WHERE, dtypes.int.vec(3), [
            new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
              new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                ], undefined),
                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                  59998,
                  59997,
                  59996,
                ]),
              ], undefined),
              new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
            new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.int.vec(3), [
            new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
              new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                ], undefined),
                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                  59999,
                  59998,
                  59997,
                ]),
              ], undefined),
              new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
            new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
        new UOp(Ops.MUL, dtypes.int, [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20000),
          ], 0),
          new UOp(Ops.CONST, dtypes.int, [], 3),
        ], undefined),
        new UOp(Ops.CONST, dtypes.int, [], 4),
      ],
      [
        new UOp(Ops.VCONST, dtypes.int.vec(3), [], [59996, 59995, 59994]),
        new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
        new UOp(Ops.RANGE, dtypes.int, [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.CONST, dtypes.int, [], 15000),
        ], 1),
        new UOp(Ops.DEFINE_ACC, dtypes.int.vec(3), [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 15000),
          ], 1),
        ], [0]),
        undefined,
        undefined,
        new UOp(Ops.ADD, dtypes.int.vec(3), [
          new UOp(Ops.ADD, dtypes.int.vec(3), [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59998,
                    59997,
                    59996,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59999,
                    59998,
                    59997,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], undefined),
          new UOp(Ops.WHERE, dtypes.int.vec(3), [
            new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
              new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 20000),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 3),
                    ], undefined),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 15000),
                      ], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                  ], undefined),
                ], undefined),
                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                  59997,
                  59996,
                  59995,
                ]),
              ], undefined),
              new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
            new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
          ], undefined),
        ], undefined),
        new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 15000),
              ], 1),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
        new UOp(Ops.MUL, dtypes.int, [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20000),
          ], 0),
          new UOp(Ops.CONST, dtypes.int, [], 3),
        ], undefined),
        new UOp(Ops.CONST, dtypes.int, [], 4),
      ],
    ],
    loop_collapse,
    'out(tiny.codegen.rewriter.loop_collapse(*data))',
  ),
)

describe(
  '_expand_arg_to_idx',
  compare(
    [
      // Basic test with single axis
      [[[0, 4]], new Map([[0, 0]])],
      [[[0, 4]], new Map([[0, 1]])],
      [[[0, 4]], new Map([[0, 2]])],
      [[[0, 4]], new Map([[0, 3]])],

      // Test with multiple axes
      [[[0, 2], [1, 3]], new Map([[0, 0], [1, 0]])],
      [[[0, 2], [1, 3]], new Map([[0, 1], [1, 0]])],
      [[[0, 2], [1, 3]], new Map([[0, 0], [1, 1]])],
      [[[0, 2], [1, 3]], new Map([[0, 1], [1, 2]])],

      // Test with larger multipliers
      [[[0, 8], [1, 4]], new Map([[0, 3], [1, 2]])],
      [[[0, 16], [1, 8], [2, 4]], new Map([[0, 10], [1, 5], [2, 2]])],

      // Edge cases
      [[], new Map()], // Empty args
      [[[0, 1]], new Map([[0, 0]])], // Single element axis
      [[[2, 3]], new Map([[2, 0], [3, 0]])],
      [[[2, 3]], new Map([[2, 1], [3, 1]])],
      [[[2, 3]], new Map([[2, 0], [3, 1]])],
      [[[2, 3]], new Map([[2, 0], [3, 3]])],
      [[[2, 3]], new Map([[2, 0], [3, 2]])],
      [[[2, 3]], new Map([[2, 1], [3, 0]])],
      [[[2, 3]], new Map([[2, 0], [3, 1]])],
      [[[2, 3]], new Map([[2, 0], [3, 3]])],
      [[[2, 3]], new Map([[2, 1], [3, 0]])],
      [[[2, 3]], new Map([[2, 0], [3, 0]])],
      [[[2, 3]], new Map([[2, 1], [3, 1]])],
      [[[2, 3]], new Map([[2, 0], [3, 2]])],
    ],
    _expand_arg_to_idx,
    'out(tiny.codegen.rewriter._expand_arg_to_idx(data[0],{int(k): v for k, v in data[1].items()}))',
  ),
)

describe(
  '_choices_from_args',
  compare(
    [
      // Basic test with single axis
      [[[0, 2]]],
      [[[0, 3]]],
      [[[0, 4]]],

      // Test with multiple axes
      [[[0, 2], [1, 2]]],
      [[[0, 2], [1, 3]]],
      [[[0, 3], [1, 2]]],

      // Test with three axes
      [[[0, 2], [1, 2], [2, 2]]],
      [[[0, 2], [1, 3], [2, 2]]],

      // Edge cases
      [[]], // Empty args
      [[[0, 1]]], // Single element axis
      [[[0, 1], [1, 1]]], // Multiple single element axes
      [[[3, 4]]],
      [[[2, 3]]],
      [[[2, 3], [3, 4]]],
    ],
    _choices_from_args,
    'out(tiny.codegen.rewriter._choices_from_args(*data))',
  ),
)

describe(
  '_swizzle_args',
  compare(
    [
      // Basic test with single axis
      [[[0, 2]], [[0, 2]], []],
      [[[0, 3]], [[0, 3]], []],
      [[[0, 4]], [[0, 4]], []],

      // Test with multiple axes
      [[[0, 2], [1, 2]], [[0, 2], [1, 2]], []],
      [[[0, 2], [1, 3]], [[0, 2], [1, 3]], []],
      [[[0, 3], [1, 2]], [[0, 3], [1, 2]], []],

      // Test with exclude args
      [[[0, 2]], [[0, 2]], [0]],
      [[[0, 2], [1, 3]], [[0, 2], [1, 3]], [0]],
      [[[0, 2], [1, 3]], [[0, 2], [1, 3]], [1]],
      [[[0, 2], [1, 3]], [[0, 2], [1, 3]], [0, 1]],

      // Edge cases
      [[], [], []], // Empty args
      [[[0, 1]], [[0, 1]], []], // Single element axis
      [[[0, 1]], [[0, 1]], [0]], // Single element with exclude
      [[[2, 3], [3, 4]], [[2, 3]], []],
      [[[2, 3], [3, 4]], [[3, 4]], []],
      [[[2, 3], [3, 4]], [[3, 4]], []],
      [[[2, 3], [3, 4]], [[2, 3]], []],
    ],
    _swizzle_args,
    'out(tiny.codegen.rewriter._swizzle_args(*data))',
  ),
)

describe(
  'do_expand',
  compare(
    [
      // Basic test with no expands
      [
        new UOp(Ops.ADD, dtypes.float32, [
          UOp.const(dtypes.float32, 1),
          UOp.const(dtypes.float32, 2),
        ]),
      ],

      // Basic expand test
      [
        new UOp(Ops.ADD, dtypes.float32, [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[
            0,
            2,
          ]]),
          UOp.const(dtypes.float32, 2),
        ]),
      ],

      // Multiple expands with same args
      [
        new UOp(Ops.ADD, dtypes.float32, [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[
            0,
            2,
          ]]),
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 2)], [[
            0,
            2,
          ]]),
        ]),
      ],

      // Multiple expands with different args
      [
        new UOp(Ops.ADD, dtypes.float32, [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[
            0,
            2,
          ]]),
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 2)], [[
            1,
            3,
          ]]),
        ]),
      ],

      // Test with vectorized dtype
      [
        new UOp(Ops.ADD, dtypes.float32.vec(2), [
          new UOp(Ops.EXPAND, dtypes.float32.vec(2), [
            UOp.const(dtypes.float32.vec(2), [1, 2]),
          ], [[0, 2]]),
          UOp.const(dtypes.float32.vec(2), [3, 4]),
        ]),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.VCONST, dtypes.int.vec(3), [], [0, 1, 2]),
          ], [[2, 3]]),
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
        ], undefined),
      ],
      [
        new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
          new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int.vec(3), [
              new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
              ], undefined),
              new UOp(Ops.VCONST, dtypes.int.vec(3), [], [0, 1, 2]),
            ], undefined),
          ], [[2, 3]]),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int.vec(3), [
              new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
              ], undefined),
              new UOp(Ops.VCONST, dtypes.int.vec(3), [], [0, 1, 2]),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 15000),
            ], 1),
            new UOp(Ops.CONST, dtypes.int, [], 4),
          ], undefined),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int.vec(3), [
              new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
              ], undefined),
              new UOp(Ops.VCONST, dtypes.int.vec(3), [], [0, 1, 2]),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.VCONST, dtypes.int.vec(4), [], [0, 1, 2, 3]),
          ], [[3, 4]]),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59998,
                    59997,
                    59996,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59999,
                    59998,
                    59997,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], [[2, 3]]),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int.vec(3), [
              new UOp(Ops.ADD, dtypes.int.vec(3), [
                new UOp(Ops.ADD, dtypes.int.vec(3), [
                  new UOp(Ops.ADD, dtypes.int.vec(3), [
                    new UOp(Ops.IDIV, dtypes.int.vec(3), [
                      new UOp(Ops.ADD, dtypes.int.vec(3), [
                        new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                          -59999,
                          -59998,
                          -59997,
                        ]),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.int.vec(3), [], 4),
                    ], undefined),
                    new UOp(Ops.IDIV, dtypes.int.vec(3), [
                      new UOp(Ops.ADD, dtypes.int.vec(3), [
                        new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                          -59998,
                          -59997,
                          -59996,
                        ]),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.int.vec(3), [], 4),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.IDIV, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int.vec(3), [
                      new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                        -59997,
                        -59996,
                        -59995,
                      ]),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.int.vec(3), [], 4),
                  ], undefined),
                ], undefined),
                new UOp(Ops.IDIV, dtypes.int.vec(3), [
                  new UOp(Ops.ADD, dtypes.int.vec(3), [
                    new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                      -59996,
                      -59995,
                      -59994,
                    ]),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int.vec(3), [], 4),
                ], undefined),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 60000),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.CONST, dtypes.int, [], -1),
        ], undefined),
      ],
    ],
    tryCatch(do_expand),
    'out(tiny.codegen.rewriter.do_expand(*data))',
  ),
)

describe(
  'do_contract',
  compare(
    [
      // CONTRACT without EXPAND
      [
        new UOp(Ops.CONTRACT, dtypes.float32.vec(2), [
          UOp.const(dtypes.float32, 1),
        ], [[0, 2]]),
      ],

      // CONTRACT with EXPAND - removing one axis
      [
        new UOp(Ops.CONTRACT, dtypes.float32.vec(2), [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[
            0,
            2,
          ], [1, 3]]),
        ], [[0, 2]]),
      ],

      // CONTRACT with EXPAND - removing multiple axes
      [
        new UOp(Ops.CONTRACT, dtypes.float32.vec(6), [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [
            [0, 2],
            [1, 3],
            [2, 4],
          ]),
        ], [[0, 2], [1, 3]]),
      ],
      [
        new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(12), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(12), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(12), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(12), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(12), [], [
                    59999,
                    59998,
                    59997,
                    59996,
                    59998,
                    59997,
                    59996,
                    59995,
                    59997,
                    59996,
                    59995,
                    59994,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(12), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(12), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(12), [], 0),
            ], undefined),
          ], [[2, 3], [3, 4]]),
        ], [[3, 4]]),
      ],
    ],
    do_contract,
    'out(tiny.codegen.rewriter.do_contract(*data))',
  ),
)

describe(
  'no_vectorized_alu',
  compare(
    [
      // No change for scalar ALU
      [
        new UOp(Ops.ADD, dtypes.float32, [
          UOp.const(dtypes.float32, 1),
          UOp.const(dtypes.float32, 2),
        ], undefined),
      ],

      // Vectorized ALU gets split into scalar ops
      [
        new UOp(Ops.ADD, dtypes.float32.vec(2), [
          UOp.const(dtypes.float32.vec(2), [1, 2]),
          UOp.const(dtypes.float32.vec(2), [3, 4]),
        ], undefined),
      ],

      // Larger vector size
      [
        new UOp(Ops.MUL, dtypes.float32.vec(4), [
          UOp.const(dtypes.float32.vec(4), [1, 2, 3, 4]),
          UOp.const(dtypes.float32.vec(4), [5, 6, 7, 8]),
        ], undefined),
      ],
      [
        new UOp(Ops.MUL, dtypes.int, [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20000),
          ], 0),
          new UOp(Ops.CONST, dtypes.int, [], 3),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
          new UOp(Ops.CONST, dtypes.int, [], 2),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
          new UOp(Ops.CONST, dtypes.int, [], 3),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
          new UOp(Ops.CONST, dtypes.int, [], 1),
        ], undefined),
      ],
      [
        new UOp(Ops.IDIV, dtypes.int, [
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
          new UOp(Ops.CONST, dtypes.int, [], 4),
        ], undefined),
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
          new UOp(Ops.CONST, dtypes.int, [], -59997),
        ], undefined),
      ],
    ],
    no_vectorized_alu,
    'out(tiny.codegen.rewriter.no_vectorized_alu(*data))',
  ),
)

describe(
  'create_gate',
  compare(
    [
      // No change for non-INDEX operation
      [
        new UOp(
          Ops.ADD,
          dtypes.float32,
          [UOp.const(dtypes.float32, 1)],
          undefined,
        ),
      ],

      // No change when INDEX has only 2 sources
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.int32, [
            UOp.const(dtypes.int32, 0),
            UOp.const(dtypes.int32, 1),
          ], undefined),
        ], undefined),
      ],

      // Gates loads after an INDEX with barrier
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.int32, [
            UOp.const(dtypes.int32, 0),
            UOp.const(dtypes.int32, 1),
            new UOp(Ops.BARRIER, dtypes.void, [], undefined),
          ], undefined),
        ], undefined),
      ],
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.UNROLL, dtypes.int, [
                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [0, 1, 2]),
              ], [[2, 3]]),
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.ASSIGN, dtypes.int, [
              new UOp(Ops.DEFINE_ACC, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 15000),
                ], 1),
              ], [0]),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.DEFINE_ACC, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 15000),
                  ], 1),
                ], [0]),
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.GEP, dtypes.int, [
                    new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                      new UOp(Ops.WHERE, dtypes.int, [
                        new UOp(Ops.CMPNE, dtypes.bool, [
                          new UOp(Ops.CMPLT, dtypes.bool, [
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.UNROLL, dtypes.int, [
                                new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                  0,
                                  1,
                                  2,
                                  3,
                                ]),
                              ], [[3, 4]]),
                              new UOp(Ops.ADD, dtypes.int, [
                                new UOp(Ops.ADD, dtypes.int, [
                                  new UOp(Ops.UNROLL, dtypes.int, [
                                    new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                      0,
                                      1,
                                      2,
                                    ]),
                                  ], [[2, 3]]),
                                  new UOp(Ops.MUL, dtypes.int, [
                                    new UOp(Ops.RANGE, dtypes.int, [
                                      new UOp(Ops.CONST, dtypes.int, [], 0),
                                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                                    ], 0),
                                    new UOp(Ops.CONST, dtypes.int, [], 3),
                                  ], undefined),
                                ], undefined),
                                new UOp(Ops.MUL, dtypes.int, [
                                  new UOp(Ops.RANGE, dtypes.int, [
                                    new UOp(Ops.CONST, dtypes.int, [], 0),
                                    new UOp(Ops.CONST, dtypes.int, [], 15000),
                                  ], 1),
                                  new UOp(Ops.CONST, dtypes.int, [], 4),
                                ], undefined),
                              ], undefined),
                            ], undefined),
                            new UOp(Ops.CONST, dtypes.int, [], 59999),
                          ], undefined),
                          new UOp(Ops.CONST, dtypes.bool, [], true),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.int, [], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                      ], undefined),
                    ], [[3, 4]]),
                  ], [3]),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.GEP, dtypes.int, [
                      new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                        new UOp(Ops.WHERE, dtypes.int, [
                          new UOp(Ops.CMPNE, dtypes.bool, [
                            new UOp(Ops.CMPLT, dtypes.bool, [
                              new UOp(Ops.ADD, dtypes.int, [
                                new UOp(Ops.UNROLL, dtypes.int, [
                                  new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                    0,
                                    1,
                                    2,
                                    3,
                                  ]),
                                ], [[3, 4]]),
                                new UOp(Ops.ADD, dtypes.int, [
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.UNROLL, dtypes.int, [
                                      new UOp(
                                        Ops.VCONST,
                                        dtypes.int.vec(3),
                                        [],
                                        [0, 1, 2],
                                      ),
                                    ], [[2, 3]]),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.RANGE, dtypes.int, [
                                        new UOp(Ops.CONST, dtypes.int, [], 0),
                                        new UOp(
                                          Ops.CONST,
                                          dtypes.int,
                                          [],
                                          20000,
                                        ),
                                      ], 0),
                                      new UOp(Ops.CONST, dtypes.int, [], 3),
                                    ], undefined),
                                  ], undefined),
                                  new UOp(Ops.MUL, dtypes.int, [
                                    new UOp(Ops.RANGE, dtypes.int, [
                                      new UOp(Ops.CONST, dtypes.int, [], 0),
                                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                                    ], 1),
                                    new UOp(Ops.CONST, dtypes.int, [], 4),
                                  ], undefined),
                                ], undefined),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.int, [], 59999),
                            ], undefined),
                            new UOp(Ops.CONST, dtypes.bool, [], true),
                          ], undefined),
                          new UOp(Ops.CONST, dtypes.int, [], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                        ], undefined),
                      ], [[3, 4]]),
                    ], [2]),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.GEP, dtypes.int, [
                        new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                          new UOp(Ops.WHERE, dtypes.int, [
                            new UOp(Ops.CMPNE, dtypes.bool, [
                              new UOp(Ops.CMPLT, dtypes.bool, [
                                new UOp(Ops.ADD, dtypes.int, [
                                  new UOp(Ops.UNROLL, dtypes.int, [
                                    new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                      0,
                                      1,
                                      2,
                                      3,
                                    ]),
                                  ], [[3, 4]]),
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.UNROLL, dtypes.int, [
                                        new UOp(
                                          Ops.VCONST,
                                          dtypes.int.vec(3),
                                          [],
                                          [0, 1, 2],
                                        ),
                                      ], [[2, 3]]),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(
                                            Ops.CONST,
                                            dtypes.int,
                                            [],
                                            20000,
                                          ),
                                        ], 0),
                                        new UOp(Ops.CONST, dtypes.int, [], 3),
                                      ], undefined),
                                    ], undefined),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.RANGE, dtypes.int, [
                                        new UOp(Ops.CONST, dtypes.int, [], 0),
                                        new UOp(
                                          Ops.CONST,
                                          dtypes.int,
                                          [],
                                          15000,
                                        ),
                                      ], 1),
                                      new UOp(Ops.CONST, dtypes.int, [], 4),
                                    ], undefined),
                                  ], undefined),
                                ], undefined),
                                new UOp(Ops.CONST, dtypes.int, [], 59999),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.bool, [], true),
                            ], undefined),
                            new UOp(Ops.CONST, dtypes.int, [], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                          ], undefined),
                        ], [[3, 4]]),
                      ], [0]),
                      new UOp(Ops.GEP, dtypes.int, [
                        new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                          new UOp(Ops.WHERE, dtypes.int, [
                            new UOp(Ops.CMPNE, dtypes.bool, [
                              new UOp(Ops.CMPLT, dtypes.bool, [
                                new UOp(Ops.ADD, dtypes.int, [
                                  new UOp(Ops.UNROLL, dtypes.int, [
                                    new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                      0,
                                      1,
                                      2,
                                      3,
                                    ]),
                                  ], [[3, 4]]),
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.UNROLL, dtypes.int, [
                                        new UOp(
                                          Ops.VCONST,
                                          dtypes.int.vec(3),
                                          [],
                                          [0, 1, 2],
                                        ),
                                      ], [[2, 3]]),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(
                                            Ops.CONST,
                                            dtypes.int,
                                            [],
                                            20000,
                                          ),
                                        ], 0),
                                        new UOp(Ops.CONST, dtypes.int, [], 3),
                                      ], undefined),
                                    ], undefined),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.RANGE, dtypes.int, [
                                        new UOp(Ops.CONST, dtypes.int, [], 0),
                                        new UOp(
                                          Ops.CONST,
                                          dtypes.int,
                                          [],
                                          15000,
                                        ),
                                      ], 1),
                                      new UOp(Ops.CONST, dtypes.int, [], 4),
                                    ], undefined),
                                  ], undefined),
                                ], undefined),
                                new UOp(Ops.CONST, dtypes.int, [], 59999),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.bool, [], true),
                            ], undefined),
                            new UOp(Ops.CONST, dtypes.int, [], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                          ], undefined),
                        ], [[3, 4]]),
                      ], [1]),
                    ], undefined),
                  ], undefined),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int, [], -1),
          ], undefined),
        ], undefined),
      ],
    ],
    tryCatch(create_gate),
    'out(tiny.codegen.rewriter.create_gate(*data))',
  ),
)

describe(
  'no_vectorized_load_store',
  compare(
    [
      // No change for non-pointer dtype
      [
        new UOp(
          Ops.LOAD,
          dtypes.float32,
          [UOp.const(dtypes.float32, 1)],
          undefined,
        ),
      ],

      // No change for vector size 1
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(
            Ops.INDEX,
            dtypes.float32.ptr(),
            [UOp.const(dtypes.int32, 0)],
            undefined,
          ),
        ], undefined),
      ],

      // Devectorizes load with vector size > 1
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(
            Ops.INDEX,
            dtypes.float32.ptr(),
            [UOp.const(dtypes.int32, 0)],
            undefined,
          ),
        ], undefined),
      ],

      // Devectorizes store with vector size > 1
      [
        new UOp(Ops.STORE, dtypes.float32, [
          new UOp(
            Ops.INDEX,
            dtypes.float32.ptr(),
            [UOp.const(dtypes.int32, 0)],
            undefined,
          ),
          UOp.const(dtypes.float32, 1),
        ], undefined),
      ],
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
          ], undefined),
          new UOp(Ops.MUL, dtypes.int, [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 20000),
            ], 0),
            new UOp(Ops.CONST, dtypes.int, [], 3),
          ], undefined),
        ], undefined),
      ],
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 2),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int, [], 2),
          ], undefined),
        ], undefined),
      ],
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 1),
            ], undefined),
          ], undefined),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int, [], 1),
          ], undefined),
        ], undefined),
      ],
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.VECTORIZE, dtypes.int.ptr(-1).vec(3), [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
            ], undefined),
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
            ], undefined),
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
          ], undefined),
          new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 1),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 2),
            ], undefined),
          ], undefined),
        ], undefined),
      ],
    ],
    tryCatch(no_vectorized_load_store),
    'out(trycatch(lambda:tiny.codegen.rewriter.no_vectorized_load_store(*data)))',
  ),
)

describe(
  'delete_redundant_gates',
  compare(
    [
      // Case 1: Store with gate that should be removed
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [
          UOp.const(dtypes.int32, 0),
          UOp.const(dtypes.int32, 1),
          UOp.const(dtypes.bool, true),
        ], undefined),
        UOp.const(dtypes.int32, 1),
        new UOp(Ops.IF, dtypes.void, [
          UOp.const(dtypes.bool, true),
          UOp.const(dtypes.int32, 1),
        ], undefined),

        new UOp(
          Ops.CAST,
          dtypes.float32.ptr(),
          [UOp.const(dtypes.int32, 0)],
          undefined,
        ),
      ],

      // Case 2: Store with gate and cast that should be removed
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [
          UOp.const(dtypes.int32, 0),
          UOp.const(dtypes.int32, 1),
          UOp.const(dtypes.bool, true),
        ], undefined),
        UOp.const(dtypes.int32, 1),
        new UOp(Ops.IF, dtypes.void, [
          UOp.const(dtypes.bool, true),
          UOp.const(dtypes.int32, 1),
        ], undefined),
        new UOp(
          Ops.CAST,
          dtypes.float32.ptr(),
          [UOp.const(dtypes.int32, 0)],
          undefined,
        ),
      ],

      // Case 3: Store without matching gate (should return undefined)
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [
          UOp.const(dtypes.int32, 0),
          UOp.const(dtypes.int32, 1),
          UOp.const(dtypes.bool, true),
        ], undefined),
        UOp.const(dtypes.int32, 1),
        new UOp(Ops.IF, dtypes.void, [
          UOp.const(dtypes.bool, false),
          UOp.const(dtypes.int32, 1),
        ], undefined),

        new UOp(
          Ops.CAST,
          dtypes.float32.ptr(),
          [UOp.const(dtypes.int32, 0)],
          undefined,
        ),
      ],

      // Case 4: Store with multiple gates but no matching one
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [
          UOp.const(dtypes.int32, 0),
          UOp.const(dtypes.int32, 1),
          UOp.const(dtypes.bool, true),
        ], undefined),
        new UOp(Ops.IF, dtypes.void, [
          UOp.const(dtypes.bool, false),
          UOp.const(dtypes.int32, 2),
        ], undefined),
        new UOp(Ops.IF, dtypes.void, [
          UOp.const(dtypes.bool, false),
          UOp.const(dtypes.int32, 3),
        ], undefined),
        new UOp(
          Ops.CAST,
          dtypes.float32.ptr(),
          [UOp.const(dtypes.int32, 0)],
          undefined,
        ),
      ],
    ],
    delete_redundant_gates,
    'out(tiny.codegen.rewriter.delete_redundant_gates(*data))',
  ),
)

describe(
  'move_mask',
  compare(
    [
      // Case 1: Load with mask
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [
            UOp.const(dtypes.int32, 0),
            UOp.const(dtypes.int32, 1),
            UOp.const(dtypes.bool, true),
          ], undefined),
        ], undefined),
        UOp.const(dtypes.int32, 0),
        UOp.const(dtypes.int32, 1),
        UOp.const(dtypes.bool, true),
      ],

      // Case 2: Store with mask
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [
            UOp.const(dtypes.int32, 0),
            UOp.const(dtypes.int32, 1),
            UOp.const(dtypes.bool, true),
          ], undefined),
          UOp.const(dtypes.float32, 1.0),
        ], undefined),
        UOp.const(dtypes.int32, 0),
        UOp.const(dtypes.int32, 1),
        UOp.const(dtypes.bool, true),
      ],

      // Case 3: Load with mask and cast
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.CAST, dtypes.float32.ptr(), [
            new UOp(Ops.INDEX, dtypes.int32.ptr(), [
              UOp.const(dtypes.int32, 0),
              UOp.const(dtypes.int32, 1),
              UOp.const(dtypes.bool, true),
            ], undefined),
          ], undefined),
        ], undefined),
        UOp.const(dtypes.int32, 0),
        UOp.const(dtypes.int32, 1),
        UOp.const(dtypes.bool, true),
        new UOp(
          Ops.CAST,
          dtypes.float32.ptr(),
          [UOp.const(dtypes.int32, 0)],
          undefined,
        ),
      ],
    ],
    move_mask,
    'out(tiny.codegen.rewriter.move_mask(*data))',
  ),
)
describe(
  'full_graph_rewrite',
  compare(
    [
      [
        new UOp(
          Ops.SINK,
          dtypes.void,
          [UOp.const(dtypes.float32, 1.0)],
          undefined,
        ),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.LOAD, dtypes.float32.vec(4), [
            new UOp(Ops.INDEX, dtypes.float32.ptr().vec(4), [
              UOp.const(dtypes.int32, 0),
            ], undefined),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.ADD, dtypes.float32, [
            UOp.const(dtypes.float32, 1.0),
            UOp.const(dtypes.float32, 2.0),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.LOAD, dtypes.float32.vec(4), [
            new UOp(Ops.INDEX, dtypes.float32.ptr().vec(4), [
              UOp.const(dtypes.int32, 0),
            ], undefined),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(
            Ops.EXP2,
            dtypes.float32,
            [UOp.const(dtypes.float32, 1.0)],
            undefined,
          ),
        ], undefined),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float32.ptr(), [
              UOp.const(dtypes.int32, 0),
              UOp.const(dtypes.int32, 0),
            ], undefined),
            UOp.const(dtypes.float32, 1.0),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.LOAD, dtypes.float32, [
            new UOp(Ops.INDEX, dtypes.float32.ptr(), [
              UOp.const(dtypes.int32, 0),
              UOp.const(dtypes.int32, 1),
              UOp.const(dtypes.bool, true),
            ], undefined),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.int.ptr(60000), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(60000), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.UNROLL, dtypes.int, [
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [0, 1, 2]),
                ], [[2, 3]]),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.ASSIGN, dtypes.int, [
                new UOp(Ops.DEFINE_ACC, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 15000),
                  ], 1),
                ], [0]),
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.DEFINE_ACC, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                    ], 1),
                  ], [0]),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.GEP, dtypes.int, [
                          new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                            new UOp(Ops.WHERE, dtypes.int, [
                              new UOp(Ops.CMPNE, dtypes.bool, [
                                new UOp(Ops.CMPLT, dtypes.bool, [
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.UNROLL, dtypes.int, [
                                      new UOp(
                                        Ops.VCONST,
                                        dtypes.int.vec(4),
                                        [],
                                        [0, 1, 2, 3],
                                      ),
                                    ], [[3, 4]]),
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.ADD, dtypes.int, [
                                        new UOp(Ops.UNROLL, dtypes.int, [
                                          new UOp(
                                            Ops.VCONST,
                                            dtypes.int.vec(3),
                                            [],
                                            [0, 1, 2],
                                          ),
                                        ], [[2, 3]]),
                                        new UOp(Ops.MUL, dtypes.int, [
                                          new UOp(Ops.RANGE, dtypes.int, [
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              0,
                                            ),
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              20000,
                                            ),
                                          ], 0),
                                          new UOp(Ops.CONST, dtypes.int, [], 3),
                                        ], undefined),
                                      ], undefined),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(
                                            Ops.CONST,
                                            dtypes.int,
                                            [],
                                            15000,
                                          ),
                                        ], 1),
                                        new UOp(Ops.CONST, dtypes.int, [], 4),
                                      ], undefined),
                                    ], undefined),
                                  ], undefined),
                                  new UOp(Ops.CONST, dtypes.int, [], 59999),
                                ], undefined),
                                new UOp(Ops.CONST, dtypes.bool, [], true),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.int, [], 1),
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                            ], undefined),
                          ], [[3, 4]]),
                        ], [0]),
                        new UOp(Ops.GEP, dtypes.int, [
                          new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                            new UOp(Ops.WHERE, dtypes.int, [
                              new UOp(Ops.CMPNE, dtypes.bool, [
                                new UOp(Ops.CMPLT, dtypes.bool, [
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.UNROLL, dtypes.int, [
                                      new UOp(
                                        Ops.VCONST,
                                        dtypes.int.vec(4),
                                        [],
                                        [0, 1, 2, 3],
                                      ),
                                    ], [[3, 4]]),
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.ADD, dtypes.int, [
                                        new UOp(Ops.UNROLL, dtypes.int, [
                                          new UOp(
                                            Ops.VCONST,
                                            dtypes.int.vec(3),
                                            [],
                                            [0, 1, 2],
                                          ),
                                        ], [[2, 3]]),
                                        new UOp(Ops.MUL, dtypes.int, [
                                          new UOp(Ops.RANGE, dtypes.int, [
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              0,
                                            ),
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              20000,
                                            ),
                                          ], 0),
                                          new UOp(Ops.CONST, dtypes.int, [], 3),
                                        ], undefined),
                                      ], undefined),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(
                                            Ops.CONST,
                                            dtypes.int,
                                            [],
                                            15000,
                                          ),
                                        ], 1),
                                        new UOp(Ops.CONST, dtypes.int, [], 4),
                                      ], undefined),
                                    ], undefined),
                                  ], undefined),
                                  new UOp(Ops.CONST, dtypes.int, [], 59999),
                                ], undefined),
                                new UOp(Ops.CONST, dtypes.bool, [], true),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.int, [], 1),
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                            ], undefined),
                          ], [[3, 4]]),
                        ], [1]),
                      ], undefined),
                      new UOp(Ops.GEP, dtypes.int, [
                        new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                          new UOp(Ops.WHERE, dtypes.int, [
                            new UOp(Ops.CMPNE, dtypes.bool, [
                              new UOp(Ops.CMPLT, dtypes.bool, [
                                new UOp(Ops.ADD, dtypes.int, [
                                  new UOp(Ops.UNROLL, dtypes.int, [
                                    new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                      0,
                                      1,
                                      2,
                                      3,
                                    ]),
                                  ], [[3, 4]]),
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.UNROLL, dtypes.int, [
                                        new UOp(
                                          Ops.VCONST,
                                          dtypes.int.vec(3),
                                          [],
                                          [0, 1, 2],
                                        ),
                                      ], [[2, 3]]),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(
                                            Ops.CONST,
                                            dtypes.int,
                                            [],
                                            20000,
                                          ),
                                        ], 0),
                                        new UOp(Ops.CONST, dtypes.int, [], 3),
                                      ], undefined),
                                    ], undefined),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.RANGE, dtypes.int, [
                                        new UOp(Ops.CONST, dtypes.int, [], 0),
                                        new UOp(
                                          Ops.CONST,
                                          dtypes.int,
                                          [],
                                          15000,
                                        ),
                                      ], 1),
                                      new UOp(Ops.CONST, dtypes.int, [], 4),
                                    ], undefined),
                                  ], undefined),
                                ], undefined),
                                new UOp(Ops.CONST, dtypes.int, [], 59999),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.bool, [], true),
                            ], undefined),
                            new UOp(Ops.CONST, dtypes.int, [], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                          ], undefined),
                        ], [[3, 4]]),
                      ], [2]),
                    ], undefined),
                    new UOp(Ops.GEP, dtypes.int, [
                      new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                        new UOp(Ops.WHERE, dtypes.int, [
                          new UOp(Ops.CMPNE, dtypes.bool, [
                            new UOp(Ops.CMPLT, dtypes.bool, [
                              new UOp(Ops.ADD, dtypes.int, [
                                new UOp(Ops.UNROLL, dtypes.int, [
                                  new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                    0,
                                    1,
                                    2,
                                    3,
                                  ]),
                                ], [[3, 4]]),
                                new UOp(Ops.ADD, dtypes.int, [
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.UNROLL, dtypes.int, [
                                      new UOp(
                                        Ops.VCONST,
                                        dtypes.int.vec(3),
                                        [],
                                        [0, 1, 2],
                                      ),
                                    ], [[2, 3]]),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.RANGE, dtypes.int, [
                                        new UOp(Ops.CONST, dtypes.int, [], 0),
                                        new UOp(
                                          Ops.CONST,
                                          dtypes.int,
                                          [],
                                          20000,
                                        ),
                                      ], 0),
                                      new UOp(Ops.CONST, dtypes.int, [], 3),
                                    ], undefined),
                                  ], undefined),
                                  new UOp(Ops.MUL, dtypes.int, [
                                    new UOp(Ops.RANGE, dtypes.int, [
                                      new UOp(Ops.CONST, dtypes.int, [], 0),
                                      new UOp(Ops.CONST, dtypes.int, [], 15000),
                                    ], 1),
                                    new UOp(Ops.CONST, dtypes.int, [], 4),
                                  ], undefined),
                                ], undefined),
                              ], undefined),
                              new UOp(Ops.CONST, dtypes.int, [], 59999),
                            ], undefined),
                            new UOp(Ops.CONST, dtypes.bool, [], true),
                          ], undefined),
                          new UOp(Ops.CONST, dtypes.int, [], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                        ], undefined),
                      ], [[3, 4]]),
                    ], [3]),
                  ], undefined),
                ], undefined),
              ], undefined),
              new UOp(Ops.WHERE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.bool, [], true),
                new UOp(Ops.CONST, dtypes.int, [], -1),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 2, false)),
        new ClangRenderer(),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(262176), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(262176), [], 0),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], [
                      'gidx0',
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.DEFINE_VAR, dtypes.int, [], [
                          'start_pos',
                          1,
                          8192,
                        ]),
                        new UOp(Ops.CONST, dtypes.int, [], 1),
                      ], undefined),
                    ]),
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.SPECIAL, dtypes.int, [], ['gidx1', 2]),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.DEFINE_VAR, dtypes.int, [], [
                            'start_pos',
                            1,
                            8192,
                          ]),
                          new UOp(Ops.CONST, dtypes.int, [], 16),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.int, [], 16),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], ['lidx0', 4]),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.DEFINE_VAR, dtypes.int, [], [
                        'start_pos',
                        1,
                        8192,
                      ]),
                      new UOp(Ops.CONST, dtypes.int, [], 1),
                    ], undefined),
                  ], undefined),
                ], undefined),
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.UNROLL, dtypes.int, [
                    new UOp(Ops.VCONST, dtypes.int.vec(4), [], [0, 1, 2, 3]),
                  ], [[5, 4]]),
                  new UOp(Ops.ADD, dtypes.int, [
                    new UOp(Ops.MUL, dtypes.int, [
                      new UOp(Ops.DEFINE_VAR, dtypes.int, [], [
                        'start_pos',
                        1,
                        8192,
                      ]),
                      new UOp(Ops.CONST, dtypes.int, [], 4),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.int, [], 4),
                  ], undefined),
                ], undefined),
              ], undefined),
              new UOp(Ops.MUL, dtypes.bool, [
                new UOp(Ops.CONST, dtypes.bool, [], true),
                new UOp(Ops.CMPNE, dtypes.bool, [
                  new UOp(Ops.CMPNE, dtypes.bool, [
                    new UOp(Ops.SPECIAL, dtypes.int, [], ['lidx1', 8]),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.bool, [], true),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.MUL, dtypes.float, [
              new UOp(Ops.ASSIGN, dtypes.float, [
                new UOp(Ops.DEFINE_ACC, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 8),
                  ], 1003),
                ], [0]),
                new UOp(Ops.ADD, dtypes.float, [
                  new UOp(Ops.DEFINE_ACC, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 0),
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 8),
                    ], 1003),
                  ], [0]),
                  new UOp(Ops.LOAD, dtypes.float, [
                    new UOp(Ops.INDEX, dtypes.float.ptr(128, true), [
                      new UOp(
                        Ops.DEFINE_LOCAL,
                        dtypes.float.ptr(128, true),
                        [],
                        ['temp1', 128],
                      ),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.UNROLL, dtypes.int, [
                          new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                            0,
                            1,
                            2,
                            3,
                          ]),
                        ], [[5, 4]]),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.SPECIAL, dtypes.int, [], ['lidx0', 4]),
                            new UOp(Ops.CONST, dtypes.int, [], 32),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 8),
                            ], 1003),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.BARRIER, dtypes.void, [
                      new UOp(Ops.STORE, dtypes.void, [
                        new UOp(Ops.INDEX, dtypes.float.ptr(128, true), [
                          new UOp(
                            Ops.DEFINE_LOCAL,
                            dtypes.float.ptr(128, true),
                            [],
                            ['temp1', 128],
                          ),
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.UNROLL, dtypes.int, [
                              new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                0,
                                1,
                                2,
                                3,
                              ]),
                            ], [[5, 4]]),
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.SPECIAL, dtypes.int, [], [
                                  'lidx0',
                                  4,
                                ]),
                                new UOp(Ops.CONST, dtypes.int, [], 32),
                              ], undefined),
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.SPECIAL, dtypes.int, [], [
                                  'lidx1',
                                  8,
                                ]),
                                new UOp(Ops.CONST, dtypes.int, [], 4),
                              ], undefined),
                            ], undefined),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.ASSIGN, dtypes.float, [
                          new UOp(Ops.DEFINE_ACC, dtypes.float, [
                            new UOp(Ops.CONST, dtypes.float, [], 0),
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 8),
                            ], 4),
                          ], [0]),
                          new UOp(Ops.ADD, dtypes.float, [
                            new UOp(Ops.DEFINE_ACC, dtypes.float, [
                              new UOp(Ops.CONST, dtypes.float, [], 0),
                              new UOp(Ops.RANGE, dtypes.int, [
                                new UOp(Ops.CONST, dtypes.int, [], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 8),
                              ], 4),
                            ], [0]),
                            new UOp(Ops.MUL, dtypes.float, [
                              new UOp(Ops.LOAD, dtypes.float, [
                                new UOp(Ops.INDEX, dtypes.float.ptr(2048), [
                                  new UOp(
                                    Ops.DEFINE_GLOBAL,
                                    dtypes.float.ptr(2048),
                                    [],
                                    1,
                                  ),
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.ADD, dtypes.int, [
                                        new UOp(Ops.SPECIAL, dtypes.int, [], [
                                          'lidx1',
                                          8,
                                        ]),
                                        new UOp(Ops.ADD, dtypes.int, [
                                          new UOp(Ops.MUL, dtypes.int, [
                                            new UOp(
                                              Ops.SPECIAL,
                                              dtypes.int,
                                              [],
                                              ['gidx1', 2],
                                            ),
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              1024,
                                            ),
                                          ], undefined),
                                          new UOp(Ops.MUL, dtypes.int, [
                                            new UOp(
                                              Ops.SPECIAL,
                                              dtypes.int,
                                              [],
                                              ['lidx0', 4],
                                            ),
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              64,
                                            ),
                                          ], undefined),
                                        ], undefined),
                                      ], undefined),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(Ops.CONST, dtypes.int, [], 8),
                                        ], 4),
                                        new UOp(Ops.CONST, dtypes.int, [], 8),
                                      ], undefined),
                                    ], undefined),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.UNROLL, dtypes.int, [
                                        new UOp(
                                          Ops.VCONST,
                                          dtypes.int.vec(4),
                                          [],
                                          [0, 1, 2, 3],
                                        ),
                                      ], [[5, 4]]),
                                      new UOp(Ops.CONST, dtypes.int, [], 256),
                                    ], undefined),
                                  ], undefined),
                                ], undefined),
                              ], undefined),
                              new UOp(Ops.LOAD, dtypes.float, [
                                new UOp(Ops.INDEX, dtypes.float.ptr(8388608), [
                                  new UOp(
                                    Ops.DEFINE_GLOBAL,
                                    dtypes.float.ptr(8388608),
                                    [],
                                    2,
                                  ),
                                  new UOp(Ops.ADD, dtypes.int, [
                                    new UOp(Ops.ADD, dtypes.int, [
                                      new UOp(Ops.SPECIAL, dtypes.int, [], [
                                        'lidx1',
                                        8,
                                      ]),
                                      new UOp(Ops.MUL, dtypes.int, [
                                        new UOp(Ops.RANGE, dtypes.int, [
                                          new UOp(Ops.CONST, dtypes.int, [], 0),
                                          new UOp(Ops.CONST, dtypes.int, [], 8),
                                        ], 4),
                                        new UOp(Ops.CONST, dtypes.int, [], 8),
                                      ], undefined),
                                    ], undefined),
                                    new UOp(Ops.MUL, dtypes.int, [
                                      new UOp(Ops.MOD, dtypes.int, [
                                        new UOp(Ops.ADD, dtypes.int, [
                                          new UOp(Ops.UNROLL, dtypes.int, [
                                            new UOp(
                                              Ops.VCONST,
                                              dtypes.int.vec(4),
                                              [],
                                              [0, 1, 2, 3],
                                            ),
                                          ], [[5, 4]]),
                                          new UOp(Ops.ADD, dtypes.int, [
                                            new UOp(Ops.MUL, dtypes.int, [
                                              new UOp(
                                                Ops.SPECIAL,
                                                dtypes.int,
                                                [],
                                                [
                                                  'gidx0',
                                                  new UOp(Ops.ADD, dtypes.int, [
                                                    new UOp(
                                                      Ops.DEFINE_VAR,
                                                      dtypes.int,
                                                      [],
                                                      ['start_pos', 1, 8192],
                                                    ),
                                                    new UOp(
                                                      Ops.CONST,
                                                      dtypes.int,
                                                      [],
                                                      1,
                                                    ),
                                                  ], undefined),
                                                ],
                                              ),
                                              new UOp(
                                                Ops.CONST,
                                                dtypes.int,
                                                [],
                                                8,
                                              ),
                                            ], undefined),
                                            new UOp(Ops.MUL, dtypes.int, [
                                              new UOp(
                                                Ops.SPECIAL,
                                                dtypes.int,
                                                [],
                                                ['gidx1', 2],
                                              ),
                                              new UOp(
                                                Ops.CONST,
                                                dtypes.int,
                                                [],
                                                4,
                                              ),
                                            ], undefined),
                                          ], undefined),
                                        ], undefined),
                                        new UOp(Ops.ADD, dtypes.int, [
                                          new UOp(Ops.MUL, dtypes.int, [
                                            new UOp(
                                              Ops.DEFINE_VAR,
                                              dtypes.int,
                                              [],
                                              ['start_pos', 1, 8192],
                                            ),
                                            new UOp(
                                              Ops.CONST,
                                              dtypes.int,
                                              [],
                                              8,
                                            ),
                                          ], undefined),
                                          new UOp(Ops.CONST, dtypes.int, [], 8),
                                        ], undefined),
                                      ], undefined),
                                      new UOp(Ops.CONST, dtypes.int, [], 64),
                                    ], undefined),
                                  ], undefined),
                                ], undefined),
                              ], undefined),
                            ], undefined),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                ], undefined),
              ], undefined),
              new UOp(Ops.WHERE, dtypes.float, [
                new UOp(Ops.CONST, dtypes.bool, [], true),
                new UOp(Ops.CONST, dtypes.float, [], 0.125),
                new UOp(Ops.CONST, dtypes.float, [], 0),
              ], undefined),
            ], undefined),
          ], undefined),
        ], new KernelInfo(1, 1, false)),
        new WGSLRenderer(),
      ],
    ],
    tryCatch(full_graph_rewrite),
    'out(tiny.codegen.rewriter.full_graph_rewrite(*data))',
  ),
)

describe(
  'reduce_collapse',
  compare(
    [
      [
        new UOp(Ops.DEFINE_ACC, dtypes.int, [
          new UOp(Ops.CONST, dtypes.int, [], 0),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 15000),
          ], 1),
        ], [0]),
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.GEP, dtypes.int, [
            new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
              new UOp(Ops.WHERE, dtypes.int, [
                new UOp(Ops.CMPNE, dtypes.bool, [
                  new UOp(Ops.CMPLT, dtypes.bool, [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.UNROLL, dtypes.int, [
                        new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                          0,
                          1,
                          2,
                          3,
                        ]),
                      ], [[3, 4]]),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.UNROLL, dtypes.int, [
                            new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                              0,
                              1,
                              2,
                            ]),
                          ], [[2, 3]]),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.int, [], 59999),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.bool, [], true),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
            ], [[3, 4]]),
          ], [3]),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.GEP, dtypes.int, [
              new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                new UOp(Ops.WHERE, dtypes.int, [
                  new UOp(Ops.CMPNE, dtypes.bool, [
                    new UOp(Ops.CMPLT, dtypes.bool, [
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.UNROLL, dtypes.int, [
                          new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                            0,
                            1,
                            2,
                            3,
                          ]),
                        ], [[3, 4]]),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.UNROLL, dtypes.int, [
                              new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                0,
                                1,
                                2,
                              ]),
                            ], [[2, 3]]),
                            new UOp(Ops.MUL, dtypes.int, [
                              new UOp(Ops.RANGE, dtypes.int, [
                                new UOp(Ops.CONST, dtypes.int, [], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 20000),
                              ], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 3),
                            ], undefined),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.int, [], 59999),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.bool, [], true),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int, [], 1),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
              ], [[3, 4]]),
            ], [2]),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.GEP, dtypes.int, [
                new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                  new UOp(Ops.WHERE, dtypes.int, [
                    new UOp(Ops.CMPNE, dtypes.bool, [
                      new UOp(Ops.CMPLT, dtypes.bool, [
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.UNROLL, dtypes.int, [
                            new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                              0,
                              1,
                              2,
                              3,
                            ]),
                          ], [[3, 4]]),
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.UNROLL, dtypes.int, [
                                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                  0,
                                  1,
                                  2,
                                ]),
                              ], [[2, 3]]),
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.RANGE, dtypes.int, [
                                  new UOp(Ops.CONST, dtypes.int, [], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                                ], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 3),
                              ], undefined),
                            ], undefined),
                            new UOp(Ops.MUL, dtypes.int, [
                              new UOp(Ops.RANGE, dtypes.int, [
                                new UOp(Ops.CONST, dtypes.int, [], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 15000),
                              ], 1),
                              new UOp(Ops.CONST, dtypes.int, [], 4),
                            ], undefined),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.int, [], 59999),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.bool, [], true),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.int, [], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                ], [[3, 4]]),
              ], [0]),
              new UOp(Ops.GEP, dtypes.int, [
                new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                  new UOp(Ops.WHERE, dtypes.int, [
                    new UOp(Ops.CMPNE, dtypes.bool, [
                      new UOp(Ops.CMPLT, dtypes.bool, [
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.UNROLL, dtypes.int, [
                            new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                              0,
                              1,
                              2,
                              3,
                            ]),
                          ], [[3, 4]]),
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.UNROLL, dtypes.int, [
                                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                  0,
                                  1,
                                  2,
                                ]),
                              ], [[2, 3]]),
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.RANGE, dtypes.int, [
                                  new UOp(Ops.CONST, dtypes.int, [], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                                ], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 3),
                              ], undefined),
                            ], undefined),
                            new UOp(Ops.MUL, dtypes.int, [
                              new UOp(Ops.RANGE, dtypes.int, [
                                new UOp(Ops.CONST, dtypes.int, [], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 15000),
                              ], 1),
                              new UOp(Ops.CONST, dtypes.int, [], 4),
                            ], undefined),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.int, [], 59999),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.bool, [], true),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.int, [], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                ], [[3, 4]]),
              ], [1]),
            ], undefined),
          ], undefined),
        ], undefined),
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.DEFINE_ACC, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 15000),
            ], 1),
          ], [0]),
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.GEP, dtypes.int, [
              new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                new UOp(Ops.WHERE, dtypes.int, [
                  new UOp(Ops.CMPNE, dtypes.bool, [
                    new UOp(Ops.CMPLT, dtypes.bool, [
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.UNROLL, dtypes.int, [
                          new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                            0,
                            1,
                            2,
                            3,
                          ]),
                        ], [[3, 4]]),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.UNROLL, dtypes.int, [
                              new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                0,
                                1,
                                2,
                              ]),
                            ], [[2, 3]]),
                            new UOp(Ops.MUL, dtypes.int, [
                              new UOp(Ops.RANGE, dtypes.int, [
                                new UOp(Ops.CONST, dtypes.int, [], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 20000),
                              ], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 3),
                            ], undefined),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.int, [], 59999),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.bool, [], true),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int, [], 1),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
              ], [[3, 4]]),
            ], [3]),
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.GEP, dtypes.int, [
                new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                  new UOp(Ops.WHERE, dtypes.int, [
                    new UOp(Ops.CMPNE, dtypes.bool, [
                      new UOp(Ops.CMPLT, dtypes.bool, [
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.UNROLL, dtypes.int, [
                            new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                              0,
                              1,
                              2,
                              3,
                            ]),
                          ], [[3, 4]]),
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.UNROLL, dtypes.int, [
                                new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                  0,
                                  1,
                                  2,
                                ]),
                              ], [[2, 3]]),
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.RANGE, dtypes.int, [
                                  new UOp(Ops.CONST, dtypes.int, [], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                                ], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 3),
                              ], undefined),
                            ], undefined),
                            new UOp(Ops.MUL, dtypes.int, [
                              new UOp(Ops.RANGE, dtypes.int, [
                                new UOp(Ops.CONST, dtypes.int, [], 0),
                                new UOp(Ops.CONST, dtypes.int, [], 15000),
                              ], 1),
                              new UOp(Ops.CONST, dtypes.int, [], 4),
                            ], undefined),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.int, [], 59999),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.bool, [], true),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.int, [], 1),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                ], [[3, 4]]),
              ], [2]),
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.GEP, dtypes.int, [
                  new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                    new UOp(Ops.WHERE, dtypes.int, [
                      new UOp(Ops.CMPNE, dtypes.bool, [
                        new UOp(Ops.CMPLT, dtypes.bool, [
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.UNROLL, dtypes.int, [
                              new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                0,
                                1,
                                2,
                                3,
                              ]),
                            ], [[3, 4]]),
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.ADD, dtypes.int, [
                                new UOp(Ops.UNROLL, dtypes.int, [
                                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                    0,
                                    1,
                                    2,
                                  ]),
                                ], [[2, 3]]),
                                new UOp(Ops.MUL, dtypes.int, [
                                  new UOp(Ops.RANGE, dtypes.int, [
                                    new UOp(Ops.CONST, dtypes.int, [], 0),
                                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                                  ], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 3),
                                ], undefined),
                              ], undefined),
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.RANGE, dtypes.int, [
                                  new UOp(Ops.CONST, dtypes.int, [], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 15000),
                                ], 1),
                                new UOp(Ops.CONST, dtypes.int, [], 4),
                              ], undefined),
                            ], undefined),
                          ], undefined),
                          new UOp(Ops.CONST, dtypes.int, [], 59999),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.bool, [], true),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.int, [], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                    ], undefined),
                  ], [[3, 4]]),
                ], [0]),
                new UOp(Ops.GEP, dtypes.int, [
                  new UOp(Ops.CONTRACT, dtypes.int.vec(4), [
                    new UOp(Ops.WHERE, dtypes.int, [
                      new UOp(Ops.CMPNE, dtypes.bool, [
                        new UOp(Ops.CMPLT, dtypes.bool, [
                          new UOp(Ops.ADD, dtypes.int, [
                            new UOp(Ops.UNROLL, dtypes.int, [
                              new UOp(Ops.VCONST, dtypes.int.vec(4), [], [
                                0,
                                1,
                                2,
                                3,
                              ]),
                            ], [[3, 4]]),
                            new UOp(Ops.ADD, dtypes.int, [
                              new UOp(Ops.ADD, dtypes.int, [
                                new UOp(Ops.UNROLL, dtypes.int, [
                                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                                    0,
                                    1,
                                    2,
                                  ]),
                                ], [[2, 3]]),
                                new UOp(Ops.MUL, dtypes.int, [
                                  new UOp(Ops.RANGE, dtypes.int, [
                                    new UOp(Ops.CONST, dtypes.int, [], 0),
                                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                                  ], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 3),
                                ], undefined),
                              ], undefined),
                              new UOp(Ops.MUL, dtypes.int, [
                                new UOp(Ops.RANGE, dtypes.int, [
                                  new UOp(Ops.CONST, dtypes.int, [], 0),
                                  new UOp(Ops.CONST, dtypes.int, [], 15000),
                                ], 1),
                                new UOp(Ops.CONST, dtypes.int, [], 4),
                              ], undefined),
                            ], undefined),
                          ], undefined),
                          new UOp(Ops.CONST, dtypes.int, [], 59999),
                        ], undefined),
                        new UOp(Ops.CONST, dtypes.bool, [], true),
                      ], undefined),
                      new UOp(Ops.CONST, dtypes.int, [], 1),
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                    ], undefined),
                  ], [[3, 4]]),
                ], [1]),
              ], undefined),
            ], undefined),
          ], undefined),
        ], undefined),
      ],
    ],
    reduce_collapse,
    'out(tiny.codegen.rewriter.reduce_collapse(*data))',
  ),
)

describe(
  'no_vectorized_acc',
  compare(
    [
      [
        new UOp(Ops.DEFINE_ACC, dtypes.float, [
          new UOp(Ops.CONST, dtypes.float, [], 0),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20),
          ], 1),
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 20),
          ], 2),
        ], [0]),
      ],
    ],
    no_vectorized_acc,
    'out(tiny.codegen.rewriter.no_vectorized_acc(*data))',
  ),
)

describe(
  'sym+devectorize',
  compare(
    [
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.IDIV, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int, [], 1),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 4),
              ], undefined),
              new UOp(Ops.IDIV, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int, [], 2),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 4),
              ], undefined),
            ], undefined),
            new UOp(Ops.IDIV, dtypes.int, [
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.IDIV, dtypes.int, [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 20000),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 3),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int, [], 4),
          ], undefined),
        ], undefined),
        undefined,
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.ADD, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.IDIV, dtypes.int, [
                new UOp(Ops.ADD, dtypes.int, [
                  new UOp(Ops.MUL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 20000),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 3),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 4),
              ], undefined),
              new UOp(Ops.IDIV, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 4),
              ], undefined),
            ], undefined),
            new UOp(Ops.IDIV, dtypes.int, [
              new UOp(Ops.ADD, dtypes.int, [
                new UOp(Ops.MUL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 20000),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 3),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int, [], 1),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 4),
            ], undefined),
          ], undefined),
          new UOp(Ops.IDIV, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 20000),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 3),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 2),
            ], undefined),
            new UOp(Ops.CONST, dtypes.int, [], 4),
          ], undefined),
        ], undefined),
        undefined,
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59999,
                    59998,
                    59997,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59998,
                    59997,
                    59996,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], [[2, 3]]),
        ], undefined),
        undefined,
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59997,
                    59996,
                    59995,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int.vec(3), [
              new UOp(Ops.WHERE, dtypes.int.vec(3), [
                new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                  new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                    new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                      59998,
                      59997,
                      59996,
                    ]),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
                new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
              ], undefined),
              new UOp(Ops.WHERE, dtypes.int.vec(3), [
                new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                  new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                    new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                      59999,
                      59998,
                      59997,
                    ]),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
                new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
              ], undefined),
            ], undefined),
          ], [[2, 3]]),
        ], undefined),
        undefined,
      ],
      [
        new UOp(Ops.ADD, dtypes.int, [
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.WHERE, dtypes.int.vec(3), [
              new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                  new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.ADD, dtypes.int, [
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 20000),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 3),
                      ], undefined),
                      new UOp(Ops.MUL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 15000),
                        ], 1),
                        new UOp(Ops.CONST, dtypes.int, [], 4),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                  new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                    59996,
                    59995,
                    59994,
                  ]),
                ], undefined),
                new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
              new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
            ], undefined),
          ], [[2, 3]]),
          new UOp(Ops.UNROLL, dtypes.int, [
            new UOp(Ops.ADD, dtypes.int.vec(3), [
              new UOp(Ops.ADD, dtypes.int.vec(3), [
                new UOp(Ops.WHERE, dtypes.int.vec(3), [
                  new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                    new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                      new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                        59998,
                        59997,
                        59996,
                      ]),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
                  new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
                ], undefined),
                new UOp(Ops.WHERE, dtypes.int.vec(3), [
                  new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                    new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                      new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                        new UOp(Ops.ADD, dtypes.int, [
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 20000),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 3),
                          ], undefined),
                          new UOp(Ops.MUL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 15000),
                            ], 1),
                            new UOp(Ops.CONST, dtypes.int, [], 4),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                        59999,
                        59998,
                        59997,
                      ]),
                    ], undefined),
                    new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
                  new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
                ], undefined),
              ], undefined),
              new UOp(Ops.WHERE, dtypes.int.vec(3), [
                new UOp(Ops.CMPNE, dtypes.bool.vec(3), [
                  new UOp(Ops.CMPLT, dtypes.bool.vec(3), [
                    new UOp(Ops.VECTORIZE, dtypes.int.vec(3), [
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.ADD, dtypes.int, [
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 20000),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 3),
                        ], undefined),
                        new UOp(Ops.MUL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 15000),
                          ], 1),
                          new UOp(Ops.CONST, dtypes.int, [], 4),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.VCONST, dtypes.int.vec(3), [], [
                      59997,
                      59996,
                      59995,
                    ]),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.bool.vec(3), [], true),
                ], undefined),
                new UOp(Ops.CONST, dtypes.int.vec(3), [], 1),
                new UOp(Ops.CONST, dtypes.int.vec(3), [], 0),
              ], undefined),
            ], undefined),
          ], [[2, 3]]),
        ], undefined),
        undefined,
      ],
    ],
    (uop: UOp, ctx: any) => sym.add(devectorize).rewrite(uop, ctx),
    'out((tiny.codegen.rewriter.sym + tiny.codegen.rewriter.devectorize).rewrite(*data))',
  ),
)
