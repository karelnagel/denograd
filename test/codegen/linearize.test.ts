import { append_to_block, BasicBlock, block_merge, block_reorder, linearize_uop } from '../../jsgrad/codegen/linearize.ts'
import { compare, tryCatch } from '../helpers.ts'
import { dtypes } from '../../jsgrad/dtype.ts'
import { KernelInfo, Ops, UOp } from '../../jsgrad/ops.ts'
import { describe as describe } from 'vitest'

describe(
  'append_to_block',
  compare(
    [
      [
        [
          new Map([
            [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 0), []],
            [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              [],
            ],
            [new UOp(Ops.CONST, dtypes.float, [], 1.0), []],
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
              [],
            ],
          ]),
          new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ],
          ], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ]], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
            ],
          ]]),
        ],
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ],
          new BasicBlock([], [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
      [
        [
          new Map([
            [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 0), []],
            [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              [],
            ],
            [new UOp(Ops.CONST, dtypes.float, [], 1.0), []],
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
              [],
            ],
          ]),
          new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ],
          ], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ]], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
            ],
          ]]),
        ],
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ],
          new BasicBlock([], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
      [
        [
          new Map([
            [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 0), []],
            [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              [],
            ],
            [new UOp(Ops.CONST, dtypes.float, [], 1.0), []],
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
              [],
            ],
          ]),
          new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ],
          ], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ]], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
            ],
          ]]),
        ],
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ],
          new BasicBlock([], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
      [
        [
          new Map([
            [new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 0), []],
            [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              [],
            ],
            [new UOp(Ops.CONST, dtypes.float, [], 1.0), []],
            [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.BITCAST, dtypes.uint, [
                    new UOp(Ops.NOOP, dtypes.float, [
                      new UOp(Ops.CONST, dtypes.float, [], 1.0),
                    ], undefined),
                  ], undefined),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
              [],
            ],
          ]),
          new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0), [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
            new UOp(Ops.NOOP, dtypes.float, [
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ]], [
            new UOp(Ops.NOOP, dtypes.float, [
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            [
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ],
          ], [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ],
          ], [
            new UOp(Ops.BITCAST, dtypes.uint, [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ],
          ], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.BITCAST, dtypes.uint, [
                    new UOp(Ops.NOOP, dtypes.float, [
                      new UOp(Ops.CONST, dtypes.float, [], 1.0),
                    ], undefined),
                  ], undefined),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
            ],
          ]]),
        ],
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
          ],
          new BasicBlock([], [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
      [
        [
          new Map([
            [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 10), []],
            [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
              [
                new UOp(Ops.BLOCKSTART, dtypes.void, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
              ],
            ],
            [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ],
            ],
            [new UOp(Ops.CONST, dtypes.float, [], 0.0), []],
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
              [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ],
            ],
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 10),
                    ], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 0.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
              [],
            ],
          ]),
          new Map([[new UOp(Ops.CONST, dtypes.int, [], 0), [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ]], [new UOp(Ops.CONST, dtypes.int, [], 10), [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ]], [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ]], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
            [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
            ],
          ], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
            ],
          ], [new UOp(Ops.CONST, dtypes.float, [], 0.0), [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ]], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 10),
                    ], 0),
                  ], undefined),
                  new UOp(Ops.CONST, dtypes.float, [], 0.0),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
            ],
          ]]),
        ],
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ),
      ],
      [
        [
          new Map([
            [new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0), []],
            [new UOp(Ops.CONST, dtypes.int, [], 0), []],
            [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              [],
            ],
            [new UOp(Ops.CONST, dtypes.float, [], 1.0), []],
            [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
              [],
            ],
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.BITCAST, dtypes.uint, [
                    new UOp(Ops.NOOP, dtypes.float, [
                      new UOp(Ops.CONST, dtypes.float, [], 1.0),
                    ], undefined),
                  ], undefined),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
              [],
            ],
          ]),
          new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0), [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
          ]], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
            new UOp(Ops.NOOP, dtypes.float, [
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ]], [
            new UOp(Ops.NOOP, dtypes.float, [
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            [
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ],
          ], [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ],
          ], [
            new UOp(Ops.BITCAST, dtypes.uint, [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], undefined),
            [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ],
          ], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
            [
              new UOp(Ops.SINK, dtypes.void, [
                new UOp(Ops.STORE, dtypes.void, [
                  new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                  ], undefined),
                  new UOp(Ops.BITCAST, dtypes.uint, [
                    new UOp(Ops.NOOP, dtypes.float, [
                      new UOp(Ops.CONST, dtypes.float, [], 1.0),
                    ], undefined),
                  ], undefined),
                ], undefined),
              ], new KernelInfo(0, 0, false)),
            ],
          ]]),
        ],
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.BITCAST, dtypes.uint, [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], undefined),
          ],
          new BasicBlock([], [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
    ],
    append_to_block,
    'out(tiny.codegen.linearize.append_to_block(*data))',
  ),
)
describe(
  'block_merge',
  compare(
    [
      [
        new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ]], [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ],
        ], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ], undefined),
        ]], [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ], undefined),
          [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ],
        ]]),
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ],
          new BasicBlock([], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
      [
        new Map([[new UOp(Ops.CONST, dtypes.int, [], 0), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.CONST, dtypes.int, [], 10), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ]], [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
          [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ],
        ], [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ],
        ], [new UOp(Ops.CONST, dtypes.float, [], 0.0), [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
        ]], [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
          [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ],
        ]]),
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new Map([[new UOp(Ops.CONST, dtypes.int, [], 0), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.CONST, dtypes.int, [], 10), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ]], [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
          [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ],
        ], [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ],
        ], [new UOp(Ops.CONST, dtypes.float, [], 0.0), [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
        ]], [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
          [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ],
        ]]),
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ],
          new BasicBlock([], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
            new UOp(Ops.ENDRANGE, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ], undefined),
        ),
      ],
      [
        new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0), [
          new UOp(Ops.INDEX, dtypes.uint.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ]], [new UOp(Ops.CONST, dtypes.int, [], 0), [
          new UOp(Ops.INDEX, dtypes.uint.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
        ]], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [
          new UOp(Ops.NOOP, dtypes.float, [
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ], undefined),
        ]], [
          new UOp(Ops.NOOP, dtypes.float, [
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ], undefined),
          [
            new UOp(Ops.BITCAST, dtypes.uint, [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], undefined),
          ],
        ], [
          new UOp(Ops.INDEX, dtypes.uint.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
          ],
        ], [
          new UOp(Ops.BITCAST, dtypes.uint, [
            new UOp(Ops.NOOP, dtypes.float, [
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
          ],
        ], [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.BITCAST, dtypes.uint, [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], undefined),
          ], undefined),
          [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ],
        ]]),
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ],
          new BasicBlock([], [
            new UOp(Ops.NOOP, dtypes.float, [
              new UOp(Ops.CONST, dtypes.float, [], 1.0),
            ], undefined),
            new UOp(Ops.BITCAST, dtypes.uint, [
              new UOp(Ops.NOOP, dtypes.float, [
                new UOp(Ops.CONST, dtypes.float, [], 1.0),
              ], undefined),
            ], undefined),
            new UOp(Ops.INDEX, dtypes.uint.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
              new UOp(Ops.BITCAST, dtypes.uint, [
                new UOp(Ops.NOOP, dtypes.float, [
                  new UOp(Ops.CONST, dtypes.float, [], 1.0),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.uint.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
                new UOp(Ops.BITCAST, dtypes.uint, [
                  new UOp(Ops.NOOP, dtypes.float, [
                    new UOp(Ops.CONST, dtypes.float, [], 1.0),
                  ], undefined),
                ], undefined),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
      [
        new Map([[new UOp(Ops.CONST, dtypes.int, [], 0), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.CONST, dtypes.int, [], 10), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ]], [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
          [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ],
        ], [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ],
        ], [new UOp(Ops.CONST, dtypes.float, [], 0.0), [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
        ]], [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
          [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ],
        ]]),
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
            new UOp(
              Ops.BLOCK,
              dtypes.void,
              [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ],
              new BasicBlock([
                new UOp(Ops.BLOCKSTART, dtypes.void, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
              ], [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
            ),
          ],
          new BasicBlock([
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ], undefined),
        ),
      ],
      [
        new Map([[new UOp(Ops.CONST, dtypes.int, [], 0), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.CONST, dtypes.int, [], 10), [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
        ]], [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ]], [
          new UOp(Ops.RANGE, dtypes.int, [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ], 0),
          [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ],
        ], [
          new UOp(Ops.INDEX, dtypes.float.ptr(), [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
          [
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ],
        ], [new UOp(Ops.CONST, dtypes.float, [], 0.0), [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
        ]], [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
          [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ],
        ]]),
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ],
          new BasicBlock([], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 10),
                ], 0),
              ], undefined),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
            new UOp(Ops.ENDRANGE, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 10),
                  ], 0),
                ], undefined),
                new UOp(Ops.CONST, dtypes.float, [], 0.0),
              ], undefined),
            ], new KernelInfo(0, 0, false)),
          ], undefined),
        ),
      ],
    ],
    block_merge,
    'out(tiny.codegen.linearize.block_merge(*data))',
  ),
)
describe(
  'block_reorder',
  compare(
    [
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [new UOp(Ops.CONST, dtypes.float, [], 0.0)],
          new BasicBlock([], [
            new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 8),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 8),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 8),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 16),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 16),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 16),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 10),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 10),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 200),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 200),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 200),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 6400),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 6400),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 6400),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [new UOp(Ops.CONST, dtypes.float, [], 0)],
          new BasicBlock([], [
            new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
              new UOp(Ops.CONST, dtypes.float, [], 0),
              new UOp(Ops.CONST, dtypes.float, [], 0),
              new UOp(Ops.CONST, dtypes.float, [], 0),
              new UOp(Ops.CONST, dtypes.float, [], 0),
            ], undefined),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(Ops.CONST, dtypes.int, [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 200),
          ],
          new BasicBlock([
            new UOp(Ops.BLOCKSTART, dtypes.void, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 200),
              ], 0),
            ], undefined),
          ], [
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 200),
            ], 0),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(
              Ops.BLOCK,
              dtypes.void,
              [new UOp(Ops.CONST, dtypes.float, [], 0)],
              new BasicBlock([], [
                new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                ], undefined),
              ], undefined),
            ),
            new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
            new UOp(Ops.CONST, dtypes.int, [], 2),
            new UOp(
              Ops.BLOCK,
              dtypes.void,
              [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 200),
              ],
              new BasicBlock([
                new UOp(Ops.BLOCKSTART, dtypes.void, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 200),
                  ], 0),
                ], undefined),
              ], [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 200),
                ], 0),
              ], undefined),
            ),
          ],
          new BasicBlock([
            new UOp(Ops.RANGE, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 200),
            ], 0),
          ], [
            new UOp(Ops.SHL, dtypes.int, [
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 200),
              ], 0),
              new UOp(Ops.CONST, dtypes.int, [], 2),
            ], undefined),
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.SHL, dtypes.int, [
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 200),
                ], 0),
                new UOp(Ops.CONST, dtypes.int, [], 2),
              ], undefined),
            ], undefined),
            new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.SHL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 200),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 2),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.STORE, dtypes.void, [
              new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
                new UOp(Ops.INDEX, dtypes.float.ptr(), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                  new UOp(Ops.SHL, dtypes.int, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 200),
                    ], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 2),
                  ], undefined),
                ], undefined),
              ], undefined),
              new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
                new UOp(Ops.CONST, dtypes.float, [], 0),
                new UOp(Ops.CONST, dtypes.float, [], 0),
                new UOp(Ops.CONST, dtypes.float, [], 0),
                new UOp(Ops.CONST, dtypes.float, [], 0),
              ], undefined),
            ], undefined),
          ], undefined),
        ),
      ],
      [
        new UOp(
          Ops.BLOCK,
          dtypes.void,
          [
            new UOp(
              Ops.BLOCKEND,
              dtypes.void,
              [
                new UOp(
                  Ops.BLOCK,
                  dtypes.void,
                  [
                    new UOp(
                      Ops.BLOCK,
                      dtypes.void,
                      [new UOp(Ops.CONST, dtypes.float, [], 0)],
                      new BasicBlock([], [
                        new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
                          new UOp(Ops.CONST, dtypes.float, [], 0),
                          new UOp(Ops.CONST, dtypes.float, [], 0),
                          new UOp(Ops.CONST, dtypes.float, [], 0),
                          new UOp(Ops.CONST, dtypes.float, [], 0),
                        ], undefined),
                      ], undefined),
                    ),
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 2),
                    new UOp(
                      Ops.BLOCK,
                      dtypes.void,
                      [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 200),
                      ],
                      new BasicBlock([
                        new UOp(Ops.BLOCKSTART, dtypes.void, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 200),
                          ], 0),
                        ], undefined),
                      ], [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 200),
                        ], 0),
                      ], undefined),
                    ),
                  ],
                  new BasicBlock([
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 200),
                    ], 0),
                  ], [
                    new UOp(Ops.SHL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 200),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 2),
                    ], undefined),
                    new UOp(Ops.INDEX, dtypes.float.ptr(), [
                      new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                      new UOp(Ops.SHL, dtypes.int, [
                        new UOp(Ops.RANGE, dtypes.int, [
                          new UOp(Ops.CONST, dtypes.int, [], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 200),
                        ], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 2),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
                      new UOp(Ops.INDEX, dtypes.float.ptr(), [
                        new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                        new UOp(Ops.SHL, dtypes.int, [
                          new UOp(Ops.RANGE, dtypes.int, [
                            new UOp(Ops.CONST, dtypes.int, [], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 200),
                          ], 0),
                          new UOp(Ops.CONST, dtypes.int, [], 2),
                        ], undefined),
                      ], undefined),
                    ], undefined),
                    new UOp(Ops.STORE, dtypes.void, [
                      new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
                        new UOp(Ops.INDEX, dtypes.float.ptr(), [
                          new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                          new UOp(Ops.SHL, dtypes.int, [
                            new UOp(Ops.RANGE, dtypes.int, [
                              new UOp(Ops.CONST, dtypes.int, [], 0),
                              new UOp(Ops.CONST, dtypes.int, [], 200),
                            ], 0),
                            new UOp(Ops.CONST, dtypes.int, [], 2),
                          ], undefined),
                        ], undefined),
                      ], undefined),
                      new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
                        new UOp(Ops.CONST, dtypes.float, [], 0),
                        new UOp(Ops.CONST, dtypes.float, [], 0),
                        new UOp(Ops.CONST, dtypes.float, [], 0),
                        new UOp(Ops.CONST, dtypes.float, [], 0),
                      ], undefined),
                    ], undefined),
                  ], undefined),
                ),
              ],
              new BasicBlock(
                [],
                [
                  new UOp(Ops.ENDRANGE, dtypes.void, [
                    new UOp(Ops.RANGE, dtypes.int, [
                      new UOp(Ops.CONST, dtypes.int, [], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 200),
                    ], 0),
                  ], undefined),
                ],
                new UOp(Ops.RANGE, dtypes.int, [
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 200),
                ], 0),
              ),
            ),
          ],
          new BasicBlock([], [
            new UOp(Ops.SINK, dtypes.void, [
              new UOp(Ops.STORE, dtypes.void, [
                new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
                  new UOp(Ops.INDEX, dtypes.float.ptr(), [
                    new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                    new UOp(Ops.SHL, dtypes.int, [
                      new UOp(Ops.RANGE, dtypes.int, [
                        new UOp(Ops.CONST, dtypes.int, [], 0),
                        new UOp(Ops.CONST, dtypes.int, [], 200),
                      ], 0),
                      new UOp(Ops.CONST, dtypes.int, [], 2),
                    ], undefined),
                  ], undefined),
                ], undefined),
                new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                  new UOp(Ops.CONST, dtypes.float, [], 0),
                ], undefined),
              ], undefined),
            ], new KernelInfo(0, 1, false)),
          ], undefined),
        ),
      ],
    ],
    block_reorder,
    'out(tiny.codegen.linearize.block_reorder(*data))',
  ),
)
describe(
  'linearize_uop',
  compare(
    [
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 1.0),
          ], undefined),
        ], new KernelInfo(0, 0, false)),
        true,
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
              new UOp(Ops.RANGE, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 10),
              ], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 0.0),
          ], undefined),
        ], new KernelInfo(0, 0, false)),
        true,
      ],
      // For some reason order fails
      // [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.uint.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.uint.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.BITCAST, dtypes.uint, [new UOp(Ops.NOOP, dtypes.float, [new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)], undefined)], undefined)], new KernelInfo(0, 0, false)), false],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.SHL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 8),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 2),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 1, false)),
        true,
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.SHL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 16),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 2),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 1, false)),
        true,
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.CAST, dtypes.float.vec(4).ptr(), [
              new UOp(Ops.INDEX, dtypes.float.ptr(), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0),
                new UOp(Ops.SHL, dtypes.int, [
                  new UOp(Ops.RANGE, dtypes.int, [
                    new UOp(Ops.CONST, dtypes.int, [], 0),
                    new UOp(Ops.CONST, dtypes.int, [], 200),
                  ], 0),
                  new UOp(Ops.CONST, dtypes.int, [], 2),
                ], undefined),
              ], undefined),
            ], undefined),
            new UOp(Ops.VECTORIZE, dtypes.float.vec(4), [
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
              new UOp(Ops.CONST, dtypes.float, [], 0.0),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 1, false)),
        true,
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float.ptr(), [
              new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0.1),
              new UOp(Ops.CONST, dtypes.int, [], 0),
            ], undefined),
            new UOp(Ops.CONST, dtypes.float, [], 1.1),
          ], undefined),
        ], new KernelInfo(0, 0, false)),
        false,
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.CAST, dtypes.float.vec(4).ptr(1), [
              new UOp(Ops.INDEX, dtypes.float.ptr(4), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
            ], undefined),
            new UOp(Ops.LOAD, dtypes.float.vec(4), [
              new UOp(Ops.CAST, dtypes.float.vec(4).ptr(1), [
                new UOp(Ops.INDEX, dtypes.float.ptr(4), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(4), [], 1),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
              ], undefined),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 1, false)),
      ],
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.CAST, dtypes.int.vec(4).ptr(1), [
              new UOp(Ops.INDEX, dtypes.int.ptr(4), [
                new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(4), [], 0),
                new UOp(Ops.CONST, dtypes.int, [], 0),
              ], undefined),
            ], undefined),
            new UOp(Ops.LOAD, dtypes.int.vec(4), [
              new UOp(Ops.CAST, dtypes.int.vec(4).ptr(1), [
                new UOp(Ops.INDEX, dtypes.int.ptr(4), [
                  new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(4), [], 1),
                  new UOp(Ops.CONST, dtypes.int, [], 0),
                ], undefined),
              ], undefined),
            ], undefined),
          ], undefined),
        ], new KernelInfo(0, 1, false)),
      ],
    ],
    tryCatch(linearize_uop),
    'out(trycatch(lambda:tiny.codegen.linearize.linearize_uop(*data)))',
  ),
)
