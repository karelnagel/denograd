import { _choices_from_args, _expand_arg_to_idx, _swizzle_args, create_gate, delete_redundant_gates, do_contract, do_expand, fix_unfoldable_image_load, fold_expanded, full_graph_rewrite, loop_collapse, move_mask, no_vectorized_alu, no_vectorized_load_store, simplify_valid_load, threefry2x32 } from '../../src/codegen/uopgraph.ts'
import { dtypes } from '../../src/dtype.ts'
import { Ops, UOp } from '../../src/ops.ts'
import { ClangRenderer } from '../../src/renderer/cstyle.ts'
import { compare, tryCatch } from '../helpers.ts'

Deno.test(
  'fold_expanded',
  compare(
    [
      [new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [])]), UOp.variable('buf')],
      [new UOp(Ops.STORE, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf')])]), UOp.variable('buf')],
      [new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf')], 0)])]), UOp.variable('buf')],
      [new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [new UOp(Ops.ADD, undefined, [UOp.variable('idx'), UOp.int(1)])])]), UOp.variable('buf')],
      [new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf'), UOp.variable('gate')])])]), UOp.variable('buf')],
    ],
    tryCatch(fold_expanded),
    'out(tiny.codegen.uopgraph.fold_expanded(*data))',
  ),
)

Deno.test(
  'fix_unfoldable_image_load',
  compare(
    [
      [
        new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)), UOp.int(5)])]),
        UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)),
      ],
      [
        new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)), UOp.variable('idx')])]),
        UOp.variable('buf', undefined, undefined, dtypes.imagef(10, 10)),
      ],
    ],
    fix_unfoldable_image_load,
    'out(tiny.codegen.uopgraph.fix_unfoldable_image_load(*data))',
  ),
)

Deno.test(
  'simplify_valid_load',
  compare(
    [
      [UOp.variable('buf'), UOp.int(5), UOp.int(0)], // idx is None case
      [UOp.variable('buf'), UOp.int(5), UOp.int(1)], // non-image buffer, same idx
      [UOp.variable('buf'), UOp.variable('idx'), UOp.variable('valid')], // non-image buffer, different idx
    ],
    simplify_valid_load,
    'out(tiny.codegen.uopgraph.simplify_valid_load(*data))',
  ),
)

Deno.test.ignore(
  'threefry2x32',
  compare(
    [
      [UOp.int(0x1234567812345678), UOp.int(0x1234567812345678)], // Test with same input and key
      [UOp.int(0x0000000000000000), UOp.int(0x0000000000000000)], // Test with zeros
      [UOp.int(0xFFFFFFFFFFFFFFFF), UOp.int(0xFFFFFFFFFFFFFFFF)], // Test with all ones
      [UOp.int(0x1234567812345678), UOp.int(0x8765432187654321)], // Test with different input and key
    ],
    threefry2x32,
    'out(tiny.codegen.uopgraph.threefry2x32(*data))',
  ),
)

Deno.test(
  'loop_collapse',
  compare(
    [
      // Basic test with positive mul
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc'), undefined, undefined, undefined, undefined, UOp.int(1)],

      // Test with negative mul
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc'), undefined, undefined, undefined, undefined, undefined, UOp.int(0), UOp.int(-1)],

      // Test with idx2 and idx3
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc'), UOp.int(2), UOp.int(3)],

      // Test with vectorization
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc'), undefined, undefined, undefined, UOp.variable('vec', undefined, undefined, dtypes.float.vec(4))],

      // Test with extra accumulation
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc'), undefined, undefined, UOp.int(5)],

      // Test with non-zero loop start (should return undefined)
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc')],

      // Test with disabled loop collapse (should return undefined)
      [UOp.int(10), UOp.int(1), new UOp(Ops.RANGE, undefined, undefined, [0, 5]), UOp.variable('acc')],
    ],
    loop_collapse,
    'out(tiny.codegen.uopgraph.loop_collapse(*data))',
  ),
)

Deno.test(
  '_expand_arg_to_idx',
  compare(
    [
      // Basic test with single axis
      [[[0, 4]], { 0: 0 }],
      [[[0, 4]], { 0: 1 }],
      [[[0, 4]], { 0: 2 }],
      [[[0, 4]], { 0: 3 }],

      // Test with multiple axes
      [[[0, 2], [1, 3]], { 0: 0, 1: 0 }],
      [[[0, 2], [1, 3]], { 0: 1, 1: 0 }],
      [[[0, 2], [1, 3]], { 0: 0, 1: 1 }],
      [[[0, 2], [1, 3]], { 0: 1, 1: 2 }],

      // Test with larger multipliers
      [[[0, 8], [1, 4]], { 0: 3, 1: 2 }],
      [[[0, 16], [1, 8], [2, 4]], { 0: 10, 1: 5, 2: 2 }],

      // Edge cases
      [[], {}], // Empty args
      [[[0, 1]], { 0: 0 }], // Single element axis
    ],
    _expand_arg_to_idx,
    'out(tiny.codegen.uopgraph._expand_arg_to_idx(data[0],{int(k): v for k, v in data[1].items()}))',
  ),
)

Deno.test(
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
    ],
    _choices_from_args,
    'out(tiny.codegen.uopgraph._choices_from_args(*data))',
  ),
)

Deno.test(
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
    ],
    _swizzle_args,
    'out(tiny.codegen.uopgraph._swizzle_args(*data))',
  ),
)

Deno.test(
  'do_expand',
  compare(
    [
      // Basic test with no expands
      [new UOp(Ops.ADD, dtypes.float32, [UOp.const(dtypes.float32, 1), UOp.const(dtypes.float32, 2)])],

      // Basic expand test
      [
        new UOp(Ops.ADD, dtypes.float32, [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[0, 2]]),
          UOp.const(dtypes.float32, 2),
        ]),
      ],

      // Multiple expands with same args
      [
        new UOp(Ops.ADD, dtypes.float32, [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[0, 2]]),
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 2)], [[0, 2]]),
        ]),
      ],

      // Multiple expands with different args
      [
        new UOp(Ops.ADD, dtypes.float32, [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[0, 2]]),
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 2)], [[1, 3]]),
        ]),
      ],

      // Test with vectorized dtype
      [
        new UOp(Ops.ADD, dtypes.float32.vec(2), [
          new UOp(Ops.EXPAND, dtypes.float32.vec(2), [UOp.const(dtypes.float32.vec(2), [1, 2])], [[0, 2]]),
          UOp.const(dtypes.float32.vec(2), [3, 4]),
        ]),
      ],
    ],
    tryCatch(do_expand),
    'out(tiny.codegen.uopgraph.do_expand(*data))',
  ),
)

Deno.test(
  'do_contract',
  compare(
    [
      // CONTRACT without EXPAND
      [
        new UOp(Ops.CONTRACT, dtypes.float32.vec(2), [UOp.const(dtypes.float32, 1)], [[0, 2]]),
      ],

      // CONTRACT with EXPAND - removing one axis
      [
        new UOp(Ops.CONTRACT, dtypes.float32.vec(2), [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[0, 2], [1, 3]]),
        ], [[0, 2]]),
      ],

      // CONTRACT with EXPAND - removing multiple axes
      [
        new UOp(Ops.CONTRACT, dtypes.float32.vec(6), [
          new UOp(Ops.EXPAND, dtypes.float32, [UOp.const(dtypes.float32, 1)], [[0, 2], [1, 3], [2, 4]]),
        ], [[0, 2], [1, 3]]),
      ],
    ],
    do_contract,
    'out(tiny.codegen.uopgraph.do_contract(*data))',
  ),
)

Deno.test(
  'no_vectorized_alu',
  compare(
    [
      // No change for scalar ALU
      [
        new UOp(Ops.ADD, dtypes.float32, [UOp.const(dtypes.float32, 1), UOp.const(dtypes.float32, 2)], undefined),
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
    ],
    no_vectorized_alu,
    'out(tiny.codegen.uopgraph.no_vectorized_alu(*data))',
  ),
)

Deno.test(
  'create_gate',
  compare(
    [
      // No change for non-INDEX operation
      [
        new UOp(Ops.ADD, dtypes.float32, [UOp.const(dtypes.float32, 1)], undefined),
      ],

      // No change when INDEX has only 2 sources
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.int32, [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1)], undefined),
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
    ],
    tryCatch(create_gate),
    'out(tiny.codegen.uopgraph.create_gate(*data))',
  ),
)

Deno.test(
  'no_vectorized_load_store',
  compare(
    [
      // No change for non-pointer dtype
      [
        new UOp(Ops.LOAD, dtypes.float32, [UOp.const(dtypes.float32, 1)], undefined),
      ],

      // No change for vector size 1
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
        ], undefined),
      ],

      // Devectorizes load with vector size > 1
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
        ], undefined),
      ],

      // Devectorizes store with vector size > 1
      [
        new UOp(Ops.STORE, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
          UOp.const(dtypes.float32, 1),
        ], undefined),
      ],
    ],
    tryCatch(no_vectorized_load_store),
    'out(trycatch(lambda:tiny.codegen.uopgraph.no_vectorized_load_store(*data)))',
  ),
)

Deno.test(
  'delete_redundant_gates',
  compare(
    [
      // Case 1: Store with gate that should be removed
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
        UOp.const(dtypes.int32, 1),
        new UOp(Ops.IF, dtypes.void, [UOp.const(dtypes.bool, true), UOp.const(dtypes.int32, 1)], undefined),

        new UOp(Ops.CAST, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
      ],

      // Case 2: Store with gate and cast that should be removed
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
        UOp.const(dtypes.int32, 1),
        new UOp(Ops.IF, dtypes.void, [UOp.const(dtypes.bool, true), UOp.const(dtypes.int32, 1)], undefined),
        new UOp(Ops.CAST, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
      ],

      // Case 3: Store without matching gate (should return undefined)
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
        UOp.const(dtypes.int32, 1),
        new UOp(Ops.IF, dtypes.void, [UOp.const(dtypes.bool, false), UOp.const(dtypes.int32, 1)], undefined),

        new UOp(Ops.CAST, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
      ],

      // Case 4: Store with multiple gates but no matching one
      [
        new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
        new UOp(Ops.IF, dtypes.void, [UOp.const(dtypes.bool, false), UOp.const(dtypes.int32, 2)], undefined),
        new UOp(Ops.IF, dtypes.void, [UOp.const(dtypes.bool, false), UOp.const(dtypes.int32, 3)], undefined),
        new UOp(Ops.CAST, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
      ],
    ],
    delete_redundant_gates,
    'out(tiny.codegen.uopgraph.delete_redundant_gates(*data))',
  ),
)

Deno.test(
  'move_mask',
  compare(
    [
      // Case 1: Load with mask
      [
        new UOp(Ops.LOAD, dtypes.float32, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
        ], undefined),
        UOp.const(dtypes.int32, 0),
        UOp.const(dtypes.int32, 1),
        UOp.const(dtypes.bool, true),
      ],

      // Case 2: Store with mask
      [
        new UOp(Ops.STORE, dtypes.void, [
          new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
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
            new UOp(Ops.INDEX, dtypes.int32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 1), UOp.const(dtypes.bool, true)], undefined),
          ], undefined),
        ], undefined),
        UOp.const(dtypes.int32, 0),
        UOp.const(dtypes.int32, 1),
        UOp.const(dtypes.bool, true),
        new UOp(Ops.CAST, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0)], undefined),
      ],
    ],
    move_mask,
    'out(tiny.codegen.uopgraph.move_mask(*data))',
  ),
)
Deno.test(
  'full_graph_rewrite',
  compare(
    [
      // Test basic sink operation
      [new UOp(Ops.SINK, dtypes.void, [UOp.const(dtypes.float32, 1.0)], undefined)],

      // Test with vectorized load that should be devectorized
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.LOAD, dtypes.float32.vec(4), [new UOp(Ops.INDEX, dtypes.float32.ptr().vec(4), [UOp.const(dtypes.int32, 0)], undefined)], undefined),
        ], undefined),
        new ClangRenderer(),
      ],

      // Test with symbolic operations
      [
        new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.ADD, dtypes.float32, [UOp.const(dtypes.float32, 1.0), UOp.const(dtypes.float32, 2.0)], undefined)], undefined),
        new ClangRenderer(),
      ],
      // Test with float4 folding when supported
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.LOAD, dtypes.float32.vec(4), [
            new UOp(Ops.INDEX, dtypes.float32.ptr().vec(4), [UOp.const(dtypes.int32, 0)], undefined),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],

      // Test with late rewrite patterns
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.EXP2, dtypes.float32, [UOp.const(dtypes.float32, 1.0)], undefined),
        ], undefined),
      ],

      // Test with migrate indexing
      [
        new UOp(Ops.SINK, dtypes.void, [
          new UOp(Ops.STORE, dtypes.void, [
            new UOp(Ops.INDEX, dtypes.float32.ptr(), [UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 0)], undefined),
            UOp.const(dtypes.float32, 1.0),
          ], undefined),
        ], undefined),
        new ClangRenderer(),
      ],

      // Test with load store indexing
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
    ],
    full_graph_rewrite,
    'out(tiny.codegen.uopgraph.full_graph_rewrite(*data))',
  ),
)
