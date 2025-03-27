import { Ops, type sint, UOp } from "../../jsgrad/ops.ts";
import { compare, tryCatch } from "../helpers.ts";
import {
  _reshape_mask,
  canonicalize_strides,
  strides_for_shape,
  View,
} from "../../jsgrad/shape/view.ts";
import { dtypes } from "../../jsgrad/dtype.ts";
import { describe } from "vitest";

describe(
  "canonicalize_strides",
  compare(
    [
      [[UOp.int(200), UOp.int(100), UOp.int(30)], [
        UOp.int(20),
        UOp.int(2100),
        UOp.int(20000),
      ]],
      [[44444, UOp.int(100), 4.4], [UOp.int(20), 5555, -5.5]],
      [[44444, 543, 4.4], [555, 5555, -5.5]],
    ],
    canonicalize_strides,
    "out(tiny.shape.view.canonicalize_strides(*data))",
  ),
);

describe(
  "strides_for_shape",
  compare(
    [
      [[4, 23, 545, 32443, 44, -1]],
      [[
        4,
        23,
        545,
        32443,
        44,
        -1,
        UOp.int(444),
        UOp.bool(true),
        UOp.float(4.44),
      ]],
      [[UOp.int(200), UOp.int(500), UOp.int(500), UOp.int(500)]],
      [[UOp.int(2), UOp.float(4.4), UOp.int(743)]],
    ],
    strides_for_shape,
    `out(tiny.shape.view.strides_for_shape(*data))`,
  ),
);

describe(
  "_reshape_mask",
  compare(
    [
      [undefined, [4, 4], [8, 1, 1, 2, 1, 1, 1]],
      [[[UOp.int(2), UOp.int(3)]], [UOp.int(44), UOp.int(44)], [
        UOp.int(444),
        UOp.int(44),
      ]],
      [[[UOp.int(2), UOp.int(3)], [UOp.int(5), UOp.int(5)]], [
        UOp.int(44),
        UOp.int(44),
      ], [UOp.int(444), UOp.int(44)]],
      [[[2, 3], [UOp.int(5), 44444]], [555, UOp.int(44)], [
        UOp.int(444),
        UOp.float(44),
      ]],
      [[[0, 11], [9, 19]], [11, 19], [209]],
      [[[7, 15]], [15], [1, 15]],
      [[[0, 3], [2, 5]], [3, 5], [1, 3, 1, 5]],
    ],
    _reshape_mask,
    `out(tiny.shape.view._reshape_mask(*data))`,
  ),
);

describe(
  "View.create",
  compare(
    [
      [[4, 34, 534]], // basic shape
      [[4, 34, 534], [3, 4, 6]], // shape and strides
      [[4, 34, 534], [3, 4, 6], 4, [[23, 32], [43, 43], [43, 43]]], // shape, strides, offset, mask
      [[UOp.int(4), UOp.int(9), UOp.int(4)]], // symbolic shape
      [[UOp.int(4), UOp.int(9), UOp.int(4)], [
        UOp.int(1),
        UOp.int(9),
        UOp.int(4),
      ]], // symbolic shape and strides
      [
        [UOp.int(4), UOp.int(9), UOp.int(4)],
        [UOp.int(1), UOp.int(9), UOp.int(4)],
        0,
        [[UOp.int(23), UOp.int(44)], [UOp.int(323), UOp.int(23)], [
          UOp.int(43),
          UOp.int(43),
        ]],
      ], // symbolic everything
      [[0, 34, 534]], // zero in shape
      [[4, 0, 534]], // zero in middle of shape
      [[4, 34, 0]], // zero at end of shape
      [[4, 34, 534], undefined, UOp.int(5)], // offset with no strides
      [[4, 34], [36, 1], 0, [[0, 4], [0, 34]]], // mask matching shape
      [[4, 34], [36, 1], 0, [[1, 3], [2, 33]]], // partial mask
      [[4, 34], [36, 1], 0, [[2, 2], [3, 4]]], // single element mask
      [[4, 34], [36, 1], 0, [[3, 2], [4, 3]]], // invalid mask (start > end)
      [[UOp.int(4), UOp.int(9)], [UOp.int(9), UOp.int(1)], UOp.int(3), [[
        UOp.int(2),
        UOp.int(3),
      ], [UOp.int(4), UOp.int(8)]]], // symbolic mask
      [[UOp.int(4), UOp.int(9)], [UOp.int(9), UOp.int(1)], UOp.int(3), [[
        UOp.int(3),
        UOp.int(2),
      ], [UOp.int(8), UOp.int(4)]]], // invalid symbolic mask
    ],
    View.create,
    "out(tiny.shape.view.View.create(*data))",
  ),
);

const view1 = View.create([4, 4], [4, 1], 0, undefined);
const view2 = View.create(
  [UOp.int(4), UOp.int(6)],
  [UOp.int(6), UOp.int(1)],
  UOp.int(8),
  [[UOp.int(1), UOp.int(3)], [UOp.int(2), UOp.int(5)]],
);
const view3 = View.create([4, 4], [-4, -1], 0, undefined);
const view4 = View.create([0, 3, 2], [6, -2, 1], 0, [[1, 3], [0, 4]]);

describe(
  "View.to_indexed_uops",
  compare(
    [
      [view1], // default case with no indices
      [view1, [UOp.int(2), UOp.int(3)]], // basic indices
      [view2], // symbolic shape with mask
      [view2, [UOp.int(1), UOp.int(4)]], // symbolic shape, mask and custom indices
      [view3], // 3D view with mask
      [view3, [UOp.int(1), UOp.int(2)]], // 3D view with mask and indices
      [view4], // 4D view with mask
      [view4, [UOp.int(0), UOp.int(2), UOp.int(1)]], // 4D view with mask and indices
    ],
    (v: View, _idxs?: UOp[], vexpr?: UOp) => v.to_indexed_uops(_idxs, vexpr),
    "out(trycatch(lambda:data[0].to_indexed_uops(*data[1:])))",
  ),
);

describe(
  "View.size",
  compare(
    [
      [view1],
      [view2],
      [view3],
      [view4],
    ],
    (v: View) => v.size(),
    "out(trycatch(lambda:data[0].size()))",
  ),
);

describe(
  "View.vars",
  compare(
    [
      [view1],
      [view2],
      [view3],
      [view4],
    ],
    (v: View) => v.vars(),
    "out(trycatch(lambda:data[0].vars(*data[1:])))",
  ),
);

describe(
  "View.add",
  compare(
    [
      [view1, view1],
      [view2, view1],
      [view1, view2],
      [view3, view1],
      [view3, view4],
      [
        new View(
          [4, 55],
          [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 1),
              new UOp(Ops.CONST, dtypes.int, [], 55),
            ], undefined),
            1,
          ],
          0,
          undefined,
          false,
        ),
        new View(
          [8, 110, 33],
          [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.MUL, dtypes.int, [
                new UOp(Ops.CONST, dtypes.int, [], 1),
                new UOp(Ops.CONST, dtypes.int, [], 33),
              ], undefined),
              new UOp(Ops.CONST, dtypes.int, [], 110),
            ], undefined),
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 1),
              new UOp(Ops.CONST, dtypes.int, [], 33),
            ], undefined),
            1,
          ],
          0,
          undefined,
          false,
        ),
      ],
      [
        new View([10], [1], 0, [[0, 5]], false),
        new View([9], [1], 0, undefined, true),
      ],
    ],
    (v1: View, v2: View) => v1.add(v2),
    "out(trycatch(lambda: data[0] + data[1] ))",
  ),
);

describe(
  "View.invert",
  compare(
    [
      [view1, [3, UOp.int(4)]],
      [view2, [UOp.int(4), UOp.int(6)]],
      [view3, [UOp.int(3), UOp.int(5), UOp.int(2)]],
      [view4, [UOp.int(2), UOp.int(4), UOp.int(3), UOp.int(2)]],
      [view4, [4, 4]],
      [view4, [3, 3, 3]],
      [view1, [4, 4]],
    ],
    (v: View, out_shape: sint[]) => v.invert(out_shape),
    "out(trycatch(lambda: data[0].invert(data[1])))",
  ),
);

describe(
  "View.minify",
  compare(
    [
      [view1],
      [view2],
      [view3],
      [view4],
    ],
    (v: View) => v.minify(),
    "out(trycatch(lambda:data[0].minify()))",
  ),
);

describe(
  "View.pad",
  compare(
    [
      [view1, [[0, 0], [0, 0]]],
      [view1, [[1, 1], [2, 2]]],
      [view2, [[0, 1], [1, 0]]],
      [view3, [[1, 0], [0, 1], [2, 1]]],
      [view4, [[0, 0], [1, 1], [2, 2], [0, 1]]],
      [view1, [[UOp.int(1), UOp.int(2)], [UOp.int(0), UOp.int(1)]]],
      [view2, [[UOp.int(2), UOp.int(0)], [UOp.int(1), UOp.int(2)]]],
      [view3, [[0, UOp.int(1)], [UOp.int(1), 0], [1, UOp.int(2)]]],
    ],
    tryCatch((v: View, arg: [sint, sint][]) => v.pad(arg)),
    "out(trycatch(lambda:data[0].pad(data[1])))",
    { stringSimilarity: 0.75 },
  ),
);

describe(
  "View.shrink",
  compare(
    [
      [view1, [[0, 2], [1, 3]]],
      [view2, [[0, 2], [2, 4]]],
      [view3, [[1, 3], [1, 3]]],
      [view4, [[0, 2], [1, 2], [0, 1]]],
      [view1, [[UOp.int(0), UOp.int(2)], [UOp.int(1), UOp.int(3)]]],
      [view2, [[UOp.int(1), UOp.int(3)], [UOp.int(2), UOp.int(4)]]],
      [view3, [[0, UOp.int(2)], [1, UOp.int(3)]]],
      [view4, [[UOp.int(0), 2], [1, UOp.int(3)], [0, UOp.int(1)]]],
    ],
    tryCatch((v: View, arg: [sint, sint][]) => v.shrink(arg)),
    "out(trycatch(lambda:data[0].shrink(data[1])))",
  ),
);
describe(
  "View.expand",
  compare(
    [
      [view4, [0, 3, 1]],
      [view4, [0, 3, 2]],
      [view4, [UOp.int(0), UOp.int(3), UOp.int(2)]],
      [view3, [4, 4]],
      [view2, [4, 6]],
      [view1, [4, 4]],
      [view1, [4, 2, 2]],
      [view2, [4, UOp.int(4)]],
    ],
    tryCatch((v: View, new_shape: sint[]) => v.expand(new_shape)),
    "out(trycatch(lambda:data[0].expand(data[1])))",
    { stringSimilarity: 0.69 },
  ),
);

describe(
  "View.permute",
  compare(
    [
      [view1, [0, 1]],
      [view1, [1, 0]],
      [view2, [1, 0]],
      [view3, [0, 1]],
      [view3, [1, 0]],
      [view4, [1, 1]],
      [view4, [0, 2, 1]],
    ],
    tryCatch((v: View, axis: number[]) => v.permute(axis)),
    "out(trycatch(lambda:data[0].permute(data[1])))",
    { stringSimilarity: 0.94 },
  ),
);

describe(
  "View.stride",
  compare(
    [
      [view1, [2]],
      [view1, [-2]],
      [view2, [2, 3]],
      [view2, [-2, -3]],
      [view3, [2, -2, 3]],
      [view4, [2, -1, 3, -2]],
    ],
    (v: View, multi: number[]) => v.stride(multi),
    "out(trycatch(lambda:data[0].stride(data[1])))",
  ),
);

describe(
  "View.reshape",
  compare(
    [
      [view1, [1, 16]],
      [view1, [4, 4]],
      [view2, [3, 8]],
      [view2, [24]],
      [view2, [UOp.int(24)]],
      [view2, [12, 1, 1, 2, 1, 1, 1]],
      [view2, [UOp.int(12), 1, 1, 2, 1, 1, 1]],
      [view3, [8, 1, 1, 2, 1, 1, 1]],
      [
        new View(
          [4, 55],
          [
            new UOp(Ops.MUL, dtypes.int, [
              new UOp(Ops.CONST, dtypes.int, [], 1),
              new UOp(Ops.CONST, dtypes.int, [], 55),
            ], undefined),
            1,
          ],
          0,
          undefined,
          false,
        ),
        [],
      ],
      [new View([], [], 0, undefined, true), [1]],
      [new View([15], [0], 0, [[7, 15]], false), [1, 15]],
      [new View([3, 5], [0, 0], 0, [[0, 3], [2, 5]], false), [1, 3, 1, 5]],
      [new View([], [], 0, undefined, true), [1, 1, 1, 1]],
      [new View([], [], 0, undefined, true), []],
      [new View([3, 5], [0, 0], 0, [[0, 3], [2, 5]], false), [1, 3, 1, 5]],
      [new View([], [], 0, undefined, true), [1, 1, 1, 1]],
      [new View([], [], 0, undefined, true), []],
      [new View([1, 3, 1, 5], [0, 5, 0, 1], 0, undefined, true), [3, 5]],
      [new View([1, 3, 1, 5], [0, 0, 0, 0], 0, undefined, false), [3, 5]],
      [new View([3, 5], [5, 1], 0, undefined, true), [15]],
      [new View([3, 5], [0, 0], 0, undefined, false), [15]],
    ],
    tryCatch((v: View, new_shape: sint[]) => v.reshape(new_shape)),
    "out(trycatch(lambda:data[0].reshape(data[1])))",
    { stringSimilarity: 0.6 },
  ),
);
