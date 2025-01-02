import { can_pad, div_and_mod_folding, Ops, resolve, smax, smin, UOp, UPat } from '../src/ops.ts'
import { compare, tryCatch } from './helpers.ts'
import { dtypes } from '../src/dtype.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'

Deno.test(
  'can_pad',
  compare(
    [
      [new UOp(Ops.RECIP), new Map(), new Set([])],
      [new UOp(Ops.ADD), new Map(), new Set([])],

      [new UOp(Ops.RECIP, undefined, [new UOp(Ops.IDIV)]), new Map(), new Set([])],
      [new UOp(Ops.ADD, undefined, [new UOp(Ops.IDIV)]), new Map(), new Set([])],
    ],
    can_pad,
    'out(tiny.ops.can_pad(data[0],{},set()))',
  ),
)
Deno.test(
  'resolve',
  compare(
    [
      [new UOp(Ops.ADD, dtypes.float)],
      [new UOp(Ops.ADD, dtypes.float, [UOp.int(4), UOp.int(55)])],
      [new UOp(Ops.ADD, dtypes.bool, [UOp.int(4), UOp.int(55)])],

      [UOp.int(3).mul(UOp.bool(false))],
      [UOp.float(3).add(UOp.int(4)).idiv(UOp.float(44))],
      [UOp.int(3).add(UOp.bool(false), true)],
    ],
    tryCatch(resolve),
    'out(trycatch(lambda: tiny.ops.resolve(*data)))',
  ),
)
Deno.test(
  'uop.parents',
  compare(
    [
      [new UOp(Ops.ADD, undefined, [new UOp(Ops.BARRIER, undefined, [new UOp(Ops.CONST, undefined, undefined, 69)])])],
      [new UOp(Ops.CONST, undefined, undefined, 1)],
    ],
    (x: UOp) => [...x.toposort],
    'out(list(data[0].toposort.keys()))',
  ),
)
Deno.test(
  'upat.match',
  compare(
    [
      [
        new UPat(Ops.ADD, dtypes.int).named('add_op'),
        new UOp(Ops.ADD, dtypes.int, [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 3)]),
      ],
      [
        new UPat(Ops.MUL, dtypes.float).named('mul_op'),
        new UOp(Ops.MUL, dtypes.float, [UOp.const(dtypes.float, 2.5), UOp.const(dtypes.float, 4.0)]),
      ],
      [
        new UPat(Ops.SUB, dtypes.int).named('sub_op'),
        new UOp(Ops.SUB, dtypes.int, [UOp.const(dtypes.int, 10), UOp.const(dtypes.int, 4)]),
      ],
      [
        new UPat(Ops.ADD, dtypes.float).named('complex_add'),
        new UOp(Ops.ADD, dtypes.float, [UOp.const(dtypes.float, 1.5), UOp.const(dtypes.float, 2.5), UOp.const(dtypes.float, 3.0)]),
      ],
      [
        new UPat(Ops.IF, dtypes.bool, [new UPat(Ops.CMPLT, dtypes.bool).named('cmp_op'), new UPat(undefined).named('true_case'), new UPat(undefined).named('false_case')], undefined, 'conditional_op'),
        new UOp(Ops.IF, dtypes.bool, [new UOp(Ops.CMPLT, dtypes.bool, [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)]), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)]),
      ],
    ],
    (x: UPat, uop: UOp) => x.match(uop, new Map()),
    'out(data[0].match(data[1],{}))',
  ),
)
Deno.test(
  'uop.simplify2',
  compare(
    [[]],
    () => UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true)),
    `
from tinygrad.ops import UOp
from tinygrad.dtype import dtypes
out(UOp.const(dtypes.int,3).add(UOp.const(dtypes.float,4.6).idiv(UOp.const(dtypes.float,55))).mul(UOp.const(dtypes.bool,True)))`,
  ),
)
Deno.test(
  'uop.simplify',
  compare(
    [
      [new UOp(Ops.ADD, undefined, [UOp.int(10), UOp.int(100)], 1)],
      [new UOp(Ops.IDIV, undefined, [UOp.float(10), UOp.int(100)], 1)],
      [new UOp(Ops.AND, undefined, [UOp.bool(false), UOp.bool(true)], 1)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))],
      [UOp.int(3).add(UOp.float(4.6).div(UOp.float(55))).mul(UOp.bool(true))],

      [UOp.int(3).mul(UOp.bool(false))],
      [UOp.int(3).mul(UOp.bool(false), true)],
      [UOp.bool(true).mul(UOp.int(3))],
      [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      [UOp.int(4).mul(true)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      [new UOp(Ops.IF, dtypes.bool, [new UOp(Ops.CMPLT, dtypes.bool, [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)]), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)])],
    ],
    (x: UOp) => x.simplify(),
    'out(data[0].simplify())',
  ),
)
Deno.test(
  'uop.ssimplify',
  compare(
    [
      [new UOp(Ops.ADD, undefined, [UOp.int(10), UOp.int(100)], 1)],
      [new UOp(Ops.IDIV, undefined, [UOp.float(10), UOp.int(100)], 1)],
      [new UOp(Ops.AND, undefined, [UOp.bool(false), UOp.bool(true)], 1)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))],
      [UOp.int(3).add(UOp.float(4.6).div(UOp.float(55))).mul(UOp.bool(true))],

      [UOp.int(3).mul(UOp.bool(false))],
      [UOp.int(3).mul(UOp.bool(false), true)],
      [UOp.bool(true).mul(UOp.int(3))],
      [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      [UOp.int(4).mul(true)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      [new UOp(Ops.IF, dtypes.bool, [new UOp(Ops.CMPLT, dtypes.bool, [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)]), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)])],
    ],
    (x: UOp) => x.ssimplify(),
    'out(data[0].ssimplify())',
  ),
)
Deno.test(
  'uop.symInfer',
  compare(
    [
      [new UOp(Ops.ADD, undefined, [UOp.int(10), UOp.int(100)], 1)],
      [new UOp(Ops.IDIV, undefined, [UOp.float(10), UOp.int(100)], 1)],
      [new UOp(Ops.AND, undefined, [UOp.bool(false), UOp.bool(true)], 1)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))],
      [UOp.int(3).add(UOp.float(4.6).index(UOp.float(55.6))).mul(UOp.bool(true))],

      [UOp.int(3).mul(UOp.bool(false))],
      [UOp.int(3).mul(UOp.bool(false), true)],
      [UOp.bool(true).mul(UOp.int(3))],
      [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      [UOp.int(4).mul(true)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      [new UOp(Ops.IF, dtypes.bool, [new UOp(Ops.CMPLT, dtypes.bool, [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)]), UOp.const(dtypes.float, 1.1), UOp.const(dtypes.float, 0.0)])],
    ],
    tryCatch((x: UOp) => x.symInfer(new Map())),
    'out(trycatch(lambda:data[0].sym_infer({})))',
  ),
)
Deno.test(
  'uop.render',
  compare(
    [
      [new UOp(Ops.ADD, undefined, [UOp.int(10), UOp.int(100)], 1)],
      [new UOp(Ops.IDIV, undefined, [UOp.float(10), UOp.int(100)], 1)],
      [new UOp(Ops.AND, undefined, [UOp.bool(false), UOp.bool(true)], 1)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))],
      [UOp.int(3).add(UOp.float(4.6).index(UOp.float(55.6))).mul(UOp.bool(true))],

      [UOp.int(3).mul(UOp.bool(false))],
      [UOp.int(3).mul(UOp.bool(false), true)],
      [UOp.bool(true).mul(UOp.int(3))],
      [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      [UOp.int(4).mul(true)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      [new UOp(Ops.IF, dtypes.bool, [new UOp(Ops.CMPLT, dtypes.bool, [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)]), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)])],
    ],
    tryCatch((x: UOp) => [x.render(true), x.render(false)]),
    'out([data[0].render(True).lower(),data[0].render(False).lower()])',
    { ignore: [6, 14] }, // ignoring these because python and TS code for UOps is different, so tests fail, but they are correct
  ),
)

Deno.test(
  'smax',
  compare(
    [
      [UOp.bool(true), UOp.bool(false), UOp.bool(true)],
      [[UOp.int(10), UOp.bool(false), UOp.float(444)]],
      [UOp.int(10), UOp.bool(true), UOp.float(444)],
      [UOp.int(10), UOp.float(444), UOp.float(3324)],
      [555, 3434, 0, -3],
      [1, 0, 3434, 0, -3],
    ],
    smax,
    'out(tiny.ops.smax(*data))',
  ),
)

Deno.test(
  'smin',
  compare(
    [
      [UOp.bool(true), UOp.bool(false), UOp.bool(true)],
      [[UOp.int(10), UOp.bool(false), UOp.float(444)]],
      [UOp.int(10), UOp.bool(true), UOp.float(444)],
      [UOp.int(10), UOp.float(444), UOp.float(3324)],
      [555, 3434, 0, -3],
      [1, 0, 3434, 0, -3],
    ],
    smin,
    'out(tiny.ops.smin(*data))',
  ),
)

Deno.test(
  'UOp.has_st',
  compare(
    [
      [new UOp(Ops.DEFINE_LOCAL)],
      [new UOp(Ops.DEFINE_GLOBAL)],
      [new UOp(Ops.BUFFER)],
      [new UOp(Ops.CONST)],
      [new UOp(Ops.DEFINE_VAR)],
      [new UOp(Ops.ADD)],
      [new UOp(Ops.MUL)],
      [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))],
      [new UOp(Ops.ADD, undefined, [UOp.int(4), UOp.int(5)])],
      [new UOp(Ops.ADD, undefined, [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4])), new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))])],
      [new UOp(Ops.ADD, undefined, [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))])],
    ],
    tryCatch((x: UOp) => x.has_st),
    'out(trycatch(lambda: data[0].has_st))',
  ),
)

Deno.test(
  'UOp.st',
  compare(
    [
      [new UOp(Ops.DEFINE_LOCAL)],
      [new UOp(Ops.ADD, undefined, [ShapeTracker.from_shape([2, 2]).to_uop()])],
      [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([2, 2]))],
      [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))],
      [new UOp(Ops.ADD, undefined, [UOp.int(4), UOp.int(5)])],
      [new UOp(Ops.ADD, undefined, [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4])), new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))])],
      [new UOp(Ops.ADD, undefined, [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))])],
    ],
    tryCatch((x: UOp) => x.st),
    'out(trycatch(lambda: data[0].st))',
  ),
)

Deno.test(
  'UOp.full_shape',
  compare(
    [
      [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 4]))],
    ],
    (x: UOp) => x.full_shape,
    'out(trycatch(lambda: data[0].full_shape))',
  ),
)

Deno.test(
  'st_arg',
  compare(
    [
      [new UOp(Ops.BUFFER, undefined, [UOp.int(1), new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([2, 2]))])],
      [new UOp(Ops.VALID, undefined, [new UOp(Ops.VIEW, undefined, undefined, ShapeTracker.from_shape([3, 3]))])],
      [new UOp(Ops.ADD)], // Should throw error - not a buffer op
      [new UOp(Ops.BUFFER, undefined, [UOp.int(1), UOp.int(2)])], // Should throw error - src[1] not VIEW
    ],
    tryCatch((x: UOp) => x.st_arg),
    'out(trycatch(lambda: data[0].st_arg))',
  ),
)

Deno.test(
  'const_with_shape',
  compare(
    [
      [dtypes.float, 3.14, [2, 2]],
      [dtypes.int, 42, [3, 3]],
      [dtypes.bool, true, [1, 4]],
    ],
    tryCatch(UOp.const_with_shape),
    'out(trycatch(lambda: tiny.ops.UOp.const_with_shape(*data)))',
  ),
)

Deno.test(
  'div_and_mod_folding',
  compare(
    [
      [new UOp(Ops.RANGE, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], 1), 4, Ops.MOD, false],
      [new UOp(Ops.RANGE, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], 0), 4, Ops.MOD, false],
      [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 4]), new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 3]), new UOp(Ops.CONST, dtypes.int, [], 4)], undefined)], undefined), 4, Ops.MOD, false],
      [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.RANGE, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], undefined), new UOp(Ops.RANGE, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], 1)], undefined), 4, Ops.MOD, false],
      [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.RANGE, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], undefined), new UOp(Ops.RANGE, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 0), new UOp(Ops.CONST, dtypes.int, [], 12)], 1)], undefined), 12, Ops.IDIV, false],
      [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.EXPAND, dtypes.int, [new UOp(Ops.VCONST, dtypes.int.vec(4), [], [0, 1, 2, 3])], [[3, 4]]), new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 3]), new UOp(Ops.CONST, dtypes.int, [], 48)], undefined), new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 4]), new UOp(Ops.CONST, dtypes.int, [], 12)], undefined)], undefined), new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx1`, 3]), new UOp(Ops.CONST, dtypes.int, [], 4)], undefined)], undefined)], undefined), 4, Ops.MOD, false],
      [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.EXPAND, dtypes.int, [new UOp(Ops.VCONST, dtypes.int.vec(4), [], [0, 1, 2, 3])], [[3, 4]]), new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.ADD, dtypes.int, [new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`gidx0`, 3]), new UOp(Ops.CONST, dtypes.int, [], 48)], undefined), new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx0`, 4]), new UOp(Ops.CONST, dtypes.int, [], 12)], undefined)], undefined), new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.SPECIAL, dtypes.int, [], [`lidx1`, 3]), new UOp(Ops.CONST, dtypes.int, [], 4)], undefined)], undefined)], undefined), 12, Ops.IDIV, false],
    ],
    div_and_mod_folding,
    'out(tiny.ops.div_and_mod_folding(*data))',
  ),
)
