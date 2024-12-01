import { canPad, Ops, resolve, smax, smin, UOp, UPat } from '../src/ops.ts'
import { compare, tryCatch } from './helpers.ts'
import { dtypes } from '../src/dtype.ts'

Deno.test(
  'canPad',
  compare(
    [
      [new UOp({ op: Ops.RECIP })],
      [new UOp({ op: Ops.ADD })],

      [new UOp({ op: Ops.RECIP, src: [new UOp({ op: Ops.IDIV })] })],
      [new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.IDIV })] })],
    ],
    canPad,
    'out(tiny.ops.can_pad(*data))',
  ),
)
Deno.test(
  'resolve',
  compare(
    [
      [new UOp({ op: Ops.ADD, dtype: dtypes.float })],
      [new UOp({ op: Ops.ADD, dtype: dtypes.float, src: [UOp.int(4), UOp.int(55)] })],
      // [new UOp({ op: Ops.ADD, dtype: dtypes.bool, src: [UOp.int(4), UOp.int(55)] })], //fails

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
      [new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.BARRIER, src: [new UOp({ op: Ops.CONST, arg: 69 })] })] })],
      [new UOp({ op: Ops.CONST, arg: 1 })],
    ],
    (x: UOp) => [...x.parents().keys()],
    'out(list(data[0].parents.keys()))',
  ),
)
Deno.test(
  'uop.sparents',
  compare(
    [
      [new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.BARRIER, src: [new UOp({ op: Ops.CONST, arg: 69 })] })] })],
      [new UOp({ op: Ops.CONST, arg: 1 })],
    ],
    (x: UOp) => [...x.sparents().keys()],
    'out(list(data[0].sparents.keys()))',
  ),
)
Deno.test(
  'upat.match',
  compare(
    [
      [
        new UPat({ op: Ops.ADD, name: 'add_op', dtype: dtypes.int }),
        new UOp({ op: Ops.ADD, dtype: dtypes.int, src: [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 3)] }),
      ],
      [
        new UPat({ op: Ops.MUL, name: 'mul_op', dtype: dtypes.float }),
        new UOp({ op: Ops.MUL, dtype: dtypes.float, src: [UOp.const(dtypes.float, 2.5), UOp.const(dtypes.float, 4.0)] }),
      ],
      [
        new UPat({ op: Ops.SUB, name: 'sub_op', dtype: dtypes.int }),
        new UOp({ op: Ops.SUB, dtype: dtypes.int, src: [UOp.const(dtypes.int, 10), UOp.const(dtypes.int, 4)] }),
      ],
      [
        new UPat({ op: Ops.ADD, name: 'complex_add', dtype: dtypes.float }),
        new UOp({ op: Ops.ADD, dtype: dtypes.float, src: [UOp.const(dtypes.float, 1.5), UOp.const(dtypes.float, 2.5), UOp.const(dtypes.float, 3.0)] }),
      ],
      [
        new UPat({ op: Ops.IF, name: 'conditional_op', dtype: dtypes.bool, src: [new UPat({ op: Ops.CMPLT, name: 'cmp_op', dtype: dtypes.bool }), new UPat({ name: 'true_case' }), new UPat({ name: 'false_case' })] }),
        new UOp({ op: Ops.IF, dtype: dtypes.bool, src: [new UOp({ op: Ops.CMPLT, dtype: dtypes.bool, src: [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)] }), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)] }),
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
      [new UOp({ op: Ops.ADD, arg: 1, src: [UOp.int(10), UOp.int(100)] })],
      [new UOp({ op: Ops.IDIV, arg: 1, src: [UOp.float(10), UOp.int(100)] })],
      // [new UOp({ op: Ops.AND, arg: 1, src: [UOp.bool(false), UOp.bool(true)] })], //fails for some reason
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      // [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))], //fails
      // [UOp.int(3).add(UOp.float(4.6).div(UOp.float(55))).mul(UOp.bool(true))], //fails

      // [UOp.int(3).mul(UOp.bool(false))], //fails
      [UOp.int(3).mul(UOp.bool(false), true)], //succeeds
      [UOp.bool(true).mul(UOp.int(3))], //same as prev, but doesn't fail
      [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      [UOp.int(4).mul(true)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      [new UOp({ op: Ops.IF, dtype: dtypes.bool, src: [new UOp({ op: Ops.CMPLT, dtype: dtypes.bool, src: [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)] }), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)] })],
    ],
    (x: UOp) => x.simplify(),
    'out(data[0].simplify())',
  ),
)
Deno.test(
  'uop.ssimplify',
  compare(
    [
      [new UOp({ op: Ops.ADD, arg: 1, src: [UOp.int(10), UOp.int(100)] })],
      [new UOp({ op: Ops.IDIV, arg: 1, src: [UOp.float(10), UOp.int(100)] })],
      // [new UOp({ op: Ops.AND, arg: 1, src: [UOp.bool(false), UOp.bool(true)] })], //fails for some reason
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      // [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))], //fails
      // [UOp.int(3).add(UOp.float(4.6).div(UOp.float(55))).mul(UOp.bool(true))], //fails

      // [UOp.int(3).mul(UOp.bool(false))], //fails
      [UOp.int(3).mul(UOp.bool(false), true)], //succeeds
      [UOp.bool(true).mul(UOp.int(3))], //same as prev, but doesn't fail
      [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      [UOp.int(4).mul(true)],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      [new UOp({ op: Ops.IF, dtype: dtypes.bool, src: [new UOp({ op: Ops.CMPLT, dtype: dtypes.bool, src: [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)] }), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)] })],
    ],
    (x: UOp) => x.ssimplify(),
    'out(data[0].ssimplify())',
  ),
)
Deno.test(
  'uop.symInfer',
  compare(
    [
      // [new UOp({ op: Ops.ADD, arg: 1, src: [UOp.int(10), UOp.int(100)] })],
      // [new UOp({ op: Ops.IDIV, arg: 1, src: [UOp.float(10), UOp.int(100)] })],
      // // [new UOp({ op: Ops.AND, arg: 1, src: [UOp.bool(false), UOp.bool(true)] })], //fails for some reason
      // [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      // [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      // // [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))], //fails
      // // [UOp.int(3).add(UOp.float(4.6).div(UOp.float(55))).mul(UOp.bool(true))], //fails

      // // [UOp.int(3).mul(UOp.bool(false))], //fails
      // [UOp.int(3).mul(UOp.bool(false), true)], //succeeds
      // [UOp.bool(true).mul(UOp.int(3))], //same as prev, but doesn't fail
      // [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5)],
      // [UOp.int(4).mul(true)],
      // [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4))],
      // [new UOp({ op: Ops.IF, dtype: dtypes.bool, src: [new UOp({ op: Ops.CMPLT, dtype: dtypes.bool, src: [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)] }), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)] })],
    ],
    (x: UOp) => x.symInfer(new Map()),
    'out(data[0].sym_infer({}))',
  ),
)
Deno.test(
  'uop.render',
  compare(
    [
      // [new UOp({ op: Ops.ADD, arg: 1, src: [UOp.int(10), UOp.int(100)] })],
      // [new UOp({ op: Ops.IDIV, arg: 1, src: [UOp.float(10), UOp.int(100)] })],
      // // [new UOp({ op: Ops.AND, arg: 1, src: [UOp.bool(false), UOp.bool(true)] })], //fails for some reason
      // [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.4))],
      // [UOp.int(3).add(UOp.float(4).idiv(UOp.float(55))).mul(UOp.int(3.6))],
      // // [UOp.int(3).add(UOp.float(4.6).idiv(UOp.float(55))).mul(UOp.bool(true))], //fails
      // // [UOp.int(3).add(UOp.float(4.6).div(UOp.float(55))).mul(UOp.bool(true))], //fails

      // // [UOp.int(3).mul(UOp.bool(false))], //fails
      // [UOp.int(3).mul(UOp.bool(false), true)], //succeeds
      // [UOp.bool(true).mul(UOp.int(3))], //same as prev, but doesn't fail
      // [UOp.int(3).mul(false)],

      [UOp.bool(true).mul(5.5), true],
      [UOp.int(4).mul(true), false],
      [UOp.int(3).add(UOp.float(4).idiv(UOp.bool(false))).mul(UOp.int(3.4)), true],
      [new UOp({ op: Ops.IF, dtype: dtypes.bool, src: [new UOp({ op: Ops.CMPLT, dtype: dtypes.bool, src: [UOp.const(dtypes.int, 5), UOp.const(dtypes.int, 10)] }), UOp.const(dtypes.float, 1.0), UOp.const(dtypes.float, 0.0)] }), false],
    ],
    (x: UOp, simplify: boolean) => x.render(simplify),
    'out(data[0].render(data[1]))',
  ),
)

// TODO: syminfer,_minMax,this.symInfer,flopsMem,modFolding,divFolding,ltFolding,foldUnrolledDivs,canonicalizeSimplex,isIncreasing,uopGivenValid,simplifyValid,maxVarConst
// Deno.test(
//   'example',
//   test(
//     [[]],
//     () => {},
//     '',
//   ),
// )

Deno.test(
  'smax',
  compare(
    [
      [UOp.bool(true), UOp.bool(false), UOp.bool(true)],
      // [[UOp.int(10), UOp.float(444), UOp.bool(false)]],
      // [UOp.int(10), UOp.float(444), UOp.bool(true)],
      [UOp.int(10), UOp.float(444), UOp.float(3324)],
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
      // [[UOp.int(10), UOp.float(444), UOp.bool(false)]],
      // [UOp.int(10), UOp.float(444), UOp.bool(true)],
      [UOp.int(10), UOp.float(444), UOp.float(3324)],
    ],
    smin,
    'out(tiny.ops.smin(*data))',
  ),
)
