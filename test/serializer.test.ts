import { Kernel, Opt, OptOps } from '../src/codegen/kernel.ts'
import { IndexContext } from '../src/codegen/lowerer.ts'
import { dtypes } from '../src/dtype.ts'
import { Ops, spec, UOp, UPat } from '../src/ops.ts'
import { ClangRenderer } from '../src/renderer/cstyle.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'
import { View } from '../src/shape/view.ts'
import { compare } from './helpers.ts'

Deno.test(
  'serialize',
  compare(
    [
      [[4, 4, 'sdf', { sdf: 69 }, [{ df: 44, sdf: 'sdf' }]]],
      [Ops.ADD],
      [Ops.ASSIGN],
      [new UOp({ op: Ops.BARRIER, dtype: dtypes.float, arg: 5445 })],
      [new UPat({ op: Ops.ASSIGN, dtype: dtypes.floats, arg: 555, name: 'sdf' })],
      [new UPat({ op: Ops.IF, name: 'conditional_op', dtype: dtypes.bool, src: [new UPat({ op: Ops.CMPLT, name: 'cmp_op', dtype: dtypes.bool }), new UPat({ name: 'true_case' }), new UPat({ name: 'false_case' })] })],
      [dtypes.floats],
      [dtypes.default_int],
      [dtypes.imagef(2, 44, 42)],
      [dtypes.bool.ptr(true)],
      [dtypes.bool.ptr(false)],
      [
        new View({
          shape: [4, 55],
          strides: [new UOp({ op: Ops.MUL, dtype: dtypes.int, arg: undefined, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int, arg: 1, src: [] }), new UOp({ op: Ops.CONST, dtype: dtypes.int, arg: 55, src: [] })] }), 1],
          offset: 0,
          mask: undefined,
          contiguous: false,
        }),
      ],
      [ShapeTracker.from_shape([UOp.float(8), UOp.int(110), UOp.int(33)])],
      [new IndexContext([UOp.int(3)], [UOp.bool(true), UOp.float(4.4)], 4)],
      [new ClangRenderer()],
      [new Opt(OptOps.PADTO, 5, 666)],
      // [new Kernel(new UOp({ op: Ops.SINK }), new ClangRenderer())],
      // [new Kernel(new UOp({ op: Ops.SINK }))],
      ...spec.patterns.map((p) => [p[0]] as any),
    ],
    (x) => x,
    'out(*data)',

  ),
)
