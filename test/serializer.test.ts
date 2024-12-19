import { Opt, OptOps } from '../src/codegen/kernel.ts'
import { IndexContext } from '../src/codegen/lowerer.ts'
import { BasicBlock } from '../src/codegen/linearize.ts'
import { DType, dtypes, PtrDType } from '../src/dtype.ts'
import { KernelInfo, Ops, spec, UOp, UPat } from '../src/ops.ts'
import { ClangRenderer } from '../src/renderer/cstyle.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'
import { View } from '../src/shape/view.ts'
import { compare } from './helpers.ts'

Deno.test(
  'serialize',
  compare(
    [
      [new View([10, 576], [576, 1], 0, undefined, true)],
      [[4, 4, 'sdf', { sdf: 69 }, [{ df: 44, sdf: 'sdf' }]]],
      [Ops.ADD],
      [Ops.ASSIGN],
      [new UOp(Ops.BARRIER, dtypes.float, undefined, 5445)],
      [new UPat(Ops.ASSIGN, dtypes.floats, undefined, 555, 'sdf')],
      [new UPat(Ops.IF, dtypes.bool, [new UPat(Ops.CMPLT, dtypes.bool).named('cmp_op'), new UPat(undefined).named('true_case'), new UPat(undefined).named('false_case')], undefined, 'conditional_op')],
      [dtypes.floats],
      [dtypes.default_int],
      [dtypes.imagef(2, 44, 42)],
      [dtypes.bool.ptr(true)],
      [dtypes.bool.ptr(false)],
      [
        new View(
          [4, 55],
          [new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.CONST, dtypes.int, [], 1), new UOp(Ops.CONST, dtypes.int, [], 55)], undefined), 1],
          0,
          undefined,
          false,
        ),
      ],
      [ShapeTracker.from_shape([UOp.float(8), UOp.int(110), UOp.int(33)])],
      [new IndexContext([UOp.int(3)], [UOp.bool(true), UOp.float(4.4)], 4)],
      [new ClangRenderer()],
      [new Opt(OptOps.PADTO, 5, 666)],
      // [new Kernel(new UOp(Ops.SINK), new ClangRenderer())],
      // [new Kernel(new UOp(Ops.SINK))],
      ...spec.patterns.map((p) => [p[0]] as any),
      [[[new Map([[new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), []], [new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0), []], [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), []], [new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), []], [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined), []], [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined)], new KernelInfo(0, 0, false)), []]]), new Map([[new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined)]], [new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined)]], [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined)]], [new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined)]], [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined), [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined)], new KernelInfo(0, 0, false))]]])], new UOp(7, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined)], new BasicBlock([], [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(35, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(62, new DType(5, 4, `int`, `i`, 1, undefined), [], 0)], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0)], undefined)], new KernelInfo(0, 0, false))], undefined))]],
    ],
    (x) => x,
    'out(*data)',
  ),
)
