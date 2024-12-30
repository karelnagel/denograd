import { Opt, OptOps } from '../src/codegen/kernel.ts'
import { IndexContext } from '../src/codegen/lowerer.ts'
import { BasicBlock } from '../src/codegen/linearize.ts'
import { DType, dtypes, PtrDType } from '../src/dtype.ts'
import { KernelInfo, Ops, spec, UOp, UPat } from '../src/ops.ts'
import { ClangRenderer } from '../src/renderer/cstyle.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'
import { View } from '../src/shape/view.ts'
import { compare } from './helpers.ts'
import { LazyBuffer } from '../src/engine/lazy.ts'
import { Metadata } from '../src/helpers.ts'
import { Tensor } from '../src/tensor.ts'

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
      [[new LazyBuffer(`PYTHON`, new ShapeTracker([new View([10], [1], 0, undefined, true)]), dtypes.int, Ops.EMPTY, undefined, undefined, undefined, undefined)]],
      [[new LazyBuffer(`PYTHON`, new ShapeTracker([new View([5, 2], [2, 1], 0, undefined, true)]), dtypes.int, undefined, undefined, undefined, new LazyBuffer(`PYTHON`, new ShapeTracker([new View([10], [1], 0, undefined, true)]), dtypes.int, Ops.EMPTY, undefined, undefined, undefined, undefined), new Metadata(`reshape`, ``, false))]],
      [new Tensor([[3, 3, 3], [5, 5, 5]])],
      // [new Kernel(new UOp(Ops.SINK), new ClangRenderer())],
      // [new Kernel(new UOp(Ops.SINK))],
      ...spec.patterns.map((p) => [p[0]] as any),
      [[[new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), []], [new UOp(Ops.CONST, dtypes.int, [], 0), []], [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), []], [new UOp(Ops.CONST, dtypes.float, [], 1.0), []], [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined), []], [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)], new KernelInfo(0, 0, false)), []]]), new Map([[new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined)]], [new UOp(Ops.CONST, dtypes.int, [], 0), [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined)]], [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)]], [new UOp(Ops.CONST, dtypes.float, [], 1.0), [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)]], [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined), [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)], new KernelInfo(0, 0, false))]]])], new UOp(Ops.BLOCK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)], new BasicBlock([], [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX, dtypes.float.ptr(), [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.CONST, dtypes.int, [], 0)], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0)], undefined)], new KernelInfo(0, 0, false))], undefined))]],
    ],
    (x) => x,
    'out(*data)',
  ),
)
