import { _assert_valid_uop, Kernel, Opt, OptOps, verify_ast } from '../../src/codegen/kernel.ts'
import { KernelInfo, Ops, UOp } from '../../src/ops.ts'
import { compare, tryCatch } from '../helpers.ts'
import { Renderer } from '../../src/renderer/index.ts'
import { DType, dtypes, PtrDType } from '../../src/dtype.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { ClangRenderer } from '../../src/renderer/cstyle.ts'
import { PythonRenderer } from '../../src/runtime/ops_python.ts'

Deno.test(
  'Opt.real_axis',
  compare(
    [
      [new Opt(OptOps.UNROLL, 0, 0), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 32, 3, 3], [288, 9, 3, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 32, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
    ],
    (opt: Opt, k: Kernel) => opt.real_axis(k),
    'out(data[0].real_axis(data[1]))',
  ),
)
export const kernelKeys = ['ast', 'opts', 'vars', 'bufs', 'applied_opts', 'group_for_reduces', 'upcasted', 'local_dims', 'tensor_core', 'tensor_core_opts', 'use_tensor_cores', 'bufs_for_tensor_core', 'dont_use_locals', 'sts', 'reduceops', 'full_buf_index'] as const
export const tsKernel = (k: Kernel) => kernelKeys.map((key) => k[key])
export const pyKernel = `out([getattr(k,key,None) for key in [${kernelKeys.map((k) => `"${k}"`)}]])`

const kernelInputs = (): [UOp, Renderer][] => [
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new PythonRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new PythonRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new PythonRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new PythonRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new PythonRenderer()],
  [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new PythonRenderer()],
]
const kernels = () => kernelInputs().map((i) => [new Kernel(...i)] as [Kernel])

Deno.test(
  'Kernel.init',
  compare(
    kernelInputs(),
    (ast: UOp, opts: Renderer) => tsKernel(new Kernel(ast, opts)),
    [
      `k = tiny.codegen.kernel.Kernel(*data)`,
      pyKernel,
    ],
  ),
)
Deno.test(
  'Kernel.membufs',
  compare(
    kernels(),
    (k: Kernel) => k.membufs,
    'out(data[0].membufs)',
  ),
)
Deno.test(
  'Kernel.float4_axis',
  compare(
    kernels().flatMap(([k]) => [0, 1].map((i) => [k, i] as [Kernel, number])),
    (k: Kernel, i: number) => k.float4_axis(i),
    'out(data[0].float4_axis(data[1]))',
  ),
)
Deno.test(
  'Kernel.upcasted_axis',
  compare(
    kernels().flatMap(([k]) => [0, 1].map((i) => [k, i] as [Kernel, number])),
    (k: Kernel, axis: number) => k.upcasted_axis(axis),
    'out(data[0].upcasted_axis(data[1]))',
  ),
)
Deno.test(
  'Kernel.first_reduce',
  compare(
    kernels(),
    (k: Kernel) => k.first_reduce,
    'out(data[0].first_reduce)',
  ),
)
Deno.test(
  'Kernel.colors',
  compare(
    kernels(),
    (k: Kernel) => k.colors(),
    'out(data[0].colors())',
  ),
)
Deno.test.ignore(
  'Kernel.colored_shape',
  compare(
    kernels(),
    (k: Kernel) => k.colored_shape(),
    'out(data[0].colored_shape())',
  ),
)

Deno.test(
  'Kernel.shift_to',
  compare(
    [
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()), 1, 10, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 3, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 3, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0), new UOp(Ops.CONST, dtypes.float, [], 0)], undefined)], undefined)], undefined), new PythonRenderer()), 0, 16, false, 1],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0), new UOp(Ops.CONST, dtypes.float, [], 0)], undefined)], undefined)], undefined), new PythonRenderer()), 0, 4, false, undefined],
    ],
    tryCatch((k: Kernel, axis: number, amount: number, top: boolean, insert_before?: number) => {
      k.shift_to(axis, amount, top, insert_before)
      return tsKernel(k)
    }),
    [
      `k = data[0]`,
      `k.shift_to(*data[1:])`,
      pyKernel,
    ],
  ),
)
Deno.test(
  'Kernel.apply_opt',
  compare(
    [
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UNROLL, 0, 0), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 3), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 3), true],
    ],
    tryCatch((k: Kernel, opt: Opt, append_opt: boolean) => {
      k.apply_opt(opt, append_opt)
      return tsKernel(k)
    }),
    [
      `k = data[0]`,
      `k.apply_opt(*data[1:])`,
      pyKernel,
    ],
  ),
)
Deno.test(
  'Kernel.required_optimizations',
  compare(
    kernels(),
    (k: Kernel) => k.required_optimizations(),
    `out(data[0].required_optimizations())`,
  ),
)
Deno.test(
  'Kernel.hand_coded_optimizations',
  compare(
    kernels(),
    (k: Kernel) => tsKernel(k.hand_coded_optimizations()),
    [
      `k = data[0].hand_coded_optimizations()`,
      pyKernel,
    ],
    {
      ignore: [8, 10], // TODO: they generate wrong shapetracked, one view is too much, if you comment last this.apply_opt(), then it's works fine
    },
  ),
)
Deno.test(
  'Kernel.name',
  compare(
    kernels(),
    (k: Kernel) => k.name,
    'out(data[0].name)',
  ),
)

Deno.test(
  'Kernel.get_optimized_ast',
  compare(
    kernels(),
    (k: Kernel) => k.get_optimized_ast(),
    'out(data[0].get_optimized_ast())',
  ),
)

Deno.test(
  'Kernel.linearize',
  compare(
    kernels(),
    (k: Kernel) => tsKernel(k.linearize()),
    [
      `k = data[0].linearize()`,
      pyKernel,
    ],
    {
      ignore: [7], // possibly bug in tinygrad
    },
  ),
)

Deno.test(
  'Kernel.to_program',
  compare(
    kernels(),
    (k: Kernel) => k.to_program(),
    'out(data[0].to_program())',
    {
      ignore: [7], // possibly bug in tinygrad(fails in python not in TS)
      ignoreKeys: ['src'], // PythonRenderer generates other b64
    },
  ),
)

Deno.test(
  '_assert_valid_uop',
  compare(
    [
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new ShapeTracker([new View([], [], 0, undefined, true)]), new Map<UOp, ShapeTracker>([])],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new ShapeTracker([new View([1], [0], 0, undefined, true)]), new Map<UOp, ShapeTracker>([])],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([5], [1], 0, undefined, true)])), new ShapeTracker([new View([5], [1], 0, undefined, true)]), new Map<UOp, ShapeTracker>([])],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new ShapeTracker([new View([32], [1], 0, undefined, true)]), new Map<UOp, ShapeTracker>([])],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new ShapeTracker([new View([10], [1], 0, undefined, true)]), new Map<UOp, ShapeTracker>([])],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([16], [1], 0, undefined, true)])), new ShapeTracker([new View([16], [1], 0, undefined, true)]), new Map<UOp, ShapeTracker>([])],
    ],
    _assert_valid_uop,
    'out(tiny.codegen.kernel._assert_valid_uop(*data))',
  ),
)

Deno.test(
  'verify_ast',
  compare(
    [
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], new KernelInfo(0, 0, false))],
    ],
    verify_ast,
    'out(tiny.codegen.kernel.verify_ast(*data))',
  ),
)
