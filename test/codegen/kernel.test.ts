// deno-fmt-ignore-file
import { _assert_valid_uop, Kernel, Opt, OptOps, verify_ast } from '../../src/codegen/kernel.ts'
import { KernelInfo, UOp } from '../../src/ops.ts'
import { compare, tryCatch } from '../helpers.ts'
import { Renderer } from '../../src/renderer/index.ts'
import { DType, PtrDType } from '../../src/dtype.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { ClangRenderer } from '../../src/renderer/cstyle.ts'

Deno.test(
  'Opt.real_axis',
  compare(
    [
      [new Opt(OptOps.UNROLL, 0, 0), new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(22, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(33, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64, 32, 3, 3], [288, 9, 3, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64, 32, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer())]
    ],
    (opt: Opt, k: Kernel) => opt.real_axis(k),
    'out(data[0].real_axis(data[1]))',
  ),
)
const keys = ["ast", "opts", "vars", "bufs", "applied_opts", "group_for_reduces", "upcasted", "local_dims", "tensor_core", "tensor_core_opts", "use_tensor_cores", "bufs_for_tensor_core", "dont_use_locals", "sts", "reduceops", "full_buf_index"]as const
const tsKernelKeys = (k:Kernel)=>Object.fromEntries(keys.map(key=>[key,k[key]]))
const pyKernelKeys = `
keys = [${keys.map(k=>`"${k}"`)}]
out({key:getattr(k, key) for key in keys})
`
Deno.test(
  'Kernel.init',
  compare(
    [
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(22, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(33, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()]
    ],
    (ast: UOp, opts: Renderer) => tsKernelKeys(new Kernel(ast, opts)),
    `k = tiny.codegen.kernel.Kernel(*data)\n${pyKernelKeys}`,
  ),
)
const kernels = () => [
  new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()),
  new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(22, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(33, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()),
  new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()),
  new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()),
  new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()),
  new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()),

]
Deno.test(
  'Kernel.membufs',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.membufs,
    'out(data[0].membufs)',
  ),
)
Deno.test(
  'Kernel.float4_axis',
  compare(
    kernels().flatMap(k=>[0,1].map(i=>[k,i] as [Kernel,number])),
    (k:Kernel,i:number) => k.float4_axis(i),
    'out(data[0].float4_axis(data[1]))',
  ),
)
Deno.test(
  'Kernel.upcasted_axis',
  compare(
    kernels().flatMap(k=>[0,1].map(i=>[k,i] as [Kernel,number])),
    (k:Kernel,axis:number) => k.upcasted_axis(axis),
    'out(data[0].upcasted_axis(data[1]))',
  ),
)
Deno.test(
  'Kernel.first_reduce',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.first_reduce,
    'out(data[0].first_reduce)',
  ),
)
Deno.test(
  'Kernel.colors',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.colors(),
    'out(data[0].colors())',
  ),
)
Deno.test.ignore(
  'Kernel.colored_shape',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.colored_shape(),
    'out(data[0].colored_shape())',
  ),
)

Deno.test(
  'Kernel.shift_to',
  compare(
    [
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(22, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(33, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()), 1, 10, false, undefined],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 3, false, undefined],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 3, false, undefined]
    ],
    tryCatch((k:Kernel,axis:number,amount:number,top:boolean,insert_before?:number)=>{
      k.shift_to(axis,amount,top,insert_before)
      return tsKernelKeys(k)
    }),
    `k = data[0]
k.shift_to(*data[1:])
${pyKernelKeys}`,
  ),
)
Deno.test(
  'Kernel.apply_opt',
  compare(
    [
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(22, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(33, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UNROLL, 0, 0), true],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 3), true],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 3), true]

    ],
    tryCatch((k:Kernel,opt:Opt,append_opt:boolean)=>{
      k.apply_opt(opt,append_opt)
      return tsKernelKeys(k)
    },),
    `k = data[0]
k.apply_opt(*data[1:])
${pyKernelKeys}
`,
  ),
)
Deno.test(
  'Kernel.required_optimizations',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.required_optimizations(),
    `out(data[0].required_optimizations())`
  ),
)
Deno.test(
  'Kernel.hand_coded_optimizations',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel)=>tsKernelKeys(k.hand_coded_optimizations()),
    `k = data[0].hand_coded_optimizations()\n${pyKernelKeys}`,
  ),
)
Deno.test(
  'Kernel.name',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.name,
    'out(data[0].name)',
  ),
)

Deno.test(
  'Kernel.get_optimized_ast',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.get_optimized_ast(),
    'out(data[0].get_optimized_ast())',
  ),
)

Deno.test(
  'Kernel.linearize',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => tsKernelKeys(k.linearize()),
    `k = data[0].linearize()\n${pyKernelKeys}`,
  ),
)

Deno.test(
  'Kernel.to_program',
  compare(
    kernels().map(k=>[k] as [Kernel]),
    (k:Kernel) => k.to_program(),
    'out(data[0].to_program())',
    { ignore:[1] }
  ),
)

Deno.test(
  '_assert_valid_uop',
  compare(
    [
      [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([], [], 0, undefined, true)])), new ShapeTracker([new View([], [], 0, undefined, true)]), new Map<UOp,ShapeTracker>([])],
      [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new ShapeTracker([new View([1], [0], 0, undefined, true)]), new Map<UOp,ShapeTracker>([])],
      [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([5], [1], 0, undefined, true)])), new ShapeTracker([new View([5], [1], 0, undefined, true)]), new Map<UOp,ShapeTracker>([])],
      [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new ShapeTracker([new View([32], [1], 0, undefined, true)]), new Map<UOp,ShapeTracker>([])],
      [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new ShapeTracker([new View([10], [1], 0, undefined, true)]), new Map<UOp,ShapeTracker>([])],
      [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([16], [1], 0, undefined, true)])), new ShapeTracker([new View([16], [1], 0, undefined, true)]), new Map<UOp,ShapeTracker>([])]
    ],
    _assert_valid_uop,
    'out(tiny.codegen.kernel._assert_valid_uop(*data))',
  ),
)

Deno.test(
  'verify_ast',
  compare(
    [
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(22, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(33, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [40, [1]])], undefined)], undefined)],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new UOp(1, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(34, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(14, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(52, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(19, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(13, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), new UOp(62, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], new KernelInfo(0, 0, false))]
    ],
    verify_ast,
    'out(tiny.codegen.kernel.verify_ast(*data))',
  ),
)
