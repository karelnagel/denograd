import { _assert_valid_uop, Kernel, Opt, OptOps, verify_ast } from '../../denograd/codegen/kernel.ts'
import { KernelInfo, Ops, UOp } from '../../denograd/ops.ts'
import { compare, tryCatch } from '../helpers.ts'
import type { Renderer } from '../../denograd/renderer/index.ts'
import { dtypes } from '../../denograd/dtype.ts'
import { ShapeTracker } from '../../denograd/shape/shapetracker.ts'
import { View } from '../../denograd/shape/view.ts'
import { ClangRenderer } from '../../denograd/renderer/cstyle.ts'
import { kernelInputs, pyKernel, tsKernel } from '../engine/kernel-inputs.ts'
import { describe } from 'vitest'

describe(
  'Opt.real_axis',
  compare(
    [
      [new Opt(OptOps.UNROLL, 0, 0), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [Ops.MAX, [1]])], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 4), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer())],
      [new Opt(OptOps.UPCAST, 0, 3), new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 32, 3, 3], [288, 9, 3, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 32, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer())],
    ],
    (opt: Opt, k: Kernel) => opt.real_axis(k),
    'out(data[0].real_axis(data[1]))',
  ),
)

const kernels = () => kernelInputs().map(([opts, ast]) => [new Kernel(ast, opts)] as [Kernel])
describe(
  'Kernel.init',
  compare(
    () => kernelInputs().map((x) => x.toReversed() as [UOp, Renderer]),
    (ast: UOp, opts: Renderer) => tsKernel(new Kernel(ast, opts)),
    [
      `k = tiny.codegen.kernel.Kernel(*data)`,
      pyKernel,
    ],
  ),
)
describe(
  'Kernel.membufs',
  compare(
    kernels,
    (k: Kernel) => k.membufs,
    'out(data[0].membufs)',
  ),
)
describe(
  'Kernel.float4_axis',
  compare(
    () => kernels().flatMap(([k]) => [0, 1].map((i) => [k, i] as [Kernel, number])),
    (k: Kernel, i: number) => k.float4_axis(i),
    'out(data[0].float4_axis(data[1]))',
  ),
)
describe(
  'Kernel.upcasted_axis',
  compare(
    () => kernels().flatMap(([k]) => [0, 1].map((i) => [k, i] as [Kernel, number])),
    (k: Kernel, axis: number) => k.upcasted_axis(axis),
    'out(data[0].upcasted_axis(data[1]))',
  ),
)
describe(
  'Kernel.first_reduce',
  compare(
    kernels,
    (k: Kernel) => k.first_reduce,
    'out(data[0].first_reduce)',
  ),
)
describe(
  'Kernel.colors',
  compare(
    kernels,
    (k: Kernel) => k.colors(),
    'out(data[0].colors())',
  ),
)
describe(
  'Kernel.colored_shape',
  { todo: true },
  compare(
    kernels,
    (k: Kernel) => k.colored_shape(),
    'out(data[0].colored_shape())',
  ),
)

describe(
  'Kernel.shift_to',
  compare(
    [
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [Ops.MAX, [1]])], undefined)], undefined), new ClangRenderer()), 1, 10, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 3, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 3, false, undefined],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0), new UOp(Ops.CONST, dtypes.float, [], 0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 16, false, 1],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0), new UOp(Ops.CONST, dtypes.float, [], 0)], undefined)], undefined)], undefined), new ClangRenderer()), 0, 4, false, undefined],
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
describe(
  'Kernel.apply_opt',
  compare(
    [
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [Ops.MAX, [1]])], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UNROLL, 0, 0), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 3), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [25, 0, 5, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32, 1, 5, 5], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 4), true],
      [new Kernel(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [576, 9, 3, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64, 64, 3, 3], [0, 0, 0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined), new ClangRenderer()), new Opt(OptOps.UPCAST, 0, 3), true],
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
describe(
  'Kernel.required_optimizations',
  compare(
    kernels,
    (k: Kernel) => k.required_optimizations(),
    `out(data[0].required_optimizations())`,
  ),
)

describe(
  'Kernel.name',
  compare(
    kernels,
    (k: Kernel) => {
      Kernel.kernel_cnt.clear()
      return k.name
    },
    'out(data[0].name)',
  ),
)

describe(
  'Kernel.get_optimized_ast',
  compare(
    kernels,
    (k: Kernel) => k.get_optimized_ast(),
    'out(data[0].get_optimized_ast())',
  ),
)

describe(
  'Kernel.linearize',
  compare(
    kernels,
    (k: Kernel) => tsKernel(k.linearize()),
    [
      `k = data[0].linearize()`,
      pyKernel,
    ],
  ),
)

describe(
  'Kernel.to_program',
  compare(
    kernels,
    (k: Kernel) => {
      Kernel.kernel_cnt.clear()
      return k.to_program()
    },
    'out(data[0].to_program())',
    {
      skip: [7], // possibly bug in tinygrad(fails in python not in TS)
    },
  ),
)

describe(
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

describe(
  'verify_ast',
  compare(
    [
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, dtypes.float, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [Ops.MAX, [1]])], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], undefined)],
      [new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.1), new UOp(Ops.CONST, dtypes.float, [], 0.1)], undefined)], undefined)], new KernelInfo(0, 0, false))],
    ],
    verify_ast,
    'out(tiny.codegen.kernel.verify_ast(*data))',
  ),
)
