import { Kernel } from '../../src/codegen/kernel.ts'
import { DType, PtrDType } from '../../src/dtype.ts'
import { get_kernel } from '../../src/engine/realize.ts'
import { Ops, UOp } from '../../src/ops.ts'
import { ClangRenderer } from '../../src/renderer/cstyle.ts'
import { Renderer } from '../../src/renderer/index.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { compare } from '../helpers.ts'

const kernelKeys: (keyof Kernel)[] = ['ast', 'opts', 'reduceops', 'vars', 'bufs', 'full_buf_index', 'sts', 'applied_opts', 'group_for_reduces', 'uops', 'upcasted', 'local_dims', 'tensor_core', 'tensor_core_opts', 'use_tensor_cores', 'bufs_for_tensor_core', 'dont_use_locals']
Deno.test(
  'realize.get_kernel',
  compare(
    [
      [new ClangRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new ClangRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new ClangRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new ClangRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
      [new ClangRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.LOAD, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [Ops.MAX, [1]])], undefined)], undefined)],
      [new ClangRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
    ],
    (renderer: Renderer, ast: UOp) => {
      const k = get_kernel(renderer, ast)
      return kernelKeys.map((key) => k[key])
    },
    [
      'k = tiny.engine.realize.get_kernel(*data)',
      `out([${kernelKeys.map((k) => `getattr(k,"${k}",None)`)}])`,
    ],
  ),
)

Deno.test(
  'CompiledRunner.call',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'BufferCopy.copy',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'BufferCopy.call',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'get_runner',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'ExecItem.run',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'lower_schedule_item',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'lower_schedule',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'run_schedule',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)
