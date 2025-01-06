import { Kernel } from '../../src/codegen/kernel.ts'
import { DeviceType } from '../../src/device.ts'
import { dtypes } from '../../src/dtype.ts'
import { get_kernel, get_runner } from '../../src/engine/realize.ts'
import { Ops, UOp } from '../../src/ops.ts'
import { Renderer } from '../../src/renderer/index.ts'
import { PythonRenderer } from '../../src/runtime/ops_python.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { pyKernel, tsKernel } from '../codegen/kernel.test.ts'
import { compare } from '../helpers.ts'

const kernelInputs = (): [Renderer, UOp][] => [
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 0.0), new UOp(Ops.CONST, dtypes.float, [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 4, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 9, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2], [1], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2], [1], 0, undefined, true)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 2, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([5], [1], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([5], [1], 5, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2, 2], [2, 1], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2, 2], [2, 1], 2, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1, 2], [0, 1], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([1, 2], [0, 1], 2, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2, 3, 2, 2], [12, 4, 2, 1], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.int, [new UOp(Ops.DEFINE_GLOBAL, dtypes.int.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([2, 3, 2, 2], [4, 1, 8, 1], 0, undefined, false)]))], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 0, undefined, true)])), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.CMPLT, dtypes.bool, [new UOp(Ops.CAST, dtypes.float, [new UOp(Ops.CAST, dtypes.int, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 0, undefined, true)]))], undefined)], undefined)], undefined), new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 0, undefined, true)]))], undefined)], undefined), new UOp(Ops.ADD, dtypes.float, [new UOp(Ops.CAST, dtypes.float, [new UOp(Ops.CAST, dtypes.int, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 0, undefined, true)]))], undefined)], undefined)], undefined), new UOp(Ops.WHERE, dtypes.float, [new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, dtypes.float, [], 1), new UOp(Ops.CONST, dtypes.float, [], 0)], undefined)], undefined), new UOp(Ops.CAST, dtypes.float, [new UOp(Ops.CAST, dtypes.int, [new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([4], [1], 0, undefined, true)]))], undefined)], undefined)], undefined)], undefined)], undefined)], undefined)],
]

Deno.test(
  'realize.get_kernel',
  compare(
    kernelInputs,
    (renderer, ast) => tsKernel(get_kernel(renderer, ast)),
    [
      'k = tiny.engine.realize.get_kernel(*data)',
      pyKernel,
    ],
  ),
)
Deno.test(
  'realize.get_optimized_ast',
  compare(
    kernelInputs,
    (renderer, ast) => get_kernel(renderer, ast).get_optimized_ast(),
    [
      'out(tiny.engine.realize.get_kernel(*data).get_optimized_ast())',
    ],
  ),
)
Deno.test(
  'realize.linearize',
  compare(
    kernelInputs,
    (renderer, ast) => get_kernel(renderer, ast).linearize().uops,
    [
      'out(tiny.engine.realize.get_kernel(*data).linearize().uops)',
    ],
  ),
)
Deno.test(
  'realize.to_program',
  compare(
    kernelInputs,
    (renderer, ast) => {
      Kernel.kernel_cnt = {}
      return get_kernel(renderer, ast).to_program()
    },
    'out(tiny.engine.realize.get_kernel(*data).to_program())',
    {
      ignoreKeys: ['src'],
    },
  ),
)

Deno.test(
  'realize.get_runner',
  compare(
    ()=>kernelInputs().map(([r, ast]) => [r.device, ast] as [DeviceType, UOp]),
    (d, ast) => {
      Kernel.kernel_cnt = {}
      const runner = get_runner(d, ast)
      return [runner.p]
    },
    [
      'runner = tiny.engine.realize.get_runner(*data)',
      'out([runner.p])',
    ],
    {
      ignoreKeys: ['src'],
    },
  ),
)

// Deno.test(
//   'CompiledRunner.init',
//   compare(
//     kernelInputs(),
//     (renderer, ast) => {
//       const kernel = get_kernel(renderer, ast)
//       const program = kernel.to_program()
//       const runner = new CompiledRunner(program)
//       runner.call()
//     },
//     'out(tiny.engine.realize.CompiledRunner(*data))',
//   ),
// )
// Deno.test.ignore(
//   'CompiledRunner.call',
//   compare(
//     [],
//     tryCatch((runner: CompiledRunner, rawbufs: Buffer[], var_vals: Map<Variable, number>, wait?: boolean) => runner.call(rawbufs, var_vals, wait)),
//     'out(data[0](*data[1:]))',
//   ),
// )

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
