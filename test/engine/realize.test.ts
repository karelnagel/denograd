import { DeviceType } from '../../src/device.ts'
import { DType, PtrDType } from '../../src/dtype.ts'
import { get_kernel, get_runner } from '../../src/engine/realize.ts'
import { Ops, UOp } from '../../src/ops.ts'
import { Renderer } from '../../src/renderer/index.ts'
import { PythonRenderer } from '../../src/runtime/ops_python.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { pyKernel, tsKernel } from '../codegen/kernel.test.ts'
import { compare } from '../helpers.ts'

const kernelInputs = (): [Renderer, UOp][] => [
  [new PythonRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([1], [0], 0, undefined, true)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 1.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([32], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([64], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10], [0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)])), new UOp(Ops.REDUCE_AXIS, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.LOAD, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([512, 10], [10, 1], 0, undefined, true)]))], undefined)], [Ops.MAX, [1]])], undefined)], undefined)],
  [new PythonRenderer(), new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [576, 1], 0, undefined, true)])), new UOp(Ops.WHERE, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.VALID, new DType(0, 1, `bool`, `?`, 1, undefined), [new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([10, 576], [0, 0], 0, undefined, false)]))], undefined), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0), new UOp(Ops.CONST, new DType(11, 4, `float`, `f`, 1, undefined), [], 0.0)], undefined)], undefined)], undefined)],
]

Deno.test(
  'realize.get_kernel',
  compare(
    kernelInputs(),
    (renderer, ast) => tsKernel(get_kernel(renderer, ast)),
    [
      'k = tiny.engine.realize.get_kernel(*data)',
      pyKernel,
    ],
    {
      ignore: [1, 2], // TS has one more view in sts
    },
  ),
)
Deno.test(
  'realize.get_kernel.to_program',
  compare(
    kernelInputs(),
    (renderer, ast) => get_kernel(renderer, ast).to_program(),
    'out(tiny.engine.realize.get_kernel(*data).to_program())',
    {
      ignoreKeys: ['src'],
      ignore: [4],
    },
  ),
)

Deno.test(
  'realize.get_runner',
  compare(
    kernelInputs().map(([r, ast]) => [r.device, ast] as [DeviceType, UOp]),
    (d, ast) => {
      const runner = get_runner(d, ast)

      return [runner.p]
    },
    [
      'runner = tiny.engine.realize.get_runner(*data)',
      'out([runner.p])',
    ],
    {
      ignoreKeys: ['src'],
      ignore: [4],
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
