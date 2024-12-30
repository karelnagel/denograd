import { Buffer } from '../../src/device.ts'
import { DType, dtypes, PtrDType } from '../../src/dtype.ts'
import { memory_planner } from '../../src/engine/memory.ts'
import { ScheduleItem } from '../../src/engine/schedule.ts'
import { Metadata } from '../../src/helpers.ts'
import { Ops, UOp } from '../../src/ops.ts'
import { ShapeTracker } from '../../src/shape/shapetracker.ts'
import { View } from '../../src/shape/view.ts'
import { compare, tryCatch } from '../helpers.ts'

Deno.test(
  '_internal_memory_planner',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

Deno.test(
  'memory_planner',
  compare(
    [
      [[new ScheduleItem(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, dtypes.float, false, 1), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined)], undefined)], undefined), [new Buffer(`PYTHON`, 1, dtypes.float, undefined, undefined, undefined, 1, undefined, 0, false), new Buffer(`PYTHON`, 10, dtypes.float, undefined, undefined, undefined, 0, undefined, 0, false)], [new Metadata(`tolist`, ``, false)], new Set([]))]],
    ],
    memory_planner,
    'out(tiny.engine.memory.memory_planner(*data))',
  ),
)
