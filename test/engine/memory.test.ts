import { Buffer } from '../../denograd/device.ts'
import { dtypes } from '../../denograd/dtype.ts'
import { memory_planner } from '../../denograd/engine/memory.ts'
import { ScheduleItem } from '../../denograd/engine/schedule.ts'
import { Metadata } from '../../denograd/helpers.ts'
import { Ops, UOp } from '../../denograd/ops.ts'
import { ShapeTracker } from '../../denograd/shape/shapetracker.ts'
import { View } from '../../denograd/shape/view.ts'
import { compare } from '../helpers.ts'

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
      [[new ScheduleItem(new UOp(Ops.SINK, dtypes.void, [new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 0), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.LOAD, dtypes.float, [new UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(), [], 1), new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined)], undefined)], undefined), [new Buffer(`PYTHON`, 1, dtypes.float, undefined, undefined, undefined, 1, undefined, 0, false), new Buffer(`PYTHON`, 10, dtypes.float, undefined, undefined, undefined, 0, undefined, 0, false)], [new Metadata(`tolist`, ``, false)], [])]],
    ],
    memory_planner,
    'out(tiny.engine.memory.memory_planner(*data))',
  ),
)
