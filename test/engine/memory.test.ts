import { Buffer } from '../../src/device.ts'
import { DType, PtrDType } from '../../src/dtype.ts'
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
      [[new ScheduleItem(new UOp(Ops.SINK, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.STORE, new DType(-1, 0, `void`, undefined, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 0), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([], [], 0, undefined, true)])), new UOp(Ops.LOAD, new DType(11, 4, `float`, `f`, 1, undefined), [new UOp(Ops.DEFINE_GLOBAL, new PtrDType(11, 4, `float`, `f`, 1, undefined, new DType(11, 4, `float`, `f`, 1, undefined), false, 1), [], 1), new UOp(Ops.VIEW, new DType(-1, 0, `void`, undefined, 1, undefined), [], new ShapeTracker([new View([], [], 0, undefined, true)]))], undefined)], undefined)], undefined), [new Buffer(`PYTHON`, 1, new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, undefined, 1, undefined, 0, false), new Buffer(`PYTHON`, 10, new DType(11, 4, `float`, `f`, 1, undefined), undefined, undefined, undefined, 0, undefined, 0, false)], [new Metadata(`tolist`, ``, false)], new Set([]))]],
    ],
    memory_planner,
    'out(tiny.engine.memory.memory_planner(*data))',
  ),
)
