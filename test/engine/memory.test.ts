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
      // TODO: no way to test this currently cause Buffer isn't serializable
    ],
    memory_planner,
    'out(tiny.engine.memory.memory_planner(*data))',
  ),
)
