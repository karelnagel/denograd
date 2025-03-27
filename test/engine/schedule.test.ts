import { dtypes } from '../../jsgrad/dtype.ts'
import { apply_swizzle } from '../../jsgrad/engine/schedule.ts'
import { Ops, UOp } from '../../jsgrad/ops.ts'
import { ShapeTracker } from '../../jsgrad/shape/shapetracker.ts'
import { View } from '../../jsgrad/shape/view.ts'
import { compare } from '../helpers.ts'
import { describe as describe } from 'vitest'

describe(
  'apply_swizzle',
  compare(
    [
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)]))],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([10], [1], 0, undefined, true)]))],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([256], [1], 0, undefined, true)]))],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([60000], [1], 0, undefined, true)]))],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([64], [1], 0, undefined, true)]))],
      [new UOp(Ops.VIEW, dtypes.void, [], new ShapeTracker([new View([512, 1], [1, 0], 0, undefined, true)]))],
    ],
    apply_swizzle,
    'out(tiny.engine.schedule.apply_swizzle(*data))',
  ),
)
