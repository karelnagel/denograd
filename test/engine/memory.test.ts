import { memory_planner } from '../../denograd/engine/memory.ts'
import { compare, test } from '../helpers.ts'

test(
  '_internal_memory_planner',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

test(
  'memory_planner',
  compare(
    [
      // TODO: no way to test this currently cause Buffer isn't serializable
    ],
    memory_planner,
    'out(tiny.engine.memory.memory_planner(*data))',
  ),
)
