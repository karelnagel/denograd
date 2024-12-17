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
    [],
    () => {},
    'out(XXX)',
  ),
)
