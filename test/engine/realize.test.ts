import { Kernel } from '../../denograd/codegen/kernel.ts'
import { get_kernel, get_runner } from '../../denograd/engine/realize.ts'
import type { UOp } from '../../denograd/ops.ts'
import { compare } from '../helpers.ts'
import { kernelInputs, pyKernel, tsKernel } from './kernel-inputs.ts'
import { describe } from 'vitest'

describe(
  'realize.get_kernel',
  compare(
    kernelInputs,
    async (renderer, ast) => tsKernel(await get_kernel(renderer, ast)),
    [
      'k = tiny.engine.realize.get_kernel(*data)',
      pyKernel,
    ],
  ),
)
describe(
  'realize.get_optimized_ast',
  compare(
    kernelInputs,
    async (renderer, ast) => (await get_kernel(renderer, ast)).get_optimized_ast(),
    [
      'out(tiny.engine.realize.get_kernel(*data).get_optimized_ast())',
    ],
  ),
)
describe(
  'realize.linearize',
  compare(
    kernelInputs,
    async (renderer, ast) => (await get_kernel(renderer, ast)).linearize().uops,
    [
      'out(tiny.engine.realize.get_kernel(*data).linearize().uops)',
    ],
  ),
)
describe(
  'realize.to_program',
  compare(
    kernelInputs,
    async (renderer, ast) => {
      Kernel.kernel_cnt.clear()
      return (await get_kernel(renderer, ast)).to_program()
    },
    'out(tiny.engine.realize.get_kernel(*data).to_program())',
    {},
  ),
)

describe(
  'realize.get_runner',
  compare(
    () => kernelInputs().map(([r, ast]) => [r.device, ast] as [string, UOp]),
    async (d, ast) => {
      Kernel.kernel_cnt.clear()
      const runner = await get_runner(d, ast)
      return [runner.p]
    },
    [
      'runner = tiny.engine.realize.get_runner(*data)',
      'out([runner.p])',
    ],
    {},
  ),
)
