import { Kernel } from '../../denograd/codegen/kernel.ts'
import type { DeviceType } from '../../denograd/device.ts'
import { get_kernel, get_runner } from '../../denograd/engine/realize.ts'
import type { UOp } from '../../denograd/ops.ts'
import { compare, test } from '../helpers.ts'
import { kernelInputs, pyKernel, tsKernel } from './kernel-inputs.ts'

test(
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
test(
  'realize.get_optimized_ast',
  compare(
    kernelInputs,
    async (renderer, ast) => (await get_kernel(renderer, ast)).get_optimized_ast(),
    [
      'out(tiny.engine.realize.get_kernel(*data).get_optimized_ast())',
    ],
  ),
)
test(
  'realize.linearize',
  compare(
    kernelInputs,
    async (renderer, ast) => (await get_kernel(renderer, ast)).linearize().uops,
    [
      'out(tiny.engine.realize.get_kernel(*data).linearize().uops)',
    ],
  ),
)
test(
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

test(
  'realize.get_runner',
  compare(
    () => kernelInputs().map(([r, ast]) => [r.device, ast] as [DeviceType, UOp]),
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

// test(
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

test(
  'BufferCopy.copy',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

test(
  'BufferCopy.call',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

test(
  'ExecItem.run',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

test(
  'lower_schedule_item',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

test(
  'lower_schedule',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)

test(
  'run_schedule',
  compare(
    [],
    () => {},
    'out(XXX)',
  ),
)
