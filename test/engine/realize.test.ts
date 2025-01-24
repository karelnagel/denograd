import { Kernel } from '../../denograd/codegen/kernel.ts'
import type { DeviceType } from '../../denograd/device.ts'
import { get_kernel, get_runner } from '../../denograd/engine/realize.ts'
import type { UOp } from '../../denograd/ops.ts'
import { compare } from '../helpers.ts'
import { kernelInputs } from './kernel-inputs.ts'

export const kernelKeys = ['ast', 'opts', 'vars', 'bufs', 'applied_opts', 'group_for_reduces', 'upcasted', 'local_dims', 'tensor_core', 'tensor_core_opts', 'use_tensor_cores', 'dont_use_locals', 'sts', 'reduceops', 'full_buf_index', 'uops'] as const
export const tsKernel = (k: Kernel) => kernelKeys.map((key) => k[key])
export const pyKernel = `out([getattr(k,key,None) for key in [${kernelKeys.map((k) => `"${k}"`)}]])`

Deno.test(
  'realize.get_kernel',
  compare(
    kernelInputs,
    (renderer, ast) => tsKernel(get_kernel(renderer, ast)),
    [
      'k = tiny.engine.realize.get_kernel(*data)',
      pyKernel,
    ],
  ),
)
Deno.test(
  'realize.get_optimized_ast',
  compare(
    kernelInputs,
    (renderer, ast) => get_kernel(renderer, ast).get_optimized_ast(),
    [
      'out(tiny.engine.realize.get_kernel(*data).get_optimized_ast())',
    ],
  ),
)
Deno.test(
  'realize.linearize',
  compare(
    kernelInputs,
    (renderer, ast) => get_kernel(renderer, ast).linearize().uops,
    [
      'out(tiny.engine.realize.get_kernel(*data).linearize().uops)',
    ],
  ),
)
Deno.test(
  'realize.to_program',
  compare(
    kernelInputs,
    (renderer, ast) => {
      Kernel.kernel_cnt = {}
      return get_kernel(renderer, ast).to_program()
    },
    'out(tiny.engine.realize.get_kernel(*data).to_program())',
    {},
  ),
)

Deno.test(
  'realize.get_runner',
  compare(
    () => kernelInputs().map(([r, ast]) => [r.device, ast] as [DeviceType, UOp]),
    (d, ast) => {
      Kernel.kernel_cnt = {}
      const runner = get_runner(d, ast)
      return [runner.p]
    },
    [
      'runner = tiny.engine.realize.get_runner(*data)',
      'out([runner.p])',
    ],
    {},
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
