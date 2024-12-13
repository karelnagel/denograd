import { Kernel, Opt } from '../../src/codegen/kernel.ts'
import { UOp } from '../../src/ops.ts'
import { compare } from '../helpers.ts'
import { Renderer } from '../../src/renderer/index.ts'

Deno.test(
  'Opt.real_axis',
  compare(
    [],
    (opt: Opt, k: Kernel) => opt.real_axis(k),
    'out(data[0].real_axis(data[1]))',
  ),
)

Deno.test(
  'Kernel.init',
  compare(
    [],
    (ast: UOp, opts: Renderer) => new Kernel(ast, opts),
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.membufs',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.float4_axis',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.upcasted_axis',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.first_reduce',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.colors',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.colored_shape',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.reshape_and_permute',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.shift_to',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.simplify_ones',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.simplify_merge_adjacent',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.apply_opt',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.required_optimizations',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.hand_coded_optimizations',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
Deno.test(
  'Kernel.name',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.get_optimized_ast',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.linearize',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'Kernel.to_program',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  '_assert_valid_uop',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)

Deno.test(
  'verify_ast',
  compare(
    [],
    () => {},
    'out(tiny.codegen.kernel.Kernel(*data))',
  ),
)
