import { _lazy_map_numbers, exponent_bias, exponent_mask, ilogb2k, ldexp2k, ldexp3k, pow2if, rintk, shl, shr, sin_poly, xexp2, xlog2 } from '../../denograd/codegen/transcendental.ts'
import { dtypes } from '../../denograd/dtype.ts'
import { UOp } from '../../denograd/ops.ts'
import { compare } from '../helpers.ts'

Deno.test(
  '_lazy_map_numbers',
  compare(
    [
      [UOp.float(3), UOp.float(Infinity), UOp.float(33), UOp.float(3), UOp.float(33)],
      [UOp.float(Infinity), UOp.float(Infinity), UOp.float(33), UOp.float(3), UOp.float(33)],
      [UOp.float(-Infinity), UOp.float(Infinity), UOp.float(33), UOp.float(3), UOp.float(33)],
      [UOp.float(NaN), UOp.float(Infinity), UOp.float(33), UOp.float(3), UOp.float(33)],
      [UOp.float(0), UOp.float(Infinity), UOp.float(33), UOp.float(3), UOp.float(33)],
      [UOp.float(-5.5), UOp.float(Infinity), UOp.float(33), UOp.float(3), UOp.float(33)],
    ],
    _lazy_map_numbers,
    'out(tiny.codegen.transcendental._lazy_map_numbers(*data))',
  ),
)

Deno.test(
  'exponent_bias',
  compare(
    [
      [dtypes.float64],
      [dtypes.float32],
      [dtypes.float16],
    ],
    exponent_bias,
    'out(tiny.codegen.transcendental.exponent_bias(*data))',
  ),
)

Deno.test(
  'exponent_mask',
  compare(
    [
      [dtypes.float64],
      [dtypes.float32],
      [dtypes.float16],
    ],
    exponent_mask,
    'out(tiny.codegen.transcendental.exponent_mask(*data))',
  ),
)

Deno.test(
  'shr',
  compare(
    [
      [UOp.int(16), 2], // 16 >> 2 = 4
      [UOp.int(8), 1], // 8 >> 1 = 4
      [UOp.int(32), 3], // 32 >> 3 = 4
      [UOp.int(64), 4], // 64 >> 4 = 4
    ],
    shr,
    'out(tiny.codegen.transcendental.shr(*data))',
  ),
)

Deno.test(
  'shl',
  compare(
    [
      [UOp.int(1), 2], // 1 << 2 = 4
      [UOp.int(2), 1], // 2 << 1 = 4
      [UOp.int(8), -1], // 8 << -1 = 4
      [UOp.int(4), 0], // 4 << 0 = 4
    ],
    shl,
    'out(tiny.codegen.transcendental.shl(*data))',
  ),
)

Deno.test(
  'rintk',
  compare(
    [
      [UOp.float(1.6)], // rounds to 2
      [UOp.float(-1.6)], // rounds to -2
      [UOp.float(2.1)], // rounds to 2
      [UOp.float(-2.1)], // rounds to -2
      [UOp.float(0.5)], // rounds to 1
      [UOp.float(-0.5)], // rounds to -1
      [UOp.float(0.1)], // rounds to 0
      [UOp.float(-0.1)], // rounds to 0
    ],
    rintk,
    'out(tiny.codegen.transcendental.rintk(*data))',
  ),
)

Deno.test(
  'pow2if',
  compare(
    [
      [UOp.int(0), dtypes.float32], // 2^0 = 1.0
      [UOp.int(1), dtypes.float32], // 2^1 = 2.0
      [UOp.int(-1), dtypes.float32], // 2^-1 = 0.5
      [UOp.int(2), dtypes.float32], // 2^2 = 4.0
      [UOp.int(-2), dtypes.float32], // 2^-2 = 0.25
      [UOp.int(3), dtypes.float32], // 2^3 = 8.0
      [UOp.int(4), dtypes.float32], // 2^4 = 16.0
      [UOp.int(5), dtypes.float32], // 2^5 = 32.0
    ],
    pow2if,
    'out(tiny.codegen.transcendental.pow2if(*data))',
  ),
)
Deno.test(
  'ilogb2k',
  compare(
    [
      [UOp.float(1.0)], // log2(1.0) = 0
      [UOp.float(2.0)], // log2(2.0) = 1
      [UOp.float(0.5)], // log2(0.5) = -1
      [UOp.float(4.0)], // log2(4.0) = 2
      [UOp.float(8.0)], // log2(8.0) = 3
      [UOp.float(16.0)], // log2(16.0) = 4
      [UOp.float(32.0)], // log2(32.0) = 5
      [UOp.float(64.0)], // log2(64.0) = 6
      [UOp.float(0.25)], // log2(0.25) = -2
      [UOp.float(0.125)], // log2(0.125) = -3
    ],
    ilogb2k,
    'out(tiny.codegen.transcendental.ilogb2k(*data))',
  ),
)
Deno.test(
  'ldexp3k',
  compare(
    [
      [UOp.float(1.0), UOp.float(0.0)], // 1.0 * 2^0 = 1.0
      [UOp.float(1.0), UOp.float(1.0)], // 1.0 * 2^1 = 2.0
      [UOp.float(1.0), UOp.float(-1.0)], // 1.0 * 2^-1 = 0.5
      [UOp.float(2.0), UOp.float(1.0)], // 2.0 * 2^1 = 4.0
      [UOp.float(2.0), UOp.float(-1.0)], // 2.0 * 2^-1 = 1.0
      [UOp.float(3.0), UOp.float(2.0)], // 3.0 * 2^2 = 12.0
      [UOp.float(4.0), UOp.float(-2.0)], // 4.0 * 2^-2 = 1.0
      [UOp.float(0.5), UOp.float(3.0)], // 0.5 * 2^3 = 4.0
    ],
    ldexp3k,
    'out(tiny.codegen.transcendental.ldexp3k(*data))',
  ),
)

Deno.test(
  'ldexp2k',
  compare(
    [
      [UOp.float(1.0), UOp.int(0)], // 1.0 * 2^0 = 1.0
      [UOp.float(1.0), UOp.int(1)], // 1.0 * 2^1 = 2.0
      [UOp.float(1.0), UOp.int(-1)], // 1.0 * 2^-1 = 0.5
      [UOp.float(2.0), UOp.int(1)], // 2.0 * 2^1 = 4.0
      [UOp.float(2.0), UOp.int(-1)], // 2.0 * 2^-1 = 1.0
      [UOp.float(3.0), UOp.int(2)], // 3.0 * 2^2 = 12.0
      [UOp.float(4.0), UOp.int(-2)], // 4.0 * 2^-2 = 1.0
      [UOp.float(0.5), UOp.int(3)], // 0.5 * 2^3 = 4.0
    ],
    ldexp2k,
    'out(tiny.codegen.transcendental.ldexp2k(*data))',
  ),
)

Deno.test(
  'sin_poly',
  compare(
    [
      [UOp.float(0.0)], // sin(0) = 0
      [UOp.float(Math.PI / 6)], // sin(π/6) ≈ 0.5
      [UOp.float(Math.PI / 4)], // sin(π/4) ≈ 0.707
      [UOp.float(Math.PI / 3)], // sin(π/3) ≈ 0.866
      [UOp.float(Math.PI / 2)], // sin(π/2) = 1
      [UOp.float(-Math.PI / 6)], // sin(-π/6) ≈ -0.5
      [UOp.float(-Math.PI / 4)], // sin(-π/4) ≈ -0.707
      [UOp.float(-Math.PI / 3)], // sin(-π/3) ≈ -0.866
      [UOp.float(-Math.PI / 2)], // sin(-π/2) = -1
    ],
    sin_poly,
    'out(tiny.codegen.transcendental.sin_poly(*data))',
  ),
)

Deno.test(
  'xexp2',
  compare(
    [
      [{ d: UOp.float(0.0) }], // exp2(0) = 1
      [{ d: UOp.float(1.0) }], // exp2(1) = 2
      [{ d: UOp.float(2.0) }], // exp2(2) = 4
      [{ d: UOp.float(3.0) }], // exp2(3) = 8
      [{ d: UOp.float(-1.0) }], // exp2(-1) = 0.5
      [{ d: UOp.float(-2.0) }], // exp2(-2) = 0.25
      [{ d: UOp.float(0.5) }], // exp2(0.5) ≈ 1.414
      [{ d: UOp.float(-0.5) }], // exp2(-0.5) ≈ 0.707
      [{ d: UOp.float(1024) }], // exp2(1024) = +inf (float64 overflow)
      [{ d: UOp.float(-2000) }], // exp2(-2000) = 0 (float64 underflow)
      [{ d: UOp.float(NaN) }], // exp2(NaN) = NaN
    ],
    xexp2,
    'out(tiny.codegen.transcendental.xexp2(data[0]["d"]))',
  ),
)

Deno.test(
  'xlog2',
  compare(
    [
      [{ d: UOp.float(1.0) }], // log2(1) = 0
      [{ d: UOp.float(2.0) }], // log2(2) = 1
      [{ d: UOp.float(4.0) }], // log2(4) = 2
      [{ d: UOp.float(8.0) }], // log2(8) = 3
      [{ d: UOp.float(0.5) }], // log2(0.5) = -1
      [{ d: UOp.float(0.25) }], // log2(0.25) = -2
      [{ d: UOp.float(0.0) }], // log2(0) = -Inf
      [{ d: UOp.float(-0.0) }], // log2(-0) = -Inf
      [{ d: UOp.float(-1.0) }], // log2(-1) = NaN
      [{ d: UOp.float(Infinity) }], // log2(Inf) = Inf
      [{ d: UOp.float(NaN) }], // log2(NaN) = NaN
      [{ d: UOp.float(1e-200) }], // Test denormal number handling
    ],
    xlog2,
    'out(tiny.codegen.transcendental.xlog2(data[0]["d"]))',
  ),
)
