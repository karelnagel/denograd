import { DType, dtypes } from '../dtype.ts'
import { assert, range } from '../helpers.ts'
import { polyN, UOp } from '../ops.ts'

export const TRANSCENDENTAL_SUPPORTED_DTYPES = [dtypes.float16, dtypes.float32, dtypes.float64]

/**replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio*/
export const _lazy_map_numbers = (x: UOp, inf: UOp, _inf: UOp, nan: UOp, ratio: UOp) => x.ne(Infinity).where(x.ne(x).where(nan, x.ne(-Infinity).where(ratio, _inf)), inf)

// *** helper functions for bit manipulation ***
export const mantissa_bits = (d: DType): number => dtypes.finfo(d)[1]
export const exponent_bias = (d: DType): number => new Map([[dtypes.float64, 1023], [dtypes.float32, 127], [dtypes.float16, 15]] as const).get(d)!
export const exponent_mask = (d: DType): number => new Map([[dtypes.float64, 2047], [dtypes.float32, 255], [dtypes.float16, 31]] as const).get(d)!

// **** utils ****
export const shr = (x: UOp, y: number): UOp => x.idiv(2 ** y)
export const shl = (x: UOp, y: number): UOp => x.mul(2 ** y)

/**round d:float to int away from 0*/
export const rintk = (d: UOp): UOp => {
  const out_dtype = new Map([[dtypes.float64, dtypes.int64], [dtypes.float32, dtypes.int32], [dtypes.float16, dtypes.int16]]).get(d.dtype)!
  return (d.add(d.lt(0.0).where(d.const_like(-0.5), d.const_like(0.5)))).cast(out_dtype)
}

/**cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]*/
export const pow2if = (q: UOp, float_dtype: DType) => {
  const out_dtype = new Map([[dtypes.int64, dtypes.float64], [dtypes.int32, dtypes.float32], [dtypes.int16, float_dtype]]).get(q.dtype)!
  return shl(q.add(exponent_bias(out_dtype)), mantissa_bits(out_dtype)).bitcast(out_dtype)
}
/**calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf).*/
export const ilogb2k = (d: UOp): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  const dint = d.bitcast(new Map([[dtypes.float64, dtypes.int64], [dtypes.float32, dtypes.int32], [dtypes.float16, dtypes.int16]]).get(d.dtype)!)
  // -1 <= ilog2bk(d) <= 128
  return (shr(dint, mantissa_bits(d.dtype)).bitwise_and(exponent_mask(d.dtype))).sub(exponent_bias(d.dtype))
}
/**d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number.*/
export const ldexp3k = (d: UOp, e: UOp): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype) && TRANSCENDENTAL_SUPPORTED_DTYPES.includes(e.dtype))
  const cast_map = new Map([[dtypes.float64, dtypes.int64], [dtypes.float32, dtypes.int32], [dtypes.float16, dtypes.int16]])
  const m1 = d.bitcast(cast_map.get(d.dtype)!)
  const m2 = shl(e.cast(cast_map.get(d.dtype)!), mantissa_bits(d.dtype))
  return (m1.add(m2)).bitcast(d.dtype).cast(d.dtype)
}
/**d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal.*/
export const ldexp2k = (d: UOp, e: UOp): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype) && [dtypes.int16, dtypes.int32, dtypes.int64].includes(e.dtype))
  return (d.mul(pow2if(shr(e, 1), d.dtype))).mul(pow2if(e.sub(shr(e, 1)), d.dtype))
}
/** frexp(v) -> (mantissa, exponent) assuming v != 0 */
export const frexp = (v: UOp): [UOp, UOp] => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(v.dtype))
  // m1 = masks for mantissa, m2 = masks to normalize the mantissa.
  const m1 = new Map([[dtypes.float64, 0x000FFFFFFFFFFFFF], [dtypes.float32, 0x807FFFFF], [dtypes.float16, 0x83FF]]).get(v.dtype)!
  const m2 = new Map([[dtypes.float64, 0x3FE0000000000000], [dtypes.float32, 0x3F000000], [dtypes.float16, 0x3800]]).get(v.dtype)!
  const bits = v.bitcast(new Map([[dtypes.float64, dtypes.uint64], [dtypes.float32, dtypes.uint32], [dtypes.float16, dtypes.uint16]]).get(v.dtype)!)
  const exponent = shr(bits, mantissa_bits(v.dtype)).bitwise_and(exponent_mask(v.dtype))
  // Set the exponent bits appropriately to normalize the mantissa into the range of [0.5, 1.0).
  const mantissa = ((bits.bitwise_and(m1)).bitwise_or(m2)).bitcast(v.dtype)
  const exp = exponent.sub(exponent_bias(v.dtype)).add(1)
  return [mantissa, exp]
}

// *** reduction algorithms for sine ***
/**
 * Performs Payne-Hanek Reduction: computes the remainder of `d` modulo pi/2 for the values `d` where 39800.0 <= d <= +Inf
 * Returns a tuple of `(r, q)`:
 * - `r`[d.dtype] is the reminder value corresponding to `round_to_nearest(x % pi/2)`.
 * - `q`[int32] is an integer, and q % 4 is corresponding to the quadrant of the original angle `d`.
 */
export const payne_hanek_reduction = (d: UOp): [UOp, UOp] => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  // https://stackoverflow.com/questions/30463616/payne-hanek-algorithm-implementation-in-c/30465751#30465751
  // 190 bits of 2/pi for Payne-Hanek style argument reduction
  const two_over_pi_f = [0x00000000, 0x28be60db, 0x9391054a, 0x7f09d5f4, 0x7d4d3770, 0x36d8a566, 0x4f10e410]

  const intermediate_dtype = d.dtype === dtypes.float16 ? dtypes.float32 : d.dtype

  let [f, e] = frexp(d)
  const ia = (f.cast(intermediate_dtype).mul(4.294967296e9)).cast(dtypes.uint64)
  // extract 96 relevant bits of 2/pi based on magnitude of argument
  const i = shr(e.cast(dtypes.uint64), 5)
  e = e.cast(dtypes.int32).bitwise_and(31)
  const offset = e.sub(32, true)

  /** an = two_over_pi_f[i+offset] */
  const _take = (an: UOp, offset: number, count = 0): UOp => {
    if (count + offset < two_over_pi_f.length - 1) {
      an = i.ne(count).where(_take(an, offset, count + 1), an.const_like(two_over_pi_f[count + offset]))
    }
    return an
  }
  const _shl_lazy = (x: UOp, y: UOp) => (x.cast(dtypes.uint64).mul(pow2if(y, d.dtype).cast(dtypes.uint64))).cast(dtypes.uint32)
  const _shr_lazy = (x: UOp, y: UOp) => (x.cast(dtypes.uint64).idiv(pow2if(y, d.dtype).cast(dtypes.uint64))).cast(dtypes.uint32)

  const a = range(4).map((i) => _take(UOp.const(dtypes.uint32, 0), i))
  //  (two_over_pi_f[Int(i) + n] << e) | (two_over_pi_f[Int(i) + n+1] >> (nbits - e))
  // Note: e >= 1 for all numbers d >= 1.0. assume e != 0
  const hi = _shl_lazy(a[0], e).bitwise_or(_shr_lazy(a[1], offset))
  const mi = _shl_lazy(a[1], e).bitwise_or(_shr_lazy(a[2], offset))
  const lo = _shl_lazy(a[2], e).bitwise_or(_shr_lazy(a[3], offset))

  const _hp_mul = (x: UOp, y: UOp) => x.cast(dtypes.uint64).mul(y.cast(dtypes.uint64))
  // compute x * 2/pi
  let p = shl(_hp_mul(ia, hi), 32).add(_hp_mul(ia, mi)).add(shr(_hp_mul(ia, lo), 32))

  // round quotient to nearest
  const q = shr(p, 62).cast(dtypes.int32)
  p = p.bitwise_and(0x3fffffffffffffffn)
  const r = (p.cast(intermediate_dtype).mul(3.4061215800865545e-19)).cast(d.dtype)

  // if fraction >= 0.5, r -= pi/2, q += 1
  return [(f.lt(0.5)).where(r, r.sub(Math.PI / 2)), (f.lt(0.5)).where(q, q.add(1))]
}

/**
 * Performs Cody-Waite Reduction: computes the reminder of `d` modulo pi/2 for the values `d` where 0 <= abs(d) <= 39800.0
 * Returns a tuple of `(r, q)`, where the output format is the same as that of `payne_hanek_reduction`.
 */
export const cody_waite_reduction = (d: UOp): [UOp, UOp] => {
  const m_1_pi = 0.318309886183790671537767526745028724
  const qdh = (d.mul(m_1_pi / 2.0 ** 24)).cast(dtypes.int64).cast(d.dtype).mul(2.0 ** 24)
  const _reduce_d = (x: UOp, q: UOp) => {
    // https://github.com/shibatch/sleef/blob/4e08851f59fc2b545f9c393c6a23dfd311a26308/src/libm/sleefdp.c#L789-L823
    if (x.dtype === dtypes.float64) {
      // https://github.com/shibatch/sleef/blob/f6d8a841fbfddd26ce712834d4da220cd76048fb/src/common/misc.h#L77
      const [PI_A, PI_B, PI_C, PI_D] = [3.1415926218032836914, 3.1786509424591713469e-08, 1.2246467864107188502e-16, 1.2736634327021899816e-24]
      d = qdh.sub(PI_A).add(x)
      d = q.mul(-PI_A).add(d)
      d = qdh.mul(-PI_B).add(d)
      d = q.mul(-PI_B).add(d)
      d = qdh.mul(-PI_C).add(d)
      d = q.mul(-PI_C).add(d)
      d = (qdh.add(q)).mul(-PI_D).add(d)
    } else if (x.dtype === dtypes.float16) {
      // [FIXME] when reducing `d`, FP16 needs FP32 precision to achieve 1.0 ULP precision.
      d = _reduce_d(x.cast(dtypes.float32), q.cast(dtypes.float32)).cast(dtypes.float16)
    } else {
      // https://github.com/shibatch/sleef/blob/4e08851f59fc2b545f9c393c6a23dfd311a26308/src/libm/sleefsp.c#L464-L503
      d = q.mul(-3.1414794921875).add(x)
      d = q.mul(-0.00011315941810607910156).add(d)
      d = q.mul(-1.9841872589410058936e-09).add(d)
      d = q.mul(-1.2154201256553420762e-10).add(d)
    }
    return d
  }
  const quadrant = d.dtype === dtypes.float64 ? rintk(d.mul(m_1_pi).sub(qdh)) : rintk(d.mul(m_1_pi))
  return [_reduce_d(d, quadrant.cast(d.dtype)), quadrant.cast(dtypes.int32)]
}
// *** approximate sine on small angle. ***
export const trig_poly = (d: UOp, coeff32: number[], coeff64: number[]) => d.mul(d.dtype === dtypes.float64 ? polyN(d.mul(d), coeff64) : polyN(d.mul(d), coeff32))
// approximate sine on [-pi/2, pi/2]
// deno-fmt-ignore
export const sin_poly = (d: UOp): UOp => {
  return trig_poly(d,
    [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0],
    [-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10, -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815, -0.166666666666666657414808, 1.0],
  )
}

const _ifand = (q: UOp, n: number) => (q.bitwise_and(n)).ne(0)

export const sin_poly_small = (d: UOp, q: UOp): UOp => {
  const r = sin_poly(d)
  return r.mul(_ifand(q, 1).where(r.const_like(-1), r.const_like(1)))
}

export const sin_poly_large = (d: UOp, q: UOp): UOp => {
  const r = sin_poly(d.add(_ifand(q, 1).where(d.const_like(Math.PI / 2), d.const_like(0))))
  return r.mul(_ifand(q, 2).where(r.const_like(-1), r.const_like(1)))
}

// *** toplevel functions for xsin/xlog2/xexp2 ***
/**
 * Implements a 1.0 ULP approximation for Ops.SIN.
 * - fast=True assumes x <= switch_over.
 * - switch_over is the threshold for switching to payne_hanek_reduction.
 */
export const xsin = ({ d, fast = false, switch_over = 30.0 }: { d: UOp; fast?: boolean; switch_over?: number }) => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  //  mask +-inf/nan as zero
  const x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  //  x_sign = sign(x)
  const x_sign = x.ne(0).where((x.lt(0)).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  const x_abs = x.mul(x_sign)
  const [r, q] = (fast ? cody_waite_reduction : payne_hanek_reduction)(x_abs)
  let result
  if (fast) result = sin_poly_small(r, q)
  else {
    // Payne Hanek Reduction assumes abs(x) >= pi/4, so for smaller values, use cody_waite_reduction.
    const [r_small, q_small] = cody_waite_reduction(x_abs)
    result = (x_abs.lt(switch_over)).where(sin_poly_small(r_small, q_small), sin_poly_large(r, q))
  }
  // adjusts the sign for abs(x)
  result = result.mul(x_sign)
  // sin(Inf) = NaN, sin(-Inf) = NaN, sin(NaN) = NaN
  return _lazy_map_numbers(d, d.const_like(NaN), d.const_like(NaN), d.const_like(NaN), result)
}

/**
 * Implements a 1.0 ULP approximation for Ops.EXP2
 * Paper: https://arxiv.org/pdf/2001.09258
 */
export const xexp2 = ({ d }: { d: UOp }): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  //   # mask +=inf/nan as zero.
  const x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
  const q = rintk(x)
  // # s = d - round(d)
  const s = x.sub(q.cast(x.dtype))
  //   # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  let u
  if (d.dtype === dtypes.float64) {
    // deno-fmt-ignore
    u = polyN(s, [0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4,
                    0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0,
                    0.6931471805599452862e+0, 0.1000000000000000000e+1])
  } else u = polyN(s, [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0])
  u = ldexp2k(u, q) // u*2^q
  const [upper, lower] = new Map([[dtypes.float64, [1024, -2000]], [dtypes.float32, [128, -150]], [dtypes.float16, [23, -22]]]).get(d.dtype)!
  //   # Replace x >= upper with +inf
  u = (d.ge(upper)).where(d.const_like(Infinity), u)
  //   # Replace x < lower with zero.
  u = (d.lt(lower)).where(d.const_like(0.0), u)
  //   # exp2(NaN) = NaN
  return d.ne(d).where(d.const_like(NaN), u)
}

/**
 * Implements a 1.0 ULP approximation for Ops.LOG2
 * Paper: https://arxiv.org/pdf/2001.09258 5.5
 */
export const xlog2 = ({ d }: { d: UOp }): UOp => {
  assert(TRANSCENDENTAL_SUPPORTED_DTYPES.includes(d.dtype))
  //   # TODO: float16 denormal need float32 to achieve precision
  if (d.dtype === dtypes.float16) return xlog2({ d: d.cast(dtypes.float32) }).cast(dtypes.float16)
  const FLT_MIN = d.const_like(d.dtype === dtypes.float16 ? 1e-6 : 1e-4)
  const is_denormal = d.lt(FLT_MIN)
  const a = is_denormal.where(d.mul(2 ** 64), d)

  let e = ilogb2k(a.mul(1.0 / 0.75)).cast(a.dtype)
  const m = ldexp3k(a, e.neg())
  e = is_denormal.where(e.sub(64), e)

  const x = (m.sub(1.0)).div(m.add(1.0))
  const x2 = x.mul(x)
  let t, s_hi, s_lo
  if (d.dtype === dtypes.float64) {
    // deno-fmt-ignore
    t = polyN(x2, [0.2211941750456081490e+0, 0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0,
                       0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449]);
    ;[s_hi, s_lo] = [e.add(x.mul(2.885390081777926774)), e.const_like(0)]
  } else {
    t = polyN(x2, [0.4374550283e+0, 0.5764790177e+0, 0.9618012905120])
    ;[s_hi, s_lo] = [e.add(x.mul(2.8853900432586669922)), x.mul(3.2734474483568488616e-08)]
  }
  let r = t.mul(x.mul(x2)).add(s_hi.add(s_lo))

  //   # log2(Inf) = Inf
  r = d.ne(Infinity).where(r, r.const_like(Infinity))
  //   # log2(x) = NaN for x < 0
  r = (d.lt(-0.0)).where(r.const_like(NaN), r)
  //   # log2(0) = -Inf, but we will compare using the value of y because 1e-200==0 is true.
  //   # log2_zero = the value of unmasked xlog2(0.0).
  const log2_zero = new Map([[dtypes.float64, -1087], [dtypes.float32, -191], [dtypes.float16, -79]]).get(d.dtype)!
  r = r.ne(log2_zero).where(r, r.const_like(-Infinity))
  //   # log2(NaN) = NaN
  r = d.ne(d).where(r.const_like(NaN), r)
  //   # log2(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
  return d.reciprocal().ne(-Infinity).where(r, r.const_like(-Infinity))
}
