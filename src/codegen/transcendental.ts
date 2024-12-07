// import math
// from typing import Tuple
// from tinygrad.dtype import dtypes, DType
// from tinygrad.helpers import polyN
// from tinygrad.ops import UOp

// TRANSCENDENTAL_SUPPORTED_DTYPES = (dtypes.float16, dtypes.float32, dtypes.float64)

// def _lazy_map_numbers(x:UOp, inf:UOp, _inf:UOp, nan:UOp, ratio:UOp):
//   """replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio"""
//   return x.ne(math.inf).where(x.ne(x).where(nan, x.ne(-math.inf).where(ratio, _inf)), inf)

// # *** helper functions for bit manipulation ***
// def mantissa_bits(d:DType) -> int: return dtypes.finfo(d)[1]
// def exponent_bias(d:DType) -> int: return {dtypes.float64: 1023, dtypes.float32: 127, dtypes.float16: 15}[d]
// def exponent_mask(d:DType) -> int: return {dtypes.float64: 2047, dtypes.float32: 255, dtypes.float16: 31}[d]

// # **** utils ****
// def shr(x:UOp, y:int) -> UOp: return x // (2**y)
// def shl(x:UOp, y:int) -> UOp: return x * (2**y)

// def rintk(d:UOp) -> UOp:
//   """round d:float to int away from 0"""
//   out_dtype = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype]
//   return (d + (d<0.0).where(d.const_like(-0.5), d.const_like(0.5))).cast(out_dtype)

// def pow2if(q:UOp, float_dtype:DType):
//   """cast(2^q, float_dtype) where q is any integer in the range of [-126, 127]"""
//   out_dtype = {dtypes.int64: dtypes.float64, dtypes.int32: dtypes.float32, dtypes.int16: float_dtype}[q.dtype]
//   return shl(q + exponent_bias(out_dtype), mantissa_bits(out_dtype)).bitcast(out_dtype)

// def ilogb2k(d:UOp) -> UOp:
//   """calculate the integer part of log2(d), where d is normalized fp value in the range of [0, +inf)."""
//   assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
//   dint = d.bitcast({dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype])
//   # -1 <= ilog2bk(d) <= 128
//   return (shr(dint, mantissa_bits(d.dtype)) & exponent_mask(d.dtype)) - exponent_bias(d.dtype)

// def ldexp3k(d:UOp, e:UOp) -> UOp:
//   """d*2^e. e is a number obtained by casting an integer in the range [-127, 127] to a float. d is any float number."""
//   assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES and e.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
//   cast_map = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}
//   m1 = d.bitcast(cast_map[d.dtype])
//   m2 = shl(e.cast(cast_map[d.dtype]), mantissa_bits(d.dtype))
//   return (m1 + m2).bitcast(d.dtype).cast(d.dtype)

// def ldexp2k(d:UOp, e:UOp) -> UOp:
//   """d*2^e. much faster than ldexp3k but risky. d > 0 and d is not denormal."""
//   assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES and e.dtype in (dtypes.int16, dtypes.int32, dtypes.int64)
//   return (d * pow2if(shr(e, 1), d.dtype)) * pow2if(e - shr(e, 1), d.dtype)

// def frexp(v:UOp) -> Tuple[UOp, UOp]:pass

// # *** reduction algorithms for sine ***
// def payne_hanek_reduction(d:UOp) -> Tuple[UOp, UOp]:pass

// def cody_waite_reduction(d:UOp) -> Tuple[UOp, UOp]:pass
// # *** approximate sine on small angle. ***
// def trig_poly(d:UOp, coeff32, coeff64): return d * (polyN(d*d, coeff64) if d.dtype == dtypes.float64 else polyN(d*d, coeff32))
// # approximate sine on [-pi/2, pi/2]
// def sin_poly(d:UOp) -> UOp:
//   return trig_poly(d, [2.6083159809786593541503e-06, -0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938, 1.0],
//                       [-7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10,
//                        -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815,
//                        -0.166666666666666657414808,    1.0])

// def _ifand(q:UOp, n:int): return (q & n).ne(0)

// def sin_poly_small(d:UOp, q:UOp) -> UOp:pass

// def sin_poly_large(d:UOp, q:UOp) -> UOp:pass

// # *** toplevel functions for xsin/xlog2/xexp2 ***

// def xsin(d:UOp, fast:bool=False, switch_over:float=30.0): pass
// def xexp2(d:UOp) -> UOp:
//   """
//   Implements a 1.0 ULP approximation for Ops.EXP2
//   - Paper: https://arxiv.org/pdf/2001.09258
//   """
//   assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
//   # mask +=inf/nan as zero.
//   x = _lazy_map_numbers(d, d.const_like(0.0), d.const_like(0.0), d.const_like(0.0), d)
//   q = rintk(x)
//   # s = d - round(d)
//   s = x - q.cast(x.dtype)
//   # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
//   if d.dtype == dtypes.float64:
//     u = polyN(s, [0.4434359082926529454e-9, 0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4,
//                   0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0,
//                   0.6931471805599452862e+0, 0.1000000000000000000e+1])
//   else: u = polyN(s, [0.1535920892e-3, 0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 1.0])
//   u = ldexp2k(u, q) # u*2^q
//   upper, lower = {dtypes.float64: (1024, -2000), dtypes.float32: (128, -150), dtypes.float16: (23, -22)}[d.dtype]
//   # Replace x >= upper with +inf
//   u = (d >= upper).where(d.const_like(math.inf), u)
//   # Replace x < lower with zero.
//   u = (d<lower).where(d.const_like(0.0), u)
//   # exp2(NaN) = NaN
//   return d.ne(d).where(d.const_like(math.nan), u)

// def xlog2(d:UOp) -> UOp:
//   """
//   Implements a 1.0 ULP approximation for Ops.LOG2
//   Paper: https://arxiv.org/pdf/2001.09258 5.5
//   """
//   assert d.dtype in TRANSCENDENTAL_SUPPORTED_DTYPES
//   # TODO: float16 denormal need float32 to achieve precision
//   if d.dtype == dtypes.float16: return xlog2(d.cast(dtypes.float32)).cast(dtypes.float16)
//   FLT_MIN = d.const_like(1e-6 if d.dtype == dtypes.float16 else 1e-4)
//   is_denormal = d<FLT_MIN
//   a = is_denormal.where(d * (2 ** 64), d)

//   e = ilogb2k(a * (1.0 / 0.75)).cast(a.dtype)
//   m = ldexp3k(a, -e)
//   e = is_denormal.where(e - 64, e)

//   x = (m - 1.0) / (m + 1.0)
//   x2 = x * x
//   if d.dtype == dtypes.float64:
//     t = polyN(x2, [0.2211941750456081490e+0, 0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0,
//                    0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449])
//     s_hi, s_lo = e+x*2.885390081777926774, e.const_like(0)
//   else:
//     t = polyN(x2, [0.4374550283e+0, 0.5764790177e+0, 0.9618012905120])
//     s_hi, s_lo = e+x*2.8853900432586669922, x*3.2734474483568488616e-08
//   r = t * (x * x2) + (s_hi + s_lo)

//   # log2(Inf) = Inf
//   r = d.ne(math.inf).where(r, r.const_like(math.inf))
//   # log2(x) = NaN for x < 0
//   r = (d<-0.0).where(r.const_like(math.nan), r)
//   # log2(0) = -Inf, but we will compare using the value of y because 1e-200==0 is true.
//   # log2_zero = the value of unmasked xlog2(0.0).
//   log2_zero = {dtypes.float64: -1087, dtypes.float32: -191, dtypes.float16: -79}[d.dtype]
//   r = r.ne(log2_zero).where(r, r.const_like(-math.inf))
//   # log2(NaN) = NaN
//   r = d.ne(d).where(r.const_like(math.nan), r)
//   # log2(-0.0) = -Inf. In certain devices like PTX, x == -0.0 won't be true. so making reciprocal.
//   return d.reciprocal().ne(-math.inf).where(r, r.const_like(-math.inf))
