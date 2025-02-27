import type { DeviceType } from '../device.ts'
import { type DType, dtypes, PtrDType } from '../dtype.ts'
import { idiv, is_less_than, strip_parens } from '../helpers.ts'
import { Ops, PatternMatcher, UOp, UPat } from '../ops.ts'
import { base_rewrite, CStyleLanguage, extra_pm, type RenderKernelArgs } from './cstyle.ts'

const sign_extend = (val: UOp, sext_am: number) => {
  return val.rshift(sext_am - 1).gt(0).where(UOp.const(dtypes.uint32, 0xffffffff).lshift(sext_am), UOp.const(dtypes.uint32, 0)).bitwise_or(val.bitcast(dtypes.uint32)).bitcast(dtypes.int)
}
// store for char: buf[idx/4] <- (var << (idx%4)*8))
const packed_store = (bidx: UOp, variable: UOp) => {
  const shift_am: UOp = (bidx.src[1].cast(dtypes.uint32).mod(UOp.const(dtypes.uint32, idiv(4, variable.dtype.itemsize)))).mul(UOp.const(dtypes.uint32, 8 * variable.dtype.itemsize))
  const new_v = (variable.bitwise_and(variable.dtype.itemsize === 1 ? 0xFF : 0xFFFF)).cast(dtypes.uint32).lshift(shift_am)
  const mask = ((shift_am.lshift(variable.dtype.itemsize === 1 ? 0xFF : 0xFFFF, true)).xor(0xFFFFFFFF)).cast(dtypes.uint32)
  const buf = new UOp(Ops.INDEX, bidx.dtype, [bidx.src[0], bidx.src[1].idiv(idiv(4, variable.dtype.itemsize))]).load([], { dtype: dtypes.uint32 })
  return new UOp(Ops.INDEX, bidx.dtype, [bidx.src[0], bidx.src[1].idiv(idiv(4, variable.dtype.itemsize))]).store([(buf.bitwise_and(mask)).bitwise_or(new_v.cast(dtypes.uint32))])
}
// load for char: sign_extend(buf[idx/4] >> ((idx%4)*8))
const packed_load = (root: UOp, bidx: UOp, dtype: DType, variable?: UOp) => {
  const div_idx = bidx.src[1].idiv(idiv(4, dtype.itemsize))
  const shift_am = (bidx.src[1].cast(dtypes.uint32).mod(UOp.const(dtypes.uint32, idiv(4, dtype.itemsize)))).mul(UOp.const(dtypes.uint32, 8 * dtype.itemsize))
  let load
  if (variable !== undefined) load = new UOp(Ops.INDEX, bidx.dtype, [bidx.src[0], div_idx]).load([variable, root.src[2]], { dtype: dtypes.uint32, arg: root.arg })
  else load = new UOp(Ops.INDEX, bidx.dtype, [bidx.src[0], div_idx]).load(root.src.slice(1), { dtype: dtypes.uint32, arg: root.arg })
  const val = (load.cast(dtypes.uint32).rshift(shift_am)).bitwise_and(dtype.itemsize === 1 ? 0xFF : 0xFFFF)
  return [dtypes.char, dtypes.short].includes(dtype) ? sign_extend(val, 8 * dtype.itemsize).cast(dtype) : val.cast(dtype)
}
export const wgsl_matcher = new PatternMatcher([
  new UPat([Ops.CMPLT, Ops.XOR], undefined, [new UPat(undefined, dtypes.bool).named('a'), new UPat().named('b')]).named('c').fn(({ a, b, c }) => a.cast(dtypes.int).alu(c.op, b.cast(dtypes.int)).cast(dtypes.bool)),
  new UPat(Ops.LOAD, undefined, [UPat.var('b')]).named('l').fn(({ l, b }) => l.dtype.itemsize < 4 ? packed_load(l, b, l.dtype) : undefined),
  new UPat(Ops.LOAD, undefined, [UPat.var('b'), UPat.var('c'), new UPat()]).named('l').fn(({ l, b, c }) => l.dtype.itemsize < 4 ? packed_load(l, b, l.dtype, c.cast(dtypes.uint32)) : undefined),
  UPat.var('bidx').store([UPat.var('var')], { allow_any_len: true }).fn((x) => x.var.dtype.itemsize < 4 ? packed_store(x.bidx, x.var) : undefined),
  // TODO: why is this needed, and only for this MUL order
  new UPat(Ops.MUL, undefined, [UPat.var('a'), UPat.var('g').where(UPat.cvar('c1'), UPat.cvar('c2'))]).fn(({ a, g, c1, c2 }) => isNaN(c1.arg) && c2.arg === 1 ? g.where(c1, a) : undefined),
]).add(extra_pm)

const is_storage = (dtype: DType, mutable: boolean) => mutable || (dtype instanceof PtrDType && dtype.size > 512) || dtype.itemsize < 4

export class WGSLRenderer extends CStyleLanguage {
  override device: DeviceType = 'WEBGPU'
  override global_max: [number, number, number] = [65535, 65535, 65535]
  override local_max: [number, number, number] = [256, 256, 64]
  override code_for_workitem: Record<string, (...x: any[]) => string> = { 'g': (x: any) => `i32(gindex.${'xyz'[Number(x)]})`, 'l': (x: any) => `i32(lindex.${'xyz'[Number(x)]})` }
  override extra_matcher = wgsl_matcher
  override supports_float4 = false
  override barrier = 'workgroupBarrier();'
  override code_for_op = new Map<Ops, (...a: (string | DType)[]) => string>([...new CStyleLanguage().code_for_op, [Ops.WHERE, (a, b, c, dtype) => `select(${c},${b},${a})`]])
  override nan = 'nan()'
  override _type_map = new Map([[dtypes.float, 'f32'], [dtypes.uchar, 'u32'], [dtypes.ushort, 'u32'], [dtypes.short, 'i32'], [dtypes.char, 'i32'], [dtypes.int32, 'i32'], [dtypes.uint32, 'u32'], [dtypes.bool, 'bool']])
  override string_rewrite = new PatternMatcher<WGSLRenderer, string | undefined>([
    new UPat(Ops.CONST, dtypes.bool).named('x').fn(({ ctx, x }) => x.arg ? 'true' : 'false'),
    new UPat(Ops.CONST, [dtypes.uchar, dtypes.ushort, dtypes.uint32]).named('x').fn(({ ctx, x }) => x.arg < 0 ? `bitcast<u32>(${x.arg})` : `${BigInt(x.arg) & 0xFFFFFFFFn}u`),
    new UPat(Ops.DEFINE_LOCAL).named('x').fn(({ ctx, x }) => `var<workgroup> ${ctx.get(x)}: array<${ctx.buf_map(x.dtype.base)}, ${x.arg[1]}>;`),
    new UPat(Ops.BITCAST).named('x').fn(({ ctx, x }) => `bitcast<${ctx.get_dtype(x.dtype)}>(${ctx.get(x.src[0])}${['&0xFF', '&0xFFFF', '', ''].at(x.dtype.itemsize - 1)})`),
    UPat.load([UPat.var('b'), UPat.var('v'), UPat.var('g')]).fn(({ ctx, b, v, g }) => `select(${ctx.get(v)}, ${ctx.render_load(ctx.get(b)!, b.src[0].dtype)}, ${ctx.get(g)})`),
    UPat.load([UPat.var('b')], { allow_any_len: true }).fn(({ ctx, b }) => ctx.render_load(ctx.get(b)!, b.src[0].dtype)),
    UPat.index(UPat.var('b'), UPat.var('idx')).fn(({ ctx, b, idx }) => {
      const data = ctx.get(b), index = idx.arg === Ops.ADD ? strip_parens(ctx.get(idx)!) : ctx.get(idx)
      const { dtype, mutable } = ctx.bufs!.get(b)!
      return is_storage(dtype, mutable) ? `${data}[${index}]` : `${data}[${index} / 4][${index} % 4]`
    }),
    // (load & mask) | var -> mask = v.src[0].src[1], var = v.src[1]
    UPat.store([UPat.var('b'), UPat.var('v')], { allow_any_len: true }).fn(({ ctx, b, v }) => b.src[0].dtype.itemsize < 4 ? `atomicAnd(&${ctx.get(b)},${ctx.get(v.src[0].src[1])});\n  atomicAdd(&${ctx.get(b)},${ctx.get(v.src[1])});` : `${ctx.get(b)} = ${ctx.get(v)};`),
    // fix nan check: 'a != a -> is_nan()'
    UPat.var('a').ne(UPat.var('a')).fn(({ ctx, a }) => `is_nan(${ctx.get(a)})`),
  ]).add(base_rewrite)

  override render_cast = (dt: DType, val: string) => `${this.get_dtype(dt)}(${val})`
  override render_dtype = (dt: DType, mutable = true) => 'var'
  render_load = (x: string, dt: DType) => dt.itemsize < 4 ? `atomicLoad(&${x})` : x
  buf_map = (dt: DType) => dt.itemsize < 4 ? 'atomic<u32>' : this.get_dtype(dt.base)!
  override render_kernel = ({ function_name, kernel, uops, prefix }: RenderKernelArgs): string => {
    let local_size = uops.filter((u) => u.op === Ops.SPECIAL && u.arg[0][0] === 'l').map((u) => u.arg).toSorted((a, b) => is_less_than(a[0], b[0]) ? -1 : 1).map(([_, num]) => num)
    if (!local_size.length) local_size = [1]
    const external_local_bufs = kernel.filter((line) => line.includes('var<workgroup>')).map((line) => line.replace('\\s', ''))
    kernel = kernel.filter((line) => !line.includes('var<workgroup>'))
    let prg = 'fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\n'
    // trick to obfuscate compiler so that nan is detected properly
    prg += 'fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }\n@group(0) @binding(0)\nvar<uniform> INFINITY : f32;\n'
    prg += [
      ...external_local_bufs,
      ...[...this.bufs!.entries()].map(([uop, { name, dtype, mutable }], i) =>
        `@group(0) @binding(${i + 1})` +
        `${is_storage(dtype, mutable) ? 'var<storage,read_write>' : 'var<uniform>'}` +
        `${name}:${is_storage(dtype, mutable) ? `array<${this.buf_map(dtype.base)}>` : dtype instanceof PtrDType ? `array<vec4<${this.buf_map(dtype.base)}>,${Math.ceil(dtype.size / 4)}>` : this.buf_map(dtype)};`
      ),
    ].join('\n')
    prg += `\n@compute @workgroup_size(${local_size.join(',')}) fn ${function_name}(@builtin(workgroup_id) gindex: vec3<u32>,`
    prg += '@builtin(local_invocation_id) lindex: vec3<u32>) {\n' + kernel.join('\n') + '\n}'
    return prg
  }
}
