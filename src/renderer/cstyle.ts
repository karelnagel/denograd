import { type DType, dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { AMX, assert, dedup, getEnv, isNone, isNotNone, prod, stripParens } from '../helpers.ts'
import { GroupOp, Ops, PatternMatcher, UOp, UPat } from '../ops.ts'
import { Renderer, TensorCore } from './index.ts'

export const baseRewrite = new PatternMatcher([
  [new UPat({ op: Ops.DEFINE_ACC, name: 'x' }), (ctx, x) => ctx[x.src[0]]],
  [new UPat({ op: Ops.ASSIGN, name: 'x' }), (ctx, x) => `${ctx[x.src[0]]} = ${ctx[x.src[1]]};`],
  [new UPat({ op: Ops.IF, name: 'x' }), (ctx, x) => `if (${ctx[x.src[0]]}) {{`],
  [new UPat({ op: [Ops.ENDIF, Ops.ENDRANGE] }), (ctx) => '}'],
  [new UPat({ op: Ops.WMMA, name: 'x' }), (ctx, x) => `__${x.arg[0]}(${ctx[x.src[0]]}, ${ctx[x.src[1]]}, ${ctx[x.src[2]]})`],
  // r method accesses
  [new UPat({ op: Ops.RANGE, name: 'x' }), (ctx, x) => `for (${ctx.renderDType(x.dtype)} ${ctx[x]} = ${ctx[x.src[0]]}; ${ctx[x]} < ${ctx[x.src[1]]}; ${ctx[x]}++) {{`],
  [new UPat({ op: Ops.VECTORIZE, name: 'x' }), (ctx, x) => `${ctx.float4.replace('float4', ctx.renderDType(x.dtype))}` + (ctx.device === 'CLANG' ? `{${x.src.map((y: any) => ctx[y]).join(',')}}` : `(${x.src.map((y: any) => ctx[y]).join(',')})`)],
  [new UPat({ op: Ops.CAST, name: 'x' }), (ctx, x) => `(${ctx.renderCast(x.dtype, ctx[x.src[0]])})`],
  [new UPat({ op: Ops.BITCAST, name: 'x' }), (ctx, x) => `(*((${ctx.bufferPrefix}${ctx.renderDType(x.dtype)}*)&${ctx[x.src[0]]}))`],
  [new UPat({ op: Ops.DEFINE_LOCAL, name: 'x' }), (ctx, x) => `${ctx.smemAlign}${ctx.smemPrefix}${ctx.renderDType(x.dtype.base)} ${ctx[x]}[${x.arg[1]}];`],
  [new UPat({ op: Ops.BARRIER }), (ctx) => ctx.barrier],
  [new UPat({ op: Ops.NOOP, name: 'x' }), (ctx, x) => ctx[x.src[0]]],
  [new UPat({ op: Ops.SPECIAL, name: 'x' }), (ctx, x) => `${ctx.codeForWorkitem[x.arg[0][0]](x.arg[0][-1])}; /* ${x.arg[1]} */`],
  // const
  [new UPat({ op: Ops.CONST, arg: Infinity, name: 'x' }), (ctx, x) => `(${ctx.renderCast(x.dtype, ctx.infinity)})`],
  [new UPat({ op: Ops.CONST, arg: -Infinity, name: 'x' }), (ctx, x) => `(${ctx.renderCast(x.dtype, `-${ctx.infinity}`)})`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.floats, name: 'x' }), (ctx, x) => !isFinite(x.arg) ? `(${ctx.renderCast(x.dtype, ctx.nan)})` : null],
  [new UPat({ op: Ops.CONST, dtype: dtypes.float, name: 'x' }), (ctx, x) => `${x.arg}f`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.int64, name: 'x' }), (ctx, x) => `${x.arg}ll`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.uint64, name: 'x' }), (ctx, x) => `${x.arg}ull`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.uint32, name: 'x' }), (ctx, x) => `${x.arg}u`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.bool, name: 'x' }), (ctx, x) => x.arg ? '1' : '0'],
  // consts are rendered to larger type and casted
  [new UPat({ op: Ops.CONST, dtype: [dtypes.bfloat16, dtypes.half], name: 'x' }), (ctx, x) => `(${ctx.renderCast(x.dtype, `${x.arg}f`)})`],
  [new UPat({ op: Ops.CONST, dtype: [dtypes.uint8, dtypes.uint16], name: 'x' }), (ctx, x) => `(${ctx.renderCast(x.dtype, `${x.arg}u`)})`],
  [new UPat({ op: Ops.CONST, dtype: [dtypes.int8, dtypes.int16], name: 'x' }), (ctx, x) => `(${ctx.renderCast(x.dtype, x.arg)})`],
  // default const render
  [new UPat({ op: Ops.CONST, name: 'x' }), (ctx, x) => x.arg.toString()],
  // new load/store
  [new UPat({ op: Ops.INDEX, src: [UPat.var('buf'), UPat.var('idx')] }), (ctx, buf, idx) => `(${ctx[buf]}+${idx.arg === Ops.ADD ? stripParens(ctx[idx]) : ctx[idx]})`],
  [new UPat({ op: Ops.LOAD, src: [UPat.var('bidx'), UPat.var('var'), UPat.var('gate')] }), (ctx, bidx, var1, gate) => `(${ctx[gate]}?*${ctx[bidx]}:${ctx[var1]})`],
  [new UPat({ op: Ops.LOAD, src: [UPat.var('bidx')], allowAnyLen: true }), (ctx, bidx) => `*${ctx[bidx]}`],
  [new UPat({ op: Ops.STORE, src: [UPat.var('bidx'), UPat.var('var')], allowAnyLen: true }), (ctx, bidx, var1) => `*${ctx[bidx]} = ${ctx[var1]};`],
  // alu/gep
  [new UPat({ op: GroupOp.ALU, name: 'x' }), (ctx, x) => ctx.codeForOp[x.op](...x.src.map((v: any) => v.op === x.op && [Ops.ADD, Ops.MUL, Ops.XOR].includes(x.op) ? stripParens(ctx[v]) : ctx[v]), x.dtype)],
  [new UPat({ op: Ops.GEP, name: 'x' }), (ctx, x) => ctx[x.src[0]] + (x.src[0].dtype.count > (['CUDA', 'NV'].includes(ctx.device) ? 8 : 4) || ctx.device == 'CLANG' ? `[${x.arg[0]}]` : `.${'xyzwabcd'[x.arg[0]]}`)],
])

export const extraPm = new PatternMatcher([
  // insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  [new UPat({ op: Ops.BITCAST, name: 'x' }), (x) => x.src[0].op !== Ops.NOOP ? new UOp({ op: Ops.BITCAST, dtype: x.dtype, src: [new UOp({ op: Ops.NOOP, dtype: x.src[0].dtype, src: x.src })] }) : null],
  // gate any stores that aren't gated with ifs
  [
    new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [new UPat({}), new UPat({}), new UPat({ dtype: dtypes.bool })], name: 'store' }),
    (store) => new UOp({ op: Ops.STORE, src: [...store.src.slice(0, -2), new UOp({ op: Ops.IF, src: [store.src[2]] })] }),
  ],
  // rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  [new UPat({ op: Ops.MAX, name: 'm' }), (m) => (m.src[0].lt(m.src[1])).where(m.src[1], m.src[0])],
])

export const uopsToDTypes = (uops: UOp[]): DType[] => dedup(uops.filter((u) => !(u.dtype instanceof ImageDType || u.dtype instanceof PtrDType)).map((u) => u.dtype))

type RenderKernelArgs = { functionName: string; kernel: string[]; bufs: [string, [DType, boolean]][]; uops: UOp[]; prefix?: string[] }
const renderKernel = (self: CStyleLanguage, { bufs, functionName, kernel, uops, prefix }: RenderKernelArgs): string => {
  const tmp = bufs.some(([_, [dtype]]) => dtype instanceof ImageDType) ? 'const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n' : ''
  const buftypes = bufs.map(([name, [dtype, mutable]]) => [name, dtype == dtypes.int ? (dtype instanceof ImageDType || dtype instanceof PtrDType) ? self.renderDType(dtype, mutable) + self.bufferSuffix : self.argIntPrefix : null])

  const prg = [`${self.kernelPrefix}void ${self.getKernelModifier(uops)}${functionName}(`, ...buftypes.map(([name, t]) => `${t} ${name}`), ...self.extra_args.join(', '), ') {\n' + tmp, kernel.join('\n'), '\n}'].join('')
  return isNone(prefix) ? prg : prefix.join('\n') + `\n${prg}`
}

export class CStyleLanguage extends Renderer {
  kernelPrefix = ''
  bufferPrefix = ''
  bufferSuffix = ''
  smemAlign = ''
  smemPrefix = ''
  smemPrefixForCast = true
  argIntPrefix = 'const int'
  barrier = ''
  codeForWorkitem: Record<'g' | 'l' | 'i', (...x: any[]) => string> = {} as any
  extra_args: string[] = []
  float4?: string
  typeMap: Map<DType, string> = new Map()
  infinity = 'INFINITY'
  nan = 'NAN'
  r?: Map<UOp, string>
  static override codeForOp: { [key in Ops]?: (...a: string[]) => string } = {
    [Ops.SQRT]: (x, dtype) => `sqrt(${x})`,
    [Ops.RECIP]: (x, dtype) => `(1/${x})`,
    [Ops.NEG]: (x, dtype) => `-${x}`,
    [Ops.EXP2]: (x, dtype) => `exp2(${x})`,
    [Ops.LOG2]: (x, dtype) => `log2(${x})`,
    [Ops.SIN]: (x, dtype) => `sin(${x})`,
    [Ops.AND]: (a, b, dtype) => `(${a}&${b})`,
    [Ops.XOR]: (a, b, dtype) => `(${a}^${b})`,
    [Ops.OR]: (a, b, dtype) => `(${a}|${b})`,
    [Ops.ADD]: (a, b, dtype) => `(${a}+${b})`,
    [Ops.SUB]: (a, b, dtype) => `(${a}-${b})`,
    [Ops.MUL]: (a, b, dtype) => `(${a}*${b})`,
    [Ops.MOD]: (a, b, dtype) => `(${a}%${b})`,
    [Ops.IDIV]: (a, b, dtype) => `(${a}/${b})`,
    [Ops.CMPNE]: (a, b, dtype) => `(${a}!=${b})`,
    [Ops.SHR]: (a, b, dtype) => `(${a}>>${b})`,
    [Ops.SHL]: (a, b, dtype) => `(${a}<<${b})`,
    [Ops.CMPLT]: (a, b, dtype) => `(${a}<${b})`,
    [Ops.WHERE]: (a, b, c, dtype) => `(${a}?${b}:${c})`,
  }
  stringRewrite = baseRewrite
  override extraMatcher = extraPm

  getKernelModifier = (uops: UOp[]) => ''
  renderKernel = (args: RenderKernelArgs) => renderKernel(this, args)
  renderCast = (dt: DType, val: string): string => `(${this.renderDType(dt)})(${val})`
  renderDType = (dt: DType, mutable = true): string => {
    if (dt instanceof ImageDType) return `${mutable ? 'write_only' : 'read_only'} image2d_t`
    if (dt instanceof PtrDType) return (dt.local && this.smemPrefixForCast ? this.smemPrefix : this.bufferPrefix) + this.renderDType(dt.base) + (dt instanceof PtrDType ? '*' : '')
    const scalar = dt.scalar()
    return (this.typeMap.get(scalar) || scalar.name) + ((dt.count) > 1 ? dt.count.toString() : '')
  }
  // __getitem__ = (key: string) => this.r?.get(key) // hacky helper
  override render = (name: string, uops: UOp[]): string => {
    const r = new Map<UOp, string>()
    this.r = r

    // TODO: it didn't seem to do anything
    // const child_count = Counter(uops.flatMap((ru) => ru.src.map((v) => v)))
    const bufs = new Map<UOp, [string, [DType, boolean]]>()
    const kernel = []
    let depth = 1
    const c: Record<string, number> = {}
    for (const u of uops) {
      if ([Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR].includes(u.op)) {
        r.set(u, u.op === Ops.DEFINE_GLOBAL ? `data${u.arg}` : u.arg[0])
        bufs.set(u, [r.get(u)!, [u.dtype, false]])
        continue
      }

      // // mark buffers that we store to writable
      if (u.op === Ops.STORE) {
        for (const up of u.src[0].sparents().keys()) {
          if (up.op === Ops.DEFINE_GLOBAL) bufs.set(up, [bufs.get(up)![0], [bufs.get(up)![1][0], true]])
        }
      }
      // // naming
      let prefix
      if (u.op === Ops.SPECIAL) r.set(u, u.arg[0])
      else {
        const prefixes = {
          [Ops.RANGE]: 'ridx',
          [Ops.WMMA]: 'wmma',
          [Ops.DEFINE_LOCAL]: 'temp',
          [Ops.CONST]: 'const',
          [Ops.CAST]: 'cast',
          [Ops.BITCAST]: 'cast',
          [Ops.GEP]: 'gep',
          [Ops.VECTORIZE]: 'cast',
          [Ops.NOOP]: 'precast',
          [Ops.INDEX]: 'bidx',
          [Ops.DEFINE_ACC]: 'acc',
          [Ops.LOAD]: 'val',
        }
        prefix = prefixes[u.op as keyof typeof prefixes] || 'alu'
        r.set(u, `${prefix}${c[prefix]}`)
      }
      let l = String(this.stringRewrite.rewrite(u, this))
      assert(isNotNone(l), `failed to render ${u.op} ${u.dtype} ${u.src.map((x) => [x.op, x.dtype])} ${u.arg}`)

      if ([Ops.ENDIF, Ops.ENDRANGE].includes(u.op)) depth -= 1
      if ([Ops.CONST, Ops.GEP, Ops.INDEX].includes(u.op) || ([Ops.VECTORIZE, ...GroupOp.ALU, Ops.CAST, Ops.BITCAST].includes(u.op) && !getEnv('EXPAND_SSA'))) {
        r.set(u, l!)
      } else {
        if ([Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL].includes(u.op) || u.dtype === dtypes.void) {
          if (u.op === Ops.ASSIGN) r.set(u, r.get(u.src[0])!)
        } else {
          l = `${this.renderDType(u.dtype)} ${r.get(u)!} = ${l}` + (u.op !== Ops.SPECIAL ? ';' : '')
        }
        kernel.push('  '.repeat(depth) + l)
        if (prefix) c[prefix] += 1 // if it was used, increment
      }
      if ([Ops.IF, Ops.RANGE].includes(u.op)) depth += 1
    }
    delete this.r

    //  NOTE: this relies on bufs dict preserving order
    return this.renderKernel({ functionName: name, kernel, bufs: bufs.values().toArray(), uops })
  }
}

export class ClangRenderer extends CStyleLanguage {
  override device = 'CLANG'
  override float4 = '(float4)'
  override hasLocal = false
  override globalMax = undefined
  override infinity = '__builtin_inff()'
  override nan = '__builtin_nanf("")'

  // language options
  override bufferSuffix = ' restrict'
  override typeMap = new Map([[dtypes.bool, '_Bool'], [dtypes.half, '__fp16']])
  static override codeForOp = {
    ...Object.fromEntries(Object.entries(CStyleLanguage.codeForOp).filter(([k]) => ![Ops.EXP2, Ops.SIN, Ops.LOG2].includes(k as unknown as Ops))),
    [Ops.SQRT]: (x: any, dtype: any) => dtype === dtypes.float64 ? `__builtin_sqrt(${x})` : `__builtin_sqrtf(${x})`,
  }
  tensorCores = !AMX
    ? null
    : [dtypes.float].map((dt) => [dt, Math.floor(64 / dt.itemsize)] as const).map(([dt, sz]) => new TensorCore({ dims: [sz, sz, 1], threads: [], reduceAxes: [], upcastAxes: [[[1, sz]], [[0, sz]], [[1, sz], [0, sz]]], dtypeIn: dt, dtypeOut: dt }))

  renderVectorPrefix = (dt: DType): string => `typedef ${this.renderDType(dt.scalar())} ${this.renderDType(dt)} __attribute__((aligned(${dt.itemsize}),vector_size(${dt.itemsize})));`

  override renderKernel = ({ bufs, functionName, kernel, uops, prefix }: RenderKernelArgs): string => {
    prefix = uopsToDTypes(uops).filter((dt) => dt.count > 1).map((dt) => this.renderVectorPrefix(dt))
    // https://github.com/corsix/amx
    for (const [name, [N, M, _], dtypeIn] of dedup(uops.filter((uop) => uop.op === Ops.WMMA).map((uop) => uop.arg))) {
      prefix = [
        ...prefix,
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      const out = this.renderDType(dtypeIn.vec(N * N))
      prefix = [
        ...prefix,
        `${out} __$${this.renderDType(dtypeIn.vec(N))} data1, ${
          this.renderDType(dtypeIn.vec(M))
        } data2, ${out} data0){{ AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }} AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull); for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}`,
      ]
    }
    return renderKernel(this, { functionName, kernel, bufs, uops, prefix })
  }
}

export class OpenCLRenderer extends CStyleLanguage {
  //   device = "GPU"

  // language options
  //   kernelPrefix = "__kernel "
  //   bufferPrefix = "__global "
  //   smemAlign = "__attribute__ ((aligned (16))) "
  //   smemPrefix = "__local "
  //   barrier = "barrier(CLK_LOCAL_MEM_FENCE);"
  //   float4 = "(float4)"
  //   codeForWorkitem = {"g": lambda x: f"get_group_id({x})", "l": lambda x: f"get_local_id({x})", "i": lambda x: f"get_global_id({x})"}
  //   typeMap = { dtypes.int8:"char", dtypes.uint8: "uchar", dtypes.uint32: "uint", dtypes.uint16: "ushort", dtypes.uint64: "ulong",
  //               dtypes.bfloat16: "ushort" }

  //   stringRewrite = PatternMatcher([
  //     (UPat(Ops.BITCAST, name="x"), (ctx,x) => f"as_{ctx.renderDType(x.dtype)}({ctx[x.src[0]]})"),
  //     # load/store image (OpenCL)
  //     (UPat(Ops.LOAD, dtype=dtypes.float.vec(4), src=(UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))), UPat.var("var"), UPat.var("gate"))),
  //       lambda ctx,buf,idx,var,gate: f"({ctx[gate]}?read_imagef({ctx[buf]}, smp, {ctx[idx]}):{ctx[var]})"),
  //     (UPat(Ops.LOAD, dtype=dtypes.float.vec(4), src=(UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))),)),
  //       lambda ctx,buf,idx: f"read_imagef({ctx[buf]}, smp, {ctx[idx]})"),
  //     (UPat(Ops.STORE, src=(UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))), UPat.var("var", dtypes.float.vec(4))), allow_any_len=True),
  //       lambda ctx,buf,idx,var: f"write_imagef({ctx[buf]}, {ctx[idx]}, {ctx[var]});"),
  //   ]) + base_rewrite

  //   def renderKernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
  //     if any(uop.dtype == dtypes.half for uop in uops): prefix = (["#pragma OPENCL EXTENSION cl_khr_fp16 : enable"] + (prefix or []))
  //     return super().renderKernel(function_name, kernel, bufs, uops, prefix)
}
export class IntelRenderer extends OpenCLRenderer {
  //   device, suffix, kernelPrefix = "GPU", "INTEL", "__attribute__((intel_reqd_sub_group_size(8)))\n" + "__kernel "
  //   tensorCores = [TensorCore(dims=(8,8,16),threads=[(0,8)],dtype_in=di,dtype_out=do,reduce_axes=[(0,16)],upcast_axes=([(0,16)],[(0,16)],[(1,8)]),
  //     st1_pattern=(((1,0),),((1,2),(1,1),(0,0))),expanded_shape=(8,2,8)) for di,do in [(dtypes.half,dtypes.float),(dtypes.bfloat16,dtypes.float)]]

  //   stringRewrite = PatternMatcher([
  //     (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=(UPat.var('x', dtype=dtypes.float))), (ctx,x) => f"intel_convert_bfloat16_as_ushort({ctx[x[0]]})"),
  //     (UPat(Ops.CAST, dtype=dtypes.float, src=(UPat.var('x', dtype=dtypes.bfloat16))), (ctx,x) => f"intel_convert_as_bfloat16_float({ctx[x[0]]})"),
  //   ]) + OpenCLRenderer.stringRewrite

  //   def renderKernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
  //     prefix = []
  //     for arg in dedup([uop.arg for uop in uops if uop.op is Ops.WMMA]):
  //       dt_in = ("ushort", "bf16") if arg[2] == dtypes.bfloat16 else (arg[2].name, "f16")
  //       prefix.append(f"""{arg[3].name}8 __{arg[0]}({dt_in[0]}16 a, {dt_in[0]}16 b, {arg[3].name}8 c) {{
  //     return intel_sub_group_{dt_in[1]}_{dt_in[1]}_matrix_mad_k16(as_int8(a), as_int8(b), c);\n}}""")
  //     return super().renderKernel(function_name, kernel, bufs, uops, prefix or None)
}

export class MetalRenderer extends CStyleLanguage {
  //   device = "METAL"
  //   sharedMax = 32768
  //   tensorCores = [TensorCore(dims=(8,8,8),threads=[(0,2),(1,4),(0,2),(1,2)],expanded_shape=(2,2,2,2),upcast_axes=([(1,2)],[(1,2)],[(1,2)]),
  //     st1_pattern=(((1,1),(0,1),(1,0),(0,3)),((0,0),(0,2),(1,3),(1,2))),st2_pattern=(((0,0),(1,1),(1,2),(0,2),(1,0)),((0,1),(0,3),(1,3))),
  //     dtype_in=di,dtype_out=do,reduce_axes=[(0,8)]) for di,do in [(dtypes.float,dtypes.float),(dtypes.half,dtypes.float),(dtypes.half,dtypes.half),
  //                                                                 (dtypes.bfloat16,dtypes.float),(dtypes.bfloat16,dtypes.bfloat16)]]
  //   def __init__(self): self.tensorCores = MetalRenderer.tensorCores if hasattr(os, 'uname') and os.uname().machine == "arm64" else []

  // language options
  //   kernelPrefix = "kernel "
  //   bufferPrefix = "device "
  //   smemPrefix = "threadgroup "
  //   arg_int_prefix = "constant int&"
  //   barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);"
  //   float4 = "float4"
  //   codeForWorkitem = {"g": lambda x: f"gid.{chr(120+int(x))}", "l": lambda x: f"lid.{chr(120+int(x))}"}
  // uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  //   extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  //   typeMap = {dtypes.bfloat16: "bfloat"}

  // precise::sin
  //   codeForOp = {**CStyleLanguage.codeForOp, Ops.SIN: lambda x,dtype: f"precise::sin({x})"}

  // upcast to float32 all the ops that don't support bfloat16
  //   extra_matcher = PatternMatcher([
  //     # NOTE: this is copied from PTX
  //     (UPat((Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN), dtype=dtypes.bfloat16, name="x"),
  //       lambda x: (UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16))),
  //   ]) + extra_pm

  //   stringRewrite = PatternMatcher([
  //     (UPat(Ops.BITCAST, name="x"), (ctx,x) => f"as_type<{ctx.renderDType(x.dtype)}>({ctx[x.src[0]]})"),
  //   ]) + base_rewrite

  //   def renderKernel(self, function_name, kernel, bufs, uops, prefix=None):
  //     prefix, wmma_args = ["#include <metal_stdlib>","using namespace metal;"], set([uop.arg for uop in uops if uop.op is Ops.WMMA])
  //     for arg in wmma_args: prefix.append(
  //   f"""{(dtype_out:=self.renderDType(arg[3].vec(2)))} __{arg[0]}({(dtype_in:=self.renderDType(arg[2].vec(2)))} a, {dtype_in} b, {dtype_out} c){{
  //   simdgroup_{self.renderDType(arg[2])}8x8 mat_a, mat_b; simdgroup_{self.renderDType(arg[3])}8x8 mat_c;
  //   mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
  //   mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
  //   simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return {dtype_out}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}}""")
  //     return super().renderKernel(function_name, kernel, bufs, uops, prefix)
}
const _nms = 'xyzwabcdefghijkl'

export class CUDARenderer extends CStyleLanguage {
  //   device = "CUDA"
  //   globalMax = (2147483647, 65535, 65535)
  //   localMax = (1024, 1024, 64)
  //   sharedMax = 49152
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float
  //   tensorCores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(1,2)], dtype_in=di, dtype_out=do, expanded_shape=(2,2,2,2,2,2),
  //     st1_pattern=(((1,1),(1,0),(0,2),(0,3),(0,4)),((1,3),(1,5),(1,2),(0,0),(0,1),(1,4))),
  //     st2_pattern=(((1,1),(1,0),(1,4),(0,0),(0,1)),((0,4),(0,2),(1,5),(0,3),(1,3),(1,2))), reduce_axes=[(0,8),(1,2)],
  //     upcast_axes=([(0,8)],[(2,2),(3,2)],[(3,2),(2,2)])) for di, do in ([(dtypes.half,dtypes.float),(dtypes.bfloat16,dtypes.float)])]
  //   def __init__(self, arch:str): self.tensorCores, self.arch = CUDARenderer.tensorCores if int(arch[3:]) >= 80 else [], arch
  //   def __reduce__(self): return self.__class__, (self.arch,)

  // language options
  //   kernelPrefix = "extern \"C\" __global__ "
  //   smemPrefix = "__shared__ "
  //   smemPrefix_for_cast = False
  //   barrier = "__syncthreads();"
  //   float4 = "make_float4"
  //   codeForWorkitem = {"g": lambda x: f"blockIdx.{chr(120+int(x))}", "l": lambda x: f"threadIdx.{chr(120+int(x))}",
  //                        "i": lambda x: f"(blockIdx.{chr(120+int(x))}*blockDim.{chr(120+int(x))}+threadIdx.{chr(120+int(x))})"}
  //   codeForOp = { **CStyleLanguage.codeForOp,
  //     Ops.SIN: lambda x,dtype: f"hsin({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sin({x})",
  //     Ops.LOG2: lambda x,dtype: f"hlog2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"log2({x})",
  //     Ops.EXP2: lambda x,dtype: f"hexp2({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"exp2({x})",
  //     Ops.SQRT: lambda x,dtype: f"hsqrt({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"sqrt({x})",
  //     Ops.RECIP: lambda x,dtype: f"hrcp({x})" if dtype in (dtypes.half, dtypes.bfloat16) else f"(1/{x})" }
  //   typeMap = {dtypes.bfloat16: "nv_bfloat16"}

  //   def render_vector_prefix(self, dt:DType) -> str:
  //     vec, scal = self.renderDType(dt), self.renderDType(dt.scalar()),
  //     elems, header = ', '.join(_nms[:dt.count]), ', '.join([f"{scal} {x}" for x in _nms[:dt.count]])
  //     return f"struct __align__({dt.itemsize}) {vec} {{ {scal} {elems}; }}; __device__ {vec} make_{vec}({header}) {{ {vec} r={{{elems}}}; return r; }}"

  //   def renderKernel(self, function_name, kernel, bufs, uops, prefix=None):
  //     # TODO: why is dtypes.bfloat16.name == "__bf16"? would be easier not override dtypes.name
  //     prefix = ["#define INFINITY (__int_as_float(0x7f800000))","#define NAN (__int_as_float(0x7fffffff))"]

  //     used_dtypes = uops_to_dtypes(uops)
  //     if any(dt.scalar() == dtypes.half for dt in used_dtypes): prefix.append("#include <cuda_fp16.h>")
  //     if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes): prefix.append("#include <cuda_bf16.h>")
  //     prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count in (4,8) and dt.scalar() in {dtypes.half, dtypes.bfloat16}]

  //     dt_map = { dtypes.half: "f16", dtypes.bfloat16: "bf16" }
  //     for name, (N, M, K), dtype_in, dtype_out, _, _, upcast_axes, _ in dedup([uop.arg for uop in uops if uop.op is Ops.WMMA]):
  //       upcast_sizes = [prod(size for _, size in upcast) for upcast in upcast_axes]
  //       wmma_dtypes = [self.renderDType(dtype.vec(size)) for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)]
  //       n_operands = [size*dtype.itemsize//4 for dtype, size in zip([dtype_in, dtype_in, dtype_out], upcast_sizes)] # 4 => CUDA reg size in bytes
  //       operands = [f"%{i}" for i in range(sum(n_operands))]

  //       # mma operands => {c}, {a}, {b}, {c}
  //       prefix.append(f"""__device__ {wmma_dtypes[2]} __{name}({wmma_dtypes[0]} a, {wmma_dtypes[1]} b, {wmma_dtypes[2]} c){{
  //   int *a_pk = (int *)(&a), *b_pk = (int *)(&b);\n  asm("mma.sync.aligned.m{M}n{N}k{K}.row.col.f32.{dt_map[dtype_in]}.{dt_map[dtype_in]}.f32"
  //       "{{{", ".join(operands[:n_operands[2]])}}}, {{{", ".join(operands[n_operands[2]:n_operands[2]+n_operands[0]])}}},"
  //       "{{{", ".join(operands[-n_operands[1]:])}}}, {{{", ".join(operands[:n_operands[2]])}}};"
  //     : {", ".join([f'"+f"(c.{_nms[i]})' for i in range(n_operands[2])])}
  //     : {", ".join([f'"r"(a_pk[{i}])' for i in range(n_operands[0])])}, {", ".join([f'"r"(b_pk[{i}])' for i in range(n_operands[1])])});
  //   return c;\n}}""")

  // return super().renderKernel(function_name, kernel, bufs, uops, prefix=prefix)
}

export const getKernelModifier = (uops: UOp[]): string => {
  const maxThreadsPerBlock = prod(uops.filter((u) => u.op === Ops.SPECIAL && u.arg[0][0] === 'l').map((u) => u.arg[1]))
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
  return `__launch_bounds__(${maxThreadsPerBlock}) `
}

export class AMDRenderer extends CStyleLanguage {
  //   device = "AMD"
  //   sharedMax = 65536
  // https://gpuopen.com/learn/wmma_on_rdna3/
  //   tensorCores = [TensorCore(dims=(16,16,16), threads=[(0,8),(0,2),(1,2)], dtype_in=di, dtype_out=do, reduce_axes=[(0,16)], opts_seq=("LC","UP"),
  //     upcast_axes = ([(0,16)],[(0,16)],[(1,8)]), st1_pattern=(((1,2),(0,2),(1,1),(0,1)),((1,0),(0,0))), expanded_shape=(16,2,4))
  //     for (di, do) in [(dtypes.half, dtypes.float), (dtypes.half, dtypes.half)]]

  // language options
  //   ockl = [(f"__ockl_get_{name}", "unsigned int", "size_t", "const") for name in ["local_id", "group_id", "local_size"]]
  //   ocml = [(f"__ocml_{name}_f{n}", f"{dt}, {dt}" if "fmax" == name else dt, dt, atr)
  //             for dt, n in [(dtype.name, dtype.itemsize * 8) for dtype in [dtypes.float, dtypes.double, dtypes.half]]
  //             for name, atr in [("fmax", "const"), ("exp2", "pure"), ("log2", "pure"), ("sqrt", "const"), ("sin", "")]]

  //   kernelPrefix = "\n".join(f'extern "C" __attribute__((device{f", {atr}" if atr else ""})) {dto} {meth}({dti});' for meth,dti,dto,atr in ockl+ocml)
  //   kernelPrefix += '\nextern "C" __attribute__((global))'
  //   codeForWorkitem = {"g": lambda x: f"__ockl_get_group_id({x})", "l": lambda x: f"__ockl_get_local_id({x})",
  //                        "i": lambda x: f"(__ockl_get_group_id({x})*__ockl_get_local_size({x})+__ockl_get_local_id({x}))"}
  //   codeForOp = { **CStyleLanguage.codeForOp,
  //     Ops.SIN: lambda x,dtype: f"__ocml_sin_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
  //     Ops.LOG2: lambda x,dtype: f"__ocml_log2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
  //     Ops.EXP2: lambda x,dtype: f"__ocml_exp2_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})",
  //     Ops.SQRT: lambda x,dtype: f"__ocml_sqrt_f{ {dtypes.half:16, dtypes.double:64}.get(dtype, 32)}({x})" }
  //   smemPrefix = "__attribute__((shared))"
  //   barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + \
  //             '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  //   float4 = "make_float4"
  //   typeMap = {dtypes.bfloat16: "hip_bfloat16"}
  //   extra_matcher = PatternMatcher([
  //     # cast bfloat16 alus to float
  //     (UPat(Ops.WHERE, src=(UPat.var("b"), UPat.var("x", dtype=dtypes.bfloat16), UPat.var("y", dtype=dtypes.bfloat16))),
  //       lambda b,x,y: UOp(Ops.WHERE, dtype=dtypes.float, src=(b,x.cast(dtypes.float),y.cast(dtypes.float))).cast(dtypes.bfloat16)),
  //     (UPat(GroupOp.ALU, dtype=dtypes.bfloat16, name="x"),
  //       lambda x: UOp(x.op, dtypes.float, tuple(vv.cast(dtypes.float) for vv in x.src), x.arg).cast(dtypes.bfloat16)),
  //     (UPat(GroupOp.ALU, dtypes.bool, name="alu", src=(UPat.var("x", dtype=dtypes.bfloat16), UPat.var("y", dtype=dtypes.bfloat16))),
  //       lambda alu,x,y: UOp(alu.op, dtypes.bool, (x.cast(dtypes.float), y.cast(dtypes.float)), alu.arg)),
  //     # add float intermediate casting for bfloat16
  //     (UPat(Ops.CAST, name="x", src=UPat.var("y", dtypes.bfloat16)),lambda x,y: y.cast(dtypes.float).cast(x.dtype) if x.dtype!=dtypes.float else None),
  //     (UPat(Ops.CAST, dtypes.bfloat16, UPat.var("x")),lambda x: x.cast(dtypes.float).cast(dtypes.bfloat16) if x.dtype!=dtypes.float else None),
  //     # bfloat16 casting
  //     (UPat.cvar('x', dtypes.bfloat16), lambda x: cast_float_to_bf16(UOp.const(dtypes.float, x.arg))),
  //     (UPat(Ops.CAST, dtype=dtypes.float, src=UPat.var("x", dtype=dtypes.bfloat16)),
  //       lambda x: (x.bitcast(dtypes.ushort).cast(dtypes.uint)<<16).bitcast(dtypes.float)),
  //     (UPat(Ops.CAST, dtype=dtypes.bfloat16, src=UPat.var("x", dtype=dtypes.float)), cast_float_to_bf16)]) + extra_pm

  //   def render_vector_prefix(self, dtype:DType) -> str:
  //     vec, scal = self.renderDType(dtype), self.renderDType(dtype.scalar())
  //     return f"typedef {scal} {vec} __attribute__((ext_vector_type({dtype.count})));\nstatic inline __attribute__((device)) "+ \
  //            f"{vec} make_{vec}({', '.join([f'{scal} {x}' for x in _nms[:dtype.count]])}) {{ return {{ {', '.join(_nms[:dtype.count])} }}; }}"

  //   def renderKernel(self, function_name, kernel, bufs, uops, prefix=None) -> str:
  //     prefix = ["#define INFINITY (__builtin_inff())","#define NAN (__builtin_nanf(\"\"))","typedef long unsigned int size_t;","#define half _Float16"]

  //     used_dtypes = uops_to_dtypes(uops)
  //     if any(dt.scalar() == dtypes.bfloat16 for dt in used_dtypes): prefix.append("typedef unsigned short hip_bfloat16;")
  //     prefix += [self.render_vector_prefix(dt) for dt in used_dtypes if dt.count > 1]

  //     for arg in dedup([uop.arg for uop in uops if uop.op is Ops.WMMA]): # TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
  //       if arg[3] == dtypes.float: prefix.append(f"#define __{arg[0]} __builtin_amdgcn_wmma_f32_16x16x16_f16_w32")
  //       else: prefix.append(f"static inline __attribute__((device)) half8 __{arg[0]}"+"""(half16 a, half16 b, half8 c) {
  //   half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
  //   c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
  //   for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}""")
  //     return super().renderKernel(function_name, kernel, bufs, uops, prefix)

  //   def get_kernel_modifier(self, uops:List[UOp]) -> str:
  //     requiredMaxThreadsPerBlock = prod(u.arg[1] for u in uops if u.op is Ops.SPECIAL and u.arg[0][0] == "l")
  //     # https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
  //     # NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
  //     return f"__attribute__((amdgpu_flat_work_group_size(1, {requiredMaxThreadsPerBlock})))"
}
export class NVRenderer extends CUDARenderer {
  override device = 'NV'
}
export class HIPRenderer extends AMDRenderer {
  override device = 'HIP'
}
export class QCOMRenderer extends OpenCLRenderer {
  override device = 'QCOM'
}
