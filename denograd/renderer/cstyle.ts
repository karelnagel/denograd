import { type DType, dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { env } from '../env/index.ts'
import { dedup, DefaultMap, floatString, idiv, prod, range, set_default, strip_parens, sum, zip } from '../helpers.ts'
import { GroupOp, Ops, PatternMatcher, UOp, UPat } from '../ops.ts'
import { Renderer, TensorCore } from './index.ts'

export const base_rewrite = new PatternMatcher<CStyleLanguage, string | undefined>([
  new UPat(Ops.DEFINE_ACC).named('x').fn(({ ctx, x }) => ctx.get(x.src[0])),
  new UPat(Ops.ASSIGN).named('x').fn(({ ctx, x }) => `${ctx.get(x.src[0])} = ${ctx.get(x.src[1])};`),
  new UPat(Ops.IF).named('x').fn(({ ctx, x }) => `if (${ctx.get(x.src[0])}) {`),
  new UPat([Ops.ENDIF, Ops.ENDRANGE]).fn(({ ctx }) => '}'),
  new UPat(Ops.WMMA).named('x').fn(({ ctx, x }) => `__${x.arg[0]}(${ctx.get(x.src[0])}, ${ctx.get(x.src[1])}, ${ctx.get(x.src[2])})`),
  // r method accesses
  new UPat(Ops.RANGE).named('x').fn(({ ctx, x }) => `for (${ctx.render_dtype(x.dtype)} ${ctx.get(x)} = ${ctx.get(x.src[0])}; ${ctx.get(x)} < ${ctx.get(x.src[1])}; ${ctx.get(x)}++) {`),
  [
    new UPat(Ops.VECTORIZE).named('x'),
    ({ ctx, x }) => `${ctx.float4!.replace('float4', ctx.render_dtype(x.dtype))}` + (ctx.device === 'CLANG' ? `{${x.src.map((y: any) => ctx.get(y)).join(',')}}` : `(${x.src.map((y: any) => ctx.get(y)).join(',')})`),
  ],
  new UPat(Ops.CAST).named('x').fn(({ ctx, x }) => `(${ctx.render_cast(x.dtype, ctx.get(x.src[0])!)})`),
  new UPat(Ops.BITCAST).named('x').fn(({ ctx, x }) => `(*((${ctx.buffer_prefix}${ctx.render_dtype(x.dtype)}*)&${ctx.get(x.src[0])}))`),
  new UPat(Ops.DEFINE_LOCAL).named('x').fn(({ ctx, x }) => `${ctx.smem_align}${ctx.smem_prefix}${ctx.render_dtype(x.dtype.base)} ${ctx.get(x)}[${x.arg[1]}];`),
  new UPat(Ops.BARRIER).fn(({ ctx }) => ctx.barrier),
  new UPat(Ops.NOOP).named('x').fn(({ ctx, x }) => ctx.get(x.src[0])),
  new UPat(Ops.SPECIAL).named('x').fn(({ ctx, x }) => `${ctx.code_for_workitem[x.arg[0][0] as keyof typeof ctx.code_for_workitem]?.(x.arg[0].at(-1)!)}; /* ${x.arg[1]} */`),
  // const
  new UPat(Ops.CONST, undefined, undefined, Infinity, 'x').fn(({ ctx, x }) => `(${ctx.render_cast(x.dtype, ctx.infinity)})`),
  new UPat(Ops.CONST, undefined, undefined, -Infinity, 'x').fn(({ ctx, x }) => `(${ctx.render_cast(x.dtype, `-${ctx.infinity}`)})`),
  new UPat(Ops.CONST, dtypes.floats).named('x').fn(({ ctx, x }) => isNaN(x.arg) ? `(${ctx.render_cast(x.dtype, ctx.nan)})` : undefined),
  new UPat(Ops.CONST, dtypes.float).named('x').fn(({ ctx, x }) => `${floatString(x.arg)}f`),
  new UPat(Ops.CONST, dtypes.int64).named('x').fn(({ ctx, x }) => `${x.arg}ll`),
  new UPat(Ops.CONST, dtypes.uint64).named('x').fn(({ ctx, x }) => `${x.arg}ull`),
  new UPat(Ops.CONST, dtypes.uint32).named('x').fn(({ ctx, x }) => `${x.arg}u`),
  new UPat(Ops.CONST, dtypes.bool).named('x').fn(({ ctx, x }) => x.arg ? '1' : '0'),
  // consts are rendered to larger type and casted
  new UPat(Ops.CONST, [dtypes.bfloat16, dtypes.half]).named('x').fn(({ ctx, x }) => `(${ctx.render_cast(x.dtype, `${floatString(x.arg)}f`)})`),
  new UPat(Ops.CONST, [dtypes.uint8, dtypes.uint16]).named('x').fn(({ ctx, x }) => `(${ctx.render_cast(x.dtype, `${x.arg}u`)})`),
  new UPat(Ops.CONST, [dtypes.int8, dtypes.int16]).named('x').fn(({ ctx, x }) => `(${ctx.render_cast(x.dtype, x.arg)})`),
  // default const render
  new UPat(Ops.CONST).named('x').fn(({ ctx, x }) => x.arg.toString()),
  // new load/store
  new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]).fn(({ ctx, buf, idx }) => `(${ctx.get(buf)}+${idx.arg === Ops.ADD ? strip_parens(ctx.get(idx)!) : ctx.get(idx)})`),
  new UPat(Ops.LOAD, undefined, [UPat.var('bidx'), UPat.var('var'), UPat.var('gate')]).fn((x) => `(${x.ctx.get(x.gate)}?*${x.ctx.get(x.bidx)}:${x.ctx.get(x.var)})`),
  new UPat(Ops.LOAD, undefined, [UPat.var('bidx')], undefined, undefined, true).fn(({ ctx, bidx }) => `*${ctx.get(bidx)}`),
  new UPat(Ops.STORE, undefined, [UPat.var('bidx'), UPat.var('var')], undefined, undefined, true).fn((p) => `*${p.ctx.get(p.bidx)} = ${p.ctx.get(p.var)};`),
  // alu/gep
  new UPat(GroupOp.ALU).named('x').fn(({ ctx, x }) => ctx.code_for_op.get(x.op)!(...x.src.map((v) => v.op === x.op && [Ops.ADD, Ops.MUL, Ops.XOR].includes(x.op) ? strip_parens(ctx.get(v)!) : ctx.get(v)!), x.dtype)),
  new UPat(Ops.GEP).named('x').fn(({ ctx, x }) => ctx.get(x.src[0]) + (x.src[0].dtype.count > (['CUDA', 'NV'].includes(ctx.device) ? 8 : 4) || ctx.device === 'CLANG' ? `[${x.arg[0]}]` : `.${'xyzwabcd'[x.arg[0]]}`)),
])

export const extra_pm = new PatternMatcher([
  // insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  new UPat(Ops.BITCAST).named('x').fn(({ x }) => x.src[0].op !== Ops.NOOP ? new UOp(Ops.BITCAST, x.dtype, [new UOp(Ops.NOOP, x.src[0].dtype, x.src)]) : undefined),
  // rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  new UPat(Ops.MAX).named('m').fn(({ m }) => (m.src[0].lt(m.src[1])).where(m.src[1], m.src[0])),
])

export const uops_to_dtypes = (uops: UOp[]): DType[] => dedup(uops.filter((u) => !(u.dtype instanceof ImageDType || u.dtype instanceof PtrDType)).map((u) => u.dtype))

export type Buf = { name: string; dtype: DType; mutable: boolean }
export type RenderKernelArgs = { function_name: string; kernel: string[]; bufs: Map<UOp, Buf>; uops: UOp[]; prefix?: string[] }

export class CStyleLanguage extends Renderer {
  kernel_prefix = ''
  buffer_prefix = ''
  buffer_suffix = ''
  smem_align = ''
  smem_prefix = ''
  smem_prefix_for_cast = true
  arg_int_prefix = 'const int'
  barrier = ''
  code_for_workitem: Record<string, (...x: any[]) => string> = {}
  extra_args: string[] = []
  float4?: string
  type_map: Map<DType, string> = new Map()
  get_dtype = (dtype: DType) => {
    const res = this.type_map.get(dtype)
    if (res) return res
    throw new Error(`DType ${dtype} not supported`)
  }
  infinity = 'INFINITY'
  nan = 'NAN'
  r?: Map<UOp, string>
  override code_for_op = new Map<Ops, (...a: (string | DType)[]) => string>([
    [Ops.SQRT, (x, dtype) => `sqrt(${x})`],
    [Ops.RECIP, (x, dtype) => `(1/${x})`],
    [Ops.NEG, (x, dtype) => `-${x}`],
    [Ops.EXP2, (x, dtype) => `exp2(${x})`],
    [Ops.LOG2, (x, dtype) => `log2(${x})`],
    [Ops.SIN, (x, dtype) => `sin(${x})`],
    [Ops.AND, (a, b, dtype) => `(${a}&${b})`],
    [Ops.XOR, (a, b, dtype) => `(${a}^${b})`],
    [Ops.OR, (a, b, dtype) => `(${a}|${b})`],
    [Ops.ADD, (a, b, dtype) => `(${a}+${b})`],
    [Ops.SUB, (a, b, dtype) => `(${a}-${b})`],
    [Ops.MUL, (a, b, dtype) => `(${a}*${b})`],
    [Ops.MOD, (a, b, dtype) => `(${a}%${b})`],
    [Ops.IDIV, (a, b, dtype) => `(${a}/${b})`],
    [Ops.CMPNE, (a, b, dtype) => `(${a}!=${b})`],
    [Ops.SHR, (a, b, dtype) => `(${a}>>${b})`],
    [Ops.SHL, (a, b, dtype) => `(${a}<<${b})`],
    [Ops.CMPLT, (a, b, dtype) => `(${a}<${b})`],
    [Ops.WHERE, (a, b, c, dtype) => `(${a}?${b}:${c})`],
  ])
  string_rewrite = base_rewrite
  override extra_matcher = extra_pm

  get_kernel_modifier(uops: UOp[]) {
    return ''
  }
  render_kernel({ bufs, function_name, kernel, uops, prefix }: RenderKernelArgs) {
    const tmp = bufs.values().some(({ dtype }) => dtype instanceof ImageDType) ? 'const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n' : ''
    const buftypes = [...bufs.values()].map(({ name, dtype, mutable }) => [name, (dtype instanceof ImageDType || dtype instanceof PtrDType) ? this.render_dtype(dtype, mutable) + this.buffer_suffix : dtype === dtypes.int ? this.arg_int_prefix : undefined])
    const prg = [`${this.kernel_prefix}void ${this.get_kernel_modifier(uops)}${function_name}(`, ...buftypes.map(([name, t]) => `${t} ${name}`).join(', '), ...this.extra_args.join(', '), ') {\n' + tmp, kernel.join('\n'), '\n}'].join('')
    return prefix === undefined ? prg : `${prefix.join('\n')}\n${prg}`
  }
  render_cast = (dt: DType, val: string): string => `(${this.render_dtype(dt)})(${val})`
  render_dtype = (dt: DType, mutable = true): string => {
    if (dt instanceof ImageDType) return `${mutable ? 'write_only' : 'read_only'} image2d_t`
    if (dt instanceof PtrDType) return (dt.local && this.smem_prefix_for_cast ? this.smem_prefix : this.buffer_prefix) + this.render_dtype(dt.base) + '*'
    const scalar = dt.scalar()
    return (this.type_map.get(scalar) || scalar.name) + ((dt.count) > 1 ? dt.count.toString() : '')
  }
  bufs?: Map<UOp, Buf>
  get = (key: UOp) => this.r?.get(key) // hacky helper
  override render = (name: string, uops: UOp[]): string => {
    const r = new Map<UOp, string>()
    this.r = r

    const child_count = new Map<UOp, number>()
    for (const u of uops) {
      for (const v of u.src) child_count.set(v, (child_count.get(v) || 0) + 1)
    }

    this.bufs = new Map()
    // TODO: Getting all bufs at once, cause webgpu needs to know which are mutable, should moved back to the for loop below
    for (const u of uops) {
      if ([Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR].includes(u.op)) {
        this.bufs.set(u, { name: u.op === Ops.DEFINE_GLOBAL ? `data${u.arg}` : u.arg[0], dtype: u.dtype, mutable: false })
        continue
      }

      // mark buffers that we store to writable
      if (u.op === Ops.STORE) {
        for (const up of u.src[0].toposort) {
          if (up.op === Ops.DEFINE_GLOBAL) this.bufs.set(up, { name: this.bufs.get(up)!.name, dtype: this.bufs.get(up)!.dtype, mutable: true })
        }
      }
    }

    const kernel = []
    let depth = 1
    const c = new DefaultMap<string, number>(undefined, () => 0)
    for (const u of uops) {
      if ([Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR].includes(u.op)) {
        r.set(u, u.op === Ops.DEFINE_GLOBAL ? `data${u.arg}` : u.arg[0])
        continue
      }

      // naming
      let prefix
      if (u.op === Ops.SPECIAL) r.set(u, u.arg[0])
      else {
        const prefixes = new Map([
          [Ops.RANGE, 'ridx'],
          [Ops.WMMA, 'wmma'],
          [Ops.DEFINE_LOCAL, 'temp'],
          [Ops.CONST, 'const'],
          [Ops.CAST, 'cast'],
          [Ops.BITCAST, 'cast'],
          [Ops.GEP, 'gep'],
          [Ops.VECTORIZE, 'cast'],
          [Ops.NOOP, 'precast'],
          [Ops.INDEX, 'bidx'],
          [Ops.DEFINE_ACC, 'acc'],
          [Ops.LOAD, 'val'],
        ])
        prefix = prefixes.get(u.op) || 'alu'

        r.set(u, `${prefix}${set_default(c, prefix, 0)}`)
      }
      let l = this.string_rewrite.rewrite(u, this)
      if (l === undefined) throw new Error(`failed to render: ${u}`)

      if ([Ops.ENDIF, Ops.ENDRANGE].includes(u.op)) depth -= 1
      if (
        [Ops.CONST, Ops.GEP, Ops.INDEX].includes(u.op) ||
        ([Ops.VECTORIZE, ...GroupOp.ALU, Ops.CAST, Ops.BITCAST].includes(u.op) &&
          (child_count.get(u) || 0) === 1 &&
          !env.get('EXPAND_SSA'))
      ) {
        r.set(u, l!)
      } else {
        if ([Ops.RANGE, Ops.ASSIGN, Ops.DEFINE_LOCAL].includes(u.op) || u.dtype === dtypes.void) {
          if (u.op === Ops.ASSIGN) r.set(u, r.get(u.src[0])!)
        } else {
          l = `${this.render_dtype(u.dtype)} ${r.get(u)!} = ${l}` + (u.op !== Ops.SPECIAL ? ';' : '')
        }
        kernel.push('  '.repeat(depth) + l)
        if (prefix) c.set(prefix, (c.get(prefix) || 0) + 1) // if it was used, increment
      }
      if ([Ops.IF, Ops.RANGE].includes(u.op)) depth += 1
    }
    //  NOTE: this relies on bufs dict preserving order
    const res = this.render_kernel({ function_name: name, kernel, bufs: this.bufs, uops })
    this.r = undefined
    this.bufs = undefined
    return res
  }
}

export class ClangRenderer extends CStyleLanguage {
  override device: string = 'CLANG'
  override float4 = '(float4)'
  override has_local = false
  override global_max = undefined
  override infinity = '__builtin_inff()'
  override nan = '__builtin_nanf("")'

  // language options
  override type_map = new Map([[dtypes.bool, '_Bool'], [dtypes.half, '__fp16']])
  override buffer_suffix = ' restrict'
  override code_for_op = new Map<Ops, (...args: (string | DType)[]) => string>([
    ...new CStyleLanguage().code_for_op.entries().filter(([k, v]) => ![Ops.EXP2, Ops.SIN, Ops.LOG2].includes(k)),
    [Ops.SQRT, (x: any, dtype: any) => dtype === dtypes.float64 ? `__builtin_sqrt(${x})` : `__builtin_sqrtf(${x})`],
  ])
  // LLVM legalizes double => half cast on systems that don't support it natively (like x86 cpus without AVX512-FP16) into a compiler-rt libcall.
  override extra_matcher = new PatternMatcher([[UPat.var('x', dtypes.float64).cast(dtypes.float16), ({ x }) => x.cast(dtypes.float32).cast(dtypes.float16)]]).add(new CStyleLanguage().extra_matcher)
  override tensor_cores = !env.AMX ? undefined : [dtypes.float].map((dt) => [dt, idiv(64, dt.itemsize)] as const).map(([dt, sz]) => new TensorCore({ dims: [sz, sz, 1], threads: 1, elements_per_thread: [sz, sz, sz * sz], dtype_in: dt, dtype_out: dt, swizzle: [undefined, [[], [4, 5, 6, 7, 0, 1, 2, 3]]], opts: ['u0', 'u0', 'u0', 'u0', 'u1', 'u1', 'u1', 'u1'] }))

  render_vector_prefix = (dt: DType): string => `typedef ${this.render_dtype(dt.scalar())} ${this.render_dtype(dt)} __attribute__((aligned(${dt.itemsize}),vector_size(${dt.itemsize})));`

  override render_kernel = ({ bufs, function_name, kernel, uops, prefix }: RenderKernelArgs): string => {
    prefix = uops_to_dtypes(uops).filter((dt) => dt.count > 1).map((dt) => this.render_vector_prefix(dt))
    // https://github.com/corsix/amx
    for (const [name, [N, M, _], dtypeIn] of dedup(uops.filter((uop) => uop.op === Ops.WMMA).map((uop) => uop.arg))) {
      prefix = [
        ...prefix,
        '#define AMX_SET(imm5) __asm("nop\\nnop\\nnop\\n.word (0x201000+(%0<<5)+%1)" : : "i"(17), "i"(imm5) : "memory")',
        '#define AMX(op, gpr, btf) __asm(".word (0x201000+(%0 << 5)+0%1-((0%1>>4)*6))" : : "i"(op), "r"((unsigned long long)(gpr)+(btf)) : "memory")',
      ]
      const out = this.render_dtype(dtypeIn.vec(N * N))
      prefix = [
        ...prefix,
        // 'static' in C roughly means that function symbol isn't exported. LLVM puts those symbols at the end of object file which allows Clang JIT
        // to just jump at the start of a shellcode whithout having to deal with symbols or trampolines at all. This is better than having to inline
        // wmma function every time it is called or wasting complexity on a symbol parsing and a memory page on trampoline.
        `static ${out} __$${this.render_dtype(dtypeIn.vec(N))} data1, ${this.render_dtype(dtypeIn.vec(M))} data2, ${out} data0){{ AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }} AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull); for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}`,
      ]
    }
    return super.render_kernel({ function_name, kernel, bufs, uops, prefix })
  }
}

class OpenCLRenderer extends CStyleLanguage {
  override device = 'GPU'

  // language options
  override kernel_prefix = '__kernel '
  override buffer_prefix = '__global '
  override smem_align = '__attribute__ ((aligned (16))) '
  override smem_prefix = '__local '
  override barrier = 'barrier(CLK_LOCAL_MEM_FENCE);'
  override float4 = '(float4)'
  override code_for_workitem: Record<string, (...x: any[]) => string> = { 'g': (x) => `get_group_id(${x})`, 'l': (x) => `get_local_id(${x})`, 'i': (x) => `get_global_id(${x})` }
  override type_map = new Map([[dtypes.int8, 'char'], [dtypes.uint8, 'uchar'], [dtypes.uint32, 'uint'], [dtypes.uint16, 'ushort'], [dtypes.uint64, 'ulong'], [dtypes.bfloat16, 'ushort']])

  override string_rewrite = new PatternMatcher<OpenCLRenderer, string | undefined>([
    new UPat(Ops.BITCAST).named('x').fn(({ ctx, x }) => `as_${ctx.render_dtype(x.dtype)}(${ctx.get(x.src[0])})`),
    // load/store image (OpenCL)
    new UPat(Ops.LOAD, dtypes.float.vec(4), [UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))), UPat.var('var'), UPat.var('gate')]).fn(
      ({ ctx, buf, idx, var: v, gate }) => `(${ctx.get(gate)}?read_imagef(${ctx.get(buf)}, smp, ${ctx.get(idx)}):${ctx.get(v)})`,
    ),
    new UPat(Ops.LOAD, dtypes.float.vec(4), [UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2)))]).fn(({ ctx, buf, idx }) => `read_imagef(${ctx.get(buf)}, smp, ${ctx.get(idx)})`),
    new UPat(Ops.STORE, undefined, [UPat.var('buf').index(UPat.var('idx', dtypes.int.vec(2))), UPat.var('var', dtypes.float.vec(4))], undefined, undefined, true).fn(({ ctx, buf, idx, var: v }) => `write_imagef(${ctx.get(buf)}, ${ctx.get(idx)}, ${ctx.get(v)});`),
  ]).add(base_rewrite)

  override render_kernel({ function_name, kernel, bufs, uops, prefix }: RenderKernelArgs) {
    if (uops.some((uop) => uop.dtype.base === dtypes.half)) prefix = ['#pragma OPENCL EXTENSION cl_khr_fp16 : enable', ...(prefix?.length ? prefix : [])]
    return super.render_kernel({ function_name, kernel, bufs, uops, prefix })
  }
}
export class IntelRenderer extends OpenCLRenderer {
  override device = 'GPU'
  override suffix = 'INTEL'
  override kernel_prefix = '__attribute__((intel_reqd_sub_group_size(8)))\n' + '__kernel '
  override tensor_cores = [new TensorCore({ dims: [8, 8, 16], threads: 8, elements_per_thread: [16, 16, 8], dtype_in: dtypes.half, dtype_out: dtypes.float, opts: ['l0', 'l0', 'l0', 'u1', 'u1', 'u1'], swizzle: [[[4, 5, 6], [0, 1, 2, 3, 7, 8, 9]], [[0, 1, 2], [7, 8, 9, 3, 4, 5, 6]]] })]

  override string_rewrite = new PatternMatcher<IntelRenderer, string | undefined>([
    new UPat(Ops.CAST, dtypes.bfloat16, [UPat.var('x', dtypes.float)]).fn(({ ctx, x }) => `intel_convert_bfloat16_as_ushort(${ctx.get(x)})`),
    new UPat(Ops.CAST, dtypes.float, [UPat.var('x', dtypes.bfloat16)]).fn(({ ctx, x }) => `intel_convert_as_bfloat16_float(${ctx.get(x)})`),
  ]).add(new OpenCLRenderer().string_rewrite)

  override render_kernel({ function_name, kernel, bufs, uops, prefix }: RenderKernelArgs) {
    prefix = []
    for (const arg of dedup(uops.filter((uop) => uop.op === Ops.WMMA).map((uop) => uop.arg))) {
      const dt_in = arg[2] === dtypes.bfloat16 ? ['ushort', 'bf16'] : [arg[2].name, 'f16']
      prefix.push(`${arg[3].name}8 __${arg[0]}(${dt_in[0]}16 a, ${dt_in[0]}16 b, ${arg[3].name}8 c) {\nreturn intel_sub_group_${dt_in[1]}_${dt_in[1]}_matrix_mad_k16(as_int8(a), as_int8(b), c);\n}`)
    }
    return super.render_kernel({ function_name, kernel, bufs, uops, prefix: prefix.length ? prefix : undefined })
  }
}
export class MetalRenderer extends CStyleLanguage {
  override device = 'METAL'
  override shared_max = 32768
  override tensor_cores: TensorCore[] = env.machine() === 'arm64'
    ? [[dtypes.float, dtypes.float], [dtypes.half, dtypes.float], [dtypes.half, dtypes.half], [dtypes.bfloat16, dtypes.float], [dtypes.bfloat16, dtypes.bfloat16]].map(
      (d) => new TensorCore({ dims: [8, 8, 8], threads: 32, elements_per_thread: [2, 2, 2], dtype_in: d[0], dtype_out: d[1], opts: ['u0', 'l0', 'l1', 'l1', 'l0', 'l1'], swizzle: [[[6, 1, 2, 7, 4], [8, 0, 3, 5]], [[0, 5, 6, 3, 7], [1, 2, 4, 8]]] }),
    )
    : []

  // language options
  override kernel_prefix = 'kernel '
  override buffer_prefix = 'device '
  override smem_prefix = 'threadgroup '
  override arg_int_prefix = 'constant int&'
  override barrier = 'threadgroup_barrier(mem_flags::mem_threadgroup);'
  override float4 = 'float4'
  override code_for_workitem: Record<string, (...x: any[]) => string> = { 'g': (x) => `gid.${String.fromCharCode(120 + Number(x))}`, 'l': (x) => `lid.${String.fromCharCode(120 + Number(x))}` }
  // uint3 used for gid/lid - TODO: this should probably be `ushort3 lid [[thread_position_in_threadgroup]]`
  override extra_args = ['uint3 gid [[threadgroup_position_in_grid]]', 'uint3 lid [[thread_position_in_threadgroup]]']
  override type_map = new Map([[dtypes.bfloat16, 'bfloat']])

  // precise::sin
  override code_for_op = new Map<Ops, (...args: (string | DType)[]) => string>([...new CStyleLanguage().code_for_op.entries(), [Ops.SIN, (x, dtype) => `precise::sin(${x})`]])

  // upcast to float32 all the ops that don't support bfloat16
  override extra_matcher = new PatternMatcher([
    // NOTE: this is copied from PTX
    new UPat([Ops.SQRT, Ops.EXP2, Ops.LOG2, Ops.SIN], dtypes.bfloat16).named('x').fn(({ x }) => (new UOp(x.op, dtypes.float, x.src.map((vv) => vv.cast(dtypes.float)), x.arg).cast(dtypes.bfloat16))),
  ]).add(extra_pm)

  override string_rewrite = new PatternMatcher<MetalRenderer, string | undefined>([
    new UPat(Ops.BITCAST).named('x').fn(({ ctx, x }) => `as_type<${ctx.render_dtype(x.dtype)}>(${ctx.get(x.src[0])})`),
  ]).add(base_rewrite)

  override render_kernel({ function_name, kernel, bufs, uops, prefix }: RenderKernelArgs) {
    prefix = ['#include <metal_stdlib>', 'using namespace metal;']
    const wmma_args = new Set(uops.filter((uop) => uop.op === Ops.WMMA).map((uop) => uop.arg))
    for (const arg of wmma_args) {
      const dtype_out = this.render_dtype(arg[3].vec(2)), dtype_in = this.render_dtype(arg[2].vec(2))
      prefix.push(
        `${dtype_out} __${arg[0]}(${dtype_in} a, ${dtype_in} b, ${dtype_out} c){
      simdgroup_${this.render_dtype(arg[2])}8x8 mat_a, mat_b; simdgroup_${this.render_dtype(arg[3])}8x8 mat_c;
      mat_a.thread_elements()[0] = a[0]; mat_b.thread_elements()[0] = b[0]; mat_c.thread_elements()[0] = c[0];
      mat_a.thread_elements()[1] = a[1]; mat_b.thread_elements()[1] = b[1]; mat_c.thread_elements()[1] = c[1];
      simdgroup_multiply_accumulate(mat_c, mat_a, mat_b, mat_c);\n  return ${dtype_out}(mat_c.thread_elements()[0], mat_c.thread_elements()[1]);\n}`,
      )
    }
    return super.render_kernel({ function_name, kernel, bufs, uops, prefix })
  }
}

const _nms = 'xyzwabcdefghijkl'
const cuda_tc_opts = ['u0', 'l0', 'l0', 'l1', 'l1', 'l1', 'u1'] // shared by all shapes with M=16 N=8

export class CUDARenderer extends CStyleLanguage {
  override device = 'CUDA'
  override global_max: [number, number, number] = [2147483647, 65535, 65535]
  override local_max = [1024, 1024, 64]
  override shared_max = 49152
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-multiply-accumulate-instructions
  static tc_81616 = [[dtypes.half, dtypes.float], [dtypes.bfloat16, dtypes.float]].map((d) => new TensorCore({ dims: [8, 16, 16], threads: 32, elements_per_thread: [8, 4, 4], dtype_in: d[0], dtype_out: d[1], opts: cuda_tc_opts, swizzle: [[[6, 7, 2, 3, 4], [0, 1, 9, 5, 10, 8]], [[6, 7, 9, 0, 1], [2, 3, 4, 10, 5, 8]]] }))
  static tc_8168_f16 = [new TensorCore({ dims: [8, 16, 8], threads: 32, elements_per_thread: [4, 2, 4], dtype_in: dtypes.half, dtype_out: dtypes.float, opts: cuda_tc_opts, swizzle: [[[6, 7, 2, 3, 4], [0, 1, 8, 5, 9]], [[6, 7, 8, 0, 1], [2, 3, 4, 9, 5]]] })]
  static tc_8168_tf32 = [new TensorCore({ dims: [8, 16, 8], threads: 32, elements_per_thread: [4, 2, 4], dtype_in: dtypes.float, dtype_out: dtypes.float, opts: cuda_tc_opts, swizzle: [[[5, 6, 2, 3, 4], [0, 1, 8, 9, 7]], [[5, 6, 8, 0, 1], [2, 3, 4, 9, 7]]] })]

  static tc_sm80 = [...CUDARenderer.tc_81616, ...CUDARenderer.tc_8168_f16, ...(env.get_num('ALLOW_TF32', 0) ? CUDARenderer.tc_8168_tf32 : [])]
  static tc_sm75 = CUDARenderer.tc_8168_f16
  constructor(public arch: string) {
    super()
    this.tensor_cores = Number(arch.slice(3)) >= 80 ? CUDARenderer.tc_sm80 : Number(arch.slice(3)) >= 75 ? CUDARenderer.tc_sm75 : []
  }

  // language options
  override kernel_prefix = 'extern "C" __global__ '
  override smem_prefix = '__shared__ '
  override smem_prefix_for_cast = false
  override barrier = '__syncthreads();'
  override float4 = 'make_float4'
  override code_for_workitem: Record<string, (...x: any[]) => string> = { 'g': (x) => `blockIdx.${String.fromCharCode(120 + Number(x))}`, 'l': (x) => `threadIdx.${String.fromCharCode(120 + Number(x))}`, 'i': (x) => `(blockIdx.${String.fromCharCode(120 + Number(x))}*blockDim.${String.fromCharCode(120 + Number(x))}+threadIdx.${String.fromCharCode(120 + Number(x))})` }
  override code_for_op = new Map<Ops, (...args: any[]) => string>([
    ...new CStyleLanguage().code_for_op,
    [Ops.SIN, (x, dtype) => [dtypes.half, dtypes.bfloat16].includes(dtype) ? `hsin(${x})` : `sin(${x})`],
    [Ops.LOG2, (x, dtype) => [dtypes.half, dtypes.bfloat16].includes(dtype) ? `hlog2(${x})` : `log2(${x})`],
    [Ops.EXP2, (x, dtype) => [dtypes.half, dtypes.bfloat16].includes(dtype) ? `hexp2(${x})` : `exp2(${x})`],
    [Ops.SQRT, (x, dtype) => [dtypes.half, dtypes.bfloat16].includes(dtype) ? `hsqrt(${x})` : `sqrt(${x})`],
    [Ops.RECIP, (x, dtype) => [dtypes.half, dtypes.bfloat16].includes(dtype) ? `hrcp(${x})` : `(1/${x})`],
  ])
  override type_map = new Map([[dtypes.bfloat16, 'nv_bfloat16']])

  render_vector_prefix(dt: DType): string {
    const vec = this.render_dtype(dt), scal = this.render_dtype(dt.scalar())
    const elems = _nms.split('').slice(0, dt.count).join(', '), header = _nms.split('').slice(0, dt.count).map((x) => `${scal} ${x}`).join(', ')
    return `struct __align__(${dt.itemsize}) ${vec} { ${scal} ${elems}; }; __device__ ${vec} make_${vec}(${header}) { ${vec} r={${elems}}; return r; }`
  }
  override render_kernel({ function_name, kernel, bufs, uops, prefix }: RenderKernelArgs) {
    // TODO: why is dtypes.bfloat16.name == "__bf16"? would be easier not override dtypes.name
    prefix = ['#define INFINITY (__int_as_float(0x7f800000))', '#define NAN (__int_as_float(0x7fffffff))']

    const used_dtypes = uops_to_dtypes(uops)
    if (used_dtypes.some((dt) => dt.scalar() === dtypes.half)) prefix.push('#include <cuda_fp16.h>')
    if (used_dtypes.some((dt) => dt.scalar() === dtypes.bfloat16)) prefix.push('#include <cuda_bf16.h>')
    prefix.push(...used_dtypes.filter((dt) => [4, 8].includes(dt.count) && [dtypes.half, dtypes.bfloat16].includes(dt.scalar())).map((dt) => this.render_vector_prefix(dt)))

    const dt_map = new Map([[dtypes.float, 'tf32'], [dtypes.half, 'f16'], [dtypes.bfloat16, 'bf16']])
    for (const [name, [N, M, K], dtype_in, dtype_out, _, _1, upcast_axes] of dedup(uops.filter((uop) => uop.op === Ops.WMMA).map((uop) => uop.arg))) {
      const upcast_sizes = upcast_axes.map((upcast: any) => prod(upcast.map(([_, size]: any) => size)))
      const wmma_dtypes = zip([dtype_in, dtype_in, dtype_out], upcast_sizes).map(([dtype, size]) => this.render_dtype(dtype.vec(size)))
      const n_operands = zip([dtype_in, dtype_in, dtype_out], upcast_sizes).map(([dtype, size]: any) => idiv(size * dtype.itemsize, 4)) // 4 => CUDA reg size in bytes
      const operands = range(sum(n_operands)).map((i) => `%${i}`)

      // mma operands => {c}, {a}, {b}, {c}
      prefix.push(`__device__ ${wmma_dtypes[2]} __${name}(${wmma_dtypes[0]} a, ${wmma_dtypes[1]} b, ${wmma_dtypes[2]} c){
        int *a_pk = (int *)(&a), *b_pk = (int *)(&b);\n  asm("mma.sync.aligned.m${M}n${N}k${K}.row.col.f32.${dt_map.get(dtype_in)}.${dt_map.get(dtype_in)}.f32"
            "{${operands.slice(0, n_operands[2]).join(', ')}}, {${operands.slice(n_operands[2], n_operands[2] + n_operands[0]).join(', ')}},"
            "{${operands.slice(-n_operands[1]).join(', ')}}, {${operands.slice(0, n_operands[2]).join(', ')}};"
          : ${range(n_operands[2]).map((i) => `"+f"(c.${_nms[i]})`).join(',')}
          : ${range(n_operands[0]).map((i) => `"r"(a_pk[${i}])`).join(', ')}, ${range(n_operands[1]).map((i) => `"r"(b_pk[${i}])`).join(', ')});
      return c;\n}}`)
    }
    return super.render_kernel({ function_name, kernel, bufs, uops, prefix })
  }
  override get_kernel_modifier(uops: UOp[]): string {
    const maxThreadsPerBlock = prod(uops.filter((u) => u.op === Ops.SPECIAL && u.arg[0][0] === 'l').map((u) => u.arg[1]))
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
    return `__launch_bounds__(${maxThreadsPerBlock}) `
  }
}
export const cast_float_to_bf16 = (x: UOp): UOp => {
  if (x.dtype !== dtypes.float) throw new Error('cast float -> bf16 must start with float')
  x = x.bitcast(dtypes.uint)
  x = x.neg().bitwise_and(0x7f800000).where(x.add((x.rshift(16)).bitwise_and(1)).add(0x7fff), (x.bitwise_and(0xffff)).where(x.bitwise_or(0x10000), x))
  return x.rshift(16).cast(dtypes.ushort).bitcast(dtypes.bfloat16)
}

export class AMDRenderer extends CStyleLanguage {
  override device = 'AMD'
  override shared_max = 65536
  // https://gpuopen.com/learn/wmma_on_rdna3/
  override tensor_cores = [[dtypes.half, dtypes.float], [dtypes.half, dtypes.half]].map(([di, doo]: any) => new TensorCore({ dims: [16, 16, 16], threads: 32, elements_per_thread: [16, 16, 8], dtype_in: di, dtype_out: doo, opts: ['l0', 'l0', 'l0', 'l0', 'l1', 'u1', 'u1', 'u1'], swizzle: [[[4, 9, 10, 11, 0], [1, 2, 3, 5, 6, 7, 8]], [[0, 1, 2, 3, 4], [9, 10, 11, 5, 6, 7, 8]]] }))

  // language options
  static ockl = ['local_id', 'group_id', 'local_size'].map((name) => [`__ockl_get_${name}`, 'unsigned int', 'size_t', 'const'])
  static ocml = [dtypes.float, dtypes.double, dtypes.half].map((dtype) => [dtype.name, dtype.itemsize * 8]).flatMap(([dt, n]) => [['fmax', 'const'], ['exp2', 'pure'], ['log2', 'pure'], ['sqrt', 'const'], ['sin', '']].map(([name, atr]) => [`__ocml_${name}_f${n}`, 'fmax' === name ? `${dt}, ${dt}` : dt, dt, atr]))

  static kernel_prefix = [...[...AMDRenderer.ockl, ...AMDRenderer.ocml].map(([meth, dti, dto, atr]) => `extern "C" __attribute__((device{f", ${atr}" if atr else ""})) ${dto} ${meth}(${dti});`), 'extern "C" __attribute__((global))'].join('\n')
  override code_for_workitem: Record<string, (...x: any[]) => string> = { 'g': (x) => `__ockl_get_group_id(${x})`, 'l': (x) => `__ockl_get_local_id({x})`, 'i': (x) => `(__ockl_get_group_id(${x})*__ockl_get_local_size(${x})+__ockl_get_local_id(${x}))` }
  override code_for_op = new Map<Ops, (...args: any[]) => string>([
    ...new CStyleLanguage().code_for_op,
    [Ops.SIN, (x, dtype) => `__ocml_sin_f${dtype === dtypes.half ? 16 : dtype === dtypes.double ? 64 : 32}(${x})`],
    [Ops.LOG2, (x, dtype) => `__ocml_log2_f${dtype === dtypes.half ? 16 : dtype === dtypes.double ? 64 : 32}(${x})`],
    [Ops.EXP2, (x, dtype) => `__ocml_exp2_f${dtype === dtypes.half ? 16 : dtype === dtypes.double ? 64 : 32}(${x})`],
    [Ops.SQRT, (x, dtype) => `__ocml_sqrt_f${dtype === dtypes.half ? 16 : dtype === dtypes.double ? 64 : 32}(${x})`],
  ])
  override smem_prefix = '__attribute__((shared))'
  override barrier = '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");' + '__builtin_amdgcn_s_barrier();' + '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");'
  override float4 = 'make_float4'
  override type_map = new Map([[dtypes.bfloat16, 'hip_bfloat16']])
  override extra_matcher = new PatternMatcher([
    // cast bfloat16 alus to float
    new UPat(Ops.WHERE, undefined, [UPat.var('b'), UPat.var('x', dtypes.bfloat16), UPat.var('y', dtypes.bfloat16)]).fn(({ b, x, y }) => new UOp(Ops.WHERE, dtypes.float, [b, x.cast(dtypes.float), y.cast(dtypes.float)]).cast(dtypes.bfloat16)),
    new UPat(GroupOp.ALU, dtypes.bfloat16).named('x').fn(({ x }) => new UOp(x.op, dtypes.float, x.src.map((vv) => vv.cast(dtypes.float)), x.arg).cast(dtypes.bfloat16)),
    new UPat(GroupOp.ALU, dtypes.bool, [UPat.var('x', dtypes.bfloat16), UPat.var('y', dtypes.bfloat16)]).named('alu').fn(({ alu, x, y }) => new UOp(alu.op, dtypes.bool, [x.cast(dtypes.float), y.cast(dtypes.float)], alu.arg)),
    // add float intermediate casting for bfloat16
    new UPat(Ops.CAST, undefined, UPat.var('y', dtypes.bfloat16)).named('x').fn(({ x, y }) => x.dtype !== dtypes.float ? y.cast(dtypes.float).cast(x.dtype) : undefined),
    new UPat(Ops.CAST, dtypes.bfloat16, UPat.var('x')).fn(({ x }) => x.dtype !== dtypes.float ? x.cast(dtypes.float).cast(dtypes.bfloat16) : undefined),
    // bfloat16 casting
    UPat.cvar('x', dtypes.bfloat16).fn(({ x }) => cast_float_to_bf16(UOp.const(dtypes.float, x.arg))),
    new UPat(Ops.CAST, dtypes.float, UPat.var('x', dtypes.bfloat16)).fn(({ x }) => (x.bitcast(dtypes.ushort).cast(dtypes.uint).lshift(16)).bitcast(dtypes.float)),
    new UPat(Ops.CAST, dtypes.bfloat16, UPat.var('x', dtypes.float)).fn(({ x }) => cast_float_to_bf16(x)),
  ]).add(extra_pm)

  render_vector_prefix(dtype: DType): string {
    const vec = this.render_dtype(dtype), scal = this.render_dtype(dtype.scalar())
    return `typedef ${scal} ${vec} __attribute__((ext_vector_type(${dtype.count})));\nstatic inline __attribute__((device)) ` + `${vec} make_${vec}(${_nms.split('').map((x) => `${scal} ${x}`).join(', ')}) { return { ${_nms.split('').slice(0, dtype.count).join(', ')} }; }`
  }
  override render_kernel({ function_name, kernel, bufs, uops, prefix }: RenderKernelArgs): string {
    prefix = ['#define INFINITY (__builtin_inff())', '#define NAN (__builtin_nanf(""))', 'typedef long unsigned int size_t;', '#define half _Float16']

    const used_dtypes = uops_to_dtypes(uops)
    if (used_dtypes.some((dt) => dt.scalar() === dtypes.bfloat16)) prefix.push('typedef unsigned short hip_bfloat16;')
    prefix.push(...used_dtypes.filter((dt) => dt.count > 1).map((dt) => this.render_vector_prefix(dt)))

    for (const arg of dedup(uops.filter((uop) => uop.op === Ops.WMMA).map((uop) => uop.arg))) { // TODO: handle TCs f32_bf16 and bf16_bf16 w/ wrapper
      if (arg[3] === dtypes.float) prefix.push(`#define __${arg[0]} __builtin_amdgcn_wmma_f32_16x16x16_f16_w32`)
      else {prefix.push(
          `static inline __attribute__((device)) half8 __${arg[0]}` + `(half16 a, half16 b, half8 c) {
        half16 c_frag = {}; half8 d; for (int n = 0; n < 8; n++) { c_frag[n*2] = c[n]; }
        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c_frag, false);
        for (int n = 0; n < 8; n++) { d[n] = c_frag[n*2]; } return d;\n}`,
        )}
    }
    return super.render_kernel({ function_name, kernel, bufs, uops, prefix })
  }
  override get_kernel_modifier(uops: UOp[]): string {
    const requiredMaxThreadsPerBlock = prod(uops.filter((u) => u.op === Ops.SPECIAL && u.arg[0][0] === 'l').map((u) => u.arg[1]))
    // https://clang.llvm.org/docs/AttributeReference.html#amdgpu-flat-work-group-size
    // NOTE: this makes hlb_cifar10 twice as fast, there may be more gains in tweaking these parameters
    return `__attribute__((amdgpu_flat_work_group_size(1, ${requiredMaxThreadsPerBlock})))`
  }
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
