import { type DType, dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { env } from '../env/index.ts'
import { dedup, DefaultMap, floatString, idiv, NotImplemented, set_default, strip_parens } from '../helpers.ts'
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
  code_for_workitem: Record<string, (...x: any[]) => string> = {} as any
  extra_args: string[] = []
  float4?: string
  _type_map: Map<DType, string> = new Map()
  get_dtype = (dtype: DType) => {
    const res = this._type_map.get(dtype)
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

  get_kernel_modifier = (uops: UOp[]) => ''
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
    return (this._type_map.get(scalar) || scalar.name) + ((dt.count) > 1 ? dt.count.toString() : '')
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
  override _type_map = new Map([[dtypes.bool, '_Bool'], [dtypes.half, '__fp16']])
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

export class OpenCLRenderer extends CStyleLanguage {
  constructor() {
    super()
    throw new NotImplemented()
  }
}

export class IntelRenderer extends OpenCLRenderer {
  constructor() {
    super()
    throw new NotImplemented()
  }
}
export class MetalRenderer extends CStyleLanguage {
  constructor() {
    super()
    throw new NotImplemented()
  }
}
export class CUDARenderer extends CStyleLanguage {
  constructor() {
    super()
    throw new NotImplemented()
  }
}
export class AMDRenderer extends CStyleLanguage {
  constructor() {
    super()
    throw new NotImplemented()
  }
}
export class NVRenderer extends CUDARenderer {
  override device = 'NV' as any
}
export class HIPRenderer extends AMDRenderer {
  override device = 'HIP' as any
}
export class QCOMRenderer extends OpenCLRenderer {
  override device = 'QCOM' as any
}
