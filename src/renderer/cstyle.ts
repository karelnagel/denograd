import { DeviceType } from '../device.ts'
import { type DType, dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { AMX, assert, dedup, get_env, isInf, isNone, isNotNone, setDefault, strip_parens } from '../helpers.ts'
import { GroupOp, Ops, PatternMatcher, UOp, UPat } from '../ops.ts'
import { Renderer, TensorCore } from './index.ts'

export const base_rewrite = new PatternMatcher<{ ctx: CStyleLanguage } & Record<string, UOp>, string | undefined>([
  [new UPat(Ops.DEFINE_ACC).named('x'), ({ ctx, x }) => ctx.get(x.src[0])],
  [new UPat(Ops.ASSIGN).named('x'), ({ ctx, x }) => `${ctx.get(x.src[0])} = ${ctx.get(x.src[1])};`],
  [new UPat(Ops.IF).named('x'), ({ ctx, x }) => `if (${ctx.get(x.src[0])}) {`],
  [new UPat([Ops.ENDIF, Ops.ENDRANGE]), ({ ctx }) => '}'],
  [new UPat(Ops.WMMA).named('x'), ({ ctx, x }) => `__${x.arg[0]}(${ctx.get(x.src[0])}, ${ctx.get(x.src[1])}, ${ctx.get(x.src[2])})`],
  // r method accesses
  [new UPat(Ops.RANGE).named('x'), ({ ctx, x }) => `for (${ctx.render_dtype(x.dtype)} ${ctx.get(x)} = ${ctx.get(x.src[0])}; ${ctx.get(x)} < ${ctx.get(x.src[1])}; ${ctx.get(x)}++) {`],
  [
    new UPat(Ops.VECTORIZE).named('x'),
    ({ ctx, x }) => `${ctx.float4!.replace('float4', ctx.render_dtype(x.dtype))}` + (ctx.device === 'CLANG' ? `{${x.src.map((y: any) => ctx.get(y)).join(',')}}` : `(${x.src.map((y: any) => ctx.get(y)).join(',')})`),
  ],
  [new UPat(Ops.CAST).named('x'), ({ ctx, x }) => `(${ctx.render_cast(x.dtype, ctx.get(x.src[0])!)})`],
  [new UPat(Ops.BITCAST).named('x'), ({ ctx, x }) => `(*((${ctx.buffer_prefix}${ctx.render_dtype(x.dtype)}*)&${ctx.get(x.src[0])}))`],
  [new UPat(Ops.DEFINE_LOCAL).named('x'), ({ ctx, x }) => `${ctx.smem_align}${ctx.smem_prefix}${ctx.render_dtype(x.dtype.base)} ${ctx.get(x)}[${x.arg[1]}];`],
  [new UPat(Ops.BARRIER), ({ ctx }) => ctx.barrier],
  [new UPat(Ops.NOOP).named('x'), ({ ctx, x }) => ctx.get(x.src[0])],
  [new UPat(Ops.SPECIAL).named('x'), ({ ctx, x }) => `${ctx.code_for_workitem[x.arg[0][0] as keyof typeof ctx.code_for_workitem]?.(x.arg[0].at(-1)!)}; /* ${x.arg[1]} */`],
  // const
  [new UPat(Ops.CONST, undefined, undefined, Infinity, 'x'), ({ ctx, x }) => `(${ctx.render_cast(x.dtype, ctx.infinity)})`],
  [new UPat(Ops.CONST, undefined, undefined, -Infinity, 'x'), ({ ctx, x }) => `(${ctx.render_cast(x.dtype, `-${ctx.infinity}`)})`],
  [new UPat(Ops.CONST, dtypes.floats).named('x'), ({ ctx, x }) => isInf(x.arg) ? `(${ctx.render_cast(x.dtype, ctx.nan)})` : undefined],
  [new UPat(Ops.CONST, dtypes.float).named('x'), ({ ctx, x }) => `${x.arg}f`],
  [new UPat(Ops.CONST, dtypes.int64).named('x'), ({ ctx, x }) => `${x.arg}ll`],
  [new UPat(Ops.CONST, dtypes.uint64).named('x'), ({ ctx, x }) => `${x.arg}ull`],
  [new UPat(Ops.CONST, dtypes.uint32).named('x'), ({ ctx, x }) => `${x.arg}u`],
  [new UPat(Ops.CONST, dtypes.bool).named('x'), ({ ctx, x }) => x.arg ? '1' : '0'],
  // consts are rendered to larger type and casted
  [new UPat(Ops.CONST, [dtypes.bfloat16, dtypes.half]).named('x'), ({ ctx, x }) => `(${ctx.render_cast(x.dtype, `${x.arg}f`)})`],
  [new UPat(Ops.CONST, [dtypes.uint8, dtypes.uint16]).named('x'), ({ ctx, x }) => `(${ctx.render_cast(x.dtype, `${x.arg}u`)})`],
  [new UPat(Ops.CONST, [dtypes.int8, dtypes.int16]).named('x'), ({ ctx, x }) => `(${ctx.render_cast(x.dtype, x.arg)})`],
  // default const render
  [new UPat(Ops.CONST).named('x'), ({ ctx, x }) => x.arg.toString()],
  // new load/store
  [new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('idx')]), ({ ctx, buf, idx }) => `(${ctx.get(buf)}+${idx.arg === Ops.ADD ? strip_parens(ctx.get(idx)!) : ctx.get(idx)})`],
  [new UPat(Ops.LOAD, undefined, [UPat.var('bidx'), UPat.var('var'), UPat.var('gate')]), ({ ctx, bidx, var1, gate }) => `(${ctx.get(gate)}?*${ctx.get(bidx)}:${ctx.get(var1)})`],
  [new UPat(Ops.LOAD, undefined, [UPat.var('bidx')], undefined, undefined, true), ({ ctx, bidx }) => `*${ctx.get(bidx)}`],
  [new UPat(Ops.STORE, undefined, [UPat.var('bidx'), UPat.var('var')], undefined, undefined, true), (p) => `*${p.ctx.get(p.bidx)} = ${p.ctx.get(p.var)};`],
  // alu/gep
  [new UPat(GroupOp.ALU).named('x'), ({ ctx, x }) => ctx.code_for_op.get(x.op)!(...x.src.map((v) => v.op === x.op && [Ops.ADD, Ops.MUL, Ops.XOR].includes(x.op) ? strip_parens(ctx.get(v)!) : ctx.get(v)!), x.dtype)],
  [new UPat(Ops.GEP).named('x'), ({ ctx, x }) => ctx.get(x.src[0]) + (x.src[0].dtype.count > (['CUDA', 'NV'].includes(ctx.device) ? 8 : 4) || ctx.device === 'CLANG' ? `[${x.arg[0]}]` : `.${'xyzwabcd'[x.arg[0]]}`)],
])

export const extra_pm = new PatternMatcher<Record<string, UOp>, UOp | undefined>([
  // insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  [new UPat(Ops.BITCAST).named('x'), ({ x }) => x.src[0].op !== Ops.NOOP ? new UOp(Ops.BITCAST, x.dtype, [new UOp(Ops.NOOP, x.src[0].dtype, x.src)]) : undefined],
  // rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  [new UPat(Ops.MAX).named('m'), ({ m }) => (m.src[0].lt(m.src[1])).where(m.src[1], m.src[0])],
])

export const uops_to_dtypes = (uops: UOp[]): DType[] => dedup(uops.filter((u) => !(u.dtype instanceof ImageDType || u.dtype instanceof PtrDType)).map((u) => u.dtype))

type RenderKernelArgs = { functionName: string; kernel: string[]; bufs: [string, [DType, boolean]][]; uops: UOp[]; prefix?: string[] }
const root_render_kernel = (self: CStyleLanguage, { bufs, functionName, kernel, uops, prefix }: RenderKernelArgs): string => {
  const tmp = bufs.some(([_, [dtype]]) => dtype instanceof ImageDType) ? 'const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n' : ''
  const buftypes = bufs.map(([name, [dtype, mutable]]) => [name, (dtype instanceof ImageDType || dtype instanceof PtrDType) ? self.render_dtype(dtype, mutable) + self.buffer_suffix : dtype === dtypes.int ? self.arg_int_prefix : undefined])
  const prg = [`${self.kernel_prefix}void ${self.get_kernel_modifier(uops)}${functionName}(`, ...buftypes.map(([name, t]) => `${t} ${name}`).join(', '), ...self.extra_args.join(', '), ') {\n' + tmp, kernel.join('\n'), '\n}'].join('')
  return isNone(prefix) ? prg : `${prefix.join('\n')}\n${prg}`
}

export class CStyleLanguage extends Renderer {
  kernel_prefix = ''
  buffer_prefix = ''
  buffer_suffix = ''
  smem_align = ''
  smem_prefix = ''
  smem_prefix_for_cast = true
  arg_int_prefix = 'const int'
  barrier = ''
  code_for_workitem: Record<'g' | 'l' | 'i', (...x: any[]) => string> = {} as any
  extra_args: string[] = []
  float4?: string
  type_map: Map<DType, string> = new Map()
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
  render_kernel = (args: RenderKernelArgs) => root_render_kernel(this, args)
  render_cast = (dt: DType, val: string): string => `(${this.render_dtype(dt)})(${val})`
  render_dtype = (dt: DType, mutable = true): string => {
    if (dt instanceof ImageDType) return `${mutable ? 'write_only' : 'read_only'} image2d_t`
    if (dt instanceof PtrDType) return (dt.local && this.smem_prefix_for_cast ? this.smem_prefix : this.buffer_prefix) + this.render_dtype(dt.base) + (dt instanceof PtrDType ? '*' : '')
    const scalar = dt.scalar()
    return (this.type_map.get(scalar) || scalar.name) + ((dt.count) > 1 ? dt.count.toString() : '')
  }
  get = (key: UOp) => this.r?.get(key) // hacky helper
  override render = (name: string, uops: UOp[]): string => {
    const r = new Map<UOp, string>()
    this.r = r

    // KAREL: it didn't seem to do anything
    // const child_count = Counter(uops.flatMap((ru) => ru.src.map((v) => v)))
    const bufs = new Map<UOp, [string, [DType, boolean]]>()
    const kernel = []
    let depth = 1
    const c = new Map<string, number>()
    for (const u of uops) {
      if ([Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR].includes(u.op)) {
        r.set(u, u.op === Ops.DEFINE_GLOBAL ? `data${u.arg}` : u.arg[0])
        bufs.set(u, [r.get(u)!, [u.dtype, false]])
        continue
      }

      // // mark buffers that we store to writable
      if (u.op === Ops.STORE) {
        for (const up of u.src[0].toposort) {
          if (up.op === Ops.DEFINE_GLOBAL) bufs.set(up, [bufs.get(up)![0], [bufs.get(up)![1][0], true]])
        }
      }
      // // naming
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

        r.set(u, `${prefix}${setDefault(c, prefix, 0)}`)
      }
      let l = this.string_rewrite.rewrite(u, this)
      assert(isNotNone(l), `failed to render ${u.op} ${u.dtype} ${u.src.map((x) => [x.op, x.dtype])} ${u.arg}`)

      if ([Ops.ENDIF, Ops.ENDRANGE].includes(u.op)) depth -= 1
      if ([Ops.CONST, Ops.GEP, Ops.INDEX].includes(u.op) || ([Ops.VECTORIZE, ...GroupOp.ALU, Ops.CAST, Ops.BITCAST].includes(u.op) && !get_env('EXPAND_SSA'))) {
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
    delete this.r
    //  NOTE: this relies on bufs dict preserving order
    return this.render_kernel({ functionName: name, kernel, bufs: [...bufs.values()], uops })
  }
}

export class ClangRenderer extends CStyleLanguage {
  override device: DeviceType = 'CLANG'
  override float4 = '(float4)'
  override has_local = false
  override global_max = undefined
  override infinity = '__builtin_inff()'
  override nan = '__builtin_nanf("")'

  // language options
  override buffer_suffix = ' restrict'
  override type_map = new Map([[dtypes.bool, '_Bool'], [dtypes.half, '__fp16']])
  override code_for_op = new Map<Ops, (...args: (string | DType)[]) => string>([
    ...new CStyleLanguage().code_for_op.entries().filter(([k, v]) => ![Ops.EXP2, Ops.SIN, Ops.LOG2].includes(k)),
    [Ops.SQRT, (x: any, dtype: any) => dtype === dtypes.float64 ? `__builtin_sqrt(${x})` : `__builtin_sqrtf(${x})`],
  ])
  override tensor_cores = !AMX ? undefined : [dtypes.float].map((dt) => [dt, Math.floor(64 / dt.itemsize)] as const).map(([dt, sz]) => new TensorCore([sz, sz, 1], dt, dt, [], [], [[[1, sz]], [[0, sz]], [[1, sz], [0, sz]]]))

  render_vector_prefix = (dt: DType): string => `typedef ${this.render_dtype(dt.scalar())} ${this.render_dtype(dt)} __attribute__((aligned(${dt.itemsize}),vector_size(${dt.itemsize})));`

  override render_kernel = ({ bufs, functionName, kernel, uops, prefix }: RenderKernelArgs): string => {
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
        `${out} __$${this.render_dtype(dtypeIn.vec(N))} data1, ${this.render_dtype(dtypeIn.vec(M))} data2, ${out} data0){{ AMX_SET(0);\n  for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(4, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }} AMX(0, (int *)(&data2), 0ull<<62); AMX(1, (int *)(&data1), 0ull<<62); AMX(12, 0, 0ull); for(int ridx0 = 0; ridx0 < 16; ridx0++){{ AMX(5, (int *)(&data0), 0ull<<62 | (ridx0*4ull)<<56 | ridx0*64ull); }}\n  AMX_SET(1);\n  return data0;\n}}`,
      ]
    }
    return root_render_kernel(this, { functionName, kernel, bufs, uops, prefix })
  }
}
