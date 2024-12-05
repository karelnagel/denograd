import { type DType, dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { AMX, assert, dedup, getEnv, isNone, isNotNone, stripParens } from '../helpers.ts'
import { GroupOp, Ops, PatternMatcher, UOp, UPat } from '../ops.ts'
import { Renderer, TensorCore } from './index.ts'

export const baseRewrite = new PatternMatcher<{ ctx: CStyleLanguage } & Record<string, UOp>, string | undefined>([
  [new UPat({ op: Ops.DEFINE_ACC, name: 'x' }), ({ ctx, x }) => ctx.get(x.src[0])],
  [new UPat({ op: Ops.ASSIGN, name: 'x' }), ({ ctx, x }) => `${ctx.get(x.src[0])} = ${ctx.get(x.src[1])};`],
  [new UPat({ op: Ops.IF, name: 'x' }), ({ ctx, x }) => `if (${ctx.get(x.src[0])}) {`],
  [new UPat({ op: [Ops.ENDIF, Ops.ENDRANGE] }), ({ ctx }) => '}'],
  [new UPat({ op: Ops.WMMA, name: 'x' }), ({ ctx, x }) => `__${x.arg[0]}(${ctx.get(x.src[0])}, ${ctx.get(x.src[1])}, ${ctx.get(x.src[2])})`],
  // r method accesses
  [new UPat({ op: Ops.RANGE, name: 'x' }), ({ ctx, x }) => `for (${ctx.renderDType(x.dtype)} ${ctx.get(x)} = ${ctx.get(x.src[0])}; ${ctx.get(x)} < ${ctx.get(x.src[1])}; ${ctx.get(x)}++) {{`],
  [
    new UPat({ op: Ops.VECTORIZE, name: 'x' }),
    ({ ctx, x }) => `${ctx.float4!.replace('float4', ctx.renderDType(x.dtype))}` + (ctx.device === 'CLANG' ? `{${x.src.map((y: any) => ctx.get(y)).join(',')}}` : `(${x.src.map((y: any) => ctx.get(y)).join(',')})`),
  ],
  [new UPat({ op: Ops.CAST, name: 'x' }), ({ ctx, x }) => `(${ctx.renderCast(x.dtype, ctx.get(x.src[0])!)})`],
  [new UPat({ op: Ops.BITCAST, name: 'x' }), ({ ctx, x }) => `(*((${ctx.bufferPrefix}${ctx.renderDType(x.dtype)}*)&${ctx.get(x.src[0])}))`],
  [new UPat({ op: Ops.DEFINE_LOCAL, name: 'x' }), ({ ctx, x }) => `${ctx.smemAlign}${ctx.smemPrefix}${ctx.renderDType(x.dtype.base)} ${ctx.get(x)}[${x.arg[1]}];`],
  [new UPat({ op: Ops.BARRIER }), ({ ctx }) => ctx.barrier],
  [new UPat({ op: Ops.NOOP, name: 'x' }), ({ ctx, x }) => ctx.get(x.src[0])],
  [new UPat({ op: Ops.SPECIAL, name: 'x' }), ({ ctx, x }) => `${ctx.codeForWorkitem[x.arg[0][0] as keyof typeof ctx.codeForWorkitem](x.arg[0][-1])}; /* ${x.arg[1]} */`],
  // const
  [new UPat({ op: Ops.CONST, arg: Infinity, name: 'x' }), ({ ctx, x }) => `(${ctx.renderCast(x.dtype, ctx.infinity)})`],
  [new UPat({ op: Ops.CONST, arg: -Infinity, name: 'x' }), ({ ctx, x }) => `(${ctx.renderCast(x.dtype, `-${ctx.infinity}`)})`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.floats, name: 'x' }), ({ ctx, x }) => !isFinite(x.arg) ? `(${ctx.renderCast(x.dtype, ctx.nan)})` : undefined],
  [new UPat({ op: Ops.CONST, dtype: dtypes.float, name: 'x' }), ({ ctx, x }) => `${x.arg}f`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.int64, name: 'x' }), ({ ctx, x }) => `${x.arg}ll`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.uint64, name: 'x' }), ({ ctx, x }) => `${x.arg}ull`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.uint32, name: 'x' }), ({ ctx, x }) => `${x.arg}u`],
  [new UPat({ op: Ops.CONST, dtype: dtypes.bool, name: 'x' }), ({ ctx, x }) => x.arg ? '1' : '0'],
  // consts are rendered to larger type and casted
  [new UPat({ op: Ops.CONST, dtype: [dtypes.bfloat16, dtypes.half], name: 'x' }), ({ ctx, x }) => `(${ctx.renderCast(x.dtype, `${x.arg}f`)})`],
  [new UPat({ op: Ops.CONST, dtype: [dtypes.uint8, dtypes.uint16], name: 'x' }), ({ ctx, x }) => `(${ctx.renderCast(x.dtype, `${x.arg}u`)})`],
  [new UPat({ op: Ops.CONST, dtype: [dtypes.int8, dtypes.int16], name: 'x' }), ({ ctx, x }) => `(${ctx.renderCast(x.dtype, x.arg)})`],
  // default const render
  [new UPat({ op: Ops.CONST, name: 'x' }), ({ ctx, x }) => x.arg.toString()],
  // new load/store
  [new UPat({ op: Ops.INDEX, src: [UPat.var('buf'), UPat.var('idx')] }), ({ ctx, buf, idx }) => `(${ctx.get(buf)}+${idx.arg === Ops.ADD ? stripParens(ctx.get(idx)!) : ctx.get(idx)})`],
  [new UPat({ op: Ops.LOAD, src: [UPat.var('bidx'), UPat.var('var'), UPat.var('gate')] }), ({ ctx, bidx, var1, gate }) => `(${ctx.get(gate)}?*${ctx.get(bidx)}:${ctx.get(var1)})`],
  [new UPat({ op: Ops.LOAD, src: [UPat.var('bidx')], allowAnyLen: true }), ({ ctx, bidx }) => `*${ctx.get(bidx)}`],
  [new UPat({ op: Ops.STORE, src: [UPat.var('bidx'), UPat.var('var')], allowAnyLen: true }), ({ ctx, bidx, var1 }) => `*${ctx.get(bidx)} = ${ctx.get(var1)};`],
  // alu/gep
  [new UPat({ op: GroupOp.ALU, name: 'x' }), ({ ctx, x }) => ctx.codeForOp[x.op]!(...x.src.map((v) => v.op === x.op && [Ops.ADD, Ops.MUL, Ops.XOR].includes(x.op) ? stripParens(ctx.get(v)!) : ctx.get(v)!), x.dtype)],
  [new UPat({ op: Ops.GEP, name: 'x' }), ({ ctx, x }) => ctx.get(x.src[0]) + (x.src[0].dtype.count > (['CUDA', 'NV'].includes(ctx.device) ? 8 : 4) || ctx.device == 'CLANG' ? `[${x.arg[0]}]` : `.${'xyzwabcd'[x.arg[0]]}`)],
])

export const extraPm = new PatternMatcher<Record<string, UOp>, UOp | undefined>([
  // insert a NOOP before BITCAST to force it to be rendered. not needed on all backends?
  [new UPat({ op: Ops.BITCAST, name: 'x' }), ({ x }) => x.src[0].op !== Ops.NOOP ? new UOp({ op: Ops.BITCAST, dtype: x.dtype, src: [new UOp({ op: Ops.NOOP, dtype: x.src[0].dtype, src: x.src })] }) : undefined],
  // gate any stores that aren't gated with ifs
  [
    new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [new UPat({}), new UPat({}), new UPat({ dtype: dtypes.bool })], name: 'store' }),
    ({ store }) => new UOp({ op: Ops.STORE, src: [...store.src.slice(0, -2), new UOp({ op: Ops.IF, src: [store.src[2]] })] }),
  ],
  // rewrite MAX to CMPLT + WHERE (max function is annoying on many cstyle backends)
  [new UPat({ op: Ops.MAX, name: 'm' }), ({ m }) => (m.src[0].lt(m.src[1])).where(m.src[1], m.src[0])],
])

export const uopsToDTypes = (uops: UOp[]): DType[] => dedup(uops.filter((u) => !(u.dtype instanceof ImageDType || u.dtype instanceof PtrDType)).map((u) => u.dtype))

type RenderKernelArgs = { functionName: string; kernel: string[]; bufs: [string, [DType, boolean]][]; uops: UOp[]; prefix?: string[] }
const renderKernel = (self: CStyleLanguage, { bufs, functionName, kernel, uops, prefix }: RenderKernelArgs): string => {
  const tmp = bufs.some(([_, [dtype]]) => dtype instanceof ImageDType) ? 'const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n' : ''
  const buftypes = bufs.map(([name, [dtype, mutable]]) => [name, dtype == dtypes.int ? (dtype instanceof ImageDType || dtype instanceof PtrDType) ? self.renderDType(dtype, mutable) + self.bufferSuffix : self.argIntPrefix : undefined])

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
  static override codeForOp: { [key in Ops]?: (...args: (string | DType)[]) => string } = {
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
  codeForOp = CStyleLanguage.codeForOp
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
  get = (key: UOp) => this.r?.get(key) // hacky helper
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
  override codeForOp = CStyleLanguage.codeForOp
  tensorCores = !AMX
    ? undefined
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
