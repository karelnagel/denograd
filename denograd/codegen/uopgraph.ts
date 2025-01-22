import { dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { all_same, AMX, assert, cache_fn, DEBUG, dedup, flatten, get_env, is_eq, isinstance, partition, range, set_default, slice, TRANSCENDENTAL, zip } from '../helpers.ts'
import { sub } from '../mod.ts'
import { add, div, graph_rewrite, GroupOp, idiv, is_increasing, mul, Ops, parse_valid, type PatternFn, PatternMatcher, prod, simplify_valid, type sint, split_uop, symbolic_flat, symbolic_simple, UOp, uop_given_valid, UPat } from '../ops.ts'
import type { Renderer } from '../renderer/index.ts'
import { TRANSCENDENTAL_SUPPORTED_DTYPES, xexp2, xlog2, xsin } from './transcendental.ts'

// if TYPE_CHECKING: from tinygrad.renderer import Renderer

// # ***** float4/image store handling *****

export const fold_expanded = (ex: UOp, buf: UOp) => {
  if (buf.dtype.base !== dtypes.float && buf.dtype.base !== dtypes.half && !isinstance(buf.dtype, ImageDType)) return undefined
  let new_srcs: (UOp | undefined)[] = dedup([...ex.src])
  const old_new_srcs = [...new_srcs]
  const [is_load, is_image] = [new_srcs[0]?.op === Ops.LOAD, isinstance(buf.dtype, ImageDType)]

  //   # first, extract all the relevant offsets
  const offsets_rootsrc = new Map<UOp, Map<string, number>>()
  for (const [i, s] of new_srcs.entries()) {
    const idx = s!.src[0].src[1]
    let root_src: any, arg
    if (s!.dtype.count !== 1 || (is_image && idx.dtype.count === 2)) continue
    if (idx.op === Ops.ADD && idx.src[1].op === Ops.CONST) [root_src, arg] = [idx.src[0], idx.src[1].arg]
    else if (idx.op === Ops.CONST) [root_src, arg] = ['CONST', idx.arg]
    else [root_src, arg] = [idx, 0]
    //     # add gates for gated
    if (s!.src[0].src.length === 3) root_src = [s!.src[0].src[2], root_src]
    if (set_default(offsets_rootsrc, root_src, new Map()).has(arg)) throw new Error(`${offsets_rootsrc.get(root_src)!.get(arg)} != ${i} with ${s?.src.length} sources`)
    offsets_rootsrc.get(root_src)!.set(arg, i)
  }
  //   # then rewrite everything we can
  const lengths = is_image ? [4] : (buf.dtype.base === dtypes.half && get_env('ALLOW_HALF8') ? [8, 4, 2] : (AMX ? [16, 8, 4, 2] : [4, 2]))
  let used: [UOp, any][] = []
  for (const [rootsrc, offsets] of offsets_rootsrc.entries()) {
    for (const o of offsets.keys()) {
      for (const fold_length of lengths) {
        if (range(fold_length).every((i) => !used.some(([a, b]) => a === rootsrc && b === (o + i)) && offsets.has(o + i))) {
          const load_1 = new_srcs[offsets.get(o)!]!
          const new_src = [...load_1.src]
          const oidx = new_src[0].src[1]
          if (oidx.divides(fold_length) === undefined) continue
          if (is_image) {
            // for images, we rewrite the index. it must evenly divide 4 from the above check
            new_src[0] = buf.index(
              new UOp(Ops.VECTORIZE, dtypes.int.vec(2), [oidx.idiv(4).mod((buf.dtype as ImageDType).shape[1]), oidx.idiv(4 * (buf.dtype as ImageDType).shape[1])]),
              isinstance(rootsrc, Array) ? rootsrc[0] as UOp : undefined,
            )
          } else {
            // for non image, we upcast the index pointer
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr(idiv((new_src[0].dtype as PtrDType).size, fold_length), (new_src[0].dtype as PtrDType).local))
          }
          //           # generate the folded new_srcs
          if (is_load) {
            const new_load = new UOp(Ops.LOAD, load_1!.dtype.vec(fold_length), new_src)
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = new_load.gep(i)
          } else { // vectorize the store
            new_src[1] = new UOp(Ops.VECTORIZE, new_src[1].dtype.vec(fold_length), range(fold_length).map((i) => new_srcs[offsets.get(o + i)!]!.src[1]))
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = i === 0 ? new UOp(Ops.STORE, dtypes.void, new_src) : undefined
          }
          used = [...used, ...range(fold_length).map((i) => [rootsrc, o + i as any] as [UOp, UOp])]
        }
      }
    }
  }
  //   # dedup expand for LOAD
  if (is_load && old_new_srcs.length !== ex.src.length) new_srcs = ex.src.map((s) => new_srcs[old_new_srcs.indexOf(s)])
  //   # remove Nones for STORE
  return used.length ? new UOp(ex.op, ex.dtype, [...new_srcs.filter((x) => x !== undefined)], ex.arg) : undefined
}

export const fix_unfoldable_image_load = (load: UOp, buf: UOp) => {
  const oidx = load.src[0].src[1]
  if (!isinstance(buf.dtype, ImageDType) || oidx.dtype.count === 2) return undefined
  const id4 = oidx.mod(4)
  const new_src = [...load.src]
  //   # TODO: copied logic from above
  new_src[0] = load.src[0].src[0].index(
    new UOp(Ops.VECTORIZE, dtypes.int.vec(2), [(oidx.idiv(4)).mod(buf.dtype.shape[1]), oidx.idiv(4 * buf.dtype.shape[1])]),
    load.src[0].src.length === 3 ? load.src[0].src[2] : undefined,
  )
  const vec_load = new UOp(Ops.LOAD, load.dtype.vec(4), [...new_src])
  return range(4).reduce((ret, i) => id4.ne(i).where(ret, vec_load.gep(i)), load.const_like(NaN))
}

export const buf_idx_pat = new UPat(Ops.INDEX, undefined, [UPat.var('buf')], undefined, undefined, true)
export const float4_folding = new PatternMatcher([
  [new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.LOAD, undefined, [buf_idx_pat], undefined, undefined, true)).named('ex'), ({ ex, buf }) => fold_expanded(ex, buf)],
  [new UPat([Ops.BARRIER, Ops.SINK], undefined, new UPat(Ops.STORE, undefined, [buf_idx_pat], undefined, undefined, true)).named('ex'), ({ ex, buf }) => fold_expanded(ex, buf)],
])

// # ***** image load valid simplification *****

export const simplify_valid_load = (buf: UOp, start_idx: UOp, valid: UOp): undefined | UOp => {
  const idx = uop_given_valid(valid, start_idx)
  if (idx === undefined) return buf.const_like(0)
  if (!isinstance(buf.dtype, ImageDType)) return idx === start_idx ? undefined : buf.index(idx, valid)

  // wait for it to be image indexed before running simplification
  if (start_idx.dtype.count !== 2) return undefined

  // can drop valid if idx is out of bound when valid is False
  let drop_stmt = []
  for (const stmt of split_uop(valid, Ops.AND)) {
    const [X, is_upper_bound, c] = parse_valid(stmt)

    // for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
    if (!is_upper_bound && c === 1 && split_uop(X, Ops.ADD).every((u) => GroupOp.Irreducible.includes(u.op) && u.vmin === 0)) {
      const testidx = split_uop(X, Ops.ADD).reduce((nowidx, u) => nowidx.substitute(new Map([[u, u.const_like(0)]])), idx).simplify()
      if (testidx.gep(0).vmax < 0 || testidx.gep(1).vmax < 0) {
        drop_stmt.push(stmt)
        continue
      }
    }
    // if X <= c, check if it's out of bound when X = c+1
    // if X >= c, check if it's out of bound when X = c-1
    const test_value = is_upper_bound ? c + 1 : c - 1
    for (const [i, b] of zip(idx.src, [buf.dtype.shape[1], buf.dtype.shape[0]])) {
      if (is_increasing(i)) {
        const rw = i.substitute(new Map([[X, X.const_like(test_value)]])).simplify()
        if (rw.vmin >= b || rw.vmax < 0) {
          drop_stmt.push(stmt)
          break
        }
      }
    }
  }
  if (!drop_stmt && idx === start_idx) return undefined
  const ss = [...split_uop(valid, Ops.AND)].filter((s) => !drop_stmt.includes(s))
  const new_valid = ss.length ? ss.reduce((acc, s) => acc.add(s)) : undefined
  return buf.index(idx, new_valid)
}
// # ***** optional patterns *****

const powers_of_two = new Map(range(64).map((i) => [2 ** i, i]))
type Pat = [UPat, PatternFn]
export const get_late_rewrite_patterns = cache_fn((ops: Ops[], force_transcendental = false) => {
  let pat: Pat[] = ([[Ops.EXP2, xexp2], [Ops.LOG2, xlog2], [Ops.SIN, xsin]] as const).filter(([op, f]) => !ops.includes(op) || force_transcendental)
    .map(([op, f]) => [new UPat(op, TRANSCENDENTAL_SUPPORTED_DTYPES, [UPat.var('d')]), (x) => f(x as any)])
  // rewrite MOD to AND (which should always be supported, but not for generic in tests): x % (2**y) -> x & (2**y-1)
  if (ops.includes(Ops.AND)) {
    pat = [...pat, [UPat.var('x', dtypes.ints).mod(UPat.cvar('c')), ({ x, c }) => powers_of_two.has(c.arg) ? x.bitwise_and(sub(c.arg, 1)) : undefined] satisfies Pat]
  }
  // rewrite MUL/IDIV to SHL+SHR: x*(2**y) -> shl(x,y) and x//(2**y) -> shr(x,y)
  if (ops.includes(Ops.SHL) && ops.includes(Ops.SHR)) {
    pat = [
      ...pat,
      [UPat.var('x', dtypes.ints).mul(UPat.cvar('c')), ({ c, x }) => powers_of_two.has(c.arg) ? x.lshift(powers_of_two.get(c.arg)!) : undefined],
      [UPat.var('x', dtypes.ints).idiv(UPat.cvar('c')), ({ x, c }) => powers_of_two.has(c.arg) ? x.rshift(powers_of_two.get(c.arg)!) : undefined],
    ]
  }
  if (ops.includes(Ops.NEG)) {
    pat = [...pat, [UPat.var('x').mul(-1), ({ x }) => x.alu(Ops.NEG)]]
    if (ops.includes(Ops.SUB)) pat = [...pat, [UPat.var('x').add(UPat.var('y').alu(Ops.NEG)), ({ x, y }) => x.alu(Ops.SUB, y)]]
  }
  if (ops.includes(Ops.MULACC)) {
    pat = [...pat, [UPat.var('a').mul(UPat.var('b')).add(UPat.var('c')), ({ a, b, c }) => a.alu(Ops.MULACC, b, c)]]
  }
  return new PatternMatcher(pat)
})
// # ***** threefry *****

export const threefry2x32 = (x: UOp, key: UOp) => {
  //   # split x into two uint32, since x in a uint64
  const [x0, x1] = [(x.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((x.idiv(2 ** 32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]

  const rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  const [key0, key1] = [(key.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((key.idiv(2 ** 32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]
  const ks = [key1, key0.xor(key1).xor(0x1BD11BDA), key0]
  let xr = [x0.add(ks.at(-1)!), x1.add(ks[0])]
  for (const i of range(5)) {
    for (const r of rotations[i % 2]) {
      const x0 = xr[0].add(xr[1])
      ;[xr[0], xr[1]] = [x0, x0.xor(xr[1].mul(2 ** r).add(xr[1].idiv(2 ** (32 - r))))]
    }
    xr = [xr[0].add(ks[i % 3]), xr[1].add(ks[(i + 1) % 3]).add(i).add(1)]
  }
  return xr[1].cast(dtypes.uint64).mul(2n ** 32n).bitwise_or(xr[0].cast(dtypes.uint64))
}

// ***** other math rewrite ****

export const sigmoid_like = (x: UOp, y: UOp) => {
  const t = div(1, add(x, 1))
  return t.mul(sub(1, t)).mul(y)
}

// # ***** main rewriter *****

export const loop_collapse = (compval: any, multconst: any, rng: UOp, acc: UOp, idx2?: any, idx3?: any, extra?: any, vec?: any, ne?: any, add = UOp.const(dtypes.int, 0), mul = UOp.const(dtypes.int, 1)) => {
  if (get_env('DISABLE_LOOP_COLLAPSE') || !acc.src.includes(rng)) return undefined // must be the right REDUCE
  let [loop_start, loop_end] = rng.src
  if (loop_start.arg !== 0) {
    //     # TODO: support and test this with other mul and loop_starts
    if (DEBUG >= 1) console.log(`WARNING, NOT FOLDING: mul:${mul.arg} loop_start:${loop_start.arg}`)
    return undefined
  }
  if (idx2 !== undefined) add = add + idx2
  if (idx3 !== undefined) add = add + idx3
  if (vec !== undefined) {
    //     # add, mul, loop_start, loop_end
    const dvec = (x: UOp) => {
      if (x.op === Ops.CONST) return UOp.const(x.dtype.vec(vec.dtype.count), x.arg)
      return new UOp(Ops.VECTORIZE, x.dtype.vec(vec.dtype.count), range(vec.dtype.count).map(() => x))
    }
    ;[add, mul, loop_start, loop_end] = [dvec(add), dvec(mul), dvec(loop_start), dvec(loop_end)]
  }

  let comprange
  if (mul.vmin > 0 && ne !== undefined) {
    comprange = loop_end.minimum(add.sub(compval)).idiv(mul).add(loop_end.sub(loop_start).maximum(loop_start))
  } else if (mul.vmax < 0 && ne === undefined) comprange = loop_end.minimum(add.sub(compval).sub(mul)).idiv(mul).add(loop_end.sub(loop_start).maximum(loop_start))
  else return undefined
  const new_reduce_op = comprange.cast(multconst.dtype).mul(multconst)
  //   # TODO: what does it mean to have the same numbered DEFINE_ACC with different ranges?
  const new_acc = acc.replace({ src: [acc.src[1], ...acc.src.slice(1).filter((x) => x !== rng)] })
  let ret = new_acc.assign(new_acc.add(new_reduce_op))
  if (extra !== undefined) ret = ret.add(acc.assign(acc.add(extra)))
  //   return ret
}

export const index_collapse = (idx: UOp, rng: UOp, buf: UOp, ld: UOp, acc: UOp, add = UOp.const(dtypes.int, 0), mul = UOp.const(dtypes.int, 1)) => {
  if (!acc.src.includes(rng)) return undefined
  const new_load = buf.index(add.add(mul.mul(idx)), idx.ge(rng.src[0]).bitwise_and(idx.lt(rng.src[1]))).load([], { dtype: ld.dtype })
  const new_acc = acc.replace({ src: [acc.src[0], ...acc.src.slice(1).filter((x) => x !== rng)] })
  return new_acc.assign(new_acc.add(new_load))
}
// TODO: there's a lot shared with no_vectorized_wmma here
export const gep_through_wmma = (gep: UOp, wmma: UOp) => {
  const out_sz: number = prod(wmma.arg[6].at(-1)!.map((x: number[]) => x[1]))
  const wmma_idxs: number[] = slice(gep.arg, { step: out_sz })
  for (const i of range(out_sz)) {
    if (!is_eq(slice(gep.arg, { start: i, step: out_sz }).map((x: any) => sub(x, i)), wmma_idxs)) return undefined
  }
  const tsrcs: UOp[] = []
  for (const [s, sz] of zip(wmma.src, wmma.arg[6] as number[][][])) {
    let src_args: number[] = []
    const ssz = prod(sz.map((x) => x[1]))
    for (const w of wmma_idxs) src_args = [...src_args, ...range(mul(idiv(w, out_sz), ssz), add(mul(idiv(w, out_sz), ssz), ssz))]
    tsrcs.push(s.gep(src_args))
  }
  return new UOp(Ops.WMMA, gep.dtype, [...tsrcs], wmma.arg)
}
export const no_vectorized_wmma = (wmma: UOp) => {
  const out_sz: number = prod(wmma.arg[6].at(-1)!.map((x: any) => x[1]))
  if (wmma.dtype.count === out_sz) return undefined
  let tsrcs: UOp[][] = []
  for (const [s, sz] of zip(wmma.src, wmma.arg[6] as number[][][])) {
    const ssz = prod(sz.map((x) => x[1]))
    tsrcs.push(range(0, s.dtype.count, ssz).map((grp) => s.gep(range(grp, grp + ssz))))
  }
  const wmmas = zip(...tsrcs).map((tsrc) => new UOp(Ops.WMMA, wmma.dtype.scalar().vec(out_sz), tsrc, wmma.arg))
  const wmma_ex = flatten(wmmas.map((e) => range(out_sz).map((i) => e.gep(i))))
  return new UOp(Ops.VECTORIZE, wmma.dtype, wmma_ex)
}
export const reduce_collapse = (acc: UOp, ret: UOp, alu: UOp) => {
  const [reduce_parented, reduce_unparented] = partition(acc.src.slice(1), (x) => ret.toposort.has(x))
  if (reduce_unparented.length === 0) return undefined
  const new_acc = acc.replace({ src: [acc.src[0], ...reduce_parented] })
  ret = new_acc.assign(new_acc.alu(alu.op, ret))
  if (alu.op === Ops.ADD) {
    for (const r of reduce_unparented) ret = ret.mul((r.src[1].sub(r.src[0])).cast(ret.dtype.scalar()).broadcast(ret.dtype.count))
  }
  return ret
}
export const [acc_pat, rng_pat] = [new UPat(Ops.DEFINE_ACC).named('acc'), new UPat(Ops.RANGE).named('rng')]
export const rng_aug = UPat.any([rng_pat, UPat.var('add').add(rng_pat), UPat.var('mul').mul(rng_pat), UPat.var('add').add(UPat.var('mul').mul(rng_pat))])

export const index_load = UPat.var('buf').index(rng_aug).load(undefined, { name: 'ld' })

export const arange_augrng = UPat.any([rng_aug, rng_aug.add(UPat.var('idx2')), rng_aug.add(UPat.var('idx2')).add(UPat.var('idx3')), new UPat(Ops.VECTORIZE, undefined, rng_aug, undefined, 'vec')])
export const arange_m = ((arange_augrng.lt(UPat.cvar('compval'))).ne(new UPat(Ops.CONST, undefined, undefined, true, 'ne'))).where(UPat.cvar('multconst'), UPat.const(undefined, 0))

// # this is symbolic 2.0
export const sym = symbolic_flat.add(
  new PatternMatcher([
    //   # self ASSIGN is just self
    [new UPat(Ops.ASSIGN, undefined, [UPat.var('x'), UPat.var('x')]), ({ x }) => x],
    //   # VECTORIZE/CONST, VECTORIZE/GEP
    [new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.CONST), undefined, 'vec'), ({ vec }) => UOp.const(vec.dtype, vec.src.map((x) => x.arg))],
    [new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.GEP, undefined, [new UPat(undefined).named('x')]), undefined, 'vec'), ({ vec, x }) => x.gep(vec.src.map((y) => y.arg[0]))],
    //   # reorder ALU/VECTORIZE
    [
      new UPat(GroupOp.ALU, undefined, [new UPat(Ops.VECTORIZE, undefined, new UPat(undefined).named('x')), new UPat(Ops.VECTORIZE, undefined, new UPat(undefined).named('y'))], undefined, 'alu'),
      ({ x, y, alu }) => new UOp(Ops.VECTORIZE, alu.dtype, range(alu.dtype.count).map((i) => new UOp(alu.op, alu.dtype.scalar(), [x, y]))),
    ],
    //   # VECTORIZE of a single element is just that element
    [new UPat(Ops.VECTORIZE, undefined, [new UPat(undefined).named('x')]), ({ x }) => x],
    //   # VECTORIZE void is SINK
    [new UPat(Ops.VECTORIZE, dtypes.void, new UPat(Ops.BARRIER).named('b')), ({ b }) => b],
    [new UPat(Ops.VECTORIZE, dtypes.void).named('x'), ({ x }) => new UOp(Ops.SINK, dtypes.void, x.src)],
    //   # GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
    [new UPat(Ops.GEP, undefined, [new UPat(Ops.GEP).named('g2')], undefined, 'g1'), ({ g1, g2 }) => g2.src[0].gep(range(g1.dtype.count).map((i) => g2.arg[g1.arg[i]]))],
    [
      new UPat(Ops.GEP, undefined, [new UPat(Ops.VECTORIZE).named('vec')], undefined, 'gep'),
      ({ gep, vec }) => new UOp(Ops.VECTORIZE, gep.dtype, gep.arg.length > 1 ? gep.arg.map((i: number) => vec.src[i]) : vec.src[gep.arg[0]]),
    ],
    [new UPat(Ops.GEP, undefined, [UPat.cvar('c', undefined, false)], undefined, 'gep'), ({ gep, c }) => gep.const_like(c.arg)],
    [new UPat(Ops.GEP, undefined, [new UPat(Ops.VCONST).named('c')], undefined, 'gep'), ({ gep, c }) => gep.const_like(gep.arg.map((x: any) => c.arg[x]))],
    //   # push all GEPs through ALUs (fix arange stuff)
    [
      new UPat(Ops.GEP, undefined, [new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST]).named('alu')], undefined, 'gep'),
      ({ gep, alu }) => new UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), alu.src.map((x) => x.gep(gep.arg)), alu.arg),
    ],
    //   # push some GEPs through WMMAs
    [new UPat(Ops.GEP, undefined, [new UPat(Ops.WMMA).named('wmma')], undefined, 'gep'), ({ wmma, gep }) => gep_through_wmma(gep, wmma)],
    //   # tensor core with a 0 input is acc
    [new UPat(Ops.WMMA, undefined, [UPat.const(undefined, 0.0), UPat.var(), UPat.var('acc')]), ({ acc }) => acc],
    [new UPat(Ops.WMMA, undefined, [UPat.var(), UPat.const(undefined, 0.0), UPat.var('acc')]), ({ acc }) => acc],
    //   # tensor core cleanups
    [UPat.var('add').add(new UPat(Ops.WMMA).named('wmma')), ({ add, wmma }) => new UOp(wmma.op, wmma.dtype, [wmma.src[0], wmma.src[1], wmma.src[2].add(add)], wmma.arg)],
    //   # threefry + remove longs
    [new UPat(Ops.THREEFRY, dtypes.uint64, [UPat.var('x'), UPat.var('key')]), ({ x, key }) => threefry2x32(x, key)],
    [UPat.var('x', dtypes.uint32).cast(dtypes.uint64).cast(dtypes.uint32), ({ x }) => x], // cast there and back is noop (TODO: genericize)
    [(UPat.var('x', dtypes.uint64).bitwise_and(0xFFFFFFFF)).cast(dtypes.uint32), ({ x }) => x.cast(dtypes.uint32)], // cast does truncation
    [(UPat.var(undefined, dtypes.uint64).mul(1n << 32n).bitwise_or(UPat.var('y', dtypes.uint32).cast(dtypes.uint64))).cast(dtypes.uint32), ({ y }) => y],
    [((UPat.var('x', dtypes.uint64).mul(1n << 32n)).bitwise_or(UPat.var(undefined, dtypes.uint32).cast(dtypes.uint64))).idiv(1n << 32n), ({ x }) => x],
    //   # hacks for threefry long removal when padded (TODO: genericize)
    [UPat.var('x', dtypes.uint32).cast(dtypes.uint64).mul(UPat.var('y').where(UPat.const(dtypes.uint64, 1n << 32n), UPat.const(dtypes.uint64, 0))), ({ x, y }) => y.where(x, UOp.const(dtypes.uint32, 0)).cast(dtypes.uint64).mul(1n << 32n)],
    [(UPat.var('x', dtypes.uint64).bitwise_and(UPat.var('y').where(UPat.const(dtypes.uint64, 0xFFFFFFFF), UPat.const(dtypes.uint64, 0)))).cast(dtypes.uint32), ({ x, y }) => y.where(x.cast(dtypes.uint32), UOp.const(dtypes.uint32, 0))],
    // # arange loop folding
    [acc_pat.assign(UPat.any([arange_m, arange_m.add(UPat.var('extra'))]).add(acc_pat)), ({ compval, multconst, rng, acc, idx2, idx3, extra, vec, ne, add, mul }) => loop_collapse(compval, multconst, rng, acc, idx2, idx3, extra, vec, ne, add, mul)],
    // # indexing, with cast or where
    [acc_pat.assign(UPat.var('idx').eq(new UPat(Ops.RANGE).named('rng')).cast().mul(index_load).add(acc_pat)), ({ idx, rng, buf, ld, axx, add, mul }) => index_collapse(idx, rng, buf, ld, axx, add, mul)],
    [acc_pat.assign(UPat.var('idx').eq(new UPat(Ops.RANGE).named('rng')).where(index_load, UPat.const(undefined, 0.0)).add(acc_pat)), ({ idx, rng, buf, ld, acc, add, mul }) => index_collapse(idx, rng, buf, ld, acc, add, mul)],
    // # parentless reduce  # TODO: add MUL
    [acc_pat.assign(new UPat([Ops.ADD, Ops.MAX], undefined, [[acc_pat, UPat.var('ret')]], undefined, 'alu')), ({ acc, ret, alu }) => reduce_collapse(acc, ret, alu)],
    //   # ** self folding **
    [new UPat(Ops.DEFINE_ACC, undefined, [UPat.var('x')]), ({ x }) => x], // a DEFINE_ACC without ranges is a CONST
    [new UPat(Ops.ASSIGN, undefined, [UPat.cvar(), UPat.var('x')]), ({ x }) => x], // an ASSIGN to a const is a NOOP
    // # x!=0 -> (bool)x
    [UPat.var('x').ne(0), ({ x }) => x.cast(dtypes.bool.vec(x.dtype.count))],
    //   # ** load/store folding **
    [new UPat(Ops.INDEX).named('index').store([new UPat(Ops.INDEX).named('index').load()]), ({ index }) => new UOp(Ops.NOOP)],
    [new UPat(Ops.INDEX).named('index').store([UPat.var('gate').where(UPat.var('alt'), new UPat(Ops.INDEX).named('index').load())]), ({ index, gate, alt }) => index.src[0].index(index.src[1], gate).store([alt])],
    // # fold gated LOAD/STORE
    [new UPat().index(new UPat(), UPat.const(dtypes.bool, true)).named('idx'), ({ idx }) => idx.replace({ src: idx.src.slice(0, 2) })], // remove True
    [new UPat().index(new UPat(), UPat.const(dtypes.bool, false)).named('idx'), ({ idx }) => idx.const_like(0)], //False -> NULL pointer
    [new UPat(Ops.LOAD, undefined, [UPat.const(undefined, 0)], undefined, 'x', true), ({ x }) => x.const_like(0)], // NULL pointer load loads 0
    [new UPat(Ops.STORE, undefined, [UPat.const(undefined, 0)], undefined, undefined, true), () => new UOp(Ops.NOOP)], // NULL pointer store does nothing
    //   # remove NOOPs from SINK
    [new UPat(Ops.SINK).named('root'), ({ root }) => {
      const a = root.src.filter((x) => x.op !== Ops.NOOP)
      return a.length !== root.src.length ? new UOp(Ops.SINK, root.dtype, a, root.arg) : undefined
    }],
    //   # remove VECTORIZE from SINK/BARRIER
    [new UPat(Ops.BARRIER, undefined, [new UPat([Ops.VECTORIZE, Ops.SINK]).named('sink')]), ({ sink }) => new UOp(Ops.BARRIER, dtypes.void, sink.src)],
    [new UPat(Ops.SINK).named('root'), ({ root }) => root.src.some((x) => [Ops.SINK, Ops.UNROLL].includes(x.op)) ? new UOp(Ops.SINK, root.dtype, flatten(root.src.map((x) => [Ops.SINK, Ops.UNROLL].includes(x.op) ? x.src : [x])), root.arg) : undefined],
    // # stable sigmoid
    [UPat.var('x').mul(((UPat.var('x').add(1)).mul(UPat.var('x').add(1))).reciprocal()), ({ x }) => sigmoid_like(x, x.const_like(1))],
    [UPat.var('x').mul(((UPat.var('x').add(1)).mul(UPat.var('x').add(1))).reciprocal().mul(UPat.var('y'))), ({ x, y }) => sigmoid_like(x, y)],
    [UPat.var('x').mul(((UPat.var('x').add(1)).mul(UPat.var('x').add(1)).mul(UPat.var('x').add(1))).reciprocal()), ({ x }) => sigmoid_like(x, (x.add(1)).reciprocal())],
  ]),
)

// # *** uop expander ***

export const _expand_arg_to_idx = (args: [number, number][], rpk: Record<number, number>): number => {
  let [idx, mul] = [0, 1]
  for (const [axis, m] of args.toReversed()) {
    idx += rpk[axis] * mul
    mul *= m
  }
  return idx
}
export const _choices_from_args = (args: [number, number][]): Record<number, number>[] => {
  return args.reduce((acc, [axis, m]) => acc.flatMap((d) => range(m).map((i) => ({ ...d, [axis]: i }))), [{}])
}
export const _swizzle_args = cache_fn((cargs: [number, number][], eargs: [number, number][], exclude_args: number[]): number[] => {
  return _choices_from_args(cargs).map((rpk) => _expand_arg_to_idx(eargs, exclude_args ? { ...rpk, ...Object.fromEntries(exclude_args.map((x) => [x, 0])) } : rpk))
})
export const do_expand = (root: UOp) => {
  const expands = root.src.filter((x) => x.op === Ops.UNROLL)
  if (expands.length === 0) return undefined
  //   # NOTE: we 0 out the reduce axis for WMMA. in theory they should all be the same, but is this always correct?
  const exclude_args = root.op === Ops.WMMA ? dedup([...root.arg.at(-1)!, ...flatten(root.arg.at(-2)).map((y: any) => y[0])]) : []
  const expands_args = expands.map((x) => x.arg)
  let expand_args: [number, number][]
  if (all_same(expands_args) && exclude_args.length === 0) {
    //     # if there's only one expand arg, it's okay to use it (optimization)
    expand_args = expands[0].arg
  } // otherwise, we sort them and GEP
  else expand_args = dedup<[number, number]>(flatten(expands_args)).toSorted().filter((x) => !exclude_args.includes((x as any)[0]))
  const expand_sz = prod(expand_args.map((x) => x[1]))
  const new_srcs = []
  for (const [i, src] of root.src.entries()) {
    if (src.op === Ops.UNROLL) {
      //         # IF means OR on first arg to IF
      if (root.op === Ops.IF && i === 0) new_srcs.push(range(expand_sz).map((i) => src.src[0].gep(i)).reduce((acc, x) => acc.bitwise_or(x)))
      //         # just remove the expand
      else if (is_eq(expand_args, src.arg)) new_srcs.push(src.src[0])
      else {
        let lst = _swizzle_args(expand_args, src.arg, exclude_args)
        //         # if the base dtype is > 1, put those at the end
        if (src.dtype.count > 1) lst = lst.flatMap((i) => range(src.dtype.count).map((j) => i * src.dtype.count + j))
        new_srcs.push(src.src[0].gep([...lst]))
      }
    } else { // non-UNROLL input
      // for the first arg of IF, just pass them through ignoring UNROLLS
      if (root.op === Ops.IF) new_srcs.push(src)
      // put any input dtype > 1 grouped together
      else if (src.dtype.count > 1) new_srcs.push(new UOp(Ops.VECTORIZE, src.dtype.scalar().vec(expand_sz * src.dtype.count), range(expand_sz).flatMap(() => range(src.dtype.count).map((i) => src.gep(i)))))
      // repeat the arg
      else new_srcs.push(src.broadcast(expand_sz))
    }
  }
  let new_arg = root.arg
  if (root.op === Ops.GEP) {
    assert(root.dtype.count === 1)
    //     # is this right?
    new_arg = range(root.arg[0], new_srcs[0].dtype.count, idiv(new_srcs[0].dtype.count, expand_sz))
  }
  const nsrc = new UOp(root.op, root.dtype.scalar().vec(root.dtype.count * expand_sz), new_srcs, new_arg)
  return new UOp(Ops.UNROLL, root.dtype, [nsrc], expand_args)
}
export const do_contract = (con: UOp) => {
  const ex = con.src[0]
  //   # CONTRACT without UNROLL repeats the element VECTORIZED
  if (ex.op !== Ops.UNROLL) return new UOp(Ops.VECTORIZE, con.dtype, range(con.dtype.count).flatMap(() => con.src))
  //   # CONTRACT may remove several axes from UNROLL
  if (con.dtype.count !== prod(con.arg.map((x: any) => x[1]))) throw new Error('dtype is wrong')
  let idxs: number[] = []
  const new_ex_args = ex.arg.filter((x: any) => !con.arg.some((arg: any[]) => is_eq(arg, x)))
  for (const rpk of _choices_from_args(new_ex_args)) {
    idxs = [...idxs, ..._choices_from_args(con.arg).map((lrpk) => _expand_arg_to_idx(ex.arg, { ...rpk, ...lrpk }))]
  }
  return new UOp(Ops.UNROLL, con.dtype, [ex.src[0].gep([...idxs])], new_ex_args)
}
export const no_vectorized_alu = (alu: UOp) => {
  if (alu.dtype.vcount === 1) return undefined
  const alus = range(alu.dtype.vcount).map((i) => new UOp(alu.op, alu.dtype.scalar(), alu.src.map((s) => s.gep(i)), alu.arg))
  return new UOp(Ops.VECTORIZE, alu.dtype, alus)
}

const _gate_srcs = cache_fn((u: UOp, gate: UOp): UOp => {
  if (u.op === Ops.BARRIER) return u
  if (u.op === Ops.LOAD && u.src.at(-1)!.op === Ops.BARRIER) return new UOp(u.op, u.dtype, [...u.src.toReversed(), new UOp(Ops.IF, dtypes.void, [gate, u.src.at(-1)!])], u.arg)
  const replace_source = u.src.map((x) => _gate_srcs(x, gate))
  return is_eq(replace_source, u.src) ? u : new UOp(u.op, u.dtype, replace_source, u.arg)
})
export const create_gate = (root: UOp): undefined | UOp => {
  let idx = root.src[0]
  if (idx.op === Ops.CAST) idx = idx.src[0]
  const ret = _gate_srcs(root, idx.src[2])
  return idx.op !== Ops.INDEX || idx.src.length === 2 || ret === root ? undefined : ret
}
export const expander = new PatternMatcher([
  //   # double expand
  [new UPat(Ops.UNROLL, undefined, [new UPat(Ops.UNROLL).named('inner')], undefined, 'outer'), ({ outer, inner }) => new UOp(Ops.UNROLL, outer.dtype, [inner.src[0]], inner.arg + outer.arg)],
  //   # do expansion
  [new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WMMA, Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.ASSIGN, Ops.VECTORIZE, Ops.IF], undefined, undefined, undefined, 'root', undefined, undefined, [Ops.UNROLL]), ({ root }) => do_expand(root)],
  [new UPat(Ops.CONTRACT).named('con'), ({ con }) => do_contract(con)],
  //   # vectorize DEFINE_ACC
  [new UPat(Ops.VECTORIZE, undefined, new UPat(Ops.DEFINE_ACC).named('acc'), undefined, 'v'), ({ acc, v }) => acc.replace({ dtype: v.dtype })],
  //   # BARRIERs aren't actually expanded
  [
    new UPat(Ops.BARRIER, undefined, [new UPat(Ops.UNROLL).named('ex')]),
    ({ ex }) => new UOp(Ops.UNROLL, dtypes.void, range(ex.src.length).map((x) => new UOp(Ops.BARRIER, dtypes.void, ex.src)), ex.arg),
  ],
  //   # empty EXPAND is NOOP
  [new UPat(Ops.UNROLL, undefined, [UPat.var('x')], []), ({ x }) => x],
  //   # EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  [
    new UPat(Ops.UNROLL, undefined, range(AMX ? 256 : 8).map((i) => UPat.var('x').gep(i).add(UPat.var('y').gep(i))), undefined, 'ex'),
    ({ ex, x, y }) => new UOp(Ops.UNROLL, ex.dtype, range(AMX ? 256 : 8).map((i) => x.add(y).gep(i)), ex.arg),
  ],
])

export const no_vectorized_load_store = (ls: UOp) => {
  const idx = ls.src[0]
  if (!isinstance(idx.dtype, PtrDType)) throw new Error()
  if (idx.dtype.v === 1) return undefined
  const tv = range(idx.dtype.v).map((i) => new UOp(ls.op, ls.dtype.scalar(), ls.src.map((j) => j.gep(i))))
  return new UOp(Ops.VECTORIZE, ls.dtype, tv)
}
export const no_vectorized_acc = (acc: UOp) => {
  if (acc.dtype.count === 1) return undefined
  const alus = [new UOp(acc.op, acc.dtype.scalar(), range(acc.dtype.count).flatMap((i) => [...acc.src.entries()].map(([j, s]) => j === 0 ? s.gep(i) : s)))]
  return new UOp(Ops.VECTORIZE, acc.dtype, alus)
}
export const devectorize = new PatternMatcher([
  //   # no ALU on vectorized dtypes
  [new UPat([...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX]).named('alu'), ({ alu }) => no_vectorized_alu(alu)],
  [new UPat(Ops.WMMA).named('wmma'), ({ wmma }) => no_vectorized_wmma(wmma)],
  [new UPat(Ops.DEFINE_ACC).named('acc'), ({ acc }) => no_vectorized_acc(acc)],
  [new UPat([Ops.LOAD, Ops.STORE]).named('ls'), ({ ls }) => no_vectorized_load_store(ls)],
])

export const delete_redundant_gates = (buf: UOp, idx: UOp, val: UOp, store_gate: UOp, cast?: UOp): undefined | UOp => {
  if (![...val.toposort].filter((gate) => gate.op === Ops.IF).map((gate) => gate.src[0]).includes(store_gate)) return undefined
  //   # remove the gate from the index
  return (cast !== undefined ? buf.index(idx).cast(cast.dtype) : buf.index(idx)).store([val])
}
const _stidx = UPat.var('buf').index(UPat.var('idx'), UPat.var('store_gate'))
export const load_store_indexing = new PatternMatcher([
  //   # late fixup of unfoldable image loads
  [new UPat(Ops.LOAD, undefined, [UPat.var('buf'), new UPat()], undefined, 'load', true), ({ load, buf }) => fix_unfoldable_image_load(load, buf)],
  //   # simplify valid
  [new UPat(Ops.AND).named('valid'), ({ valid }) => simplify_valid(valid)],
  //   # image load valid idx simplification
  [new UPat(Ops.INDEX, undefined, [UPat.var('buf'), UPat.var('start_idx'), UPat.var('valid')]), ({ buf, start_idx, valid }) => simplify_valid_load(buf, start_idx, valid)],
  //   # delete_redundant_gates (after expand)
  [new UPat(Ops.STORE, undefined, [UPat.any([_stidx, _stidx.cast(undefined).named('cast')]), UPat.var('val')]), ({ buf, idx, val, store_gate, cast }) => delete_redundant_gates(buf, idx, val, store_gate, cast)],
])

export const migrate_indexing = new PatternMatcher([
  //   # create gate MUST BE BEFORE expander
  [new UPat(Ops.STORE).named('root'), ({ root }) => create_gate(root)],
])

export const move_mask = (x: UOp, buf: UOp, idx: UOp, mask: UOp, cast?: UOp): UOp => {
  //   # this moves the mask from the indexing to the load/store op for rendering
  const nidx = cast !== undefined ? buf.index(idx).cast(cast.dtype) : buf.index(idx)
  return x.op === Ops.LOAD ? nidx.load([x.const_like(0), mask, ...x.src.slice(1)], { dtype: x.dtype }) : nidx.store([x.src[1], mask, ...x.src.slice(2)])
}
const _masked_index = new UPat(Ops.INDEX, undefined, [new UPat(undefined).named('buf'), new UPat(undefined).named('idx'), new UPat(undefined).named('mask')])
export const pm_render = new PatternMatcher([
  //   # for rendering, we use explicit VECTORIZE
  [new UPat(Ops.CONST).named('c'), ({ c }) => c.dtype.vcount > 1 ? new UOp(Ops.VECTORIZE, c.dtype, range(c.dtype.vcount).map(() => UOp.const(c.dtype.scalar(), c.arg))) : undefined],
  [new UPat(Ops.VCONST).named('c'), ({ c }) => new UOp(Ops.VECTORIZE, c.dtype, c.arg.map((x: number) => UOp.const(c.dtype.scalar(), x)))],
  [new UPat(Ops.GEP).named('gep'), ({ gep }) => new UOp(Ops.VECTORIZE, gep.dtype, gep.arg.length > 1 ? gep.arg.map((x: number) => gep.src[0].gep(x)) : undefined)],
  [new UPat(Ops.VECTORIZE, undefined, [new UPat(undefined).named('x')]), ({ x }) => x],
  //   # move masks of loads/stores
  [
    new UPat([Ops.LOAD, Ops.STORE], undefined, [UPat.any([_masked_index, _masked_index.cast(undefined).named('cast')])], undefined, 'x', true),
    ({ x, buf, idx, mask, cast }) => move_mask(x, buf, idx, mask, cast),
  ],
  //   # gate any stores that aren't gated with ifs
  [
    new UPat(Ops.STORE, dtypes.void, [new UPat(), new UPat(), new UPat(undefined, dtypes.bool)], undefined, 'store'),
    ({ store }) => new UOp(Ops.STORE, undefined, [...store.src.slice(0, 2), new UOp(Ops.IF, undefined, [store.src[2]])]),
  ],
])

// # *** uop graph ***

export const full_graph_rewrite = (sink: UOp, opts?: Renderer): UOp => {
  if (sink.op !== Ops.SINK) throw new Error(`sink isn't sink, it's ${sink.op}`)
  const supported_ops = opts !== undefined ? [...opts.code_for_op.keys()] : []
  const extra_matcher = opts !== undefined && opts.extra_matcher !== undefined ? opts.extra_matcher : new PatternMatcher([])

  //   # initial symbolic + migrate indexing (remove this)
  sink = graph_rewrite(sink, sym.add(migrate_indexing))
  //   # expand
  sink = graph_rewrite(sink, sym.add(expander))

  //   # devectorize + load_store_indexing
  sink = graph_rewrite(sink, sym.add(opts !== undefined && opts.supports_float4 ? devectorize.add(float4_folding) : devectorize).add(load_store_indexing))

  //   # final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple.add(get_late_rewrite_patterns(supported_ops as any as Ops[], TRANSCENDENTAL >= 2)).add(pm_render).add(extra_matcher))
  return sink
}
