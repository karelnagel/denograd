import { dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { all_same, AMX, assert, DEBUG, dedup, flatten, getEnv, isEq, isinstance, isNone, isNotNone, prod, range, sorted, TRANSCENDENTAL } from '../helpers.ts'
import { graph_rewrite, GroupOp, idiv, Ops, PatternMatcher, simplify_valid, symbolic_flat, symbolic_simple, UOp, uop_given_valid, UPat } from '../ops.ts'
import { Renderer } from '../renderer/index.ts'
import { TRANSCENDENTAL_SUPPORTED_DTYPES, xexp2, xlog2, xsin } from './transcendental.ts'

// if TYPE_CHECKING: from tinygrad.renderer import Renderer

// # ***** float4/image store handling *****

export const fold_expanded = (ex: UOp, buf: UOp) => {
  if (buf.dtype.base !== dtypes.float && buf.dtype.base !== dtypes.half && !isinstance(buf.dtype, ImageDType)) return undefined
  let new_srcs: (UOp | undefined)[] = dedup([...ex.src])
  const old_new_srcs = [...new_srcs]
  const [is_load, is_image] = [new_srcs[0]?.op === Ops.LOAD, isinstance(buf.dtype, ImageDType)]

  //   # first, extract all the relevant offsets
  const offsets_rootsrc = new Map<any, Map<string, number>>()
  for (const [i, s] of new_srcs.entries()) {
    const idx = s!.src[0].src[1]
    let root_src, arg
    if (s!.dtype.count !== 1 || (is_image && idx.dtype.count == 2)) continue
    if (idx.op === Ops.ADD && idx.src[1].op === Ops.CONST) [root_src, arg] = [idx.src[0], idx.src[1].arg]
    else if (idx.op === Ops.CONST) [root_src, arg] = ['CONST', idx.arg]
    else [root_src, arg] = [idx, 0]
    //     # add gates for gated
    if (s!.src[0].src.length === 3) root_src = [s!.src[0].src[2], root_src]
    assert(!offsets_rootsrc.get(root_src)!.has(arg), `${offsets_rootsrc.get(root_src)?.get(arg)} != ${i} with ${s?.src.length} sources`)
    offsets_rootsrc.get(root_src)!.set(arg, i)
  }
  //   # then rewrite everything we can
  const lengths = is_image ? [4] : (buf.dtype.base === dtypes.half && getEnv('ALLOW_HALF8') ? [8, 4, 2] : (AMX ? [16, 8, 4, 2] : [4, 2]))
  let used = new Set<[UOp, UOp]>()
  for (const [rootsrc, offsets] of offsets_rootsrc.entries()) {
    for (const o of offsets.keys()) { //TODO:????
      for (const fold_length of lengths) {
        if (range(fold_length).every((i) => !used.has([rootsrc, o + i as any]) && offsets.has(o + i))) {
          const load_1 = new_srcs[offsets.get(o)!]
          const new_src = [...load_1!.src]
          const oidx = new_src[0].src[1]
          if (isNone(oidx.divides(fold_length))) continue
          if (is_image) {
            //             # for images, we rewrite the index. it must evenly divide 4 from the above check
            new_src[0] = buf.index(
              new UOp({ op: Ops.VECTORIZE, dtype: dtypes.int.vec(2), src: [oidx.idiv(4).mod((buf.dtype as ImageDType).shape[1]), oidx.idiv(4 * (buf.dtype as ImageDType).shape[1])] }),
              isinstance(rootsrc, Array) ? rootsrc[0] as UOp : undefined,
            )
          } else {
            //             # for non image, we upcast the index pointer
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr((new_src[0].dtype as PtrDType).local))
          }
          //           # generate the folded new_srcs
          if (is_load) {
            const new_load = new UOp({ op: Ops.LOAD, dtype: load_1!.dtype.vec(fold_length), src: new_src })
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = new_load.gep(i)
          } else { // vectorize the store
            new_src[1] = new UOp({ op: Ops.VECTORIZE, dtype: new_src[1].dtype.vec(fold_length), src: range(fold_length).map((i) => new_srcs[offsets.get(o + i)!]!.src[1]) })
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = i == 0 ? new UOp({ op: Ops.STORE, dtype: dtypes.void, src: new_src }) : undefined
          }
          used = new Set([...used, ...range(fold_length).map((i) => [rootsrc, o + i as any] as [UOp, UOp])])
        }
      }
    }
  }
  //   # dedup expand for LOAD
  if (is_load && old_new_srcs.length !== ex.src.length) new_srcs = ex.src.map((s) => new_srcs[old_new_srcs.indexOf(s)])
  //   # remove Nones for STORE
  return used.size ? new UOp({ op: ex.op, dtype: ex.dtype, src: [...new_srcs.filter((x) => isNotNone(x))], arg: ex.arg }) : undefined
}

export const fix_unfoldable_image_load = (load: UOp, buf: UOp) => {
  const oidx = load.src[0].src[1]
  if (!isinstance(buf.dtype, ImageDType) || oidx.dtype.count === 2) return undefined
  const id4 = oidx.mod(4)
  const new_src = [...load.src]
  //   # TODO: copied logic from above
  new_src[0] = load.src[0].src[0].index(
    new UOp({ op: Ops.VECTORIZE, dtype: dtypes.int.vec(2), src: [(oidx.idiv(4)).mod(buf.dtype.shape[1]), oidx.idiv(4 * buf.dtype.shape[1])] }),
    load.src[0].src.length === 3 ? load.src[0].src[2] : undefined,
  )
  const vec_load = new UOp({ op: Ops.LOAD, dtype: load.dtype.vec(4), src: [...new_src] })
  return range(4).reduce((ret, i) => id4.ne(i).where(ret, vec_load.gep(i)), load.const_like(NaN))
}

export const buf_idx_pat = new UPat({ op: Ops.INDEX, src: [UPat.var('buf')], allow_any_len: true })
export const float4_folding = new PatternMatcher([
  [new UPat({ op: Ops.VECTORIZE, src: new UPat({ op: Ops.LOAD, src: [buf_idx_pat], allow_any_len: true }), name: 'ex' }), ({ ex, buf }) => fold_expanded(ex, buf)],
  [new UPat({ op: [Ops.BARRIER, Ops.SINK], src: new UPat({ op: Ops.STORE, src: [buf_idx_pat], allow_any_len: true }), name: 'ex' }), ({ ex, buf }) => fold_expanded(ex, buf)],
])

// # ***** image load valid simplification *****

export const simplify_valid_load = (buf: UOp, start_idx: UOp, valid: UOp): undefined | UOp => {
  const idx = uop_given_valid(valid, start_idx)
  if (isNone(idx)) return buf.const_like(0)
  if (!isinstance(buf.dtype, ImageDType)) return idx === start_idx ? undefined : buf.index(idx, valid)
  throw new Error('not implemented')
  //   # wait for it to be image indexed before running simplification
  //   # TODO:not needed for mnist
  //   # if start_idx.dtype.count != 2: return None

  //   # # can drop valid if idx is out of bound when valid is False
  //   # drop_stmt = []
  //   # for stmt in split_uop(valid, Ops.AND):
  //   #   X, is_upper_bound, c = parse_valid(stmt)

  //   #   # for X0 + X1 + ... >= 1, check if it's out of bound when Xi = 0 for all i
  //   #   if not is_upper_bound and c == 1 and all(u.op in GroupOp.Irreducible and u.vmin == 0 for u in split_uop(X, Ops.ADD)):
  //   #     testidx = functools.reduce(lambda nowidx,u: nowidx.substitute({u:u.const_like(0)}), split_uop(X, Ops.ADD), idx)
  //   #     testidx = testidx.simplify()
  //   #     if testidx.gep(0).vmax < 0 or testidx.gep(1).vmax < 0:
  //   #       drop_stmt.append(stmt)
  //   #       continue

  //   #   # if X <= c, check if it's out of bound when X = c+1
  //   #   # if X >= c, check if it's out of bound when X = c-1
  //   #   test_value = c + 1 if is_upper_bound else c - 1
  //   #   for i,b in zip(idx.src, (buf.dtype.shape[1], buf.dtype.shape[0])):
  //   #     if is_increasing(i):
  //   #       rw = i.substitute({X:X.const_like(test_value)}).simplify()
  //   #       if rw.vmin >= b or rw.vmax < 0:
  //   #         drop_stmt.append(stmt)
  //   #         break

  //   # if not drop_stmt and idx is start_idx: return None
  //   # new_valid = functools.reduce(operator.and_, ss) if (ss:=[s for s in split_uop(valid, Ops.AND) if s not in drop_stmt]) else None
  //   # return buf.index(idx, new_valid)
}
// # ***** optional patterns *****

const powers_of_two = Object.fromEntries(range(64).map((i) => [2 ** i, i]))
// @functools.lru_cache(None)
type Pat = [UPat, (a: Record<'d' | 'base' | 'const' | 'div' | 'mul' | 'x' | 'y' | 'a' | 'b' | 'c', UOp>) => UOp | undefined]
export const get_late_rewrite_patterns = (ops: Ops[], force_transcendental = false) => {
  let pat: Pat[] = ([[Ops.EXP2, xexp2], [Ops.LOG2, xlog2], [Ops.SIN, xsin]] as const).filter(([op, f]) => !ops.includes(op) || force_transcendental)
    .map(([op, f]) => [new UPat({ op, dtype: TRANSCENDENTAL_SUPPORTED_DTYPES, src: [UPat.var('d')] }), f] as const)
  //   # rewrite MOD to AND (which should always be supported, but not for generic in tests)
  if (ops.includes(Ops.AND)) {
    pat = [...pat, [new UPat({ op: Ops.MOD, src: [UPat.var('base'), UPat.cvar('const')] }), (args: { base: UOp; const: UOp }) => powers_of_two[args.const.arg] ? args.base.bitwise_and(args.const.arg - 1) : undefined]]
  }
  //   # rewrite MUL/IDIV to SHL+SHR
  if (ops.includes(Ops.SHL) && ops.includes(Ops.SHR)) {
    pat = [
      ...pat,
      [new UPat({ op: Ops.MUL, dtype: dtypes.ints, src: [[UPat.cvar('const'), UPat.var('mul')]] }), (a) => powers_of_two[a.const.arg] ? a.mul.lshift(powers_of_two[a.const.arg]) : undefined], // (x  * (2**y)) -> shl(x,y)
      [new UPat({ op: Ops.IDIV, src: [UPat.var('div'), UPat.cvar('const')] }), (a) => powers_of_two[a.const.arg] ? a.div.rshift(powers_of_two[a.const.arg]) : undefined], // (x // (2**y)) -> shr(x,y)
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
}
// # ***** threefry *****

export const threefry2x32 = (x: UOp, key: UOp) => {
  //   # split x into two uint32, since x in a uint64
  const [x0, x1] = [(x.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((x.idiv(2 ** 32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]

  const rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  const [key0, key1] = [(key.bitwise_and(0xffffffff)).cast(dtypes.uint32), ((key.idiv(2 ** 32)).bitwise_and(0xffffffff)).cast(dtypes.uint32)]
  const ks = [key1, key0.xor(key1).xor(0x1BD11BDA), key0]
  let xr = [x0.add(ks[-1]), x1.add(ks[0])]
  for (const i of range(5)) {
    for (const r of rotations[i % 2]) {
      const x0 = xr[0].add(xr[1])
      ;[xr[0], xr[1]] = [x0, x0.xor(xr[1].mul(2 ** r).add(xr[1].idiv(2 ** (32 - r))))]
      xr = [xr[0].add(ks[i % 3]), xr[1].add(ks[(i + 1) % 3]).add(i + 1)]
    }
  }
  return xr[1].cast(dtypes.uint64).mul(2 ** 32).bitwise_or(xr[0].cast(dtypes.uint64))
}
// # ***** main rewriter *****

export const loop_collapse = (compval: any, multconst: any, rng: UOp, acc: UOp, idx2?: any, idx3?: any, extra?: any, vec?: any, ne?: any, add = UOp.const(dtypes.int, 0), mul = UOp.const(dtypes.int, 1)) => {
  if (getEnv('DISABLE_LOOP_COLLAPSE') || !acc.src.includes(rng)) return undefined // must be the right REDUCE
  let [loop_start, loop_end] = rng.src
  if (loop_start.arg !== 0) {
    //     # TODO: support and test this with other mul and loop_starts
    if (DEBUG >= 1) console.log(`WARNING, NOT FOLDING: mul:${mul.arg} loop_start:${loop_start.arg}`)
    return undefined
  }
  if (isNotNone(idx2)) add = add + idx2
  if (isNotNone(idx3)) add = add + idx3
  if (isNotNone(vec)) {
    //     # add, mul, loop_start, loop_end
    const dvec = (x: UOp) => {
      if (x.op === Ops.CONST) return UOp.const(x.dtype.vec(vec.dtype.count), x.arg)
      return new UOp({ op: Ops.VECTORIZE, dtype: x.dtype.vec(vec.dtype.count), src: range(vec.dtype.count).map(() => x) })
    }
    ;[add, mul, loop_start, loop_end] = [dvec(add), dvec(mul), dvec(loop_start), dvec(loop_end)]
  }

  let comprange
  if (mul.vmin > 0 && isNotNone(ne)) {
    comprange = loop_end.minimum(add.sub(compval)).idiv(mul).add(loop_end.sub(loop_start).maximum(loop_start))
  } else if (mul.vmax < 0 && isNone(ne)) comprange = loop_end.minimum(add.sub(compval).sub(mul)).idiv(mul).add(loop_end.sub(loop_start).maximum(loop_start))
  else return undefined
  const new_reduce_op = comprange.cast(multconst.dtype).mul(multconst)
  //   # TODO: what does it mean to have the same numbered DEFINE_ACC with different ranges?
  const new_acc = acc.replace({ src: [acc.src[1], ...acc.src.slice(1).filter((x) => x !== rng)] })
  let ret = new_acc.assign(new_acc.add(new_reduce_op))
  if (isNotNone(extra)) ret = ret.add(acc.assign(acc.add(extra)))
  //   return ret
}
export const index_collapse = (idx: UOp, rng: UOp, buf: UOp, ld: UOp, acc: UOp, add = UOp.const(dtypes.int, 0), mul = UOp.const(dtypes.int, 1)) => {
  throw new Error()
}

// # TODO: there's a lot shared with no_vectorized_wmma here
export const gep_through_wmma = (gep: UOp, wmma: UOp) => {
  throw new Error()
}
export const no_vectorized_wmma = (wmma: UOp) => {
  throw new Error()
}
export const reduce_collapse = (acc: UOp, ret: UOp, alu: UOp) => {
  throw new Error()
}
export const [acc_pat, rng_pat] = [new UPat({ op: Ops.DEFINE_ACC, name: 'acc' }), new UPat({ op: Ops.RANGE, name: 'rng' })]
export const rng_aug = UPat.any([rng_pat, UPat.var('add').add(rng_pat), UPat.var('mul').mul(rng_pat), UPat.var('add').add(UPat.var('mul').mul(rng_pat))])

export const index_load = UPat.var('buf').index(rng_aug).load(undefined, { name: 'ld' })

export const arange_augrng = UPat.any([rng_aug, rng_aug.add(UPat.var('idx2')), rng_aug.add(UPat.var('idx2')).add(UPat.var('idx3')), new UPat({ op: Ops.VECTORIZE, name: 'vec', src: rng_aug })])
export const arange_m = ((arange_augrng.lt(UPat.cvar('compval'))).ne(new UPat({ op: Ops.CONST, name: 'ne', arg: true }))).where(UPat.cvar('multconst'), UPat.const(undefined, 0))

// # this is symbolic 2.0
export const sym = symbolic_flat.add(
  new PatternMatcher([
    //   # self ASSIGN is just self
    [new UPat({ op: Ops.ASSIGN, src: [UPat.var('x'), UPat.var('x')] }), ({ x }) => x],
    //   # ASSIGN to global is just self
    [new UPat({ op: Ops.ASSIGN, src: [new UPat({ op: Ops.DEFINE_GLOBAL }), UPat.var('x')] }), ({ x }) => x],
    //   # VECTORIZE/CONST, VECTORIZE/GEP
    [new UPat({ op: Ops.VECTORIZE, src: new UPat({ op: Ops.CONST }), name: 'vec' }), ({ vec }) => UOp.const(vec.dtype, vec.src.map((x) => x.arg))],
    [new UPat({ op: Ops.VECTORIZE, src: new UPat({ op: Ops.GEP, src: [new UPat({ name: 'x' })] }), name: 'vec' }), ({ vec, x }) => x.gep(vec.src.map((y) => y.arg[0]))],
    //   # reorder ALU/VECTORIZE
    [
      new UPat({ op: GroupOp.ALU, src: [new UPat({ op: Ops.VECTORIZE, src: new UPat({ name: 'x' }) }), new UPat({ op: Ops.VECTORIZE, src: new UPat({ name: 'y' }) })], name: 'alu' }),
      ({ x, y, alu }) => new UOp({ op: Ops.VECTORIZE, dtype: alu.dtype, src: range(alu.dtype.count).map((i) => new UOp({ op: alu.op, dtype: alu.dtype.scalar(), src: [x, y] })) }),
    ],
    //   # VECTORIZE of a single element is just that element
    [new UPat({ op: Ops.VECTORIZE, src: [new UPat({ name: 'x' })] }), ({ x }) => x],
    //   # VECTORIZE void is SINK
    [new UPat({ op: Ops.VECTORIZE, dtype: dtypes.void, src: new UPat({ op: Ops.BARRIER, name: 'b' }) }), ({ b }) => b],
    [new UPat({ op: Ops.VECTORIZE, dtype: dtypes.void, name: 'x' }), ({ x }) => new UOp({ op: Ops.SINK, dtype: dtypes.void, src: x.src })],
    //   # GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
    [new UPat({ op: Ops.GEP, src: [new UPat({ op: Ops.GEP, name: 'g2' })], name: 'g1' }), ({ g1, g2 }) => g2.src[0].gep(range(g1.dtype.count).map((i) => g2.arg[g1.arg[i]]))],
    [
      new UPat({ op: Ops.GEP, src: [new UPat({ op: Ops.VECTORIZE, name: 'vec' })], name: 'gep' }),
      ({ gep, vec }) => new UOp({ op: Ops.VECTORIZE, dtype: gep.dtype, src: gep.arg.length > 1 ? gep.arg.map((i: number) => vec.src[i]) : vec.src[gep.arg[0]] }),
    ],
    [new UPat({ op: Ops.GEP, src: [UPat.cvar('c', undefined, false)], name: 'gep' }), ({ gep, c }) => gep.const_like(c.arg)],
    [new UPat({ op: Ops.GEP, src: [new UPat({ op: Ops.VCONST, name: 'c' })], name: 'gep' }), ({ gep, c }) => gep.const_like(gep.arg.map((x: any) => c.arg[x]))],
    //   # push all GEPs through ALUs (fix arange stuff)
    [
      new UPat({ op: Ops.GEP, src: [new UPat({ op: [...GroupOp.ALU, Ops.CAST, Ops.BITCAST], name: 'alu' })], name: 'gep' }),
      ({ gep, alu }) => new UOp({ op: alu.op, dtype: alu.dtype.scalar().vec(gep.dtype.count), src: alu.src.map((x) => x.gep(gep.arg)), arg: alu.arg }),
    ],
    //   # push some GEPs through WMMAs
    [new UPat({ op: Ops.GEP, src: [new UPat({ op: Ops.WMMA, name: 'wmma' })], name: 'gep' }), ({ wmma, gep }) => gep_through_wmma(gep, wmma)],
    //   # tensor core with a 0 input is acc
    [new UPat({ op: Ops.WMMA, src: [UPat.const(undefined, 0.0), UPat.var(), UPat.var('acc')] }), ({ acc }) => acc],
    [new UPat({ op: Ops.WMMA, src: [UPat.var(), UPat.const(undefined, 0.0), UPat.var('acc')] }), ({ acc }) => acc],
    //   # tensor core cleanups
    [UPat.var('add').add(new UPat({ op: Ops.WMMA, name: 'wmma' })), ({ add, wmma }) => new UOp({ op: wmma.op, dtype: wmma.dtype, src: [wmma.src[0], wmma.src[1], wmma.src[2].add(add)], arg: wmma.arg })],
    //   # threefry + remove longs
    [new UPat({ op: Ops.THREEFRY, dtype: dtypes.uint64, src: [UPat.var('x'), UPat.var('key')] }), ({ x, key }) => threefry2x32(x, key)],
    [UPat.var('x', dtypes.uint32).cast(dtypes.uint64).cast(dtypes.uint32), ({ x }) => x], // cast there and back is noop (TODO: genericize)
    [(UPat.var('x', dtypes.uint64).bitwise_and(0xFFFFFFFF)).cast(dtypes.uint32), ({ x }) => x.cast(dtypes.uint32)], // cast does truncation
    [(UPat.var(undefined, dtypes.uint64).mul(2 ** 32).bitwise_or(UPat.var('y', dtypes.uint32).cast(dtypes.uint64))).cast(dtypes.uint32), ({ y }) => y],
    [((UPat.var('x', dtypes.uint64).mul(2 ** 32)).bitwise_or(UPat.var(undefined, dtypes.uint32).cast(dtypes.uint64))).idiv(2 ** 32), ({ x }) => x],
    //   # hacks for threefry long removal when padded (TODO: genericize)
    [UPat.var('x', dtypes.uint32).cast(dtypes.uint64).mul(UPat.var('y').where(UPat.const(dtypes.uint64, 2 ** 32), UPat.const(dtypes.uint64, 0))), ({ x, y }) => y.where(x, UOp.const(dtypes.uint32, 0)).cast(dtypes.uint64).mul(2 ** 32)],
    [(UPat.var('x', dtypes.uint64).bitwise_and(UPat.var('y').where(UPat.const(dtypes.uint64, 0xFFFFFFFF), UPat.const(dtypes.uint64, 0)))).cast(dtypes.uint32), ({ x, y }) => y.where(x.cast(dtypes.uint32), UOp.const(dtypes.uint32, 0))],
    // # arange loop folding
    // [acc_pat.assign(UPat.any([arange_m, arange_m.add(UPat.var('extra'))]).add(acc_pat)), ({ compval, multconst, rng, acc, idx2, idx3, extra, vec, ne, add, mul }) => loop_collapse(compval, multconst, rng, acc, idx2, idx3, extra, vec, ne, add, mul)],
    // # indexing, with cast or where
    [acc_pat.assign(UPat.var('idx').eq(new UPat({ op: Ops.RANGE, name: 'rng' })).cast().mul(index_load).add(acc_pat)), ({ idx, rng, buf, ld, axx, add, mul }) => index_collapse(idx, rng, buf, ld, axx, add, mul)],
    [acc_pat.assign(UPat.var('idx').eq(new UPat({ op: Ops.RANGE, name: 'rng' })).where(index_load, UPat.const(undefined, 0.0)).add(acc_pat)), ({ idx, rng, buf, ld, acc, add, mul }) => index_collapse(idx, rng, buf, ld, acc, add, mul)],
    // # parentless reduce  # TODO: add MUL
    [acc_pat.assign(new UPat({ op: [Ops.ADD, Ops.MAX], src: [[acc_pat, UPat.var('ret')]], name: 'alu' })), ({ acc, ret, alu }) => reduce_collapse(acc, ret, alu)],
    //   # ** self folding **
    [new UPat({ op: Ops.DEFINE_ACC, src: [UPat.var('x')] }), ({ x }) => x], // a DEFINE_ACC without ranges is a CONST
    [new UPat({ op: Ops.ASSIGN, src: [UPat.cvar(), UPat.var('x')] }), ({ x }) => x], // an ASSIGN to a const is a NOOP
    // # x!=0 -> (bool)x
    [UPat.var('x').ne(0), ({ x }) => x.cast(dtypes.bool.vec(x.dtype.count))],
    //   # ** load/store folding **
    [new UPat({ op: Ops.INDEX, name: 'index' }).store([new UPat({ op: Ops.INDEX, name: 'index' }).load()]), ({ index }) => new UOp({ op: Ops.NOOP })],
    [new UPat({ op: Ops.INDEX, name: 'index' }).store([UPat.var('gate').where(UPat.var('alt'), new UPat({ op: Ops.INDEX, name: 'index' }).load())]), ({ index, gate, alt }) => index.src[0].index(index.src[1], gate).store([alt])],
    // # fold gated LOAD/STORE
    [new UPat({}).index(new UPat({}), UPat.const(dtypes.bool, true)).named('idx'), ({ idx }) => idx.replace({ src: idx.src.slice(0, 2) })], // remove True
    [new UPat({}).index(new UPat({}), UPat.const(dtypes.bool, false)).named('idx'), ({ idx }) => idx.const_like(0)], //False -> NULL pointer
    [new UPat({ op: Ops.LOAD, src: [UPat.const(undefined, 0)], allow_any_len: true, name: 'x' }), ({ x }) => x.const_like(0)], // NULL pointer load loads 0
    [new UPat({ op: Ops.STORE, src: [UPat.const(undefined, 0)], allow_any_len: true }), () => new UOp({ op: Ops.NOOP })], // NULL pointer store does nothing
    //   # remove NOOPs from SINK
    [new UPat({ op: Ops.SINK, name: 'root' }), ({ root }) => {
      const a = root.src.filter((x) => x.op !== Ops.NOOP)
      return a.length !== root.src.length ? new UOp({ op: Ops.SINK, dtype: root.dtype, src: a, arg: root.arg }) : undefined
    }],
    //   # remove EXPANDs from SINK/BARRIER
    [new UPat({ op: Ops.BARRIER, src: [new UPat({ op: [Ops.VECTORIZE, Ops.SINK], name: 'sink' })] }), ({ sink }) => new UOp({ op: Ops.BARRIER, dtype: dtypes.void, src: sink.src })],
    [
      new UPat({ op: Ops.SINK, name: 'root' }),
      ({ root }) => root.src.every((x) => [Ops.SINK, Ops.EXPAND].includes(x.op)) ? new UOp({ op: Ops.SINK, dtype: root.dtype, src: root.src.flatMap((x) => [Ops.SINK, Ops.EXPAND].includes(x.op) ? x.src : [x]), arg: root.arg }) : undefined,
    ],
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
  return args.reduce((acc, [axis, m]) => acc.flatMap((d) => range(m).map((i) => ({ ...d, [axis]: i }))), [{}]) // TODO: Can likely be wrong
}
export const _swizzle_args = (cargs: [number, number][], eargs: [number, number][], exclude_args: number[]): number[] => {
  return _choices_from_args(cargs).map((rpk) => _expand_arg_to_idx(eargs, exclude_args ? Object.fromEntries(exclude_args.map((x) => [x, 0])) : rpk))
}
export const do_expand = (root: UOp) => {
  const expands = root.src.filter((x) => x.op === Ops.EXPAND)
  if (expands.length === 0) return undefined
  //   # NOTE: we 0 out the reduce axis for WMMA. in theory they should all be the same, but is this always correct?
  const exclude_args = root.op === Ops.WMMA ? dedup([...root.arg[-1], ...flatten(root.arg.slice(-2)).map((y: any) => y[0])]) : []
  const expands_args = expands.map((x) => x.arg)
  let expand_args
  if (all_same(expands_args) && exclude_args.length === 0) {
    //     # if there's only one expand arg, it's okay to use it (optimization)
    expand_args = expands[0].arg
  } // otherwise, we sort them and GEP
  else expand_args = sorted(dedup(flatten(expands_args)) as any).filter((x) => !exclude_args.includes((x as any)[0]))
  const expand_sz = prod(exclude_args.map((x) => x[1]))
  const new_srcs = []
  for (const [i, src] of root.src.entries()) {
    if (src.op === Ops.EXPAND) {
      //         # IF means OR on first arg to IF
      if (root.op === Ops.IF && i === 0) new_srcs.push(range(expand_sz).map((i) => src.src[0].gep(i)).reduce((acc, x) => acc.bitwise_or(x)))
      //         # just remove the expand
      else if (expand_args == src.arg) new_srcs.push(src.src[0])
      else {
        let lst = _swizzle_args(expand_args, src.arg, exclude_args)
        //         # if the base dtype is > 1, put those at the end
        if (src.dtype.count > 1) lst = lst.flatMap((i) => range(src.dtype.count).map((j) => i * src.dtype.count + j))
        new_srcs.push(src.src[0].gep([...lst]))
      }
    } //       # non-EXPAND input
    else {
      //         # for the first arg of IF, just pass them through ignoring EXPANDS
      if (root.op === Ops.IF) new_srcs.push(src)
      //         # put any input dtype > 1 grouped together
      else if (src.dtype.count > 1) new_srcs.push(new UOp({ op: Ops.VECTORIZE, dtype: src.dtype.scalar().vec(expand_sz * src.dtype.count), src: range(expand_sz).flatMap(() => range(src.dtype.count).map((i) => src.gep(i))) })) //TODO: this src.mul() might not be right
      //         # repeat the arg
      else new_srcs.push(src.broadcast(expand_sz))
    }
  }
  let new_arg = root.arg
  if (root.op === Ops.GEP) {
    assert(root.dtype.count === 1)
    //     # is this right?
    new_arg = range(root.arg[0], new_srcs[0].dtype.count, idiv(new_srcs[0].dtype.count, expand_sz))
  }
  const nsrc = new UOp({ op: root.op, dtype: root.dtype.scalar().vec(root.dtype.count * expand_sz), src: new_srcs, arg: new_arg })
  return new UOp({ op: Ops.EXPAND, dtype: root.dtype, src: [nsrc], arg: expand_args })
}
export const do_contract = (con: UOp) => {
  const ex = con.src[0]
  //   # CONTRACT without EXPAND repeats the element VECTORIZED
  if (ex.op !== Ops.EXPAND) return new UOp({ op: Ops.VECTORIZE, dtype: con.dtype, src: range(con.dtype.count).flatMap(() => con.src.map((x) => x.mul(con.dtype.count))) }) // TODO: not sure
  //   # CONTRACT may remove several axes from EXPAND
  assert(con.dtype.count === prod(con.arg.map((x: any) => x[1])), 'dtype is wrong')
  let idxs: number[] = []
  const new_ex_args = ex.arg.filter((x: any) => !con.arg.includes(x))
  for (const rpk of _choices_from_args(new_ex_args)) {
    idxs = [...idxs, ..._choices_from_args(con.arg).map((lrpk) => _expand_arg_to_idx(ex.arg, { ...rpk, ...lrpk }))]
  }
  return new UOp({ op: Ops.EXPAND, dtype: con.dtype, src: [ex.src[0].gep([...idxs])], arg: new_ex_args })
}
export const no_vectorized_alu = (alu: UOp) => {
  if (alu.dtype.vcount === 1) return undefined
  const alus = range(alu.dtype.vcount).map((i) => new UOp({ op: alu.op, dtype: alu.dtype.scalar(), src: alu.src.map((s) => s.gep(i)), arg: alu.arg }))
  return new UOp({ op: Ops.VECTORIZE, dtype: alu.dtype, src: alus })
}
export const create_gate = (root: UOp): undefined | UOp => {
  const _gate_srcs = (u: UOp, gate: UOp): UOp => {
    if (u.op === Ops.BARRIER) return u
    if (u.op === Ops.LOAD && u.src[-1].op === Ops.BARRIER) return new UOp({ op: u.op, dtype: u.dtype, src: [...u.src.toReversed(), new UOp({ op: Ops.IF, dtype: dtypes.void, src: [gate, u.src.at(-1)!] })], arg: u.arg })
    const replace_source = u.src.map((x) => _gate_srcs(x, gate))
    return isEq(replace_source, u.src) ? u : new UOp({ op: u.op, dtype: u.dtype, src: replace_source, arg: u.arg })
  }
  let idx = root.src[0]
  if (idx.op === Ops.CAST) idx = idx.src[0]
  const ret = _gate_srcs(root, idx.src[2])
  return idx.op !== Ops.INDEX || idx.src.length === 2 || ret === root ? undefined : ret
}
export const expander = new PatternMatcher([
  //   # double expand
  [new UPat({ op: Ops.EXPAND, name: 'outer', src: [new UPat({ op: Ops.EXPAND, name: 'inner' })] }), ({ outer, inner }) => new UOp({ op: Ops.EXPAND, dtype: outer.dtype, src: [inner.src[0]], arg: inner.arg + outer.arg })],
  //   # do expansion
  [new UPat({ op: [...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WMMA, Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.ASSIGN, Ops.VECTORIZE, Ops.IF], name: 'root', custom_early_reject: [Ops.EXPAND] }), ({ root }) => do_expand(root)],
  [new UPat({ op: Ops.CONTRACT, name: 'con' }), ({ con }) => do_contract(con)],
  //   # vectorize DEFINE_ACC
  [new UPat({ op: Ops.VECTORIZE, src: new UPat({ op: Ops.DEFINE_ACC, name: 'acc' }), name: 'v' }), ({ acc, v }) => acc.replace({ dtype: v.dtype })],
  //   # BARRIERs aren't actually expanded
  [
    new UPat({ op: Ops.BARRIER, src: [new UPat({ op: Ops.EXPAND, name: 'ex' })] }),
    ({ ex }) => new UOp({ op: Ops.EXPAND, dtype: dtypes.void, src: range(ex.src.length).map((x) => new UOp({ op: Ops.BARRIER, dtype: dtypes.void, src: ex.src })), arg: ex.arg }),
  ],
  //   # empty EXPAND is NOOP
  [new UPat({ op: Ops.EXPAND, src: [UPat.var('x')], arg: [] }), ({ x }) => x],
  //   # EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  [
    new UPat({ op: Ops.EXPAND, name: 'ex', src: range(AMX ? 256 : 8).map((i) => UPat.var('x').gep(i).add(UPat.var('y').gep(i))) }),
    ({ ex, x, y }) => new UOp({ op: Ops.EXPAND, dtype: ex.dtype, src: range(AMX ? 256 : 8).map((i) => x.add(y).gep(i)), arg: ex.arg }),
  ],
])

export const no_vectorized_load_store = (ls: UOp) => {
  const idx = ls.src[0]
  if (!isinstance(idx.dtype, PtrDType)) throw new Error()
  if (idx.dtype.v === 1) return undefined
  const tv = range(idx.dtype.v).map((i) => new UOp({ op: ls.op, dtype: ls.dtype.scalar(), src: ls.src.map((j) => j.gep(i)) }))
  return new UOp({ op: Ops.VECTORIZE, dtype: ls.dtype, src: tv })
}
export const no_vectorized_acc = (acc: UOp) => {
  if (acc.dtype.count === 1) return undefined
  const alus = [new UOp({ op: acc.op, dtype: acc.dtype.scalar(), src: range(acc.dtype.count).flatMap((i) => [...acc.src.entries()].map(([j, s]) => j === 0 ? s.gep(i) : s)) })]
  return new UOp({ op: Ops.VECTORIZE, dtype: acc.dtype, src: alus })
}
export const devectorize = new PatternMatcher([
  //   # no ALU on vectorized dtypes
  [new UPat({ op: [...GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX], name: 'alu' }), ({ alu }) => no_vectorized_alu(alu)],
  [new UPat({ op: Ops.WMMA, name: 'wmma' }), ({ wmma }) => no_vectorized_wmma(wmma)],
  [new UPat({ op: Ops.DEFINE_ACC, name: 'acc' }), ({ acc }) => no_vectorized_acc(acc)],
  [new UPat({ op: [Ops.LOAD, Ops.STORE], name: 'ls' }), ({ ls }) => no_vectorized_load_store(ls)],
])

export const delete_redundant_gates = (buf: UOp, idx: UOp, val: UOp, store_gate: UOp, cast?: UOp): undefined | UOp => {
  if (![...val.toposort].filter((gate) => gate.op === Ops.IF).map((gate) => gate.src[0]).includes(store_gate)) return undefined
  //   # remove the gate from the index
  return (isNotNone(cast) ? buf.index(idx).cast(cast.dtype) : buf.index(idx)).store([val])
}
const _stidx = UPat.var('buf').index(UPat.var('idx'), UPat.var('store_gate'))
export const load_store_indexing = new PatternMatcher([
  //   # late fixup of unfoldable image loads
  [new UPat({ op: Ops.LOAD, src: [UPat.var('buf'), new UPat({})], allow_any_len: true, name: 'load' }), ({ load, buf }) => fix_unfoldable_image_load(load, buf)],
  //   # simplify valid
  [new UPat({ op: Ops.AND, name: 'valid' }), ({ valid }) => simplify_valid(valid)],
  //   # image load valid idx simplification
  [new UPat({ op: Ops.INDEX, src: [UPat.var('buf'), UPat.var('start_idx'), UPat.var('valid')] }), ({ buf, start_idx, valid }) => simplify_valid_load(buf, start_idx, valid)],
  //   # delete_redundant_gates (after expand)
  [new UPat({ op: Ops.STORE, src: [UPat.any([_stidx, _stidx.cast(undefined).named('cast')]), UPat.var('val')] }), ({ buf, idx, val, store_gate, cast }) => delete_redundant_gates(buf, idx, val, store_gate, cast)],
])

export const migrate_indexing = new PatternMatcher([
  //   # create gate MUST BE BEFORE expander
  [new UPat({ op: Ops.STORE, name: 'root' }), ({ root }) => create_gate(root)],
])

export const move_mask = (x: UOp, buf: UOp, idx: UOp, mask: UOp, cast?: UOp): UOp => {
  //   # this moves the mask from the indexing to the load/store op for rendering
  const nidx = isNotNone(cast) ? buf.index(idx).cast(cast.dtype) : buf.index(idx)
  return x.op === Ops.LOAD ? nidx.load([x.const_like(0), mask, ...x.src.slice(1)], { dtype: x.dtype }) : nidx.store([x.src[1], mask, ...x.src.slice(2)])
}
const _masked_index = new UPat({ op: Ops.INDEX, src: [new UPat({ name: 'buf' }), new UPat({ name: 'idx' }), new UPat({ name: 'mask' })] })
export const pm_render = new PatternMatcher([
  //   # for rendering, we use explicit VECTORIZE
  [new UPat({ op: Ops.CONST, name: 'c' }), ({ c }) => new UOp({ op: Ops.VECTORIZE, dtype: c.dtype, src: c.dtype.vcount > 1 ? range(c.dtype.vcount).map(() => UOp.const(c.dtype.scalar(), c.arg)) : undefined })],
  [new UPat({ op: Ops.VCONST, name: 'c' }), ({ c }) => new UOp({ op: Ops.VECTORIZE, dtype: c.dtype, src: c.arg.map((x: number) => UOp.const(c.dtype.scalar(), x)) })],
  [new UPat({ op: Ops.GEP, name: 'gep' }), ({ gep }) => new UOp({ op: Ops.VECTORIZE, dtype: gep.dtype, src: gep.arg.length > 1 ? gep.arg.map((x: number) => gep.src[0].gep(x)) : undefined })],
  [new UPat({ op: Ops.VECTORIZE, src: [new UPat({ name: 'x' })] }), ({ x }) => x],
  //   # move masks of loads/stores
  [
    new UPat({ op: [Ops.LOAD, Ops.STORE], src: [UPat.any([_masked_index, _masked_index.cast(undefined).named('cast')])], allow_any_len: true, name: 'x' }),
    ({ x, buf, idx, mask, cast }) => move_mask(x, buf, idx, mask, cast),
  ],
  //   # gate any stores that aren't gated with ifs
  [
    new UPat({ op: Ops.STORE, dtype: dtypes.void, src: [new UPat({}), new UPat({}), new UPat({ dtype: dtypes.bool })], name: 'store' }),
    ({ store }) => new UOp({ op: Ops.STORE, src: [...store.src.slice(0, 2), new UOp({ op: Ops.IF, src: [store.src[2]] })] }),
  ],
])

// # *** uop graph ***

export const full_graph_rewrite = (sink: UOp, opts?: Renderer): UOp => {
  assert(sink.op === Ops.SINK, `sink isn't sink, it's ${sink.op}`)
  const supported_ops = isNotNone(opts) ? Object.keys(opts?.code_for_op) : []
  const extra_matcher = isNotNone(opts) && isNotNone(opts.extra_matcher) ? opts.extra_matcher : new PatternMatcher([])

  //   # initial symbolic + migrate indexing (remove this)
  sink = graph_rewrite(sink, sym.add(migrate_indexing))

  //   # expand
  sink = graph_rewrite(sink, sym.add(expander))

  //   # devectorize + load_store_indexing
  sink = graph_rewrite(sink, sym.add(isNotNone(opts) && opts.supports_float4 ? devectorize.add(float4_folding) : devectorize).add(load_store_indexing))

  //   # final rules for the renderer (without sym)
  sink = graph_rewrite(sink, symbolic_simple.add(get_late_rewrite_patterns(supported_ops as unknown as Ops[], TRANSCENDENTAL >= 2)).add(pm_render).add(extra_matcher))
  return sink
}
