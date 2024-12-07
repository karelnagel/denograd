import { dtypes, ImageDType, PtrDType } from '../dtype.ts'
import { allSame, AMX, assert, DEBUG, dedup, flatten, getEnv, isEq, isinstance, isNone, isNotNone, prod, range, sorted, TRANSCENDENTAL } from '../helpers.ts'
import { graph_rewrite, idiv, Ops, PatternMatcher, symbolic_flat, symbolic_simple, UOp, uop_given_valid, UPat } from '../ops.ts'
import { Renderer } from '../renderer/index.ts'
import { TRANSCENDENTAL_SUPPORTED_DTYPES, xexp2, xlog2, xsin } from './transcendental.ts'

// if TYPE_CHECKING: from tinygrad.renderer import Renderer

// # ***** float4/image store handling *****

const fold_expanded = (ex: UOp, buf: UOp) => {
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
    assert(!offsets_rootsrc.get(root_src)!.has(arg), `${offsets_rootsrc.get(root_src)?.get(arg)} != ${i} with ${s.src.length} sources`)
    offsets_rootsrc.get(root_src)!.set(arg, i)
  }
  //   # then rewrite everything we can
  const lengths = is_image ? [4] : (buf.dtype.base === dtypes.half && getEnv('ALLOW_HALF8') ? [8, 4, 2] : (AMX ? [16, 8, 4, 2] : [4, 2]))
  let used = new Set<[UOp, UOp]>()
  for (const [rootsrc, offsets] of offsets_rootsrc.entries()) {
    for (const o of offsets.keys()) { //TODO:????
      for (const fold_length of lengths) {
        if (range(fold_length).every((i) => !used.has([rootsrc, o + i]) && offsets.has(o + i))) {
          const load_1 = new_srcs[offsets.get(o)!]
          const new_src = [...load_1!.src]
          const oidx = new_src[0].src[1]
          if (isNone(oidx.divides(fold_length))) continue
          if (is_image) {
            //             # for images, we rewrite the index. it must evenly divide 4 from the above check
            new_src[0] = buf.index(
              new UOp({ op: Ops.VECTORIZE, dtype: dtypes.int.vec(2), src: [oidx.idiv(4).mod((buf.dtype as ImageDType).shape[1]), oidx.idiv(4 * (buf.dtype as ImageDType).shape[1])] }),
              isinstance(rootsrc, Array) ? rootsrc[0] : undefined,
            )
          } else {
            //             # for non image, we upcast the index pointer
            new_src[0] = new_src[0].cast(new_src[0].dtype.base.vec(fold_length).ptr((new_src[0].dtype as PtrDType).local))
          }
          //           # generate the folded new_srcs
          if (is_load) {
            const new_load = new UOp({ op: Ops.LOAD, dtype: load_1.dtype.vec(fold_length), src: new_src })
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = new_load.gep(i)
          } else { // vectorize the store
            new_src[1] = new UOp({ op: Ops.VECTORIZE, dtype: new_src[1].dtype.vec(fold_length), src: range(fold_length).map((i) => new_srcs[offsets.get(o + i)!]!.src[1]) })
            for (const i of range(fold_length)) new_srcs[offsets.get(o + i)!] = i == 0 ? new UOp({ op: Ops.STORE, dtype: dtypes.void, src: new_src }) : undefined
          }
          used = new Set([...used, ...range(fold_length).map((x) => [rootsrc, o + i] as [UOp, UOp])])
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
  [new UPat({ op: Ops.VECTORIZE, src: new UPat({ op: Ops.LOAD, src: [buf_idx_pat], allow_any_len: true }), name: 'ex' }), fold_expanded],
  [new UPat({ op: [Ops.BARRIER, Ops.SINK], src: new UPat({ op: Ops.STORE, src: [buf_idx_pat], allow_any_len: true }), name: 'ex' }), fold_expanded],
])

// # ***** image load valid simplification *****

export const simplify_valid_load = (buf: UOp, start_idx: UOp, valid: UOp): undefined | UOp => {
  const idx = uop_given_valid(valid, start_idx)
  if (isNone(idx)) return buf.const_like(0)
  if (!isinstance(buf.dtype, ImageDType)) return idx === start_idx ? undefined : buf.index(idx, valid)

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
    pat = [...pat, [new UPat({ op: Ops.MOD, src: [UPat.var('base'), UPat.cvar('const')] }), (args: { base: UOp; const: UOp }) => powers_of_two[args.const.arg] ? args.base.bitwiseAnd(args.const.arg - 1) : undefined]]
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
  const [x0, x1] = [(x.bitwiseAnd(0xffffffff)).cast(dtypes.uint32), ((x.idiv(2 ** 32)).bitwiseAnd(0xffffffff)).cast(dtypes.uint32)]

  const rotations = [[13, 15, 26, 6], [17, 29, 16, 24]]
  const [key0, key1] = [(key.bitwiseAnd(0xffffffff)).cast(dtypes.uint32), ((key.idiv(2 ** 32)).bitwiseAnd(0xffffffff)).cast(dtypes.uint32)]
  const ks = [key1, key0.xor(key1).xor(0x1BD11BDA), key0]
  let xr = [x0.add(ks[-1]), x1.add(ks[0])]
  for (const i of range(5)) {
    for (const r of rotations[i % 2]) {
      const x0 = xr[0].add(xr[1])
      ;[xr[0], xr[1]] = [x0, x0.xor(xr[1].mul(2 ** r).add(xr[1].idiv(2 ** (32 - r))))]
      xr = [xr[0].add(ks[i % 3]), xr[1].add(ks[(i + 1) % 3]).add(i + 1)]
    }
  }
  return xr[1].cast(dtypes.uint64).mul(2 ** 32).bitwiseOr(xr[0].cast(dtypes.uint64))
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
  new PatternMatcher<Record<string, UOp>, UOp | undefined>([
    //   # self ASSIGN is just self
    //   (UPat(Ops.ASSIGN, src=(UPat.var('x'), UPat.var('x'))), lambda x: x),
    //   # ASSIGN to global is just self
    //   (UPat(Ops.ASSIGN, src=(UPat(Ops.DEFINE_GLOBAL), UPat.var("x"))), lambda x: x),
    //   # VECTORIZE/CONST, VECTORIZE/GEP
    //   (UPat(Ops.VECTORIZE, src=UPat(Ops.CONST), name="vec"), lambda vec: UOp.const(vec.dtype, tuple(x.arg for x in vec.src))),
    //   (UPat(Ops.VECTORIZE, src=UPat(Ops.GEP, src=(UPat(name="x"),)), name="vec"), lambda vec,x: x.gep(tuple(y.arg[0] for y in vec.src))),
    //   # reorder ALU/VECTORIZE
    //   (UPat(GroupOp.ALU, src=(UPat(Ops.VECTORIZE, src=UPat(name='x')), UPat(Ops.VECTORIZE, src=UPat(name='y'))), name='alu'),
    //    lambda x,y,alu: UOp(Ops.VECTORIZE, alu.dtype, (UOp(alu.op, alu.dtype.scalar(), (x,y)),)*alu.dtype.count)),
    //   # VECTORIZE of a single element is just that element
    //   (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
    //   # VECTORIZE void is SINK
    //   (UPat(Ops.VECTORIZE, dtype=dtypes.void, src=UPat(Ops.BARRIER, name='b')), lambda b: b),
    //   (UPat(Ops.VECTORIZE, dtype=dtypes.void, name='x'), lambda x: UOp(Ops.SINK, dtypes.void, x.src)),
    //   # GEP/VECTORIZE, GEP/GEP, GEP/CONST, GEP/VCONST
    //   (UPat(Ops.GEP, src=(UPat(Ops.GEP, name='g2'),), name='g1'),
    //    lambda g1, g2: g2.src[0].gep(tuple(g2.arg[g1.arg[i]] for i in range(g1.dtype.count)))),
    //   (UPat(Ops.GEP, src=(UPat(Ops.VECTORIZE, name="vec"),), name="gep"),
    //    lambda gep, vec: UOp(Ops.VECTORIZE, gep.dtype, tuple(vec.src[i] for i in gep.arg)) if len(gep.arg) > 1 else vec.src[gep.arg[0]]),
    //   (UPat(Ops.GEP, src=(UPat.cvar("c", vec=False),), name="gep"), lambda gep, c: gep.const_like(c.arg)),
    //   (UPat(Ops.GEP, src=(UPat(Ops.VCONST, name="c"),), name="gep"), lambda gep, c: gep.const_like(tuple(c.arg[x] for x in gep.arg))),
    //   # push all GEPs through ALUs (fix arange stuff)
    //   (UPat(Ops.GEP, src=(UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST), name='alu'),), name='gep'),
    //    lambda gep,alu: UOp(alu.op, alu.dtype.scalar().vec(gep.dtype.count), tuple(x.gep(gep.arg) for x in alu.src), alu.arg)),
    //   # push some GEPs through WMMAs
    //   (UPat(Ops.GEP, src=(UPat(Ops.WMMA, name="wmma"),), name="gep"), gep_through_wmma),
    //   # tensor core with a 0 input is acc
    //   (UPat(Ops.WMMA, src=(UPat.const(None, 0.0), UPat.var(), UPat.var("acc"))), lambda acc: acc),
    //   (UPat(Ops.WMMA, src=(UPat.var(), UPat.const(None, 0.0), UPat.var("acc"))), lambda acc: acc),
    //   # tensor core cleanups
    //   (UPat.var("add") + UPat(Ops.WMMA, name="wmma"),
    //     lambda add, wmma: UOp(wmma.op, wmma.dtype, (wmma.src[0], wmma.src[1], wmma.src[2]+add), wmma.arg)),
    //   # threefry + remove longs
    //   (UPat(Ops.THREEFRY, dtype=dtypes.uint64, src=(UPat.var("x"), UPat.var("key"))), threefry2x32),
    //   (UPat.var('x', dtypes.uint32).cast(dtypes.uint64).cast(dtypes.uint32), lambda x: x),   # cast there and back is noop (TODO: genericize)
    //   ((UPat.var('x', dtypes.uint64)&0xFFFFFFFF).cast(dtypes.uint32), lambda x: x.cast(dtypes.uint32)),  # cast does truncation
    //   (((UPat.var(None, dtypes.uint64)*(1<<32)) | UPat.var('y',  dtypes.uint32).cast(dtypes.uint64)).cast(dtypes.uint32), lambda y: y),
    //   (((UPat.var('x',  dtypes.uint64)*(1<<32)) | UPat.var(None, dtypes.uint32).cast(dtypes.uint64))//(1<<32), lambda x: x),
    //   # hacks for threefry long removal when padded (TODO: genericize)
    //   (UPat.var('x', dtypes.uint32).cast(dtypes.uint64) * UPat.var('y').where(UPat.const(dtypes.uint64, 1<<32), UPat.const(dtypes.uint64, 0)),
    //    lambda x,y: y.where(x, UOp.const(dtypes.uint32, 0)).cast(dtypes.uint64) * (1<<32)),
    //   ((UPat.var('x', dtypes.uint64)&(UPat.var('y').where(UPat.const(dtypes.uint64, 0xFFFFFFFF), UPat.const(dtypes.uint64, 0)))).cast(dtypes.uint32),
    //    lambda x,y: y.where(x.cast(dtypes.uint32), UOp.const(dtypes.uint32, 0))),
    //   # arange loop folding
    //   (acc_pat.assign(UPat.any(arange_m, arange_m+UPat.var("extra"))+acc_pat), loop_collapse),
    //   # indexing, with cast or where
    //   (acc_pat.assign(UPat.var("idx").eq(UPat(Ops.RANGE, name="rng")).cast()*index_load+acc_pat), index_collapse),
    //   (acc_pat.assign(UPat.var("idx").eq(UPat(Ops.RANGE, name="rng")).where(index_load, UPat.const(None, 0.0))+acc_pat), index_collapse),
    //   # parentless reduce  # TODO: add MUL
    //   (acc_pat.assign(UPat((Ops.ADD, Ops.MAX), src=[acc_pat, UPat.var("ret")], name="alu")), reduce_collapse),
    //   # ** self folding **
    //   (UPat(Ops.DEFINE_ACC, src=(UPat.var("x"),)), lambda x: x),            # a DEFINE_ACC without ranges is a CONST
    //   (UPat(Ops.ASSIGN, src=(UPat.cvar(),UPat.var("x"))), lambda x: x),     # an ASSIGN to a const is a NOOP
    //   # x!=0 -> (bool)x
    //   (UPat.var("x")!=0, lambda x: x.cast(dtypes.bool.vec(x.dtype.count))),
    //   # ** load/store folding **
    //   (UPat.store(UPat(Ops.INDEX, name="index"), UPat.load(UPat(Ops.INDEX, name="index"))), lambda index: UOp(Ops.NOOP)),
    //   (UPat.store(UPat(Ops.INDEX, name="index"), UPat.var("gate").where(UPat.var("alt"), UPat.load(UPat(Ops.INDEX, name="index")))),
    //    lambda index, gate, alt: UOp.store(index.src[0].index(index.src[1], gate), alt)),
    //   # fold gated LOAD/STORE
    //   (UPat().index(UPat(), UPat.const(dtypes.bool, True)).named("idx"), lambda idx: idx.replace(src=idx.src[0:2])), # remove True
    //   (UPat().index(UPat(), UPat.const(dtypes.bool, False)).named("idx"), lambda idx: idx.const_like(0)),      # False -> NULL pointer
    //   (UPat(Ops.LOAD, src=(UPat.const(None, 0),), allow_any_len=True, name="x"), lambda x: x.const_like(0)),  # NULL pointer load loads 0
    //   (UPat(Ops.STORE, src=(UPat.const(None, 0),), allow_any_len=True), lambda: UOp(Ops.NOOP)),  # NULL pointer store does nothing
    //   # remove NOOPs from SINK
    //   (UPat(Ops.SINK, name="root"),
    //     lambda root: UOp(Ops.SINK, root.dtype, a, root.arg) if len(a:=tuple(x for x in root.src if x.op is not Ops.NOOP)) != len(root.src) else None),
    //   # remove EXPANDs from SINK/BARRIER
    //   (UPat(Ops.BARRIER, src=(UPat((Ops.VECTORIZE, Ops.SINK), name='sink'),)), lambda sink: UOp(Ops.BARRIER, dtypes.void, sink.src)),
    //   (UPat(Ops.SINK, name="root"),
    //     lambda root: UOp(Ops.SINK, root.dtype, tuple(flatten(x.src if x.op in {Ops.SINK, Ops.EXPAND} else (x,) for x in root.src)), root.arg)
    //       if any(x.op in {Ops.SINK, Ops.EXPAND} for x in root.src) else None),
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
  if (allSame(expands_args) && exclude_args.length === 0) {
    //     # if there's only one expand arg, it's okay to use it (optimization)
    expand_args = expands[0].arg
  } // otherwise, we sort them and GEP
  else expand_args = sorted(dedup(flatten(expands_args)) as any).filter((x) => !exclude_args.includes((x as any)[0]))
  const expand_sz = prod(exclude_args.map((x) => x[1]))
  let new_srcs = []
  for (const [i, src] of root.src.entries()) {
    if (src.op === Ops.EXPAND) {
      //         # IF means OR on first arg to IF
      if (root.op === Ops.IF && i === 0) new_srcs.push(range(expand_sz).map((i) => src.src[0].gep(i)).reduce((acc, x) => acc.bitwiseOr(x)))
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
      else if (src.dtype.count > 1) new_srcs.push(new UOp({ op: Ops.VECTORIZE, dtype: src.dtype.scalar().vec(expand_sz * src.dtype.count), src: range(src.dtype.count).map((i) => src.gep(i).mul(expand_sz)) })) //TODO: this src.mul() might not be right
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
  if (ex.op !== Ops.EXPAND) return new UOp({ op: Ops.VECTORIZE, dtype: con.dtype, src: con.src.map((x) => x.mul(con.dtype.count)) }) // TODO: not sure
  //   # CONTRACT may remove several axes from EXPAND
  assert(con.dtype.count === prod(con.arg.map((x) => x[1])), 'dtype is wrong')
  let idxs: number[] = []
  const new_ex_args = ex.arg.filter((x) => !con.arg.includes(x))
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
export const expander = new PatternMatcher<Record<string, UOp>, UOp | undefined>([
  //   # double expand
  //   (UPat(Ops.EXPAND, name="outer", src=(UPat(Ops.EXPAND, name="inner"),)),
  //    lambda outer, inner: UOp(Ops.EXPAND, outer.dtype, (inner.src[0],), inner.arg+outer.arg)),
  //   # do expansion
  //   (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WMMA, Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.ASSIGN,
  //          Ops.VECTORIZE, Ops.IF), name="root", custom_early_reject=set([Ops.EXPAND])), do_expand),
  //   (UPat(Ops.CONTRACT, name="con"), do_contract),
  //   # vectorize DEFINE_ACC
  //   (UPat(Ops.VECTORIZE, src=UPat(Ops.DEFINE_ACC, name="acc"), name="v"), lambda acc,v: acc.replace(dtype=v.dtype)),
  //   # BARRIERs aren't actually expanded
  //   (UPat(Ops.BARRIER, src=(UPat(Ops.EXPAND, name="ex"),)),
  //    lambda ex: UOp(Ops.EXPAND, dtypes.void, (UOp(Ops.BARRIER, dtypes.void, ex.src),)*len(ex.src), ex.arg)),
  //   # empty EXPAND is NOOP
  //   (UPat(Ops.EXPAND, src=(UPat.var('x'),), arg=()), lambda x: x),
  //   # EXPAND GEP (needed for WMMA, generalize this) -> vectorized ALU
  //   (UPat(Ops.EXPAND, name="ex", src=tuple(UPat.var('x').gep(i)+UPat.var('y').gep(i) for i in range(256 if AMX else 8))),
  //     lambda ex,x,y: UOp(Ops.EXPAND, ex.dtype, tuple((x+y).gep(i) for i in range(256 if AMX else 8)), ex.arg)),
])

export const no_vectorized_load_store = (ls: UOp) => {
  let idx = ls.src[0]
  if (!isinstance(idx.dtype, PtrDType)) throw new Error()
  if (idx.dtype.v === 1) return undefined
  const tv = range(idx.dtype.v).map((i) => new UOp({ op: ls.op, dtype: ls.dtype.scalar(), src: ls.src.map((j) => j.gep(i)) }))
  return new UOp({ op: Ops.VECTORIZE, dtype: ls.dtype, src: tv })
}
export const no_vectorized_acc = (acc: UOp) => {
  if (acc.dtype.count === 1) return undefined
  const alus = [new UOp({ op: acc.op, dtype: acc.dtype.scalar(), src: range(acc.dtype.count).flatMap((i) => acc.src.entries().map(([j, s]) => j === 0 ? s.gep(i) : s)) })]
  return new UOp({ op: Ops.VECTORIZE, dtype: acc.dtype, src: alus })
}
export const devectorize = new PatternMatcher([
  //   # no ALU on vectorized dtypes
  //   (UPat((*GroupOp.ALU, Ops.CAST, Ops.BITCAST, Ops.ASSIGN, Ops.INDEX), name="alu"), no_vectorized_alu),
  //   (UPat(Ops.WMMA, name="wmma"), no_vectorized_wmma),
  //   (UPat(Ops.DEFINE_ACC, name="acc"), no_vectorized_acc),
  //   (UPat((Ops.LOAD, Ops.STORE), name="ls"), no_vectorized_load_store),
])

export const delete_redundant_gates = (buf: UOp, idx: UOp, val: UOp, store_gate: UOp, cast?: UOp): undefined | UOp => {
  if (![...val.toposort].filter((gate) => gate.op === Ops.IF).map((gate) => gate.src[0]).includes(store_gate)) return undefined
  //   # remove the gate from the index
  return (isNotNone(cast) ? buf.index(idx).cast(cast.dtype) : buf.index(idx)).store([ val])
}
export const load_store_indexing = new PatternMatcher([
  //   # late fixup of unfoldable image loads
  //   (UPat(Ops.LOAD, src=(UPat.var("buf"), UPat()), allow_any_len=True, name="load"), fix_unfoldable_image_load),
  //   # simplify valid
  //   (UPat(Ops.AND, name="valid"), simplify_valid),
  //   # image load valid idx simplification
  //   (UPat(Ops.INDEX, src=(UPat.var("buf"), UPat.var("start_idx"), UPat.var("valid"))), simplify_valid_load),
  //   # delete_redundant_gates (after expand)
  //   (UPat(Ops.STORE, src=(UPat.any(stidx:=UPat.var("buf").index(UPat.var("idx"), UPat.var("store_gate")), stidx.cast().named("cast")),
  //                                   UPat.var("val"))), delete_redundant_gates),
])

export const migrate_indexing = new PatternMatcher([
  //   # create gate MUST BE BEFORE expander
  //   (UPat(Ops.STORE, name="root"), create_gate),
])

export const move_mask = (x: UOp, buf: UOp, idx: UOp, mask: UOp, cast?: UOp): UOp => {
  //   # this moves the mask from the indexing to the load/store op for rendering
  const nidx = isNotNone(cast) ? buf.index(idx).cast(cast.dtype) : buf.index(idx)
  return x.op === Ops.LOAD ? nidx.load([x.const_like(0), mask, ...x.src.slice(1)], { dtype: x.dtype }) : nidx.store([x.src[1], mask, ...x.src.slice(2)])
}
export const pm_render = new PatternMatcher([
  //   # for rendering, we use explicit VECTORIZE
  //   (UPat(Ops.CONST, name='c'),
  //    lambda c: UOp(Ops.VECTORIZE, c.dtype, (UOp.const(c.dtype.scalar(), c.arg),)*c.dtype.vcount) if c.dtype.vcount > 1 else None),
  //   (UPat(Ops.VCONST, name='c'), lambda c: UOp(Ops.VECTORIZE, c.dtype, tuple(UOp.const(c.dtype.scalar(), x) for x in c.arg))),
  //   (UPat(Ops.GEP, name='gep'), lambda gep: UOp(Ops.VECTORIZE, gep.dtype, tuple(gep.src[0].gep(x) for x in gep.arg)) if len(gep.arg) > 1 else None),
  //   (UPat(Ops.VECTORIZE, src=(UPat(name='x'),)), lambda x: x),
  //   # move masks of loads/stores
  //   (UPat((Ops.LOAD, Ops.STORE), src=(UPat.any(masked_index:=UPat(Ops.INDEX, src=(UPat(name="buf"), UPat(name="idx"), UPat(name="mask"))),
  //                                                masked_index.cast(None).named("cast")),), allow_any_len=True, name="x"), move_mask),
  //   # gate any stores that aren't gated with ifs
  //   (UPat(Ops.STORE, dtype=dtypes.void, src=(UPat(), UPat(), UPat(dtype=dtypes.bool)), name="store"),
  //     lambda store: UOp(Ops.STORE, src=store.src[:2]+(UOp(Ops.IF, src=(store.src[2],)),))),
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
