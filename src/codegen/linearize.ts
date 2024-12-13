import { dtypes, PtrDType } from '../dtype.ts'
import { assert, dedup, isEq, isinstance, isLessThan, isNotNone, len, min, partition, setDefault } from '../helpers.ts'
import { graph_rewrite, GroupOp, Ops, opsString, PatternMatcher, UOp, UPat } from '../ops.ts'

const DONT_PLACE_IN_BLOCK = [Ops.DEFINE_GLOBAL, Ops.DEFINE_LOCAL, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.CONST, ...GroupOp.Block]

export const disp = (y: UOp): string => {
  if (y.op === Ops.BLOCKSTART) return 'w' + disp(y.src[0])
  if (y.op === Ops.IF) return `IF${y.key}` // TODO: not sure
  if (y.op === Ops.RANGE) return y.arg.toString()
  return '<NONE>'
}

class BasicBlock {
  ctx: UOp[]
  lst: UOp[]
  end?: UOp
  // deno-fmt-ignore
  constructor(ctx:UOp[],lst:UOp[],end?:UOp){ this.ctx=ctx; this.lst=lst; this.end=end }
  lt = (o: BasicBlock) => isLessThan([...this.ctx, ...this.lst].map((x) => x.tuplize()), [...o.ctx, ...o.lst].map((x) => x.tuplize()))
  toString = () => `${isNotNone(this.end) ? (disp(this.end) + ' ') : ''}` + `${this.ctx.map((y) => disp(y))} ${len(this.lst)}` + '\n' + this.lst.map((x) => opsString(x.op)).join('\n')
}
type CTX = [Map<UOp, UOp[]>, Map<UOp, UOp[]>]
export const append_to_block = (ctx: CTX, x: UOp): UOp | undefined => {
  const [block_ctxs, children] = ctx
  const in_this_block = new Set(x.arg.lst)

  //   # collections to build
  let new_srcs: UOp[] = []
  const to_append: UOp[] = []
  const old_blocks = new Map<UOp[], UOp>()
  const new_blocks = new Map<UOp[], UOp[]>()

  for (const u of x.src) {
    if (u.op === Ops.BLOCK) {
      //       # merge sibling blocks. NOTE: blocks must only have one output source
      assert(!old_blocks.has(u.arg.ctx), 'sibiling should never have been created')
      old_blocks.set(u.arg.ctx, u)
    } else if (!DONT_PLACE_IN_BLOCK.includes(u.op) && new Set(children.get(u)).isSubsetOf(in_this_block)) {
      //       # if it can go in blocks and all its children are in the block, we add it to the block
      const block_ctx = block_ctxs.get(u)
      if (block_ctx === x.arg.ctx) {
        //         # if it's the same context, we place the UOp in this block and append the parents to its srcs
        new_srcs = [...new_srcs, ...u.src]
        to_append.push(u)
      } //         # if it's a different context, we create a new block with this UOp

      else setDefault(new_blocks, block_ctx, []).push(u)
    } //       # otherwise, we keep it in the srcs
    else new_srcs.push(u)
  }
  if (to_append.length === 0 && new_blocks.size === 0) return undefined

  for (let [rng, lst] of new_blocks.entries()) {
    let srcs = lst.flatMap((y) => y.src)
    const old_block = old_blocks.get(rng)
    old_blocks.delete(rng)
    if (isNotNone(old_block)) {
      //       # NOTE: order shouldn't matter here
      srcs = [...srcs, ...old_block.src]
      lst = [...lst, old_block.arg.lst]
    }
    let new_block = new UOp({ op: Ops.BLOCK, dtype: dtypes.void, src: dedup(srcs), arg: new BasicBlock(rng, lst) })
    let lrng = rng
    for (const r of rng.toReversed()) {
      if (!x.arg.ctx.includes(r) && r.op !== Ops.BLOCKSTART) {
        lrng = lrng.filter((x) => x !== r)
        new_block = new UOp({ op: Ops.BLOCKEND, src: [new_block], arg: new BasicBlock(lrng, [new UOp({ op: r.op === Ops.IF ? Ops.ENDIF : Ops.ENDRANGE, src: [r] })], r) })
      }
    }
    new_srcs.push(new_block)
  }
  return new UOp({ op: Ops.BLOCK, dtype: dtypes.void, src: dedup([...old_blocks.values(), ...new_srcs]), arg: new BasicBlock(x.arg.ctx, [...to_append, ...x.arg.lst]) })
}
export const make_basic_blocks = new PatternMatcher<Record<string, UOp> & { ctx: CTX }, UOp | undefined>([
  [new UPat({ op: Ops.SINK, name: 'x' }), ({ x }) => new UOp({ op: Ops.BLOCK, src: x.src, arg: new BasicBlock([], [x]) })],
  [new UPat({ op: Ops.BLOCK, name: 'x' }), ({ ctx, x }) => append_to_block(ctx, x)],
])

export const block_merge = (ctx: Map<UOp, UOp[]>, x: UOp): UOp | undefined => {
  if (!(x.arg instanceof BasicBlock)) throw new Error('Has to be basic block,maybe??')
  //   # ctx is children here
  if (x.op === Ops.BLOCKEND) {
    //     # if it's a BLOCKEND, see if we are done with placement. if all the children of the range are in here
    const in_this_block = new Set(x.arg.lst)
    if (ctx.get(x.arg.end!)!.filter((y) => !in_this_block.has(y)).length === 0) {
      //       # find the parent block that has the BLOCKSTART in the ctx
      const parent_blocks = x.src.filter((y) => y.op === Ops.BLOCK && y.arg.ctx.includes(new UOp({ op: Ops.BLOCKSTART, src: [x.arg.end] })))
      assert(parent_blocks.length <= 1, 'should never have two parent blocks')
      if (parent_blocks.length === 1) {
        const parent_block = parent_blocks[0]
        // range needs DEFINE_ACC to be before the range (never in DEFINE_ACC for if)
        const [early_ops, late_ops] = partition(x.arg.lst, (y) => y.op === Ops.DEFINE_ACC && y.src.includes(x.arg.end))
        return new UOp({ op: Ops.BLOCK, dtype: dtypes.void, src: [...x.src.filter((y) => y !== parent_block), ...parent_block.src], arg: new BasicBlock(x.arg.ctx.filter((y) => y !== x.arg.end), [...early_ops, ...parent_block.arg.lst, ...late_ops]) })
      }
    }
  }
  let new_srcs: UOp[] = []
  let to_append: UOp[] = []
  let new_ctx = x.arg.ctx
  const placed = new Set()
  for (const u of x.src) {
    if (u.op === Ops.BLOCK && (isEq(u.arg.ctx, x.arg.ctx) || (isNotNone(x.arg.end) && u.arg.ctx.includes(x.arg.end)))) {
      //       # NOTE: this can't appear in srcs twice or it would be a BLOCKFORK
      new_ctx = [...new_ctx, ...u.arg.ctx.filter((y: UOp) => !x.arg.ctx.includes(y))]
      new_srcs = [...new_srcs, ...u.src]
      to_append = [...to_append, ...u.arg.lst]
    } else if (u.op === Ops.BLOCKFORK && x.src.filter((a) => a === u).length === u.arg) { // block fork appears # of times in srcs
      if (!placed.has(u)) {
        new_srcs = [...new_srcs, ...u.src]
        placed.add(u)
      }
    } else {
      //       # keep it in srcs
      new_srcs.push(u)
    }
  }
  if (to_append.length === 0 && placed.size === 0) return undefined
  return new UOp({ op: x.op, dtype: dtypes.void, src: new_srcs, arg: new BasicBlock(new_ctx.toSorted((a, b) => isLessThan(a.tuplize(), b.tuplize()) ? -1 : 1), /**TODO:not sure about sort */ [...to_append, ...x.arg.lst], x.arg.end) })
}
export const pm_block_merge = new PatternMatcher<{ctx:Map<UOp,UOp[]>,x:UOp}>([[new UPat({ op: [Ops.BLOCKEND, Ops.BLOCK], name: 'x' }), ({ ctx, x }) => block_merge(ctx, x)]])

// # NOTE: any toposort should be valid here, unlike last time this isn't required, it's just for speed
export const block_reorder = (in_block: UOp): UOp => {
  const in_this_block = new Set(in_block.arg.lst)
  const local_children = new Map<UOp, UOp[]>()
  const in_degree = new Map<UOp, number>()
  const priorities = new Map<UOp, number>()

  //   # get local children and assign priorities
  for (const u of in_block.arg.lst.toReversed()) {
    for (const s of u.src) {
      if (in_this_block.has(u)) {
        local_children.get(s)!.push(u)
        in_degree.set(u, in_degree.get(u)! + 1)
      }
    }
    //     # put loads in the beginning of the block and prevent priority inversion
    priorities.set(u, min([u.op === Ops.LOAD ? -1000 : 0, ...local_children.get(u)!.map((x) => priorities.get(x)!)]))
  }

  //   # placement queue
  const queue: UOp[] = [];
  const push = (u: UOp) => {
    queue.push(u);
    queue.sort((a, b) => {
      const priA = priorities.get(a) || 0;
      const priB = priorities.get(b) || 0;
      if (priA !== priB) return priA - priB;
      // Compare tuplize as secondary sort key
      // Assuming tuplize comparison works similar to Python
      return JSON.stringify(a.tuplize) < JSON.stringify(b.tuplize) ? -1 : 1;
    });
  };

  //   # place the first ones that don't have deps
  for (const u of in_block.arg.lst) if (!in_degree.has(u)) push(u)

  const newlst = []
  while (queue.length) {
    const x = queue.shift()!;
    newlst.push(x)
    for (const u of local_children.get(x)!) {
      in_degree.set(u, in_degree.get(u)! - 1)
      if (in_degree.get(u) === 0) push(u)
    }
  }
  assert(newlst.length === in_block.arg.lst.length, `len mismatch ${len(newlst)} != ${len(in_block.arg.lst)}`)
  return in_block.replace({ arg: new BasicBlock(in_block.arg.ctx, newlst) })
}

export const linearize_uop = (sink: UOp, skip_check = false): UOp[] => {
  assert(sink.op === Ops.SINK, `sink isn't sink, it's ${sink.op}`)

  //   # get children and all block contexts
  const temp_block_ctxs = new Map<UOp, UOp[]>()
  const children = new Map<UOp, UOp[]>()
  for (const u of sink.toposort) {
    let this_block_ctx: UOp[] = []
    for (const s of u.src) {
      //       # save children
      setDefault(children, s, []).push(u)
      //       # compute block ctx
      if ([Ops.RANGE, Ops.IF].includes(s.op)) this_block_ctx.push(s)
      //       # don't flow (fully) through assign and store
      else if (s.op === Ops.STORE) {
        //         # ugh, deal with non-reduce locals. probably wrong
        if (isinstance(s.src[0].dtype, PtrDType) && s.src[0].dtype.local) {
          const [idx_context, store_context] = [temp_block_ctxs.get(s.src[0]), temp_block_ctxs.get(s)]
          this_block_ctx = [...this_block_ctx, ...store_context!.filter((x) => !idx_context!.includes(x) && x.op === Ops.RANGE)]
        }
      } else if (s.op === Ops.ASSIGN) {
        //         # flow though assign, but remove the ranges used in the assign
        //         assert s.src[0].op is Ops.DEFINE_ACC
        //         this_block_ctx += [x for x in temp_block_ctxs[s.src[1]] if x not in s.src[0].src[1:]]
      } //         # flow though everything else

      else this_block_ctx = [...this_block_ctx, ...temp_block_ctxs.get(s)!]
    }
    temp_block_ctxs.set(u, dedup(this_block_ctx).toSorted((a, b) => isLessThan(a.tuplize(), b.tuplize()) ? -1 : 1))
  }
  //   # make final block_ctxs, add BLOCKSTART to block_ctxs for IF and RANGE
  const block_ctxs = new Map<UOp, UOp[]>()
  for (const u of sink.toposort) {
    block_ctxs.set(u, [new UOp({ op: Ops.BLOCKSTART, src: [u] }), ...([Ops.IF, Ops.RANGE].includes(u.op) ? temp_block_ctxs.get(u)! : temp_block_ctxs.get(u)!)])
  }

  //   # TODO: there's probably a clever way to remove this while loop
  while (true) {
    sink = graph_rewrite(sink, make_basic_blocks, [block_ctxs, children])

    // # add BLOCKFORK (slow!)
    const block_parent_count = [...sink.toposort].filter((x) => x.op === Ops.BLOCK).flatMap((x) => x.src).reduce((acc, src) => {
      acc.set(src, (acc.get(src) || 0) + 1)
      return acc
    }, new Map<UOp, number>())
    const non_block_parents = new Set([...sink.toposort].filter((x) => x.op !== Ops.BLOCK).flatMap((x) => x.src))
    const forks = new Map<UOp, UOp>(
      block_parent_count.entries().filter(([u, child_count]) => !DONT_PLACE_IN_BLOCK.includes(u.op) && child_count > 1 && !non_block_parents.has(u))
        .map(([u, child_count]) => [u, new UOp({ op: Ops.BLOCKFORK, src: [new UOp({ op: Ops.BLOCK, src: u.src, arg: new BasicBlock(block_ctxs.get(u)!, [u]) })], arg: child_count })]),
    )
    if (!forks.size) break
    sink = sink.substitute(forks)
  }
  //   # combine matching BLOCKENDS
  const blockends_to_arg = new Map<UOp, UOp[]>()
  for (const be of sink.toposort) if (be.op === Ops.BLOCKEND) setDefault(blockends_to_arg, be.arg.end, []).push(be)
  const new_forks = new Map<UOp, UOp>()
  for (const [k, v] of blockends_to_arg.entries()) {
    //     # NOTE: if any BLOCKEND is the parent of any other with the same arg, this algo fails
    if (v.length > 1) {
      const out = new UOp({ op: Ops.BLOCKFORK, src: [new UOp({ op: Ops.BLOCKEND, src: v.flatMap((x) => x.src), arg: new BasicBlock(dedup(v.flatMap((y) => y.arg.ctx)), v[0].arg.lst, k) })], arg: v.length })
      for (const u of v) new_forks.set(u, out)
    }
  }
  sink = sink.substitute(new_forks)

  //   # reorder ops in block for speed
  sink = sink.substitute(new Map([...sink.toposort].filter((u) => u.op === Ops.BLOCK).map((u) => [u, block_reorder(u)] as [UOp, UOp]).filter(([u, newu]) => newu !== u)))

  //   # final rewrite to merge all blocks into one
  sink = graph_rewrite(sink, pm_block_merge, children)

  //   # there should just be one block left, with a few parents with 0 srcs
  assert(sink.op === Ops.BLOCK)
  let _uops = dedup(sink.src).toSorted((a, b) => isLessThan(a.tuplize(), b.tuplize()) ? -1 : 1)
  assert(_uops.every((x) => x.src.length === 0 && ![Ops.BLOCK, Ops.BLOCKSTART, Ops.BLOCKEND, Ops.BLOCKFORK].includes(x.op)))
  _uops = [..._uops, ...sink.arg.lst]

  //   # sanity checks (NOTE: these can cause things to be skipped in BEAM)
  //   if (!skip_check) type_verify(_uops)

  //   # strip the SINK
  return _uops.slice(0, -1)
}
