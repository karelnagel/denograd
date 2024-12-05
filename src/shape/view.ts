import { dtypes } from '../dtype.ts'
import { allInt, argsort, assert, flatten, isEq, isInt, isLessThan, isNone, isNotNone, listStr, prod, range, zip } from '../helpers.ts'
import { gt, le, ne, sint_ceildiv, sint_prod, sint_sorted, smax, smin, symInfer, type Variable } from '../ops.ts'
import { add, ge, idiv, lt, mod, mul, resolve, type sint, sintToUOp, sub, UOp } from '../ops.ts'

export const canonicalize_strides = (shape: sint[], strides: sint[]): sint[] => {
  return shape.map((s, i) => ({ s, st: strides[i] })).map(({ s, st }) => s == 1 ? 0 : st)
}

export const strides_for_shape = (shape: sint[]): sint[] => {
  if (!shape?.length) return []
  const strides = shape.slice(1).reverse().reduce((acc, curr) => [...acc, mul(acc.at(-1)!, curr)], [1 as sint])
  return canonicalize_strides(shape, [...strides].reverse())
}

export const _merge_dims = (shape: sint[], strides: sint[], mask?: [sint, sint][]): [sint, sint, sint][] => {
  // merge contiguous sub-parts or zero strided dims. ret = Tuple[(merged_size, stride, merged size w/o zero stride), ...]
  if (!shape.length) return []
  assert(shape.length === strides.length && (isNone(mask) || shape.length === mask?.length))
  const ret = [[shape[0], strides[0], strides[0] !== 0 ? shape[0] : 0] as [sint, sint, sint]]
  // merge this dim to next dim if size is 1
  let merging = isNotNone(mask) ? sub(mask[0][1], mask[0][0]) === 1 : shape[0] === 1
  for (let i = 1; i < shape.length; i++) {
    const s = shape[i], st = strides[i]
    const [last_s, last_st, last_pre_expand_s] = ret.at(-1)!
    // always merge 1
    if (s === 1) continue
    // merge last dim with this dim if merging or strides matched
    if (merging || last_st === mul(s, st)) ret[ret.length - 1] = [mul(last_s, s), st, st != 0 ? (merging ? s : mul(last_pre_expand_s, s)) : 0]
    else ret.push([s, st, st != 0 ? s : 0])
    // merge this dim to next dim if size is 1
    merging = isNotNone(mask) ? sub(mask[i][1], mask[i][0]) === 1 : s === 1
  }
  return ret
}
const iterator = <T>(items: T[], def: T) => {
  const it = items[Symbol.iterator]()
  return { ...it, next: () => it.next().value || def }
}
/**Returns the new mask if reshape is possible, and None if not possible.*/
export const _reshape_mask = (_mask: undefined | [sint, sint][], old_shape: sint[], new_shape: sint[]): [sint, sint][] | undefined => {
  if (isNone(_mask)) return new_shape.map((s) => [0, s])
  if (_mask.some((m) => typeof m[0] !== 'number' || typeof m[1] !== 'number')) return undefined
  if (_mask.some((m) => lt(sub(m[1], m[0]), 1))) return range(new_shape.length).map((x) => [0, 0]) //zero mask

  const new_mask: [sint, sint][] = []
  // _mask is all int here
  const [r_masks, r_shape, r_new_shape] = [iterator(_mask.toReversed(), [0, 1]), iterator(old_shape.toReversed(), 1), iterator(new_shape.toReversed(), 1)]
  let [curr_stride, old_dim, new_dim, mask] = [1 as sint, r_shape.next(), r_new_shape.next(), r_masks.next()]

  while (new_mask.length < new_shape.length) {
    const [[l, r], next_stride] = [mask, mul(new_dim, curr_stride)]
    if (ge(old_dim, next_stride)) { // need to split mask.
      if (old_dim === next_stride) { // simply copy the mask and get next batch for merging
        new_mask.push([idiv(l, curr_stride), add(idiv(sub(r, 1), curr_stride), 1)])
        ;[curr_stride, old_dim, new_dim, mask] = [1, r_shape.next(), r_new_shape.next(), r_masks.next()]
      } else { // mask can only be splitted if reshape doesn't cut across the mask.
        if (((mod(l, next_stride) !== 0 || mod(r, next_stride) !== 0) && idiv(l, next_stride) !== idiv(sub(r, 1), next_stride)) || mod(old_dim, next_stride) !== 0) return undefined
        new_mask.push([idiv(mod(l, next_stride), curr_stride), idiv(mod(sub(r, 1), next_stride), add(curr_stride, 1))])
        ;[curr_stride, new_dim] = [next_stride, r_new_shape.next()] // need to get mask for next dimension
      }
    } else {
      const next_mask = r_masks.next()
      // combine if the mask can unfold continuously
      if ((mask[0] !== 0 && mask[1] !== old_dim) && sub(next_mask[1], next_mask[0]) !== 1) return undefined
      ;[mask, old_dim] = [[add(mul(next_mask[0], old_dim), l), add(mul(sub(next_mask[1], 1), old_dim), r)], mul(old_dim, r_shape.next())]
    }
  }
  for (const mask of r_masks) { // if the old shape has leading 1s, need to make sure their mask is (0,1)
    if (mask[0] !== 0 || mask[1] !== 1) return range(new_shape.length).map(() => [0, 0]) // invalid mask
  }

  return new_mask.toReversed()
}

export const un1d = (shape: sint[], offs: sint): sint[] => {
  const result: sint[] = []
  for (const stride of strides_for_shape(shape)) {
    const here = stride !== 0 ? idiv(offs, stride) : 0
    result.push(here)
    offs = sub(offs, mul(here, stride))
  }
  return result
}

export class View {
  shape: sint[]
  strides: sint[]
  offset: sint
  mask?: [sint, sint][]
  contiguous: boolean
  // deno-fmt-ignore
  constructor(a: { shape: sint[]; strides: sint[]; offset: sint; mask?: [sint, sint][]; contiguous: boolean }) {
    this.shape=a.shape; this.strides=a.strides; this.offset=a.offset; this.mask=a.mask; this.contiguous=a.contiguous;
  }

  get t() {
    return [...this.shape, ...this.strides, this.offset, ...(isNotNone(this.mask) ? flatten(this.mask) : [])].map((x) => x instanceof UOp ? x.tuplize() : [x])
  }
  lt = (o: View) => isLessThan(this.t, o.t)
  toString = () => `new View({shape:${listStr(this.shape)},strides:${listStr(this.strides)},offset:${this.offset},mask:${listStr(this.mask)},contiguous:${this.contiguous}})`
  to_indexed_uops = (_idxs?: UOp[], vexpr = UOp.const(dtypes.bool, true)): [UOp, UOp] => {
    const idxs = isNone(_idxs) ? this.shape.map((s, i) => UOp.range(dtypes.int, 0, s, i)) : _idxs
    let iexpr = sintToUOp(this.offset)
    for (const [idx, sh, st, m] of zip(idxs, this.shape, this.strides, isNotNone(this.mask) ? this.mask : this.shape.map((x) => undefined))) {
      if (resolve(sh !== 1) && resolve(st !== 0)) iexpr = iexpr.add(idx.mul(st))
      if (isNotNone(m)) {
        if (resolve(m[0] !== 0)) vexpr = vexpr.mul(idx.ge(m[0]))
        if (resolve(m[1] !== sh)) vexpr = vexpr.mul(idx.lt(m[1]))
      }
    }
    return [iexpr, vexpr]
  }
  size = (): number => {
    const ret = prod(this.shape.map((x) => x instanceof UOp ? x.vmax : x))
    assert(typeof ret === 'number', `${ret} is not int`)
    return ret
  }

  static create = (shape: sint[], strides?: sint[], offset: sint = 0, mask?: [sint, sint][]) => {
    // TODO: this resolve shouldn't be needed
    if (!shape.some((s) => resolve(ge(s, 0)))) throw new Error(`Trying to create View with negative dimension: ${shape}`)
    strides = strides ? canonicalize_strides(shape, strides) : strides_for_shape(shape)
    // # canonicalize 0 in shape
    if (shape.includes(0)) return new View({ shape, strides: range(shape.length).map(() => 0), offset: 0, mask: undefined, contiguous: true })
    // # canonicalize empty mask
    if (isNotNone(mask) && zip(mask, shape).every(([m, s]) => isEq(m, [0, s]))) mask = undefined
    // # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
    // # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
    const elim = mask?.map(([b, e]) => !resolve(lt(add(b, 1), e)))
    if (mask && elim?.some((x) => x)) {
      if (mask.some(([b, e]) => !resolve(lt(b, e)))) [strides, offset, mask] = [range(shape.length).map((x) => 0), 0, range(shape.length).map((x) => [0, 0])]
      offset = add(offset, elim.reduce((acc, e, i) => add(acc, e ? mul(strides![i], mask![i][0]) : 0), 0 as sint))
      strides = zip(strides, elim).map(([st, e]) => e ? 0 : st)
    }
    // # simplify as we go
    if (offset instanceof UOp) offset = offset.ssimplify()
    shape = shape.map((x) => x instanceof UOp ? x.ssimplify() : x)
    // # TODO: enabling stride simplification breaks it
    // """
    // strides = tuple(x.ssimplify() if isinstance(x, UOp) else x for x in strides)
    // if mask: mask = tuple((s.ssimplify() if isinstance(s, UOp) else s, e.ssimplify() if isinstance(e, UOp) else e) for s,e in mask)
    // """
    const contiguous = offset === 0 && isNone(mask) && isEq(strides, strides_for_shape(shape))
    return new View({ shape, strides, offset, mask, contiguous })
  }

  vars = (): Variable[] => {
    const flatten_mask = isNotNone(this.mask) ? this.mask.flatMap((m) => m.map((x) => x)) : []
    return [...new Set([...this.shape, ...this.strides, this.offset, ...flatten_mask].filter((x) => x instanceof UOp).reduce((acc, x) => [...acc, ...x.vars()], [] as UOp[]))]
  }
  unbind = (): [View, Map<Variable, number>] => {
    const var_unboundvar_val = this.vars().map((v) => [v, v.unbind()] as const)
    const unbound_vars = new Map(var_unboundvar_val.map(([v, [uv, _]]) => [v, uv]))
    const substitute = (x: sint) => typeof x === 'number' ? x : x.substitute(unbound_vars)
    const new_shape = this.shape.map((x) => substitute(x))
    const new_strides = this.strides.map((x) => substitute(x))
    const new_offset = substitute(this.offset)
    const new_mask = isNotNone(this.mask) ? this.mask.map((x) => [substitute(x[0]), substitute(x[1])] as [sint, sint]) : undefined
    return [View.create(new_shape, new_strides, new_offset, new_mask), Object.fromEntries(var_unboundvar_val.map((x) => x[1]))]
  }
  __add__ = (vm1: View): View | undefined => {
    const vm2 = new View({ ...this })
    if (vm2.contiguous) return vm1
    if (vm1.contiguous && isEq(vm1.shape, vm2.shape)) return vm2
    const ret = vm2.reshape(vm1.shape)
    if (vm1.contiguous && vm1.size() === vm2.size() && isNotNone(ret)) return ret
    if (vm1.mask) {
      for (const [b, e] of vm1.mask) {
        if (resolve(b >= e, false)) return View.create(vm1.shape, range(vm1.shape.length).map(() => 0), 0, range(vm1.shape.length).map((x) => [0, 0] as const))
      }
      const merged = vm2.__add__(vm1.shrink(vm1.mask))
      return merged && merged.pad(zip(vm1.mask, vm1.shape).map(([[b, e], s]) => [b, sub(s, e)]))
    }
    //     # Project vm1's offset and strides on to vm2.
    const origin = un1d(vm2.shape, vm1.offset)
    const terms: [number, sint][][] = origin.map((x) => [])
    const strides: sint[] = range(vm1.shape.length).map((x) => 0)
    for (const [d1, st] of vm1.strides.entries()) {
      if (st === 0) continue
      for (let [d2, [o, s1]] of zip(origin, un1d(vm2.shape, add(vm1.offset, st))).entries()) {
        s1 = sub(s1, o)
        if (s1 === 0) continue
        terms[d2].push([d1, s1])
        strides[d1] = add(strides[d1], mul(s1, vm2.strides[d2]))
      }
    }
    //     # Merge dimensions in vm2 if required.
    //     # NB: Merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required.
    if (!allInt(vm1.shape)) return undefined
    const idxs: UOp[] = [...vm1.shape.map((s, i) => UOp.variable(`idx${i}`, 0, s - 1))]
    let [merged_size, merged_term] = [1, UOp.const(dtypes.int, 0)] as [sint, UOp]
    const extents: [sint, UOp][] = []
    for (const [term, s, o] of zip(terms.toReversed(), vm2.shape.toReversed(), origin.toReversed())) {
      merged_term = merged_term.add(term.map(([d1, s1]) => idxs[d1].mul(s1)).reduce((acc, x) => acc.add(x)).add(o).mul(merged_size))
      merged_size = mul(merged_size, s)
      if (resolve(merged_term < merged_size, false) && resolve(le(0, merged_term), false)) {
        extents.push([merged_size, merged_term])
        ;[merged_size, merged_term] = [1, UOp.const(dtypes.int, 0)]
      }
    }
    if (resolve(merged_term.ne(0))) return undefined
    const vm2_shape = extents.toReversed().map(([s, _]) => s)
    if (!isEq(vm2_shape, vm2.shape)) {
      const reshaped_vm2 = vm2.reshape(vm2_shape)
      if (isNone(reshaped_vm2)) return undefined
      if (!isEq(reshaped_vm2.shape, vm2.shape)) return reshaped_vm2.__add__(vm1)
    }
    if (vm2.mask) {
      //       # Try to project vm2's mask on to vm1.
      let [newb, newe, bad] = [range(vm1.shape.length).map((x) => 0), vm1.shape, false]
      for (const [[b, e], o, term, [_, t]] of zip(vm2.mask, origin, terms, extents.toReversed())) {
        if (resolve(le(b, t.vmin) && lt(t.vmax, e), false)) continue
        if (typeof o !== 'number' || typeof b !== 'number' || typeof e !== 'number') {
          bad = true
          continue
        }
        if (term.length !== 1) {
          if (!term && newe) newe[0] = 0
          else bad = true
          continue
        }

        const [d1, s1] = term[0]
        if (typeof s1 !== 'number' || typeof newe[d1] !== 'number') {
          bad = true
          continue
        }
        newb[d1] = Math.max(newb[d1], Math.ceil((s1 > 0 ? b - o : e - o - 1) / s1))
        newe[d1] = Math.min(newe[d1], idiv(s1 < 0 ? b - o : e - o - 1, s1) + 1)
      }
      //       # If any of vm1 was masked off, try again with that mask in place.
      for (const [b, e, s] of zip(newb, newe, vm1.shape)) {
        if (!isEq([b, e], [0, s])) {
          return vm2.__add__(View.create(vm1.shape, vm1.strides, vm1.offset, zip(newb, newe)))
        }
      }
      //       # Otherwise if vm2's mask was violated, then cannot merge.
      if (bad) return undefined
    }
    return View.create(vm1.shape, strides, add(zip(origin, vm2.strides).reduce((acc, [o, s]) => add(acc, add(o, s)), 0 as sint), vm2.offset))
  }
  invert = (out_shape: sint[]): View | undefined => {
    let ret = View.create(this.shape)
    if (this.mask) ret = ret.shrink(this.mask)
    ret = ret.stride(this.strides.map((x) => lt(x, 0) ? -1 : 1)).permute(argsort(this.strides.map((x) => gt(x, 0) ? -x : x)))
    return isEq(sint_prod(ret.shape), sint_prod(out_shape)) ? ret : undefined // don't support shrink, expand, or stride != (-1, 1)
  }
  minify = () => this.reshape(_merge_dims(this.shape, this.strides, this.mask).map((x) => x[0])) || this
  __unsafe_resize = (arg: [sint, sint][], mask?: [sint, sint][]): View => {
    const offset = zip(this.strides, arg).reduce((acc, [s, x]) => add(acc, mul(s, x[0])), 0 as sint)
    if (this.mask) {
      //       # move the old mask
      const nmask = zip(this.mask, arg).map(([[mx, my], [ax, ay]]) => [smax(0, smin(sub(mx, ax), sub(ay, ax))), smax(0, smin(sub(my, ax), sub(ay, ax)))] as [sint, sint])
      //       # merge the masks if we have two
      mask = isNotNone(mask) ? zip(nmask, mask).map(([[mx1, my1], [mx2, my2]]) => [smax(mx1, mx2), smin(my1, my2)] as [sint, sint]) : nmask
    }
    const shape = arg.map(([x, y]) => sub(y, x))
    if (isNotNone(mask) && zip(mask, shape).every(([m, s]) => m[0] === 0 && m[1] === s)) mask = undefined
    return View.create(shape.map((s) => s instanceof UOp ? s.ssimplify() : s), this.strides, add(this.offset, offset), mask)
  }
  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  pad = (arg: [sint, sint][]): View => {
    assert(arg.length === this.shape.length, `invalid pad ${arg} for ${this.shape}`)
    //     # NOTE: not checking for symbolic arg
    for (const [b, e] of arg) assert((typeof b !== 'number' || typeof e !== 'number') || (b >= 0 && e >= 0), `invalid pad ${arg} for ${this.shape}`)
    if (arg.some(([b, e]) => resolve(ne(b, 0)) || resolve(ne(e, 0)))) {
      const zvarg = zip(this.shape, arg).map(([s, [b, e]]) => [mul(b, -1), add(s, e)] as [sint, sint])
      const mask = zip(this.shape, arg).map(([s, [b, _]]) => [b, add(s, b)] as [sint, sint])
      return this.__unsafe_resize(zvarg, mask)
    }
    return this
  }
  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  shrink = (arg: [sint, sint][]): View => {
    assert(arg.length === this.shape.length, `invalid shrink ${arg} for ${this.shape}`)
    // # NOTE: not checking for symbolic arg
    for (const [s, [b, e]] of zip(this.shape, arg)) assert(!(isInt(b) && isInt(e) && isInt(s)) || (0 <= b && b <= e && e <= s), `invalid shrink ${arg} for ${this.shape}`)
    return this.__unsafe_resize(arg)
  }
  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  expand = (new_shape: sint[]): View => {
    assert(new_shape.length === this.shape.length, `expand arg ${new_shape} must have same number of dimensions as shape ${this.shape}`)
    if (this.shape.includes(0)) {
      assert(zip(this.shape, new_shape).every(([s, x]) => (s === x && x === 0) || (gt(s, 0) && mod(x, s) === 0)), `can't expand ${this.shape} into ${new_shape}`)
      return View.create(new_shape)
    }
    //     # TODO: this resolve might be wrong
    assert(zip(this.shape, new_shape).every(([s, x]) => !resolve(ne(s, x), false) || s === 1), `can't expand ${this.shape} into ${new_shape}`)
    //     # NOTE: can the mask ever be (0,0)?
    //     # TODO: this resolve may not be needed, but it's hard because vars need to be sorted
    const mask = this.mask ? zip(this.mask, this.shape, new_shape).map(([m, s, ns]) => resolve(ne(s, ns), false) ? (!isEq(m, [0, 1]) ? [0, 0] : [0, ns]) as [sint, sint] : m) : undefined
    return View.create(new_shape, this.strides, this.offset, mask)
  }

  permute = (axis: number[]): View => {
    assert(isEq(sint_sorted(axis), range(this.shape.length)), `invalid permutation ${axis} of len ${this.shape.length}`)
    return View.create(axis.map((a) => this.shape[a]), axis.map((a) => this.strides[a]), this.offset, isNotNone(this.mask) ? axis.map((a) => this.mask![a]) : undefined)
  }
  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  stride = (multi: number[]): View => {
    //     # except for the negative case, you can build this from the others. invertible in the negative case
    assert(multi.every((x) => typeof x === 'number' && x !== 0), `invalid stride ${mul} for ${this.shape}`)
    const strides = zip(this.strides, multi).map(([z, m]) => mul(z, m))
    const new_shape = zip(this.shape, multi).map(([s, m]) => sint_ceildiv(s, Math.abs(m)))
    const offset = zip(this.shape, this.strides, multi).filter(([s, z, m]) => m < 0).reduce((acc, [s, z, m]) => add(acc, mul(sub(s, 1), z)), 0 as sint)
    const mask = isNotNone(this.mask) ? zip(this.mask, this.shape, multi).map(([[mx, my], s, m]) => [sint_ceildiv(m > 0 ? mx : sub(s, my), Math.abs(m)), sint_ceildiv(m > 0 ? my : sub(s, mx), Math.abs(m))] as [sint, sint]) : undefined
    return View.create(new_shape, strides, add(this.offset, offset), mask)
  }

  reshape = (new_shape: sint[]): View | undefined => {
    if (isEq(this.shape, new_shape)) return this

    //     # TODO: this resolve shouldn't be needed
    assert(new_shape.every((x) => resolve(ge(x, 0))), `shape can't contain negative numbers ${new_shape}`)
    if (this.shape.includes(0)) {
      assert(new_shape.includes(0), `cannot reshape 0 size to ${new_shape}`)
      return View.create(new_shape)
    }
    //     # check for the same size
    const self_all_int = allInt(this.shape)
    if (self_all_int) {
      assert(new_shape.every((s) => s instanceof UOp || typeof s === 'number'), `${this.shape} -> ${new_shape} contains non (int, Variable) dim`)
      if (resolve(ne(sint_prod(this.shape), sint_prod(new_shape)), false)) throw new Error(`size mismatched, can't reshape ${this.shape} -> ${new_shape}`)
    }
    if (new_shape.length === 0 && this.mask && this.mask.some(([mx, my]) => mx === my)) return undefined

    //     # after the asserts, it's okay to check contiguous
    if (this.contiguous) return View.create(new_shape)

    //     # if it's not contiguous and new shape is symbolic, check if it's directly replaceable
    if (self_all_int && !allInt(new_shape)) {
      if (this.shape.length !== new_shape.length) throw new Error(`cannot symbolic reshape non-contiguous ${this} -> ${new_shape}`)
      for (let [si, so] of zip(this.shape, new_shape)) {
        if (typeof so !== 'number') so = symInfer(so, new Map([...so.vars()].map((v) => v.unbind())))
        if (si !== so) throw new Error(`cannot symbolic reshape non-contiguous ${this} -> ${new_shape}`)
        //       # all dimensions matched, return the new view directly
      }
      return new View({ shape: new_shape, strides: this.strides, offset: this.offset, mask: this.mask, contiguous: this.contiguous })
    }
    let [strides, r_new_shape] = [[] as sint[], new_shape.toReversed()]
    let exitedWithBreak = false
    for (let [merged_dim, new_stride, real_dim] of _merge_dims(this.shape, this.strides, this.mask).toReversed()) {
      let acc = 1 as sint
      //       # TODO: third resolve shouldn't be needed
      for (const new_dim of r_new_shape) {
        if (!(resolve(le(acc, merged_dim)) && resolve(ne(acc, merged_dim)) && resolve(ge(new_dim, 0)))) {
          exitedWithBreak = true
          break
        }
        strides.push(new_stride)
        if (resolve(ne(new_dim, 1))) {
          acc = mul(acc, new_dim)
          new_stride = mul(new_stride, resolve(lt(acc, real_dim)) ? new_dim : 0)
        }
      }
      if (resolve(ne(acc, merged_dim))) {
        exitedWithBreak = true
        break
      }
    }
    if (!exitedWithBreak) {
      strides = [...strides, ...range(new_shape.length - strides.length).map((x) => 0)]
      const new_mask = _reshape_mask(this.mask, this.shape, new_shape)
      if (isNotNone(new_mask)) {
        const new_strides = canonicalize_strides(new_mask.map(([b, e]) => sub(e, b)), strides.toReversed())
        const extra_offset = sub(
          this.mask ? zip(this.mask, this.strides).reduce((acc, [m, s]) => add(acc, mul(m[0], s)), 0 as sint) : 0,
          zip(new_mask, new_strides).reduce((acc, [m, s]) => add(acc, mul(m[0], s)), 0 as sint),
        )
        return View.create(new_shape, new_strides, add(this.offset, extra_offset), new_mask)
      }
    }
    return undefined
  }
}
