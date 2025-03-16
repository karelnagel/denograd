// deno-lint-ignore-file no-this-alias
import { dtypes } from '../dtype.ts'
import { all_int, argsort, assert, cache, cache_fn, flatten, get_key, is_eq, isInt, list_str, next, range, sorted, WeakValueMap, zip } from '../helpers.ts'
import { resolve, type sint, sint_to_uop, smax, smin, sym_infer, UOp, type Variable } from '../ops.ts'
import { add, and, ceildiv, ge, gt, idiv, le, lt, mod, mul, ne, neg, prod, sub, sum } from '../helpers.ts'

export const canonicalize_strides = cache_fn((shape: sint[], strides: sint[]): sint[] => {
  return shape.map((s, i) => ({ s, st: strides[i] })).map(({ s, st }) => s === 1 ? 0 : st)
})

export const strides_for_shape = cache_fn((shape: sint[]): sint[] => {
  if (!shape?.length) return []
  const strides = shape.slice(1).reverse().reduce((acc, curr) => [...acc, mul(acc.at(-1)!, curr)], [1 as sint])
  return canonicalize_strides(shape, [...strides].reverse())
})

export const merge_dims = cache_fn((shape: sint[], strides: sint[], mask?: [sint, sint][]): [sint, sint, sint][] => {
  // merge contiguous sub-parts or zero strided dims
  // any stride 0, masked from dim=1, or contiguous part is merged into next dim.
  // stride != 0 to stride == 0 starts a new merging block
  // ret = tuple[(merged_size, stride, merged size w/o zero stride), ...]
  if (!shape.length) return []
  assert(shape.length === strides.length && (mask === undefined || shape.length === mask?.length))
  const ret = [[shape[0], strides[0], strides[0] !== 0 ? shape[0] : 0] as [sint, sint, sint]]
  // merge this dim to next dim if size is 1
  let merging = mask !== undefined ? sub(mask[0][1], mask[0][0]) === 1 : shape[0] === 1
  for (const [i, [s, st]] of zip(shape, strides).entries()) {
    if (i === 0) continue // skipping the first, in py start=1
    // always merge 1
    if (s === 1) continue
    const [last_s, last_st, last_pre_expand_s] = ret.at(-1)!
    // merge last dim with this dim if merging or strides matched
    if (merging || last_st === mul(s, st)) ret[ret.length - 1] = [mul(last_s, s), st, merging ? s : mul(last_pre_expand_s, s)]
    else ret.push([s, st, s])
    // merge this dim to next dim if size is 1
    merging = mask !== undefined ? sub(mask[i][1], mask[i][0]) === 1 : s === 1
  }
  return ret
})

export const _reshape_mask = cache_fn((_mask: undefined | [sint, sint][], old_shape: sint[], new_shape: sint[]): [sint, sint][] | undefined => {
  if (_mask === undefined) return new_shape.map((s) => [0, s])
  if (!all_int(flatten(_mask))) return undefined

  const new_mask: [sint, sint][] = []
  // _mask is all int here
  const r_masks = _mask.toReversed().values(), r_shape = old_shape.toReversed().values(), r_new_shape = new_shape.toReversed().values()
  let curr_stride: sint = 1, old_dim = next(r_shape, 1), new_dim = next(r_new_shape, 1), mask = next(r_masks, [0, 1])
  while (new_mask.length < new_shape.length) {
    const [l, r] = mask, next_stride: sint = mul(new_dim, curr_stride)
    // need to split mask.
    if (old_dim === next_stride) { // simply copy the mask and get next batch for merging
      new_mask.push([idiv(l, curr_stride), add(idiv(sub(r, 1), curr_stride), 1)])
      curr_stride = 1, old_dim = next(r_shape, 1), new_dim = next(r_new_shape, 1), mask = next(r_masks, [0, 1])
    } else if (old_dim > next_stride) {
      if (mod(old_dim, next_stride) !== 0) return undefined
      if ((mod(l, next_stride) !== 0 || mod(r, next_stride) !== 0) && idiv(l, next_stride) !== idiv(sub(r, 1), next_stride)) return undefined
      new_mask.push([idiv(mod(l, next_stride), curr_stride), add(idiv(mod(sub(r, 1), next_stride), curr_stride), 1)])
      curr_stride = next_stride, new_dim = next(r_new_shape, 1) // need to get mask for next dimension
    } else {
      const next_mask = next(r_masks, [0, 1])
      // combine if the mask can unfold continuously
      if (!is_eq(mask, [0, old_dim]) && l !== r && (next_mask[1] as number) - (next_mask[0] as number) !== 1) return undefined
      mask = [add(mul(next_mask[0], old_dim), l), add(mul(sub(next_mask[1], 1), old_dim), r)], old_dim = mul(old_dim, next(r_shape, 1))
    }
  }

  return new_mask.toReversed()
})

export const unravel = (shape: sint[], offset: sint): sint[] => {
  // find the position of offset on each dimension based on shape
  // similar to unravel_index in numpy/torch
  let acc: sint = 1, idxs = []
  for (const d of shape.toReversed()) {
    idxs.push(mod(idiv(offset, acc), d))
    acc = mul(acc, d)
  }
  return idxs.toReversed()
}

export class View {
  key: string
  static cache = new WeakValueMap<string, View>()
  constructor(public shape: sint[], public strides: sint[], public offset: sint, public mask?: [sint, sint][], public contiguous?: boolean) {
    this.key = get_key(shape, strides, offset, mask, contiguous)
    Object.freeze(this)
    return View.cache.setDefault(this.key, this)
  }

  toString() {
    return `new View(${list_str(this.shape)}, ${list_str(this.strides)}, ${this.offset}, ${list_str(this.mask)}, ${this.contiguous})`
  }
  [Symbol.for('nodejs.util.inspect.custom')](_depth: number, _options: any) {
    return this.toString()
  }
  to_indexed_uops(_idxs?: UOp[], vexpr = UOp.const(dtypes.bool, true)): [UOp, UOp] {
    const idxs = _idxs === undefined ? this.shape.map((s, i) => UOp.range(dtypes.int, 0, s, i)) : _idxs
    let iexpr = sint_to_uop(this.offset)
    for (const [idx, sh, st, m] of zip(idxs, this.shape, this.strides, this.mask !== undefined ? this.mask : this.shape.map((x) => undefined))) {
      if (resolve(ne(sh, 1)) && resolve(ne(st, 0))) iexpr = iexpr.add(idx.mul(st))
      if (m !== undefined) {
        if (resolve(ne(m[0], 0))) vexpr = vexpr.mul(idx.ge(m[0]))
        if (resolve(ne(m[1], sh))) vexpr = vexpr.mul(idx.lt(m[1]))
      }
    }
    return [iexpr, vexpr]
  }
  size = cache((): number => {
    const ret = prod(this.shape.map((x) => x instanceof UOp ? x.vmax : x))
    if (typeof ret !== 'number') throw new Error(`${ret} is not int`)
    return ret
  })
  static create = cache((shape: sint[], strides?: sint[], offset: sint = 0, mask?: [sint, sint][]) => {
    // TODO: this resolve shouldn't be needed
    if (!shape.every((s) => resolve(ge(s, 0)))) throw new Error(`Trying to create View with negative dimension: shape=${shape}`)
    strides = strides?.length ? canonicalize_strides(shape, strides) : strides_for_shape(shape)
    // canonicalize 0 in shape
    if (shape.includes(0)) return new View(shape, range(shape.length).map(() => 0), 0, undefined, true)
    // canonicalize no-op mask
    if (mask !== undefined && zip(mask, shape).every(([m, s]) => is_eq(m, [0, s]))) mask = undefined
    // if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
    // then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
    const elim = mask?.map(([b, e]) => !resolve(lt(add(b, 1), e)))
    if (mask?.length && elim?.some((x) => x)) {
      if (mask.some(([b, e]) => !resolve(lt(b, e)))) [strides, offset, mask] = [range(shape.length).map((x) => 0), 0, range(shape.length).map((x) => [0, 0])]
      offset = add(offset, elim.reduce((acc, e, i) => add(acc, e ? mul(strides![i], mask![i][0]) : 0), 0 as sint))
      strides = zip(strides, elim).map(([st, e]) => e ? 0 : st)
    }
    // simplify as we go
    if (offset instanceof UOp) offset = offset.ssimplify()
    shape = shape.map((x) => x instanceof UOp ? x.ssimplify() : x)
    // TODO: enabling stride simplification breaks symbolic jit
    // """
    // strides = tuple(x.ssimplify() if isinstance(x, UOp) else x for x in strides)
    // if mask: mask = tuple((s.ssimplify() if isinstance(s, UOp) else s, e.ssimplify() if isinstance(e, UOp) else e) for s,e in mask)
    // """
    const contiguous = offset === 0 && mask === undefined && is_eq(strides, strides_for_shape(shape))
    return new View(shape, strides, offset, mask, contiguous)
  })
  vars = cache((): Variable[] => {
    const flatten_mask = this.mask !== undefined ? this.mask.flatMap((m) => m.map((x) => x)) : []
    return [...new Set([...this.shape, ...this.strides, this.offset, ...flatten_mask].filter((x) => x instanceof UOp).reduce((acc, x) => [...acc, ...x.vars()], [] as UOp[]))]
  })
  unbind = cache((): [View, Map<Variable, number>] => {
    const var_unboundvar_val = this.vars().map((v) => [v, v.unbind()] as const)
    const unbound_vars = new Map(var_unboundvar_val.map(([v, [uv, _]]) => [v, uv]))
    const substitute = (x: sint) => typeof x === 'number' ? x : x.substitute(unbound_vars)
    const new_shape = this.shape.map((x) => substitute(x))
    const new_strides = this.strides.map((x) => substitute(x))
    const new_offset = substitute(this.offset)
    const new_mask = this.mask !== undefined ? this.mask.map((x) => [substitute(x[0]), substitute(x[1])] as [sint, sint]) : undefined
    return [View.create(new_shape, new_strides, new_offset, new_mask), new Map(var_unboundvar_val.map((x) => x[1]))]
  })
  add = cache((vm1: View): View | undefined => {
    const vm2 = this
    if (vm2.contiguous) return vm1
    if (vm1.contiguous && is_eq(vm1.shape, vm2.shape)) return vm2
    if (vm1.contiguous && vm1.size() === vm2.size()) {
      const ret = vm2.reshape(vm1.shape)
      if (ret !== undefined) return ret
    }
    if (vm1.mask?.length) {
      for (const [b, e] of vm1.mask) {
        if (resolve(ge(b, e), false)) return View.create(vm1.shape, range(vm1.shape.length).map(() => 0), 0, range(vm1.shape.length).map((x) => [0, 0]))
      }
      const merged = vm2.add(vm1.shrink(vm1.mask))
      return merged && merged.pad(zip(vm1.mask, vm1.shape).map(([[b, e], s]) => [b, sub(s, e)]))
    }
    if (vm1.mask?.length) {
      const new_vm1 = vm1.shrink(vm1.mask), merged = vm2.add(new_vm1)
      if (new_vm1 === vm1 || merged === undefined) return undefined
      return merged.pad(zip(vm1.mask, vm1.shape).map(([[b, e], s]) => [b, sub(s, e)]))
    }
    if (!all_int(vm1.shape)) return undefined

    // Project vm1's offset and strides on to vm2.
    const origin = unravel(vm2.shape, vm1.offset)
    const terms: [number, sint][][] = vm2.shape.map(() => [])
    const strides: sint[] = range(vm1.shape.length).map((x) => 0)
    for (const [d1, st] of vm1.strides.entries()) {
      if (st === 0) continue
      for (let [d2, [o, s1]] of zip(origin, unravel(vm2.shape, add(vm1.offset, st))).entries()) {
        s1 = sub(s1, o)
        if (s1 === 0) continue
        terms[d2].push([d1, s1])
        strides[d1] = add(strides[d1], mul(s1, vm2.strides[d2]))
      }
    }
    // Merge dimensions in vm2 if required.
    // NB: Merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required.
    const idxs: UOp[] = [...vm1.shape.map((s, i) => UOp.variable(`idx${i}`, 0, s - 1))]
    let merged_size: sint = 1, merged_term = UOp.int(0)
    const extents: [sint, UOp][] = []
    for (const [term, s, o] of zip(terms.toReversed(), vm2.shape.toReversed(), origin.toReversed())) {
      merged_term = merged_term.add(mul(add(term.reduce((acc, [d1, s1]) => add(acc, mul(idxs[d1], s1)), 0 as sint), o), merged_size))
      merged_size = mul(merged_size, s)
      if (resolve(lt(merged_term, merged_size), false) && resolve(le(0, merged_term), false)) {
        extents.push([merged_size, merged_term])
        merged_size = 1, merged_term = UOp.int(0)
      }
    }
    if (resolve(merged_term.ne(0))) return undefined
    const vm2_shape = extents.toReversed().map(([s, _]) => s)
    if (!is_eq(vm2_shape, vm2.shape)) {
      const reshaped_vm2 = vm2.reshape(vm2_shape)
      if (reshaped_vm2 === undefined) return undefined
      // NOTE: this != to prevent infinite loop
      if (!is_eq(reshaped_vm2.shape, vm2.shape)) return reshaped_vm2.add(vm1)
    }
    if (vm2.mask?.length) {
      // Try to project vm2's mask on to vm1.
      let newb = range(vm1.shape.length).map((x) => 0), newe = [...vm1.shape], bad = false
      for (const [[b, e], o, term, [_, t]] of zip(vm2.mask as [number, number][], origin as number[], terms, extents.toReversed())) {
        if (resolve(and(le(b, t.vmin), lt(t.vmax, e)), false)) continue
        if (term.length !== 1) {
          if (!term && newe.length) newe[0] = 0
          else bad = true
          continue
        }
        const [d1, s1] = term[0]
        newb[d1] = Math.max(newb[d1], ceildiv((s1 as number) > 0 ? b - o : e - o - 1, s1) as number)
        newe[d1] = Math.min(newe[d1], idiv((s1 as number) < 0 ? b - o : e - o - 1, s1 as number) + 1)
      }
      // If any of vm1 was masked off, try again with that mask in place.
      if (zip(newb, newe, vm1.shape).some(([b, e, s]) => !is_eq([b, e], [0, s]))) {
        return vm2.add(View.create(vm1.shape, vm1.strides, vm1.offset, zip(newb, newe)))
      }
      // Otherwise if vm2's mask was violated, then cannot merge.
      if (bad) return undefined
    }
    return View.create(vm1.shape, strides, add(zip(origin, vm2.strides).reduce((acc, [o, s]) => add(acc, mul(o, s)), 0 as sint), vm2.offset))
  })
  invert = cache((out_shape: sint[]): View | undefined => {
    let ret = View.create(this.shape)
    if (this.mask?.length) ret = ret.shrink(this.mask)
    ret = ret.stride(this.strides.map((x) => lt(x, 0) ? -1 : 1)).permute(argsort(this.strides.map((x) => gt(x, 0) ? -x : x)))
    return is_eq(prod(ret.shape), prod(out_shape)) ? ret : undefined // don't support shrink, expand, or stride !== (-1, 1)
  })
  minify = cache((): View => {
    return this.reshape(merge_dims(this.shape, this.strides, this.mask).map((x) => x[0])) || this
  })
  __unsafe_resize(arg: [sint, sint][], mask?: [sint, sint][]): View {
    const offset = zip(this.strides, arg).reduce((acc, [s, x]) => add(acc, mul(s, x[0])), 0 as sint)
    if (this.mask?.length) {
      // move the old mask
      const nmask = zip(this.mask, arg).map(([[mx, my], [ax, ay]]) => [smax(0, smin(sub(mx, ax), sub(ay, ax))), smax(0, smin(sub(my, ax), sub(ay, ax)))] as [sint, sint])
      // merge the masks if we have two
      mask = mask !== undefined ? zip(nmask, mask).map(([[mx1, my1], [mx2, my2]]) => [smax(mx1, mx2), smin(my1, my2)] as [sint, sint]) : nmask
    }
    return View.create(arg.map(([x, y]) => sub(y, x)), this.strides, add(this.offset, offset), mask)
  }
  pad = cache((arg: [sint, sint][]): View => {
    if (arg.length !== this.shape.length) throw new Error(`invalid pad ${list_str(arg)} for ${list_str(this.shape)}`)
    // NOTE: not checking for symbolic arg
    for (const [b, e] of arg) if (!((typeof b !== 'number' || typeof e !== 'number') || (b >= 0 && e >= 0))) throw new Error(`invalid pad ${list_str(arg)} for ${list_str(this.shape)}`)
    if (arg.some(([b, e]) => resolve(ne(b, 0)) || resolve(ne(e, 0)))) {
      const zvarg = zip(this.shape, arg).map(([s, [b, e]]) => [neg(b), add(s, e)] as [sint, sint])
      const mask = zip(this.shape, arg).map(([s, [b, _]]) => [b, add(s, b)] as [sint, sint])
      return this.__unsafe_resize(zvarg, mask)
    }
    return this
  })
  shrink = cache((arg: [sint, sint][]): View => {
    if (arg.length !== this.shape.length) throw new Error(`invalid shrink ${list_str(arg)} for ${list_str(this.shape)}`)
    // NOTE: not checking for symbolic arg
    for (const [s, [b, e]] of zip(this.shape, arg)) if ((isInt(b) && isInt(e) && isInt(s)) && !(0 <= b && b <= e && e <= s)) throw new Error(`invalid shrink ${list_str(arg)} for ${list_str(this.shape)}`)
    return this.__unsafe_resize(arg)
  })
  expand = cache((new_shape: sint[]): View => {
    if (new_shape.length !== this.shape.length) throw new Error(`expand arg new_shape=${list_str(new_shape)} must have same number of dimensions as shape self.shape=${list_str(this.shape)}`)
    if (this.shape.includes(0)) {
      if (!zip(this.shape, new_shape).every(([s, x]) => (s === x && x === 0) || (gt(s, 0) && mod(x, s) === 0))) throw new Error(`can't expand ${list_str(this.shape)} into ${list_str(new_shape)}`)
      return View.create(new_shape)
    }
    // TODO: this resolve might be wrong
    if (!zip(this.shape, new_shape).every(([s, x]) => !resolve(ne(s, x), false) || s === 1)) throw new Error(`can't expand ${list_str(this.shape)} into ${list_str(new_shape)}`)
    // TODO: this resolve may not be needed, but it's hard because vars need to be sorted
    const mask = this.mask?.length ? zip(this.mask, this.shape, new_shape).map(([m, s, ns]) => resolve(ne(s, ns), false) ? (!is_eq(m, [0, 1]) ? [0, 0] : [0, ns]) as [sint, sint] : m) : undefined
    return View.create(new_shape, this.strides, this.offset, mask)
  })
  permute = cache((axis: number[]): View => {
    if (!is_eq(sorted(axis), range(this.shape.length))) throw new Error(`invalid permutation ${list_str(sorted(axis))} of len ${this.shape.length}`)
    return View.create(axis.map((a) => this.shape[a]), axis.map((a) => this.strides[a]), this.offset, this.mask !== undefined ? axis.map((a) => this.mask![a]) : undefined)
  })
  stride = cache((multi: number[]): View => {
    // except for the negative case, you can build this from the others. invertible in the negative case
    if (!multi.every((x) => typeof x === 'number' && x !== 0)) throw new Error(`invalid stride ${multi} for ${this.shape}`)
    const strides = zip(this.strides, multi).map(([z, m]) => mul(z, m))
    const new_shape = zip(this.shape, multi).map(([s, m]) => ceildiv(s, Math.abs(m)))
    const offset = zip(this.shape, this.strides, multi).filter(([s, z, m]) => m < 0).reduce((acc, [s, z, m]) => add(acc, mul(sub(s, 1), z)), 0 as sint)
    const mask = this.mask !== undefined ? zip(this.mask, this.shape, multi).map(([[mx, my], s, m]) => [ceildiv(m > 0 ? mx : sub(s, my), Math.abs(m)), ceildiv(m > 0 ? my : sub(s, mx), Math.abs(m))] as [sint, sint]) : undefined
    return View.create(new_shape, strides, add(this.offset, offset), mask)
  })
  reshape = cache((new_shape: sint[]): View | undefined => {
    if (is_eq(this.shape, new_shape)) return this
    if (new_shape.some((x) => x as number < 0)) throw new Error(`shape can't contain negative numbers ${list_str(new_shape)}`)
    // check for the same size
    const self_all_int = all_int(this.shape)
    if (self_all_int) {
      if (new_shape.some((s) => !(s instanceof UOp) && typeof s !== 'number')) throw new Error(`${list_str(this.shape)} -> ${list_str(new_shape)} contains non (int, Variable) dim`)
      if (resolve(ne(prod(this.shape), prod(new_shape)), false)) throw new Error(`size mismatched, can't reshape self.shape=${list_str(this.shape)} -> new_shape=${list_str(new_shape)}`)
    }

    if (this.shape.includes(0)) return View.create(new_shape)
    if (new_shape.length === 0 && this.mask?.length && this.mask.some(([mx, my]) => mx === my)) return undefined

    // after the asserts, it's okay to check contiguous
    if (this.contiguous) return View.create(new_shape)

    // if it's not contiguous and new shape is symbolic, check if it's directly replaceable
    if (self_all_int && !all_int(new_shape)) {
      if (this.shape.length !== new_shape.length) throw new Error(`cannot symbolic reshape non-contiguous ${this} -> ${new_shape}`)
      for (let [si, so] of zip(this.shape, new_shape)) {
        if (typeof so !== 'number') so = sym_infer(so, new Map(so.vars().map((v) => v.unbind())))
        if (si !== so) throw new Error(`cannot symbolic reshape non-contiguous ${this} -> ${new_shape}`)
        // all dimensions matched, return the new view directly
      }
      return new View(new_shape, this.strides, this.offset, this.mask, this.contiguous)
    }

    const r_strides: sint[] = [], r_new_shape = new_shape.toReversed()
    for (let [merged_size, new_stride, real_size] of merge_dims(this.shape, this.strides, this.mask).toReversed()) {
      // TODO: write with get_contraction
      let acc = 1 as sint
      // TODO: third resolve shouldn't be needed
      let new_dim: sint
      while (resolve(le(acc, merged_size)) && resolve(ne(acc, merged_size)) && resolve(gt(new_dim = r_new_shape.shift() || 0, 0))) {
        r_strides.push(mul(new_stride, acc))
        acc = mul(acc, new_dim)
        if (!resolve(lt(acc, real_size))) new_stride = 0
      }
      if (resolve(ne(acc, merged_size))) return undefined
    }
    const new_strides = [...range(new_shape.length - r_strides.length).map((_) => 0), ...r_strides.toReversed()]
    const new_mask = _reshape_mask(this.mask, this.shape, new_shape)
    if (new_mask !== undefined) {
      const extra_offset = sub(
        this.mask?.length ? sum(zip(this.mask, this.strides).map(([m, s]) => mul(m[0], s))) : 0,
        sum(zip(new_mask, new_strides).map(([m, s]) => mul(m[0], s))),
      )
      return View.create(new_shape, new_strides, add(this.offset, extra_offset), new_mask)
    }
    return undefined
  })
}
