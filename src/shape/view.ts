// from __future__ import annotations
// import functools, operator, itertools, math
// from dataclasses import dataclass
// from typing import Tuple, List, Optional, Dict, Set, cast
// from tinygrad.dtype import dtypes
// from tinygrad.ops import resolve, UOp, Variable, sint, sym_infer, smax, smin, sint_to_uop
// from tinygrad.helpers import prod, all_int, argsort, flatten, ceildiv

import { assert, isNone, isNotNone, range } from '../helpers.ts'
import { UOp } from '../ops.ts'

export const canonicalize_strides = (shape: UOp[], strides: UOp[]): UOp[] => {
  return shape.map((s, i) => ({ s, st: strides[i] })).map(({ s, st }) => s.eq(s.constLike(1)).arg ? s.constLike(0) : st)
}

export const strides_for_shape = (shape: UOp[]): UOp[] => {
  if (!shape?.length) return []
  const strides = shape.slice(1).reverse().reduce((acc, curr) => [...acc, acc.at(-1)!.mul(curr)], [UOp.int(1)])
  return canonicalize_strides(shape, [...strides].reverse())
}

// @functools.lru_cache(maxsize=None)
export const _merge_dims = (shape: number[], strides: number[], mask?: [number, number][]): [number, number, number][] => {
  // merge contiguous sub-parts or zero strided dims. ret = Tuple[(merged_size, stride, merged size w/o zero stride), ...]
  if (!shape.length) return []
  assert(shape.length === strides.length && (isNone(mask) || shape.length === mask?.length))
  const ret: [number, number, number][] = [[shape[0], strides[0], strides[0] !== 0 ? shape[0] : 0]]
  // merge this dim to next dim if size is 1
  let merging = isNotNone(mask) ? (mask[0][1] - mask[0][0] === 1) : shape[0] === 1
  for (let i = 1; i < shape.length; i++) {
    const s = shape[i], st = strides[i]
    const [last_s, last_st, last_pre_expand_s] = ret.at(-1)!
    // always merge 1
    if (s === 1) continue
    // merge last dim with this dim if merging or strides matched
    if (merging || last_st === s * st) ret[ret.length - 1] = [last_s * s, st, st != 0 ? (merging ? s : last_pre_expand_s * s) : 0]
    else ret.push([s, st, st != 0 ? s : 0])
    // merge this dim to next dim if size is 1
    merging = isNotNone(mask) ? (mask[i][1] - mask[i][0] === 1) : s === 1
  }
  return ret
}
const iterator = <T>(items: T[], def: T) => {
  const it = items[Symbol.iterator]()
  return { ...it, next: () => it.next().value || def }
}
/**Returns the new mask if reshape is possible, and None if not possible.*/
export const _reshape_mask = (_mask: undefined | [UOp, UOp][], old_shape: UOp[], new_shape: UOp[]): [UOp, UOp][] | undefined => {
  if (isNone(_mask)) return new_shape.map((s) => [UOp.int(0), s])
  //   if any(not isinstance(m[0], int) or not isinstance(m[1], int) for m in _mask): return None
  if (_mask.some((m) => m[1].sub(m[0]).lt(UOp.int(1)).arg)) return range(new_shape.length).map((x) => [UOp.int(0), UOp.int(0)]) //zero mask

  const new_mask: [UOp, UOp][] = []
  // _mask is all int here
  const [r_masks, r_shape, r_new_shape] = [iterator(_mask.toReversed(), [UOp.int(0), UOp.int(1)]), iterator(old_shape.toReversed(), UOp.int(1)), iterator(new_shape.toReversed(), UOp.int(1))]
  let [curr_stride, old_dim, new_dim, mask] = [UOp.int(1), r_shape.next(), r_new_shape.next(), r_masks.next()]

  while (new_mask.length < new_shape.length) {
    const [[l, r], next_stride] = [mask, new_dim.mul(curr_stride)]
    console.log(new_dim.toString())
    if (old_dim.ge(next_stride)) { // need to split mask.
      if (old_dim === next_stride) { // simply copy the mask and get next batch for merging
        new_mask.push([l.idiv(curr_stride), (r.sub(1)).idiv(curr_stride).add(1)])
        ;[curr_stride, old_dim, new_dim, mask] = [UOp.int(1), r_shape.next(), r_new_shape.next(), r_masks.next()]
      } else { // mask can only be splitted if reshape doesn't cut across the mask.
        if (((l.mod(next_stride).ne(0) || r.mod(next_stride).ne(0)) && l.idiv(next_stride).ne(r.sub(1)).idiv(next_stride)) || old_dim.mod(next_stride).ne(0)) return undefined
        new_mask.push([l.mod(next_stride).idiv(curr_stride), (r.sub(1)).mod(next_stride).idiv(curr_stride.add(1))])
        ;[curr_stride, new_dim] = [next_stride, r_new_shape.next()] // need to get mask for next dimension
      }
    } else {
      const next_mask = r_masks.next()
      // combine if the mask can unfold continuously
      if ((mask[0].arg !== 0 || mask[1].arg !== old_dim) && next_mask[1].sub(next_mask[0]).ne(1).arg) return undefined
      ;[mask, old_dim] = [[next_mask[0].mul(old_dim).add(l), (next_mask[1].sub(1)).mul(old_dim).add(r)], old_dim.mul(r_shape.next())]
    }
  }
  for (const mask of r_masks) { // if the old shape has leading 1s, need to make sure their mask is (0,1)
    if (mask[0].arg !== 0 || mask[1].arg !== 1) return range(new_shape.length).map((x) => [UOp.int(0), UOp.int(0)]) // invalid mask
  }

  return new_mask.toReversed()
}
// const un1d=(shape:UOp., offs:sint) -> List[sint]:
//   result = []
//   for stride in strides_for_shape(shape):
//     here = offs // stride if stride != 0 else 0
//     result.append(here)
//     offs -= here * stride
//   return result

// @dataclass(frozen=True)

export class View {
  //   shape:Tuple[sint, ...]
  //   strides:Tuple[sint, ...]
  //   offset:sint
  //   mask:Optional[Tuple[Tuple[sint, sint], ...]]
  //   contiguous:bool

  //   @functools.cached_property
  //   def t(self):
  //     return tuple(x.tuplize if isinstance(x, UOp) else (x,) \
  //                  for x in self.shape+self.strides+(self.offset,)+(tuple(flatten(self.mask)) if self.mask is not None else tuple()))
  //   def __lt__(self, o:View): return self.t < o.t

  //   def to_indexed_uops(self:View, _idxs:Optional[List[UOp]]=None, vexpr:UOp=UOp.const(dtypes.bool, True)) -> Tuple[UOp, UOp]:
  //     idxs = [UOp.range(dtypes.int, 0, s, i) for i,s in enumerate(self.shape)] if _idxs is None else _idxs
  //     iexpr = sint_to_uop(self.offset)
  //     for idx,sh,st,m in zip(idxs, self.shape, self.strides, self.mask if self.mask is not None else [None]*len(self.shape)):
  //       if resolve(sh != 1) and resolve(st != 0): iexpr = iexpr + idx*st
  //       if m is not None:
  //         if resolve(m[0] != 0): vexpr = vexpr * idx.ge(m[0])
  //         if resolve(m[1] != sh): vexpr = vexpr * idx.lt(m[1])
  //     return iexpr, vexpr

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def size(self) -> int:
  //     ret = prod([x.vmax if isinstance(x, UOp) else x for x in self.shape])
  //     assert isinstance(ret, int), f"{ret=} is not int"
  //     return ret

  //   @staticmethod
  //   @functools.lru_cache(maxsize=None)
  //   def create(shape:Tuple[sint, ...], strides:Optional[Tuple[sint, ...]]=None, offset:sint=0, mask:Optional[Tuple[Tuple[sint, sint], ...]]=None):
  //     # TODO: this resolve shouldn't be needed
  //     if not all(resolve(s >= 0) for s in shape): raise ValueError(f"Trying to create View with negative dimension: {shape=}")
  //     strides = canonicalize_strides(shape, strides) if strides else strides_for_shape(shape)
  //     # canonicalize 0 in shape
  //     if 0 in shape: return View(shape, (0,) * len(shape), offset=0, mask=None, contiguous=True)
  //     # canonicalize empty mask
  //     if mask is not None and all(m == (0,s) for m,s in zip(mask, shape)): mask = None
  //     # if any dimension has size >1, but is masked such that only one index in the dimension is unmasked
  //     # then its stride can also be set to 0, albeit with a corresponding adjustment required to the offset
  //     if mask and any(elim := [not resolve(b+1 < e) for b,e in mask]):
  //       if any(not resolve(b < e) for b,e in mask):
  //         strides, offset, mask = (0,) * len(shape), 0, ((0,0),) * len(shape)
  //       offset += sum((strides[i] * mask[i][0]) if e else 0 for i, e in enumerate(elim))
  //       strides = tuple(0 if e else st for st,e in zip(strides, elim))
  //     # simplify as we go
  //     if isinstance(offset, UOp): offset = cast(sint, offset.ssimplify())
  //     shape = tuple(cast(sint, x.ssimplify()) if isinstance(x, UOp) else x for x in shape)
  //     # TODO: enabling stride simplification breaks it
  //     """
  //     strides = tuple(x.ssimplify() if isinstance(x, UOp) else x for x in strides)
  //     if mask: mask = tuple((s.ssimplify() if isinstance(s, UOp) else s, e.ssimplify() if isinstance(e, UOp) else e) for s,e in mask)
  //     """
  //     contiguous = offset == 0 and mask is None and strides == strides_for_shape(shape)
  //     return View(shape, strides, offset, mask, contiguous)

  //   @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  //   def vars(self) -> Set[Variable]:
  //     flatten_mask = tuple(x for m in self.mask for x in m) if self.mask is not None else tuple()
  //     return functools.reduce(operator.or_, [x.vars() for x in self.shape+self.strides+(self.offset,)+flatten_mask if isinstance(x, UOp)], set())

  //   @functools.lru_cache(None)  # pylint: disable=method-cache-max-size-none
  //   def unbind(self) -> Tuple[View, Dict[Variable, int]]:
  //     var_unboundvar_val = [(v, v.unbind()) for v in self.vars()]
  //     unbound_vars = {v:uv for v,(uv,_) in var_unboundvar_val}
  //     def substitute(x): return x if isinstance(x, int) else x.substitute(unbound_vars)
  //     new_shape = tuple(map(substitute, self.shape))
  //     new_strides = tuple(map(substitute, self.strides))
  //     new_offset = substitute(self.offset)
  //     new_mask = tuple((substitute(x[0]), substitute(x[1])) for x in self.mask) if self.mask is not None else None
  //     return View.create(new_shape, new_strides, new_offset, new_mask), dict(x[1] for x in var_unboundvar_val)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def __add__(self, vm1:View) -> Optional[View]:
  //     vm2 = self
  //     if vm2.contiguous: return vm1
  //     if vm1.contiguous and vm1.shape == vm2.shape: return vm2
  //     if vm1.contiguous and vm1.size() == vm2.size() and (ret := vm2.reshape(vm1.shape)) is not None: return ret
  //     if vm1.mask:
  //       for b,e in vm1.mask:
  //         if resolve(b >= e, False): return View.create(vm1.shape, (0,) * len(vm1.shape), 0, ((0,0),) * len(vm1.shape))
  //       return (merged := vm2 + vm1.shrink(vm1.mask)) and merged.pad(tuple((b,s-e) for (b,e),s in zip(vm1.mask, vm1.shape)))

  //     # Project vm1's offset and strides on to vm2.
  //     origin = un1d(vm2.shape, vm1.offset)
  //     terms: List[List[Tuple[int, sint]]] = [[] for _ in origin]
  //     strides: List[sint] = [0] * len(vm1.shape)
  //     for d1, st in enumerate(vm1.strides):
  //       if st == 0: continue
  //       for d2, (o, s1) in enumerate(zip(origin, un1d(vm2.shape, vm1.offset + st))):
  //         if (s1 := s1 - o) == 0: continue
  //         terms[d2].append((d1, s1))
  //         strides[d1] += s1 * vm2.strides[d2]

  //     # Merge dimensions in vm2 if required.
  //     # NB: Merging too many dimensions can make it difficult to project vm2's mask, hence only combining when required.
  //     if not all_int(vm1.shape): return None
  //     idxs: List[UOp] = [UOp.variable(f"idx{i}", 0, s-1) for i,s in enumerate(vm1.shape)]
  //     merged_size, merged_term = 1, UOp.const(dtypes.int, 0)
  //     extents: List[Tuple[sint, UOp]] = []
  //     for term, s, o in zip(reversed(terms), reversed(vm2.shape), reversed(origin)):
  //       merged_term += (sum([idxs[d1] * s1 for d1, s1 in term]) + o) * merged_size
  //       merged_size *= s
  //       if resolve(merged_term < merged_size, False) and resolve(0 <= merged_term, False):
  //         extents.append((merged_size, merged_term))
  //         merged_size, merged_term = 1, UOp.const(dtypes.int, 0)
  //     if resolve(merged_term != 0): return None
  //     if (vm2_shape := tuple(s for s,_ in reversed(extents))) != vm2.shape:
  //       reshaped_vm2 = vm2.reshape(vm2_shape)
  //       if reshaped_vm2 is None: return None
  //       if reshaped_vm2.shape != vm2.shape: return reshaped_vm2 + vm1

  //     if vm2.mask:
  //       # Try to project vm2's mask on to vm1.
  //       newb, newe, bad = [0] * len(vm1.shape), list(vm1.shape), False
  //       for (b, e), o, term, (_, t) in zip(vm2.mask, origin, terms, reversed(extents)):
  //         if resolve(b <= t.vmin and t.vmax < e, False): continue
  //         if not all_int([o, b, e]):
  //           bad = True
  //           continue
  //         if len(term) != 1:
  //           if not term and newe: newe[0] = 0
  //           else: bad = True
  //           continue
  //         d1, s1 = term[0]
  //         if not isinstance(s1, int) or not isinstance(newe[d1], int):
  //           bad = True
  //           continue
  //         newb[d1] = max(newb[d1], math.ceil((b - o if s1 > 0 else e - o - 1) / s1))
  //         newe[d1] = min(newe[d1], (b - o if s1 < 0 else e - o - 1) // s1 + 1)

  //       # If any of vm1 was masked off, try again with that mask in place.
  //       for b, e, s in zip(newb, newe, vm1.shape):
  //         if (b, e) != (0, s):
  //           return vm2 + View.create(vm1.shape, vm1.strides, vm1.offset, tuple(zip(newb, newe)))
  //       # Otherwise if vm2's mask was violated, then cannot merge.
  //       if bad: return None

  //     return View.create(vm1.shape, tuple(strides), sum(o * s for o, s in zip(origin, vm2.strides)) + vm2.offset)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def invert(self, out_shape:Tuple[sint, ...]) -> Optional[View]:
  //     ret = View.create(self.shape)
  //     if self.mask: ret = ret.shrink(self.mask)
  //     ret = ret.stride(tuple(-1 if x < 0 else 1 for x in self.strides)).permute(argsort(tuple(-x if x > 0 else x for x in self.strides)))
  //     return ret if prod(ret.shape) == prod(out_shape) else None   # don't support shrink, expand, or stride != (-1, 1)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def minify(self):
  //     min_shape = tuple(x[0] for x in _merge_dims(self.shape, self.strides, self.mask))
  //     return nv if (nv := self.reshape(min_shape)) else self

  //   def __unsafe_resize(self, arg: Tuple[Tuple[sint, sint], ...], mask=None) -> View:
  //     offset = sum([s * x[0] for s, x in zip(self.strides,arg)])
  //     if self.mask:
  //       # move the old mask
  //       nmask = tuple([(smax(0, smin(mx-ax,ay-ax)), smax(0, smin(my-ax,ay-ax))) for (mx,my),(ax,ay) in zip(self.mask, arg)])
  //       # merge the masks if we have two
  //       mask = tuple([(smax(mx1, mx2), smin(my1, my2)) for (mx1, my1), (mx2, my2) in zip(nmask, mask)]) if mask is not None else nmask
  //     shape = [y-x for x,y in arg]
  //     if mask is not None and all(m[0] == 0 and m[1] == s for m,s in zip(mask, shape)): mask = None
  //     return View.create(tuple(s.ssimplify() if isinstance(s, UOp) else s for s in shape), self.strides, self.offset+offset, mask)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def pad(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
  //     assert len(arg) == len(self.shape), f"invalid pad {arg} for {self.shape}"
  //     # NOTE: not checking for symbolic arg
  //     for b,e in arg: assert not all_int([b,e]) or b>=0 and e>=0, f"invalid pad {arg} for {self.shape}"
  //     if any(resolve(b!=0) or resolve(e!=0) for b, e in arg):
  //       zvarg = tuple([(-b,s+e) for s,(b,e) in zip(self.shape, arg)])
  //       mask = tuple([(b,s+b) for s,(b,_) in zip(self.shape, arg)])
  //       return self.__unsafe_resize(zvarg, mask=mask)
  //     return self

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def shrink(self, arg: Tuple[Tuple[sint, sint], ...]) -> View:
  //     assert len(arg) == len(self.shape), f"invalid shrink {arg} for {self.shape}"
  //     # NOTE: not checking for symbolic arg
  //     for s,(b,e) in zip(self.shape,arg): assert not all_int([s,b,e]) or (0<=b<=e<=s), f"invalid shrink {arg} for {self.shape}"
  //     return self.__unsafe_resize(arg)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def expand(self, new_shape: Tuple[sint, ...]) -> View:
  //     if len(new_shape) != len(self.shape): raise ValueError(f"expand arg {new_shape=} must have same number of dimensions as shape {self.shape=}")
  //     if 0 in self.shape:
  //       assert all((s == x == 0) or (s > 0 and (x % s) == 0) for s,x in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
  //       return View.create(new_shape)
  //     # TODO: this resolve might be wrong
  //     assert all((not resolve(s != x, False) or s == 1) for s,x in zip(self.shape, new_shape)), f"can't expand {self.shape} into {new_shape}"
  //     # NOTE: can the mask ever be (0,0)?
  //     # TODO: this resolve may not be needed, but it's hard because vars need to be sorted
  //     mask = tuple([(((0,0) if m != (0,1) else (0,ns)) if resolve(s != ns, False) else m) \
  //                   for m,s,ns in zip(self.mask, self.shape, new_shape)]) if self.mask else None
  //     return View.create(new_shape, self.strides, self.offset, mask)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def permute(self, axis: Tuple[int, ...]) -> View:
  //     assert sorted(axis) == list(range(len(self.shape))), f"invalid permutation {axis} of len {len(self.shape)}"
  //     return View.create(tuple(self.shape[a] for a in axis), tuple(self.strides[a] for a in axis), self.offset,
  //                        tuple(self.mask[a] for a in axis) if self.mask is not None else None)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def stride(self, mul: Tuple[int, ...]) -> View:
  //     # except for the negative case, you can build this from the others. invertible in the negative case
  //     assert all(isinstance(x, int) and x != 0 for x in mul), f"invalid stride {mul} for {self.shape}"
  //     strides = tuple([z*m for z,m in zip(self.strides, mul)])
  //     new_shape = tuple([ceildiv(s, abs(m)) for s,m in zip(self.shape, mul)])
  //     offset = sum([(s-1)*z for s,z,m in zip(self.shape, self.strides, mul) if m < 0])
  //     mask = tuple([(ceildiv(mx if m > 0 else s-my, abs(m)), ceildiv(my if m > 0 else s-mx, abs(m))) \
  //                   for (mx,my),s,m in zip(self.mask, self.shape, mul)]) if self.mask is not None else None
  //     return View.create(new_shape, strides, self.offset + offset, mask)

  //   @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  //   def reshape(self, new_shape: Tuple[sint, ...]) -> Optional[View]:
  //     if self.shape == new_shape: return self

  //     # TODO: this resolve shouldn't be needed
  //     assert all(resolve(x >= 0) for x in new_shape), f"shape can't contain negative numbers {new_shape}"
  //     if 0 in self.shape:
  //       assert 0 in new_shape, f"cannot reshape 0 size to {new_shape}"
  //       return View.create(new_shape)
  //     # check for the same size
  //     if (self_all_int := all_int(self.shape)):
  //       assert all(isinstance(s, (int, UOp)) for s in new_shape), f"{self.shape=} -> {new_shape=} contains non (int, Variable) dim"
  //       if resolve(prod(self.shape) != prod(new_shape), False): raise ValueError(f"size mismatched, can't reshape {self.shape=} -> {new_shape=}")

  //     if new_shape == () and self.mask and any(mx==my for (mx,my) in self.mask): return None

  //     # after the asserts, it's okay to check contiguous
  //     if self.contiguous: return View.create(new_shape)

  //     # if it's not contiguous and new shape is symbolic, check if it's directly replaceable
  //     if self_all_int and not all_int(new_shape):
  //       if len(self.shape) != len(new_shape): raise ValueError(f"cannot symbolic reshape non-contiguous {self} -> {new_shape}")
  //       for si, so in zip(self.shape, new_shape):
  //         if not isinstance(so, int): so = sym_infer(so, dict([v.unbind() for v in so.vars()]))
  //         if si != so: raise ValueError(f"cannot symbolic reshape non-contiguous {self} -> {new_shape}")
  //       # all dimensions matched, return the new view directly
  //       return View(new_shape, self.strides, self.offset, self.mask, self.contiguous)

  //     strides, r_new_shape = [], reversed(new_shape)
  //     for merged_dim, new_stride, real_dim in reversed(_merge_dims(self.shape, self.strides, self.mask)):
  //       acc = 1
  //       # TODO: third resolve shouldn't be needed
  //       while resolve(acc <= merged_dim) and resolve(acc != merged_dim) and resolve((new_dim := next(r_new_shape, 0)) > 0):
  //         strides.append(new_stride)
  //         if resolve(new_dim != 1): new_stride *= (new_dim if resolve((acc := acc * new_dim) < real_dim) else 0)
  //       if resolve(acc != merged_dim): break
  //     else:
  //       strides += [0,] * (len(new_shape) - len(strides))
  //       new_mask = _reshape_mask(self.mask, self.shape, new_shape)
  //       if new_mask is not None:
  //         new_strides = canonicalize_strides(tuple(e-b for b,e in new_mask), tuple(reversed(strides)))
  //         extra_offset = (sum(m[0] * s for m,s in zip(self.mask, self.strides)) if self.mask else 0) - \
  //                        (sum(m[0] * s for m,s in zip(new_mask, new_strides)))
  //         return View.create(new_shape, new_strides, self.offset + extra_offset, new_mask)

  //     return None
}