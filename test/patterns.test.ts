import { expect } from 'expect/expect'
import { _substitute, merge_views, Ops, type PatternMatcher, renderer, spec, symbolic_flat, UOp, type UPat, view_left } from '../denograd/ops.ts'
import { asdict, python, removeKeys, tryCatch } from './helpers.ts'
import { base_rewrite, extra_pm } from '../denograd/renderer/cstyle.ts'
import { entries, zip } from '../denograd/helpers.ts'
import { dtypes } from '../denograd/dtype.ts'
import { symbolic_simple } from '../denograd/ops.ts'
import { symbolic } from '../denograd/ops.ts'
import { ShapeTracker } from '../denograd/shape/shapetracker.ts'
import { make_basic_blocks, pm_block_merge } from '../denograd/codegen/linearize.ts'
import { pm_lowerer } from '../denograd/codegen/lowerer.ts'
import { devectorize, expander, float4_folding, get_late_rewrite_patterns, load_store_indexing, migrate_indexing, pm_render, sym } from '../denograd/codegen/uopgraph.ts'
import { break_sched, do_realize, multioutput, ops_folding, to_si, view_right } from '../denograd/engine/schedule.ts'

const ALL_PATTERN_MATCHERS: Record<string, { matcher: PatternMatcher<any, any>; uops?: UOp[] }> = {
  'tiny.ops.spec': {
    matcher: spec,
    uops: [
      new UOp(Ops.DEFINE_GLOBAL),
      //   new UOp(Ops.DEFINE_GLOBAL, dtypes.imagef(2, 2)),//probably correct after serialization fix
      new UOp(Ops.DEFINE_LOCAL),
      new UOp(Ops.DEFINE_ACC, undefined, [new UOp(Ops.CONST, dtypes.float32), new UOp(Ops.RANGE, undefined, [UOp.int(0), UOp.int(10)])]),
      new UOp(Ops.DEFINE_VAR, undefined, undefined, [null, 0, 1]),
      new UOp(Ops.RANGE, undefined, [UOp.int(0), UOp.int(10)]),
      new UOp(Ops.SPECIAL),

      new UOp(Ops.VIEW, dtypes.void),
      new UOp(Ops.VIEW, undefined, [new UOp(Ops.CONST, dtypes.float32)]),
      new UOp(Ops.VALID, dtypes.bool, [new UOp(Ops.VIEW)]),
      new UOp(Ops.CONST, dtypes.float32, undefined, 1.4), // fails when 1.0
      new UOp(Ops.CONST, dtypes.int, undefined, 33.0),
      new UOp(Ops.CONST, dtypes.bool, undefined, true),

      new UOp(Ops.LOAD, undefined, [new UOp(Ops.DEFINE_GLOBAL), new UOp(Ops.VIEW)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.DEFINE_GLOBAL), new UOp(Ops.VIEW), new UOp(Ops.STORE)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.DEFINE_LOCAL), new UOp(Ops.VIEW)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.DEFINE_LOCAL), new UOp(Ops.VIEW), new UOp(Ops.STORE)]),
      new UOp(Ops.STORE, undefined, [new UOp(Ops.DEFINE_GLOBAL), new UOp(Ops.VIEW), new UOp(Ops.CONST)]),
      new UOp(Ops.STORE, undefined, [new UOp(Ops.DEFINE_LOCAL), new UOp(Ops.VIEW), new UOp(Ops.CONST)]),
      new UOp(Ops.INDEX, undefined, [new UOp(Ops.DEFINE_GLOBAL), new UOp(Ops.CONST)]),
      new UOp(Ops.INDEX, undefined, [new UOp(Ops.DEFINE_LOCAL), new UOp(Ops.CONST)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX), new UOp(Ops.IF)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX), new UOp(Ops.BARRIER)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX), new UOp(Ops.CONST), new UOp(Ops.CONST, dtypes.bool)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.CAST), new UOp(Ops.CONST), new UOp(Ops.CONST, dtypes.bool)]),
      new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX), new UOp(Ops.CONST)]),
      new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX), new UOp(Ops.CONST), new UOp(Ops.CONST, dtypes.bool)]),
      new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.INDEX), new UOp(Ops.CONST), new UOp(Ops.IF)]),
      new UOp(Ops.STORE, dtypes.void, [new UOp(Ops.CAST), new UOp(Ops.CONST), new UOp(Ops.IF)]),

      new UOp(Ops.WHERE, dtypes.float32, [new UOp(Ops.CONST, dtypes.bool), new UOp(Ops.CONST, dtypes.float32), new UOp(Ops.CONST, dtypes.float32)]),
      new UOp(Ops.CMPLT, dtypes.bool, [new UOp(Ops.CONST, dtypes.float32), new UOp(Ops.CONST, dtypes.float32)]),
      new UOp(Ops.CMPNE, dtypes.bool, [new UOp(Ops.CONST, dtypes.float32), new UOp(Ops.CONST, dtypes.float32)]),
      new UOp(Ops.SHL, dtypes.int32, [new UOp(Ops.CONST, dtypes.int32), new UOp(Ops.CONST, dtypes.uint)]),
      new UOp(Ops.SHR, dtypes.int32, [new UOp(Ops.CONST, dtypes.int32), new UOp(Ops.CONST, dtypes.int32)]),
      new UOp(Ops.IDIV, dtypes.int32, [new UOp(Ops.CONST, dtypes.int32), new UOp(Ops.CONST, dtypes.int32)]),
      new UOp(Ops.ADD, dtypes.float32, [new UOp(Ops.CONST, dtypes.float32), new UOp(Ops.CONST, dtypes.float32)]),
      new UOp(Ops.ASSIGN, undefined, [new UOp(Ops.DEFINE_ACC), new UOp(Ops.CONST)]),
      new UOp(Ops.ENDRANGE, dtypes.void, [new UOp(Ops.RANGE)]),

      new UOp(Ops.WMMA, undefined, [new UOp(Ops.CONST), new UOp(Ops.CONST), new UOp(Ops.CONST)]),
      new UOp(Ops.CONTRACT, dtypes.float32.vec(4), [new UOp(Ops.CONST, dtypes.float32)], [[0, 4]]),
      new UOp(Ops.EXPAND, dtypes.float32, [new UOp(Ops.CONST, dtypes.float32.vec(4))], [[0, 4]]),
      new UOp(Ops.IF, dtypes.void, [new UOp(Ops.CONST, dtypes.bool)]),
      new UOp(Ops.IF, dtypes.void, [new UOp(Ops.CONST, dtypes.bool), new UOp(Ops.BARRIER)]),
      new UOp(Ops.ENDIF, dtypes.void, [new UOp(Ops.IF)]),
      new UOp(Ops.REDUCE_AXIS, undefined, undefined, [Ops.ADD, 4]),
      new UOp(Ops.GEP, dtypes.float32, [new UOp(Ops.DEFINE_GLOBAL)]),
      new UOp(Ops.VECTORIZE, dtypes.float32.vec(4), [new UOp(Ops.CONST), new UOp(Ops.CONST), new UOp(Ops.CONST), new UOp(Ops.CONST)]),
      new UOp(Ops.BITCAST, dtypes.float32, [new UOp(Ops.CONST)]),
      new UOp(Ops.CAST, dtypes.float32, [new UOp(Ops.CONST)]),
      new UOp(Ops.BARRIER, dtypes.void, [new UOp(Ops.STORE, undefined, [new UOp(Ops.DEFINE_LOCAL), new UOp(Ops.CONST), new UOp(Ops.CONST)])]),
      new UOp(Ops.SINK, dtypes.void),
      new UOp(Ops.NOOP),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.CONST, dtypes.int64)]),
      new UOp(Ops.STORE, undefined, [new UOp(Ops.CONST, dtypes.int64), new UOp(Ops.CONST)]),
      new UOp(Ops.BARRIER, dtypes.void, [new UOp(Ops.STORE, undefined, [new UOp(Ops.CONST, dtypes.int64), new UOp(Ops.CONST)])]),
    ],
  },

  'tiny.ops.symbolic_simple': {
    matcher: symbolic_simple,
    uops: [
      UOp.variable('x').add(0),
      UOp.variable('x').mul(UOp.int(1)),
      UOp.variable('x').idiv(UOp.variable('x')),
      UOp.variable('x').idiv(1),
      UOp.variable('x').idiv(-1),
      UOp.variable('x').div(UOp.variable('x')),
      UOp.variable('x').mul(UOp.variable('x2')).div(UOp.variable('x2')),
      UOp.variable('base').mod(UOp.variable('y')).mod(UOp.variable('y')),
      UOp.variable('x').mod(UOp.int(1)).add(UOp.variable('x').idiv(UOp.int(1)).mul(UOp.int(1))),
      UOp.variable('x', false, true, dtypes.bool).bitwise_and(UOp.bool(false)),
      UOp.variable('x', false, true, dtypes.bool).bitwise_or(UOp.bool(false)),

      UOp.variable('x').maximum(UOp.variable('x')),
      UOp.variable('x').bitwise_and(UOp.variable('x')),
      UOp.variable('x').bitwise_or(UOp.variable('x')),
      UOp.variable('x', false, true, dtypes.bool).logical_not().logical_not(),
      UOp.variable('x', false, true, dtypes.bool).where(UOp.bool(true), UOp.bool(false)),

      UOp.variable('x').lt(UOp.variable('x')),
      UOp.variable('x').ne(UOp.variable('x')),

      UOp.variable('x').mul(0),
      UOp.variable('x').mul(UOp.int(0)),
      new UOp(Ops.ADD, undefined, [new UOp(Ops.CONST, undefined, undefined, 4), new UOp(Ops.CONST, undefined, undefined, 66)]),
      UOp.variable('x', false, true, dtypes.bool).mul(UOp.variable('y', false, true, dtypes.bool)),
      UOp.variable('x', false, true, dtypes.bool).add(UOp.variable('y', false, true, dtypes.bool)),
      UOp.variable('x', false, true, dtypes.bool).maximum(UOp.variable('y', false, true, dtypes.bool)),
      new UOp(Ops.CAST, dtypes.int, [new UOp(Ops.CAST, dtypes.float, undefined, 44.55)]),
      new UOp(Ops.CAST, dtypes.float, [new UOp(Ops.MUL, dtypes.float)]),
    ],
  },
  'tiny.ops.symbolic': {
    matcher: symbolic,
    uops: [
      UOp.variable('x').add(UOp.variable('y')).add(UOp.variable('x').mul(UOp.int(5))), // group like
      UOp.variable('x').bitwise_or(UOp.variable('x').bitwise_and(UOp.variable('y'))), // boolean algebra
      UOp.variable('x').mul(UOp.int(2)).add(UOp.variable('x').mul(UOp.int(3))), // combine terms
      UOp.variable('x').add(UOp.variable('x').mul(UOp.int(3))), // x + x*c -> x*(c+1)
      UOp.variable('x').add(UOp.variable('x')), // x + x -> x*2
      UOp.variable('x').div(UOp.variable('x2')).div(UOp.variable('x3')), // (x/x2)/x3 -> x/(x2*x3)
      UOp.variable('x').add(UOp.int(5)).mul(-1), // -(x+c) -> -x + -c

      UOp.variable('val').where(UOp.variable('val'), UOp.variable('val')), // same results either way is noop
      UOp.bool(false).where(UOp.variable('c0'), UOp.variable('c1')), // const gate folding
      UOp.bool(true).where(UOp.variable('c0'), UOp.variable('c1')), // const gate folding
      new UOp(Ops.MUL, dtypes.int, [new UOp(Ops.DEFINE_VAR, undefined, undefined, ['x', 0, 0]), new UOp(Ops.DEFINE_VAR, undefined, undefined, ['x', 0, 0])]), // ALU min==max -> CONST
      UOp.variable('x').maximum(UOp.variable('y')), // max folding when x.vmax <= y.vmin
      UOp.variable('x').mul(UOp.int(2)).maximum(UOp.variable('x').mul(UOp.int(3))), // maxVarConst

      UOp.variable('x').add(UOp.int(2)).add(UOp.int(3)), // (x+c1)+c2 -> x+(c1+c2)
      UOp.variable('x').mul(UOp.int(2)).mul(UOp.int(3)), // (x*c1)*c2 -> x*(c1*c2)
      UOp.variable('x').bitwise_and(UOp.int(2)).bitwise_and(UOp.int(3)), // (x&c1)&c2 -> x&(c1&c2)
      UOp.variable('x').bitwise_or(UOp.int(2)).bitwise_or(UOp.int(3)), // (x|c1)|c2 -> x|(c1|c2)
      UOp.int(2).add(UOp.variable('x')).lt(UOp.int(5)), // c0+x<c1 -> x<c1-c0
      UOp.variable('x').idiv(UOp.int(2)).idiv(UOp.int(3)), // (x//c1)//c2 -> x//(c1*c2)
      UOp.int(2).mul(UOp.variable('x')).lt(UOp.int(5)), // 2x < 5 -> x < ceil(5/2)
      UOp.int(-2).mul(UOp.variable('x')).lt(UOp.int(-5)), // -2x < -5 -> -x < -floor(-(-5)/-(-2))
      UOp.variable('x').idiv(UOp.int(2)).lt(UOp.int(3)), // x//2 < 3 -> x < 3*2
      UOp.int(4).mul(UOp.variable('x')).add(UOp.variable('x2')).lt(UOp.int(12)), // (4x + x2) < 12 -> x < 12/4 when 12%4=0 and 4>x2.vmax and x2.vmin>=0
      UOp.variable('x').lt(UOp.int(5)), // x < 5 when 0 < 5
      UOp.variable('x').lt(1).ne(true), // not x < 1 -> X > 0
      UOp.variable('x').idiv(UOp.int(3)), // x//3 when 0 < 3
      UOp.variable('x').mod(UOp.int(4)), // x%4 when 0 < 4
    ],
  },
  'tiny.ops.symbolic_flat': {
    matcher: symbolic_flat,
    uops: [
      UOp.variable('x').add(UOp.variable('y')).mul(-1),
      UOp.variable('x').add(UOp.variable('y')).mul(UOp.int(3)),
    ],
  },
  'tiny.ops._substitute': {
    matcher: _substitute,
    uops: [
      new UOp(Ops.ADD),
      new UOp(Ops.MUL),
    ],
  },
  'tiny.ops.renderer': {
    matcher: renderer,
    uops: [
      UOp.variable('x', 0, 9999),
      new UOp(Ops.DEFINE_VAR, undefined, undefined, ['x']),
      new UOp(Ops.RANGE, undefined, undefined, [1]),
      new UOp(Ops.CONST, undefined, undefined, 42),
      new UOp(Ops.BIND, undefined, [new UOp(Ops.NOOP, undefined, undefined, 'x')]),
      new UOp(Ops.NEG, undefined, [new UOp(Ops.NOOP, undefined, undefined, '5')]),
      new UOp(Ops.MAX, undefined, [new UOp(Ops.NOOP, undefined, undefined, 'a'), new UOp(Ops.NOOP, undefined, undefined, 'b')]),
      new UOp(Ops.MULACC, undefined, [new UOp(Ops.NOOP, undefined, undefined, 'x'), new UOp(Ops.NOOP, undefined, undefined, 'y'), new UOp(Ops.NOOP, undefined, undefined, 'z')]),
      // new UOp(Ops.WHERE, undefined, [new UOp(Ops.NOOP, undefined, undefined, 'cond'), new UOp(Ops.NOOP, undefined, undefined, 'true_val'), new UOp(Ops.NOOP, undefined, undefined, 'false_val')]),
      new UOp(Ops.ADD, undefined, [new UOp(Ops.NOOP, undefined, undefined, 'a'), new UOp(Ops.NOOP, undefined, undefined, 'b')]),
    ],
  },
  'tiny.ops.merge_views': {
    matcher: merge_views,
    uops: [
      new UOp(Ops.VIEW, undefined, [UOp.int(20)], new ShapeTracker([])),
    ],
  },
  // TODO: not sure if these trigger any patterns at all
  'tiny.ops.view_left': {
    matcher: view_left,
    uops: [
      new UOp(Ops.ADD, undefined, [new UOp(Ops.CONST), new UOp(Ops.CONST)]),
      new UOp(Ops.CAST, undefined, [new UOp(Ops.CONST)]),
      new UOp(Ops.BITCAST, undefined, [new UOp(Ops.CONST)]),
      new UOp(Ops.ASSIGN, undefined, [new UOp(Ops.CONST), new UOp(Ops.CONST)]),
      new UOp(Ops.LOAD, undefined, [new UOp(Ops.VIEW)]),
    ],
  },
  'tiny.renderer.cstyle.base_rewrite': {
    matcher: base_rewrite,
  },
  'tiny.renderer.cstyle.extra_pm': {
    matcher: extra_pm,
  },
  'tiny.codegen.linearize.make_basic_blocks': {
    matcher: make_basic_blocks,
  },

  'tiny.codegen.linearize.pm_block_merge': {
    matcher: pm_block_merge,
  },

  'tiny.codegen.lowerer.pm_lowerer': {
    matcher: pm_lowerer,
  },

  'tiny.codegen.uopgraph.float4_folding': {
    matcher: float4_folding,
    uops: [
      new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf')])])]),
      new UOp(Ops.VECTORIZE, undefined, [new UOp(Ops.LOAD, undefined, [new UOp(Ops.INDEX, undefined, [UOp.variable('buf', false, true, dtypes.bool)])])]),
    ],
  },

  'tiny.codegen.uopgraph.get_late_rewrite_patterns((Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.AND, Ops.SHL, Ops.NEG, Ops.MULACC))': {
    matcher: get_late_rewrite_patterns([Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.AND, Ops.SHL, Ops.NEG, Ops.MULACC]),
  },
  'tiny.codegen.uopgraph.sym': {
    matcher: sym,
  },

  'tiny.codegen.uopgraph.expander': {
    matcher: expander,
  },
  'tiny.codegen.uopgraph.devectorize': {
    matcher: devectorize,
  },
  'tiny.codegen.uopgraph.load_store_indexing': {
    matcher: load_store_indexing,
  },
  'tiny.codegen.uopgraph.migrate_indexing': {
    matcher: migrate_indexing,
  },
  'tiny.codegen.uopgraph.pm_render': {
    matcher: pm_render,
  },
  'tiny.engine.schedule.view_right': {
    matcher: view_right,
  },

  'tiny.engine.schedule.to_si': {
    matcher: to_si,
  },
  'tiny.engine.schedule.multioutput': {
    matcher: multioutput,
  },
  'tiny.engine.schedule.ops_folding': {
    matcher: ops_folding,
  },
  'tiny.engine.schedule.do_realize': {
    matcher: do_realize,
  },
  'tiny.engine.schedule.break_sched': {
    matcher: break_sched,
  },
}

for (const [name, { matcher, uops }] of entries(ALL_PATTERN_MATCHERS)) {
  const splits = name.split('.')
  const pyImport = ``
  const pyCode = name
  Deno.test(`${name}_patterns`, async (t) => {
    const TSPatterns = matcher.patterns.map((pattern) => pattern[0])
    const PYPatterns = await python(`${pyImport}\nout([pattern[0] for pattern in ${pyCode}.patterns])`)
    for (const [i, [ts, py]] of zip(TSPatterns, PYPatterns).entries()) {
      await t.step(i.toString(), async () => {
        expect(removeKeys(await asdict(ts), ['location', 'op'])).toEqual(removeKeys(await asdict(py), ['location', 'op']))
      })
    }
  })

  Deno.test(`${name}_pdict`, async (t) => {
    const PYDict = await python<Map<Ops, [UPat, undefined, Ops[], boolean][]>>(`${pyImport}\nout(${pyCode}.pdict)`)
    for (const [key, ts] of matcher.pdict.entries()) {
      const py = PYDict.get(key)!
      for (const [i, [ts1, py1]] of zip(ts as any[], py).entries()) {
        await t.step(i.toString(), async () => {
          expect(removeKeys(await asdict(ts1[0]), ['location', 'op'])).toEqual(removeKeys(await asdict(py1[0]), ['location', 'op'])) //UPat
          expect((await asdict(ts1[2])).toSorted()).toEqual((await asdict(py1[2])).toSorted()) // Ops[]
          expect(await asdict(ts1[3])).toEqual(await asdict(py1[3])) // has ctx?
        })
      }
    }
  })

  for (const [i, uop] of uops?.entries() || []) {
    Deno.test(`${name}_${i}`, async () => {
      const ts = tryCatch(() => matcher.rewrite(uop, new Map([[uop, 'somectxvalue']])))()
      const py = await python(`${pyImport}\nout(${pyCode}.rewrite(data,{data:"somectxvalue"}))`, uop)
      expect(await asdict(ts)).toEqual(await asdict(py))
    })
  }
}
