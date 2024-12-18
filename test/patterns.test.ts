import { expect } from 'expect/expect'
import { _substitute, merge_views, Ops, PatternMatcher, renderer, spec, symbolic_flat, UOp, UPat, view_left } from '../src/ops.ts'
import { asdict, python, removeKeys, tryCatch } from './helpers.ts'
import { base_rewrite, extra_pm } from '../src/renderer/cstyle.ts'
import { entries, zip } from '../src/helpers.ts'
import { dtypes } from '../src/dtype.ts'
import { symbolic_simple } from '../src/ops.ts'
import { symbolic } from '../src/ops.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'
import { make_basic_blocks, pm_block_merge } from '../src/codegen/linearize.ts'
import { pm_lowerer } from '../src/codegen/lowerer.ts'
import { devectorize, expander, float4_folding, get_late_rewrite_patterns, load_store_indexing, migrate_indexing, pm_render, sym } from '../src/codegen/uopgraph.ts'
import { append_bufs, append_load, break_sched, check_preload, do_realize, lazy, multioutput, ops_folding, to_si, view_right } from '../src/engine/schedule.ts'

const ALL_PATTERN_MATCHERS: Record<string, { matcher: PatternMatcher<any, any>; uops?: UOp[] }> = {
  'tiny.ops.spec': {
    matcher: spec,
    uops: [
      new UOp({ op: Ops.DEFINE_GLOBAL }),
      //   new UOp({ op: Ops.DEFINE_GLOBAL, dtype: dtypes.imagef(2, 2) }),//probably correct after serialization fix
      new UOp({ op: Ops.DEFINE_LOCAL }),
      new UOp({ op: Ops.DEFINE_ACC, src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 }), new UOp({ op: Ops.RANGE, src: [UOp.int(0), UOp.int(10)] })] }),
      new UOp({ op: Ops.DEFINE_VAR, arg: [null, 0, 1] }),
      new UOp({ op: Ops.RANGE, src: [UOp.int(0), UOp.int(10)] }),
      new UOp({ op: Ops.SPECIAL }),

      new UOp({ op: Ops.VIEW, dtype: dtypes.void }),
      new UOp({ op: Ops.VIEW, src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 })] }),
      new UOp({ op: Ops.VALID, dtype: dtypes.bool, src: [new UOp({ op: Ops.VIEW })] }),
      new UOp({ op: Ops.CONST, dtype: dtypes.float32, arg: 1.4 }), // fails when 1.0
      new UOp({ op: Ops.CONST, dtype: dtypes.int, arg: 33.0 }),
      new UOp({ op: Ops.CONST, dtype: dtypes.bool, arg: true }),

      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.DEFINE_GLOBAL }), new UOp({ op: Ops.VIEW })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.DEFINE_GLOBAL }), new UOp({ op: Ops.VIEW }), new UOp({ op: Ops.STORE })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.DEFINE_LOCAL }), new UOp({ op: Ops.VIEW })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.DEFINE_LOCAL }), new UOp({ op: Ops.VIEW }), new UOp({ op: Ops.STORE })] }),
      new UOp({ op: Ops.STORE, src: [new UOp({ op: Ops.DEFINE_GLOBAL }), new UOp({ op: Ops.VIEW }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.STORE, src: [new UOp({ op: Ops.DEFINE_LOCAL }), new UOp({ op: Ops.VIEW }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.INDEX, src: [new UOp({ op: Ops.DEFINE_GLOBAL }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.INDEX, src: [new UOp({ op: Ops.DEFINE_LOCAL }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.INDEX })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.INDEX }), new UOp({ op: Ops.IF })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.INDEX }), new UOp({ op: Ops.BARRIER })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.INDEX }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST, dtype: dtypes.bool })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.CAST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST, dtype: dtypes.bool })] }),
      new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [new UOp({ op: Ops.INDEX }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [new UOp({ op: Ops.INDEX }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST, dtype: dtypes.bool })] }),
      new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [new UOp({ op: Ops.INDEX }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.IF })] }),
      new UOp({ op: Ops.STORE, dtype: dtypes.void, src: [new UOp({ op: Ops.CAST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.IF })] }),

      new UOp({ op: Ops.WHERE, dtype: dtypes.float32, src: [new UOp({ op: Ops.CONST, dtype: dtypes.bool }), new UOp({ op: Ops.CONST, dtype: dtypes.float32 }), new UOp({ op: Ops.CONST, dtype: dtypes.float32 })] }),
      new UOp({ op: Ops.CMPLT, dtype: dtypes.bool, src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 }), new UOp({ op: Ops.CONST, dtype: dtypes.float32 })] }),
      new UOp({ op: Ops.CMPNE, dtype: dtypes.bool, src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 }), new UOp({ op: Ops.CONST, dtype: dtypes.float32 })] }),
      new UOp({ op: Ops.SHL, dtype: dtypes.int32, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int32 }), new UOp({ op: Ops.CONST, dtype: dtypes.uint })] }),
      new UOp({ op: Ops.SHR, dtype: dtypes.int32, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int32 }), new UOp({ op: Ops.CONST, dtype: dtypes.int32 })] }),
      new UOp({ op: Ops.IDIV, dtype: dtypes.int32, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int32 }), new UOp({ op: Ops.CONST, dtype: dtypes.int32 })] }),
      new UOp({ op: Ops.ADD, dtype: dtypes.float32, src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 }), new UOp({ op: Ops.CONST, dtype: dtypes.float32 })] }),
      new UOp({ op: Ops.ASSIGN, src: [new UOp({ op: Ops.DEFINE_ACC }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.ENDRANGE, dtype: dtypes.void, src: [new UOp({ op: Ops.RANGE })] }),

      new UOp({ op: Ops.WMMA, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.CONTRACT, dtype: dtypes.float32.vec(4), arg: [[0, 4]], src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 })] }),
      new UOp({ op: Ops.EXPAND, dtype: dtypes.float32, arg: [[0, 4]], src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32.vec(4) })] }),
      new UOp({ op: Ops.IF, dtype: dtypes.void, src: [new UOp({ op: Ops.CONST, dtype: dtypes.bool })] }),
      new UOp({ op: Ops.IF, dtype: dtypes.void, src: [new UOp({ op: Ops.CONST, dtype: dtypes.bool }), new UOp({ op: Ops.BARRIER })] }),
      new UOp({ op: Ops.ENDIF, dtype: dtypes.void, src: [new UOp({ op: Ops.IF })] }),
      new UOp({ op: Ops.REDUCE_AXIS, arg: [Ops.ADD, 4] }),
      new UOp({ op: Ops.GEP, src: [new UOp({ op: Ops.DEFINE_GLOBAL })], dtype: dtypes.float32 }),
      new UOp({ op: Ops.VECTORIZE, dtype: dtypes.float32.vec(4), src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.BITCAST, src: [new UOp({ op: Ops.CONST })], dtype: dtypes.float32 }),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.CONST })], dtype: dtypes.float32 }),
      new UOp({ op: Ops.BARRIER, dtype: dtypes.void, src: [new UOp({ op: Ops.STORE, src: [new UOp({ op: Ops.DEFINE_LOCAL }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] })] }),
      new UOp({ op: Ops.SINK, dtype: dtypes.void }),
      new UOp({ op: Ops.NOOP }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int64 })] }),
      new UOp({ op: Ops.STORE, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int64 }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.BARRIER, dtype: dtypes.void, src: [new UOp({ op: Ops.STORE, src: [new UOp({ op: Ops.CONST, dtype: dtypes.int64 }), new UOp({ op: Ops.CONST })] })] }),
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
      new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.CONST, arg: 4 }), new UOp({ op: Ops.CONST, arg: 66 })] }),
      UOp.variable('x', false, true, dtypes.bool).mul(UOp.variable('y', false, true, dtypes.bool)),
      UOp.variable('x', false, true, dtypes.bool).add(UOp.variable('y', false, true, dtypes.bool)),
      UOp.variable('x', false, true, dtypes.bool).maximum(UOp.variable('y', false, true, dtypes.bool)),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.CAST, dtype: dtypes.float, arg: 44.55 })], dtype: dtypes.int }),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.MUL, dtype: dtypes.float })], dtype: dtypes.float }),
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
      new UOp({ op: Ops.MUL, dtype: dtypes.int, src: [new UOp({ op: Ops.DEFINE_VAR, arg: ['x', 0, 0] }), new UOp({ op: Ops.DEFINE_VAR, arg: ['x', 0, 0] })] }), // ALU min==max -> CONST
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
      new UOp({ op: Ops.ADD }),
      new UOp({ op: Ops.MUL }),
    ],
  },
  'tiny.ops.renderer': {
    matcher: renderer,
    uops: [
      UOp.variable('x', 0, 9999),
      new UOp({ op: Ops.DEFINE_VAR, arg: ['x'] }),
      new UOp({ op: Ops.RANGE, arg: [1] }),
      new UOp({ op: Ops.CONST, arg: 42 }),
      new UOp({ op: Ops.BIND, src: [new UOp({ op: Ops.NOOP, arg: 'x' })] }),
      new UOp({ op: Ops.NEG, src: [new UOp({ op: Ops.NOOP, arg: '5' })] }),
      new UOp({ op: Ops.MAX, src: [new UOp({ op: Ops.NOOP, arg: 'a' }), new UOp({ op: Ops.NOOP, arg: 'b' })] }),
      new UOp({ op: Ops.MULACC, src: [new UOp({ op: Ops.NOOP, arg: 'x' }), new UOp({ op: Ops.NOOP, arg: 'y' }), new UOp({ op: Ops.NOOP, arg: 'z' })] }),
      new UOp({ op: Ops.WHERE, src: [new UOp({ op: Ops.NOOP, arg: 'cond' }), new UOp({ op: Ops.NOOP, arg: 'true_val' }), new UOp({ op: Ops.NOOP, arg: 'false_val' })] }),
      new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.NOOP, arg: 'a' }), new UOp({ op: Ops.NOOP, arg: 'b' })] }),
    ],
  },
  'tiny.ops.merge_views': {
    matcher: merge_views,
    uops: [
      new UOp({ op: Ops.VIEW, src: [UOp.int(20)], arg: new ShapeTracker([]) }),
    ],
  },
  // TODO: not sure if these trigger any patterns at all
  'tiny.ops.view_left': {
    matcher: view_left,
    uops: [
      new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.BITCAST, src: [new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.ASSIGN, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.VIEW })] }),
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
      new UOp({ op: Ops.VECTORIZE, src: [new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.INDEX, src: [UOp.variable('buf')] })] })] }),
      new UOp({ op: Ops.VECTORIZE, src: [new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.INDEX, src: [UOp.variable('buf', false, true, dtypes.bool)] })] })] }),
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

  'tiny.engine.schedule.append_bufs': {
    matcher: append_bufs,
  },
  'tiny.engine.schedule.check_preload': {
    matcher: check_preload,
  },
  'tiny.engine.schedule.to_si': {
    matcher: to_si,
  },
  'tiny.engine.schedule.lazy': {
    matcher: lazy,
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
      await t.step(i.toString(), () => {
        expect(removeKeys(asdict(ts), ['location', 'op'])).toEqual(removeKeys(asdict(py), ['location', 'op']))
      })
    }
  })

  Deno.test(`${name}_pdict`, async (t) => {
    const PYDict = await python<Map<Ops, [UPat, undefined, Ops[], boolean][]>>(`${pyImport}\nout(${pyCode}.pdict)`)
    for (const [key, ts] of matcher.pdict.entries()) {
      const py = PYDict.get(key)!
      for (const [i, [ts1, py1]] of zip(ts as any[], py).entries()) {
        await t.step(i.toString(), () => {
          expect(removeKeys(asdict(ts1[0]), ['location', 'op'])).toEqual(removeKeys(asdict(py1[0]), ['location', 'op'])) //UPat
          expect(asdict(ts1[2]).toSorted()).toEqual(asdict(py1[2]).toSorted()) // Ops[]
          expect(asdict(ts1[3])).toEqual(asdict(py1[3])) // has ctx?
        })
      }
    }
  })

  for (const [i, uop] of uops?.entries() || []) {
    Deno.test(`${name}_${i}`, async () => {
      const ts = tryCatch(() => matcher.rewrite(uop, new Map([[uop, 'somectxvalue']])))()
      const py = await python(`${pyImport}\nout(${pyCode}.rewrite(data,{data:"somectxvalue"}))`, uop)
      expect(asdict(ts)).toEqual(asdict(py))
    })
  }
}
