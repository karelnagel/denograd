import { expect } from 'expect/expect'
import { _substitute, merge_views, Ops, renderer, spec, symbolicFlat, UOp, type UPat, view_left } from '../src/ops.ts'
import { asdict, python, removeKeys, tryCatch } from './helpers.ts'
import { base_rewrite, extra_pm } from '../src/renderer/cstyle.ts'
import { entries, zip } from '../src/helpers.ts'
import { type DType, dtypes } from '../src/dtype.ts'
import { symbolicSimple } from '../src/ops.ts'
import { symbolic } from '../src/ops.ts'
import { ShapeTracker } from '../src/shape/shapetracker.ts'

const ALL_PATTERN_MATCHERS = {
  'tinygrad.ops.spec': {
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

  'tinygrad.ops.symbolic_simple': {
    matcher: symbolicSimple,
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
      UOp.variable('x', false, true, dtypes.bool).bitwiseAnd(UOp.bool(false)),
      UOp.variable('x', false, true, dtypes.bool).bitwiseOr(UOp.bool(false)),

      UOp.variable('x').maximum(UOp.variable('x')),
      UOp.variable('x').bitwiseAnd(UOp.variable('x')),
      UOp.variable('x').bitwiseOr(UOp.variable('x')),
      UOp.variable('x', false, true, dtypes.bool).logicalNot().logicalNot(),
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
  'tinygrad.ops.symbolic': {
    matcher: symbolic,
    uops: [
      UOp.variable('x').add(UOp.variable('y')).add(UOp.variable('x').mul(UOp.int(5))), // group like
      UOp.variable('x').bitwiseOr(UOp.variable('x').bitwiseAnd(UOp.variable('y'))), // boolean algebra
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
      UOp.variable('x').bitwiseAnd(UOp.int(2)).bitwiseAnd(UOp.int(3)), // (x&c1)&c2 -> x&(c1&c2)
      UOp.variable('x').bitwiseOr(UOp.int(2)).bitwiseOr(UOp.int(3)), // (x|c1)|c2 -> x|(c1|c2)
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
  'tinygrad.ops.symbolic_flat': {
    matcher: symbolicFlat,
    uops: [
      UOp.variable('x').add(UOp.variable('y')).mul(-1),
      UOp.variable('x').add(UOp.variable('y')).mul(UOp.int(3)),
    ],
  },
  'tinygrad.ops._substitute': {
    matcher: _substitute,
    uops: [
      new UOp({ op: Ops.ADD }),
      new UOp({ op: Ops.MUL }),
    ],
  },
  'tinygrad.ops.renderer': {
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
  'tinygrad.ops.merge_views': {
    matcher: merge_views,
    uops: [
      new UOp({ op: Ops.VIEW, src: [UOp.int(20)], arg: new ShapeTracker([]) }),
    ],
  },
  // TODO: not sure if these trigger any patterns at all
  'tinygrad.ops.view_left': {
    matcher: view_left,
    uops: [
      new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.BITCAST, src: [new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.ASSIGN, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      new UOp({ op: Ops.LOAD, src: [new UOp({ op: Ops.VIEW })] }),
    ],
  },
  'tinygrad.renderer.cstyle.base_rewrite': {
    matcher: base_rewrite,
    uops: [
      // These should be tested with CStyleLanguage.render() instead, cause it needs self.h
      //   new UOp({ op: Ops.DEFINE_ACC, src: [new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.ASSIGN, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.IF, src: [new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.ENDIF }),
      //   new UOp({ op: Ops.WMMA, arg: ['wmma.load_matrix_sync'], src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),

      //   new UOp({ op: Ops.RANGE, dtype: dtypes.int32, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.VECTORIZE, dtype: dtypes.float32.vec(4), src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.CAST, dtype: dtypes.float32, src: [new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.BITCAST, dtype: dtypes.float32, src: [new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.DEFINE_LOCAL, dtype: dtypes.float32, arg: [0, 128], src: [] }),
      //   new UOp({ op: Ops.BARRIER }),
      //   new UOp({ op: Ops.NOOP, src: [new UOp({ op: Ops.CONST })] }),
      //   new UOp({ op: Ops.SPECIAL, arg: [['workitem', 0], 'comment'], src: [] })
    ],
  },
  'tinygrad.renderer.cstyle.extra_pm': {
    matcher: extra_pm,
    uops: [],
  },
}
for (const [name, { matcher, uops }] of entries(ALL_PATTERN_MATCHERS)) {
  const splits = name.split('.')
  const pythonImport = `from ${splits.slice(0, -2).join('.')} import ${splits.at(-2)}`

  Deno.test(`${name}_patterns`, async (t) => {
    const TSPatterns = matcher.patterns.map((pattern) => pattern[0])
    const PYPatterns = await python(`${pythonImport}\nout([pattern[0] for pattern in ${splits.slice(-2).join('.')}.patterns])`)
    for (const [i, [ts, py]] of zip(TSPatterns, PYPatterns).entries()) {
      await t.step(i.toString(), () => {
        expect(asdict(removeKeys(ts, ['location', 'op']))).toEqual(asdict(removeKeys(py, ['location', 'op'])))
      })
    }
  })

  Deno.test(`${name}_pdict`, async (t) => {
    const PYDict = await python<Record<string, [UPat, undefined, Ops[], boolean][]>>(`${pythonImport}\nout(${splits.slice(-2).join('.')}.pdict)`)
    for (const [key, ts] of matcher.pdict.entries()) {
      const py = PYDict[key]
      for (const [i, [ts1, py1]] of zip(ts as any[], py).entries()) {
        await t.step(i.toString(), () => {
          expect(asdict(removeKeys(ts1[0], ['location', 'op']))).toEqual(asdict(removeKeys(py1[0], ['location', 'op']))) //UPat
          expect([...ts1[2]].toSorted()).toEqual(py1[2].toSorted()) // Ops[]
          expect(ts1[3]).toEqual(py1[3]) // has ctx?
        })
      }
    }
  })

  for (const [i, uop] of uops.entries()) {
    Deno.test(`${name}_${i}`, async () => {
      const ts = tryCatch(() => matcher.rewrite(uop, new Map([[uop, 'somectxvalue']])))()
      const py = await python(`${pythonImport}\nout(${splits.slice(-2).join('.')}.rewrite(data,{data:"somectxvalue"}))`, uop)
      expect(asdict(ts)).toEqual(asdict(py))
    })
  }
}
