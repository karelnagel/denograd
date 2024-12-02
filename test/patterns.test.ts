import { expect } from 'expect/expect'
import { _substitute, Ops, renderer, spec, symbolicFlat, UOp, type UPat } from '../src/ops.ts'
import { asdict, python, removeKeys, tryCatch } from './helpers.ts'
import { baseRewrite, extraPm } from '../src/renderer/cstyle.ts'
import { entries, zip } from '../src/helpers.ts'
import { type DType, dtypes } from '../src/dtype.ts'
import { symbolicSimple } from '../src/ops.ts'
import { symbolic } from '../src/ops.ts'

const variable = (name: string, dtype: DType = dtypes.int) => UOp.variable(name, 0, 999, dtype)
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
      variable('x').add(0),
      variable('x').mul(UOp.int(1)),
      variable('x').idiv(variable('x')),
      variable('x').idiv(1),
      variable('x').idiv(-1),
      variable('x').div(variable('x')),
      variable('x').mul(variable('x2')).div(variable('x2')),
      variable('base').mod(variable('y')).mod(variable('y')),
      variable('x').mod(UOp.int(1)).add(variable('x').idiv(UOp.int(1)).mul(UOp.int(1))),
      variable('x', dtypes.bool).bitwiseAnd(UOp.bool(false)),
      variable('x', dtypes.bool).bitwiseOr(UOp.bool(false)),

      variable('x').maximum(variable('x')),
      variable('x').bitwiseAnd(variable('x')),
      variable('x').bitwiseOr(variable('x')),
      variable('x', dtypes.bool).logicalNot().logicalNot(),
      variable('x', dtypes.bool).where(UOp.bool(true), UOp.bool(false)),

      variable('x').lt(variable('x')),
      variable('x', dtypes.int32).ne(variable('x', dtypes.int32)),
      variable('x', dtypes.int16).ne(variable('x', dtypes.int16)),
      variable('x', dtypes.int8).ne(variable('x', dtypes.int8)),
      variable('x', dtypes.int64).ne(variable('x', dtypes.int64)),

      variable('x').mul(0),
      variable('x').mul(UOp.int(0)),
      new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.CONST }), new UOp({ op: Ops.CONST })] }),
      variable('x', dtypes.bool).mul(variable('y', dtypes.bool)),
      variable('x', dtypes.bool).add(variable('y', dtypes.bool)),
      variable('x', dtypes.bool).maximum(variable('y', dtypes.bool)),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.CONST })], dtype: dtypes.float32 }),
      new UOp({ op: Ops.CAST, src: [new UOp({ op: Ops.CONST, dtype: dtypes.float32 })], dtype: dtypes.float32 }),
    ],
  },
  'tinygrad.ops.symbolic': {
    matcher: symbolic,
    uops: [
      variable('x').add(variable('y')).add(variable('x').mul(UOp.int(5))), // group like
      variable('x').bitwiseOr(variable('x').bitwiseAnd(variable('y'))), // boolean algebra
      variable('x').mul(UOp.int(2)).add(variable('x').mul(UOp.int(3))), // combine terms
      variable('x').add(variable('x').mul(UOp.int(3))), // x + x*c -> x*(c+1)
      variable('x').add(variable('x')), // x + x -> x*2
      variable('x').div(variable('x2')).div(variable('x3')), // (x/x2)/x3 -> x/(x2*x3)
      variable('x').add(UOp.int(5)).mul(-1), // -(x+c) -> -x + -c

      variable('val').where(variable('val'), variable('val')), // same results either way is noop
      UOp.bool(false).where(variable('c0'), variable('c1')), // const gate folding
      UOp.bool(true).where(variable('c0'), variable('c1')), // const gate folding
      new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.CONST, arg: 5 })] }), // ALU min==max -> CONST
      variable('x').maximum(variable('y')), // max folding when x.vmax <= y.vmin
      variable('x').mul(UOp.int(2)).maximum(variable('x').mul(UOp.int(3))), // maxVarConst

      variable('x').add(UOp.int(2)).add(UOp.int(3)), // (x+c1)+c2 -> x+(c1+c2)
      variable('x').mul(UOp.int(2)).mul(UOp.int(3)), // (x*c1)*c2 -> x*(c1*c2)
      variable('x').bitwiseAnd(UOp.int(2)).bitwiseAnd(UOp.int(3)), // (x&c1)&c2 -> x&(c1&c2)
      variable('x').bitwiseOr(UOp.int(2)).bitwiseOr(UOp.int(3)), // (x|c1)|c2 -> x|(c1|c2)
      UOp.int(2).add(variable('x')).lt(UOp.int(5)), // c0+x<c1 -> x<c1-c0
      variable('x').idiv(UOp.int(2)).idiv(UOp.int(3)), // (x//c1)//c2 -> x//(c1*c2)
      UOp.int(2).mul(variable('x', dtypes.int)).lt(UOp.int(5)), // 2x < 5 -> x < ceil(5/2)
      UOp.int(-2).mul(variable('x', dtypes.int)).lt(UOp.int(-5)), // -2x < -5 -> -x < -floor(-(-5)/-(-2))
      variable('x', dtypes.int).idiv(UOp.int(2)).lt(UOp.int(3)), // x//2 < 3 -> x < 3*2
      UOp.int(4).mul(variable('x')).add(variable('x2')).lt(UOp.int(12)), // (4x + x2) < 12 -> x < 12/4 when 12%4=0 and 4>x2.vmax() and x2.vmin()>=0
      variable('x', dtypes.int).lt(UOp.int(5)), // x < 5 when 0 < 5
      variable('x', dtypes.int).lt(1).ne(true), // not x < 1 -> X > 0
      variable('x', dtypes.int).idiv(UOp.int(3)), // x//3 when 0 < 3
      variable('x').mod(UOp.int(4)), // x%4 when 0 < 4
    ],
  },
  'tinygrad.ops.symbolic_flat': {
    matcher: symbolicFlat,
    uops: [
      variable('x').add(variable('y')).mul(-1),
      variable('x', dtypes.int).add(variable('y')).mul(UOp.int(3)),
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
  'tinygrad.renderer.cstyle.base_rewrite': {
    matcher: baseRewrite,
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
    matcher: extraPm,
    uops: [],
  },
}
for (const [name, { matcher, uops }] of entries(ALL_PATTERN_MATCHERS)) {
  const splits = name.split('.')
  const pythonImport = `from ${splits.slice(0, -2).join('.')} import ${splits.at(-2)}`

  Deno.test(name, async (t) => {
    await t.step(`${name}_patterns`, async () => {
      const TSPatterns = matcher.patterns.map((pattern) => pattern[0])
      const PYPatterns = await python(`${pythonImport}\nout([pattern[0] for pattern in ${splits.slice(-2).join('.')}.patterns])`)
      for (const [ts, py] of zip(TSPatterns, PYPatterns)) {
        expect(asdict(removeKeys(ts, ['location', 'op']))).toEqual(asdict(removeKeys(py, ['location', 'op'])))
      }
    })

    await t.step(`${name}_pdict`, async () => {
      const PYDict = await python<Record<string, [UPat, undefined, Ops[], boolean][]>>(`${pythonImport}\nout(${splits.slice(-2).join('.')}.pdict)`)
      for (const [key, ts] of matcher.pdict.entries()) {
        const py = PYDict[key]
        for (const [ts1, py1] of zip(ts as any[], py)) {
          expect(asdict(removeKeys(ts1[0], ['location', 'op']))).toEqual(asdict(removeKeys(py1[0], ['location', 'op']))) //UPat
          expect([...ts1[2]].toSorted()).toEqual(py1[2].toSorted()) // Ops[]
          expect(ts1[3]).toEqual(py1[3]) // has ctx?
        }
      }
    })

    for (const [i, uop] of uops.entries()) {
      await t.step(`${name}_${i}_${uop}`, async () => {
        const ts = tryCatch(() => matcher.rewrite(uop, new Map([[uop, 'somectxvalue']])))()
        const py = await python(`${pythonImport}\nout(${splits.slice(-2).join('.')}.rewrite(data,{data:"somectxvalue"}))`, uop)
        expect(asdict(ts), "Shouldn't be the same as initial").not.toEqual(asdict(uop))
        expect(asdict(ts)).toEqual(asdict(py))
      })
    }
  })
}
