import { expect } from 'expect/expect'
import { _substitute, Ops, renderer, spec, symbolicFlat, UOp } from '../src/ops.ts'
import { asdict, python, removeKeys } from './helpers.ts'
import { baseRewrite, extraPm } from '../src/renderer/cstyle.ts'
import { entries, zip } from '../src/helpers.ts'
import { dtypes } from '../src/dtype.ts'

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
  'tinygrad.ops.symbolic_flat': {
    matcher: symbolicFlat,
    uops: [
      UOp.variable('x', 0, 999).add(UOp.int(0)),
      UOp.variable('x', 0, 999).idiv(UOp.variable('x', 0, 999)),
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
      UOp.int(4),
    ],
  },
  'tinygrad.renderer.cstyle.base_rewrite': {
    matcher: baseRewrite,
    uops: [
      UOp.int(4),
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
    for (const [i, uop] of uops.entries()) {
      await t.step(`${name}_${i}_${uop}`, async () => {
        const ts = matcher.rewrite(uop, new Map([[uop, 'somectxvalue']]))
        const py = await python(`${pythonImport}\nout(${splits.slice(-2).join('.')}.rewrite(data,{data:"somectxvalue"}))`, uop)
        expect(asdict(ts), "Shouldn't be the same as initial").not.toEqual(asdict(uop))
        expect(asdict(ts)).toEqual(asdict(py))
      })
    }
  })
}
