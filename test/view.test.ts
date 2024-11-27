import { expect } from 'expect'
import { Ops, type sint, spec, UOp, UPat } from '../src/ops.ts'
import { runPython,asdict } from './helpers.ts'
import { dtypes } from '../src/dtype.ts'

Deno.test('serialization', async () => {
    const data = [
        Ops.ADD,
        Ops.ASSIGN,
        new UOp({ op: Ops.BARRIER, dtype: dtypes.float, arg: 5445,  }),
        new UPat({ op: Ops.ASSIGN, dtype: dtypes.floats, arg: 555, name: 'sdf' }),
        new UPat({ op: Ops.ASSIGN,}),
        dtypes.floats,
        dtypes.defaultFloat,
        // ...spec.patterns.map(p=>p[0])
    ]
    const res2 = await runPython('out(data)', data)
    expect(asdict(res2)).toEqual(asdict(data))
})
