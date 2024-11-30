import { expect } from 'expect/expect'
import { _substitute, canPad, Ops, resolve, spec, symbolicFlat, UOp, type UPat } from '../src/ops.ts'
import { asdict, python, removeKeys, test, tryCatch } from './helpers.ts'
import { renderer } from '../src/ops.ts'
import { baseRewrite, extraPm } from '../src/renderer/cstyle.ts'
import { range } from '../src/helpers.ts'
import { dtypes } from '../src/dtype.ts'

Deno.test(
    'canPad',
    test(
        [
            [new UOp({ op: Ops.RECIP })],
            [new UOp({ op: Ops.ADD })],

            [new UOp({ op: Ops.RECIP, src: [new UOp({ op: Ops.IDIV })] })],
            [new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.IDIV })] })],
        ],
        canPad,
        'out(tiny.ops.can_pad(*data))',
    ),
)
Deno.test(
    'resolve',
    test(
        [
            [new UOp({ op: Ops.ADD, dtype: dtypes.float })],
            [new UOp({ op: Ops.ADD, dtype: dtypes.bool, src: [new UOp({ op: Ops.IDIV })] }), false],
        ],
        tryCatch(resolve),
        'out(trycatch(lambda: tiny.ops.resolve(*data)))',
    ),
)
Deno.test(
    'uop.parents',
    test(
        [
            [new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.BARRIER, src: [new UOp({ op: Ops.CONST, arg: 69 })] })] })],
            [new UOp({ op: Ops.CONST, arg: 1 })],
        ],
        (x: UOp) => [...x.parents().keys()],
        'out(list(data[0].parents.keys()))',
    ),
)
Deno.test(
    'uop.sparents',
    test(
        [
            [new UOp({ op: Ops.ADD, src: [new UOp({ op: Ops.BARRIER, src: [new UOp({ op: Ops.CONST, arg: 69 })] })] })],
            [new UOp({ op: Ops.CONST, arg: 1 })],
        ],
        (x: UOp) => [...x.sparents().keys()],
        'out(list(data[0].sparents.keys()))',
    ),
)
Deno.test(
    'uop.simplify',
    test(
        [
            [new UOp({ op: Ops.ADD, arg: 1, src: [UOp.int(10), UOp.int(100)] })],
            [new UOp({ op: Ops.IDIV, arg: 1, src: [UOp.float(10), UOp.int(100)] })],
            [new UOp({ op: Ops.AND, arg: 1, src: [UOp.boolean(false), UOp.boolean(true)] })],
        ],
        (x: UOp) => tryCatch(x.simplify)(),
        'out(data[0].simplify())',
    ),
)

Deno.test('pdict.symbolic_flat', async () => {
    const res = await python<Record<number, [UPat, undefined, Ops[], boolean][]>>(`out(tiny.ops.symbolic_flat.pdict)`)
    for (const key in res) {
        for (const [i, py] of res[key].entries()) {
            const ts = symbolicFlat.pdict.get(Number(key))![i]
            expect(asdict(removeKeys(ts[0], ['location', 'op']))).toEqual(asdict(removeKeys(py[0], ['location', 'op'])))
            expect([...ts[2]].toSorted()).toEqual(py[2].toSorted())
            expect(ts[3]).toEqual(py[3])
        }
    }
})
Deno.test('pdict.baseRewrite', async () => {
    const res = await python<Record<number, [UPat, undefined, Ops[], boolean][]>>(`from tinygrad.renderer import cstyle\nout(cstyle.base_rewrite.pdict)`)
    for (const key in res) {
        for (const [i, py] of res[key].entries()) {
            const ts = baseRewrite.pdict.get(Number(key))![i]
            expect(asdict(removeKeys(ts[0], ['location', 'op']))).toEqual(asdict(removeKeys(py[0], ['location', 'op'])))
            expect([...ts[2]].toSorted()).toEqual(py[2].toSorted())
            expect(ts[3]).toEqual(py[3])
        }
    }
})

Deno.test(
    'spec',
    test(
        range(spec.patterns.length).map((x) => [x, [
            new UOp({ op: Ops.ADD, arg: [1, 2, 4, 5], src: [new UOp({ op: Ops.CONST, arg: 1 })] }),
            new UOp({ op: Ops.ADD, arg: [1, 2, 4, 5], src: [new UOp({ op: Ops.CONST, arg: 1 })] }),
            new UOp({ op: Ops.ADD, arg: [1, 2, 4, 5], src: [new UOp({ op: Ops.CONST, arg: 1 })] }),
        ]]) as any,
        (x: number, args: UOp[]) => {
            const pattern = spec.patterns[x]
            return { str: pattern[0].__repr__(), value: pattern[1](args[0], args[1], args[2]) }
        },
        `
pattern = tiny.ops.spec.patterns[data[0]]
arg_count = pattern[1].__code__.co_argcount
out({
    "str": str(pattern[0]),
    "value": pattern[1](*data[1][:arg_count]),
})`,
    ),
)

// symbolicSimple+symbolic+symbolicfalt
// They are mostly correct, but dtypes and op sorting is wrong and python version has x1:= variables in it
Deno.test('symbolicFlat', async (t) => {
    const patterns = await python(`out([str(pattern[0]) for pattern in tiny.ops.symbolic_flat.patterns])`)

    expect(symbolicFlat.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step({
            name: i.toString(),
            ignore: [2, 5, 6, 7, 8, 11, 12, 13, 16, 17, 25, 26, 27, 28, 29, 30, 39, 33, 37, 44, 45, 46, 51, 52, 53, 56].includes(i),
            fn: () => expect(symbolicFlat.patterns[i][0].__repr__().replaceAll('[', '(').replaceAll(']', ')')).toEqual(pattern.replaceAll('[', '(').replaceAll(']', ')')),
        })
    }
})

Deno.test('renderer', async (t) => {
    const patterns = await python(`out([str(pattern[0]) for pattern in tiny.ops.renderer.patterns])`)

    expect(renderer.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(renderer.patterns[i][0].__repr__()).toEqual(pattern))
    }
})
Deno.test('_substitute', async (t) => {
    const patterns = await python(`out([str(pattern[0]) for pattern in tiny.ops._substitute.patterns])`)

    expect(_substitute.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(_substitute.patterns[i][0].__repr__()).toEqual(pattern))
    }
})

Deno.test('baseRewrite', async (t) => {
    const patterns = await python(`from tinygrad.renderer import cstyle\nout([str(pattern[0]) for pattern in cstyle.base_rewrite.patterns])`)

    expect(baseRewrite.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step({
            name: i.toString(),
            ignore: [15, 21, 23].includes(i),
            fn: () => expect(baseRewrite.patterns[i][0].__repr__()).toEqual(pattern),
        })
    }
})

Deno.test('extraPm', async (t) => {
    const patterns = await python(`from tinygrad.renderer import cstyle\nout([str(pattern[0]) for pattern in cstyle.extra_pm.patterns])`)

    expect(extraPm.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(extraPm.patterns[i][0].__repr__()).toEqual(pattern))
    }
})
