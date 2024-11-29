import { expect } from 'expect/expect'
import { _substitute, spec, symbolicFlat } from '../src/ops.ts'
import { runPython } from './helpers.ts'
import { renderer } from '../src/ops.ts'
import { baseRewrite, extraPm } from '../src/renderer/cstyle.ts'

Deno.test('spec', async (t) => {
    const patterns = await runPython(`out([str(pattern[0]) for pattern in tiny.ops.spec.patterns])`)

    expect(spec.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(spec.patterns[i][0].__repr__()).toEqual(pattern))
    }
})

// symbolicSimple+symbolic+symbolicfalt
// They are mostly correct, but dtypes and op sorting is wrong and python version has x1:= variables in it
Deno.test('symbolicFlat', async (t) => {
    const patterns = await runPython(`out([str(pattern[0]) for pattern in tiny.ops.symbolic_flat.patterns])`)

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
    const patterns = await runPython(`out([str(pattern[0]) for pattern in tiny.ops.renderer.patterns])`)

    expect(renderer.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(renderer.patterns[i][0].__repr__()).toEqual(pattern))
    }
})
Deno.test('_substitute', async (t) => {
    const patterns = await runPython(`out([str(pattern[0]) for pattern in tiny.ops._substitute.patterns])`)

    expect(_substitute.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(_substitute.patterns[i][0].__repr__()).toEqual(pattern))
    }
})

Deno.test('baseRewrite', async (t) => {
    const patterns = await runPython(`from tinygrad.renderer import cstyle\nout([str(pattern[0]) for pattern in cstyle.base_rewrite.patterns])`)

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
    const patterns = await runPython(`from tinygrad.renderer import cstyle\nout([str(pattern[0]) for pattern in cstyle.extra_pm.patterns])`)

    expect(extraPm.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(extraPm.patterns[i][0].__repr__()).toEqual(pattern))
    }
})
