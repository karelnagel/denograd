import { expect } from 'expect/expect'
import { _substitute, spec, symbolicFlat } from '../src/ops.ts'
import { runPython } from './helpers.ts'
import { renderer } from '../src/ops.ts'

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
        await t.step(i.toString(), () => expect(symbolicFlat.patterns[i][0].__repr__().replaceAll("[","(").replaceAll("]",")")).toEqual(pattern.replaceAll("[","(").replaceAll("]",")")))
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
