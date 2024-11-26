import { expect } from 'expect/expect'
import { spec, symbolicFlat } from '../src/ops.ts'
import { runPython } from './helpers.ts'

Deno.test('spec', async (t) => {
    const patterns = await runPython(`out([str(pattern[0]) for pattern in tiny.ops.spec.patterns])`)

    expect(spec.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(spec.patterns[i][0].__repr__()).toEqual(pattern))
    }
})

// symbolicSimple+symbolic+symbolicfalt
Deno.test('symbolicFlat', async (t) => {
    const patterns = await runPython(`out([str(pattern[0]) for pattern in tiny.ops.symbolic_flat.patterns])`)

    expect(symbolicFlat.patterns.length).toBe(patterns.length)
    for (const [i, pattern] of patterns.entries()) {
        await t.step(i.toString(), () => expect(symbolicFlat.patterns[i][0].__repr__()).toEqual(pattern))
    }
})
