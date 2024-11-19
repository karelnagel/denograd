import * as helpers from '../src/helpers.ts'
import { tiny } from './helpers.ts'
import { assertEquals } from 'jsr:@std/assert'

Deno.test('helpers', async (t) => {
    const equals = async <T extends any[]>(name: string, inputs: T[], fn: (...args: T) => any, python: (...args: T) => string) => {
        await t.step(name, async () => {
            for (const input of inputs) {
                assertEquals(fn(...input), await tiny`${python(...input)}`)
            }
        })
    }
    await equals('argfix', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argfix, (x) => `out(tiny.helpers.argfix(${JSON.stringify(x)}))`)
    await equals('argfix2', [[1, 2, 3], [1, 4, 2]], helpers.argfix, (x, y, z) => `out(tiny.helpers.argfix(${x},${y},${z}))`)

    await equals('argsort', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argsort, (x) => `out(tiny.helpers.argsort(${JSON.stringify(x)}))`)

    await equals('memsizeToStr', [[123], [100_000], [2_450_000]], helpers.memsizeToStr, (x) => `out(tiny.helpers.memsize_to_str(${x}))`)

    await equals('dedup', [[[1, 2, 3, 2, 1]], [[1, 2, 3]]], helpers.dedup, (x) => `out(tiny.helpers.dedup(${JSON.stringify(x)}))`)

    await equals('allSame', [[[1, 1, 1]], [[1, 2, 3]]], helpers.allSame, (x) => `out(tiny.helpers.all_same(${JSON.stringify(x)}))`)

    await equals('allInt', [[[1, 2, 3]], [[1, 2.5, 3]]], helpers.allInt, (x) => `out(tiny.helpers.all_int(${JSON.stringify(x)}))`)

    await equals('colored', [['test', 'red'], ['test', 'green']], helpers.colored, (x, y) => `out(tiny.helpers.colored(${JSON.stringify(x)}, ${JSON.stringify(y)}))`)

    await equals('colorizeFloat', [[0.5], [1.0], [1.2]], helpers.colorizeFloat, (x) => `out(tiny.helpers.colorize_float(${x}))`)

    await equals('ansistrip', [['\u001b[31mtest\u001b[0m'], ['\u001b[31mtest'], ['test']], helpers.ansistrip, (x) => `out(tiny.helpers.ansistrip(${JSON.stringify(x)}))`)

    await equals('ansilen', [['\u001b[31mtest\u001b[0m'], ['\u001b[31mtest'], ['test']], helpers.ansilen, (x) => `out(tiny.helpers.ansilen(${JSON.stringify(x)}))`)

    await equals('makeTuple', [[[1, 2, 3], 3], [[1, 2, 3], 2], [[1, 2, 3], 1], [1, 3], [1, 2], [1, 1]], helpers.makeTuple, (x, cnt) => `out(tiny.helpers.make_tuple(${JSON.stringify(x)}, ${cnt}))`)

    await equals('flatten', [[[[1, 2], [3, 4]]], [[[1, 2, 3, 4]]], [[[1, 2, 3, [4, 5], 6]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]] as any, helpers.flatten, (x) => `out(tiny.helpers.flatten(${JSON.stringify(x)}))`)

    await equals('fully_flatten', [[[1, 2, 3]], [[1, 2, 3, 4, 5]], [[[1, 2], [3, 4], [5]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]], helpers.fullyFlatten, (x) => `out(tiny.helpers.fully_flatten(${JSON.stringify(x)}))`)

    // TODO: fromimport

    await equals('stripParens', [['sdfsdf'], ['(sdfsdf'], ['(sdfsdf)']], helpers.stripParens, (x) => `out(tiny.helpers.strip_parens(${JSON.stringify(x)}))`)

    await equals('ceildiv', [[180, 3], [-10, 3], [10.5, 3], [-10.5, 3]], helpers.ceildiv, (x, y) => `out(tiny.helpers.ceildiv(${x}, ${y}))`)

    await equals('roundUp', [[180, 3], [10, 3], [11, 3], [12, 3]], helpers.roundUp, (x, y) => `out(tiny.helpers.round_up(${x}, ${y}))`)

    await equals('data64', [[333], [45443], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64, (x) => `out(tiny.helpers.data64(${x}))`)

    await equals('data64_le', [[333], [3434], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64_le, (x) => `out(tiny.helpers.data64_le(${x}))`)

    await equals('mergeDicts', [[[{ a: 1 }, { b: 2 }], [{ x: 1, y: 2 }, { z: 3 }]]] as any, helpers.mergeDicts, (x) => `out(tiny.helpers.merge_dicts(${JSON.stringify(x)}))`)

    await equals('partition', [[[1, 2, 3, 4, 5], (x: number) => x % 2 === 0]], helpers.partition, (arr) => `out(tiny.helpers.partition(${JSON.stringify(arr)}, lambda x: x % 2 == 0))`)
    await equals('partition', [[[2, 4, 6, 8, 10], (x: number) => x > 5]], helpers.partition, (arr) => `out(tiny.helpers.partition(${JSON.stringify(arr)}, lambda x: x > 5))`)
    await equals('partition', [[['a', 'b', 'c', 'd'], (x: string) => x > 'b']], helpers.partition, (arr) => `out(tiny.helpers.partition(${JSON.stringify(arr)}, lambda x: x > "b"))`)
})
