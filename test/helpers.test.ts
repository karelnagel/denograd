import * as helpers from '../src/helpers.ts'
import { test } from './helpers.ts'

Deno.test('argfix', test([[[1, 2, 3]], [[1, 4, 2]]], helpers.argfix, `out(tiny.helpers.argfix(*data))`))

Deno.test('argfix2', test([[1, 2, 3], [1, 4, 2]], helpers.argfix, `out(tiny.helpers.argfix(*data))`))

Deno.test('argsort', test([[[1, 2, 3]], [[1, 4, 2]]], helpers.argsort, 'out(tiny.helpers.argsort(*data))'))

Deno.test('memsizeToStr', test([[123], [100_000], [2_450_000]], helpers.memsizeToStr, 'out(tiny.helpers.memsize_to_str(*data))'))

Deno.test('dedup', test([[[1, 2, 3, 2, 1]], [[1, 2, 3]]], helpers.dedup, 'out(tiny.helpers.dedup(*data))'))

Deno.test('allSame', test([[[1, 1, 1]], [[1, 2, 3]]], helpers.allSame, 'out(tiny.helpers.all_same(*data))'))

Deno.test('allInt', test([[[1, 2, 3]], [[1, 2.5, 3]]], helpers.allInt, 'out(tiny.helpers.all_int(*data))'))

Deno.test('colored', test([['test', 'red'], ['test', 'green']], helpers.colored, 'out(tiny.helpers.colored(*data))'))

Deno.test('colorizeFloat', test([[0.5], [1.0], [1.2]], helpers.colorizeFloat, 'out(tiny.helpers.colorize_float(*data))'))

Deno.test('ansistrip', test([[' sdf sdf sdf sdfasdf'], ['test']], helpers.ansistrip, 'out(tiny.helpers.ansistrip(*data))'))

Deno.test('ansilen', test([['asdf asdf asdf asdf '], ['test']], helpers.ansilen, 'out(tiny.helpers.ansilen(*data))'))

Deno.test('makeTuple', test([[[1, 2, 3], 3], [[1, 2, 3], 2], [[1, 2, 3], 1], [1, 3], [1, 2], [1, 1]], helpers.makeTuple, 'out(tiny.helpers.make_tuple(*data))'))

Deno.test('flatten', test([[[[1, 2], [3, 4]]], [[[1, 2, 3, 4]]], [[[1, 2, 3, [4, 5], 6]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]] as any, helpers.flatten, 'out(tiny.helpers.flatten(*data))'))

Deno.test('fully_flatten', test([[[1, 2, 3]], [[1, 2, 3, 4, 5]], [[[1, 2], [3, 4], [5]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]], helpers.fullyFlatten, 'out(tiny.helpers.fully_flatten(*data))'))

// // TODO: fromimport

Deno.test('stripParens', test([['sdfsdf'], ['(sdfsdf'], ['(sdfsdf)']], helpers.stripParens, 'out(tiny.helpers.strip_parens(*data))'))

Deno.test('ceildiv', test([[180, 3], [-10, 3], [10.5, 3], [-10.5, 3]], helpers.ceildiv, 'out(tiny.helpers.ceildiv(*data))'))

Deno.test('roundUp', test([[180, 3], [10, 3], [11, 3], [12, 3]], helpers.roundUp, 'out(tiny.helpers.round_up(*data))'))

Deno.test('data64', test([[333], [45443], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64, 'out(tiny.helpers.data64(*data))'))

Deno.test('data64_le', test([[333], [3434], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64Le, 'out(tiny.helpers.data64_le(*data))'))

Deno.test('mergeDicts', test([[[{ a: 1 }, { b: 2 }], [{ x: 1, y: 2 }, { z: 3 }]]] as any, helpers.mergeDicts, 'out(tiny.helpers.merge_dicts(data[0]))'))

Deno.test('partition', test([[[1, 2, 3, 4, 5], (x: number) => x % 2 === 0]], helpers.partition, 'out(tiny.helpers.partition(data[0], lambda x: x % 2 == 0))'))
Deno.test('partition', test([[[2, 4, 6, 8, 10], (x: number) => x > 5]], helpers.partition, 'out(tiny.helpers.partition(data[0], lambda x: x > 5))'))
Deno.test('partition', test([[['a', 'b', 'c', 'd'], (x: string) => x > 'b']], helpers.partition, "out(tiny.helpers.partition(data[0], lambda x: x > 'b'))"))

Deno.test('unwrap', test([[1], [2], ['sdf']], helpers.unwrap, 'out(tiny.helpers.unwrap(*data))'))

Deno.test('get_child', test([[{ a: 1, b: { c: 2 } }, 'b.c'], [{ a: { x: [33, 54] }, b: { c: 2 } }, 'a.x.0'], [[3, { a: { v: 'true' } }], '1.a.v']], helpers.getChild, 'out(tiny.helpers.get_child(*data))'))

Deno.test('word_wrap', test([['This is a long string that needs to be wrapped to fit within 80 characters. Sfasdf dsafg sdf sdf sdf sdf sdf s dfs df']], helpers.wordWrap, 'out(tiny.helpers.word_wrap(*data))'))

Deno.test('polyN', test([[2, [1, 2, 3]], [2, [1, 2, 3]]], helpers.polyN, 'out(tiny.helpers.polyN(*data))'))

Deno.test('to_function_name', test([['test'], ['not sure how this should work'], ['letsTryThisOne']], helpers.toFunctionName, `out(tiny.helpers.to_function_name(*data))`))

Deno.test('getenv', test([['key', 'value']], helpers.getEnv, 'out(tiny.helpers.getenv(*data))'))

Deno.test('temp', test([['file.txt']], helpers.temp, 'out(tiny.helpers.temp(*data))'))
