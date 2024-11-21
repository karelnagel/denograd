import * as helpers from '../src/helpers.ts'
import { tinyTest } from './helpers.ts'

tinyTest('argfix', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argfix, (x) => `out(tiny.helpers.argfix(${JSON.stringify(x)}))`)
tinyTest('argfix2', [[1, 2, 3], [1, 4, 2]], helpers.argfix, (x, y, z) => `out(tiny.helpers.argfix(${x},${y},${z}))`)

tinyTest('argsort', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argsort, (x) => `out(tiny.helpers.argsort(${JSON.stringify(x)}))`)

tinyTest('memsizeToStr', [[123], [100_000], [2_450_000]], helpers.memsizeToStr, (x) => `out(tiny.helpers.memsize_to_str(${x}))`)

tinyTest('dedup', [[[1, 2, 3, 2, 1]], [[1, 2, 3]]], helpers.dedup, (x) => `out(tiny.helpers.dedup(${JSON.stringify(x)}))`)

tinyTest('allSame', [[[1, 1, 1]], [[1, 2, 3]]], helpers.allSame, (x) => `out(tiny.helpers.all_same(${JSON.stringify(x)}))`)

tinyTest('allInt', [[[1, 2, 3]], [[1, 2.5, 3]]], helpers.allInt, (x) => `out(tiny.helpers.all_int(${JSON.stringify(x)}))`)

tinyTest('colored', [['test', 'red'], ['test', 'green']], helpers.colored, (x, y) => `out(tiny.helpers.colored(${JSON.stringify(x)}, ${JSON.stringify(y)}))`)

tinyTest('colorizeFloat', [[0.5], [1.0], [1.2]], helpers.colorizeFloat, (x) => `out(tiny.helpers.colorize_float(${x}))`)

tinyTest('ansistrip', [['\u001b[31mtest\u001b[0m'], ['\u001b[31mtest'], ['test']], helpers.ansistrip, (x) => `out(tiny.helpers.ansistrip(${JSON.stringify(x)}))`)

tinyTest('ansilen', [['\u001b[31mtest\u001b[0m'], ['\u001b[31mtest'], ['test']], helpers.ansilen, (x) => `out(tiny.helpers.ansilen(${JSON.stringify(x)}))`)

tinyTest('makeTuple', [[[1, 2, 3], 3], [[1, 2, 3], 2], [[1, 2, 3], 1], [1, 3], [1, 2], [1, 1]], helpers.makeTuple, (x, cnt) => `out(tiny.helpers.make_tuple(${JSON.stringify(x)}, ${cnt}))`)

tinyTest('flatten', [[[[1, 2], [3, 4]]], [[[1, 2, 3, 4]]], [[[1, 2, 3, [4, 5], 6]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]] as any, helpers.flatten, (x) => `out(tiny.helpers.flatten(${JSON.stringify(x)}))`)

tinyTest('fully_flatten', [[[1, 2, 3]], [[1, 2, 3, 4, 5]], [[[1, 2], [3, 4], [5]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]], helpers.fullyFlatten, (x) => `out(tiny.helpers.fully_flatten(${JSON.stringify(x)}))`)

// TODO: fromimport

tinyTest('stripParens', [['sdfsdf'], ['(sdfsdf'], ['(sdfsdf)']], helpers.stripParens, (x) => `out(tiny.helpers.strip_parens(${JSON.stringify(x)}))`)

tinyTest('ceildiv', [[180, 3], [-10, 3], [10.5, 3], [-10.5, 3]], helpers.ceildiv, (x, y) => `out(tiny.helpers.ceildiv(${x}, ${y}))`)

tinyTest('roundUp', [[180, 3], [10, 3], [11, 3], [12, 3]], helpers.roundUp, (x, y) => `out(tiny.helpers.round_up(${x}, ${y}))`)

tinyTest('data64', [[333], [45443], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64, (x) => `out(tiny.helpers.data64(${x}))`)

tinyTest('data64_le', [[333], [3434], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64Le, (x) => `out(tiny.helpers.data64_le(${x}))`)

tinyTest('mergeDicts', [[[{ a: 1 }, { b: 2 }], [{ x: 1, y: 2 }, { z: 3 }]]] as any, helpers.mergeDicts, (x) => `out(tiny.helpers.merge_dicts(${JSON.stringify(x)}))`)

tinyTest('partition', [[[1, 2, 3, 4, 5], (x: number) => x % 2 === 0]], helpers.partition, (arr) => `out(tiny.helpers.partition(${JSON.stringify(arr)}, lambda x: x % 2 == 0))`)
tinyTest('partition', [[[2, 4, 6, 8, 10], (x: number) => x > 5]], helpers.partition, (arr) => `out(tiny.helpers.partition(${JSON.stringify(arr)}, lambda x: x > 5))`)
tinyTest('partition', [[['a', 'b', 'c', 'd'], (x: string) => x > 'b']], helpers.partition, (arr) => `out(tiny.helpers.partition(${JSON.stringify(arr)}, lambda x: x > "b"))`)

tinyTest('unwrap', [[1], [2], ['sdf']], helpers.unwrap, (x) => `out(tiny.helpers.unwrap(${JSON.stringify(x)}))`)

tinyTest('get_child', [[{ a: 1, b: { c: 2 } }, 'b.c'], [{ a: { x: [33, 54] }, b: { c: 2 } }, 'a.x.0'], [[3, { a: { v: 'true' } }], '1.a.v']], helpers.getChild, (x, y) => `out(tiny.helpers.get_child(${JSON.stringify(x)}, ${JSON.stringify(y)}))`)

tinyTest('word_wrap', [['This is a long string that needs to be wrapped to fit within 80 characters. Sfasdf dsafg sdf sdf sdf sdf sdf s dfs df']], helpers.wordWrap, (x) => `out(tiny.helpers.word_wrap(${JSON.stringify(x)}))`)

tinyTest('polyN', [[2, [1, 2, 3]], [2, [1, 2, 3]]], helpers.polyN, (x, p) => `out(tiny.helpers.polyN(${x}, ${JSON.stringify(p)}))`)

tinyTest('to_function_name', [['test'], ['not sure how this should work'], ['letsTryThisOne']], helpers.toFunctionName, (s) => `out(tiny.helpers.to_function_name(${JSON.stringify(s)}))`)

tinyTest('getenv', [['key', 0]], helpers.getenv, (key, defaultVal) => `out(tiny.helpers.getenv(${JSON.stringify(key)}, ${defaultVal}))`)

tinyTest('temp', [['file.txt']], helpers.temp, (x) => `out(tiny.helpers.temp(${JSON.stringify(x)}))`)
