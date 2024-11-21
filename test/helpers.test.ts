import * as helpers from '../src/helpers.ts'
import { tiny, tinyTest } from './helpers.ts'

tinyTest('argfix', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argfix, (x) => tiny`out(tiny.helpers.argfix(${x}))`)
tinyTest('argfix2', [[1, 2, 3], [1, 4, 2]], helpers.argfix, (x, y, z) => tiny`out(tiny.helpers.argfix(${x},${y},${z}))`)

tinyTest('argsort', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argsort, (x) => tiny`out(tiny.helpers.argsort(${x}))`)

tinyTest('memsizeToStr', [[123], [100_000], [2_450_000]], helpers.memsizeToStr, (x) => tiny`out(tiny.helpers.memsize_to_str(${x}))`)

tinyTest('dedup', [[[1, 2, 3, 2, 1]], [[1, 2, 3]]], helpers.dedup, (x) => tiny`out(tiny.helpers.dedup(${x}))`)

tinyTest('allSame', [[[1, 1, 1]], [[1, 2, 3]]], helpers.allSame, (x) => tiny`out(tiny.helpers.all_same(${x}))`)

tinyTest('allInt', [[[1, 2, 3]], [[1, 2.5, 3]]], helpers.allInt, (x) => tiny`out(tiny.helpers.all_int(${x}))`)

tinyTest('colored', [['test', 'red'], ['test', 'green']], helpers.colored, (x, y) => tiny`out(tiny.helpers.colored(${x}, ${y}))`)

tinyTest('colorizeFloat', [[0.5], [1.0], [1.2]], helpers.colorizeFloat, (x) => tiny`out(tiny.helpers.colorize_float(${x}))`)

tinyTest('ansistrip', [['\u001b[31mtest\u001b[0m'], ['\u001b[31mtest'], ['test']], helpers.ansistrip, (x) => tiny`out(tiny.helpers.ansistrip(${x}))`)

tinyTest('ansilen', [['\u001b[31mtest\u001b[0m'], ['\u001b[31mtest'], ['test']], helpers.ansilen, (x) => tiny`out(tiny.helpers.ansilen(${x}))`)

tinyTest('makeTuple', [[[1, 2, 3], 3], [[1, 2, 3], 2], [[1, 2, 3], 1], [1, 3], [1, 2], [1, 1]], helpers.makeTuple, (x, cnt) => tiny`out(tiny.helpers.make_tuple(${x}, ${cnt}))`)

tinyTest('flatten', [[[[1, 2], [3, 4]]], [[[1, 2, 3, 4]]], [[[1, 2, 3, [4, 5], 6]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]] as any, helpers.flatten, (x) => tiny`out(tiny.helpers.flatten(${x}))`)

tinyTest('fully_flatten', [[[1, 2, 3]], [[1, 2, 3, 4, 5]], [[[1, 2], [3, 4], [5]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]], helpers.fullyFlatten, (x) => tiny`out(tiny.helpers.fully_flatten(${x}))`)

// TODO: fromimport

tinyTest('stripParens', [['sdfsdf'], ['(sdfsdf'], ['(sdfsdf)']], helpers.stripParens, (x) => tiny`out(tiny.helpers.strip_parens(${x}))`)

tinyTest('ceildiv', [[180, 3], [-10, 3], [10.5, 3], [-10.5, 3]], helpers.ceildiv, (x, y) => tiny`out(tiny.helpers.ceildiv(${x}, ${y}))`)

tinyTest('roundUp', [[180, 3], [10, 3], [11, 3], [12, 3]], helpers.roundUp, (x, y) => tiny`out(tiny.helpers.round_up(${x}, ${y}))`)

tinyTest('data64', [[333], [45443], [0], [-1], [NaN], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64, (x) => tiny`out(tiny.helpers.data64(${x}))`)

tinyTest('data64_le', [[333], [3434], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64Le, (x) => tiny`out(tiny.helpers.data64_le(${x}))`)

tinyTest('mergeDicts', [[[{ a: 1 }, { b: 2 }], [{ x: 1, y: 2 }, { z: 3 }]]] as any, helpers.mergeDicts, (x) => tiny`out(tiny.helpers.merge_dicts(${x}))`)

tinyTest('partition', [[[1, 2, 3, 4, 5], (x: number) => x % 2 === 0]], helpers.partition, (arr) => tiny`out(tiny.helpers.partition(${arr}, lambda x: x % 2 == 0))`)
tinyTest('partition', [[[2, 4, 6, 8, 10], (x: number) => x > 5]], helpers.partition, (arr) => tiny`out(tiny.helpers.partition(${arr}, lambda x: x > 5))`)
tinyTest('partition', [[['a', 'b', 'c', 'd'], (x: string) => x > 'b']], helpers.partition, (arr) => tiny`out(tiny.helpers.partition(${arr}, lambda x: x > "b"))`)

tinyTest('unwrap', [[1], [2], ['sdf']], helpers.unwrap, (x) => tiny`out(tiny.helpers.unwrap(${x}))`)

tinyTest('get_child', [[{ a: 1, b: { c: 2 } }, 'b.c'], [{ a: { x: [33, 54] }, b: { c: 2 } }, 'a.x.0'], [[3, { a: { v: 'true' } }], '1.a.v']], helpers.getChild, (x, y) => tiny`out(tiny.helpers.get_child(${x}, ${y}))`)

tinyTest('word_wrap', [['This is a long string that needs to be wrapped to fit within 80 characters. Sfasdf dsafg sdf sdf sdf sdf sdf s dfs df']], helpers.wordWrap, (x) => tiny`out(tiny.helpers.word_wrap(${x}))`)

tinyTest('polyN', [[2, [1, 2, 3]], [2, [1, 2, 3]]], helpers.polyN, (x, p) => tiny`out(tiny.helpers.polyN(${x}, ${p}))`)

tinyTest('to_function_name', [['test'], ['not sure how this should work'], ['letsTryThisOne']], helpers.toFunctionName, (s) => tiny`out(tiny.helpers.to_function_name(${s}))`)

tinyTest('getenv', [['key', 0]], helpers.getenv, (key, defaultVal) => tiny`out(tiny.helpers.getenv(${key}, ${defaultVal}))`)

tinyTest('temp', [['file.txt']], helpers.temp, (x) => tiny`out(tiny.helpers.temp(${x}))`)
