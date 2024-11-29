import * as helpers from '../src/helpers.ts'
import { runPython, tinyTest } from './helpers.ts'

tinyTest('argfix', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argfix, (x) => runPython(`out(tiny.helpers.argfix(data))`, x))
tinyTest('argfix2', [[1, 2, 3], [1, 4, 2]], helpers.argfix, (...data) => runPython(`out(tiny.helpers.argfix(*data))`, data))

tinyTest('argsort', [[[1, 2, 3]], [[1, 4, 2]]], helpers.argsort, (x) => runPython(`out(tiny.helpers.argsort(data))`, x))

tinyTest('memsizeToStr', [[123], [100_000], [2_450_000]], helpers.memsizeToStr, (x) => runPython(`out(tiny.helpers.memsize_to_str(data))`, x))

tinyTest('dedup', [[[1, 2, 3, 2, 1]], [[1, 2, 3]]], helpers.dedup, (x) => runPython(`out(tiny.helpers.dedup(data))`, x))

tinyTest('allSame', [[[1, 1, 1]], [[1, 2, 3]]], helpers.allSame, (x) => runPython(`out(tiny.helpers.all_same(data))`, x))

tinyTest('allInt', [[[1, 2, 3]], [[1, 2.5, 3]]], helpers.allInt, (x) => runPython(`out(tiny.helpers.all_int(data))`, x))

tinyTest('colored', [['test', 'red'], ['test', 'green']], helpers.colored, (...data) => runPython(`out(tiny.helpers.colored(*data))`, data))

tinyTest('colorizeFloat', [[0.5], [1.0], [1.2]], helpers.colorizeFloat, (x) => runPython(`out(tiny.helpers.colorize_float(data))`, x))

tinyTest('ansistrip', [[' sdf sdf sdf sdfasdf'], ['test']], helpers.ansistrip, (x) => runPython(`out(tiny.helpers.ansistrip(data))`, x))

tinyTest('ansilen', [['asdf asdf asdf asdf '], ['test']], helpers.ansilen, (x) => runPython(`out(tiny.helpers.ansilen(data))`, x))

tinyTest('makeTuple', [[[1, 2, 3], 3], [[1, 2, 3], 2], [[1, 2, 3], 1], [1, 3], [1, 2], [1, 1]], helpers.makeTuple, (...data) => runPython(`out(tiny.helpers.make_tuple(*data))`, data))

tinyTest('flatten', [[[[1, 2], [3, 4]]], [[[1, 2, 3, 4]]], [[[1, 2, 3, [4, 5], 6]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]] as any, helpers.flatten, (x) => runPython(`out(tiny.helpers.flatten(data))`, x))

tinyTest('fully_flatten', [[[1, 2, 3]], [[1, 2, 3, 4, 5]], [[[1, 2], [3, 4], [5]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]], helpers.fullyFlatten, (x) => runPython(`out(tiny.helpers.fully_flatten(data))`, x))

// TODO: fromimport

tinyTest('stripParens', [['sdfsdf'], ['(sdfsdf'], ['(sdfsdf)']], helpers.stripParens, (x) => runPython(`out(tiny.helpers.strip_parens(data))`, x))

tinyTest('ceildiv', [[180, 3], [-10, 3], [10.5, 3], [-10.5, 3]], helpers.ceildiv, (...data) => runPython(`out(tiny.helpers.ceildiv(*data))`, data))

tinyTest('roundUp', [[180, 3], [10, 3], [11, 3], [12, 3]], helpers.roundUp, (...data) => runPython(`out(tiny.helpers.round_up(*data))`, data))

tinyTest('data64', [[333], [45443], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64, (x) => runPython(`out(tiny.helpers.data64(data))`, x))

tinyTest('data64_le', [[333], [3434], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64Le, (x) => runPython(`out(tiny.helpers.data64_le(data))`, x))

tinyTest('mergeDicts', [[[{ a: 1 }, { b: 2 }], [{ x: 1, y: 2 }, { z: 3 }]]] as any, helpers.mergeDicts, (x) => runPython(`out(tiny.helpers.merge_dicts(data))`, x))

tinyTest('partition', [[[1, 2, 3, 4, 5], (x: number) => x % 2 === 0]], helpers.partition, (arr) => runPython(`out(tiny.helpers.partition(data, lambda x: x % 2 == 0))`, arr))
tinyTest('partition', [[[2, 4, 6, 8, 10], (x: number) => x > 5]], helpers.partition, (arr) => runPython(`out(tiny.helpers.partition(data, lambda x: x > 5))`, arr))
tinyTest('partition', [[['a', 'b', 'c', 'd'], (x: string) => x > 'b']], helpers.partition, (arr) => runPython(`out(tiny.helpers.partition(data, lambda x: x > "b"))`, arr))

tinyTest('unwrap', [[1], [2], ['sdf']], helpers.unwrap, (x) => runPython(`out(tiny.helpers.unwrap(data))`,x))

tinyTest('get_child', [[{ a: 1, b: { c: 2 } }, 'b.c'], [{ a: { x: [33, 54] }, b: { c: 2 } }, 'a.x.0'], [[3, { a: { v: 'true' } }], '1.a.v']], helpers.getChild, (...data) => runPython(`out(tiny.helpers.get_child(*data))`, data))

tinyTest('word_wrap', [['This is a long string that needs to be wrapped to fit within 80 characters. Sfasdf dsafg sdf sdf sdf sdf sdf s dfs df']], helpers.wordWrap, (x) => runPython(`out(tiny.helpers.word_wrap(data))`, x))

tinyTest('polyN', [[2, [1, 2, 3]], [2, [1, 2, 3]]], helpers.polyN, (...data) => runPython(`out(tiny.helpers.polyN(*data))`, data))

tinyTest('to_function_name', [['test'], ['not sure how this should work'], ['letsTryThisOne']], helpers.toFunctionName, (s) => runPython(`out(tiny.helpers.to_function_name(data))`, s))

tinyTest('getenv', [['key', 'value']], helpers.getEnv, (...data) => runPython(`out(tiny.helpers.getenv(*data))`, data))

tinyTest('temp', [['file.txt']], helpers.temp, (x) => runPython(`out(tiny.helpers.temp(data))`, x))
