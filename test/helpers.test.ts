import * as helpers from '../denograd/helpers.ts'
import { compare, test } from './helpers.ts'

test('argsort', compare([[[1, 2, 3]], [[1, 4, 2]]], helpers.argsort, 'out(tiny.helpers.argsort(*data))'))

test('memsizeToStr', compare([[123], [100_000], [2_450_000]], helpers.memsize_to_str, 'out(tiny.helpers.memsize_to_str(*data))'))

test('dedup', compare([[[1, 2, 3, 2, 1]], [[1, 2, 3]]], helpers.dedup, 'out(tiny.helpers.dedup(*data))'))

test('allSame', compare([[[1, 1, 1]], [[1, 2, 3]]], helpers.all_same, 'out(tiny.helpers.all_same(*data))'))

test('allInt', compare([[[1, 2, 3]], [[1, 2.5, 3]]], helpers.all_int, 'out(tiny.helpers.all_int(*data))'))

test('colored', compare([['test', 'red'], ['test', 'green']], helpers.colored, 'out(tiny.helpers.colored(*data))'))

test('colorizeFloat', compare([[0.5], [1.0], [1.2]], helpers.colorize_float, 'out(tiny.helpers.colorize_float(*data))'))

test('ansistrip', compare([[' sdf sdf sdf sdfasdf'], ['test']], helpers.ansistrip, 'out(tiny.helpers.ansistrip(*data))'))

test('ansilen', compare([['asdf asdf asdf asdf '], ['test']], helpers.ansilen, 'out(tiny.helpers.ansilen(*data))'))

test('makeTuple', compare([[[1, 2, 3], 3], [[1, 2, 3], 2], [[1, 2, 3], 1], [1, 3], [1, 2], [1, 1]], helpers.make_tuple, 'out(tiny.helpers.make_tuple(*data))'))

test('flatten', compare([[[[1, 2], [3, 4]]], [[[1, 2, 3, 4]]], [[[1, 2, 3, [4, 5], 6]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]] as any, helpers.flatten, 'out(tiny.helpers.flatten(*data))'))

test('fully_flatten', compare([[[1, 2, 3]], [[1, 2, 3, 4, 5]], [[[1, 2], [3, 4], [5]]], [[[1, 2, 3, 4, 5, 6, 7, 8]]]], helpers.fully_flatten, 'out(tiny.helpers.fully_flatten(*data))'))

test('stripParens', compare([['sdfsdf'], ['(sdfsdf'], ['(sdfsdf)']], helpers.strip_parens, 'out(tiny.helpers.strip_parens(*data))'))

test('roundUp', compare([[180, 3], [10, 3], [11, 3], [12, 3]], helpers.round_up, 'out(tiny.helpers.round_up(*data))'))

test('data64', compare([[333], [45443], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64, 'out(tiny.helpers.data64(*data))'))

test('data64_le', compare([[333], [3434], [0], [-1], [Number.MAX_SAFE_INTEGER], [Number.MIN_SAFE_INTEGER]], helpers.data64Le, 'out(tiny.helpers.data64_le(*data))'))

test('mergeDicts', compare([[[{ a: 1 }, { b: 2 }], [{ x: 1, y: 2 }, { z: 3 }]]] as any, helpers.merge_dicts, 'out(tiny.helpers.merge_dicts(data[0]))'))

test('partition', compare([[[1, 2, 3, 4, 5], (x: number) => x % 2 === 0]], helpers.partition, 'out(tiny.helpers.partition(data[0], lambda x: x % 2 == 0))'))
test('partition', compare([[[2, 4, 6, 8, 10], (x: number) => x > 5]], helpers.partition, 'out(tiny.helpers.partition(data[0], lambda x: x > 5))'))
test('partition', compare([[['a', 'b', 'c', 'd'], (x: string) => x > 'b']], helpers.partition, "out(tiny.helpers.partition(data[0], lambda x: x > 'b'))"))

test('unwrap', compare([[1], [2], ['sdf']], helpers.unwrap, 'out(tiny.helpers.unwrap(*data))'))

test(
  'get_child',
  compare(
    [
      [{ a: 1, b: { c: 2 } }, 'b.c'],
      [{ a: { x: [33, 54] }, b: { c: 2 } }, 'a.x.0'],
      [[3, { a: { v: 'true' } }], '1.a.v'],
    ],
    helpers.getChild,
    'out(tiny.helpers.get_child(*data))',
  ),
)

test('word_wrap', compare([['This is a long string that needs to be wrapped to fit within 80 characters. Sfasdf dsafg sdf sdf sdf sdf sdf s dfs df']], helpers.word_wrap, 'out(tiny.helpers.word_wrap(*data))'))

test('to_function_name', compare([['test'], ['not sure how this should work'], ['letsTryThisOne']], helpers.to_function_name, `out(tiny.helpers.to_function_name(*data))`))

test('getenv', compare([['key', 'value']], helpers.get_env, 'out(tiny.helpers.getenv(*data))'))

test(
  'slice',
  compare(
    [
      [[1, 2, 3, 4], -2, undefined, -2],
      [[1, 2, 3, 4, 5], 1, 4, 2],
      [[1, 2, 3, 4, 5], undefined, undefined, -1],
      [[1, 2, 3, 4, 5], 0, 5, 1],
      [[1, 2, 3, 4, 5], -3, -1, 1],
      [[1, 2, 3, 4, 5], 0, -2, 2],
    ],
    (arr: number[], start?: number, stop?: number, step?: number) => helpers.slice(arr, { start, stop, step }),
    'out(data[0][data[1]:data[2]:data[3]])',
  ),
)
