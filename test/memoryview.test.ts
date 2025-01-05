import { FmtStr } from '../src/dtype.ts'
import { _get_recursive_parents, bitcast } from '../src/dtype.ts'
import { MemoryView } from '../src/memoryview.ts'
import { compare } from './helpers.ts'

Deno.test(
  'dtype.bitcast',
  compare(
    [
      [[4, 4, 55, 23.34, 54], 'f', 'i'],
      [[4, 4, 55, 23, 54], 'i', 'f'],
      // [[4, 4, 55, 23, 54], 'i', 'B'],
      [[127, 0, 255], 'B', 'b'], // Test unsigned to signed byte conversion
      [[65535, 0, 32768], 'H', 'h'], // Test unsigned to signed short
      [[2147483647, 0, -2147483648], 'i', 'I'], // Test signed to unsigned int
      // [[true, false, true], '?', 'B'], // Test boolean to byte
    ],
    bitcast,
    [
      'import struct',
      `out(list(struct.unpack(f"{len(data[0])}{data[2]}", struct.pack(f"{len(data[0])}{data[1]}", *data[0]))))`,
    ],
  ),
)

Deno.test(
  'memoryview',
  compare(
    [
      [[2, 2, 5, 5, 3, 65, 76], 'B', 'B'],
      [[2, 2, 4, 0, 0], 'B', 'B'],
      [[1, 0, 1, 1, 1], 'B', '?'],
      [[2, 4, 4, 4], 'B', 'i'],
      [[2, 4, 4, 4], 'B', 'f'],
      [[2.3, 4.4, 4.4, 4.3], 'f', 'B'],
      [[2, 4, 4, 69], 'i', 'B'],
      [[2, 4, 4, 4], 'B', 'f'],
      [[2, 4, 4, 4], 'B', 'f'],
    ],
    (data: number[], from: FmtStr, to: FmtStr) => {
      return new MemoryView(data, { fmt: from }).cast(to).slice(1, -1).to1DList()
    },
    [
      'from array import array',
      'out(memoryview(array(data[1],data[0])).cast(data[2])[1:-1].tolist())',
    ],
  ),
)
Deno.test(
  'memoryview.shape',
  compare(
    [
      [[4, 4, 4, 6, 6, 6], [2, 3]],
      [[4, 4, 4, 6, 6, 6], [6]],
      [[4, 4, 4, 6, 6, 6], [1, 1, 1, 1, 1, 6]],
      [[4, 4, 4, 6, 6, 6], [1, 1, 1, 1, 2, 3]],
    ],
    (data: number[], shape: number[]) => {
      const view = new MemoryView(data, { fmt: 'B' }).cast('B', shape)
      return [view.shape, view.strides]
    },
    [
      'from array import array',
      'view = memoryview(array("B",data[0])).cast("B", data[1])',
      'out([view.shape,view.strides])',
    ],
  ),
)
