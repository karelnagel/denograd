import { Tensor } from '../src/tensor.ts'
import { compare, tryCatch } from './helpers.ts'
import process from 'node:process'

process.env.PYTHON = '1'

Deno.test(
  'Tensor.numel',
  compare(
    [
      [4, 4, 4, 2, 6.5],
      [4],
      [true, false],
    ],
    (...data: (number | boolean)[]) => new Tensor(data).numel(),
    'out(tiny.Tensor(data).numel())',
  ),
)
Deno.test(
  'Tensor.item',
  compare(
    [
      [4],
      [4, 4, 3, 66],
    ],
    (...data: (number | boolean)[]) => new Tensor(data).item(),
    'out(tiny.Tensor(data).item())',
  ),
)

Deno.test(
  'Tensor._data',
  compare(
    [
      [2, 4],
    ],
    (...data: (number | boolean)[]) => new Tensor(data)._data(),
    'out(tiny.Tensor(data)._data())',
  ),
)

Deno.test(
  'Tensor.tolist',
  compare(
    [
      [4, 4, 4, 2, 6.5],
      [4],
      [4, 5],
      [true, false],
    ],
    (...data: (number | boolean)[]) => new Tensor(data).tolist(),
    'out(tiny.Tensor(data).tolist())',
  ),
)

Deno.test(
  'Tensor.add',
  compare(
    [
      [[4, 4, 4, 2, 6.5], [4, 4, 3, 3, 3]],
    ],
    (data0: number[], data1: number[]) => new Tensor(data0).add(new Tensor(data1)).tolist(),
    'out((tiny.Tensor(data[0]) + tiny.Tensor(data[1])).tolist())',
  ),
)

Deno.test(
  'Tensor.mul',
  compare(
    [
      [[4, 4, 4, 2, 6.5], [4, 4, 3, 3, 3]],
    ],
    (data0: number[], data1: number[]) => new Tensor(data0).mul(new Tensor(data1)).tolist(),
    'out((tiny.Tensor(data[0]) * tiny.Tensor(data[1])).tolist())',
  ),
)
Deno.test(
  'Tensor.matmul',
  compare(
    [
      [[4, 4, 4, 2, 6.5], [4, 4, 3, 3, 3]],
    ],
    (data0: number[], data1: number[]) => new Tensor(data0).matmul(new Tensor(data1)).tolist(),
    'out((tiny.Tensor(data[0]) @ tiny.Tensor(data[1])).tolist())',
  ),
)

Deno.test(
  'Tensor.get',
  compare(
    [
      [4, 4, 4, 2, 6.5],
      [4],
      [true, false],
    ],
    (...data: (number | boolean)[]) => new Tensor(data).get({ start: 1 }).numel(),
    'out(tiny.Tensor(data)[1:].numel())',
  ),
)
