import { ConstType, dtypes } from '../src/dtype.ts'
import { LazyBuffer } from '../src/engine/lazy.ts'
import { UOp } from '../src/ops.ts'
import { Tensor, TensorOptions } from '../src/tensor.ts'
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
  'Tensor.tolist',
  compare(
    [
      [[4, 4, 4, 2, 6.5], { dtype: dtypes.half }],
      [[], {}],
      [[4], {}],
      [[555], {}],
      [[-1], {}],
      [[255], {}],
      [1.2, {}],
      [255, {}],
      [[256], {}],
      [[257], {}],
      [[4, 5], {}],
      [[4, Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER], {}],
      [[true, false], {}],
      [new Uint8Array([2, 3]), {}],
    ],
    tryCatch((data: ConstType | UOp | Uint8Array | any[] | LazyBuffer | Tensor | string, opts: TensorOptions) => {
      const t = new Tensor(data, opts)
      return [t.tolist(), t.dtype]
    }),
    [
      't = tiny.Tensor(data[0], dtype=data[1].get("dtype"))',
      'out([t.tolist(), t.dtype])',
    ],
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
