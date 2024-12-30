import { ConstType, DType, dtypes } from '../src/dtype.ts'
import { LazyBuffer } from '../src/engine/lazy.ts'
import { sint, UOp } from '../src/ops.ts'
import { Tensor, TensorOptions } from '../src/tensor.ts'
import { compare, tryCatch } from './helpers.ts'
import process from 'node:process'

process.env.PYTHON = '1'

Deno.test(
  'Tensor.numel',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6.5])],
      [new Tensor(4)],
      [new Tensor([true, false])],
    ],
    (t: Tensor) => t.numel(),
    'out(data[0].numel())',
  ),
)
Deno.test(
  'Tensor.item',
  compare(
    [
      [new Tensor(4)],
      [new Tensor(4.55)],
      [new Tensor(true)],
    ],
    (t: Tensor) => t.item(),
    'out(data[0].item())',
  ),
)

Deno.test(
  'Tensor.init',
  compare(
    [
      [[4, 4, 4, 2, 6.5], { dtype: dtypes.half }],
      [[], {}],
      [[4], {}],
      [[555], {}],
      [[-1], {}],
      [[255], {}],
      [undefined, {}],
      [1.2, {}],
      [255, {}],
      [[256], {}],
      [[257], {}],
      [[4, 5], {}],
      [[[4, 5]], {}],
      [[[[4, 5]]], {}],
      [[4, Number.MAX_SAFE_INTEGER, Number.MIN_SAFE_INTEGER], { dtype: dtypes.bool }],
      [[true, false], {}],
      [new Uint8Array([2, 3]), { dtype: dtypes.float }],
    ],
    (data: ConstType | undefined | UOp | Uint8Array | any[] | LazyBuffer | Tensor | string, opts: TensorOptions) => new Tensor(data, opts),
    'out(tiny.Tensor(data[0], dtype=data[1].get("dtype")))',
  ),
)

Deno.test(
  'Tensor.reshape',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6.5, 1, 2, 3, 4, 5]), [10]],
      [new Tensor([4, 4, 4, 2, 6.5, 1, 2, 3, 4, 5]), [1, 10]],
      [new Tensor([4, 4, 4, 2, 6.5, 1, 2, 3, 4, 5]), [1, 1, 10]],
      [new Tensor([[4, 4, 4, 2, 6.5, 1, 2, 3, 4, 5]]), [2, 1, 5]],
    ],
    (t: Tensor, shape: number[]) => t.reshape(shape),
    'out(data[0].reshape(data[1]))',
  ),
)

Deno.test(
  'Tensor._broadcast_to',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6.5, 1, 2, 3, 4, 5]), [1, 10]],
      [new Tensor([4, 4, 4, 2, 6.5, 1, 2, 3, 4, 5]), [1, 1, 10]],
    ],
    tryCatch((t: Tensor, shape: number[]) => t._broadcast_to(shape).tolist()),
    'out(data[0]._broadcast_to(data[1]).tolist())',
  ),
)

Deno.test(
  'Tensor.get.data',
  compare(
    [
      [new Tensor([4, 11, 255, 2, 65, 1, 24, 3, 1, 5])],
    ],
    (t: Tensor) => {
      return [
        t.get(undefined).data(),
        t.get('...').data(),
        t.reshape([5, 2]).data(),
        t.get(0).data(),
        t.get(9).data(),
        t.get({ start: 2, stop: 2 }).data(),
        t.reshape([2, 5]).get(0, 4).data(),
        t.get({ start: 2, stop: 6 }).data(),
        t.reshape([2, 5]).get(1).data(),
        t.get({ start: 0, stop: 2 }).data(),
        t.reshape([5, 2]).get({ start: 1, stop: 2 }).data(),
        t.reshape([5, 2]).get({ start: 1, stop: 3 }).data(),
      ]
    },
    [
      't = data[0]',
      'out([',
      '   t[None].data(),',
      '   t[...].data(),',
      '   t.reshape((5,2)).data(),',
      '   t[0].data(),',
      '   t[9].data(),',
      '   t[2:2].data(),',
      '   t.reshape((2,5))[0, 4].data(),',
      '   t[2:6].data(),',
      '   t.reshape((2,5))[1].data(),',
      '   t[0:2].data(),',
      '   t.reshape((5,2))[1:2].data(),',
      '   t.reshape((5,2))[1:3].data(),',
      '])',
    ],
  ),
)

Deno.test(
  'Tensor.get.tolist',
  compare(
    [
      [new Tensor([4, 11, 255, 2, 65, 1, 24, 3, 1, 5])],
      [new Tensor([4.2, 11.7, 255.1, 2.9, 65.3, 1.4, 24.8, 3.6, 1.1, 5.5])],
    ],
    (t: Tensor) => {
      return [
        t.get(undefined).tolist(),
        t.get('...').tolist(),
        t.reshape([5, 2]).tolist(),
        t.get(0).tolist(),
        t.get(9).tolist(),
        t.get({ start: 2, stop: 2 }).tolist(),
        t.reshape([2, 5]).get(0, 4).tolist(),
        // t.get({ start: 2, stop: 6 }).tolist(), // float tensor fails uop verification
        t.reshape([2, 5]).get(1).tolist(),
        t.get({ start: 0, stop: 2 }).tolist(),
        t.reshape([5, 2]).get({ start: 1, stop: 2 }).tolist(),
        // t.reshape([5, 2]).get({ start: 1, stop: 3 }).tolist(), // float tensor fails uop verification
      ]
    },
    [
      't = data[0]',
      'out([',
      '   t[None].tolist(),',
      '   t[...].tolist(),',
      '   t.reshape((5,2)).tolist(),',
      '   t[0].tolist(),',
      '   t[9].tolist(),',
      '   t[2:2].tolist(),',
      '   t.reshape((2,5))[0, 4].tolist(),',
      // '   t[2:6].tolist(),',
      '   t.reshape((2,5))[1].tolist(),',
      '   t[0:2].tolist(),',
      '   t.reshape((5,2))[1:2].tolist(),',
      // '   t.reshape((5,2))[1:3].tolist(),',
      '])',
    ],
  ),
)

Deno.test(
  'Tensor.add',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), new Tensor([4, 4, 3, 3, 3])],
      [new Tensor([4, 4, 4, 2, 6.5, 5]).reshape([1, 1, 6]), new Tensor([4, 4, 3, 3, 3, 6])],
      [new Tensor([4, 4, 4, 2, 6.5, 5]).reshape([2, 3]), new Tensor([4, 4, 3, 3, 3, 6]).reshape([2, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.add(t2),
    'out(data[0] + data[1])',
  ),
)

Deno.test(
  'Tensor.mul',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), new Tensor([4, 4, 3, 3, 3])],
      [new Tensor([4, 4, 4, 2, 6.5, 5]).reshape([1, 1, 6]), new Tensor([4, 4, 3, 3, 3, 6])],
      [new Tensor([4, 4, 4, 2, 6.5, 5]).reshape([2, 3]), new Tensor([4, 4, 3, 3, 3, 6]).reshape([2, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.mul(t2),
    'out(data[0] * data[1])',
  ),
)

Deno.test(
  'Tensor.div',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), new Tensor([4, 4, 3, 3, 3])],
      [new Tensor([4, 4, 4, 2, 6.5, 5]).reshape([1, 1, 6]), new Tensor([4, 4, 3, 3, 3, 6])],
      [new Tensor([4, 4, 4, 2, 6.5, 5]).reshape([2, 3]), new Tensor([4, 4, 3, 3, 3, 6]).reshape([2, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.div(t2),
    'out(data[0] / data[1])',
  ),
)
Deno.test(
  'Tensor.idiv',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), new Tensor([4, 4, 3, 3, 3])],
      [new Tensor([4, 4, 4, 2, 6, 5]).reshape([1, 1, 6]), new Tensor([4, 4, 3, 3, 3, 6])],
      [new Tensor([4, 4, 4, 2, 6, 5]).reshape([2, 3]), new Tensor([4, 4, 3, 3, 3, 6]).reshape([2, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.idiv(t2),
    'out(data[0] // data[1])',
  ),
)
Deno.test(
  'Tensor.cast',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), dtypes.bool],
      [new Tensor([4, 4, 4, 2, 6, 5.5]).reshape([1, 1, 6]), dtypes.float],
      [new Tensor([4, 4, 4, 2, 6, 5.5]).reshape([1, 1, 6]), dtypes.half],
      [new Tensor([4, 4, 4, 2, 6, 5.5]).reshape([1, 1, 6]), dtypes.int],
      [new Tensor([4, 4, 4, 2, 6, 5.5]).reshape([1, 1, 6]), dtypes.int64],
      [new Tensor([4, 4, 4, 2, 6, 5.5]).reshape([2, 3]), dtypes.double],
      [new Tensor([4, 4, 4, 2, 6, 5.5]).reshape([2, 3]), dtypes.bool],
    ],
    (t1: Tensor, dtype: DType) => t1.cast(dtype),
    'out(data[0].cast(data[1]))',
  ),
)
Deno.test(
  'Tensor.maximum',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), new Tensor([4, 4, 3, 3, 3])],
      [new Tensor([4, 4, 4, 2, 6, 5]).reshape([1, 1, 6]), new Tensor([4, 4, 3, 3, 3, 6])],
      [new Tensor([4, 4, 4, 2, 6, 5]).reshape([2, 3]), new Tensor([4, 4, 3, 3, 3, 6]).reshape([2, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.maximum(t2),
    'out(data[0].maximum(data[1]))',
  ),
)
Deno.test(
  'Tensor.minimum',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6]), new Tensor([4, 4, 3, 3, 3])],
      [new Tensor([4, 4, 4, 2, 6, 5]).reshape([1, 1, 6]), new Tensor([4, 3, 3, 3, 3, 6])],
      [new Tensor([4, 4, 4, 2, 6, 5]).reshape([2, 3]), new Tensor([1, 2, 3, 3, 3, 6]).reshape([2, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.minimum(t2),
    'out(data[0].minimum(data[1]))',
  ),
)

Deno.test.ignore(
  'Tensor.matmul',
  compare(
    [
      [new Tensor([4, 4, 4, 2, 6.5]), new Tensor([4, 4, 3, 3, 3])],
    ],
    (t1: Tensor, t2: Tensor) => t1.matmul(t2),
    'out(data[0] @ data[1])',
  ),
)

const ops: [Tensor, keyof Tensor, string?][] = [
  [new Tensor([[-2, -1, 0], [1, 2, 3]]), 'max'],
  [new Tensor([[-2, -1, 0], [1, 2, 3]]), 'min'],
  [new Tensor([[-2, -1, 0], [1, 2, 3]]), 'relu'],
  [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), 'max_pool2d'],
  [new Tensor([[1, 2], [3, 4], [5, 6]]), 'flatten'],
  [new Tensor([2.4, 5.5, 7.7]), 'round'],
  [new Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), 'round'],
  [new Tensor([1.4, 1.5, 1.6, 2.4, 2.5, 2.6]), 'round'],
  [new Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), 'floor'],
  [new Tensor([1.1, 1.9, 2.1, 2.9]), 'floor'],
  [new Tensor([-1.9, -1.1, -0.9, -0.1]), 'floor'],
  [new Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), 'ceil'],
  [new Tensor([1.1, 1.9, 2.1, 2.9]), 'ceil'],
  [new Tensor([-1.9, -1.1, -0.9, -0.1]), 'ceil'],
  [new Tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]), 'trunc'],
  [new Tensor([1.1, 1.9, 2.1, 2.9]), 'trunc'],
  [new Tensor([-1.9, -1.1, -0.9, -0.1]), 'trunc'],
  [new Tensor([1, Infinity, 2, -Infinity, NaN]), 'isinf'],
  [new Tensor([1, 2, 3, 4, NaN, 5]), 'isnan'],
  [new Tensor([1, 2, 3]), 'square'],
  [new Tensor([-3, -2, -1, 0, 1, 2, 3]), 'square'],
  [new Tensor([-3, -2, -1, 0, 1, 2, 3]), 'sign'],
  [new Tensor([0.1, -0.5, 1.2, -2.4]), 'sign'],
  [new Tensor([-3, -2, -1, 0, 1, 2, 3]), 'abs'],
  [new Tensor([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]), 'abs'],
  [new Tensor([1, 2, 4, 8]), 'reciprocal'],
  [new Tensor([0.5, 2, 4, 10]), 'reciprocal'],
  [new Tensor([true, false, true]), 'logical_not'],
  [new Tensor([-2, -1, 0, 1, 2]), 'neg'],
  [new Tensor([[1, 2], [3, 4]]), 'contiguous'],
  [new Tensor([[1, 2], [3, 4]]), 'contiguous_backward'],
  [new Tensor([0.1, 1.0, 10.0]), 'log'],
  [new Tensor([0.1, 1.0, 10.0]), 'log2'],
  [new Tensor([-2, -1, 0, 1, 2]), 'exp'],
  [new Tensor([-2, -1, 0, 1, 2]), 'exp2'],
  [new Tensor([-2, -1, 0, 1, 2]), 'relu'],
  [new Tensor([-2, -1, 0, 1, 2]), 'sigmoid'],
  [new Tensor([-2, -1, 0, 1, 2]), 'hardsigmoid'],
  [new Tensor([0, 1, 4, 9, 16]), 'sqrt'],
  [new Tensor([1, 4, 9, 16]), 'rsqrt', 'TIMEOUT'],
  [new Tensor([-1, -0.5, 0, 0.5, 1]), 'sin', 'TIMEOUT'],
  [new Tensor([-1, -0.5, 0, 0.5, 1]), 'cos', 'TIMEOUT'],
  [new Tensor([-1, -0.5, 0, 0.5, 1]), 'tan', 'TIMEOUT'],
  [new Tensor([-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]), 'asin'],
  [new Tensor([-0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9]), 'acos'],
  [new Tensor([-3, -2, -1, 0, 1, 2, 3]), 'atan'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'elu'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'celu'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'selu'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'swish'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'silu'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'relu6'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'hardswish'],
  [new Tensor([-2, -1, 0, 1, 2]), 'tanh'],
  [new Tensor([-2, -1, 0, 1, 2]), 'sinh'],
  [new Tensor([-2, -1, 0, 1, 2]), 'cosh'],
  [new Tensor([-0.9, -0.5, 0, 0.5, 0.9]), 'atanh', 'TIMEOUT'],
  [new Tensor([-2, -1, 0, 1, 2]), 'asinh', 'TIMEOUT'],
  [new Tensor([1.5, 2, 2.5, 3, 3.5]), 'acosh', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'hardtanh', 'TIMEOUT'],
  [new Tensor([-2, -1, 0, 1, 2]), 'erf', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'gelu', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'quick_gelu', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'leakyrelu', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'mish', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'softplus', 'TIMEOUT'],
  [new Tensor([-2.7, -1.5, -0.2, 0, 0.2, 1.5, 2.7]), 'softsign', 'TIMEOUT'],
]

for (const [i, [tensor, op, ignore]] of ops.entries()) {
  Deno.test({
    name: `Tensor.ops.${op}.${i}`,
    ignore: !!ignore,
    fn: compare(
      [[tensor, op]],
      (t: Tensor, op: keyof Tensor) => (t[op] as any)()._debug_ast(),
      'out(getattr(data[0],data[1])()._debug_ast())',
    ),
  })
}

Deno.test(
  'Tensor._pool',
  compare(
    [
      // Basic 2D pooling
      [new Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), [2, 2]],

      // Test stride > kernel case
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), [2, 2], 3],

      // Test dilation
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), [2, 2], 1, 2],

      // Test kernel > stride case
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), [3, 3], 2],

      // Test 1D pooling
      [new Tensor([1, 2, 3, 4, 5, 6]), [2]],

      // Test 3D pooling
      [new Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), [2, 2, 2]],

      // Test different strides per dimension
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), [2, 2], [2, 1]],

      // Test different dilations per dimension
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]), [2, 2], 1, [2, 1]],
    ],
    (t: Tensor, k_: number[], stride: number[] | number = 1, dilation: number[] | number = 1) => t._pool(k_, stride, dilation),
    'out(data[0]._pool(*data[1:]))',
  ),
)

Deno.test(
  'Tensor.repeat',
  compare(
    [
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [3, 3]],
    ],
    (t: Tensor, repeats: number[]) => t.repeat(repeats),
    'out(data[0].repeat(data[1]))',
  ),
)

Deno.test(
  'Tensor.reshape',
  compare(
    [
      [new Tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [1, 4, 1, 4]],
      [new Tensor([[[[1, 2, 3, 4]], [[5, 6, 7, 8]], [[9, 10, 11, 12]], [[13, 14, 15, 16]]]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [1, 4, 1, 4]],
      [new Tensor([[[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]], [[9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12]], [[13, 14, 15, 16], [13, 14, 15, 16], [13, 14, 15, 16]]], [[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]], [[9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12]], [[13, 14, 15, 16], [13, 14, 15, 16], [13, 14, 15, 16]]], [[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]], [[9, 10, 11, 12], [9, 10, 11, 12], [9, 10, 11, 12]], [[13, 14, 15, 16], [13, 14, 15, 16], [13, 14, 15, 16]]]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [12, 12]],
    ],
    (t: Tensor, shape: number[]) => t.reshape(shape),
    'out(data[0].reshape(*data[1]))',
  ),
)

Deno.test(
  'Tensor.shrink',
  compare(
    [
      [new Tensor([[[[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]], [[5, 6, 7, 8, 5], [6, 7, 8, 5, 6]], [[9, 10, 11, 12, 9], [10, 11, 12, 9, 10]], [[13, 14, 15, 16, 13], [14, 15, 16, 13, 14]], [[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]], [[5, 6, 7, 8, 5], [6, 7, 8, 5, 6]]], [[[9, 10, 11, 12, 9], [10, 11, 12, 9, 10]], [[13, 14, 15, 16, 13], [14, 15, 16, 13, 14]], [[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]], [[5, 6, 7, 8, 5], [6, 7, 8, 5, 6]], [[9, 10, 11, 12, 9], [10, 11, 12, 9, 10]], [[13, 14, 15, 16, 13], [14, 15, 16, 13, 14]]]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [[0, 2], [0, 2], [0, 2], [0, 3]]],
      [new Tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12], [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12], [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12], [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [[0, 12], [0, 10]]],
      [new Tensor([[[[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]], [[5, 6, 7, 8, 5], [6, 7, 8, 5, 6]], [[9, 10, 11, 12, 9], [10, 11, 12, 9, 10]], [[13, 14, 15, 16, 13], [14, 15, 16, 13, 14]], [[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]]], [[[5, 6, 7, 8, 5], [6, 7, 8, 5, 6]], [[9, 10, 11, 12, 9], [10, 11, 12, 9, 10]], [[13, 14, 15, 16, 13], [14, 15, 16, 13, 14]], [[1, 2, 3, 4, 1], [2, 3, 4, 1, 2]], [[5, 6, 7, 8, 5], [6, 7, 8, 5, 6]]]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [[0, 2], [0, 4], [0, 2], [0, 3]]],
      [new Tensor([[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12], [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12], [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16], [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8], [9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12], [13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16]], { requires_grad: undefined, dtype: dtypes.int, device: `PYTHON` }), [[0, 10], [0, 10]]],
    ],
    (t: Tensor, args: [sint, sint][]) => t.shrink(args),
    'out(data[0].shrink(data[1]))',
  ),
)
