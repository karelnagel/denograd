import { Conv2d } from '../../src/nn/index.ts'
import { Tensor } from '../../src/tensor.ts'
import { compare } from '../helpers.ts'

Deno.test(
  'Conv2d.init',
  compare(
    [
      [],
    ],
    () => {
      Tensor.manual_seed(3)
      const conv = new Conv2d(1, 1, 3)
      return [conv.bias, conv.weight]
    },
    [
      'from tinygrad.nn import Conv2d',
      'tiny.Tensor.manual_seed(3)',
      'conv = Conv2d(1,1,3)',
      'out([conv.bias, conv.weight])',
    ],
  ),
)
