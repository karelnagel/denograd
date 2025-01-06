import { Conv2d } from '../../src/nn/index.ts'
import { Tensor } from '../../src/tensor.ts'
import { compare } from '../helpers.ts'

Deno.test(
  'Conv2d.init',
  compare(
    [
      [1, 1, 3], // basic case
      [3, 2, 3], // different in/out channels
      [1, 1, [3, 3]], // kernel size as array
      [2, 4, 5, 2], // with stride
      [1, 1, 3, 1, 1], // with padding
      [1, 1, 3, 1, [1, 1]], // padding as array
      [1, 1, 3, 1, 'same'], // padding as string
      [2, 2, 3, 1, 0, 2], // with dilation
      [4, 2, 3, 1, 0, 1, 2], // with groups
      [1, 1, 3, 1, 0, 1, 1, false], // without bias
    ],
    (in_channels: number, out_channels: number, kernel_size: number | number[], stride?: number, padding?: number | number[] | string, dilation?: number, groups?: number, bias?: boolean) => {
      Tensor.manual_seed(3)
      const conv = new Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
      return [conv.bias, conv.weight]
    },
    [
      'from tinygrad.nn import Conv2d',
      'tiny.Tensor.manual_seed(3)',
      'conv = Conv2d(*data)',
      'out([conv.bias, conv.weight])',
    ],
  ),
)

Deno.test(
  'Conv2d.call',
  compare(
    [
      [1, 1, 3, [1, 1, 4, 7]],
      [3, 2, 3, [2, 3, 3, 3]],
    ],
    (in_channels: number, out_channels: number, kernel_size: number | number[], rand_shape: number[]) => {
      Tensor.manual_seed(3)
      const conv = new Conv2d(in_channels, out_channels, kernel_size)
      const t = Tensor.rand(rand_shape)
      return conv.call(t)
    },
    [
      'from tinygrad.nn import Conv2d',
      'tiny.Tensor.manual_seed(3)',
      'conv = Conv2d(*data[:3])',
      't = tiny.Tensor.rand(*data[3])',
      'out(conv(t))',
    ],
  ),
)
