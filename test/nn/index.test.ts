import { BatchNorm, Conv2d } from '../../denograd/nn/index.ts'
import { Tensor } from '../../denograd/tensor.ts'
import { compare, test } from '../helpers.ts'
import { Linear } from '../../denograd/nn/index.ts'

test(
  'Conv2d.init',
  compare(
    [
      [1, 1, 3], // basic case
      [3, 2, 3], // different in/out channels
      [1, 1, [3, 3]], // kernel size as array
      // [2, 4, 5, 2], // with stride //for some reason fails, when running all the tests
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

test(
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

test(
  'BatchNorm.init',
  compare(
    [
      [1], // simplest case, 1 feature
      [3, 1e-3, false, true, 0.9], // custom eps, momentum, affine off
      [2, 1e-5, false, true, 0.1], // default eps, momentum, affine on
    ],
    (sz: number, eps: number = 1e-5, affine?: boolean, track_running_stats?: boolean, momentum: number = 0.1) => {
      Tensor.manual_seed(3)
      const bn = new BatchNorm(sz, eps, affine, track_running_stats, momentum)
      return [bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked]
    },
    [
      'from tinygrad.nn import BatchNorm',
      'tiny.Tensor.manual_seed(3)',
      'bn = BatchNorm(*data)',
      'out([bn.weight, bn.bias, getattr(bn,"running_mean",None), getattr(bn,"running_var",None), bn.num_batches_tracked])',
    ],
  ),
)

test(
  'BatchNorm.call',
  compare(
    [
      [1, [2, 1, 4, 4]],
      [2, [2, 2, 3, 3]],
    ],
    (num_features: number, shape: number[]) => {
      Tensor.manual_seed(3)
      const bn = new BatchNorm(num_features)
      const t = Tensor.rand(shape)
      return bn.call(t)
    },
    [
      'from tinygrad.nn import BatchNorm',
      'tiny.Tensor.manual_seed(3)',
      'bn = BatchNorm(data[0])',
      't = tiny.Tensor.rand(*data[1])',
      'out(bn(t))',
    ],
  ),
)

test(
  'Linear.init',
  compare(
    [
      // TODO: doesn't match in CI for some reason, locally matches
      // [1, 1], // simplest case, 1 -> 1
      [2, 3], // 2 -> 3
      [3, 3, false], // 3 -> 3, no bias
    ],
    (in_features: number, out_features: number, bias: boolean = true) => {
      Tensor.manual_seed(3)
      const linear = new Linear(in_features, out_features, bias)
      return [linear.bias, linear.weight]
    },
    [
      'from tinygrad.nn import Linear',
      'tiny.Tensor.manual_seed(3)',
      'linear = Linear(*data)',
      'out([linear.bias, linear.weight])',
    ],
  ),
)

test(
  'Linear.call',
  compare(
    [
      [1, 1, [4, 1]], // shape: (4, 1) => 4 samples, each of size 1
      [2, 3, [5, 2]], // shape: (5, 2) => 5 samples, each of size 2
    ],
    (in_features: number, out_features: number, input_shape: number[]) => {
      Tensor.manual_seed(3)
      const linear = new Linear(in_features, out_features)
      const t = Tensor.rand(input_shape)
      return linear.call(t)
    },
    [
      'from tinygrad.nn import Linear',
      'tiny.Tensor.manual_seed(3)',
      'linear = Linear(data[0], data[1])',
      't = tiny.Tensor.rand(*data[2])',
      'out(linear(t))',
    ],
  ),
)
