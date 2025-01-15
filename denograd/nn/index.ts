import { is_dtype_supported } from '../device.ts'
import { dtypes } from '../dtype.ts'
import { assert, flatten, is_eq, make_tuple, range, zip } from '../helpers.ts'
import { div, idiv, mul, prod, sub } from '../ops.ts'
import { Layer, Tensor } from '../tensor.ts'
import { get_state_dict } from './state.ts'
import { load_state_dict, safe_load, safe_save } from './state.ts'
export * as optim from './optim.ts'
export * as state from './state.ts'

/**
 * Abstract model to simplify calling, loading and saving different models.
 * You only need to implement list of layers to create a new model
 * ```ts
 * export class MNIST extends Model {
 *  layers: Layer[] = [
 *    new nn.Conv2d(1, 32, 5),
 *    Tensor.relu,
 *    // ...other layers
 *  ]
 * }
 * const mnist = new MNIST()
 *
 * // call the model with some input
 * mnist.call(Tensor.rand([1, 1, 28, 28]))
 *
 * // load weigths from ./mnist.safetensors
 * await mnist.load("./mnist.safetensors")
 *
 * //save weigths to ./mnist.safetensors
 * await mnist.save("./mnist.safetensors")
 * ```
 */
export abstract class Model {
  abstract layers: Layer[]
  /**
   * Call model with a Tensor input, returns a Tensor with output.
   * ```ts
   * const model = new Model()
   * const res = model.call(Tensor.rand([1, 1, 28, 28]))
   * console.log(await res.tolist())
   * ```
   */
  call = (x: Tensor) => x.sequential(this.layers)
  /**
   * Load model weights from a .safetensors file at the given path
   * ```ts
   * const model = new Model()
   * await model.load("./model.safetensors")
   * ```
   */
  load = async (path: string | Tensor) => await load_state_dict(this, await safe_load(path))
  /**
   * Save model weights to a .safetensors file at the given path
   * ```ts
   * const model = new Model()
   * await model.save("./model.safetensors")
   * ```
   */
  save = async (path: string) => await safe_save(get_state_dict(this), path)
}
/**
 * Applies Batch Normalization over a 2D || 3D input.
 *
 * - Described: https://paperswithcode.com/method/batch-normalization
 * - Paper: https://arxiv.org/abs/1502.03167v3
 *
 * See: `Tensor.batchnorm`
 *
 * ```python exec="true" session="tensor"
 * from tinygrad import Tensor, dtypes, nn
 * import numpy as np
 * np.set_printoptions(precision=4)
 * ```
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * norm = nn.BatchNorm(3)
 * t = Tensor.rand(2, 3, 4, 4)
 * console.log(t.mean().item(), t.std().item())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = norm(t)
 * console.log(t.mean().item(), t.std().item())
 * ```
 */
export class BatchNorm {
  eps: number
  track_running_stats: boolean
  momentum: number
  weight?: Tensor
  bias?: Tensor
  num_batches_tracked: Tensor
  running_mean?: Tensor
  running_var?: Tensor
  constructor(sz: number, eps = 1e-5, affine = true, track_running_stats = true, momentum = 0.1) {
    this.eps = eps, this.track_running_stats = track_running_stats, this.momentum = momentum

    this.weight = affine ? Tensor.ones([sz]) : undefined
    this.bias = affine ? Tensor.zeros([sz]) : undefined

    this.num_batches_tracked = Tensor.zeros([1], { dtype: is_dtype_supported(dtypes.long) ? dtypes.long : dtypes.int, requires_grad: false })
    if (track_running_stats) this.running_mean = Tensor.zeros([sz], { requires_grad: false }), this.running_var = Tensor.ones([sz], { requires_grad: false })
  }
  calc_stats = (x: Tensor): [Tensor, Tensor] => {
    const shape_mask: number[] = [1, -1, ...(range(x.ndim - 2).map((x) => 1))]
    if (this.track_running_stats && !Tensor.training) return [this.running_mean!, this.running_var!.reshape(shape_mask).expand(x.shape)]
    // This requires two full memory accesses to x
    // https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
    // There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance//Welford's_Online_algorithm
    const reduce_axes = range(x.ndim).filter((x) => x !== 1)
    const batch_mean = x.mean(reduce_axes)
    const y = x.sub(batch_mean.detach().reshape(shape_mask)) // d(var)/d(mean) = 0
    const batch_var = (y.mul(y)).mean(reduce_axes)
    return [batch_mean, batch_var]
  }
  call = (x: Tensor): Tensor => {
    const [batch_mean, batch_var] = this.calc_stats(x)
    // NOTE: wow, this === done all throughout training in most PyTorch models
    if (this.track_running_stats && Tensor.training) {
      this.running_mean!.assign(this.running_mean!.mul(1 - this.momentum, true).add(batch_mean.detach().mul(this.momentum, true)))
      this.running_var!.assign(this.running_var!.mul(1 - this.momentum, true).add(batch_var.detach().mul(div(mul(this.momentum, x.numel()), sub(x.numel(), x.shape[1])) as number, true)))
      this.num_batches_tracked = this.num_batches_tracked.add(1)
    }
    return x.batchnorm(this.weight, this.bias, batch_mean, batch_var.add(this.eps).rsqrt())
  }
}
export const BatchNorm2d = BatchNorm
export const BatchNorm3d = BatchNorm

/**
 * Applies a 1D convolution over an input signal composed of several input planes.
 *
 * See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * conv = nn.Conv1d(1, 1, 3)
 * t = Tensor.rand(1, 1, 4)
 * console.log(t.numpy())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = conv(t)
 * console.log(t.numpy())
 * ```
 */
export const Conv1d = (in_channels: number, out_channels: number, kernel_size: number, stride = 1, padding: number | string = 0, dilation = 1, groups = 1, bias = true): Conv2d => {
  return new Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
}
/**
 * Applies a 2D convolution over an input signal composed of several input planes.
 *
 * See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * conv = nn.Conv2d(1, 1, 3)
 * t = Tensor.rand(1, 1, 4, 4)
 * console.log(t.numpy())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = conv(t)
 * console.log(t.numpy())
 * ```
 */
export class Conv2d {
  kernel_size: number[]
  stride: number
  dilation: number
  groups: number
  padding: number | number[]
  weight: Tensor
  bias?: Tensor
  constructor(in_channels: number, out_channels: number, kernel_size: number | number[], stride = 1, padding: number | number[] | string = 0, dilation = 1, groups = 1, bias = true) {
    this.kernel_size = make_tuple(kernel_size, 2)
    if (typeof padding === 'string') {
      if (padding.toLowerCase() !== 'same') throw new Error(`Invalid padding string ${padding}, only 'same' is supported`)
      if (stride !== 1) throw new Error("padding='same' is not supported for strided convolutions")
      const pad = zip(make_tuple(dilation, this.kernel_size.length), this.kernel_size.toReversed()).map(([d, k]) => [idiv(d * (k - 1), 2), d * (k - 1) - idiv(d * (k - 1), 2)])
      padding = flatten(pad)
    }
    this.stride = stride, this.dilation = dilation, this.groups = groups, this.padding = padding
    const scale = 1 / Math.sqrt(in_channels * prod(this.kernel_size))
    this.weight = Tensor.uniform([out_channels, idiv(in_channels, groups), ...this.kernel_size], -scale, scale)
    this.bias = bias ? Tensor.uniform([out_channels], -scale, scale) : undefined
  }
  call = (x: Tensor): Tensor => x.conv2d(this.weight, this.bias, this.groups, this.stride, this.dilation, this.padding)
}
/**
 * Applies a linear transformation to the incoming data.
 *
 * See: https://pytorch.org/docs/stable/generated/torch.nn.Linear
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * lin = nn.Linear(3, 4)
 * t = Tensor.rand(2, 3)
 * console.log(t.numpy())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = lin(t)
 * console.log(t.numpy())
 * ```
 */
export class Linear {
  weight: Tensor
  bias?: Tensor
  constructor(in_features: number, out_features: number, bias = true) {
    const bound = 1 / Math.sqrt(in_features)
    this.weight = Tensor.uniform([out_features, in_features], -bound, bound)
    this.bias = bias ? Tensor.uniform([out_features], -bound, bound) : undefined
  }
  call = (x: Tensor): Tensor => x.linear(this.weight.transpose(), this.bias)
}
/**
 * Applies Layer Normalization over a mini-batch of inputs.
 *
 * - Described: https://paperswithcode.com/method/layer-normalization
 * - Paper: https://arxiv.org/abs/1607.06450v1
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * norm = nn.LayerNorm(3)
 * t = Tensor.rand(2, 5, 3) * 2 + 1
 * console.log(t.mean().item(), t.std().item())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = norm(t)
 * console.log(t.mean().item(), t.std().item())
 * ```
 */
export class LayerNorm {
  normalized_shape: number[]
  axis: number[]
  eps: number
  elementwise_affine: boolean
  weight?: Tensor
  bias?: Tensor
  constructor(normalized_shape: number | number[], eps = 1e-5, elementwise_affine = true) {
    this.normalized_shape = make_tuple(normalized_shape, 1)
    this.axis = range(this.normalized_shape.length).map((i) => -1 - i), this.eps = eps, this.elementwise_affine = elementwise_affine
    this.weight = elementwise_affine ? Tensor.ones(this.normalized_shape) : undefined
    this.bias = elementwise_affine ? Tensor.zeros(this.normalized_shape) : undefined
  }
  call(x: Tensor): Tensor {
    if (!is_eq(this.normalized_shape, x.shape.slice(this.normalized_shape.length))) throw new Error(`last dimensions of ${x.shape} must match ${this.normalized_shape}`)
    x = x.layernorm(this.axis, this.eps)
    if (!this.elementwise_affine) return x
    return x.mul(this.weight!).add(this.bias!)
  }
}
/**
 * Applies Layer Normalization over a mini-batch of 2D inputs.
 *
 * See: `LayerNorm`
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * norm = nn.LayerNorm2d(3)
 * t = Tensor.rand(2, 3, 4, 4) * 2 + 1
 * console.log(t.mean().item(), t.std().item())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = norm(t)
 * console.log(t.mean().item(), t.std().item())
 * ```
 */
export class LayerNorm2d extends LayerNorm {
  override call(x: Tensor): Tensor {
    return super.call(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
  }
}

/**
 * Applies Root Mean Square Normalization to input.
 *
 * - Described: https://paperswithcode.com/method/rmsnorm
 * - Paper: https://arxiv.org/abs/1910.07467
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * norm = nn.RMSNorm(4)
 * t = Tensor.arange(12, dtype=dtypes.number).reshape(3, 4)
 * console.log(t.numpy())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * console.log(norm(t).numpy())
 * ```
 */
export class RMSNorm {
  eps: number
  weight: Tensor
  constructor(dim: number, eps = 1e-6) {
    this.eps = eps, this.weight = Tensor.ones([dim])
  }

  _norm = (x: Tensor): Tensor => x.mul((x.square().mean(-1, true).add(this.eps)).rsqrt())

  call = (x: Tensor): Tensor => this._norm(x.float()).cast(x.dtype).mul(this.weight)
}
