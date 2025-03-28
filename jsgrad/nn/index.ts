import { is_dtype_supported } from '../device.ts'
import { dtypes } from '../dtype.ts'
import { flatten, is_eq, make_tuple, num, range, zip } from '../helpers/helpers.ts'
import { div, idiv, mul, prod, sub } from '../helpers/helpers.ts'
import type { sint } from '../ops.ts'
import { type Layer, Tensor } from '../tensor.ts'
import { get_state_dict } from './state.ts'
import { load_state_dict, safe_load, safe_save } from './state.ts'

export * from './optim.ts'
export * from './state.ts'
export * from './datasets.ts'

/**
 * Abstract model to simplify calling, loading and saving different models.
 * You only need to implement list of layers to create a new model
 * ```ts
 * import { Layer, Tensor } from "@jsgrad/jsgrad"
 * export class MNIST extends Model {
 *  layers: Layer[] = [
 *    new Conv2d(1, 32, 5),
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
  DEFAULT_LOAD?: string
  abstract layers: Layer[]
  /**
   * Call model with a Tensor input, returns a Tensor with output.
   * ```ts
   * import { MNIST, Tensor } from "@jsgrad/jsgrad"
   *
   * const model = new MNIST()
   * const res = model.call(Tensor.rand([1, 1, 28, 28]))
   * console.log(await res.tolist())
   * ```
   */
  call = (x: Tensor) => x.sequential(this.layers)
  /**
   * Load model weights from a .safetensors file at the given path or absolute URL
   * ```ts
   * import { MNIST } from "@jsgrad/jsgrad"
   * const model = new MNIST()
   * await model.load("./model.safetensors")
   * ```
   */
  load = async (path?: string | Tensor) => {
    if (!path && !this.DEFAULT_LOAD) {
      throw new Error(
        `You need to specify model path, can be URL or local path!`,
      )
    }
    await load_state_dict(this, await safe_load(path || this.DEFAULT_LOAD!))
    return this
  }
  /**
   * Save model weights to a .safetensors file at the given path
   * ```ts
   * import { MNIST } from "@jsgrad/jsgrad"
   *
   * const model = new MNIST()
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
  constructor(
    sz: number,
    eps = 1e-5,
    affine = true,
    track_running_stats = true,
    momentum = 0.1,
  ) {
    this.eps = eps, this.track_running_stats = track_running_stats, this.momentum = momentum

    this.weight = affine ? Tensor.ones([sz]) : undefined
    this.bias = affine ? Tensor.zeros([sz]) : undefined

    this.num_batches_tracked = Tensor.zeros([1], {
      dtype: is_dtype_supported(dtypes.long) ? dtypes.long : dtypes.int,
      requires_grad: false,
    })
    if (track_running_stats) {
      this.running_mean = Tensor.zeros([sz], { requires_grad: false }), this.running_var = Tensor.ones([sz], { requires_grad: false })
    }
  }
  calc_stats = (x: Tensor): [Tensor, Tensor] => {
    const shape_mask: number[] = [1, -1, ...(range(x.ndim - 2).map((x) => 1))]
    if (this.track_running_stats && !Tensor.training) {
      return [
        this.running_mean!,
        this.running_var!.reshape(shape_mask).expand(x.shape),
      ]
    }
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
      this.running_mean!.assign(
        this.running_mean!.mul(1 - this.momentum, true).add(
          batch_mean.detach().mul(this.momentum, true),
        ),
      )
      this.running_var!.assign(
        this.running_var!.mul(1 - this.momentum, true).add(
          batch_var.detach().mul(
            num(div(mul(this.momentum, x.numel()), sub(x.numel(), x.shape[1]))),
            true,
          ),
        ),
      )
      this.num_batches_tracked = this.num_batches_tracked.add(1)
    }
    return x.batchnorm(
      this.weight,
      this.bias,
      batch_mean,
      batch_var.add(this.eps).rsqrt(),
    )
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
export const Conv1d = (
  in_channels: number,
  out_channels: number,
  kernel_size: number,
  stride = 1,
  padding: number | string = 0,
  dilation = 1,
  groups = 1,
  bias = true,
): Conv2d => {
  return new Conv2d(
    in_channels,
    out_channels,
    [kernel_size],
    stride,
    padding,
    dilation,
    groups,
    bias,
  )
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
  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | number[],
    stride = 1,
    padding: number | number[] | string = 0,
    dilation = 1,
    groups = 1,
    bias = true,
  ) {
    this.kernel_size = make_tuple(kernel_size, 2)
    if (typeof padding === 'string') {
      if (padding.toLowerCase() !== 'same') {
        throw new Error(
          `Invalid padding string ${padding}, only 'same' is supported`,
        )
      }
      if (stride !== 1) {
        throw new Error(
          "padding='same' is not supported for strided convolutions",
        )
      }
      const pad = zip(
        make_tuple(dilation, this.kernel_size.length),
        this.kernel_size.toReversed(),
      ).map((
        [d, k],
      ) => [idiv(d * (k - 1), 2), d * (k - 1) - idiv(d * (k - 1), 2)])
      padding = flatten(pad)
    }
    this.stride = stride, this.dilation = dilation, this.groups = groups, this.padding = padding
    const scale = 1 / Math.sqrt(in_channels * prod(this.kernel_size))
    this.weight = Tensor.uniform(
      [out_channels, idiv(in_channels, groups), ...this.kernel_size],
      -scale,
      scale,
    )
    this.bias = bias ? Tensor.uniform([out_channels], -scale, scale) : undefined
  }
  call = (x: Tensor): Tensor =>
    x.conv2d(
      this.weight,
      this.bias,
      this.groups,
      this.stride,
      this.dilation,
      this.padding,
    )
}

/**
 *
 * Applies a 1D transposed convolution operator over an input signal composed of several input planes.

 * See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

 * ```python exec="true" source="above" session="tensor" result="python"
 * conv = nn.ConvTranspose1d(1, 1, 3)
 * t = Tensor.rand(1, 1, 4)
 * print(t.numpy())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = conv(t)
 * print(t.numpy())
 * ```
 */
export const ConvTranspose1d = (
  in_channels: number,
  out_channels: number,
  kernel_size: number,
  stride = 1,
  padding = 0,
  output_padding = 0,
  dilation = 1,
  groups = 1,
  bias = true,
): ConvTranspose2d => {
  return new ConvTranspose2d(
    in_channels,
    out_channels,
    [kernel_size],
    stride,
    padding,
    output_padding,
    dilation,
    groups,
    bias,
  )
}

/**
 * Applies a 2D transposed convolution operator over an input image.
 *
 * See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * conv = nn.ConvTranspose2d(1, 1, 3)
 * t = Tensor.rand(1, 1, 4, 4)
 * print(t.numpy())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = conv(t)
 * print(t.numpy())
 * ```
 */
export class ConvTranspose2d extends Conv2d {
  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number | number[],
    stride = 1,
    padding = 0,
    public output_padding = 0,
    dilation = 1,
    groups = 1,
    bias = true,
  ) {
    super(
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      bias,
    )
    const scale = 1 / Math.sqrt(in_channels * prod(this.kernel_size))
    this.weight = Tensor.uniform(
      [in_channels, idiv(out_channels, groups), ...this.kernel_size],
      -scale,
      scale,
    )
  }
  override call = (x: Tensor): Tensor => {
    return x.conv_transpose2d(
      this.weight,
      this.bias,
      this.groups,
      this.stride,
      this.dilation,
      this.padding,
      this.output_padding,
    )
  }
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
 *
 * Applies Group Normalization over a mini-batch of inputs.

 * - Described: https://paperswithcode.com/method/group-normalization
 * - Paper: https://arxiv.org/abs/1803.08494v3

 * ```python exec="true" source="above" session="tensor" result="python"
 * norm = nn.GroupNorm(2, 12)
 * t = Tensor.rand(2, 12, 4, 4) * 2 + 1
 * print(t.mean().item(), t.std().item())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = norm(t)
 * print(t.mean().item(), t.std().item())
 * ```
 */
export class GroupNorm {
  weight: Tensor | undefined
  bias: Tensor | undefined
  constructor(
    public num_groups: number,
    public num_channels: number,
    public eps = 1e-5,
    affine = true,
  ) {
    this.weight = affine ? Tensor.ones([num_channels]) : undefined
    this.bias = affine ? Tensor.zeros([num_channels]) : undefined
  }
  call = (x: Tensor): Tensor => {
    // reshape for layernorm to work as group norm
    // subtract mean and divide stddev
    x = x.reshape([x.shape[0], this.num_groups, -1]).layernorm(
      undefined,
      this.eps,
    ).reshape(x.shape)

    if (this.weight === undefined || this.bias === undefined) return x
    // elementwise_affine on channels
    return x.mul(
      this.weight.reshape([1, -1, ...range(x.ndim - 2).map(() => 1)]),
    ).add(this.bias.reshape([1, -1, ...range(x.ndim - 2).map(() => 1)]))
  }
}

/**
 * Applies Instance Normalization over a mini-batch of inputs.
 *
 * - Described: https://paperswithcode.com/method/instance-normalization
 * - Paper: https://arxiv.org/abs/1607.08022v3
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * norm = nn.InstanceNorm(3)
 * t = Tensor.rand(2, 3, 4, 4) * 2 + 1
 * print(t.mean().item(), t.std().item())
 * ```
 * ```python exec="true" source="above" session="tensor" result="python"
 * t = norm(t)
 * print(t.mean().item(), t.std().item())
 * ```
 */
export class InstanceNorm {
  weight: Tensor | undefined
  bias: Tensor | undefined
  constructor(public num_features: number, public eps = 1e-5, affine = true) {
    this.weight = affine ? Tensor.ones([num_features]) : undefined
    this.bias = affine ? Tensor.zeros([num_features]) : undefined
  }

  call = (x: Tensor): Tensor => {
    x = x.reshape([x.shape[0], this.num_features, -1]).layernorm(
      undefined,
      this.eps,
    ).reshape(x.shape)
    if (this.weight === undefined || this.bias === undefined) return x
    return x.mul(
      this.weight.reshape([1, -1, ...range(x.ndim - 2).map(() => 1)]),
    ).add(this.bias.reshape([1, -1, ...range(x.ndim - 2).map(() => 1)]))
  }
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
  constructor(
    normalized_shape: number | number[],
    eps = 1e-5,
    elementwise_affine = true,
  ) {
    this.normalized_shape = make_tuple(normalized_shape, 1)
    this.axis = range(this.normalized_shape.length).map((i) => -1 - i), this.eps = eps, this.elementwise_affine = elementwise_affine
    this.weight = elementwise_affine ? Tensor.ones(this.normalized_shape) : undefined
    this.bias = elementwise_affine ? Tensor.zeros(this.normalized_shape) : undefined
  }
  call(x: Tensor): Tensor {
    if (
      !is_eq(
        this.normalized_shape,
        x.shape.slice(-this.normalized_shape.length),
      )
    ) {
      throw new Error(
        `last dimensions of ${x.shape} must match ${this.normalized_shape}`,
      )
    }
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
  weight: Tensor
  constructor(dim: number, public eps = 1e-6) {
    this.weight = Tensor.ones([dim])
  }

  _norm = (x: Tensor): Tensor => x.mul(x.square().mean(-1, true).add(this.eps).rsqrt())

  call = (x: Tensor): Tensor => this._norm(x.float()).cast(x.dtype).mul(this.weight)
}
/**
 * A simple lookup table that stores embeddings of a fixed dictionary and size.
 *
 * See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * emb = nn.Embedding(10, 3)
 * print(emb(Tensor([1, 2, 3, 1])).numpy())
 * ```
 */
export class Embedding {
  weight: Tensor
  arange: Tensor | undefined
  constructor(public vocab_size: number, public embed_size: number) {
    this.weight = Tensor.glorot_uniform([vocab_size, embed_size])
  }

  call = (idx: Tensor): Tensor => {
    if (this.arange === undefined) {
      this.arange = Tensor.arange(this.vocab_size, undefined, undefined, {
        requires_grad: false,
        device: this.weight.device,
      }).unsqueeze(-1)
    }
    const big_shp = [...idx.shape, this.vocab_size, this.embed_size]
    const arange = this.arange.expand(big_shp),
      vals = this.weight!.expand(big_shp)
    idx = idx.reshape([...idx.shape, 1, 1]).expand(big_shp)
    return arange.eq(idx).mul(vals).sum(-2, undefined, vals.dtype)
  }
}

/**
 * A long short-term memory (LSTM) cell.
 *
 * Args:
 *   input_size: The number of expected features in the input `x`
 *   hidden_size: The number of features in the hidden state `h`
 *   bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`
 */
export class LSTMCell {
  weight_ih: Tensor
  weight_hh: Tensor
  bias_ih: Tensor | undefined
  bias_hh: Tensor | undefined
  constructor(input_size: number, hidden_size: number, bias = true) {
    const stdv = 1.0 / Math.sqrt(hidden_size)
    this.weight_ih = Tensor.uniform([hidden_size * 4, input_size], -stdv, stdv)
    this.weight_hh = Tensor.uniform(
      [hidden_size * 4, hidden_size],
      -stdv,
      stdv,
    )
    this.bias_ih = bias ? Tensor.zeros([hidden_size * 4]) : undefined
    this.bias_hh = bias ? Tensor.zeros([hidden_size * 4]) : undefined
  }

  call = (x: Tensor, hc?: [Tensor, Tensor]): [Tensor, Tensor] => {
    if (hc === undefined) {
      hc = range(2).map(() =>
        Tensor.zeros([x.size(0) as sint, this.weight_hh.size(1) as sint], {
          dtype: x.dtype,
          device: x.device,
        })
      ) as [Tensor, Tensor]
    }
    const gates = x.linear(this.weight_ih.T, this.bias_ih).add(
      hc[0].linear(this.weight_hh.T, this.bias_hh),
    )
    let [i, f, g, o] = gates.chunk(4, 1)
    i = i.sigmoid(), f = f.sigmoid(), g = g.tanh(), o = o.sigmoid()
    const new_c = f.mul(hc[1]).add(i.mul(g))
    const new_h = o.mul(new_c.tanh())
    return [new_h.contiguous(), new_c.contiguous()]
  }
}
