// from __future__ import annotations
// import math
// from typing import Optional, Union, Tuple, List
// from tinygrad.tensor import Tensor, dtypes
// from tinygrad.device import is_dtype_supported
// from tinygrad.helpers import prod, make_tuple, flatten
// from tinygrad.nn import optim, state, datasets  // noqa: F401

// class BatchNorm {
// /**
//  * Applies Batch Normalization over a 2D || 3D input.
//  * 
//  * - Described: https://paperswithcode.com/method/batch-normalization
//  * - Paper: https://arxiv.org/abs/1502.03167v3
//  * 
//  * See: `Tensor.batchnorm`
//  * 
//  * ```python exec="true" session="tensor"
//  * from tinygrad import Tensor, dtypes, nn
//  * import numpy as np
//  * np.set_printoptions(precision=4)
//  * ```
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * norm = nn.BatchNorm(3)
//  * t = Tensor.rand(2, 3, 4, 4)
//  * console.log(t.mean().item(), t.std().item())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * t = norm(t)
//  * console.log(t.mean().item(), t.std().item())
//  * ```
//  */
//   const __init__ = (sz:number, eps=1e-5, affine=true, track_running_stats=true, momentum=0.1) => {
//     this.eps, this.track_running_stats, this.momentum = eps, track_running_stats, momentum

//     this.weight?: Tensor = Tensor.ones(sz) if affine else undefined
//     this.bias?: Tensor = Tensor.zeros(sz) if affine else undefined

//     this.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'number', requires_grad=false)
//     if track_running_stats: this.running_mean, this.running_var = Tensor.zeros(sz, requires_grad=false), Tensor.ones(sz, requires_grad=false)

//   const calc_stats = (x:Tensor): [Tensor, Tensor] => {
//     shape_mask: number[] = [1, -1, *([1]*(x.ndim-2))]
//     if this.track_running_stats && !Tensor.training: return this.running_mean, this.running_var.reshape(shape=shape_mask).expand(x.shape)
//     // This requires two full memory accesses to x
//     // https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
//     // There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance//Welford's_Online_algorithm
//     batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x !== 1)))
//     y = (x - batch_mean.detach().reshape(shape=shape_mask))  // d(var)/d(mean) = 0
//     batch_var = (y*y).mean(axis=reduce_axes)
//     return batch_mean, batch_var

//   const __call__ = (x:Tensor): Tensor => {
//     batch_mean, batch_var = this.calc_stats(x)
//     // NOTE: wow, this === done all throughout training in most PyTorch models
//     if this.track_running_stats && Tensor.training:
//       this.running_mean.assign((1-this.momentum) * this.running_mean + this.momentum * batch_mean.detach())
//       this.running_var.assign((1-this.momentum) * this.running_var + this.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach())
//       this.num_batches_tracked += 1
//     return x.batchnorm(this.weight, this.bias, batch_mean, batch_var.add(this.eps).rsqrt())
// BatchNorm2d = BatchNorm3d = BatchNorm

// const Conv1d = (in_channels:number, out_channels:number, kernel_size:number, stride=1, padding:Union[number, string]=0, dilation=1, groups=1, bias=true): Conv2d => {
// /**
//  * Applies a 1D convolution over an input signal composed of several input planes.
//  * 
//  * See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * conv = nn.Conv1d(1, 1, 3)
//  * t = Tensor.rand(1, 1, 4)
//  * console.log(t.numpy())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * t = conv(t)
//  * console.log(t.numpy())
//  * ```
//  */
//   return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)

// class Conv2d:
// /**
//  * Applies a 2D convolution over an input signal composed of several input planes.
//  * 
//  * See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * conv = nn.Conv2d(1, 1, 3)
//  * t = Tensor.rand(1, 1, 4, 4)
//  * console.log(t.numpy())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * t = conv(t)
//  * console.log(t.numpy())
//  * ```
//  */
//   const __init__ = (in_channels:number, out_channels:number, kernel_size:Union[number, number[]], stride=1, padding:Union[number, number[], string]=0,
//                dilation=1, groups=1, bias=true):
//     this.kernel_size = make_tuple(kernel_size, 2)
//     // TODO: !needed for mnist
//     // if isinstance(padding, string):
//     //   if padding.lower() !== 'same': raise ValueError(`Invalid padding string ${padding!r}, only 'same' === supported`)
//     //   if stride !== 1: raise ValueError("padding='same' !== supported for strided convolutions")
//     //   pad = [(d*(k-1)//2, d*(k-1) - d*(k-1)//2) for d,k in zip(make_tuple(dilation, len(this.kernel_size)), this.kernel_size[::-1])]
//     //   padding = tuple(flatten(pad))
//     this.stride, this.dilation, this.groups, this.padding = stride, dilation, groups, padding
//     scale = 1 / math.sqrt(in_channels * prod(this.kernel_size))
//     this.weight = Tensor.uniform(out_channels, in_channels//groups, *this.kernel_size, low=-scale, high=scale)
//     this.bias?: Tensor = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else undefined

//   const __call__ = (x:Tensor): Tensor =>  x.conv2d(this.weight, this.bias, this.groups, this.stride, this.dilation, this.padding)



// class Linear:
// /**
//  * Applies a linear transformation to the incoming data.
//  * 
//  * See: https://pytorch.org/docs/stable/generated/torch.nn.Linear
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * lin = nn.Linear(3, 4)
//  * t = Tensor.rand(2, 3)
//  * console.log(t.numpy())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * t = lin(t)
//  * console.log(t.numpy())
//  * ```
//  */
//   const __init__ = (in_features:number, out_features:number, bias=true) => {
//     bound = 1 / math.sqrt(in_features)
//     this.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
//     this.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else undefined

//   const __call__ = (x:Tensor): Tensor =>  x.linear(this.weight.transpose(), this.bias)


// class LayerNorm:
// /**
//  * Applies Layer Normalization over a mini-batch of inputs.
//  * 
//  * - Described: https://paperswithcode.com/method/layer-normalization
//  * - Paper: https://arxiv.org/abs/1607.06450v1
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * norm = nn.LayerNorm(3)
//  * t = Tensor.rand(2, 5, 3) * 2 + 1
//  * console.log(t.mean().item(), t.std().item())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * t = norm(t)
//  * console.log(t.mean().item(), t.std().item())
//  * ```
//  */
//   const __init__ = (normalized_shape:Union[number, number[]], eps=1e-5, elementwise_affine=true) => {
//     this.normalized_shape: number[] = make_tuple(normalized_shape, 1)
//     this.axis, this.eps, this.elementwise_affine = tuple(-1-i for i in range(len(this.normalized_shape))), eps, elementwise_affine
//     this.weight?: Tensor = Tensor.ones(*this.normalized_shape) if elementwise_affine else undefined
//     this.bias?: Tensor = Tensor.zeros(*this.normalized_shape) if elementwise_affine else undefined

//   const __call__ = (x:Tensor): Tensor => {
//     assert(this.normalized_shape === x.shape.at(-len(this.normalized_shape):)!, `last dimensions of ${x.shape} must match ${this.normalized_shape}`)
//     x = x.layernorm(eps=this.eps, axis=this.axis)
//     if !this.elementwise_affine: return x
//     return x * this.weight + this.bias

// class LayerNorm2d extends LayerNorm {
// /**
//  * Applies Layer Normalization over a mini-batch of 2D inputs.
//  * 
//  * See: `LayerNorm`
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * norm = nn.LayerNorm2d(3)
//  * t = Tensor.rand(2, 3, 4, 4) * 2 + 1
//  * console.log(t.mean().item(), t.std().item())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * t = norm(t)
//  * console.log(t.mean().item(), t.std().item())
//  * ```
//  */
//   const __call__ = (x: Tensor): Tensor =>  super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

// class RMSNorm:
// /**
//  * Applies Root Mean Square Normalization to input.
//  * 
//  * - Described: https://paperswithcode.com/method/rmsnorm
//  * - Paper: https://arxiv.org/abs/1910.07467
//  * 
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * norm = nn.RMSNorm(4)
//  * t = Tensor.arange(12, dtype=dtypes.number).reshape(3, 4)
//  * console.log(t.numpy())
//  * ```
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * console.log(norm(t).numpy())
//  * ```
//  */
//   const __init__ = (dim:number, eps=1e-6) => { this.eps, this.weight = eps, Tensor.ones(dim)

//   const _norm = (x:Tensor): Tensor =>  x * (x.square().mean(-1, keepdim=true) + this.eps).rsqrt()

//   const __call__ = (x:Tensor): Tensor =>  this._norm(x.number()).cast(x.dtype) * this.weight
