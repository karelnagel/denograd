/**
 * This === where the forwards && backwards passes live.
 */
import { DType, dtypes, sum_acc_dtype } from './dtype.ts'
import { LazyBuffer } from './engine/lazy.ts'
import { argsort, range, zip } from './helpers.ts'
import { add, ne, Ops, resolve, sint, sub } from './ops.ts'
import { Function } from './tensor.ts'

export class Contiguous extends Function {
  override forward = (x: LazyBuffer): LazyBuffer => x.contiguous()
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output
}

export class ContiguousBackward extends Function {
  override forward = (x: LazyBuffer): LazyBuffer => x
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.contiguous()
}

export class Cast extends Function {
  input_dtype!: DType
  bitcast!: boolean
  override forward = ({ x, dtype, bitcast = false }: { x: LazyBuffer; dtype: DType; bitcast?: boolean }): LazyBuffer => {
    this.input_dtype = x.dtype, this.bitcast = bitcast
    return this.bitcast ? x.bitcast(dtype) : x.cast(dtype)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => {
    if (this.bitcast) throw new Error('bitcast can!backward')
    return grad_output.cast(this.input_dtype!)
  }
}

// // ************* unary ops *************

export class Reciprocal extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = x.reciprocal()
    return this.ret
  }

  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.neg().mul(this.ret).mul(this.ret)
}
export class Sin extends Function {
  x!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.x = x
    return x.sin()
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => (this.x.sub(Math.PI / 2, true)).sin().mul(grad_output)
}
export class Relu extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = x.maximum(0)
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => (this.ret.gt(0)).cast(grad_output.dtype).mul(grad_output)
}
export class Log extends Function {
  x!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.x = x
    return x.log2().mul(Math.log(2))
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.div(this.x)
}
export class Exp extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = (x.mul(1 / Math.log(2))).exp2()
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => this.ret.mul(grad_output)
}
export class Sqrt extends Function {
  ret!: LazyBuffer
  override forward = (x: LazyBuffer): LazyBuffer => {
    this.ret = x.sqrt()
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.div(this.ret.mul(2))
}
// // NOTE: the implicit derivative of sigmoid !== stable
// // https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
// // TODO: have the backend automatically find this
export class Sigmoid extends Function {
}
export class Sign extends Function {
  override forward = (x: LazyBuffer): LazyBuffer => x.ne(0).where((x.lt(0)).where(x.const_like(-1), x.const_like(1)), x.const_like(0))
  //   // backward always return 0 to match torch
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.const_like(0)
}
// // ************* binary ops *************

export class Less extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.lt(y)
  override backward = (grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [undefined, undefined]
}
export class Neq extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.ne(y)
  override backward = (grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [undefined, undefined]
}
export class Xor extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.xor(y)
}
export class BitwiseAnd extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.bitwiseAnd(y)
}
export class BitwiseOr extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.bitwiseOr(y)
}
export class Threefry extends Function {
  override forward = ({ x, seed }: { x: LazyBuffer; seed: LazyBuffer }): LazyBuffer => x.threefry(seed)
}

export class Add extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.add(y)

  override backward = (grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [this.needs_input_grad[0] ? grad_output : undefined, this.needs_input_grad[1] ? grad_output : undefined]
}
export class Mul extends Function {
  x!: LazyBuffer
  y!: LazyBuffer
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => {
    this.x, this.y = x, y
    return x.mul(y)
  }
  override backward = (grad_output: LazyBuffer): [LazyBuffer | undefined, LazyBuffer | undefined] => [this.needs_input_grad[0] ? (this.y.mul(grad_output)) : undefined, this.needs_input_grad[1] ? (this.x.mul(grad_output)) : undefined]
}
export class IDiv extends Function {
  override forward = ({ x, y }: { x: LazyBuffer; y: LazyBuffer }): LazyBuffer => x.idiv(y)
}
// // ************* ternary ops *************

export class Where extends Function {
  x!: LazyBuffer
  override forward = ({ x, y, z }: { x: LazyBuffer; y: LazyBuffer; z: LazyBuffer }): LazyBuffer => {
    this.x = x
    return this.x.where(y, z)
  }
  override backward = (grad_output: LazyBuffer): [undefined, LazyBuffer | undefined, LazyBuffer | undefined] => [
    undefined,
    this.needs_input_grad[1] ? this.x.where(grad_output, grad_output.const_like(0)) : undefined,
    this.needs_input_grad[2] ? this.x.where(grad_output.const_like(0), grad_output) : undefined,
  ]
}
// // ************* reduce ops *************

export class Sum extends Function {
  input_shape!: sint[]
  override forward = ({ x, axis }: { x: LazyBuffer; axis: number[] }): LazyBuffer => {
    this.input_shape = x.shape
    return x.r(Ops.ADD, axis)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.expand(this.input_shape)
}
export class Prod extends Function {
}
export class Max extends Function {
  x!: LazyBuffer
  ret!: LazyBuffer
  axis!: number[]
  override forward = ({ x, axis }: { x: LazyBuffer; axis: number[] }): LazyBuffer => {
    this.x = x, this.ret = x.r(Ops.MAX, axis), this.axis = axis
    return this.ret
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => {
    // 1s in locations where the max was chosen (can be two locations)
    const max_is_1s = this.x.ne(this.ret.expand(this.x.shape)).ne(this.x.const_like(1).cast(dtypes.bool)).cast(grad_output.dtype)
    const div = max_is_1s.r(Ops.ADD, this.axis).expand(this.x.shape)
    return (max_is_1s.div(div)).mul(grad_output.expand(this.x.shape))
  }
}
// // ************* movement ops *************

// // NOTE: this === sum in reverse
export class Expand extends Function {
  expanded_axis!: number[]
  override forward = ({ x, shape }: { x: LazyBuffer; shape: number[] }): LazyBuffer => {
    this.expanded_axis = zip(x.shape, shape).filter(([si, so]) => resolve(ne(si, so))).map((_, i) => i)
    return x.expand(shape)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => {
    return grad_output.cast(sum_acc_dtype(grad_output.dtype)).r(Ops.ADD, this.expanded_axis).cast(grad_output.dtype)
  }
}

export class Reshape extends Function {
  input_shape!: sint[]
  override forward = ({ x, shape }: { x: LazyBuffer; shape: number[] }): LazyBuffer => {
    this.input_shape = x.shape
    return x.reshape(shape)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.reshape(this.input_shape)
}
export class Permute extends Function {
  input_order!: number[]
  override forward = ({ x, order }: { x: LazyBuffer; order: number[] }): LazyBuffer => {
    this.input_order = order
    return x.permute(order)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.permute(argsort(this.input_order))
}
export class Pad extends Function {
  narg!: [sint, sint][]
  override forward = ({ x, arg }: { x: LazyBuffer; arg: [number, number][] }): LazyBuffer => {
    this.narg = zip(x.shape, arg).map(([s, p]) => [p[0], add(s, p[0])])
    return x.pad(arg)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.shrink(this.narg)
}
export class Shrink extends Function {
  narg!: [sint, sint][]
  override forward = ({ x, arg }: { x: LazyBuffer; arg: [sint, sint][] }): LazyBuffer => {
    this.narg = zip(x.shape, arg).map(([s, p]) => [p[0], sub(s, p[1])])
    return x.shrink(arg)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.pad(this.narg)
}
export class Flip extends Function {
  arg!: number[]
  override forward = ({ x, axis }: { x: LazyBuffer; axis: number[] }): LazyBuffer => {
    this.arg = range(x.shape.length).map((i) => axis.includes(i) ? -1 : 1)
    return x.stride(this.arg)
  }
  override backward = (grad_output: LazyBuffer): LazyBuffer => grad_output.stride(this.arg)
}
