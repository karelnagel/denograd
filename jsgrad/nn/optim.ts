import { dtypes, least_upper_dtype } from '../dtype.ts'
import { dedup, NotImplemented, vars, zip } from '../helpers/helpers.ts'
import { Tensor } from '../tensor.ts'

export class Optimizer {
  params: Tensor[]
  device: string | string[]
  buffers: Tensor[]
  lr: Tensor
  constructor(params: Tensor[], lr: number) {
    // if it's undefined, but being put into an optimizer, set it to true
    for (const x of params) if (x.requires_grad === undefined) x.requires_grad = true

    this.params = dedup(params.filter((x) => x.requires_grad))
    if (this.params.length === 0) throw new Error('optimizer must have at least one param')
    this.device = this.params[0].device
    this.buffers = dedup(params.filter((x) => !x.requires_grad)) // buffers are still realized
    // store lr in at least float32 precision
    this.lr = new Tensor(vars.get('CONST_LR') ? lr : [lr], {
      requires_grad: false,
      device: this.device,
      dtype: least_upper_dtype(dtypes.default_float, dtypes.float32),
    })
  }
  /**
   * Zeroes the gradients of all the parameters.
   */
  zero_grad = () => {
    for (const param of this.params) param.grad = undefined
  }
  /**
   * Performs a single optimization step.
   */
  step = async () => await Tensor.realize(this.schedule_step())
  /**
   * Returns the tensors that need to be realized to perform a single optimization step.
   */
  schedule_step = (): Tensor[] => {
    if (!Tensor.training) throw new Error(`Tensor.training=${Tensor.training}, Tensor.training must be enabled to use the optimizer. Consider setting Tensor.training=True before calling Optimizer.step().`)
    return [...this.schedule_step_with_grads(this.params.map((t) => t.grad!)), ...this.params, ...this.buffers]
  }
  schedule_step_with_grads = (grads: Tensor[]): Tensor[] => {
    throw new NotImplemented()
  }
}

/**
 * Combines multiple optimizers into one.
 */
export class OptimizerGroup extends Optimizer {
  optimizers: Optimizer[]
  constructor(optimizers: Optimizer[]) {
    super([], 0)
    this.optimizers = optimizers
    this.params = optimizers.flatMap((o) => o.params)
    this.buffers = optimizers.flatMap((o) => o.buffers)
  }
  get = (i: number) => this.optimizers[i]
  override zero_grad = () => this.optimizers.map((o) => o.zero_grad())
  override schedule_step = (): Tensor[] => this.optimizers.flatMap((o) => o.schedule_step())
}

// LARS === essentially just trust ratio to SGD so if we just set the trust coeff 0.0 its just standard SGD.
// """
// Stochastic Gradient Descent (SGD) optimizer with optional momentum and weight decay.

// `classic` is a boolean flag that determines whether to use the popular momentum update rule or the classic momentum update rule.

// - Described: https://paperswithcode.com/method/sgd
// """
export const SGD = (params: Tensor[], lr = 0.001, momentum = 0.0, weight_decay = 0.0, nesterov = false, classic = false) => {
  return new LARS(params, lr, momentum, weight_decay, nesterov, classic, 0.0)
}

// """
// Layer-wise Adaptive Rate Scaling (LARS) optimizer with optional momentum and weight decay.

// - Described: https://paperswithcode.com/method/lars
// - Paper: https://arxiv.org/abs/1708.03888v3
// """
export class LARS extends Optimizer {
  b: Tensor[]
  constructor(params: Tensor[], lr = 0.001, public momentum = 0.9, public weight_decay = 1e-4, public nesterov = false, public classic = true, public tcoef = 0.001) {
    super(params, lr)
    this.b = this.momentum ? this.params.map((t) => Tensor.zeros(t.shape, { dtype: t.dtype, device: t.device, requires_grad: false })) : []
  }

  override schedule_step_with_grads = (grads: Tensor[]): Tensor[] => {
    for (let [i, [t, g]] of zip(this.params, grads).entries()) {
      // contiguous is needed since the grads can allegedly form a "diamond"
      // TODO: fix this in lazy.py
      g = g.contiguous()
      let r
      if (this.tcoef !== 0) {
        const r1 = t.detach().square().sum().sqrt()
        const r2 = g.square().sum().sqrt()
        r = (r1.gt(0)).where((r2.gt(0)).where(r1.mul(this.tcoef, true).div(r2.add(r1.mul(this.weight_decay, true))), 1.0), 1.0)
      } else r = 1.0
      g = g.add(t.detach().mul(this.weight_decay, true))
      // classic momentum does post learning rate update
      if (this.classic) g = g.mul(r).mul(this.lr)
      if (this.momentum) {
        this.b[i].assign(this.b[i].mul(this.momentum, true).add(g)) // NOTE: self.b[i] is zero on the first run, no if required
        g = this.nesterov ? (g.add(this.b[i].mul(this.momentum))) : this.b[i]
      }
      // popular momentum does pre learning rate update
      if (!this.classic) g = g.mul(r).mul(this.lr)
      t.assign((t.detach().sub(g)).cast(t.dtype))
    }
    return this.b
  }
}

// LAMB === essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
/**
 * AdamW optimizer with optional weight decay.
 *
 * - Described: https://paperswithcode.com/method/adamw
 * - Paper: https://arxiv.org/abs/1711.05101v3
 */
export const AdamW = (params: Tensor[], lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8, weight_decay = 0.01) => {
  return new LAMB(params, lr, b1, b2, eps, weight_decay, true)
}

/**
 * Adam optimizer.
 *
 * - Described: https://paperswithcode.com/method/adam
 * - Paper: https://arxiv.org/abs/1412.6980
 */
export const Adam = (params: Tensor[], lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8) => {
  return new LAMB(params, lr, b1, b2, eps, 0.0, true)
}

/**
 * LAMB optimizer with optional weight decay.
 *
 * - Described: https://paperswithcode.com/method/lamb
 * - Paper: https://arxiv.org/abs/1904.00962
 */
class LAMB extends Optimizer {
  b1: number
  b2: number
  eps: number
  wd: number
  adam: boolean
  b1_t: Tensor
  b2_t: Tensor
  m: Tensor[]
  v: Tensor[]
  constructor(params: Tensor[], lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-6, weight_decay = 0.0, adam = false) {
    super(params, lr)
    this.b1 = b1, this.b2 = b2, this.eps = eps, this.wd = weight_decay, this.adam = adam
    ;[this.b1_t, this.b2_t] = [b1, b2].map((_) => Tensor.ones([1], { dtype: dtypes.float32, device: this.device, requires_grad: false }).contiguous())
    this.m = this.params.map((t) => Tensor.zeros(t.shape, { dtype: dtypes.float32, device: t.device, requires_grad: false }).contiguous())
    this.v = this.params.map((t) => Tensor.zeros(t.shape, { dtype: dtypes.float32, device: t.device, requires_grad: false }).contiguous())
  }
  override schedule_step_with_grads = (grads: Tensor[]): Tensor[] => {
    this.b1_t = this.b1_t.mul(this.b1)
    this.b2_t = this.b2_t.mul(this.b2)
    for (const [i, [t, g]] of zip(this.params, grads).entries()) {
      this.m[i].assign(this.m[i].mul(this.b1, true).add(g.mul(1.0 - this.b1, true)))
      this.v[i].assign(this.v[i].mul(this.b2, true).add((g.mul(g)).mul(1.0 - this.b2, true)))
      const m_hat = this.m[i].div(this.b1_t.sub(1.0, true))
      const v_hat = this.v[i].div(this.b2_t.sub(1.0, true))
      const up = (m_hat.div(v_hat.sqrt().add(this.eps))).add(t.detach().mul(this.wd, true))
      let r
      if (!this.adam) {
        const r1 = t.detach().square().sum().sqrt()
        const r2 = up.square().sum().sqrt()
        r = r1.gt(0).where(r2.gt(0).where(r1.div(r2), 1.0), 1.0)
      } else {
        r = 1.0
      }
      t.assign((t.detach().sub(this.lr.mul(r).mul(up))).cast(t.dtype))
    }
    return [this.b1_t, this.b2_t, ...this.m, ...this.v]
  }
}
