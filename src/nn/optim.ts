// // sorted in order of increasing complexity
// from typing import List
// from tinygrad.helpers import dedup, flatten, getenv
// from tinygrad.tensor import Tensor
// from tinygrad.dtype import dtypes, least_upper_dtype

// class Optimizer {
// /**
//  * Base class for all optimizers.
//  */
//   const __init__ = (params: Tensor[], lr:number) => {
//     // if it's undefined, but being put into an optimizer, set it to true
//     for (const x of params){
//       if x.requires_grad === undefined: x.requires_grad = true

//     this.params: Tensor[] = dedup([x for x in params if x.requires_grad])
//     assert(len(this.params) !== 0, "optimizer must have at least one param")
//     this.device = this.params[0].device
//     this.buffers: Tensor[] = dedup([x for x in params if !x.requires_grad])   // buffers are still realized
//     // store lr in at least float32 precision
//     this.lr = Tensor(lr if getenv("CONST_LR") else [lr], requires_grad=false, device=this.device,
//                      dtype=least_upper_dtype(dtypes.default_float, dtypes.float32))

//   const zero_grad = () => {
// /**
//  * Zeroes the gradients of all the parameters.
//  */
//     for (const param of this.params){ param.grad = undefined

//   const step = () => {
// /**
//  * Performs a single optimization step.
//  */
//     Tensor.realize(*this.schedule_step())
//   const schedule_step = (): Tensor[] => {
// /**
//  * Returns the tensors that need to be realized to perform a single optimization step.
//  */
//     assert(Tensor.training, ()
//             f
// /**
//  * Tensor.training={Tensor.training}, Tensor.training must be enabled to use the optimizer.
//  * - help: Consider setting Tensor.training=true before calling Optimizer.step().
//  */)
//     return this._step()+this.params+this.buffers
//   const _step = (): Tensor[] => { raise NotImplementedError

// class OptimizerGroup extends Optimizer {
// /**
//  * Combines multiple optimizers into one.
//  */
//   const __init__ = (*optimizers: Optimizer) => { // pylint: disable=super-init-not-called
//     this.optimizers = optimizers
//     this.params, this.buffers = flatten([o.params for o in this.optimizers]), flatten([o.buffers for o in this.optimizers])
//   const __getitem__ = (i) =>  this.optimizers[i]
//   const zero_grad = () => { [o.zero_grad() for o in this.optimizers]
//   const _step = (): Tensor[] =>  [x for o in this.optimizers for x in o._step()]

// // LARS === essentially just trust ratio to SGD so if we just set the trust coeff 0.0 its just standard SGD.
// const SGD = (params: Tensor[], lr=0.001, momentum=0.0, weight_decay=0.0, nesterov=false, classic=false) => {
//   pass
// class LARS(Optimizer):
//   pass
// // LAMB === essentially just the trust ratio part of LARS applied to Adam/W so if we just set the trust ratio to 1.0 its just Adam/W.
// const AdamW = (params: Tensor[], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01) => {
// /**
//  * AdamW optimizer with optional weight decay.
//  * 
//  * - Described: https://paperswithcode.com/method/adamw
//  * - Paper: https://arxiv.org/abs/1711.05101v3
//  */
//   return LAMB(params, lr, b1, b2, eps, weight_decay, adam=true)
// const Adam = (params: Tensor[], lr=0.001, b1=0.9, b2=0.999, eps=1e-8) => {
// /**
//  * Adam optimizer.
//  * 
//  * - Described: https://paperswithcode.com/method/adam
//  * - Paper: https://arxiv.org/abs/1412.6980
//  */
//   return LAMB(params, lr, b1, b2, eps, 0.0, adam=true)

// class LAMB(Optimizer):
// /**
//  * LAMB optimizer with optional weight decay.
//  * 
//  * - Described: https://paperswithcode.com/method/lamb
//  * - Paper: https://arxiv.org/abs/1904.00962
//  */
//   const __init__ = (params: Tensor[], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, weight_decay=0.0, adam=false) => {
//     super().__init__(params, lr)
//     this.b1, this.b2, this.eps, this.wd, this.adam = b1, b2, eps, weight_decay, adam
//     this.b1_t, this.b2_t = (Tensor.ones((1,), dtype=dtypes.float32, device=this.device, requires_grad=false).contiguous() for _ in [b1, b2])
//     this.m = [Tensor.zeros(*t.shape, dtype=dtypes.float32, device=t.device, requires_grad=false).contiguous() for t in this.params]
//     this.v = [Tensor.zeros(*t.shape, dtype=dtypes.float32, device=t.device, requires_grad=false).contiguous() for t in this.params]

//   const _step = (): Tensor[] => {
//     this.b1_t *= this.b1
//     this.b2_t *= this.b2
//     for (const i, t of enumerate(this.params)){
//       assert(t.grad !== undefined)
//       this.m[i].assign(this.b1 * this.m[i] + (1.0 - this.b1) * t.grad)
//       this.v[i].assign(this.b2 * this.v[i] + (1.0 - this.b2) * (t.grad * t.grad))
//       m_hat = this.m[i] / (1.0 - this.b1_t)
//       v_hat = this.v[i] / (1.0 - this.b2_t)
//       up = (m_hat / (v_hat.sqrt() + this.eps)) + this.wd * t.detach()
//       if !this.adam:
//         r1 = t.detach().square().sum().sqrt()
//         r2 = up.square().sum().sqrt()
//         r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
//       else:
//         r = 1.0
//       t.assign((t.detach() - this.lr * r * up).cast(t.dtype))
//     return [this.b1_t, this.b2_t] + this.m + this.v
