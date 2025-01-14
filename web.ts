import { Tensor } from './denograd/tensor.ts'

const a = new Tensor([2.3])
const b = new Tensor([6.4])

console.log(await a.div(b).add(4).tolist())
