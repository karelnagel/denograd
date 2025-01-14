import { Tensor } from './denograd/tensor.ts'

const a = new Tensor([2])
const b = new Tensor([6])

console.log(await a.add(b).tolist())
