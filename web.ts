import { Tensor } from './denograd/tensor.ts'

const a = new Tensor([4])
const b = new Tensor([5])

console.log(await a.add(b).tolist())
