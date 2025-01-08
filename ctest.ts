import { Tensor } from './src/tensor.ts'

const a = new Tensor([2, 4, 5, 5, 5, 5, 5])
const b = new Tensor([2])

console.log(await a.mul(b).tolist())
