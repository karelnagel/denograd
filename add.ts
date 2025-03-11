import { Tensor } from './denograd/mod.ts'

const a = new Tensor([3, 3, 3, 3, 3])
const b = new Tensor([3, 4, 4, 4, 4])
console.log(await a.add(b).tolist())
