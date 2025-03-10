import { Tensor } from './denograd/mod.ts'

const a = new Tensor([3, 3, 4,5,5,6,4,77,75,2,5,5,5,5,5,5,5,5,5,5,5,5,5])
console.log(await a.add(4.4).tolist())
