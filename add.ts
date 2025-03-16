import { Tensor } from './denograd/mod.ts'

const a = new Tensor([4])
const b = new Tensor([43])
console.log(await a.add(b).tolist())
