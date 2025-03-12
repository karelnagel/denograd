import { Tensor } from './denograd/mod.ts'

const a = Tensor.rand([2])
const b = Tensor.rand([2])

await a.add(b).tolist()
