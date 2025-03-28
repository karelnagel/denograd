import { Tensor } from '@jsgrad/jsgrad'

const a = new Tensor([1, 2, 3, 4, 5])
const b = new Tensor([6, 7, 8, 9, 10])

alert(await a.add(b).tolist())
