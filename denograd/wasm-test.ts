import { Device } from './device.ts'
import { Tensor } from './mod.ts'

console.log(Device.DEFAULT)

const a = new Tensor([3.1, 2.3, 1.3, 4.4])
const b = true
console.log(await a.eq(b).tolist())
