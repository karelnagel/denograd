import { Device } from './device.ts'
import { Tensor } from './mod.ts'

console.log(Device.DEFAULT)

const a = new Tensor([10, 20, 30, 40, 50, 60, 70, 80])
const b = new Tensor([1, 2, 3, 4, 5, 6, 7, 8])

const res = await a.add(b).tolist()

console.log(res)
