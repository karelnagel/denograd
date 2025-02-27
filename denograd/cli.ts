import { type AllDevices, Device } from './device.ts'
import { colored, is_eq } from './helpers.ts'
import { Tensor } from './tensor.ts'

if (import.meta.main) {
  for (const device in Device.DEVICES) {
    let result
    try {
      const test = await new Tensor([1, 2, 3], { device: device as AllDevices }).mul(2).tolist()
      if (!is_eq(test, [2, 4, 6])) throw new Error(`got ${test} instead of [2, 4, 6]`)
      result = colored('PASS', 'green')
    } catch (e: any) {
      result = `${colored('FAIL', 'red')} - ${e.message}`
    }
    console.log(`${device === Device.DEFAULT ? '*' : ' '} ${device.padEnd(10)}: ${result}`)
  }
}
