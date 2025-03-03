import { DEVICES } from '../denograd/device.ts'
import { colored, Device, is_eq, Tensor } from '../denograd/mod.ts'
for (const device in DEVICES) {
  if (device === 'DISK') continue
  let result
  try {
    await new DEVICES[device](device).init()
    const test = await new Tensor([1, 2, 3], { device }).mul(2).tolist()
    if (!is_eq(test, [2, 4, 6])) throw new Error(`got ${test} instead of [2, 4, 6]`)
    result = colored('PASS', 'green')
  } catch (e: any) {
    console.log(e)
    result = `${colored('FAIL', 'red')} - ${e.message}`
  }
  console.log(`${device === Device.DEFAULT ? '*' : ' '} ${device.padEnd(10)}: ${result}`)
}
