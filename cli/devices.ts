import { colored, Device, env, is_eq, Tensor } from '../jsgrad/node.ts'

let result: string[] = [], error = false
for (const device in env.DEVICES) {
  if (['DISK', 'CLOUD'].includes(device)) continue
  let res
  try {
    await new env.DEVICES[device](device).init()
    const test = await new Tensor([1, 2, 3], { device }).mul(2).tolist()
    if (!is_eq(test, [2, 4, 6])) {
      throw new Error(`got ${test} instead of [2, 4, 6]`)
    }
    res = colored('PASS', 'green')
  } catch (e: any) {
    console.error(`${device} error:`)
    console.error(e)
    res = `${colored('FAIL', 'red')} - ${e.message}`
    error = true
  }
  result.push(
    `${device === Device.DEFAULT ? '*' : ' '} ${device.padEnd(10)}: ${res.slice(0, 120)}`,
  )
}
console.log(result.join('\n'))
if (error) env.exit(1)
