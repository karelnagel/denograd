import { expect } from 'expect/expect'
import { Device, DeviceType } from '../denograd/device.ts'
import { compare } from './helpers.ts'

Deno.test(
  'Device.get',
  compare(
    [
      ['PYTHON'],
      //   ['CLANG'],
      ['DISK'],
    ],
    (device: DeviceType) => Device.get(device).device,
    'out(tiny.device.Device[data[0]].device)',
  ),
)

Deno.test(
  'Device.get_available_devices',
  () => {
    const devices = Device.get_available_devices()
    expect(devices).toEqual(['CLANG', 'WEBGPU', 'PYTHON'])
  },
)

Deno.test(
  'Device.DEFAULT',
  compare(
    [[]],
    () => Device.DEFAULT,
    'out(tiny.device.Device.DEFAULT)',
  ),
)
