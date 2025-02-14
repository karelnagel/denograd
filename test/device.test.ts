import { expect } from 'expect/expect'
import { Device, type DeviceType } from '../denograd/device.ts'
import { compare, test } from './helpers.ts'

test(
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

test(
  'Device.get_available_devices',
  () => {
    const devices = Device.get_available_devices()
    expect(devices).toEqual(['CLANG', 'WEBGPU', 'WASM', 'PYTHON'])
  },
)

test(
  'Device.DEFAULT',
  compare(
    [[]],
    () => Device.DEFAULT,
    'out(tiny.device.Device.DEFAULT)',
  ),
)
