import { Device, DeviceType } from '../src/device.ts'
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
  compare(
    [[]],
    () => Device.get_available_devices(),
    'out([*tiny.device.Device.get_available_devices()])',
  ),
)

Deno.test(
  'Device.DEFAULT',
  compare(
    [[]],
    () => Device.DEFAULT,
    'out(tiny.device.Device.DEFAULT)',
  ),
)
