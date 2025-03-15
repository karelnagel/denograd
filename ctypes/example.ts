import * as c from './dawn.ts'

const descriptor = c.InstanceDescriptor.new({
  features: c.InstanceFeatures.new({ timedWaitAnyEnable: c.U32.new(1), timedWaitAnyMaxCount: c.U64.new(64n) }),
})
descriptor.$features.$timedWaitAnyEnable.set(1)
const instance = c.createInstance(descriptor.ptr())
const wait = (future: c.Future) => {
  const res = c.instanceWaitAny(instance, c.Size.new(1n), c.FutureWaitInfo.new({ future }).ptr(), c.U64.new(2n ** 64n - 1n))
  if (res.value !== c.WaitStatus.Success.value) throw new Error('Future failed')
}
const from_wgpu_string = (ptr: c.StringView): string => {
  throw new Error()
}
const to_wgpu_string = (str: string): c.StringView => {
  throw new Error()
}

const feats = new c.InstanceFeatures()
const f2 = c.getInstanceFeatures(feats.ptr())
console.log(feats)
console.log(f2)
console.log(descriptor)
console.log(instance)

// const init = () => {
//   let adapter_result: [c.RequestAdapterStatus, c.Adapter, string] | undefined

//   const cb_info = new c.RequestAdapterCallbackInfo(
//     new c.Pointer(null),
//     c.CallbackMode.WaitAnyOnly,
//     new c.RequestAdapterCallback(
//       new Deno.UnsafeCallback({ parameters: ['u32', 'pointer', 'pointer', 'pointer'], result: 'void' }, (status, adapter, msg, _) => {
//         console.log('hello', status, adapter, msg)
//         adapter_result = [new c.RequestAdapterStatus(status), new c.Adapter(adapter), from_wgpu_string(new c.StringView(msg))]
//       }).pointer,
//     ),
//     new c.Pointer(null),
//   )

//   wait(c.instanceRequestAdapterF(instance, new c.RequestAdapterOptions(new c.Pointer(null), new c.Surface(null), c.FeatureLevel.Undefined, c.PowerPreference.HighPerformance, c.BackendType.Null, new c.Bool(0), new c.Bool(0)).ptr(), cb_info))

//   if (adapter_result![0].value !== c.RequestAdapterStatus.Success.value) {
//     throw new Error(`Error requesting adapter: [${adapter_result![0].value}] ${adapter_result![2]}`)
//   }

//   const supported_features = new c.SupportedFeatures(new c.Size(0n), c.FeatureName.ShaderF16.ptr())
//   const supported_features_ptr = supported_features.ptr()
//   c.adapterGetFeatures(adapter_result![1], supported_features_ptr)
//   supported_features_ptr.load(supported_features)
//   const supported: any[] = []
//   for (let i = 0; i < supported_features.items[0].value; i++) {
//     supported.push(supported_features.items[1][i])
//   }
//   const features = [c.FeatureName.TimestampQuery, c.FeatureName.ShaderF16].filter((feat) => supported.includes(feat.value))
//   const dev_desc = new c.DeviceDescriptor(new c.Pointer(null), to_wgpu_string(''), new c.Size(BigInt(features.length)), c.FeatureName.ShaderF16.ptr(), new c.Pointer(null), new c.QueueDescriptor(new c.Pointer(null), to_wgpu_string('')), new c.Pointer(null), new c.Pointer(null))

//   const supported_limits = new c.SupportedLimits(new c.Pointer(null), new c.Pointer(null))
//   c.adapterGetLimits(adapter_result![1], supported_limits.ptr())
//   const limits = new c.RequiredLimits(new c.Pointer(null), supported_limits.items[1])
//   dev_desc.items[4] = limits.ptr()

//   // Requesting a device
//   let device_result: [c.RequestDeviceStatus, c.Device, string] | undefined

//   // const dev_cb=(status, device_impl, msg, _)=> device_result[:] = status, device_impl, from_wgpu_str(msg)

//   const cb_info2 = new c.RequestDeviceCallbackInfo(
//     new c.Pointer(null),
//     c.CallbackMode.WaitAnyOnly,
//     new c.RequestDeviceCallback(
//       new Deno.UnsafeCallback({ parameters: ['u32', 'pointer', 'pointer', 'pointer'], result: 'void' }, (status, device_impl, msg, _) => {
//         console.log('hi')
//         device_result = [new c.RequestDeviceStatus(status), new c.Device(device_impl), from_wgpu_string(new c.StringView(msg))]
//       }).pointer,
//     ),
//     new c.Pointer(null),
//   )
//   wait(c.adapterRequestDeviceF(adapter_result![1], dev_desc.ptr(), cb_info2))

//   if (device_result![0].value !== c.RequestDeviceStatus.Success.value) {
//     throw new Error(`Failed to request device: [${device_result![0].value}] ${device_result![2]}`)
//   }
//   const device = device_result![1]
//   console.log(`We successfully got device!!!:${device}`)
// }
// init()
