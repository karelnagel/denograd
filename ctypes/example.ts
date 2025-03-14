import * as c from './mod.ts'
import * as dawn from './dawn.ts'

const descriptor = new dawn.InstanceDescriptor(new c.Pointer(null), new dawn.InstanceFeatures(new c.Pointer(null), new c.U32(1), new c.U64(10n)))
const instance = dawn.createInstance(descriptor.ptr())

const wait = (future: dawn.Future) => {
  const res = dawn.instanceWaitAny(instance, new c.Size(1n), new dawn.FutureWaitInfo(future, new dawn.Bool(0)).ptr(), new c.U64(2n ** 64n - 1n))
  if (res.value !== dawn.WaitStatus.Success.value) throw new Error('Future failed')
}
const from_wgpu_string = (ptr: dawn.StringView): string => {
}
const to_wgpu_string = (str: string): dawn.StringView => {
}

// const feats = new dawn.InstanceFeatures(new c.Pointer(null), new c.U32(9), new c.U64(0n))
// const ptr = feats.ptr()
// const features = dawn.getInstanceFeatures(ptr)
// const f2 = dawn.getInstanceFeatures(ptr)
// console.log(ptr.load(feats), features, f2)

const init = () => {
  let adapter_result: [dawn.RequestAdapterStatus, dawn.Adapter, string] | undefined

  const cb_info = new dawn.RequestAdapterCallbackInfo(
    new c.Pointer(null),
    dawn.CallbackMode.WaitAnyOnly,
    new dawn.RequestAdapterCallback(
      new Deno.UnsafeCallback({ parameters: ['u32', 'pointer', 'pointer', 'pointer'], result: 'void' }, (status, adapter, msg, _) => {
        console.log('hello', status, adapter, msg)
        adapter_result = [new dawn.RequestAdapterStatus(status), new dawn.Adapter(adapter), from_wgpu_string(new dawn.StringView(msg))]
      }).pointer,
    ),
    new c.Pointer(null),
  )

  wait(dawn.instanceRequestAdapterF(instance, new dawn.RequestAdapterOptions(new c.Pointer(null), new dawn.Surface(null), dawn.FeatureLevel.Undefined, dawn.PowerPreference.HighPerformance, dawn.BackendType.Null, new dawn.Bool(0), new dawn.Bool(0)).ptr(), cb_info))

  if (adapter_result![0].value !== dawn.RequestAdapterStatus.Success.value) {
    throw new Error(`Error requesting adapter: [${adapter_result![0].value}] ${adapter_result![2]}`)
  }

  const supported_features = new dawn.SupportedFeatures(new c.Size(0n), dawn.FeatureName.ShaderF16.ptr())
  const supported_features_ptr = supported_features.ptr()
  dawn.adapterGetFeatures(adapter_result![1], supported_features_ptr)
  supported_features_ptr.load(supported_features)
  const supported: any[] = []
  for (let i = 0; i < supported_features.items[0].value; i++) {
    supported.push(supported_features.items[1][i])
  }
  const features = [dawn.FeatureName.TimestampQuery, dawn.FeatureName.ShaderF16].filter((feat) => supported.includes(feat.value))
  const dev_desc = new dawn.DeviceDescriptor(new c.Pointer(null), to_wgpu_string(''), new c.Size(BigInt(features.length)), dawn.FeatureName.ShaderF16.ptr(), new c.Pointer(null), new dawn.QueueDescriptor(new c.Pointer(null), to_wgpu_string('')), new c.Pointer(null), new c.Pointer(null))

  const supported_limits = new dawn.SupportedLimits(new c.Pointer(null), new c.Pointer(null))
  dawn.adapterGetLimits(adapter_result![1], supported_limits.ptr())
  const limits = new dawn.RequiredLimits(new c.Pointer(null), supported_limits.items[1])
  dev_desc.items[4] = limits.ptr()

  // Requesting a device
  let device_result: [dawn.RequestDeviceStatus, dawn.Device, string] | undefined

  // const dev_cb=(status, device_impl, msg, _)=> device_result[:] = status, device_impl, from_wgpu_str(msg)

  const cb_info2 = new dawn.RequestDeviceCallbackInfo(
    new c.Pointer(null),
    dawn.CallbackMode.WaitAnyOnly,
    new dawn.RequestDeviceCallback(
      new Deno.UnsafeCallback({ parameters: ['u32', 'pointer', 'pointer', 'pointer'], result: 'void' }, (status, device_impl, msg, _) => {
        console.log('hi')
        device_result = [new dawn.RequestDeviceStatus(status), new dawn.Device(device_impl), from_wgpu_string(new dawn.StringView(msg))]
      }).pointer,
    ),
    new c.Pointer(null),
  )
  wait(dawn.adapterRequestDeviceF(adapter_result![1], dev_desc.ptr(), cb_info2))

  if (device_result![0].value !== dawn.RequestDeviceStatus.Success.value) {
    throw new Error(`Failed to request device: [${device_result![0].value}] ${device_result![2]}`)
  }
  const device = device_result![1]
  console.log(`We successfully got device!!!:${device}`)
}
init()
