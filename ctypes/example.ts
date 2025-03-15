import * as c from './dawn.ts'

const desc = new c.InstanceDescriptor()
desc.$features.$timedWaitAnyEnable.set(1)

const instance = c.createInstance(desc.ptr())
if (!instance.value) throw new Error(`Failed creating instance!`)
const wgpu_wait = (future: c.Future) => {
  const res = c.instanceWaitAny(instance, c.Size.new(1n), c.FutureWaitInfo.new({ future }).ptr(), c.U64.new(2n ** 64n - 1n))
  if (res.value !== c.WaitStatus.Success.value) throw new Error('Future failed')
}
const from_wgpu_str = (_str: c.StringView): string => {
  return _str.$data.value ? Deno.UnsafePointerView.getCString(_str.$data.native as Deno.PointerObject) : ''
}
const to_wgpu_str = (str: string): c.StringView => {
  throw new Error()
}

const init = () => {
  let adapter_result: [c.RequestAdapterStatus, c.Adapter, string] | undefined

  const cb_info = new c.RequestAdapterCallbackInfo()
  cb_info.$mode.set(c.CallbackMode.WaitAnyOnly.value)
  cb_info.$callback.set((status, adapter, msg) => {
    adapter_result = [status, adapter, from_wgpu_str(msg)]
  })
  wgpu_wait(c.instanceRequestAdapterF(instance, c.RequestAdapterOptions.new({ powerPreference: c.PowerPreference.HighPerformance }).ptr(), cb_info))

  if (!adapter_result || adapter_result![0].value !== c.RequestAdapterStatus.Success.value) {
    throw new Error(`Error requesting adapter: [${adapter_result![0].value}] ${adapter_result![2]}`)
  }
  console.log(`Adapter: ${adapter_result[1]}`)

  const supported_features = new c.SupportedFeatures()
  c.adapterGetFeatures(adapter_result![1], supported_features.ptr())
  const supported: c.FeatureName[] = []
  for (let i = 0n; i < supported_features.$featureCount.value; i++) {
    supported.push(new c.FeatureName().loadFromPtr(c.Pointer.new(supported_features.$features.value + i)))
  }
  const features = [c.FeatureName.TimestampQuery, c.FeatureName.ShaderF16].filter((feat) => supported.some((s) => s.value === feat.value))

  // TODO: do array correcly
  const len = features.reduce((acc, x) => acc + x.byteLength, 0)
  const featArr = new c.Type(new ArrayBuffer(len), 0, len, len)
  let offset = 0
  for (const feat of features) {
    featArr.bytes.set(feat.bytes, offset)
    offset += feat.byteLength
  }
  const dev_desc = c.DeviceDescriptor.new({ requiredFeatureCount: c.Size.new(BigInt(features.length)), requiredFeatures: featArr.ptr() })

  const supported_limits = new c.SupportedLimits()
  c.adapterGetLimits(adapter_result![1], supported_limits.ptr())
  const limits = c.RequiredLimits.new({ limits: supported_limits.$limits })
  dev_desc.$requiredLimits.set(limits.ptr().value)

  // Requesting a device
  let device_result: [c.RequestDeviceStatus, c.Device, string] | undefined

  const cb_info2 = new c.RequestDeviceCallbackInfo()
  cb_info2.$mode.set(c.CallbackMode.WaitAnyOnly.value)
  cb_info2.$callback.set((status, device_impl, msg, _) => {
    device_result = [status, device_impl, from_wgpu_str(msg)]
  })
  wgpu_wait(c.adapterRequestDeviceF(adapter_result![1], dev_desc.ptr(), cb_info2))

  if (!device_result || device_result[0].value !== c.RequestDeviceStatus.Success.value) {
    throw new Error(`Failed to request device: [${device_result![0].value}] ${device_result![2]}`)
  }
  const device = device_result[1]
  console.log(`Device: ${device}`)
}
init()
