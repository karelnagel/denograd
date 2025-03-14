import * as c from './mod.ts'

export class WGPUSType extends c.Enum<c.U64> {
  static ShaderSourceSPIRV = new WGPUSType(new c.U64(1n))
  static ShaderSourceWGSL = new WGPUSType(new c.U64(2n))
}
export class WGPUStatus extends c.I32 {
  static Success = new WGPUStatus(0x00000001)
  static Error = new WGPUStatus(0x00000002)
  static Force32 = new WGPUStatus(0x7FFFFFFF)
}

export class WGPUChainedStruct extends c.Struct<[next: c.Pointer<WGPUChainedStruct>, sType: WGPUSType]> {}
export class WGPUInstanceFeatures extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, timedWaitAnyEnable: c.U32, timedWaitAnyMaxCount: c.U64]> {}
export class WGPUInstanceDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, features: WGPUInstanceFeatures]> {}
type WGPUInstance = any

const lib = Deno.dlopen('/opt/homebrew/Cellar/dawn/0.1.6/lib/libwebgpu_dawn.dylib', {
  wgpuCreateInstance: {
    parameters: ['pointer'],
    result: 'pointer',
  },
  wgpuGetInstanceFeatures: {
    parameters: ['pointer'],
    result: 'i32',
  },
})

export const createInstance = (descriptor: c.Pointer<WGPUInstanceDescriptor>): c.Pointer<WGPUInstance> => {
  return new c.Pointer(lib.symbols.wgpuCreateInstance(descriptor.value))
}
export const getInstanceFeatures = (features: c.Pointer<WGPUInstanceFeatures>): WGPUStatus => {
  return new WGPUStatus(lib.symbols.wgpuGetInstanceFeatures(features.value))
}
const desc = new WGPUInstanceDescriptor(
  new c.Pointer(null),
  new WGPUInstanceFeatures(new c.Pointer(null), new c.U32(1), new c.U64(100n)),
)
const instance = createInstance(desc.ptr())

const feats = new WGPUInstanceFeatures(new c.Pointer(null), new c.U32(9), new c.U64(0n))
const ptr = feats.ptr()
const features = getInstanceFeatures(ptr)
const f2 = getInstanceFeatures(ptr)
console.log(ptr.load(feats), features, f2)
