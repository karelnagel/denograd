import * as c from './mod.ts'

export class WGPUSType extends c.Enum<c.U64> {
  static ShaderSourceSPIRV = new WGPUSType(new c.U64(1n))
  static ShaderSourceWGSL = new WGPUSType(new c.U64(2n))
}

export class WGPUStatus extends c.Type<number> {
  static Success = new WGPUStatus(0x00000001)
  static Error = new WGPUStatus(0x00000002)
  static Force32 = new WGPUStatus(0x7FFFFFFF)
  get buffer() {
    return new c.U32(this.value).buffer
  }
  override fromBuffer(buf: ArrayBuffer): this {
    const res = new c.U32(this.value).fromBuffer(buf)
    this._value = res._value
    return this
  }
}

export class WGPUChainedStruct extends c.Struct<[next: c.Pointer<WGPUChainedStruct>, sType: WGPUSType]> {}
export class WGPUInstanceFeatures extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, timedWaitAnyEnable: c.U32, timedWaitAnyMaxCount: c.U64]> {}
export class WGPUInstanceDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, features: WGPUInstanceFeatures]> {}

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

type WGPUInstance = any
export const createInstance = (descriptor: c.Pointer<WGPUInstanceDescriptor>): c.Pointer<WGPUInstance> => {
  return new c.Pointer(lib.symbols.wgpuCreateInstance(descriptor.value))
}
export const getInstanceFeatures = (features: c.Pointer<WGPUInstanceFeatures>): WGPUStatus => {
  return new WGPUStatus(lib.symbols.wgpuGetInstanceFeatures(features.value))
}

const instance = createInstance(new WGPUInstanceDescriptor(
  new c.Pointer(null),
  new WGPUInstanceFeatures(new c.Pointer(null), new c.U32(0), new c.U64(1n)),
).ptr())

const feats = new WGPUInstanceFeatures(new c.Pointer(null), new c.U32(9), new c.U64(0n))
console.log(feats)
const len = feats.buffer.byteLength
const ptr = feats.ptr()
console.log(new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(ptr.value, len, 0)))
const features = getInstanceFeatures(ptr)
console.log(features)
console.log(new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(ptr.value, len, 0)))
console.log(ptr.load(feats))
