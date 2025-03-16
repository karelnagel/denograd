import * as c from './ctypes.ts'
export * from './ctypes.ts'
import os from "node:os"
import process from "node:process";

const FILE = process.platform==="darwin" ? 'libwebgpu_dawn.dylib':"libwebgpu_dawn.so"
const URL = `https://github.com/wpmed92/pydawn/releases/download/v0.1.6/${FILE}`
const PATH = `${os.homedir()}/.cache/${FILE}`

const stat = await Deno.stat(PATH).catch(() => undefined)
if (!stat) {
  console.log(`Downloading ${FILE} to ${PATH}`)
  const res = await fetch(URL).then((x) => x.arrayBuffer())
  await Deno.writeFile(PATH, new Uint8Array(res))
  console.log("done")
}

const lib = Deno.dlopen(PATH, {
  wgpuAdapterInfoFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuAdapterPropertiesMemoryHeapsFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuCreateInstance: { parameters: ['pointer'], result: 'pointer' },
  wgpuDrmFormatCapabilitiesFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuGetInstanceFeatures: { parameters: ['pointer'], result: 'u32' },
  wgpuGetProcAddress: { parameters: ['buffer'], result: 'function' },
  wgpuSharedBufferMemoryEndAccessStateFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuSharedTextureMemoryEndAccessStateFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuSupportedFeaturesFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuSurfaceCapabilitiesFreeMembers: { parameters: ['buffer'], result: 'void' },
  wgpuAdapterCreateDevice: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuAdapterGetFeatures: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuAdapterGetFormatCapabilities: { parameters: ['pointer', 'u32', 'pointer'], result: 'u32' },
  wgpuAdapterGetInfo: { parameters: ['pointer', 'pointer'], result: 'u32' },
  wgpuAdapterGetInstance: { parameters: ['pointer'], result: 'pointer' },
  wgpuAdapterGetLimits: { parameters: ['pointer', 'pointer'], result: 'u32' },
  wgpuAdapterHasFeature: { parameters: ['pointer', 'u32'], result: 'u32' },
  wgpuAdapterRequestDevice: { parameters: ['pointer', 'pointer', 'function', 'pointer'], result: 'void' },
  wgpuAdapterRequestDevice2: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuAdapterRequestDeviceF: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuAdapterAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuAdapterRelease: { parameters: ['pointer'], result: 'void' },
  wgpuBindGroupSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuBindGroupAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuBindGroupRelease: { parameters: ['pointer'], result: 'void' },
  wgpuBindGroupLayoutSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuBindGroupLayoutAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuBindGroupLayoutRelease: { parameters: ['pointer'], result: 'void' },
  wgpuBufferDestroy: { parameters: ['pointer'], result: 'void' },
  wgpuBufferGetConstMappedRange: { parameters: ['pointer', 'usize', 'usize'], result: 'pointer' },
  wgpuBufferGetMapState: { parameters: ['pointer'], result: 'u32' },
  wgpuBufferGetMappedRange: { parameters: ['pointer', 'usize', 'usize'], result: 'pointer' },
  wgpuBufferGetSize: { parameters: ['pointer'], result: 'u64' },
  wgpuBufferGetUsage: { parameters: ['pointer'], result: 'u64' },
  wgpuBufferMapAsync: { parameters: ['pointer', 'u64', 'usize', 'usize', 'function', 'pointer'], result: 'void' },
  wgpuBufferMapAsync2: { parameters: ['pointer', 'u64', 'usize', 'usize', 'buffer'], result: 'buffer' },
  wgpuBufferMapAsyncF: { parameters: ['pointer', 'u64', 'usize', 'usize', 'buffer'], result: 'buffer' },
  wgpuBufferSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuBufferUnmap: { parameters: ['pointer'], result: 'void' },
  wgpuBufferAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuBufferRelease: { parameters: ['pointer'], result: 'void' },
  wgpuCommandBufferSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuCommandBufferAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuCommandBufferRelease: { parameters: ['pointer'], result: 'void' },
  wgpuCommandEncoderBeginComputePass: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuCommandEncoderBeginRenderPass: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuCommandEncoderClearBuffer: { parameters: ['pointer', 'pointer', 'u64', 'u64'], result: 'void' },
  wgpuCommandEncoderCopyBufferToBuffer: { parameters: ['pointer', 'pointer', 'u64', 'pointer', 'u64', 'u64'], result: 'void' },
  wgpuCommandEncoderCopyBufferToTexture: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void' },
  wgpuCommandEncoderCopyTextureToBuffer: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void' },
  wgpuCommandEncoderCopyTextureToTexture: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void' },
  wgpuCommandEncoderFinish: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuCommandEncoderInjectValidationError: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuCommandEncoderInsertDebugMarker: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuCommandEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void' },
  wgpuCommandEncoderPushDebugGroup: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuCommandEncoderResolveQuerySet: { parameters: ['pointer', 'pointer', 'u32', 'u32', 'pointer', 'u64'], result: 'void' },
  wgpuCommandEncoderSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuCommandEncoderWriteBuffer: { parameters: ['pointer', 'pointer', 'u64', 'pointer', 'u64'], result: 'void' },
  wgpuCommandEncoderWriteTimestamp: { parameters: ['pointer', 'pointer', 'u32'], result: 'void' },
  wgpuCommandEncoderAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuCommandEncoderRelease: { parameters: ['pointer'], result: 'void' },
  wgpuComputePassEncoderDispatchWorkgroups: { parameters: ['pointer', 'u32', 'u32', 'u32'], result: 'void' },
  wgpuComputePassEncoderDispatchWorkgroupsIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void' },
  wgpuComputePassEncoderEnd: { parameters: ['pointer'], result: 'void' },
  wgpuComputePassEncoderInsertDebugMarker: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuComputePassEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void' },
  wgpuComputePassEncoderPushDebugGroup: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuComputePassEncoderSetBindGroup: { parameters: ['pointer', 'u32', 'pointer', 'usize', 'pointer'], result: 'void' },
  wgpuComputePassEncoderSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuComputePassEncoderSetPipeline: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuComputePassEncoderWriteTimestamp: { parameters: ['pointer', 'pointer', 'u32'], result: 'void' },
  wgpuComputePassEncoderAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuComputePassEncoderRelease: { parameters: ['pointer'], result: 'void' },
  wgpuComputePipelineGetBindGroupLayout: { parameters: ['pointer', 'u32'], result: 'pointer' },
  wgpuComputePipelineSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuComputePipelineAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuComputePipelineRelease: { parameters: ['pointer'], result: 'void' },
  wgpuDeviceCreateBindGroup: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateBindGroupLayout: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateBuffer: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateCommandEncoder: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateComputePipeline: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateComputePipelineAsync: { parameters: ['pointer', 'pointer', 'function', 'pointer'], result: 'void' },
  wgpuDeviceCreateComputePipelineAsync2: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuDeviceCreateComputePipelineAsyncF: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuDeviceCreateErrorBuffer: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateErrorExternalTexture: { parameters: ['pointer'], result: 'pointer' },
  wgpuDeviceCreateErrorShaderModule: { parameters: ['pointer', 'pointer', 'buffer'], result: 'pointer' },
  wgpuDeviceCreateErrorTexture: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateExternalTexture: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreatePipelineLayout: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateQuerySet: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateRenderBundleEncoder: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateRenderPipeline: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateRenderPipelineAsync: { parameters: ['pointer', 'pointer', 'function', 'pointer'], result: 'void' },
  wgpuDeviceCreateRenderPipelineAsync2: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuDeviceCreateRenderPipelineAsyncF: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuDeviceCreateSampler: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateShaderModule: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceCreateTexture: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceDestroy: { parameters: ['pointer'], result: 'void' },
  wgpuDeviceForceLoss: { parameters: ['pointer', 'u32', 'buffer'], result: 'void' },
  wgpuDeviceGetAHardwareBufferProperties: { parameters: ['pointer', 'pointer', 'pointer'], result: 'u32' },
  wgpuDeviceGetAdapter: { parameters: ['pointer'], result: 'pointer' },
  wgpuDeviceGetAdapterInfo: { parameters: ['pointer', 'pointer'], result: 'u32' },
  wgpuDeviceGetFeatures: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuDeviceGetLimits: { parameters: ['pointer', 'pointer'], result: 'u32' },
  wgpuDeviceGetLostFuture: { parameters: ['pointer'], result: 'buffer' },
  wgpuDeviceGetQueue: { parameters: ['pointer'], result: 'pointer' },
  wgpuDeviceHasFeature: { parameters: ['pointer', 'u32'], result: 'u32' },
  wgpuDeviceImportSharedBufferMemory: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceImportSharedFence: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceImportSharedTextureMemory: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuDeviceInjectError: { parameters: ['pointer', 'u32', 'buffer'], result: 'void' },
  wgpuDevicePopErrorScope: { parameters: ['pointer', 'function', 'pointer'], result: 'void' },
  wgpuDevicePopErrorScope2: { parameters: ['pointer', 'buffer'], result: 'buffer' },
  wgpuDevicePopErrorScopeF: { parameters: ['pointer', 'buffer'], result: 'buffer' },
  wgpuDevicePushErrorScope: { parameters: ['pointer', 'u32'], result: 'void' },
  wgpuDeviceSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuDeviceSetLoggingCallback: { parameters: ['pointer', 'function', 'pointer'], result: 'void' },
  wgpuDeviceTick: { parameters: ['pointer'], result: 'void' },
  wgpuDeviceValidateTextureDescriptor: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuDeviceAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuDeviceRelease: { parameters: ['pointer'], result: 'void' },
  wgpuExternalTextureDestroy: { parameters: ['pointer'], result: 'void' },
  wgpuExternalTextureExpire: { parameters: ['pointer'], result: 'void' },
  wgpuExternalTextureRefresh: { parameters: ['pointer'], result: 'void' },
  wgpuExternalTextureSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuExternalTextureAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuExternalTextureRelease: { parameters: ['pointer'], result: 'void' },
  wgpuInstanceCreateSurface: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuInstanceEnumerateWGSLLanguageFeatures: { parameters: ['pointer', 'pointer'], result: 'usize' },
  wgpuInstanceHasWGSLLanguageFeature: { parameters: ['pointer', 'u32'], result: 'u32' },
  wgpuInstanceProcessEvents: { parameters: ['pointer'], result: 'void' },
  wgpuInstanceRequestAdapter: { parameters: ['pointer', 'pointer', 'function', 'pointer'], result: 'void' },
  wgpuInstanceRequestAdapter2: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuInstanceRequestAdapterF: { parameters: ['pointer', 'pointer', 'buffer'], result: 'buffer' },
  wgpuInstanceWaitAny: { parameters: ['pointer', 'usize', 'pointer', 'u64'], result: 'u32' },
  wgpuInstanceAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuInstanceRelease: { parameters: ['pointer'], result: 'void' },
  wgpuPipelineLayoutSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuPipelineLayoutAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuPipelineLayoutRelease: { parameters: ['pointer'], result: 'void' },
  wgpuQuerySetDestroy: { parameters: ['pointer'], result: 'void' },
  wgpuQuerySetGetCount: { parameters: ['pointer'], result: 'u32' },
  wgpuQuerySetGetType: { parameters: ['pointer'], result: 'u32' },
  wgpuQuerySetSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuQuerySetAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuQuerySetRelease: { parameters: ['pointer'], result: 'void' },
  wgpuQueueCopyExternalTextureForBrowser: { parameters: ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], result: 'void' },
  wgpuQueueCopyTextureForBrowser: { parameters: ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], result: 'void' },
  wgpuQueueOnSubmittedWorkDone: { parameters: ['pointer', 'function', 'pointer'], result: 'void' },
  wgpuQueueOnSubmittedWorkDone2: { parameters: ['pointer', 'buffer'], result: 'buffer' },
  wgpuQueueOnSubmittedWorkDoneF: { parameters: ['pointer', 'buffer'], result: 'buffer' },
  wgpuQueueSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuQueueSubmit: { parameters: ['pointer', 'usize', 'pointer'], result: 'void' },
  wgpuQueueWriteBuffer: { parameters: ['pointer', 'pointer', 'u64', 'pointer', 'usize'], result: 'void' },
  wgpuQueueWriteTexture: { parameters: ['pointer', 'pointer', 'pointer', 'usize', 'pointer', 'pointer'], result: 'void' },
  wgpuQueueAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuQueueRelease: { parameters: ['pointer'], result: 'void' },
  wgpuRenderBundleSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderBundleAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuRenderBundleRelease: { parameters: ['pointer'], result: 'void' },
  wgpuRenderBundleEncoderDraw: { parameters: ['pointer', 'u32', 'u32', 'u32', 'u32'], result: 'void' },
  wgpuRenderBundleEncoderDrawIndexed: { parameters: ['pointer', 'u32', 'u32', 'u32', 'i32', 'u32'], result: 'void' },
  wgpuRenderBundleEncoderDrawIndexedIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void' },
  wgpuRenderBundleEncoderDrawIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void' },
  wgpuRenderBundleEncoderFinish: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuRenderBundleEncoderInsertDebugMarker: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderBundleEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void' },
  wgpuRenderBundleEncoderPushDebugGroup: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderBundleEncoderSetBindGroup: { parameters: ['pointer', 'u32', 'pointer', 'usize', 'pointer'], result: 'void' },
  wgpuRenderBundleEncoderSetIndexBuffer: { parameters: ['pointer', 'pointer', 'u32', 'u64', 'u64'], result: 'void' },
  wgpuRenderBundleEncoderSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderBundleEncoderSetPipeline: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuRenderBundleEncoderSetVertexBuffer: { parameters: ['pointer', 'u32', 'pointer', 'u64', 'u64'], result: 'void' },
  wgpuRenderBundleEncoderAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuRenderBundleEncoderRelease: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPassEncoderBeginOcclusionQuery: { parameters: ['pointer', 'u32'], result: 'void' },
  wgpuRenderPassEncoderDraw: { parameters: ['pointer', 'u32', 'u32', 'u32', 'u32'], result: 'void' },
  wgpuRenderPassEncoderDrawIndexed: { parameters: ['pointer', 'u32', 'u32', 'u32', 'i32', 'u32'], result: 'void' },
  wgpuRenderPassEncoderDrawIndexedIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void' },
  wgpuRenderPassEncoderDrawIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void' },
  wgpuRenderPassEncoderEnd: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPassEncoderEndOcclusionQuery: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPassEncoderExecuteBundles: { parameters: ['pointer', 'usize', 'pointer'], result: 'void' },
  wgpuRenderPassEncoderInsertDebugMarker: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderPassEncoderMultiDrawIndexedIndirect: { parameters: ['pointer', 'pointer', 'u64', 'u32', 'pointer', 'u64'], result: 'void' },
  wgpuRenderPassEncoderMultiDrawIndirect: { parameters: ['pointer', 'pointer', 'u64', 'u32', 'pointer', 'u64'], result: 'void' },
  wgpuRenderPassEncoderPixelLocalStorageBarrier: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPassEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPassEncoderPushDebugGroup: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderPassEncoderSetBindGroup: { parameters: ['pointer', 'u32', 'pointer', 'usize', 'pointer'], result: 'void' },
  wgpuRenderPassEncoderSetBlendConstant: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuRenderPassEncoderSetIndexBuffer: { parameters: ['pointer', 'pointer', 'u32', 'u64', 'u64'], result: 'void' },
  wgpuRenderPassEncoderSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderPassEncoderSetPipeline: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuRenderPassEncoderSetScissorRect: { parameters: ['pointer', 'u32', 'u32', 'u32', 'u32'], result: 'void' },
  wgpuRenderPassEncoderSetStencilReference: { parameters: ['pointer', 'u32'], result: 'void' },
  wgpuRenderPassEncoderSetVertexBuffer: { parameters: ['pointer', 'u32', 'pointer', 'u64', 'u64'], result: 'void' },
  wgpuRenderPassEncoderSetViewport: { parameters: ['pointer', 'f32', 'f32', 'f32', 'f32', 'f32', 'f32'], result: 'void' },
  wgpuRenderPassEncoderWriteTimestamp: { parameters: ['pointer', 'pointer', 'u32'], result: 'void' },
  wgpuRenderPassEncoderAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPassEncoderRelease: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPipelineGetBindGroupLayout: { parameters: ['pointer', 'u32'], result: 'pointer' },
  wgpuRenderPipelineSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuRenderPipelineAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuRenderPipelineRelease: { parameters: ['pointer'], result: 'void' },
  wgpuSamplerSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuSamplerAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuSamplerRelease: { parameters: ['pointer'], result: 'void' },
  wgpuShaderModuleGetCompilationInfo: { parameters: ['pointer', 'function', 'pointer'], result: 'void' },
  wgpuShaderModuleGetCompilationInfo2: { parameters: ['pointer', 'buffer'], result: 'buffer' },
  wgpuShaderModuleGetCompilationInfoF: { parameters: ['pointer', 'buffer'], result: 'buffer' },
  wgpuShaderModuleSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuShaderModuleAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuShaderModuleRelease: { parameters: ['pointer'], result: 'void' },
  wgpuSharedBufferMemoryBeginAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'u32' },
  wgpuSharedBufferMemoryCreateBuffer: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuSharedBufferMemoryEndAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'u32' },
  wgpuSharedBufferMemoryGetProperties: { parameters: ['pointer', 'pointer'], result: 'u32' },
  wgpuSharedBufferMemoryIsDeviceLost: { parameters: ['pointer'], result: 'u32' },
  wgpuSharedBufferMemorySetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuSharedBufferMemoryAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuSharedBufferMemoryRelease: { parameters: ['pointer'], result: 'void' },
  wgpuSharedFenceExportInfo: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuSharedFenceAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuSharedFenceRelease: { parameters: ['pointer'], result: 'void' },
  wgpuSharedTextureMemoryBeginAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'u32' },
  wgpuSharedTextureMemoryCreateTexture: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuSharedTextureMemoryEndAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'u32' },
  wgpuSharedTextureMemoryGetProperties: { parameters: ['pointer', 'pointer'], result: 'u32' },
  wgpuSharedTextureMemoryIsDeviceLost: { parameters: ['pointer'], result: 'u32' },
  wgpuSharedTextureMemorySetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuSharedTextureMemoryAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuSharedTextureMemoryRelease: { parameters: ['pointer'], result: 'void' },
  wgpuSurfaceConfigure: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuSurfaceGetCapabilities: { parameters: ['pointer', 'pointer', 'pointer'], result: 'u32' },
  wgpuSurfaceGetCurrentTexture: { parameters: ['pointer', 'pointer'], result: 'void' },
  wgpuSurfacePresent: { parameters: ['pointer'], result: 'void' },
  wgpuSurfaceSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuSurfaceUnconfigure: { parameters: ['pointer'], result: 'void' },
  wgpuSurfaceAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuSurfaceRelease: { parameters: ['pointer'], result: 'void' },
  wgpuTextureCreateErrorView: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuTextureCreateView: { parameters: ['pointer', 'pointer'], result: 'pointer' },
  wgpuTextureDestroy: { parameters: ['pointer'], result: 'void' },
  wgpuTextureGetDepthOrArrayLayers: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureGetDimension: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureGetFormat: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureGetHeight: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureGetMipLevelCount: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureGetSampleCount: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureGetUsage: { parameters: ['pointer'], result: 'u64' },
  wgpuTextureGetWidth: { parameters: ['pointer'], result: 'u32' },
  wgpuTextureSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuTextureAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuTextureRelease: { parameters: ['pointer'], result: 'void' },
  wgpuTextureViewSetLabel: { parameters: ['pointer', 'buffer'], result: 'void' },
  wgpuTextureViewAddRef: { parameters: ['pointer'], result: 'void' },
  wgpuTextureViewRelease: { parameters: ['pointer'], result: 'void' },
})

// consts
export const BufferUsage_None = c.U64.new(0n)
export const BufferUsage_MapRead = c.U64.new(1n)
export const BufferUsage_MapWrite = c.U64.new(2n)
export const BufferUsage_CopySrc = c.U64.new(4n)
export const BufferUsage_CopyDst = c.U64.new(8n)
export const BufferUsage_Index = c.U64.new(16n)
export const BufferUsage_Vertex = c.U64.new(32n)
export const BufferUsage_Uniform = c.U64.new(64n)
export const BufferUsage_Storage = c.U64.new(128n)
export const BufferUsage_Indirect = c.U64.new(256n)
export const BufferUsage_QueryResolve = c.U64.new(512n)
export const ColorWriteMask_None = c.U64.new(0n)
export const ColorWriteMask_Red = c.U64.new(1n)
export const ColorWriteMask_Green = c.U64.new(2n)
export const ColorWriteMask_Blue = c.U64.new(4n)
export const ColorWriteMask_Alpha = c.U64.new(8n)
export const ColorWriteMask_All = c.U64.new(15n)
export const HeapProperty_DeviceLocal = c.U64.new(1n)
export const HeapProperty_HostVisible = c.U64.new(2n)
export const HeapProperty_HostCoherent = c.U64.new(4n)
export const HeapProperty_HostUncached = c.U64.new(8n)
export const HeapProperty_HostCached = c.U64.new(16n)
export const MapMode_None = c.U64.new(0n)
export const MapMode_Read = c.U64.new(1n)
export const MapMode_Write = c.U64.new(2n)
export const ShaderStage_None = c.U64.new(0n)
export const ShaderStage_Vertex = c.U64.new(1n)
export const ShaderStage_Fragment = c.U64.new(2n)
export const ShaderStage_Compute = c.U64.new(4n)
export const TextureUsage_None = c.U64.new(0n)
export const TextureUsage_CopySrc = c.U64.new(1n)
export const TextureUsage_CopyDst = c.U64.new(2n)
export const TextureUsage_TextureBinding = c.U64.new(4n)
export const TextureUsage_StorageBinding = c.U64.new(8n)
export const TextureUsage_RenderAttachment = c.U64.new(16n)
export const TextureUsage_TransientAttachment = c.U64.new(32n)
export const TextureUsage_StorageAttachment = c.U64.new(64n)

// enums
export class WGSLFeatureName extends c.U32 {
  static 'ReadonlyAndReadwriteStorageTextures' = WGSLFeatureName.new(1)
  static 'Packed4x8IntegerDotProduct' = WGSLFeatureName.new(2)
  static 'UnrestrictedPointerParameters' = WGSLFeatureName.new(3)
  static 'PointerCompositeAccess' = WGSLFeatureName.new(4)
  static 'ChromiumTestingUnimplemented' = WGSLFeatureName.new(327680)
  static 'ChromiumTestingUnsafeExperimental' = WGSLFeatureName.new(327681)
  static 'ChromiumTestingExperimental' = WGSLFeatureName.new(327682)
  static 'ChromiumTestingShippedWithKillswitch' = WGSLFeatureName.new(327683)
  static 'ChromiumTestingShipped' = WGSLFeatureName.new(327684)
  static 'Force32' = WGSLFeatureName.new(2147483647)
}
export class AdapterType extends c.U32 {
  static 'DiscreteGPU' = AdapterType.new(1)
  static 'IntegratedGPU' = AdapterType.new(2)
  static 'CPU' = AdapterType.new(3)
  static 'Unknown' = AdapterType.new(4)
  static 'Force32' = AdapterType.new(2147483647)
}
export class AddressMode extends c.U32 {
  static 'Undefined' = AddressMode.new(0)
  static 'ClampToEdge' = AddressMode.new(1)
  static 'Repeat' = AddressMode.new(2)
  static 'MirrorRepeat' = AddressMode.new(3)
  static 'Force32' = AddressMode.new(2147483647)
}
export class AlphaMode extends c.U32 {
  static 'Opaque' = AlphaMode.new(1)
  static 'Premultiplied' = AlphaMode.new(2)
  static 'Unpremultiplied' = AlphaMode.new(3)
  static 'Force32' = AlphaMode.new(2147483647)
}
export class BackendType extends c.U32 {
  static 'Undefined' = BackendType.new(0)
  static 'Null' = BackendType.new(1)
  static 'WebGPU' = BackendType.new(2)
  static 'D3D11' = BackendType.new(3)
  static 'D3D12' = BackendType.new(4)
  static 'Metal' = BackendType.new(5)
  static 'Vulkan' = BackendType.new(6)
  static 'OpenGL' = BackendType.new(7)
  static 'OpenGLES' = BackendType.new(8)
  static 'Force32' = BackendType.new(2147483647)
}
export class BlendFactor extends c.U32 {
  static 'Undefined' = BlendFactor.new(0)
  static 'Zero' = BlendFactor.new(1)
  static 'One' = BlendFactor.new(2)
  static 'Src' = BlendFactor.new(3)
  static 'OneMinusSrc' = BlendFactor.new(4)
  static 'SrcAlpha' = BlendFactor.new(5)
  static 'OneMinusSrcAlpha' = BlendFactor.new(6)
  static 'Dst' = BlendFactor.new(7)
  static 'OneMinusDst' = BlendFactor.new(8)
  static 'DstAlpha' = BlendFactor.new(9)
  static 'OneMinusDstAlpha' = BlendFactor.new(10)
  static 'SrcAlphaSaturated' = BlendFactor.new(11)
  static 'Constant' = BlendFactor.new(12)
  static 'OneMinusConstant' = BlendFactor.new(13)
  static 'Src1' = BlendFactor.new(14)
  static 'OneMinusSrc1' = BlendFactor.new(15)
  static 'Src1Alpha' = BlendFactor.new(16)
  static 'OneMinusSrc1Alpha' = BlendFactor.new(17)
  static 'Force32' = BlendFactor.new(2147483647)
}
export class BlendOperation extends c.U32 {
  static 'Undefined' = BlendOperation.new(0)
  static 'Add' = BlendOperation.new(1)
  static 'Subtract' = BlendOperation.new(2)
  static 'ReverseSubtract' = BlendOperation.new(3)
  static 'Min' = BlendOperation.new(4)
  static 'Max' = BlendOperation.new(5)
  static 'Force32' = BlendOperation.new(2147483647)
}
export class BufferBindingType extends c.U32 {
  static 'BindingNotUsed' = BufferBindingType.new(0)
  static 'Uniform' = BufferBindingType.new(1)
  static 'Storage' = BufferBindingType.new(2)
  static 'ReadOnlyStorage' = BufferBindingType.new(3)
  static 'Force32' = BufferBindingType.new(2147483647)
}
export class BufferMapAsyncStatus extends c.U32 {
  static 'Success' = BufferMapAsyncStatus.new(1)
  static 'InstanceDropped' = BufferMapAsyncStatus.new(2)
  static 'ValidationError' = BufferMapAsyncStatus.new(3)
  static 'Unknown' = BufferMapAsyncStatus.new(4)
  static 'DeviceLost' = BufferMapAsyncStatus.new(5)
  static 'DestroyedBeforeCallback' = BufferMapAsyncStatus.new(6)
  static 'UnmappedBeforeCallback' = BufferMapAsyncStatus.new(7)
  static 'MappingAlreadyPending' = BufferMapAsyncStatus.new(8)
  static 'OffsetOutOfRange' = BufferMapAsyncStatus.new(9)
  static 'SizeOutOfRange' = BufferMapAsyncStatus.new(10)
  static 'Force32' = BufferMapAsyncStatus.new(2147483647)
}
export class BufferMapState extends c.U32 {
  static 'Unmapped' = BufferMapState.new(1)
  static 'Pending' = BufferMapState.new(2)
  static 'Mapped' = BufferMapState.new(3)
  static 'Force32' = BufferMapState.new(2147483647)
}
export class CallbackMode extends c.U32 {
  static 'WaitAnyOnly' = CallbackMode.new(1)
  static 'AllowProcessEvents' = CallbackMode.new(2)
  static 'AllowSpontaneous' = CallbackMode.new(3)
  static 'Force32' = CallbackMode.new(2147483647)
}
export class CompareFunction extends c.U32 {
  static 'Undefined' = CompareFunction.new(0)
  static 'Never' = CompareFunction.new(1)
  static 'Less' = CompareFunction.new(2)
  static 'Equal' = CompareFunction.new(3)
  static 'LessEqual' = CompareFunction.new(4)
  static 'Greater' = CompareFunction.new(5)
  static 'NotEqual' = CompareFunction.new(6)
  static 'GreaterEqual' = CompareFunction.new(7)
  static 'Always' = CompareFunction.new(8)
  static 'Force32' = CompareFunction.new(2147483647)
}
export class CompilationInfoRequestStatus extends c.U32 {
  static 'Success' = CompilationInfoRequestStatus.new(1)
  static 'InstanceDropped' = CompilationInfoRequestStatus.new(2)
  static 'Error' = CompilationInfoRequestStatus.new(3)
  static 'DeviceLost' = CompilationInfoRequestStatus.new(4)
  static 'Unknown' = CompilationInfoRequestStatus.new(5)
  static 'Force32' = CompilationInfoRequestStatus.new(2147483647)
}
export class CompilationMessageType extends c.U32 {
  static 'Error' = CompilationMessageType.new(1)
  static 'Warning' = CompilationMessageType.new(2)
  static 'Info' = CompilationMessageType.new(3)
  static 'Force32' = CompilationMessageType.new(2147483647)
}
export class CompositeAlphaMode extends c.U32 {
  static 'Auto' = CompositeAlphaMode.new(0)
  static 'Opaque' = CompositeAlphaMode.new(1)
  static 'Premultiplied' = CompositeAlphaMode.new(2)
  static 'Unpremultiplied' = CompositeAlphaMode.new(3)
  static 'Inherit' = CompositeAlphaMode.new(4)
  static 'Force32' = CompositeAlphaMode.new(2147483647)
}
export class CreatePipelineAsyncStatus extends c.U32 {
  static 'Success' = CreatePipelineAsyncStatus.new(1)
  static 'InstanceDropped' = CreatePipelineAsyncStatus.new(2)
  static 'ValidationError' = CreatePipelineAsyncStatus.new(3)
  static 'InternalError' = CreatePipelineAsyncStatus.new(4)
  static 'DeviceLost' = CreatePipelineAsyncStatus.new(5)
  static 'DeviceDestroyed' = CreatePipelineAsyncStatus.new(6)
  static 'Unknown' = CreatePipelineAsyncStatus.new(7)
  static 'Force32' = CreatePipelineAsyncStatus.new(2147483647)
}
export class CullMode extends c.U32 {
  static 'Undefined' = CullMode.new(0)
  static 'None' = CullMode.new(1)
  static 'Front' = CullMode.new(2)
  static 'Back' = CullMode.new(3)
  static 'Force32' = CullMode.new(2147483647)
}
export class DeviceLostReason extends c.U32 {
  static 'Unknown' = DeviceLostReason.new(1)
  static 'Destroyed' = DeviceLostReason.new(2)
  static 'InstanceDropped' = DeviceLostReason.new(3)
  static 'FailedCreation' = DeviceLostReason.new(4)
  static 'Force32' = DeviceLostReason.new(2147483647)
}
export class ErrorFilter extends c.U32 {
  static 'Validation' = ErrorFilter.new(1)
  static 'OutOfMemory' = ErrorFilter.new(2)
  static 'Internal' = ErrorFilter.new(3)
  static 'Force32' = ErrorFilter.new(2147483647)
}
export class ErrorType extends c.U32 {
  static 'NoError' = ErrorType.new(1)
  static 'Validation' = ErrorType.new(2)
  static 'OutOfMemory' = ErrorType.new(3)
  static 'Internal' = ErrorType.new(4)
  static 'Unknown' = ErrorType.new(5)
  static 'DeviceLost' = ErrorType.new(6)
  static 'Force32' = ErrorType.new(2147483647)
}
export class ExternalTextureRotation extends c.U32 {
  static 'Rotate0Degrees' = ExternalTextureRotation.new(1)
  static 'Rotate90Degrees' = ExternalTextureRotation.new(2)
  static 'Rotate180Degrees' = ExternalTextureRotation.new(3)
  static 'Rotate270Degrees' = ExternalTextureRotation.new(4)
  static 'Force32' = ExternalTextureRotation.new(2147483647)
}
export class FeatureLevel extends c.U32 {
  static 'Undefined' = FeatureLevel.new(0)
  static 'Compatibility' = FeatureLevel.new(1)
  static 'Core' = FeatureLevel.new(2)
  static 'Force32' = FeatureLevel.new(2147483647)
}
export class FeatureName extends c.U32 {
  static 'DepthClipControl' = FeatureName.new(1)
  static 'Depth32FloatStencil8' = FeatureName.new(2)
  static 'TimestampQuery' = FeatureName.new(3)
  static 'TextureCompressionBC' = FeatureName.new(4)
  static 'TextureCompressionETC2' = FeatureName.new(5)
  static 'TextureCompressionASTC' = FeatureName.new(6)
  static 'IndirectFirstInstance' = FeatureName.new(7)
  static 'ShaderF16' = FeatureName.new(8)
  static 'RG11B10UfloatRenderable' = FeatureName.new(9)
  static 'BGRA8UnormStorage' = FeatureName.new(10)
  static 'Float32Filterable' = FeatureName.new(11)
  static 'Float32Blendable' = FeatureName.new(12)
  static 'Subgroups' = FeatureName.new(13)
  static 'SubgroupsF16' = FeatureName.new(14)
  static 'DawnInternalUsages' = FeatureName.new(327680)
  static 'DawnMultiPlanarFormats' = FeatureName.new(327681)
  static 'DawnNative' = FeatureName.new(327682)
  static 'ChromiumExperimentalTimestampQueryInsidePasses' = FeatureName.new(327683)
  static 'ImplicitDeviceSynchronization' = FeatureName.new(327684)
  static 'ChromiumExperimentalImmediateData' = FeatureName.new(327685)
  static 'TransientAttachments' = FeatureName.new(327686)
  static 'MSAARenderToSingleSampled' = FeatureName.new(327687)
  static 'DualSourceBlending' = FeatureName.new(327688)
  static 'D3D11MultithreadProtected' = FeatureName.new(327689)
  static 'ANGLETextureSharing' = FeatureName.new(327690)
  static 'PixelLocalStorageCoherent' = FeatureName.new(327691)
  static 'PixelLocalStorageNonCoherent' = FeatureName.new(327692)
  static 'Unorm16TextureFormats' = FeatureName.new(327693)
  static 'Snorm16TextureFormats' = FeatureName.new(327694)
  static 'MultiPlanarFormatExtendedUsages' = FeatureName.new(327695)
  static 'MultiPlanarFormatP010' = FeatureName.new(327696)
  static 'HostMappedPointer' = FeatureName.new(327697)
  static 'MultiPlanarRenderTargets' = FeatureName.new(327698)
  static 'MultiPlanarFormatNv12a' = FeatureName.new(327699)
  static 'FramebufferFetch' = FeatureName.new(327700)
  static 'BufferMapExtendedUsages' = FeatureName.new(327701)
  static 'AdapterPropertiesMemoryHeaps' = FeatureName.new(327702)
  static 'AdapterPropertiesD3D' = FeatureName.new(327703)
  static 'AdapterPropertiesVk' = FeatureName.new(327704)
  static 'R8UnormStorage' = FeatureName.new(327705)
  static 'FormatCapabilities' = FeatureName.new(327706)
  static 'DrmFormatCapabilities' = FeatureName.new(327707)
  static 'Norm16TextureFormats' = FeatureName.new(327708)
  static 'MultiPlanarFormatNv16' = FeatureName.new(327709)
  static 'MultiPlanarFormatNv24' = FeatureName.new(327710)
  static 'MultiPlanarFormatP210' = FeatureName.new(327711)
  static 'MultiPlanarFormatP410' = FeatureName.new(327712)
  static 'SharedTextureMemoryVkDedicatedAllocation' = FeatureName.new(327713)
  static 'SharedTextureMemoryAHardwareBuffer' = FeatureName.new(327714)
  static 'SharedTextureMemoryDmaBuf' = FeatureName.new(327715)
  static 'SharedTextureMemoryOpaqueFD' = FeatureName.new(327716)
  static 'SharedTextureMemoryZirconHandle' = FeatureName.new(327717)
  static 'SharedTextureMemoryDXGISharedHandle' = FeatureName.new(327718)
  static 'SharedTextureMemoryD3D11Texture2D' = FeatureName.new(327719)
  static 'SharedTextureMemoryIOSurface' = FeatureName.new(327720)
  static 'SharedTextureMemoryEGLImage' = FeatureName.new(327721)
  static 'SharedFenceVkSemaphoreOpaqueFD' = FeatureName.new(327722)
  static 'SharedFenceSyncFD' = FeatureName.new(327723)
  static 'SharedFenceVkSemaphoreZirconHandle' = FeatureName.new(327724)
  static 'SharedFenceDXGISharedHandle' = FeatureName.new(327725)
  static 'SharedFenceMTLSharedEvent' = FeatureName.new(327726)
  static 'SharedBufferMemoryD3D12Resource' = FeatureName.new(327727)
  static 'StaticSamplers' = FeatureName.new(327728)
  static 'YCbCrVulkanSamplers' = FeatureName.new(327729)
  static 'ShaderModuleCompilationOptions' = FeatureName.new(327730)
  static 'DawnLoadResolveTexture' = FeatureName.new(327731)
  static 'DawnPartialLoadResolveTexture' = FeatureName.new(327732)
  static 'MultiDrawIndirect' = FeatureName.new(327733)
  static 'ClipDistances' = FeatureName.new(327734)
  static 'DawnTexelCopyBufferRowAlignment' = FeatureName.new(327735)
  static 'FlexibleTextureViews' = FeatureName.new(327736)
  static 'Force32' = FeatureName.new(2147483647)
}
export class FilterMode extends c.U32 {
  static 'Undefined' = FilterMode.new(0)
  static 'Nearest' = FilterMode.new(1)
  static 'Linear' = FilterMode.new(2)
  static 'Force32' = FilterMode.new(2147483647)
}
export class FrontFace extends c.U32 {
  static 'Undefined' = FrontFace.new(0)
  static 'CCW' = FrontFace.new(1)
  static 'CW' = FrontFace.new(2)
  static 'Force32' = FrontFace.new(2147483647)
}
export class IndexFormat extends c.U32 {
  static 'Undefined' = IndexFormat.new(0)
  static 'Uint16' = IndexFormat.new(1)
  static 'Uint32' = IndexFormat.new(2)
  static 'Force32' = IndexFormat.new(2147483647)
}
export class LoadOp extends c.U32 {
  static 'Undefined' = LoadOp.new(0)
  static 'Load' = LoadOp.new(1)
  static 'Clear' = LoadOp.new(2)
  static 'ExpandResolveTexture' = LoadOp.new(327683)
  static 'Force32' = LoadOp.new(2147483647)
}
export class LoggingType extends c.U32 {
  static 'Verbose' = LoggingType.new(1)
  static 'Info' = LoggingType.new(2)
  static 'Warning' = LoggingType.new(3)
  static 'Error' = LoggingType.new(4)
  static 'Force32' = LoggingType.new(2147483647)
}
export class MapAsyncStatus extends c.U32 {
  static 'Success' = MapAsyncStatus.new(1)
  static 'InstanceDropped' = MapAsyncStatus.new(2)
  static 'Error' = MapAsyncStatus.new(3)
  static 'Aborted' = MapAsyncStatus.new(4)
  static 'Unknown' = MapAsyncStatus.new(5)
  static 'Force32' = MapAsyncStatus.new(2147483647)
}
export class MipmapFilterMode extends c.U32 {
  static 'Undefined' = MipmapFilterMode.new(0)
  static 'Nearest' = MipmapFilterMode.new(1)
  static 'Linear' = MipmapFilterMode.new(2)
  static 'Force32' = MipmapFilterMode.new(2147483647)
}
export class OptionalBool extends c.U32 {
  static 'False' = OptionalBool.new(0)
  static 'True' = OptionalBool.new(1)
  static 'Undefined' = OptionalBool.new(2)
  static 'Force32' = OptionalBool.new(2147483647)
}
export class PopErrorScopeStatus extends c.U32 {
  static 'Success' = PopErrorScopeStatus.new(1)
  static 'InstanceDropped' = PopErrorScopeStatus.new(2)
  static 'Force32' = PopErrorScopeStatus.new(2147483647)
}
export class PowerPreference extends c.U32 {
  static 'Undefined' = PowerPreference.new(0)
  static 'LowPower' = PowerPreference.new(1)
  static 'HighPerformance' = PowerPreference.new(2)
  static 'Force32' = PowerPreference.new(2147483647)
}
export class PresentMode extends c.U32 {
  static 'Fifo' = PresentMode.new(1)
  static 'FifoRelaxed' = PresentMode.new(2)
  static 'Immediate' = PresentMode.new(3)
  static 'Mailbox' = PresentMode.new(4)
  static 'Force32' = PresentMode.new(2147483647)
}
export class PrimitiveTopology extends c.U32 {
  static 'Undefined' = PrimitiveTopology.new(0)
  static 'PointList' = PrimitiveTopology.new(1)
  static 'LineList' = PrimitiveTopology.new(2)
  static 'LineStrip' = PrimitiveTopology.new(3)
  static 'TriangleList' = PrimitiveTopology.new(4)
  static 'TriangleStrip' = PrimitiveTopology.new(5)
  static 'Force32' = PrimitiveTopology.new(2147483647)
}
export class QueryType extends c.U32 {
  static 'Occlusion' = QueryType.new(1)
  static 'Timestamp' = QueryType.new(2)
  static 'Force32' = QueryType.new(2147483647)
}
export class QueueWorkDoneStatus extends c.U32 {
  static 'Success' = QueueWorkDoneStatus.new(1)
  static 'InstanceDropped' = QueueWorkDoneStatus.new(2)
  static 'Error' = QueueWorkDoneStatus.new(3)
  static 'Unknown' = QueueWorkDoneStatus.new(4)
  static 'DeviceLost' = QueueWorkDoneStatus.new(5)
  static 'Force32' = QueueWorkDoneStatus.new(2147483647)
}
export class RequestAdapterStatus extends c.U32 {
  static 'Success' = RequestAdapterStatus.new(1)
  static 'InstanceDropped' = RequestAdapterStatus.new(2)
  static 'Unavailable' = RequestAdapterStatus.new(3)
  static 'Error' = RequestAdapterStatus.new(4)
  static 'Unknown' = RequestAdapterStatus.new(5)
  static 'Force32' = RequestAdapterStatus.new(2147483647)
}
export class RequestDeviceStatus extends c.U32 {
  static 'Success' = RequestDeviceStatus.new(1)
  static 'InstanceDropped' = RequestDeviceStatus.new(2)
  static 'Error' = RequestDeviceStatus.new(3)
  static 'Unknown' = RequestDeviceStatus.new(4)
  static 'Force32' = RequestDeviceStatus.new(2147483647)
}
export class SType extends c.U32 {
  static 'ShaderSourceSPIRV' = SType.new(1)
  static 'ShaderSourceWGSL' = SType.new(2)
  static 'RenderPassMaxDrawCount' = SType.new(3)
  static 'SurfaceSourceMetalLayer' = SType.new(4)
  static 'SurfaceSourceWindowsHWND' = SType.new(5)
  static 'SurfaceSourceXlibWindow' = SType.new(6)
  static 'SurfaceSourceWaylandSurface' = SType.new(7)
  static 'SurfaceSourceAndroidNativeWindow' = SType.new(8)
  static 'SurfaceSourceXCBWindow' = SType.new(9)
  static 'AdapterPropertiesSubgroups' = SType.new(10)
  static 'TextureBindingViewDimensionDescriptor' = SType.new(131072)
  static 'SurfaceSourceCanvasHTMLSelector_Emscripten' = SType.new(262144)
  static 'SurfaceDescriptorFromWindowsCoreWindow' = SType.new(327680)
  static 'ExternalTextureBindingEntry' = SType.new(327681)
  static 'ExternalTextureBindingLayout' = SType.new(327682)
  static 'SurfaceDescriptorFromWindowsSwapChainPanel' = SType.new(327683)
  static 'DawnTextureInternalUsageDescriptor' = SType.new(327684)
  static 'DawnEncoderInternalUsageDescriptor' = SType.new(327685)
  static 'DawnInstanceDescriptor' = SType.new(327686)
  static 'DawnCacheDeviceDescriptor' = SType.new(327687)
  static 'DawnAdapterPropertiesPowerPreference' = SType.new(327688)
  static 'DawnBufferDescriptorErrorInfoFromWireClient' = SType.new(327689)
  static 'DawnTogglesDescriptor' = SType.new(327690)
  static 'DawnShaderModuleSPIRVOptionsDescriptor' = SType.new(327691)
  static 'RequestAdapterOptionsLUID' = SType.new(327692)
  static 'RequestAdapterOptionsGetGLProc' = SType.new(327693)
  static 'RequestAdapterOptionsD3D11Device' = SType.new(327694)
  static 'DawnRenderPassColorAttachmentRenderToSingleSampled' = SType.new(327695)
  static 'RenderPassPixelLocalStorage' = SType.new(327696)
  static 'PipelineLayoutPixelLocalStorage' = SType.new(327697)
  static 'BufferHostMappedPointer' = SType.new(327698)
  static 'DawnExperimentalSubgroupLimits' = SType.new(327699)
  static 'AdapterPropertiesMemoryHeaps' = SType.new(327700)
  static 'AdapterPropertiesD3D' = SType.new(327701)
  static 'AdapterPropertiesVk' = SType.new(327702)
  static 'DawnWireWGSLControl' = SType.new(327703)
  static 'DawnWGSLBlocklist' = SType.new(327704)
  static 'DrmFormatCapabilities' = SType.new(327705)
  static 'ShaderModuleCompilationOptions' = SType.new(327706)
  static 'ColorTargetStateExpandResolveTextureDawn' = SType.new(327707)
  static 'RenderPassDescriptorExpandResolveRect' = SType.new(327708)
  static 'SharedTextureMemoryVkDedicatedAllocationDescriptor' = SType.new(327709)
  static 'SharedTextureMemoryAHardwareBufferDescriptor' = SType.new(327710)
  static 'SharedTextureMemoryDmaBufDescriptor' = SType.new(327711)
  static 'SharedTextureMemoryOpaqueFDDescriptor' = SType.new(327712)
  static 'SharedTextureMemoryZirconHandleDescriptor' = SType.new(327713)
  static 'SharedTextureMemoryDXGISharedHandleDescriptor' = SType.new(327714)
  static 'SharedTextureMemoryD3D11Texture2DDescriptor' = SType.new(327715)
  static 'SharedTextureMemoryIOSurfaceDescriptor' = SType.new(327716)
  static 'SharedTextureMemoryEGLImageDescriptor' = SType.new(327717)
  static 'SharedTextureMemoryInitializedBeginState' = SType.new(327718)
  static 'SharedTextureMemoryInitializedEndState' = SType.new(327719)
  static 'SharedTextureMemoryVkImageLayoutBeginState' = SType.new(327720)
  static 'SharedTextureMemoryVkImageLayoutEndState' = SType.new(327721)
  static 'SharedTextureMemoryD3DSwapchainBeginState' = SType.new(327722)
  static 'SharedFenceVkSemaphoreOpaqueFDDescriptor' = SType.new(327723)
  static 'SharedFenceVkSemaphoreOpaqueFDExportInfo' = SType.new(327724)
  static 'SharedFenceSyncFDDescriptor' = SType.new(327725)
  static 'SharedFenceSyncFDExportInfo' = SType.new(327726)
  static 'SharedFenceVkSemaphoreZirconHandleDescriptor' = SType.new(327727)
  static 'SharedFenceVkSemaphoreZirconHandleExportInfo' = SType.new(327728)
  static 'SharedFenceDXGISharedHandleDescriptor' = SType.new(327729)
  static 'SharedFenceDXGISharedHandleExportInfo' = SType.new(327730)
  static 'SharedFenceMTLSharedEventDescriptor' = SType.new(327731)
  static 'SharedFenceMTLSharedEventExportInfo' = SType.new(327732)
  static 'SharedBufferMemoryD3D12ResourceDescriptor' = SType.new(327733)
  static 'StaticSamplerBindingLayout' = SType.new(327734)
  static 'YCbCrVkDescriptor' = SType.new(327735)
  static 'SharedTextureMemoryAHardwareBufferProperties' = SType.new(327736)
  static 'AHardwareBufferProperties' = SType.new(327737)
  static 'DawnExperimentalImmediateDataLimits' = SType.new(327738)
  static 'DawnTexelCopyBufferRowAlignmentLimits' = SType.new(327739)
  static 'Force32' = SType.new(2147483647)
}
export class SamplerBindingType extends c.U32 {
  static 'BindingNotUsed' = SamplerBindingType.new(0)
  static 'Filtering' = SamplerBindingType.new(1)
  static 'NonFiltering' = SamplerBindingType.new(2)
  static 'Comparison' = SamplerBindingType.new(3)
  static 'Force32' = SamplerBindingType.new(2147483647)
}
export class SharedFenceType extends c.U32 {
  static 'VkSemaphoreOpaqueFD' = SharedFenceType.new(1)
  static 'SyncFD' = SharedFenceType.new(2)
  static 'VkSemaphoreZirconHandle' = SharedFenceType.new(3)
  static 'DXGISharedHandle' = SharedFenceType.new(4)
  static 'MTLSharedEvent' = SharedFenceType.new(5)
  static 'Force32' = SharedFenceType.new(2147483647)
}
export class Status extends c.U32 {
  static 'Success' = Status.new(1)
  static 'Error' = Status.new(2)
  static 'Force32' = Status.new(2147483647)
}
export class StencilOperation extends c.U32 {
  static 'Undefined' = StencilOperation.new(0)
  static 'Keep' = StencilOperation.new(1)
  static 'Zero' = StencilOperation.new(2)
  static 'Replace' = StencilOperation.new(3)
  static 'Invert' = StencilOperation.new(4)
  static 'IncrementClamp' = StencilOperation.new(5)
  static 'DecrementClamp' = StencilOperation.new(6)
  static 'IncrementWrap' = StencilOperation.new(7)
  static 'DecrementWrap' = StencilOperation.new(8)
  static 'Force32' = StencilOperation.new(2147483647)
}
export class StorageTextureAccess extends c.U32 {
  static 'BindingNotUsed' = StorageTextureAccess.new(0)
  static 'WriteOnly' = StorageTextureAccess.new(1)
  static 'ReadOnly' = StorageTextureAccess.new(2)
  static 'ReadWrite' = StorageTextureAccess.new(3)
  static 'Force32' = StorageTextureAccess.new(2147483647)
}
export class StoreOp extends c.U32 {
  static 'Undefined' = StoreOp.new(0)
  static 'Store' = StoreOp.new(1)
  static 'Discard' = StoreOp.new(2)
  static 'Force32' = StoreOp.new(2147483647)
}
export class SurfaceGetCurrentTextureStatus extends c.U32 {
  static 'Success' = SurfaceGetCurrentTextureStatus.new(1)
  static 'Timeout' = SurfaceGetCurrentTextureStatus.new(2)
  static 'Outdated' = SurfaceGetCurrentTextureStatus.new(3)
  static 'Lost' = SurfaceGetCurrentTextureStatus.new(4)
  static 'OutOfMemory' = SurfaceGetCurrentTextureStatus.new(5)
  static 'DeviceLost' = SurfaceGetCurrentTextureStatus.new(6)
  static 'Error' = SurfaceGetCurrentTextureStatus.new(7)
  static 'Force32' = SurfaceGetCurrentTextureStatus.new(2147483647)
}
export class TextureAspect extends c.U32 {
  static 'Undefined' = TextureAspect.new(0)
  static 'All' = TextureAspect.new(1)
  static 'StencilOnly' = TextureAspect.new(2)
  static 'DepthOnly' = TextureAspect.new(3)
  static 'Plane0Only' = TextureAspect.new(327680)
  static 'Plane1Only' = TextureAspect.new(327681)
  static 'Plane2Only' = TextureAspect.new(327682)
  static 'Force32' = TextureAspect.new(2147483647)
}
export class TextureDimension extends c.U32 {
  static 'Undefined' = TextureDimension.new(0)
  static '1D' = TextureDimension.new(1)
  static '2D' = TextureDimension.new(2)
  static '3D' = TextureDimension.new(3)
  static 'Force32' = TextureDimension.new(2147483647)
}
export class TextureFormat extends c.U32 {
  static 'Undefined' = TextureFormat.new(0)
  static 'R8Unorm' = TextureFormat.new(1)
  static 'R8Snorm' = TextureFormat.new(2)
  static 'R8Uint' = TextureFormat.new(3)
  static 'R8Sint' = TextureFormat.new(4)
  static 'R16Uint' = TextureFormat.new(5)
  static 'R16Sint' = TextureFormat.new(6)
  static 'R16Float' = TextureFormat.new(7)
  static 'RG8Unorm' = TextureFormat.new(8)
  static 'RG8Snorm' = TextureFormat.new(9)
  static 'RG8Uint' = TextureFormat.new(10)
  static 'RG8Sint' = TextureFormat.new(11)
  static 'R32Float' = TextureFormat.new(12)
  static 'R32Uint' = TextureFormat.new(13)
  static 'R32Sint' = TextureFormat.new(14)
  static 'RG16Uint' = TextureFormat.new(15)
  static 'RG16Sint' = TextureFormat.new(16)
  static 'RG16Float' = TextureFormat.new(17)
  static 'RGBA8Unorm' = TextureFormat.new(18)
  static 'RGBA8UnormSrgb' = TextureFormat.new(19)
  static 'RGBA8Snorm' = TextureFormat.new(20)
  static 'RGBA8Uint' = TextureFormat.new(21)
  static 'RGBA8Sint' = TextureFormat.new(22)
  static 'BGRA8Unorm' = TextureFormat.new(23)
  static 'BGRA8UnormSrgb' = TextureFormat.new(24)
  static 'RGB10A2Uint' = TextureFormat.new(25)
  static 'RGB10A2Unorm' = TextureFormat.new(26)
  static 'RG11B10Ufloat' = TextureFormat.new(27)
  static 'RGB9E5Ufloat' = TextureFormat.new(28)
  static 'RG32Float' = TextureFormat.new(29)
  static 'RG32Uint' = TextureFormat.new(30)
  static 'RG32Sint' = TextureFormat.new(31)
  static 'RGBA16Uint' = TextureFormat.new(32)
  static 'RGBA16Sint' = TextureFormat.new(33)
  static 'RGBA16Float' = TextureFormat.new(34)
  static 'RGBA32Float' = TextureFormat.new(35)
  static 'RGBA32Uint' = TextureFormat.new(36)
  static 'RGBA32Sint' = TextureFormat.new(37)
  static 'Stencil8' = TextureFormat.new(38)
  static 'Depth16Unorm' = TextureFormat.new(39)
  static 'Depth24Plus' = TextureFormat.new(40)
  static 'Depth24PlusStencil8' = TextureFormat.new(41)
  static 'Depth32Float' = TextureFormat.new(42)
  static 'Depth32FloatStencil8' = TextureFormat.new(43)
  static 'BC1RGBAUnorm' = TextureFormat.new(44)
  static 'BC1RGBAUnormSrgb' = TextureFormat.new(45)
  static 'BC2RGBAUnorm' = TextureFormat.new(46)
  static 'BC2RGBAUnormSrgb' = TextureFormat.new(47)
  static 'BC3RGBAUnorm' = TextureFormat.new(48)
  static 'BC3RGBAUnormSrgb' = TextureFormat.new(49)
  static 'BC4RUnorm' = TextureFormat.new(50)
  static 'BC4RSnorm' = TextureFormat.new(51)
  static 'BC5RGUnorm' = TextureFormat.new(52)
  static 'BC5RGSnorm' = TextureFormat.new(53)
  static 'BC6HRGBUfloat' = TextureFormat.new(54)
  static 'BC6HRGBFloat' = TextureFormat.new(55)
  static 'BC7RGBAUnorm' = TextureFormat.new(56)
  static 'BC7RGBAUnormSrgb' = TextureFormat.new(57)
  static 'ETC2RGB8Unorm' = TextureFormat.new(58)
  static 'ETC2RGB8UnormSrgb' = TextureFormat.new(59)
  static 'ETC2RGB8A1Unorm' = TextureFormat.new(60)
  static 'ETC2RGB8A1UnormSrgb' = TextureFormat.new(61)
  static 'ETC2RGBA8Unorm' = TextureFormat.new(62)
  static 'ETC2RGBA8UnormSrgb' = TextureFormat.new(63)
  static 'EACR11Unorm' = TextureFormat.new(64)
  static 'EACR11Snorm' = TextureFormat.new(65)
  static 'EACRG11Unorm' = TextureFormat.new(66)
  static 'EACRG11Snorm' = TextureFormat.new(67)
  static 'ASTC4x4Unorm' = TextureFormat.new(68)
  static 'ASTC4x4UnormSrgb' = TextureFormat.new(69)
  static 'ASTC5x4Unorm' = TextureFormat.new(70)
  static 'ASTC5x4UnormSrgb' = TextureFormat.new(71)
  static 'ASTC5x5Unorm' = TextureFormat.new(72)
  static 'ASTC5x5UnormSrgb' = TextureFormat.new(73)
  static 'ASTC6x5Unorm' = TextureFormat.new(74)
  static 'ASTC6x5UnormSrgb' = TextureFormat.new(75)
  static 'ASTC6x6Unorm' = TextureFormat.new(76)
  static 'ASTC6x6UnormSrgb' = TextureFormat.new(77)
  static 'ASTC8x5Unorm' = TextureFormat.new(78)
  static 'ASTC8x5UnormSrgb' = TextureFormat.new(79)
  static 'ASTC8x6Unorm' = TextureFormat.new(80)
  static 'ASTC8x6UnormSrgb' = TextureFormat.new(81)
  static 'ASTC8x8Unorm' = TextureFormat.new(82)
  static 'ASTC8x8UnormSrgb' = TextureFormat.new(83)
  static 'ASTC10x5Unorm' = TextureFormat.new(84)
  static 'ASTC10x5UnormSrgb' = TextureFormat.new(85)
  static 'ASTC10x6Unorm' = TextureFormat.new(86)
  static 'ASTC10x6UnormSrgb' = TextureFormat.new(87)
  static 'ASTC10x8Unorm' = TextureFormat.new(88)
  static 'ASTC10x8UnormSrgb' = TextureFormat.new(89)
  static 'ASTC10x10Unorm' = TextureFormat.new(90)
  static 'ASTC10x10UnormSrgb' = TextureFormat.new(91)
  static 'ASTC12x10Unorm' = TextureFormat.new(92)
  static 'ASTC12x10UnormSrgb' = TextureFormat.new(93)
  static 'ASTC12x12Unorm' = TextureFormat.new(94)
  static 'ASTC12x12UnormSrgb' = TextureFormat.new(95)
  static 'R16Unorm' = TextureFormat.new(327680)
  static 'RG16Unorm' = TextureFormat.new(327681)
  static 'RGBA16Unorm' = TextureFormat.new(327682)
  static 'R16Snorm' = TextureFormat.new(327683)
  static 'RG16Snorm' = TextureFormat.new(327684)
  static 'RGBA16Snorm' = TextureFormat.new(327685)
  static 'R8BG8Biplanar420Unorm' = TextureFormat.new(327686)
  static 'R10X6BG10X6Biplanar420Unorm' = TextureFormat.new(327687)
  static 'R8BG8A8Triplanar420Unorm' = TextureFormat.new(327688)
  static 'R8BG8Biplanar422Unorm' = TextureFormat.new(327689)
  static 'R8BG8Biplanar444Unorm' = TextureFormat.new(327690)
  static 'R10X6BG10X6Biplanar422Unorm' = TextureFormat.new(327691)
  static 'R10X6BG10X6Biplanar444Unorm' = TextureFormat.new(327692)
  static 'External' = TextureFormat.new(327693)
  static 'Force32' = TextureFormat.new(2147483647)
}
export class TextureSampleType extends c.U32 {
  static 'BindingNotUsed' = TextureSampleType.new(0)
  static 'Float' = TextureSampleType.new(1)
  static 'UnfilterableFloat' = TextureSampleType.new(2)
  static 'Depth' = TextureSampleType.new(3)
  static 'Sint' = TextureSampleType.new(4)
  static 'Uint' = TextureSampleType.new(5)
  static 'Force32' = TextureSampleType.new(2147483647)
}
export class TextureViewDimension extends c.U32 {
  static 'Undefined' = TextureViewDimension.new(0)
  static '1D' = TextureViewDimension.new(1)
  static '2D' = TextureViewDimension.new(2)
  static '2DArray' = TextureViewDimension.new(3)
  static 'Cube' = TextureViewDimension.new(4)
  static 'CubeArray' = TextureViewDimension.new(5)
  static '3D' = TextureViewDimension.new(6)
  static 'Force32' = TextureViewDimension.new(2147483647)
}
export class VertexFormat extends c.U32 {
  static 'Uint8' = VertexFormat.new(1)
  static 'Uint8x2' = VertexFormat.new(2)
  static 'Uint8x4' = VertexFormat.new(3)
  static 'Sint8' = VertexFormat.new(4)
  static 'Sint8x2' = VertexFormat.new(5)
  static 'Sint8x4' = VertexFormat.new(6)
  static 'Unorm8' = VertexFormat.new(7)
  static 'Unorm8x2' = VertexFormat.new(8)
  static 'Unorm8x4' = VertexFormat.new(9)
  static 'Snorm8' = VertexFormat.new(10)
  static 'Snorm8x2' = VertexFormat.new(11)
  static 'Snorm8x4' = VertexFormat.new(12)
  static 'Uint16' = VertexFormat.new(13)
  static 'Uint16x2' = VertexFormat.new(14)
  static 'Uint16x4' = VertexFormat.new(15)
  static 'Sint16' = VertexFormat.new(16)
  static 'Sint16x2' = VertexFormat.new(17)
  static 'Sint16x4' = VertexFormat.new(18)
  static 'Unorm16' = VertexFormat.new(19)
  static 'Unorm16x2' = VertexFormat.new(20)
  static 'Unorm16x4' = VertexFormat.new(21)
  static 'Snorm16' = VertexFormat.new(22)
  static 'Snorm16x2' = VertexFormat.new(23)
  static 'Snorm16x4' = VertexFormat.new(24)
  static 'Float16' = VertexFormat.new(25)
  static 'Float16x2' = VertexFormat.new(26)
  static 'Float16x4' = VertexFormat.new(27)
  static 'Float32' = VertexFormat.new(28)
  static 'Float32x2' = VertexFormat.new(29)
  static 'Float32x3' = VertexFormat.new(30)
  static 'Float32x4' = VertexFormat.new(31)
  static 'Uint32' = VertexFormat.new(32)
  static 'Uint32x2' = VertexFormat.new(33)
  static 'Uint32x3' = VertexFormat.new(34)
  static 'Uint32x4' = VertexFormat.new(35)
  static 'Sint32' = VertexFormat.new(36)
  static 'Sint32x2' = VertexFormat.new(37)
  static 'Sint32x3' = VertexFormat.new(38)
  static 'Sint32x4' = VertexFormat.new(39)
  static 'Unorm10_10_10_2' = VertexFormat.new(40)
  static 'Unorm8x4BGRA' = VertexFormat.new(41)
  static 'Force32' = VertexFormat.new(2147483647)
}
export class VertexStepMode extends c.U32 {
  static 'Undefined' = VertexStepMode.new(0)
  static 'Vertex' = VertexStepMode.new(1)
  static 'Instance' = VertexStepMode.new(2)
  static 'Force32' = VertexStepMode.new(2147483647)
}
export class WaitStatus extends c.U32 {
  static 'Success' = WaitStatus.new(1)
  static 'TimedOut' = WaitStatus.new(2)
  static 'UnsupportedTimeout' = WaitStatus.new(3)
  static 'UnsupportedCount' = WaitStatus.new(4)
  static 'UnsupportedMixedSources' = WaitStatus.new(5)
  static 'Unknown' = WaitStatus.new(6)
  static 'Force32' = WaitStatus.new(2147483647)
}

// structs
export class AdapterImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new AdapterImpl().set(val)
}
export class BindGroupImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new BindGroupImpl().set(val)
}
export class BindGroupLayoutImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new BindGroupLayoutImpl().set(val)
}
export class BufferImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new BufferImpl().set(val)
}
export class CommandBufferImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new CommandBufferImpl().set(val)
}
export class CommandEncoderImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new CommandEncoderImpl().set(val)
}
export class ComputePassEncoderImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new ComputePassEncoderImpl().set(val)
}
export class ComputePipelineImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new ComputePipelineImpl().set(val)
}
export class DeviceImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new DeviceImpl().set(val)
}
export class ExternalTextureImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new ExternalTextureImpl().set(val)
}
export class InstanceImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new InstanceImpl().set(val)
}
export class PipelineLayoutImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new PipelineLayoutImpl().set(val)
}
export class QuerySetImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new QuerySetImpl().set(val)
}
export class QueueImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new QueueImpl().set(val)
}
export class RenderBundleImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new RenderBundleImpl().set(val)
}
export class RenderBundleEncoderImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new RenderBundleEncoderImpl().set(val)
}
export class RenderPassEncoderImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new RenderPassEncoderImpl().set(val)
}
export class RenderPipelineImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new RenderPipelineImpl().set(val)
}
export class SamplerImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new SamplerImpl().set(val)
}
export class ShaderModuleImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new ShaderModuleImpl().set(val)
}
export class SharedBufferMemoryImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new SharedBufferMemoryImpl().set(val)
}
export class SharedFenceImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new SharedFenceImpl().set(val)
}
export class SharedTextureMemoryImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new SharedTextureMemoryImpl().set(val)
}
export class SurfaceImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new SurfaceImpl().set(val)
}
export class TextureImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new TextureImpl().set(val)
}
export class TextureViewImpl extends c.Struct<{}> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 0, 0)
  }
  static new = (val: Partial<{}>) => new TextureViewImpl().set(val)
}
export class INTERNAL__HAVE_EMDAWNWEBGPU_HEADER extends c.Struct<{ unused: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 4, 4)
  }
  get $unused(){ return new Bool(this.buffer, this.offset + 0) }
  protected override _value = () => ({unused: this.$unused})
  static new = (val: Partial<{ unused: Bool }>) => new INTERNAL__HAVE_EMDAWNWEBGPU_HEADER().set(val)
}
export class AdapterPropertiesD3D extends c.Struct<{ chain: ChainedStructOut; shaderModel: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $shaderModel(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, shaderModel: this.$shaderModel})
  static new = (val: Partial<{ chain: ChainedStructOut; shaderModel: c.U32 }>) => new AdapterPropertiesD3D().set(val)
}
export class AdapterPropertiesSubgroups extends c.Struct<{ chain: ChainedStructOut; subgroupMinSize: c.U32; subgroupMaxSize: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $subgroupMinSize(){ return new c.U32(this.buffer, this.offset + 16) }
  get $subgroupMaxSize(){ return new c.U32(this.buffer, this.offset + 20) }
  protected override _value = () => ({chain: this.$chain, subgroupMinSize: this.$subgroupMinSize, subgroupMaxSize: this.$subgroupMaxSize})
  static new = (val: Partial<{ chain: ChainedStructOut; subgroupMinSize: c.U32; subgroupMaxSize: c.U32 }>) => new AdapterPropertiesSubgroups().set(val)
}
export class AdapterPropertiesVk extends c.Struct<{ chain: ChainedStructOut; driverVersion: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $driverVersion(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, driverVersion: this.$driverVersion})
  static new = (val: Partial<{ chain: ChainedStructOut; driverVersion: c.U32 }>) => new AdapterPropertiesVk().set(val)
}
export class BindGroupEntry extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; binding: c.U32; buffer: Buffer; offset: c.U64; size: c.U64; sampler: Sampler; textureView: TextureView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 56, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $binding(){ return new c.U32(this.buffer, this.offset + 8) }
  get $buffer(){ return new Buffer(this.buffer, this.offset + 16) }
  get $offset(){ return new c.U64(this.buffer, this.offset + 24) }
  get $size(){ return new c.U64(this.buffer, this.offset + 32) }
  get $sampler(){ return new Sampler(this.buffer, this.offset + 40) }
  get $textureView(){ return new TextureView(this.buffer, this.offset + 48) }
  protected override _value = () => ({nextInChain: this.$nextInChain, binding: this.$binding, buffer: this.$buffer, offset: this.$offset, size: this.$size, sampler: this.$sampler, textureView: this.$textureView})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; binding: c.U32; buffer: Buffer; offset: c.U64; size: c.U64; sampler: Sampler; textureView: TextureView }>) => new BindGroupEntry().set(val)
}
export class BlendComponent extends c.Struct<{ operation: BlendOperation; srcFactor: BlendFactor; dstFactor: BlendFactor }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 12, 4)
  }
  get $operation(){ return new BlendOperation(this.buffer, this.offset + 0) }
  get $srcFactor(){ return new BlendFactor(this.buffer, this.offset + 4) }
  get $dstFactor(){ return new BlendFactor(this.buffer, this.offset + 8) }
  protected override _value = () => ({operation: this.$operation, srcFactor: this.$srcFactor, dstFactor: this.$dstFactor})
  static new = (val: Partial<{ operation: BlendOperation; srcFactor: BlendFactor; dstFactor: BlendFactor }>) => new BlendComponent().set(val)
}
export class BufferBindingLayout extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; type: BufferBindingType; hasDynamicOffset: Bool; minBindingSize: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $type(){ return new BufferBindingType(this.buffer, this.offset + 8) }
  get $hasDynamicOffset(){ return new Bool(this.buffer, this.offset + 12) }
  get $minBindingSize(){ return new c.U64(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, type: this.$type, hasDynamicOffset: this.$hasDynamicOffset, minBindingSize: this.$minBindingSize})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; type: BufferBindingType; hasDynamicOffset: Bool; minBindingSize: c.U64 }>) => new BufferBindingLayout().set(val)
}
export class BufferHostMappedPointer extends c.Struct<{ chain: ChainedStruct; pointer: c.Pointer<c.Void>; disposeCallback: Callback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $pointer(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $disposeCallback(){ return new Callback(this.buffer, this.offset + 24) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({chain: this.$chain, pointer: this.$pointer, disposeCallback: this.$disposeCallback, userdata: this.$userdata})
  static new = (val: Partial<{ chain: ChainedStruct; pointer: c.Pointer<c.Void>; disposeCallback: Callback; userdata: c.Pointer<c.Void> }>) => new BufferHostMappedPointer().set(val)
}
export class BufferMapCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: BufferMapCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new BufferMapCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: BufferMapCallback; userdata: c.Pointer<c.Void> }>) => new BufferMapCallbackInfo().set(val)
}
export class Color extends c.Struct<{ r: c.F64; g: c.F64; b: c.F64; a: c.F64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $r(){ return new c.F64(this.buffer, this.offset + 0) }
  get $g(){ return new c.F64(this.buffer, this.offset + 8) }
  get $b(){ return new c.F64(this.buffer, this.offset + 16) }
  get $a(){ return new c.F64(this.buffer, this.offset + 24) }
  protected override _value = () => ({r: this.$r, g: this.$g, b: this.$b, a: this.$a})
  static new = (val: Partial<{ r: c.F64; g: c.F64; b: c.F64; a: c.F64 }>) => new Color().set(val)
}
export class ColorTargetStateExpandResolveTextureDawn extends c.Struct<{ chain: ChainedStruct; enabled: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $enabled(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, enabled: this.$enabled})
  static new = (val: Partial<{ chain: ChainedStruct; enabled: Bool }>) => new ColorTargetStateExpandResolveTextureDawn().set(val)
}
export class CompilationInfoCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CompilationInfoCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new CompilationInfoCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CompilationInfoCallback; userdata: c.Pointer<c.Void> }>) => new CompilationInfoCallbackInfo().set(val)
}
export class ComputePassTimestampWrites extends c.Struct<{ querySet: QuerySet; beginningOfPassWriteIndex: c.U32; endOfPassWriteIndex: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $querySet(){ return new QuerySet(this.buffer, this.offset + 0) }
  get $beginningOfPassWriteIndex(){ return new c.U32(this.buffer, this.offset + 8) }
  get $endOfPassWriteIndex(){ return new c.U32(this.buffer, this.offset + 12) }
  protected override _value = () => ({querySet: this.$querySet, beginningOfPassWriteIndex: this.$beginningOfPassWriteIndex, endOfPassWriteIndex: this.$endOfPassWriteIndex})
  static new = (val: Partial<{ querySet: QuerySet; beginningOfPassWriteIndex: c.U32; endOfPassWriteIndex: c.U32 }>) => new ComputePassTimestampWrites().set(val)
}
export class CopyTextureForBrowserOptions extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; flipY: Bool; needsColorSpaceConversion: Bool; srcAlphaMode: AlphaMode; srcTransferFunctionParameters: c.Pointer<c.F32>; conversionMatrix: c.Pointer<c.F32>; dstTransferFunctionParameters: c.Pointer<c.F32>; dstAlphaMode: AlphaMode; internalUsage: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 56, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $flipY(){ return new Bool(this.buffer, this.offset + 8) }
  get $needsColorSpaceConversion(){ return new Bool(this.buffer, this.offset + 12) }
  get $srcAlphaMode(){ return new AlphaMode(this.buffer, this.offset + 16) }
  get $srcTransferFunctionParameters(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 24) }
  get $conversionMatrix(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 32) }
  get $dstTransferFunctionParameters(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 40) }
  get $dstAlphaMode(){ return new AlphaMode(this.buffer, this.offset + 48) }
  get $internalUsage(){ return new Bool(this.buffer, this.offset + 52) }
  protected override _value = () => ({nextInChain: this.$nextInChain, flipY: this.$flipY, needsColorSpaceConversion: this.$needsColorSpaceConversion, srcAlphaMode: this.$srcAlphaMode, srcTransferFunctionParameters: this.$srcTransferFunctionParameters, conversionMatrix: this.$conversionMatrix, dstTransferFunctionParameters: this.$dstTransferFunctionParameters, dstAlphaMode: this.$dstAlphaMode, internalUsage: this.$internalUsage})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; flipY: Bool; needsColorSpaceConversion: Bool; srcAlphaMode: AlphaMode; srcTransferFunctionParameters: c.Pointer<c.F32>; conversionMatrix: c.Pointer<c.F32>; dstTransferFunctionParameters: c.Pointer<c.F32>; dstAlphaMode: AlphaMode; internalUsage: Bool }>) => new CopyTextureForBrowserOptions().set(val)
}
export class CreateComputePipelineAsyncCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateComputePipelineAsyncCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new CreateComputePipelineAsyncCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateComputePipelineAsyncCallback; userdata: c.Pointer<c.Void> }>) => new CreateComputePipelineAsyncCallbackInfo().set(val)
}
export class CreateRenderPipelineAsyncCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateRenderPipelineAsyncCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new CreateRenderPipelineAsyncCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateRenderPipelineAsyncCallback; userdata: c.Pointer<c.Void> }>) => new CreateRenderPipelineAsyncCallbackInfo().set(val)
}
export class DawnWGSLBlocklist extends c.Struct<{ chain: ChainedStruct; blocklistedFeatureCount: c.Size; blocklistedFeatures: c.Pointer<c.Pointer<c.U8>> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $blocklistedFeatureCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $blocklistedFeatures(){ return new c.Pointer<c.Pointer<c.U8>>(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, blocklistedFeatureCount: this.$blocklistedFeatureCount, blocklistedFeatures: this.$blocklistedFeatures})
  static new = (val: Partial<{ chain: ChainedStruct; blocklistedFeatureCount: c.Size; blocklistedFeatures: c.Pointer<c.Pointer<c.U8>> }>) => new DawnWGSLBlocklist().set(val)
}
export class DawnAdapterPropertiesPowerPreference extends c.Struct<{ chain: ChainedStructOut; powerPreference: PowerPreference }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $powerPreference(){ return new PowerPreference(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, powerPreference: this.$powerPreference})
  static new = (val: Partial<{ chain: ChainedStructOut; powerPreference: PowerPreference }>) => new DawnAdapterPropertiesPowerPreference().set(val)
}
export class DawnBufferDescriptorErrorInfoFromWireClient extends c.Struct<{ chain: ChainedStruct; outOfMemory: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $outOfMemory(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, outOfMemory: this.$outOfMemory})
  static new = (val: Partial<{ chain: ChainedStruct; outOfMemory: Bool }>) => new DawnBufferDescriptorErrorInfoFromWireClient().set(val)
}
export class DawnEncoderInternalUsageDescriptor extends c.Struct<{ chain: ChainedStruct; useInternalUsages: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $useInternalUsages(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, useInternalUsages: this.$useInternalUsages})
  static new = (val: Partial<{ chain: ChainedStruct; useInternalUsages: Bool }>) => new DawnEncoderInternalUsageDescriptor().set(val)
}
export class DawnExperimentalImmediateDataLimits extends c.Struct<{ chain: ChainedStructOut; maxImmediateDataRangeByteSize: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $maxImmediateDataRangeByteSize(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, maxImmediateDataRangeByteSize: this.$maxImmediateDataRangeByteSize})
  static new = (val: Partial<{ chain: ChainedStructOut; maxImmediateDataRangeByteSize: c.U32 }>) => new DawnExperimentalImmediateDataLimits().set(val)
}
export class DawnExperimentalSubgroupLimits extends c.Struct<{ chain: ChainedStructOut; minSubgroupSize: c.U32; maxSubgroupSize: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $minSubgroupSize(){ return new c.U32(this.buffer, this.offset + 16) }
  get $maxSubgroupSize(){ return new c.U32(this.buffer, this.offset + 20) }
  protected override _value = () => ({chain: this.$chain, minSubgroupSize: this.$minSubgroupSize, maxSubgroupSize: this.$maxSubgroupSize})
  static new = (val: Partial<{ chain: ChainedStructOut; minSubgroupSize: c.U32; maxSubgroupSize: c.U32 }>) => new DawnExperimentalSubgroupLimits().set(val)
}
export class DawnRenderPassColorAttachmentRenderToSingleSampled extends c.Struct<{ chain: ChainedStruct; implicitSampleCount: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $implicitSampleCount(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, implicitSampleCount: this.$implicitSampleCount})
  static new = (val: Partial<{ chain: ChainedStruct; implicitSampleCount: c.U32 }>) => new DawnRenderPassColorAttachmentRenderToSingleSampled().set(val)
}
export class DawnShaderModuleSPIRVOptionsDescriptor extends c.Struct<{ chain: ChainedStruct; allowNonUniformDerivatives: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $allowNonUniformDerivatives(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, allowNonUniformDerivatives: this.$allowNonUniformDerivatives})
  static new = (val: Partial<{ chain: ChainedStruct; allowNonUniformDerivatives: Bool }>) => new DawnShaderModuleSPIRVOptionsDescriptor().set(val)
}
export class DawnTexelCopyBufferRowAlignmentLimits extends c.Struct<{ chain: ChainedStructOut; minTexelCopyBufferRowAlignment: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $minTexelCopyBufferRowAlignment(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, minTexelCopyBufferRowAlignment: this.$minTexelCopyBufferRowAlignment})
  static new = (val: Partial<{ chain: ChainedStructOut; minTexelCopyBufferRowAlignment: c.U32 }>) => new DawnTexelCopyBufferRowAlignmentLimits().set(val)
}
export class DawnTextureInternalUsageDescriptor extends c.Struct<{ chain: ChainedStruct; internalUsage: TextureUsage }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $internalUsage(){ return new TextureUsage(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, internalUsage: this.$internalUsage})
  static new = (val: Partial<{ chain: ChainedStruct; internalUsage: TextureUsage }>) => new DawnTextureInternalUsageDescriptor().set(val)
}
export class DawnTogglesDescriptor extends c.Struct<{ chain: ChainedStruct; enabledToggleCount: c.Size; enabledToggles: c.Pointer<c.Pointer<c.U8>>; disabledToggleCount: c.Size; disabledToggles: c.Pointer<c.Pointer<c.U8>> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 48, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $enabledToggleCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $enabledToggles(){ return new c.Pointer<c.Pointer<c.U8>>(this.buffer, this.offset + 24) }
  get $disabledToggleCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $disabledToggles(){ return new c.Pointer<c.Pointer<c.U8>>(this.buffer, this.offset + 40) }
  protected override _value = () => ({chain: this.$chain, enabledToggleCount: this.$enabledToggleCount, enabledToggles: this.$enabledToggles, disabledToggleCount: this.$disabledToggleCount, disabledToggles: this.$disabledToggles})
  static new = (val: Partial<{ chain: ChainedStruct; enabledToggleCount: c.Size; enabledToggles: c.Pointer<c.Pointer<c.U8>>; disabledToggleCount: c.Size; disabledToggles: c.Pointer<c.Pointer<c.U8>> }>) => new DawnTogglesDescriptor().set(val)
}
export class DawnWireWGSLControl extends c.Struct<{ chain: ChainedStruct; enableExperimental: Bool; enableUnsafe: Bool; enableTesting: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $enableExperimental(){ return new Bool(this.buffer, this.offset + 16) }
  get $enableUnsafe(){ return new Bool(this.buffer, this.offset + 20) }
  get $enableTesting(){ return new Bool(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, enableExperimental: this.$enableExperimental, enableUnsafe: this.$enableUnsafe, enableTesting: this.$enableTesting})
  static new = (val: Partial<{ chain: ChainedStruct; enableExperimental: Bool; enableUnsafe: Bool; enableTesting: Bool }>) => new DawnWireWGSLControl().set(val)
}
export class DeviceLostCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: DeviceLostCallbackNew; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new DeviceLostCallbackNew(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: DeviceLostCallbackNew; userdata: c.Pointer<c.Void> }>) => new DeviceLostCallbackInfo().set(val)
}
export class DrmFormatProperties extends c.Struct<{ modifier: c.U64; modifierPlaneCount: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $modifier(){ return new c.U64(this.buffer, this.offset + 0) }
  get $modifierPlaneCount(){ return new c.U32(this.buffer, this.offset + 8) }
  protected override _value = () => ({modifier: this.$modifier, modifierPlaneCount: this.$modifierPlaneCount})
  static new = (val: Partial<{ modifier: c.U64; modifierPlaneCount: c.U32 }>) => new DrmFormatProperties().set(val)
}
export class Extent2D extends c.Struct<{ width: c.U32; height: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 4)
  }
  get $width(){ return new c.U32(this.buffer, this.offset + 0) }
  get $height(){ return new c.U32(this.buffer, this.offset + 4) }
  protected override _value = () => ({width: this.$width, height: this.$height})
  static new = (val: Partial<{ width: c.U32; height: c.U32 }>) => new Extent2D().set(val)
}
export class Extent3D extends c.Struct<{ width: c.U32; height: c.U32; depthOrArrayLayers: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 12, 4)
  }
  get $width(){ return new c.U32(this.buffer, this.offset + 0) }
  get $height(){ return new c.U32(this.buffer, this.offset + 4) }
  get $depthOrArrayLayers(){ return new c.U32(this.buffer, this.offset + 8) }
  protected override _value = () => ({width: this.$width, height: this.$height, depthOrArrayLayers: this.$depthOrArrayLayers})
  static new = (val: Partial<{ width: c.U32; height: c.U32; depthOrArrayLayers: c.U32 }>) => new Extent3D().set(val)
}
export class ExternalTextureBindingEntry extends c.Struct<{ chain: ChainedStruct; externalTexture: ExternalTexture }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $externalTexture(){ return new ExternalTexture(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, externalTexture: this.$externalTexture})
  static new = (val: Partial<{ chain: ChainedStruct; externalTexture: ExternalTexture }>) => new ExternalTextureBindingEntry().set(val)
}
export class ExternalTextureBindingLayout extends c.Struct<{ chain: ChainedStruct }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  protected override _value = () => ({chain: this.$chain})
  static new = (val: Partial<{ chain: ChainedStruct }>) => new ExternalTextureBindingLayout().set(val)
}
export class FormatCapabilities extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  protected override _value = () => ({nextInChain: this.$nextInChain})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut> }>) => new FormatCapabilities().set(val)
}
export class Future extends c.Struct<{ id: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 8)
  }
  get $id(){ return new c.U64(this.buffer, this.offset + 0) }
  protected override _value = () => ({id: this.$id})
  static new = (val: Partial<{ id: c.U64 }>) => new Future().set(val)
  override _setNative=(val: ArrayBuffer | Deno.PointerValue<unknown>) =>{
    if (val instanceof ArrayBuffer) this.buffer = val
   else this.set({id:c.U64.new(Deno.UnsafePointer.value(val as Deno.PointerObject))})
  }
}
export class InstanceFeatures extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; timedWaitAnyEnable: Bool; timedWaitAnyMaxCount: c.Size }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $timedWaitAnyEnable(){ return new Bool(this.buffer, this.offset + 8) }
  get $timedWaitAnyMaxCount(){ return new c.Size(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, timedWaitAnyEnable: this.$timedWaitAnyEnable, timedWaitAnyMaxCount: this.$timedWaitAnyMaxCount})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; timedWaitAnyEnable: Bool; timedWaitAnyMaxCount: c.Size }>) => new InstanceFeatures().set(val)
}
export class Limits extends c.Struct<{ maxTextureDimension1D: c.U32; maxTextureDimension2D: c.U32; maxTextureDimension3D: c.U32; maxTextureArrayLayers: c.U32; maxBindGroups: c.U32; maxBindGroupsPlusVertexBuffers: c.U32; maxBindingsPerBindGroup: c.U32; maxDynamicUniformBuffersPerPipelineLayout: c.U32; maxDynamicStorageBuffersPerPipelineLayout: c.U32; maxSampledTexturesPerShaderStage: c.U32; maxSamplersPerShaderStage: c.U32; maxStorageBuffersPerShaderStage: c.U32; maxStorageTexturesPerShaderStage: c.U32; maxUniformBuffersPerShaderStage: c.U32; maxUniformBufferBindingSize: c.U64; maxStorageBufferBindingSize: c.U64; minUniformBufferOffsetAlignment: c.U32; minStorageBufferOffsetAlignment: c.U32; maxVertexBuffers: c.U32; maxBufferSize: c.U64; maxVertexAttributes: c.U32; maxVertexBufferArrayStride: c.U32; maxInterStageShaderComponents: c.U32; maxInterStageShaderVariables: c.U32; maxColorAttachments: c.U32; maxColorAttachmentBytesPerSample: c.U32; maxComputeWorkgroupStorageSize: c.U32; maxComputeInvocationsPerWorkgroup: c.U32; maxComputeWorkgroupSizeX: c.U32; maxComputeWorkgroupSizeY: c.U32; maxComputeWorkgroupSizeZ: c.U32; maxComputeWorkgroupsPerDimension: c.U32; maxStorageBuffersInVertexStage: c.U32; maxStorageTexturesInVertexStage: c.U32; maxStorageBuffersInFragmentStage: c.U32; maxStorageTexturesInFragmentStage: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 160, 8)
  }
  get $maxTextureDimension1D(){ return new c.U32(this.buffer, this.offset + 0) }
  get $maxTextureDimension2D(){ return new c.U32(this.buffer, this.offset + 4) }
  get $maxTextureDimension3D(){ return new c.U32(this.buffer, this.offset + 8) }
  get $maxTextureArrayLayers(){ return new c.U32(this.buffer, this.offset + 12) }
  get $maxBindGroups(){ return new c.U32(this.buffer, this.offset + 16) }
  get $maxBindGroupsPlusVertexBuffers(){ return new c.U32(this.buffer, this.offset + 20) }
  get $maxBindingsPerBindGroup(){ return new c.U32(this.buffer, this.offset + 24) }
  get $maxDynamicUniformBuffersPerPipelineLayout(){ return new c.U32(this.buffer, this.offset + 28) }
  get $maxDynamicStorageBuffersPerPipelineLayout(){ return new c.U32(this.buffer, this.offset + 32) }
  get $maxSampledTexturesPerShaderStage(){ return new c.U32(this.buffer, this.offset + 36) }
  get $maxSamplersPerShaderStage(){ return new c.U32(this.buffer, this.offset + 40) }
  get $maxStorageBuffersPerShaderStage(){ return new c.U32(this.buffer, this.offset + 44) }
  get $maxStorageTexturesPerShaderStage(){ return new c.U32(this.buffer, this.offset + 48) }
  get $maxUniformBuffersPerShaderStage(){ return new c.U32(this.buffer, this.offset + 52) }
  get $maxUniformBufferBindingSize(){ return new c.U64(this.buffer, this.offset + 56) }
  get $maxStorageBufferBindingSize(){ return new c.U64(this.buffer, this.offset + 64) }
  get $minUniformBufferOffsetAlignment(){ return new c.U32(this.buffer, this.offset + 72) }
  get $minStorageBufferOffsetAlignment(){ return new c.U32(this.buffer, this.offset + 76) }
  get $maxVertexBuffers(){ return new c.U32(this.buffer, this.offset + 80) }
  get $maxBufferSize(){ return new c.U64(this.buffer, this.offset + 88) }
  get $maxVertexAttributes(){ return new c.U32(this.buffer, this.offset + 96) }
  get $maxVertexBufferArrayStride(){ return new c.U32(this.buffer, this.offset + 100) }
  get $maxInterStageShaderComponents(){ return new c.U32(this.buffer, this.offset + 104) }
  get $maxInterStageShaderVariables(){ return new c.U32(this.buffer, this.offset + 108) }
  get $maxColorAttachments(){ return new c.U32(this.buffer, this.offset + 112) }
  get $maxColorAttachmentBytesPerSample(){ return new c.U32(this.buffer, this.offset + 116) }
  get $maxComputeWorkgroupStorageSize(){ return new c.U32(this.buffer, this.offset + 120) }
  get $maxComputeInvocationsPerWorkgroup(){ return new c.U32(this.buffer, this.offset + 124) }
  get $maxComputeWorkgroupSizeX(){ return new c.U32(this.buffer, this.offset + 128) }
  get $maxComputeWorkgroupSizeY(){ return new c.U32(this.buffer, this.offset + 132) }
  get $maxComputeWorkgroupSizeZ(){ return new c.U32(this.buffer, this.offset + 136) }
  get $maxComputeWorkgroupsPerDimension(){ return new c.U32(this.buffer, this.offset + 140) }
  get $maxStorageBuffersInVertexStage(){ return new c.U32(this.buffer, this.offset + 144) }
  get $maxStorageTexturesInVertexStage(){ return new c.U32(this.buffer, this.offset + 148) }
  get $maxStorageBuffersInFragmentStage(){ return new c.U32(this.buffer, this.offset + 152) }
  get $maxStorageTexturesInFragmentStage(){ return new c.U32(this.buffer, this.offset + 156) }
  protected override _value = () => ({maxTextureDimension1D: this.$maxTextureDimension1D, maxTextureDimension2D: this.$maxTextureDimension2D, maxTextureDimension3D: this.$maxTextureDimension3D, maxTextureArrayLayers: this.$maxTextureArrayLayers, maxBindGroups: this.$maxBindGroups, maxBindGroupsPlusVertexBuffers: this.$maxBindGroupsPlusVertexBuffers, maxBindingsPerBindGroup: this.$maxBindingsPerBindGroup, maxDynamicUniformBuffersPerPipelineLayout: this.$maxDynamicUniformBuffersPerPipelineLayout, maxDynamicStorageBuffersPerPipelineLayout: this.$maxDynamicStorageBuffersPerPipelineLayout, maxSampledTexturesPerShaderStage: this.$maxSampledTexturesPerShaderStage, maxSamplersPerShaderStage: this.$maxSamplersPerShaderStage, maxStorageBuffersPerShaderStage: this.$maxStorageBuffersPerShaderStage, maxStorageTexturesPerShaderStage: this.$maxStorageTexturesPerShaderStage, maxUniformBuffersPerShaderStage: this.$maxUniformBuffersPerShaderStage, maxUniformBufferBindingSize: this.$maxUniformBufferBindingSize, maxStorageBufferBindingSize: this.$maxStorageBufferBindingSize, minUniformBufferOffsetAlignment: this.$minUniformBufferOffsetAlignment, minStorageBufferOffsetAlignment: this.$minStorageBufferOffsetAlignment, maxVertexBuffers: this.$maxVertexBuffers, maxBufferSize: this.$maxBufferSize, maxVertexAttributes: this.$maxVertexAttributes, maxVertexBufferArrayStride: this.$maxVertexBufferArrayStride, maxInterStageShaderComponents: this.$maxInterStageShaderComponents, maxInterStageShaderVariables: this.$maxInterStageShaderVariables, maxColorAttachments: this.$maxColorAttachments, maxColorAttachmentBytesPerSample: this.$maxColorAttachmentBytesPerSample, maxComputeWorkgroupStorageSize: this.$maxComputeWorkgroupStorageSize, maxComputeInvocationsPerWorkgroup: this.$maxComputeInvocationsPerWorkgroup, maxComputeWorkgroupSizeX: this.$maxComputeWorkgroupSizeX, maxComputeWorkgroupSizeY: this.$maxComputeWorkgroupSizeY, maxComputeWorkgroupSizeZ: this.$maxComputeWorkgroupSizeZ, maxComputeWorkgroupsPerDimension: this.$maxComputeWorkgroupsPerDimension, maxStorageBuffersInVertexStage: this.$maxStorageBuffersInVertexStage, maxStorageTexturesInVertexStage: this.$maxStorageTexturesInVertexStage, maxStorageBuffersInFragmentStage: this.$maxStorageBuffersInFragmentStage, maxStorageTexturesInFragmentStage: this.$maxStorageTexturesInFragmentStage})
  static new = (val: Partial<{ maxTextureDimension1D: c.U32; maxTextureDimension2D: c.U32; maxTextureDimension3D: c.U32; maxTextureArrayLayers: c.U32; maxBindGroups: c.U32; maxBindGroupsPlusVertexBuffers: c.U32; maxBindingsPerBindGroup: c.U32; maxDynamicUniformBuffersPerPipelineLayout: c.U32; maxDynamicStorageBuffersPerPipelineLayout: c.U32; maxSampledTexturesPerShaderStage: c.U32; maxSamplersPerShaderStage: c.U32; maxStorageBuffersPerShaderStage: c.U32; maxStorageTexturesPerShaderStage: c.U32; maxUniformBuffersPerShaderStage: c.U32; maxUniformBufferBindingSize: c.U64; maxStorageBufferBindingSize: c.U64; minUniformBufferOffsetAlignment: c.U32; minStorageBufferOffsetAlignment: c.U32; maxVertexBuffers: c.U32; maxBufferSize: c.U64; maxVertexAttributes: c.U32; maxVertexBufferArrayStride: c.U32; maxInterStageShaderComponents: c.U32; maxInterStageShaderVariables: c.U32; maxColorAttachments: c.U32; maxColorAttachmentBytesPerSample: c.U32; maxComputeWorkgroupStorageSize: c.U32; maxComputeInvocationsPerWorkgroup: c.U32; maxComputeWorkgroupSizeX: c.U32; maxComputeWorkgroupSizeY: c.U32; maxComputeWorkgroupSizeZ: c.U32; maxComputeWorkgroupsPerDimension: c.U32; maxStorageBuffersInVertexStage: c.U32; maxStorageTexturesInVertexStage: c.U32; maxStorageBuffersInFragmentStage: c.U32; maxStorageTexturesInFragmentStage: c.U32 }>) => new Limits().set(val)
}
export class MemoryHeapInfo extends c.Struct<{ properties: HeapProperty; size: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $properties(){ return new HeapProperty(this.buffer, this.offset + 0) }
  get $size(){ return new c.U64(this.buffer, this.offset + 8) }
  protected override _value = () => ({properties: this.$properties, size: this.$size})
  static new = (val: Partial<{ properties: HeapProperty; size: c.U64 }>) => new MemoryHeapInfo().set(val)
}
export class MultisampleState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; count: c.U32; mask: c.U32; alphaToCoverageEnabled: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $count(){ return new c.U32(this.buffer, this.offset + 8) }
  get $mask(){ return new c.U32(this.buffer, this.offset + 12) }
  get $alphaToCoverageEnabled(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, count: this.$count, mask: this.$mask, alphaToCoverageEnabled: this.$alphaToCoverageEnabled})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; count: c.U32; mask: c.U32; alphaToCoverageEnabled: Bool }>) => new MultisampleState().set(val)
}
export class Origin2D extends c.Struct<{ x: c.U32; y: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 8, 4)
  }
  get $x(){ return new c.U32(this.buffer, this.offset + 0) }
  get $y(){ return new c.U32(this.buffer, this.offset + 4) }
  protected override _value = () => ({x: this.$x, y: this.$y})
  static new = (val: Partial<{ x: c.U32; y: c.U32 }>) => new Origin2D().set(val)
}
export class Origin3D extends c.Struct<{ x: c.U32; y: c.U32; z: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 12, 4)
  }
  get $x(){ return new c.U32(this.buffer, this.offset + 0) }
  get $y(){ return new c.U32(this.buffer, this.offset + 4) }
  get $z(){ return new c.U32(this.buffer, this.offset + 8) }
  protected override _value = () => ({x: this.$x, y: this.$y, z: this.$z})
  static new = (val: Partial<{ x: c.U32; y: c.U32; z: c.U32 }>) => new Origin3D().set(val)
}
export class PipelineLayoutStorageAttachment extends c.Struct<{ offset: c.U64; format: TextureFormat }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $offset(){ return new c.U64(this.buffer, this.offset + 0) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 8) }
  protected override _value = () => ({offset: this.$offset, format: this.$format})
  static new = (val: Partial<{ offset: c.U64; format: TextureFormat }>) => new PipelineLayoutStorageAttachment().set(val)
}
export class PopErrorScopeCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: PopErrorScopeCallback; oldCallback: ErrorCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new PopErrorScopeCallback(this.buffer, this.offset + 16) }
  get $oldCallback(){ return new ErrorCallback(this.buffer, this.offset + 24) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, oldCallback: this.$oldCallback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: PopErrorScopeCallback; oldCallback: ErrorCallback; userdata: c.Pointer<c.Void> }>) => new PopErrorScopeCallbackInfo().set(val)
}
export class PrimitiveState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; topology: PrimitiveTopology; stripIndexFormat: IndexFormat; frontFace: FrontFace; cullMode: CullMode; unclippedDepth: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $topology(){ return new PrimitiveTopology(this.buffer, this.offset + 8) }
  get $stripIndexFormat(){ return new IndexFormat(this.buffer, this.offset + 12) }
  get $frontFace(){ return new FrontFace(this.buffer, this.offset + 16) }
  get $cullMode(){ return new CullMode(this.buffer, this.offset + 20) }
  get $unclippedDepth(){ return new Bool(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, topology: this.$topology, stripIndexFormat: this.$stripIndexFormat, frontFace: this.$frontFace, cullMode: this.$cullMode, unclippedDepth: this.$unclippedDepth})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; topology: PrimitiveTopology; stripIndexFormat: IndexFormat; frontFace: FrontFace; cullMode: CullMode; unclippedDepth: Bool }>) => new PrimitiveState().set(val)
}
export class QueueWorkDoneCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: QueueWorkDoneCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new QueueWorkDoneCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: QueueWorkDoneCallback; userdata: c.Pointer<c.Void> }>) => new QueueWorkDoneCallbackInfo().set(val)
}
export class RenderPassDepthStencilAttachment extends c.Struct<{ view: TextureView; depthLoadOp: LoadOp; depthStoreOp: StoreOp; depthClearValue: c.F32; depthReadOnly: Bool; stencilLoadOp: LoadOp; stencilStoreOp: StoreOp; stencilClearValue: c.U32; stencilReadOnly: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $view(){ return new TextureView(this.buffer, this.offset + 0) }
  get $depthLoadOp(){ return new LoadOp(this.buffer, this.offset + 8) }
  get $depthStoreOp(){ return new StoreOp(this.buffer, this.offset + 12) }
  get $depthClearValue(){ return new c.F32(this.buffer, this.offset + 16) }
  get $depthReadOnly(){ return new Bool(this.buffer, this.offset + 20) }
  get $stencilLoadOp(){ return new LoadOp(this.buffer, this.offset + 24) }
  get $stencilStoreOp(){ return new StoreOp(this.buffer, this.offset + 28) }
  get $stencilClearValue(){ return new c.U32(this.buffer, this.offset + 32) }
  get $stencilReadOnly(){ return new Bool(this.buffer, this.offset + 36) }
  protected override _value = () => ({view: this.$view, depthLoadOp: this.$depthLoadOp, depthStoreOp: this.$depthStoreOp, depthClearValue: this.$depthClearValue, depthReadOnly: this.$depthReadOnly, stencilLoadOp: this.$stencilLoadOp, stencilStoreOp: this.$stencilStoreOp, stencilClearValue: this.$stencilClearValue, stencilReadOnly: this.$stencilReadOnly})
  static new = (val: Partial<{ view: TextureView; depthLoadOp: LoadOp; depthStoreOp: StoreOp; depthClearValue: c.F32; depthReadOnly: Bool; stencilLoadOp: LoadOp; stencilStoreOp: StoreOp; stencilClearValue: c.U32; stencilReadOnly: Bool }>) => new RenderPassDepthStencilAttachment().set(val)
}
export class RenderPassDescriptorExpandResolveRect extends c.Struct<{ chain: ChainedStruct; x: c.U32; y: c.U32; width: c.U32; height: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $x(){ return new c.U32(this.buffer, this.offset + 16) }
  get $y(){ return new c.U32(this.buffer, this.offset + 20) }
  get $width(){ return new c.U32(this.buffer, this.offset + 24) }
  get $height(){ return new c.U32(this.buffer, this.offset + 28) }
  protected override _value = () => ({chain: this.$chain, x: this.$x, y: this.$y, width: this.$width, height: this.$height})
  static new = (val: Partial<{ chain: ChainedStruct; x: c.U32; y: c.U32; width: c.U32; height: c.U32 }>) => new RenderPassDescriptorExpandResolveRect().set(val)
}
export class RenderPassMaxDrawCount extends c.Struct<{ chain: ChainedStruct; maxDrawCount: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $maxDrawCount(){ return new c.U64(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, maxDrawCount: this.$maxDrawCount})
  static new = (val: Partial<{ chain: ChainedStruct; maxDrawCount: c.U64 }>) => new RenderPassMaxDrawCount().set(val)
}
export class RenderPassTimestampWrites extends c.Struct<{ querySet: QuerySet; beginningOfPassWriteIndex: c.U32; endOfPassWriteIndex: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $querySet(){ return new QuerySet(this.buffer, this.offset + 0) }
  get $beginningOfPassWriteIndex(){ return new c.U32(this.buffer, this.offset + 8) }
  get $endOfPassWriteIndex(){ return new c.U32(this.buffer, this.offset + 12) }
  protected override _value = () => ({querySet: this.$querySet, beginningOfPassWriteIndex: this.$beginningOfPassWriteIndex, endOfPassWriteIndex: this.$endOfPassWriteIndex})
  static new = (val: Partial<{ querySet: QuerySet; beginningOfPassWriteIndex: c.U32; endOfPassWriteIndex: c.U32 }>) => new RenderPassTimestampWrites().set(val)
}
export class RequestAdapterCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestAdapterCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new RequestAdapterCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestAdapterCallback; userdata: c.Pointer<c.Void> }>) => new RequestAdapterCallbackInfo().set(val)
}
export class RequestAdapterOptions extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; compatibleSurface: Surface; featureLevel: FeatureLevel; powerPreference: PowerPreference; backendType: BackendType; forceFallbackAdapter: Bool; compatibilityMode: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $compatibleSurface(){ return new Surface(this.buffer, this.offset + 8) }
  get $featureLevel(){ return new FeatureLevel(this.buffer, this.offset + 16) }
  get $powerPreference(){ return new PowerPreference(this.buffer, this.offset + 20) }
  get $backendType(){ return new BackendType(this.buffer, this.offset + 24) }
  get $forceFallbackAdapter(){ return new Bool(this.buffer, this.offset + 28) }
  get $compatibilityMode(){ return new Bool(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, compatibleSurface: this.$compatibleSurface, featureLevel: this.$featureLevel, powerPreference: this.$powerPreference, backendType: this.$backendType, forceFallbackAdapter: this.$forceFallbackAdapter, compatibilityMode: this.$compatibilityMode})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; compatibleSurface: Surface; featureLevel: FeatureLevel; powerPreference: PowerPreference; backendType: BackendType; forceFallbackAdapter: Bool; compatibilityMode: Bool }>) => new RequestAdapterOptions().set(val)
}
export class RequestDeviceCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestDeviceCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new RequestDeviceCallback(this.buffer, this.offset + 16) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestDeviceCallback; userdata: c.Pointer<c.Void> }>) => new RequestDeviceCallbackInfo().set(val)
}
export class SamplerBindingLayout extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; type: SamplerBindingType }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $type(){ return new SamplerBindingType(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, type: this.$type})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; type: SamplerBindingType }>) => new SamplerBindingLayout().set(val)
}
export class ShaderModuleCompilationOptions extends c.Struct<{ chain: ChainedStruct; strictMath: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $strictMath(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, strictMath: this.$strictMath})
  static new = (val: Partial<{ chain: ChainedStruct; strictMath: Bool }>) => new ShaderModuleCompilationOptions().set(val)
}
export class ShaderSourceSPIRV extends c.Struct<{ chain: ChainedStruct; codeSize: c.U32; code: c.Pointer<c.U32> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $codeSize(){ return new c.U32(this.buffer, this.offset + 16) }
  get $code(){ return new c.Pointer<c.U32>(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, codeSize: this.$codeSize, code: this.$code})
  static new = (val: Partial<{ chain: ChainedStruct; codeSize: c.U32; code: c.Pointer<c.U32> }>) => new ShaderSourceSPIRV().set(val)
}
export class SharedBufferMemoryBeginAccessDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $initialized(){ return new Bool(this.buffer, this.offset + 8) }
  get $fenceCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $fences(){ return new c.Pointer<SharedFence>(this.buffer, this.offset + 24) }
  get $signaledValues(){ return new c.Pointer<c.U64>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, initialized: this.$initialized, fenceCount: this.$fenceCount, fences: this.$fences, signaledValues: this.$signaledValues})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }>) => new SharedBufferMemoryBeginAccessDescriptor().set(val)
}
export class SharedBufferMemoryEndAccessState extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $initialized(){ return new Bool(this.buffer, this.offset + 8) }
  get $fenceCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $fences(){ return new c.Pointer<SharedFence>(this.buffer, this.offset + 24) }
  get $signaledValues(){ return new c.Pointer<c.U64>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, initialized: this.$initialized, fenceCount: this.$fenceCount, fences: this.$fences, signaledValues: this.$signaledValues})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }>) => new SharedBufferMemoryEndAccessState().set(val)
}
export class SharedBufferMemoryProperties extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; usage: BufferUsage; size: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $usage(){ return new BufferUsage(this.buffer, this.offset + 8) }
  get $size(){ return new c.U64(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, usage: this.$usage, size: this.$size})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; usage: BufferUsage; size: c.U64 }>) => new SharedBufferMemoryProperties().set(val)
}
export class SharedFenceDXGISharedHandleDescriptor extends c.Struct<{ chain: ChainedStruct; handle: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $handle(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStruct; handle: c.Pointer<c.Void> }>) => new SharedFenceDXGISharedHandleDescriptor().set(val)
}
export class SharedFenceDXGISharedHandleExportInfo extends c.Struct<{ chain: ChainedStructOut; handle: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $handle(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStructOut; handle: c.Pointer<c.Void> }>) => new SharedFenceDXGISharedHandleExportInfo().set(val)
}
export class SharedFenceMTLSharedEventDescriptor extends c.Struct<{ chain: ChainedStruct; sharedEvent: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $sharedEvent(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, sharedEvent: this.$sharedEvent})
  static new = (val: Partial<{ chain: ChainedStruct; sharedEvent: c.Pointer<c.Void> }>) => new SharedFenceMTLSharedEventDescriptor().set(val)
}
export class SharedFenceMTLSharedEventExportInfo extends c.Struct<{ chain: ChainedStructOut; sharedEvent: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $sharedEvent(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, sharedEvent: this.$sharedEvent})
  static new = (val: Partial<{ chain: ChainedStructOut; sharedEvent: c.Pointer<c.Void> }>) => new SharedFenceMTLSharedEventExportInfo().set(val)
}
export class SharedFenceExportInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; type: SharedFenceType }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $type(){ return new SharedFenceType(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, type: this.$type})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; type: SharedFenceType }>) => new SharedFenceExportInfo().set(val)
}
export class SharedFenceSyncFDDescriptor extends c.Struct<{ chain: ChainedStruct; handle: c.I32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $handle(){ return new c.I32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStruct; handle: c.I32 }>) => new SharedFenceSyncFDDescriptor().set(val)
}
export class SharedFenceSyncFDExportInfo extends c.Struct<{ chain: ChainedStructOut; handle: c.I32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $handle(){ return new c.I32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStructOut; handle: c.I32 }>) => new SharedFenceSyncFDExportInfo().set(val)
}
export class SharedFenceVkSemaphoreOpaqueFDDescriptor extends c.Struct<{ chain: ChainedStruct; handle: c.I32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $handle(){ return new c.I32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStruct; handle: c.I32 }>) => new SharedFenceVkSemaphoreOpaqueFDDescriptor().set(val)
}
export class SharedFenceVkSemaphoreOpaqueFDExportInfo extends c.Struct<{ chain: ChainedStructOut; handle: c.I32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $handle(){ return new c.I32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStructOut; handle: c.I32 }>) => new SharedFenceVkSemaphoreOpaqueFDExportInfo().set(val)
}
export class SharedFenceVkSemaphoreZirconHandleDescriptor extends c.Struct<{ chain: ChainedStruct; handle: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $handle(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStruct; handle: c.U32 }>) => new SharedFenceVkSemaphoreZirconHandleDescriptor().set(val)
}
export class SharedFenceVkSemaphoreZirconHandleExportInfo extends c.Struct<{ chain: ChainedStructOut; handle: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $handle(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle})
  static new = (val: Partial<{ chain: ChainedStructOut; handle: c.U32 }>) => new SharedFenceVkSemaphoreZirconHandleExportInfo().set(val)
}
export class SharedTextureMemoryD3DSwapchainBeginState extends c.Struct<{ chain: ChainedStruct; isSwapchain: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $isSwapchain(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, isSwapchain: this.$isSwapchain})
  static new = (val: Partial<{ chain: ChainedStruct; isSwapchain: Bool }>) => new SharedTextureMemoryD3DSwapchainBeginState().set(val)
}
export class SharedTextureMemoryDXGISharedHandleDescriptor extends c.Struct<{ chain: ChainedStruct; handle: c.Pointer<c.Void>; useKeyedMutex: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $handle(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $useKeyedMutex(){ return new Bool(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle, useKeyedMutex: this.$useKeyedMutex})
  static new = (val: Partial<{ chain: ChainedStruct; handle: c.Pointer<c.Void>; useKeyedMutex: Bool }>) => new SharedTextureMemoryDXGISharedHandleDescriptor().set(val)
}
export class SharedTextureMemoryEGLImageDescriptor extends c.Struct<{ chain: ChainedStruct; image: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $image(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, image: this.$image})
  static new = (val: Partial<{ chain: ChainedStruct; image: c.Pointer<c.Void> }>) => new SharedTextureMemoryEGLImageDescriptor().set(val)
}
export class SharedTextureMemoryIOSurfaceDescriptor extends c.Struct<{ chain: ChainedStruct; ioSurface: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $ioSurface(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, ioSurface: this.$ioSurface})
  static new = (val: Partial<{ chain: ChainedStruct; ioSurface: c.Pointer<c.Void> }>) => new SharedTextureMemoryIOSurfaceDescriptor().set(val)
}
export class SharedTextureMemoryAHardwareBufferDescriptor extends c.Struct<{ chain: ChainedStruct; handle: c.Pointer<c.Void>; useExternalFormat: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $handle(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $useExternalFormat(){ return new Bool(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, handle: this.$handle, useExternalFormat: this.$useExternalFormat})
  static new = (val: Partial<{ chain: ChainedStruct; handle: c.Pointer<c.Void>; useExternalFormat: Bool }>) => new SharedTextureMemoryAHardwareBufferDescriptor().set(val)
}
export class SharedTextureMemoryBeginAccessDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; concurrentRead: Bool; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $concurrentRead(){ return new Bool(this.buffer, this.offset + 8) }
  get $initialized(){ return new Bool(this.buffer, this.offset + 12) }
  get $fenceCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $fences(){ return new c.Pointer<SharedFence>(this.buffer, this.offset + 24) }
  get $signaledValues(){ return new c.Pointer<c.U64>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, concurrentRead: this.$concurrentRead, initialized: this.$initialized, fenceCount: this.$fenceCount, fences: this.$fences, signaledValues: this.$signaledValues})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; concurrentRead: Bool; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }>) => new SharedTextureMemoryBeginAccessDescriptor().set(val)
}
export class SharedTextureMemoryDmaBufPlane extends c.Struct<{ fd: c.I32; offset: c.U64; stride: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $fd(){ return new c.I32(this.buffer, this.offset + 0) }
  get $offset(){ return new c.U64(this.buffer, this.offset + 8) }
  get $stride(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({fd: this.$fd, offset: this.$offset, stride: this.$stride})
  static new = (val: Partial<{ fd: c.I32; offset: c.U64; stride: c.U32 }>) => new SharedTextureMemoryDmaBufPlane().set(val)
}
export class SharedTextureMemoryEndAccessState extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $initialized(){ return new Bool(this.buffer, this.offset + 8) }
  get $fenceCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $fences(){ return new c.Pointer<SharedFence>(this.buffer, this.offset + 24) }
  get $signaledValues(){ return new c.Pointer<c.U64>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, initialized: this.$initialized, fenceCount: this.$fenceCount, fences: this.$fences, signaledValues: this.$signaledValues})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; initialized: Bool; fenceCount: c.Size; fences: c.Pointer<SharedFence>; signaledValues: c.Pointer<c.U64> }>) => new SharedTextureMemoryEndAccessState().set(val)
}
export class SharedTextureMemoryOpaqueFDDescriptor extends c.Struct<{ chain: ChainedStruct; vkImageCreateInfo: c.Pointer<c.Void>; memoryFD: c.I32; memoryTypeIndex: c.U32; allocationSize: c.U64; dedicatedAllocation: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 48, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $vkImageCreateInfo(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $memoryFD(){ return new c.I32(this.buffer, this.offset + 24) }
  get $memoryTypeIndex(){ return new c.U32(this.buffer, this.offset + 28) }
  get $allocationSize(){ return new c.U64(this.buffer, this.offset + 32) }
  get $dedicatedAllocation(){ return new Bool(this.buffer, this.offset + 40) }
  protected override _value = () => ({chain: this.$chain, vkImageCreateInfo: this.$vkImageCreateInfo, memoryFD: this.$memoryFD, memoryTypeIndex: this.$memoryTypeIndex, allocationSize: this.$allocationSize, dedicatedAllocation: this.$dedicatedAllocation})
  static new = (val: Partial<{ chain: ChainedStruct; vkImageCreateInfo: c.Pointer<c.Void>; memoryFD: c.I32; memoryTypeIndex: c.U32; allocationSize: c.U64; dedicatedAllocation: Bool }>) => new SharedTextureMemoryOpaqueFDDescriptor().set(val)
}
export class SharedTextureMemoryVkDedicatedAllocationDescriptor extends c.Struct<{ chain: ChainedStruct; dedicatedAllocation: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $dedicatedAllocation(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, dedicatedAllocation: this.$dedicatedAllocation})
  static new = (val: Partial<{ chain: ChainedStruct; dedicatedAllocation: Bool }>) => new SharedTextureMemoryVkDedicatedAllocationDescriptor().set(val)
}
export class SharedTextureMemoryVkImageLayoutBeginState extends c.Struct<{ chain: ChainedStruct; oldLayout: c.I32; newLayout: c.I32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $oldLayout(){ return new c.I32(this.buffer, this.offset + 16) }
  get $newLayout(){ return new c.I32(this.buffer, this.offset + 20) }
  protected override _value = () => ({chain: this.$chain, oldLayout: this.$oldLayout, newLayout: this.$newLayout})
  static new = (val: Partial<{ chain: ChainedStruct; oldLayout: c.I32; newLayout: c.I32 }>) => new SharedTextureMemoryVkImageLayoutBeginState().set(val)
}
export class SharedTextureMemoryVkImageLayoutEndState extends c.Struct<{ chain: ChainedStructOut; oldLayout: c.I32; newLayout: c.I32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $oldLayout(){ return new c.I32(this.buffer, this.offset + 16) }
  get $newLayout(){ return new c.I32(this.buffer, this.offset + 20) }
  protected override _value = () => ({chain: this.$chain, oldLayout: this.$oldLayout, newLayout: this.$newLayout})
  static new = (val: Partial<{ chain: ChainedStructOut; oldLayout: c.I32; newLayout: c.I32 }>) => new SharedTextureMemoryVkImageLayoutEndState().set(val)
}
export class SharedTextureMemoryZirconHandleDescriptor extends c.Struct<{ chain: ChainedStruct; memoryFD: c.U32; allocationSize: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $memoryFD(){ return new c.U32(this.buffer, this.offset + 16) }
  get $allocationSize(){ return new c.U64(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, memoryFD: this.$memoryFD, allocationSize: this.$allocationSize})
  static new = (val: Partial<{ chain: ChainedStruct; memoryFD: c.U32; allocationSize: c.U64 }>) => new SharedTextureMemoryZirconHandleDescriptor().set(val)
}
export class StaticSamplerBindingLayout extends c.Struct<{ chain: ChainedStruct; sampler: Sampler; sampledTextureBinding: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $sampler(){ return new Sampler(this.buffer, this.offset + 16) }
  get $sampledTextureBinding(){ return new c.U32(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, sampler: this.$sampler, sampledTextureBinding: this.$sampledTextureBinding})
  static new = (val: Partial<{ chain: ChainedStruct; sampler: Sampler; sampledTextureBinding: c.U32 }>) => new StaticSamplerBindingLayout().set(val)
}
export class StencilFaceState extends c.Struct<{ compare: CompareFunction; failOp: StencilOperation; depthFailOp: StencilOperation; passOp: StencilOperation }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 4)
  }
  get $compare(){ return new CompareFunction(this.buffer, this.offset + 0) }
  get $failOp(){ return new StencilOperation(this.buffer, this.offset + 4) }
  get $depthFailOp(){ return new StencilOperation(this.buffer, this.offset + 8) }
  get $passOp(){ return new StencilOperation(this.buffer, this.offset + 12) }
  protected override _value = () => ({compare: this.$compare, failOp: this.$failOp, depthFailOp: this.$depthFailOp, passOp: this.$passOp})
  static new = (val: Partial<{ compare: CompareFunction; failOp: StencilOperation; depthFailOp: StencilOperation; passOp: StencilOperation }>) => new StencilFaceState().set(val)
}
export class StorageTextureBindingLayout extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; access: StorageTextureAccess; format: TextureFormat; viewDimension: TextureViewDimension }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $access(){ return new StorageTextureAccess(this.buffer, this.offset + 8) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 12) }
  get $viewDimension(){ return new TextureViewDimension(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, access: this.$access, format: this.$format, viewDimension: this.$viewDimension})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; access: StorageTextureAccess; format: TextureFormat; viewDimension: TextureViewDimension }>) => new StorageTextureBindingLayout().set(val)
}
export class StringView extends c.Struct<{ data: c.Pointer<c.U8>; length: c.Size }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $data(){ return new c.Pointer<c.U8>(this.buffer, this.offset + 0) }
  get $length(){ return new c.Size(this.buffer, this.offset + 8) }
  protected override _value = () => ({data: this.$data, length: this.$length})
  static new = (val: Partial<{ data: c.Pointer<c.U8>; length: c.Size }>) => new StringView().set(val)
  protected override _setNative= (val: ArrayBuffer | Deno.PointerValue) =>{
    if (val instanceof ArrayBuffer) this.buffer = val
    else if (!val) return 
    else {
      const string = Deno.UnsafePointerView.getCString(val as Deno.PointerObject)
      const buf = new TextEncoder().encode(string + '\0')
      const str = new c.Type(buf.buffer as ArrayBuffer, 0, buf.length, 8)
      this.set({ data: str.ptr(), length: c.Size.new(BigInt(str.byteLength)) })
    }
  }
}
export class SupportedFeatures extends c.Struct<{ featureCount: c.Size; features: c.Pointer<FeatureName> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $featureCount(){ return new c.Size(this.buffer, this.offset + 0) }
  get $features(){ return new c.Pointer<FeatureName>(this.buffer, this.offset + 8) }
  protected override _value = () => ({featureCount: this.$featureCount, features: this.$features})
  static new = (val: Partial<{ featureCount: c.Size; features: c.Pointer<FeatureName> }>) => new SupportedFeatures().set(val)
}
export class SurfaceCapabilities extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; usages: TextureUsage; formatCount: c.Size; formats: c.Pointer<TextureFormat>; presentModeCount: c.Size; presentModes: c.Pointer<PresentMode>; alphaModeCount: c.Size; alphaModes: c.Pointer<CompositeAlphaMode> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $usages(){ return new TextureUsage(this.buffer, this.offset + 8) }
  get $formatCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $formats(){ return new c.Pointer<TextureFormat>(this.buffer, this.offset + 24) }
  get $presentModeCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $presentModes(){ return new c.Pointer<PresentMode>(this.buffer, this.offset + 40) }
  get $alphaModeCount(){ return new c.Size(this.buffer, this.offset + 48) }
  get $alphaModes(){ return new c.Pointer<CompositeAlphaMode>(this.buffer, this.offset + 56) }
  protected override _value = () => ({nextInChain: this.$nextInChain, usages: this.$usages, formatCount: this.$formatCount, formats: this.$formats, presentModeCount: this.$presentModeCount, presentModes: this.$presentModes, alphaModeCount: this.$alphaModeCount, alphaModes: this.$alphaModes})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; usages: TextureUsage; formatCount: c.Size; formats: c.Pointer<TextureFormat>; presentModeCount: c.Size; presentModes: c.Pointer<PresentMode>; alphaModeCount: c.Size; alphaModes: c.Pointer<CompositeAlphaMode> }>) => new SurfaceCapabilities().set(val)
}
export class SurfaceConfiguration extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; device: Device; format: TextureFormat; usage: TextureUsage; viewFormatCount: c.Size; viewFormats: c.Pointer<TextureFormat>; alphaMode: CompositeAlphaMode; width: c.U32; height: c.U32; presentMode: PresentMode }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $device(){ return new Device(this.buffer, this.offset + 8) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 16) }
  get $usage(){ return new TextureUsage(this.buffer, this.offset + 24) }
  get $viewFormatCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $viewFormats(){ return new c.Pointer<TextureFormat>(this.buffer, this.offset + 40) }
  get $alphaMode(){ return new CompositeAlphaMode(this.buffer, this.offset + 48) }
  get $width(){ return new c.U32(this.buffer, this.offset + 52) }
  get $height(){ return new c.U32(this.buffer, this.offset + 56) }
  get $presentMode(){ return new PresentMode(this.buffer, this.offset + 60) }
  protected override _value = () => ({nextInChain: this.$nextInChain, device: this.$device, format: this.$format, usage: this.$usage, viewFormatCount: this.$viewFormatCount, viewFormats: this.$viewFormats, alphaMode: this.$alphaMode, width: this.$width, height: this.$height, presentMode: this.$presentMode})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; device: Device; format: TextureFormat; usage: TextureUsage; viewFormatCount: c.Size; viewFormats: c.Pointer<TextureFormat>; alphaMode: CompositeAlphaMode; width: c.U32; height: c.U32; presentMode: PresentMode }>) => new SurfaceConfiguration().set(val)
}
export class SurfaceDescriptorFromWindowsCoreWindow extends c.Struct<{ chain: ChainedStruct; coreWindow: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $coreWindow(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, coreWindow: this.$coreWindow})
  static new = (val: Partial<{ chain: ChainedStruct; coreWindow: c.Pointer<c.Void> }>) => new SurfaceDescriptorFromWindowsCoreWindow().set(val)
}
export class SurfaceDescriptorFromWindowsSwapChainPanel extends c.Struct<{ chain: ChainedStruct; swapChainPanel: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $swapChainPanel(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, swapChainPanel: this.$swapChainPanel})
  static new = (val: Partial<{ chain: ChainedStruct; swapChainPanel: c.Pointer<c.Void> }>) => new SurfaceDescriptorFromWindowsSwapChainPanel().set(val)
}
export class SurfaceSourceXCBWindow extends c.Struct<{ chain: ChainedStruct; connection: c.Pointer<c.Void>; window: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $connection(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $window(){ return new c.U32(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, connection: this.$connection, window: this.$window})
  static new = (val: Partial<{ chain: ChainedStruct; connection: c.Pointer<c.Void>; window: c.U32 }>) => new SurfaceSourceXCBWindow().set(val)
}
export class SurfaceSourceAndroidNativeWindow extends c.Struct<{ chain: ChainedStruct; window: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $window(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, window: this.$window})
  static new = (val: Partial<{ chain: ChainedStruct; window: c.Pointer<c.Void> }>) => new SurfaceSourceAndroidNativeWindow().set(val)
}
export class SurfaceSourceMetalLayer extends c.Struct<{ chain: ChainedStruct; layer: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $layer(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, layer: this.$layer})
  static new = (val: Partial<{ chain: ChainedStruct; layer: c.Pointer<c.Void> }>) => new SurfaceSourceMetalLayer().set(val)
}
export class SurfaceSourceWaylandSurface extends c.Struct<{ chain: ChainedStruct; display: c.Pointer<c.Void>; surface: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $display(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $surface(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, display: this.$display, surface: this.$surface})
  static new = (val: Partial<{ chain: ChainedStruct; display: c.Pointer<c.Void>; surface: c.Pointer<c.Void> }>) => new SurfaceSourceWaylandSurface().set(val)
}
export class SurfaceSourceWindowsHWND extends c.Struct<{ chain: ChainedStruct; hinstance: c.Pointer<c.Void>; hwnd: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $hinstance(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $hwnd(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, hinstance: this.$hinstance, hwnd: this.$hwnd})
  static new = (val: Partial<{ chain: ChainedStruct; hinstance: c.Pointer<c.Void>; hwnd: c.Pointer<c.Void> }>) => new SurfaceSourceWindowsHWND().set(val)
}
export class SurfaceSourceXlibWindow extends c.Struct<{ chain: ChainedStruct; display: c.Pointer<c.Void>; window: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $display(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $window(){ return new c.U64(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, display: this.$display, window: this.$window})
  static new = (val: Partial<{ chain: ChainedStruct; display: c.Pointer<c.Void>; window: c.U64 }>) => new SurfaceSourceXlibWindow().set(val)
}
export class SurfaceTexture extends c.Struct<{ texture: Texture; suboptimal: Bool; status: SurfaceGetCurrentTextureStatus }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $texture(){ return new Texture(this.buffer, this.offset + 0) }
  get $suboptimal(){ return new Bool(this.buffer, this.offset + 8) }
  get $status(){ return new SurfaceGetCurrentTextureStatus(this.buffer, this.offset + 12) }
  protected override _value = () => ({texture: this.$texture, suboptimal: this.$suboptimal, status: this.$status})
  static new = (val: Partial<{ texture: Texture; suboptimal: Bool; status: SurfaceGetCurrentTextureStatus }>) => new SurfaceTexture().set(val)
}
export class TextureBindingLayout extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; sampleType: TextureSampleType; viewDimension: TextureViewDimension; multisampled: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $sampleType(){ return new TextureSampleType(this.buffer, this.offset + 8) }
  get $viewDimension(){ return new TextureViewDimension(this.buffer, this.offset + 12) }
  get $multisampled(){ return new Bool(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, sampleType: this.$sampleType, viewDimension: this.$viewDimension, multisampled: this.$multisampled})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; sampleType: TextureSampleType; viewDimension: TextureViewDimension; multisampled: Bool }>) => new TextureBindingLayout().set(val)
}
export class TextureBindingViewDimensionDescriptor extends c.Struct<{ chain: ChainedStruct; textureBindingViewDimension: TextureViewDimension }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $textureBindingViewDimension(){ return new TextureViewDimension(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, textureBindingViewDimension: this.$textureBindingViewDimension})
  static new = (val: Partial<{ chain: ChainedStruct; textureBindingViewDimension: TextureViewDimension }>) => new TextureBindingViewDimensionDescriptor().set(val)
}
export class TextureDataLayout extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; offset: c.U64; bytesPerRow: c.U32; rowsPerImage: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $offset(){ return new c.U64(this.buffer, this.offset + 8) }
  get $bytesPerRow(){ return new c.U32(this.buffer, this.offset + 16) }
  get $rowsPerImage(){ return new c.U32(this.buffer, this.offset + 20) }
  protected override _value = () => ({nextInChain: this.$nextInChain, offset: this.$offset, bytesPerRow: this.$bytesPerRow, rowsPerImage: this.$rowsPerImage})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; offset: c.U64; bytesPerRow: c.U32; rowsPerImage: c.U32 }>) => new TextureDataLayout().set(val)
}
export class UncapturedErrorCallbackInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; callback: ErrorCallback; userdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $callback(){ return new ErrorCallback(this.buffer, this.offset + 8) }
  get $userdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, callback: this.$callback, userdata: this.$userdata})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; callback: ErrorCallback; userdata: c.Pointer<c.Void> }>) => new UncapturedErrorCallbackInfo().set(val)
}
export class VertexAttribute extends c.Struct<{ format: VertexFormat; offset: c.U64; shaderLocation: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $format(){ return new VertexFormat(this.buffer, this.offset + 0) }
  get $offset(){ return new c.U64(this.buffer, this.offset + 8) }
  get $shaderLocation(){ return new c.U32(this.buffer, this.offset + 16) }
  protected override _value = () => ({format: this.$format, offset: this.$offset, shaderLocation: this.$shaderLocation})
  static new = (val: Partial<{ format: VertexFormat; offset: c.U64; shaderLocation: c.U32 }>) => new VertexAttribute().set(val)
}
export class YCbCrVkDescriptor extends c.Struct<{ chain: ChainedStruct; vkFormat: c.U32; vkYCbCrModel: c.U32; vkYCbCrRange: c.U32; vkComponentSwizzleRed: c.U32; vkComponentSwizzleGreen: c.U32; vkComponentSwizzleBlue: c.U32; vkComponentSwizzleAlpha: c.U32; vkXChromaOffset: c.U32; vkYChromaOffset: c.U32; vkChromaFilter: FilterMode; forceExplicitReconstruction: Bool; externalFormat: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 72, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $vkFormat(){ return new c.U32(this.buffer, this.offset + 16) }
  get $vkYCbCrModel(){ return new c.U32(this.buffer, this.offset + 20) }
  get $vkYCbCrRange(){ return new c.U32(this.buffer, this.offset + 24) }
  get $vkComponentSwizzleRed(){ return new c.U32(this.buffer, this.offset + 28) }
  get $vkComponentSwizzleGreen(){ return new c.U32(this.buffer, this.offset + 32) }
  get $vkComponentSwizzleBlue(){ return new c.U32(this.buffer, this.offset + 36) }
  get $vkComponentSwizzleAlpha(){ return new c.U32(this.buffer, this.offset + 40) }
  get $vkXChromaOffset(){ return new c.U32(this.buffer, this.offset + 44) }
  get $vkYChromaOffset(){ return new c.U32(this.buffer, this.offset + 48) }
  get $vkChromaFilter(){ return new FilterMode(this.buffer, this.offset + 52) }
  get $forceExplicitReconstruction(){ return new Bool(this.buffer, this.offset + 56) }
  get $externalFormat(){ return new c.U64(this.buffer, this.offset + 64) }
  protected override _value = () => ({chain: this.$chain, vkFormat: this.$vkFormat, vkYCbCrModel: this.$vkYCbCrModel, vkYCbCrRange: this.$vkYCbCrRange, vkComponentSwizzleRed: this.$vkComponentSwizzleRed, vkComponentSwizzleGreen: this.$vkComponentSwizzleGreen, vkComponentSwizzleBlue: this.$vkComponentSwizzleBlue, vkComponentSwizzleAlpha: this.$vkComponentSwizzleAlpha, vkXChromaOffset: this.$vkXChromaOffset, vkYChromaOffset: this.$vkYChromaOffset, vkChromaFilter: this.$vkChromaFilter, forceExplicitReconstruction: this.$forceExplicitReconstruction, externalFormat: this.$externalFormat})
  static new = (val: Partial<{ chain: ChainedStruct; vkFormat: c.U32; vkYCbCrModel: c.U32; vkYCbCrRange: c.U32; vkComponentSwizzleRed: c.U32; vkComponentSwizzleGreen: c.U32; vkComponentSwizzleBlue: c.U32; vkComponentSwizzleAlpha: c.U32; vkXChromaOffset: c.U32; vkYChromaOffset: c.U32; vkChromaFilter: FilterMode; forceExplicitReconstruction: Bool; externalFormat: c.U64 }>) => new YCbCrVkDescriptor().set(val)
}
export class AHardwareBufferProperties extends c.Struct<{ yCbCrInfo: YCbCrVkDescriptor }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 72, 8)
  }
  get $yCbCrInfo(){ return new YCbCrVkDescriptor(this.buffer, this.offset + 0) }
  protected override _value = () => ({yCbCrInfo: this.$yCbCrInfo})
  static new = (val: Partial<{ yCbCrInfo: YCbCrVkDescriptor }>) => new AHardwareBufferProperties().set(val)
}
export class AdapterInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; vendor: StringView; architecture: StringView; device: StringView; description: StringView; backendType: BackendType; adapterType: AdapterType; vendorID: c.U32; deviceID: c.U32; compatibilityMode: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 96, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $vendor(){ return new StringView(this.buffer, this.offset + 8) }
  get $architecture(){ return new StringView(this.buffer, this.offset + 24) }
  get $device(){ return new StringView(this.buffer, this.offset + 40) }
  get $description(){ return new StringView(this.buffer, this.offset + 56) }
  get $backendType(){ return new BackendType(this.buffer, this.offset + 72) }
  get $adapterType(){ return new AdapterType(this.buffer, this.offset + 76) }
  get $vendorID(){ return new c.U32(this.buffer, this.offset + 80) }
  get $deviceID(){ return new c.U32(this.buffer, this.offset + 84) }
  get $compatibilityMode(){ return new Bool(this.buffer, this.offset + 88) }
  protected override _value = () => ({nextInChain: this.$nextInChain, vendor: this.$vendor, architecture: this.$architecture, device: this.$device, description: this.$description, backendType: this.$backendType, adapterType: this.$adapterType, vendorID: this.$vendorID, deviceID: this.$deviceID, compatibilityMode: this.$compatibilityMode})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; vendor: StringView; architecture: StringView; device: StringView; description: StringView; backendType: BackendType; adapterType: AdapterType; vendorID: c.U32; deviceID: c.U32; compatibilityMode: Bool }>) => new AdapterInfo().set(val)
}
export class AdapterPropertiesMemoryHeaps extends c.Struct<{ chain: ChainedStructOut; heapCount: c.Size; heapInfo: c.Pointer<MemoryHeapInfo> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $heapCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $heapInfo(){ return new c.Pointer<MemoryHeapInfo>(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, heapCount: this.$heapCount, heapInfo: this.$heapInfo})
  static new = (val: Partial<{ chain: ChainedStructOut; heapCount: c.Size; heapInfo: c.Pointer<MemoryHeapInfo> }>) => new AdapterPropertiesMemoryHeaps().set(val)
}
export class BindGroupDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; layout: BindGroupLayout; entryCount: c.Size; entries: c.Pointer<BindGroupEntry> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 48, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $layout(){ return new BindGroupLayout(this.buffer, this.offset + 24) }
  get $entryCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $entries(){ return new c.Pointer<BindGroupEntry>(this.buffer, this.offset + 40) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, layout: this.$layout, entryCount: this.$entryCount, entries: this.$entries})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; layout: BindGroupLayout; entryCount: c.Size; entries: c.Pointer<BindGroupEntry> }>) => new BindGroupDescriptor().set(val)
}
export class BindGroupLayoutEntry extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; binding: c.U32; visibility: ShaderStage; buffer: BufferBindingLayout; sampler: SamplerBindingLayout; texture: TextureBindingLayout; storageTexture: StorageTextureBindingLayout }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 112, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $binding(){ return new c.U32(this.buffer, this.offset + 8) }
  get $visibility(){ return new ShaderStage(this.buffer, this.offset + 16) }
  get $buffer(){ return new BufferBindingLayout(this.buffer, this.offset + 24) }
  get $sampler(){ return new SamplerBindingLayout(this.buffer, this.offset + 48) }
  get $texture(){ return new TextureBindingLayout(this.buffer, this.offset + 64) }
  get $storageTexture(){ return new StorageTextureBindingLayout(this.buffer, this.offset + 88) }
  protected override _value = () => ({nextInChain: this.$nextInChain, binding: this.$binding, visibility: this.$visibility, buffer: this.$buffer, sampler: this.$sampler, texture: this.$texture, storageTexture: this.$storageTexture})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; binding: c.U32; visibility: ShaderStage; buffer: BufferBindingLayout; sampler: SamplerBindingLayout; texture: TextureBindingLayout; storageTexture: StorageTextureBindingLayout }>) => new BindGroupLayoutEntry().set(val)
}
export class BlendState extends c.Struct<{ color: BlendComponent; alpha: BlendComponent }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 4)
  }
  get $color(){ return new BlendComponent(this.buffer, this.offset + 0) }
  get $alpha(){ return new BlendComponent(this.buffer, this.offset + 12) }
  protected override _value = () => ({color: this.$color, alpha: this.$alpha})
  static new = (val: Partial<{ color: BlendComponent; alpha: BlendComponent }>) => new BlendState().set(val)
}
export class BufferDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; usage: BufferUsage; size: c.U64; mappedAtCreation: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 48, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $usage(){ return new BufferUsage(this.buffer, this.offset + 24) }
  get $size(){ return new c.U64(this.buffer, this.offset + 32) }
  get $mappedAtCreation(){ return new Bool(this.buffer, this.offset + 40) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, usage: this.$usage, size: this.$size, mappedAtCreation: this.$mappedAtCreation})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; usage: BufferUsage; size: c.U64; mappedAtCreation: Bool }>) => new BufferDescriptor().set(val)
}
export class CommandBufferDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new CommandBufferDescriptor().set(val)
}
export class CommandEncoderDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new CommandEncoderDescriptor().set(val)
}
export class CompilationMessage extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; message: StringView; type: CompilationMessageType; lineNum: c.U64; linePos: c.U64; offset: c.U64; length: c.U64; utf16LinePos: c.U64; utf16Offset: c.U64; utf16Length: c.U64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 88, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $message(){ return new StringView(this.buffer, this.offset + 8) }
  get $type(){ return new CompilationMessageType(this.buffer, this.offset + 24) }
  get $lineNum(){ return new c.U64(this.buffer, this.offset + 32) }
  get $linePos(){ return new c.U64(this.buffer, this.offset + 40) }
  get $offset(){ return new c.U64(this.buffer, this.offset + 48) }
  get $length(){ return new c.U64(this.buffer, this.offset + 56) }
  get $utf16LinePos(){ return new c.U64(this.buffer, this.offset + 64) }
  get $utf16Offset(){ return new c.U64(this.buffer, this.offset + 72) }
  get $utf16Length(){ return new c.U64(this.buffer, this.offset + 80) }
  protected override _value = () => ({nextInChain: this.$nextInChain, message: this.$message, type: this.$type, lineNum: this.$lineNum, linePos: this.$linePos, offset: this.$offset, length: this.$length, utf16LinePos: this.$utf16LinePos, utf16Offset: this.$utf16Offset, utf16Length: this.$utf16Length})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; message: StringView; type: CompilationMessageType; lineNum: c.U64; linePos: c.U64; offset: c.U64; length: c.U64; utf16LinePos: c.U64; utf16Offset: c.U64; utf16Length: c.U64 }>) => new CompilationMessage().set(val)
}
export class ComputePassDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; timestampWrites: c.Pointer<ComputePassTimestampWrites> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $timestampWrites(){ return new c.Pointer<ComputePassTimestampWrites>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, timestampWrites: this.$timestampWrites})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; timestampWrites: c.Pointer<ComputePassTimestampWrites> }>) => new ComputePassDescriptor().set(val)
}
export class ConstantEntry extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; key: StringView; value: c.F64 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $key(){ return new StringView(this.buffer, this.offset + 8) }
  get $value(){ return new c.F64(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, key: this.$key, value: this.$value})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; key: StringView; value: c.F64 }>) => new ConstantEntry().set(val)
}
export class DawnCacheDeviceDescriptor extends c.Struct<{ chain: ChainedStruct; isolationKey: StringView; loadDataFunction: DawnLoadCacheDataFunction; storeDataFunction: DawnStoreCacheDataFunction; functionUserdata: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 56, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $isolationKey(){ return new StringView(this.buffer, this.offset + 16) }
  get $loadDataFunction(){ return new DawnLoadCacheDataFunction(this.buffer, this.offset + 32) }
  get $storeDataFunction(){ return new DawnStoreCacheDataFunction(this.buffer, this.offset + 40) }
  get $functionUserdata(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 48) }
  protected override _value = () => ({chain: this.$chain, isolationKey: this.$isolationKey, loadDataFunction: this.$loadDataFunction, storeDataFunction: this.$storeDataFunction, functionUserdata: this.$functionUserdata})
  static new = (val: Partial<{ chain: ChainedStruct; isolationKey: StringView; loadDataFunction: DawnLoadCacheDataFunction; storeDataFunction: DawnStoreCacheDataFunction; functionUserdata: c.Pointer<c.Void> }>) => new DawnCacheDeviceDescriptor().set(val)
}
export class DepthStencilState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; format: TextureFormat; depthWriteEnabled: OptionalBool; depthCompare: CompareFunction; stencilFront: StencilFaceState; stencilBack: StencilFaceState; stencilReadMask: c.U32; stencilWriteMask: c.U32; depthBias: c.I32; depthBiasSlopeScale: c.F32; depthBiasClamp: c.F32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 72, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 8) }
  get $depthWriteEnabled(){ return new OptionalBool(this.buffer, this.offset + 12) }
  get $depthCompare(){ return new CompareFunction(this.buffer, this.offset + 16) }
  get $stencilFront(){ return new StencilFaceState(this.buffer, this.offset + 20) }
  get $stencilBack(){ return new StencilFaceState(this.buffer, this.offset + 36) }
  get $stencilReadMask(){ return new c.U32(this.buffer, this.offset + 52) }
  get $stencilWriteMask(){ return new c.U32(this.buffer, this.offset + 56) }
  get $depthBias(){ return new c.I32(this.buffer, this.offset + 60) }
  get $depthBiasSlopeScale(){ return new c.F32(this.buffer, this.offset + 64) }
  get $depthBiasClamp(){ return new c.F32(this.buffer, this.offset + 68) }
  protected override _value = () => ({nextInChain: this.$nextInChain, format: this.$format, depthWriteEnabled: this.$depthWriteEnabled, depthCompare: this.$depthCompare, stencilFront: this.$stencilFront, stencilBack: this.$stencilBack, stencilReadMask: this.$stencilReadMask, stencilWriteMask: this.$stencilWriteMask, depthBias: this.$depthBias, depthBiasSlopeScale: this.$depthBiasSlopeScale, depthBiasClamp: this.$depthBiasClamp})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; format: TextureFormat; depthWriteEnabled: OptionalBool; depthCompare: CompareFunction; stencilFront: StencilFaceState; stencilBack: StencilFaceState; stencilReadMask: c.U32; stencilWriteMask: c.U32; depthBias: c.I32; depthBiasSlopeScale: c.F32; depthBiasClamp: c.F32 }>) => new DepthStencilState().set(val)
}
export class DrmFormatCapabilities extends c.Struct<{ chain: ChainedStructOut; propertiesCount: c.Size; properties: c.Pointer<DrmFormatProperties> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $propertiesCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $properties(){ return new c.Pointer<DrmFormatProperties>(this.buffer, this.offset + 24) }
  protected override _value = () => ({chain: this.$chain, propertiesCount: this.$propertiesCount, properties: this.$properties})
  static new = (val: Partial<{ chain: ChainedStructOut; propertiesCount: c.Size; properties: c.Pointer<DrmFormatProperties> }>) => new DrmFormatCapabilities().set(val)
}
export class ExternalTextureDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; plane0: TextureView; plane1: TextureView; cropOrigin: Origin2D; cropSize: Extent2D; apparentSize: Extent2D; doYuvToRgbConversionOnly: Bool; yuvToRgbConversionMatrix: c.Pointer<c.F32>; srcTransferFunctionParameters: c.Pointer<c.F32>; dstTransferFunctionParameters: c.Pointer<c.F32>; gamutConversionMatrix: c.Pointer<c.F32>; mirrored: Bool; rotation: ExternalTextureRotation }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 112, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $plane0(){ return new TextureView(this.buffer, this.offset + 24) }
  get $plane1(){ return new TextureView(this.buffer, this.offset + 32) }
  get $cropOrigin(){ return new Origin2D(this.buffer, this.offset + 40) }
  get $cropSize(){ return new Extent2D(this.buffer, this.offset + 48) }
  get $apparentSize(){ return new Extent2D(this.buffer, this.offset + 56) }
  get $doYuvToRgbConversionOnly(){ return new Bool(this.buffer, this.offset + 64) }
  get $yuvToRgbConversionMatrix(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 72) }
  get $srcTransferFunctionParameters(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 80) }
  get $dstTransferFunctionParameters(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 88) }
  get $gamutConversionMatrix(){ return new c.Pointer<c.F32>(this.buffer, this.offset + 96) }
  get $mirrored(){ return new Bool(this.buffer, this.offset + 104) }
  get $rotation(){ return new ExternalTextureRotation(this.buffer, this.offset + 108) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, plane0: this.$plane0, plane1: this.$plane1, cropOrigin: this.$cropOrigin, cropSize: this.$cropSize, apparentSize: this.$apparentSize, doYuvToRgbConversionOnly: this.$doYuvToRgbConversionOnly, yuvToRgbConversionMatrix: this.$yuvToRgbConversionMatrix, srcTransferFunctionParameters: this.$srcTransferFunctionParameters, dstTransferFunctionParameters: this.$dstTransferFunctionParameters, gamutConversionMatrix: this.$gamutConversionMatrix, mirrored: this.$mirrored, rotation: this.$rotation})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; plane0: TextureView; plane1: TextureView; cropOrigin: Origin2D; cropSize: Extent2D; apparentSize: Extent2D; doYuvToRgbConversionOnly: Bool; yuvToRgbConversionMatrix: c.Pointer<c.F32>; srcTransferFunctionParameters: c.Pointer<c.F32>; dstTransferFunctionParameters: c.Pointer<c.F32>; gamutConversionMatrix: c.Pointer<c.F32>; mirrored: Bool; rotation: ExternalTextureRotation }>) => new ExternalTextureDescriptor().set(val)
}
export class FutureWaitInfo extends c.Struct<{ future: Future; completed: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $future(){ return new Future(this.buffer, this.offset + 0) }
  get $completed(){ return new Bool(this.buffer, this.offset + 8) }
  protected override _value = () => ({future: this.$future, completed: this.$completed})
  static new = (val: Partial<{ future: Future; completed: Bool }>) => new FutureWaitInfo().set(val)
}
export class ImageCopyBuffer extends c.Struct<{ layout: TextureDataLayout; buffer: Buffer }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $layout(){ return new TextureDataLayout(this.buffer, this.offset + 0) }
  get $buffer(){ return new Buffer(this.buffer, this.offset + 24) }
  protected override _value = () => ({layout: this.$layout, buffer: this.$buffer})
  static new = (val: Partial<{ layout: TextureDataLayout; buffer: Buffer }>) => new ImageCopyBuffer().set(val)
}
export class ImageCopyExternalTexture extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; externalTexture: ExternalTexture; origin: Origin3D; naturalSize: Extent2D }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $externalTexture(){ return new ExternalTexture(this.buffer, this.offset + 8) }
  get $origin(){ return new Origin3D(this.buffer, this.offset + 16) }
  get $naturalSize(){ return new Extent2D(this.buffer, this.offset + 28) }
  protected override _value = () => ({nextInChain: this.$nextInChain, externalTexture: this.$externalTexture, origin: this.$origin, naturalSize: this.$naturalSize})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; externalTexture: ExternalTexture; origin: Origin3D; naturalSize: Extent2D }>) => new ImageCopyExternalTexture().set(val)
}
export class ImageCopyTexture extends c.Struct<{ texture: Texture; mipLevel: c.U32; origin: Origin3D; aspect: TextureAspect }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $texture(){ return new Texture(this.buffer, this.offset + 0) }
  get $mipLevel(){ return new c.U32(this.buffer, this.offset + 8) }
  get $origin(){ return new Origin3D(this.buffer, this.offset + 12) }
  get $aspect(){ return new TextureAspect(this.buffer, this.offset + 24) }
  protected override _value = () => ({texture: this.$texture, mipLevel: this.$mipLevel, origin: this.$origin, aspect: this.$aspect})
  static new = (val: Partial<{ texture: Texture; mipLevel: c.U32; origin: Origin3D; aspect: TextureAspect }>) => new ImageCopyTexture().set(val)
}
export class InstanceDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; features: InstanceFeatures }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $features(){ return new InstanceFeatures(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, features: this.$features})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; features: InstanceFeatures }>) => new InstanceDescriptor().set(val)
}
export class PipelineLayoutDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; bindGroupLayoutCount: c.Size; bindGroupLayouts: c.Pointer<BindGroupLayout>; immediateDataRangeByteSize: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 48, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $bindGroupLayoutCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $bindGroupLayouts(){ return new c.Pointer<BindGroupLayout>(this.buffer, this.offset + 32) }
  get $immediateDataRangeByteSize(){ return new c.U32(this.buffer, this.offset + 40) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, bindGroupLayoutCount: this.$bindGroupLayoutCount, bindGroupLayouts: this.$bindGroupLayouts, immediateDataRangeByteSize: this.$immediateDataRangeByteSize})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; bindGroupLayoutCount: c.Size; bindGroupLayouts: c.Pointer<BindGroupLayout>; immediateDataRangeByteSize: c.U32 }>) => new PipelineLayoutDescriptor().set(val)
}
export class PipelineLayoutPixelLocalStorage extends c.Struct<{ chain: ChainedStruct; totalPixelLocalStorageSize: c.U64; storageAttachmentCount: c.Size; storageAttachments: c.Pointer<PipelineLayoutStorageAttachment> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $totalPixelLocalStorageSize(){ return new c.U64(this.buffer, this.offset + 16) }
  get $storageAttachmentCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $storageAttachments(){ return new c.Pointer<PipelineLayoutStorageAttachment>(this.buffer, this.offset + 32) }
  protected override _value = () => ({chain: this.$chain, totalPixelLocalStorageSize: this.$totalPixelLocalStorageSize, storageAttachmentCount: this.$storageAttachmentCount, storageAttachments: this.$storageAttachments})
  static new = (val: Partial<{ chain: ChainedStruct; totalPixelLocalStorageSize: c.U64; storageAttachmentCount: c.Size; storageAttachments: c.Pointer<PipelineLayoutStorageAttachment> }>) => new PipelineLayoutPixelLocalStorage().set(val)
}
export class QuerySetDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; type: QueryType; count: c.U32 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $type(){ return new QueryType(this.buffer, this.offset + 24) }
  get $count(){ return new c.U32(this.buffer, this.offset + 28) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, type: this.$type, count: this.$count})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; type: QueryType; count: c.U32 }>) => new QuerySetDescriptor().set(val)
}
export class QueueDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new QueueDescriptor().set(val)
}
export class RenderBundleDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new RenderBundleDescriptor().set(val)
}
export class RenderBundleEncoderDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; colorFormatCount: c.Size; colorFormats: c.Pointer<TextureFormat>; depthStencilFormat: TextureFormat; sampleCount: c.U32; depthReadOnly: Bool; stencilReadOnly: Bool }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 56, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $colorFormatCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $colorFormats(){ return new c.Pointer<TextureFormat>(this.buffer, this.offset + 32) }
  get $depthStencilFormat(){ return new TextureFormat(this.buffer, this.offset + 40) }
  get $sampleCount(){ return new c.U32(this.buffer, this.offset + 44) }
  get $depthReadOnly(){ return new Bool(this.buffer, this.offset + 48) }
  get $stencilReadOnly(){ return new Bool(this.buffer, this.offset + 52) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, colorFormatCount: this.$colorFormatCount, colorFormats: this.$colorFormats, depthStencilFormat: this.$depthStencilFormat, sampleCount: this.$sampleCount, depthReadOnly: this.$depthReadOnly, stencilReadOnly: this.$stencilReadOnly})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; colorFormatCount: c.Size; colorFormats: c.Pointer<TextureFormat>; depthStencilFormat: TextureFormat; sampleCount: c.U32; depthReadOnly: Bool; stencilReadOnly: Bool }>) => new RenderBundleEncoderDescriptor().set(val)
}
export class RenderPassColorAttachment extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; view: TextureView; depthSlice: c.U32; resolveTarget: TextureView; loadOp: LoadOp; storeOp: StoreOp; clearValue: Color }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 72, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $view(){ return new TextureView(this.buffer, this.offset + 8) }
  get $depthSlice(){ return new c.U32(this.buffer, this.offset + 16) }
  get $resolveTarget(){ return new TextureView(this.buffer, this.offset + 24) }
  get $loadOp(){ return new LoadOp(this.buffer, this.offset + 32) }
  get $storeOp(){ return new StoreOp(this.buffer, this.offset + 36) }
  get $clearValue(){ return new Color(this.buffer, this.offset + 40) }
  protected override _value = () => ({nextInChain: this.$nextInChain, view: this.$view, depthSlice: this.$depthSlice, resolveTarget: this.$resolveTarget, loadOp: this.$loadOp, storeOp: this.$storeOp, clearValue: this.$clearValue})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; view: TextureView; depthSlice: c.U32; resolveTarget: TextureView; loadOp: LoadOp; storeOp: StoreOp; clearValue: Color }>) => new RenderPassColorAttachment().set(val)
}
export class RenderPassStorageAttachment extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; offset: c.U64; storage: TextureView; loadOp: LoadOp; storeOp: StoreOp; clearValue: Color }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $offset(){ return new c.U64(this.buffer, this.offset + 8) }
  get $storage(){ return new TextureView(this.buffer, this.offset + 16) }
  get $loadOp(){ return new LoadOp(this.buffer, this.offset + 24) }
  get $storeOp(){ return new StoreOp(this.buffer, this.offset + 28) }
  get $clearValue(){ return new Color(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, offset: this.$offset, storage: this.$storage, loadOp: this.$loadOp, storeOp: this.$storeOp, clearValue: this.$clearValue})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; offset: c.U64; storage: TextureView; loadOp: LoadOp; storeOp: StoreOp; clearValue: Color }>) => new RenderPassStorageAttachment().set(val)
}
export class RequiredLimits extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; limits: Limits }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 168, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $limits(){ return new Limits(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, limits: this.$limits})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; limits: Limits }>) => new RequiredLimits().set(val)
}
export class SamplerDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; addressModeU: AddressMode; addressModeV: AddressMode; addressModeW: AddressMode; magFilter: FilterMode; minFilter: FilterMode; mipmapFilter: MipmapFilterMode; lodMinClamp: c.F32; lodMaxClamp: c.F32; compare: CompareFunction; maxAnisotropy: c.U16 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $addressModeU(){ return new AddressMode(this.buffer, this.offset + 24) }
  get $addressModeV(){ return new AddressMode(this.buffer, this.offset + 28) }
  get $addressModeW(){ return new AddressMode(this.buffer, this.offset + 32) }
  get $magFilter(){ return new FilterMode(this.buffer, this.offset + 36) }
  get $minFilter(){ return new FilterMode(this.buffer, this.offset + 40) }
  get $mipmapFilter(){ return new MipmapFilterMode(this.buffer, this.offset + 44) }
  get $lodMinClamp(){ return new c.F32(this.buffer, this.offset + 48) }
  get $lodMaxClamp(){ return new c.F32(this.buffer, this.offset + 52) }
  get $compare(){ return new CompareFunction(this.buffer, this.offset + 56) }
  get $maxAnisotropy(){ return new c.U16(this.buffer, this.offset + 60) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, addressModeU: this.$addressModeU, addressModeV: this.$addressModeV, addressModeW: this.$addressModeW, magFilter: this.$magFilter, minFilter: this.$minFilter, mipmapFilter: this.$mipmapFilter, lodMinClamp: this.$lodMinClamp, lodMaxClamp: this.$lodMaxClamp, compare: this.$compare, maxAnisotropy: this.$maxAnisotropy})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; addressModeU: AddressMode; addressModeV: AddressMode; addressModeW: AddressMode; magFilter: FilterMode; minFilter: FilterMode; mipmapFilter: MipmapFilterMode; lodMinClamp: c.F32; lodMaxClamp: c.F32; compare: CompareFunction; maxAnisotropy: c.U16 }>) => new SamplerDescriptor().set(val)
}
export class ShaderModuleDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new ShaderModuleDescriptor().set(val)
}
export class ShaderSourceWGSL extends c.Struct<{ chain: ChainedStruct; code: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $code(){ return new StringView(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, code: this.$code})
  static new = (val: Partial<{ chain: ChainedStruct; code: StringView }>) => new ShaderSourceWGSL().set(val)
}
export class SharedBufferMemoryDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new SharedBufferMemoryDescriptor().set(val)
}
export class SharedFenceDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new SharedFenceDescriptor().set(val)
}
export class SharedTextureMemoryAHardwareBufferProperties extends c.Struct<{ chain: ChainedStructOut; yCbCrInfo: YCbCrVkDescriptor }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 88, 8)
  }
  get $chain(){ return new ChainedStructOut(this.buffer, this.offset + 0) }
  get $yCbCrInfo(){ return new YCbCrVkDescriptor(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, yCbCrInfo: this.$yCbCrInfo})
  static new = (val: Partial<{ chain: ChainedStructOut; yCbCrInfo: YCbCrVkDescriptor }>) => new SharedTextureMemoryAHardwareBufferProperties().set(val)
}
export class SharedTextureMemoryDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new SharedTextureMemoryDescriptor().set(val)
}
export class SharedTextureMemoryDmaBufDescriptor extends c.Struct<{ chain: ChainedStruct; size: Extent3D; drmFormat: c.U32; drmModifier: c.U64; planeCount: c.Size; planes: c.Pointer<SharedTextureMemoryDmaBufPlane> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 56, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $size(){ return new Extent3D(this.buffer, this.offset + 16) }
  get $drmFormat(){ return new c.U32(this.buffer, this.offset + 28) }
  get $drmModifier(){ return new c.U64(this.buffer, this.offset + 32) }
  get $planeCount(){ return new c.Size(this.buffer, this.offset + 40) }
  get $planes(){ return new c.Pointer<SharedTextureMemoryDmaBufPlane>(this.buffer, this.offset + 48) }
  protected override _value = () => ({chain: this.$chain, size: this.$size, drmFormat: this.$drmFormat, drmModifier: this.$drmModifier, planeCount: this.$planeCount, planes: this.$planes})
  static new = (val: Partial<{ chain: ChainedStruct; size: Extent3D; drmFormat: c.U32; drmModifier: c.U64; planeCount: c.Size; planes: c.Pointer<SharedTextureMemoryDmaBufPlane> }>) => new SharedTextureMemoryDmaBufDescriptor().set(val)
}
export class SharedTextureMemoryProperties extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; usage: TextureUsage; size: Extent3D; format: TextureFormat }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $usage(){ return new TextureUsage(this.buffer, this.offset + 8) }
  get $size(){ return new Extent3D(this.buffer, this.offset + 16) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 28) }
  protected override _value = () => ({nextInChain: this.$nextInChain, usage: this.$usage, size: this.$size, format: this.$format})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; usage: TextureUsage; size: Extent3D; format: TextureFormat }>) => new SharedTextureMemoryProperties().set(val)
}
export class SupportedLimits extends c.Struct<{ nextInChain: c.Pointer<ChainedStructOut>; limits: Limits }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 168, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $limits(){ return new Limits(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, limits: this.$limits})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStructOut>; limits: Limits }>) => new SupportedLimits().set(val)
}
export class SurfaceDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView }>) => new SurfaceDescriptor().set(val)
}
export class SurfaceSourceCanvasHTMLSelector_Emscripten extends c.Struct<{ chain: ChainedStruct; selector: StringView }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $selector(){ return new StringView(this.buffer, this.offset + 16) }
  protected override _value = () => ({chain: this.$chain, selector: this.$selector})
  static new = (val: Partial<{ chain: ChainedStruct; selector: StringView }>) => new SurfaceSourceCanvasHTMLSelector_Emscripten().set(val)
}
export class TextureDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; usage: TextureUsage; dimension: TextureDimension; size: Extent3D; format: TextureFormat; mipLevelCount: c.U32; sampleCount: c.U32; viewFormatCount: c.Size; viewFormats: c.Pointer<TextureFormat> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 80, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $usage(){ return new TextureUsage(this.buffer, this.offset + 24) }
  get $dimension(){ return new TextureDimension(this.buffer, this.offset + 32) }
  get $size(){ return new Extent3D(this.buffer, this.offset + 36) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 48) }
  get $mipLevelCount(){ return new c.U32(this.buffer, this.offset + 52) }
  get $sampleCount(){ return new c.U32(this.buffer, this.offset + 56) }
  get $viewFormatCount(){ return new c.Size(this.buffer, this.offset + 64) }
  get $viewFormats(){ return new c.Pointer<TextureFormat>(this.buffer, this.offset + 72) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, usage: this.$usage, dimension: this.$dimension, size: this.$size, format: this.$format, mipLevelCount: this.$mipLevelCount, sampleCount: this.$sampleCount, viewFormatCount: this.$viewFormatCount, viewFormats: this.$viewFormats})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; usage: TextureUsage; dimension: TextureDimension; size: Extent3D; format: TextureFormat; mipLevelCount: c.U32; sampleCount: c.U32; viewFormatCount: c.Size; viewFormats: c.Pointer<TextureFormat> }>) => new TextureDescriptor().set(val)
}
export class TextureViewDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; format: TextureFormat; dimension: TextureViewDimension; baseMipLevel: c.U32; mipLevelCount: c.U32; baseArrayLayer: c.U32; arrayLayerCount: c.U32; aspect: TextureAspect; usage: TextureUsage }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 24) }
  get $dimension(){ return new TextureViewDimension(this.buffer, this.offset + 28) }
  get $baseMipLevel(){ return new c.U32(this.buffer, this.offset + 32) }
  get $mipLevelCount(){ return new c.U32(this.buffer, this.offset + 36) }
  get $baseArrayLayer(){ return new c.U32(this.buffer, this.offset + 40) }
  get $arrayLayerCount(){ return new c.U32(this.buffer, this.offset + 44) }
  get $aspect(){ return new TextureAspect(this.buffer, this.offset + 48) }
  get $usage(){ return new TextureUsage(this.buffer, this.offset + 56) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, format: this.$format, dimension: this.$dimension, baseMipLevel: this.$baseMipLevel, mipLevelCount: this.$mipLevelCount, baseArrayLayer: this.$baseArrayLayer, arrayLayerCount: this.$arrayLayerCount, aspect: this.$aspect, usage: this.$usage})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; format: TextureFormat; dimension: TextureViewDimension; baseMipLevel: c.U32; mipLevelCount: c.U32; baseArrayLayer: c.U32; arrayLayerCount: c.U32; aspect: TextureAspect; usage: TextureUsage }>) => new TextureViewDescriptor().set(val)
}
export class VertexBufferLayout extends c.Struct<{ arrayStride: c.U64; stepMode: VertexStepMode; attributeCount: c.Size; attributes: c.Pointer<VertexAttribute> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $arrayStride(){ return new c.U64(this.buffer, this.offset + 0) }
  get $stepMode(){ return new VertexStepMode(this.buffer, this.offset + 8) }
  get $attributeCount(){ return new c.Size(this.buffer, this.offset + 16) }
  get $attributes(){ return new c.Pointer<VertexAttribute>(this.buffer, this.offset + 24) }
  protected override _value = () => ({arrayStride: this.$arrayStride, stepMode: this.$stepMode, attributeCount: this.$attributeCount, attributes: this.$attributes})
  static new = (val: Partial<{ arrayStride: c.U64; stepMode: VertexStepMode; attributeCount: c.Size; attributes: c.Pointer<VertexAttribute> }>) => new VertexBufferLayout().set(val)
}
export class BindGroupLayoutDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; entryCount: c.Size; entries: c.Pointer<BindGroupLayoutEntry> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $entryCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $entries(){ return new c.Pointer<BindGroupLayoutEntry>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, entryCount: this.$entryCount, entries: this.$entries})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; entryCount: c.Size; entries: c.Pointer<BindGroupLayoutEntry> }>) => new BindGroupLayoutDescriptor().set(val)
}
export class ColorTargetState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; format: TextureFormat; blend: c.Pointer<BlendState>; writeMask: ColorWriteMask }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $format(){ return new TextureFormat(this.buffer, this.offset + 8) }
  get $blend(){ return new c.Pointer<BlendState>(this.buffer, this.offset + 16) }
  get $writeMask(){ return new ColorWriteMask(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, format: this.$format, blend: this.$blend, writeMask: this.$writeMask})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; format: TextureFormat; blend: c.Pointer<BlendState>; writeMask: ColorWriteMask }>) => new ColorTargetState().set(val)
}
export class CompilationInfo extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; messageCount: c.Size; messages: c.Pointer<CompilationMessage> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 24, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $messageCount(){ return new c.Size(this.buffer, this.offset + 8) }
  get $messages(){ return new c.Pointer<CompilationMessage>(this.buffer, this.offset + 16) }
  protected override _value = () => ({nextInChain: this.$nextInChain, messageCount: this.$messageCount, messages: this.$messages})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; messageCount: c.Size; messages: c.Pointer<CompilationMessage> }>) => new CompilationInfo().set(val)
}
export class ComputeState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; module: ShaderModule; entryPoint: StringView; constantCount: c.Size; constants: c.Pointer<ConstantEntry> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 48, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $module(){ return new ShaderModule(this.buffer, this.offset + 8) }
  get $entryPoint(){ return new StringView(this.buffer, this.offset + 16) }
  get $constantCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $constants(){ return new c.Pointer<ConstantEntry>(this.buffer, this.offset + 40) }
  protected override _value = () => ({nextInChain: this.$nextInChain, module: this.$module, entryPoint: this.$entryPoint, constantCount: this.$constantCount, constants: this.$constants})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; module: ShaderModule; entryPoint: StringView; constantCount: c.Size; constants: c.Pointer<ConstantEntry> }>) => new ComputeState().set(val)
}
export class DeviceDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; requiredFeatureCount: c.Size; requiredFeatures: c.Pointer<FeatureName>; requiredLimits: c.Pointer<RequiredLimits>; defaultQueue: QueueDescriptor; deviceLostCallbackInfo2: DeviceLostCallbackInfo2; uncapturedErrorCallbackInfo2: UncapturedErrorCallbackInfo2 }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 144, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $requiredFeatureCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $requiredFeatures(){ return new c.Pointer<FeatureName>(this.buffer, this.offset + 32) }
  get $requiredLimits(){ return new c.Pointer<RequiredLimits>(this.buffer, this.offset + 40) }
  get $defaultQueue(){ return new QueueDescriptor(this.buffer, this.offset + 48) }
  get $deviceLostCallbackInfo2(){ return new DeviceLostCallbackInfo2(this.buffer, this.offset + 72) }
  get $uncapturedErrorCallbackInfo2(){ return new UncapturedErrorCallbackInfo2(this.buffer, this.offset + 112) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, requiredFeatureCount: this.$requiredFeatureCount, requiredFeatures: this.$requiredFeatures, requiredLimits: this.$requiredLimits, defaultQueue: this.$defaultQueue, deviceLostCallbackInfo2: this.$deviceLostCallbackInfo2, uncapturedErrorCallbackInfo2: this.$uncapturedErrorCallbackInfo2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; requiredFeatureCount: c.Size; requiredFeatures: c.Pointer<FeatureName>; requiredLimits: c.Pointer<RequiredLimits>; defaultQueue: QueueDescriptor; deviceLostCallbackInfo2: DeviceLostCallbackInfo2; uncapturedErrorCallbackInfo2: UncapturedErrorCallbackInfo2 }>) => new DeviceDescriptor().set(val)
}
export class RenderPassDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; colorAttachmentCount: c.Size; colorAttachments: c.Pointer<RenderPassColorAttachment>; depthStencilAttachment: c.Pointer<RenderPassDepthStencilAttachment>; occlusionQuerySet: QuerySet; timestampWrites: c.Pointer<RenderPassTimestampWrites> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $colorAttachmentCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $colorAttachments(){ return new c.Pointer<RenderPassColorAttachment>(this.buffer, this.offset + 32) }
  get $depthStencilAttachment(){ return new c.Pointer<RenderPassDepthStencilAttachment>(this.buffer, this.offset + 40) }
  get $occlusionQuerySet(){ return new QuerySet(this.buffer, this.offset + 48) }
  get $timestampWrites(){ return new c.Pointer<RenderPassTimestampWrites>(this.buffer, this.offset + 56) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, colorAttachmentCount: this.$colorAttachmentCount, colorAttachments: this.$colorAttachments, depthStencilAttachment: this.$depthStencilAttachment, occlusionQuerySet: this.$occlusionQuerySet, timestampWrites: this.$timestampWrites})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; colorAttachmentCount: c.Size; colorAttachments: c.Pointer<RenderPassColorAttachment>; depthStencilAttachment: c.Pointer<RenderPassDepthStencilAttachment>; occlusionQuerySet: QuerySet; timestampWrites: c.Pointer<RenderPassTimestampWrites> }>) => new RenderPassDescriptor().set(val)
}
export class RenderPassPixelLocalStorage extends c.Struct<{ chain: ChainedStruct; totalPixelLocalStorageSize: c.U64; storageAttachmentCount: c.Size; storageAttachments: c.Pointer<RenderPassStorageAttachment> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $chain(){ return new ChainedStruct(this.buffer, this.offset + 0) }
  get $totalPixelLocalStorageSize(){ return new c.U64(this.buffer, this.offset + 16) }
  get $storageAttachmentCount(){ return new c.Size(this.buffer, this.offset + 24) }
  get $storageAttachments(){ return new c.Pointer<RenderPassStorageAttachment>(this.buffer, this.offset + 32) }
  protected override _value = () => ({chain: this.$chain, totalPixelLocalStorageSize: this.$totalPixelLocalStorageSize, storageAttachmentCount: this.$storageAttachmentCount, storageAttachments: this.$storageAttachments})
  static new = (val: Partial<{ chain: ChainedStruct; totalPixelLocalStorageSize: c.U64; storageAttachmentCount: c.Size; storageAttachments: c.Pointer<RenderPassStorageAttachment> }>) => new RenderPassPixelLocalStorage().set(val)
}
export class VertexState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; module: ShaderModule; entryPoint: StringView; constantCount: c.Size; constants: c.Pointer<ConstantEntry>; bufferCount: c.Size; buffers: c.Pointer<VertexBufferLayout> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $module(){ return new ShaderModule(this.buffer, this.offset + 8) }
  get $entryPoint(){ return new StringView(this.buffer, this.offset + 16) }
  get $constantCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $constants(){ return new c.Pointer<ConstantEntry>(this.buffer, this.offset + 40) }
  get $bufferCount(){ return new c.Size(this.buffer, this.offset + 48) }
  get $buffers(){ return new c.Pointer<VertexBufferLayout>(this.buffer, this.offset + 56) }
  protected override _value = () => ({nextInChain: this.$nextInChain, module: this.$module, entryPoint: this.$entryPoint, constantCount: this.$constantCount, constants: this.$constants, bufferCount: this.$bufferCount, buffers: this.$buffers})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; module: ShaderModule; entryPoint: StringView; constantCount: c.Size; constants: c.Pointer<ConstantEntry>; bufferCount: c.Size; buffers: c.Pointer<VertexBufferLayout> }>) => new VertexState().set(val)
}
export class ComputePipelineDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; layout: PipelineLayout; compute: ComputeState }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 80, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $layout(){ return new PipelineLayout(this.buffer, this.offset + 24) }
  get $compute(){ return new ComputeState(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, layout: this.$layout, compute: this.$compute})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; layout: PipelineLayout; compute: ComputeState }>) => new ComputePipelineDescriptor().set(val)
}
export class FragmentState extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; module: ShaderModule; entryPoint: StringView; constantCount: c.Size; constants: c.Pointer<ConstantEntry>; targetCount: c.Size; targets: c.Pointer<ColorTargetState> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 64, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $module(){ return new ShaderModule(this.buffer, this.offset + 8) }
  get $entryPoint(){ return new StringView(this.buffer, this.offset + 16) }
  get $constantCount(){ return new c.Size(this.buffer, this.offset + 32) }
  get $constants(){ return new c.Pointer<ConstantEntry>(this.buffer, this.offset + 40) }
  get $targetCount(){ return new c.Size(this.buffer, this.offset + 48) }
  get $targets(){ return new c.Pointer<ColorTargetState>(this.buffer, this.offset + 56) }
  protected override _value = () => ({nextInChain: this.$nextInChain, module: this.$module, entryPoint: this.$entryPoint, constantCount: this.$constantCount, constants: this.$constants, targetCount: this.$targetCount, targets: this.$targets})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; module: ShaderModule; entryPoint: StringView; constantCount: c.Size; constants: c.Pointer<ConstantEntry>; targetCount: c.Size; targets: c.Pointer<ColorTargetState> }>) => new FragmentState().set(val)
}
export class RenderPipelineDescriptor extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; layout: PipelineLayout; vertex: VertexState; primitive: PrimitiveState; depthStencil: c.Pointer<DepthStencilState>; multisample: MultisampleState; fragment: c.Pointer<FragmentState> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 168, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $label(){ return new StringView(this.buffer, this.offset + 8) }
  get $layout(){ return new PipelineLayout(this.buffer, this.offset + 24) }
  get $vertex(){ return new VertexState(this.buffer, this.offset + 32) }
  get $primitive(){ return new PrimitiveState(this.buffer, this.offset + 96) }
  get $depthStencil(){ return new c.Pointer<DepthStencilState>(this.buffer, this.offset + 128) }
  get $multisample(){ return new MultisampleState(this.buffer, this.offset + 136) }
  get $fragment(){ return new c.Pointer<FragmentState>(this.buffer, this.offset + 160) }
  protected override _value = () => ({nextInChain: this.$nextInChain, label: this.$label, layout: this.$layout, vertex: this.$vertex, primitive: this.$primitive, depthStencil: this.$depthStencil, multisample: this.$multisample, fragment: this.$fragment})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; label: StringView; layout: PipelineLayout; vertex: VertexState; primitive: PrimitiveState; depthStencil: c.Pointer<DepthStencilState>; multisample: MultisampleState; fragment: c.Pointer<FragmentState> }>) => new RenderPipelineDescriptor().set(val)
}
export class ChainedStruct extends c.Struct<{ next: c.Pointer<ChainedStruct>; sType: SType }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $next(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $sType(){ return new SType(this.buffer, this.offset + 8) }
  protected override _value = () => ({next: this.$next, sType: this.$sType})
  static new = (val: Partial<{ next: c.Pointer<ChainedStruct>; sType: SType }>) => new ChainedStruct().set(val)
}
export class ChainedStructOut extends c.Struct<{ next: c.Pointer<ChainedStructOut>; sType: SType }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 16, 8)
  }
  get $next(){ return new c.Pointer<ChainedStructOut>(this.buffer, this.offset + 0) }
  get $sType(){ return new SType(this.buffer, this.offset + 8) }
  protected override _value = () => ({next: this.$next, sType: this.$sType})
  static new = (val: Partial<{ next: c.Pointer<ChainedStructOut>; sType: SType }>) => new ChainedStructOut().set(val)
}
export class BufferMapCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: BufferMapCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new BufferMapCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: BufferMapCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new BufferMapCallbackInfo2().set(val)
}
export class CompilationInfoCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CompilationInfoCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new CompilationInfoCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CompilationInfoCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new CompilationInfoCallbackInfo2().set(val)
}
export class CreateComputePipelineAsyncCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateComputePipelineAsyncCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new CreateComputePipelineAsyncCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateComputePipelineAsyncCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new CreateComputePipelineAsyncCallbackInfo2().set(val)
}
export class CreateRenderPipelineAsyncCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateRenderPipelineAsyncCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new CreateRenderPipelineAsyncCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: CreateRenderPipelineAsyncCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new CreateRenderPipelineAsyncCallbackInfo2().set(val)
}
export class DeviceLostCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: DeviceLostCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new DeviceLostCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: DeviceLostCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new DeviceLostCallbackInfo2().set(val)
}
export class PopErrorScopeCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: PopErrorScopeCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new PopErrorScopeCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: PopErrorScopeCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new PopErrorScopeCallbackInfo2().set(val)
}
export class QueueWorkDoneCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: QueueWorkDoneCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new QueueWorkDoneCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: QueueWorkDoneCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new QueueWorkDoneCallbackInfo2().set(val)
}
export class RequestAdapterCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestAdapterCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new RequestAdapterCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestAdapterCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new RequestAdapterCallbackInfo2().set(val)
}
export class RequestDeviceCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestDeviceCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 40, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $mode(){ return new CallbackMode(this.buffer, this.offset + 8) }
  get $callback(){ return new RequestDeviceCallback2(this.buffer, this.offset + 16) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 32) }
  protected override _value = () => ({nextInChain: this.$nextInChain, mode: this.$mode, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; mode: CallbackMode; callback: RequestDeviceCallback2; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new RequestDeviceCallbackInfo2().set(val)
}
export class UncapturedErrorCallbackInfo2 extends c.Struct<{ nextInChain: c.Pointer<ChainedStruct>; callback: UncapturedErrorCallback; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, 32, 8)
  }
  get $nextInChain(){ return new c.Pointer<ChainedStruct>(this.buffer, this.offset + 0) }
  get $callback(){ return new UncapturedErrorCallback(this.buffer, this.offset + 8) }
  get $userdata1(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 16) }
  get $userdata2(){ return new c.Pointer<c.Void>(this.buffer, this.offset + 24) }
  protected override _value = () => ({nextInChain: this.$nextInChain, callback: this.$callback, userdata1: this.$userdata1, userdata2: this.$userdata2})
  static new = (val: Partial<{ nextInChain: c.Pointer<ChainedStruct>; callback: UncapturedErrorCallback; userdata1: c.Pointer<c.Void>; userdata2: c.Pointer<c.Void> }>) => new UncapturedErrorCallbackInfo2().set(val)
}

// types
export class Flags extends c.U64 {}
export class Bool extends c.U32 {}
export class Adapter extends c.Pointer<AdapterImpl> {}
export class BindGroup extends c.Pointer<BindGroupImpl> {}
export class BindGroupLayout extends c.Pointer<BindGroupLayoutImpl> {}
export class Buffer extends c.Pointer<BufferImpl> {}
export class CommandBuffer extends c.Pointer<CommandBufferImpl> {}
export class CommandEncoder extends c.Pointer<CommandEncoderImpl> {}
export class ComputePassEncoder extends c.Pointer<ComputePassEncoderImpl> {}
export class ComputePipeline extends c.Pointer<ComputePipelineImpl> {}
export class Device extends c.Pointer<DeviceImpl> {}
export class ExternalTexture extends c.Pointer<ExternalTextureImpl> {}
export class Instance extends c.Pointer<InstanceImpl> {}
export class PipelineLayout extends c.Pointer<PipelineLayoutImpl> {}
export class QuerySet extends c.Pointer<QuerySetImpl> {}
export class Queue extends c.Pointer<QueueImpl> {}
export class RenderBundle extends c.Pointer<RenderBundleImpl> {}
export class RenderBundleEncoder extends c.Pointer<RenderBundleEncoderImpl> {}
export class RenderPassEncoder extends c.Pointer<RenderPassEncoderImpl> {}
export class RenderPipeline extends c.Pointer<RenderPipelineImpl> {}
export class Sampler extends c.Pointer<SamplerImpl> {}
export class ShaderModule extends c.Pointer<ShaderModuleImpl> {}
export class SharedBufferMemory extends c.Pointer<SharedBufferMemoryImpl> {}
export class SharedFence extends c.Pointer<SharedFenceImpl> {}
export class SharedTextureMemory extends c.Pointer<SharedTextureMemoryImpl> {}
export class Surface extends c.Pointer<SurfaceImpl> {}
export class Texture extends c.Pointer<TextureImpl> {}
export class TextureView extends c.Pointer<TextureViewImpl> {}
export class BufferUsage extends Flags {}
export class ColorWriteMask extends Flags {}
export class HeapProperty extends Flags {}
export class MapMode extends Flags {}
export class ShaderStage extends Flags {}
export class TextureUsage extends Flags {}
export class RenderPassDescriptorMaxDrawCount extends RenderPassMaxDrawCount {}
export class ShaderModuleSPIRVDescriptor extends ShaderSourceSPIRV {}
export class ShaderModuleWGSLDescriptor extends ShaderSourceWGSL {}
export class SurfaceDescriptorFromAndroidNativeWindow extends SurfaceSourceAndroidNativeWindow {}
export class SurfaceDescriptorFromCanvasHTMLSelector extends SurfaceSourceCanvasHTMLSelector_Emscripten {}
export class SurfaceDescriptorFromMetalLayer extends SurfaceSourceMetalLayer {}
export class SurfaceDescriptorFromWaylandSurface extends SurfaceSourceWaylandSurface {}
export class SurfaceDescriptorFromWindowsHWND extends SurfaceSourceWindowsHWND {}
export class SurfaceDescriptorFromXcbWindow extends SurfaceSourceXCBWindow {}
export class SurfaceDescriptorFromXlibWindow extends SurfaceSourceXlibWindow {}

// callbacks
export class BufferMapCallback extends c.Function<[status: BufferMapAsyncStatus, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer'])
  }
  protected override _fn = (fn: (status: BufferMapAsyncStatus, userdata: c.Pointer<any>) => void) => (status: any, userdata: any) => void fn(new BufferMapAsyncStatus().setNative(status), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: BufferMapAsyncStatus, userdata: c.Pointer<any>) => void) => new BufferMapCallback().set(fn)
}
export class Callback extends c.Function<[userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (userdata: c.Pointer<any>) => void) => (userdata: any) => void fn(new c.Pointer<any>().setNative(userdata))
  static new = (fn: (userdata: c.Pointer<any>) => void) => new Callback().set(fn)
}
export class CompilationInfoCallback extends c.Function<[status: CompilationInfoRequestStatus, compilationInfo: c.Pointer<CompilationInfo>, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: CompilationInfoRequestStatus, compilationInfo: c.Pointer<CompilationInfo>, userdata: c.Pointer<any>) => void) => (status: any, compilationInfo: any, userdata: any) => void fn(new CompilationInfoRequestStatus().setNative(status), new c.Pointer<CompilationInfo>().setNative(compilationInfo), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: CompilationInfoRequestStatus, compilationInfo: c.Pointer<CompilationInfo>, userdata: c.Pointer<any>) => void) => new CompilationInfoCallback().set(fn)
}
export class CreateComputePipelineAsyncCallback extends c.Function<[status: CreatePipelineAsyncStatus, pipeline: ComputePipeline, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (status: CreatePipelineAsyncStatus, pipeline: ComputePipeline, message: StringView, userdata: c.Pointer<any>) => void) => (status: any, pipeline: any, message: any, userdata: any) => void fn(new CreatePipelineAsyncStatus().setNative(status), new ComputePipeline().setNative(pipeline), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: CreatePipelineAsyncStatus, pipeline: ComputePipeline, message: StringView, userdata: c.Pointer<any>) => void) => new CreateComputePipelineAsyncCallback().set(fn)
}
export class CreateRenderPipelineAsyncCallback extends c.Function<[status: CreatePipelineAsyncStatus, pipeline: RenderPipeline, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (status: CreatePipelineAsyncStatus, pipeline: RenderPipeline, message: StringView, userdata: c.Pointer<any>) => void) => (status: any, pipeline: any, message: any, userdata: any) => void fn(new CreatePipelineAsyncStatus().setNative(status), new RenderPipeline().setNative(pipeline), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: CreatePipelineAsyncStatus, pipeline: RenderPipeline, message: StringView, userdata: c.Pointer<any>) => void) => new CreateRenderPipelineAsyncCallback().set(fn)
}
export class DawnLoadCacheDataFunction extends c.Function<[key: c.Pointer<any>, keySize: c.Size, value: c.Pointer<any>, valueSize: c.Size, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (key: c.Pointer<any>, keySize: c.Size, value: c.Pointer<any>, valueSize: c.Size, userdata: c.Pointer<any>) => void) => (key: any, keySize: any, value: any, valueSize: any, userdata: any) => void fn(new c.Pointer<any>().setNative(key), new c.Size().setNative(keySize), new c.Pointer<any>().setNative(value), new c.Size().setNative(valueSize), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (key: c.Pointer<any>, keySize: c.Size, value: c.Pointer<any>, valueSize: c.Size, userdata: c.Pointer<any>) => void) => new DawnLoadCacheDataFunction().set(fn)
}
export class DawnStoreCacheDataFunction extends c.Function<[key: c.Pointer<any>, keySize: c.Size, value: c.Pointer<any>, valueSize: c.Size, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (key: c.Pointer<any>, keySize: c.Size, value: c.Pointer<any>, valueSize: c.Size, userdata: c.Pointer<any>) => void) => (key: any, keySize: any, value: any, valueSize: any, userdata: any) => void fn(new c.Pointer<any>().setNative(key), new c.Size().setNative(keySize), new c.Pointer<any>().setNative(value), new c.Size().setNative(valueSize), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (key: c.Pointer<any>, keySize: c.Size, value: c.Pointer<any>, valueSize: c.Size, userdata: c.Pointer<any>) => void) => new DawnStoreCacheDataFunction().set(fn)
}
export class DeviceLostCallback extends c.Function<[reason: DeviceLostReason, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (reason: DeviceLostReason, message: StringView, userdata: c.Pointer<any>) => void) => (reason: any, message: any, userdata: any) => void fn(new DeviceLostReason().setNative(reason), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (reason: DeviceLostReason, message: StringView, userdata: c.Pointer<any>) => void) => new DeviceLostCallback().set(fn)
}
export class DeviceLostCallbackNew extends c.Function<[device: c.Pointer<Device>, reason: DeviceLostReason, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (device: c.Pointer<Device>, reason: DeviceLostReason, message: StringView, userdata: c.Pointer<any>) => void) => (device: any, reason: any, message: any, userdata: any) => void fn(new c.Pointer<Device>().setNative(device), new DeviceLostReason().setNative(reason), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (device: c.Pointer<Device>, reason: DeviceLostReason, message: StringView, userdata: c.Pointer<any>) => void) => new DeviceLostCallbackNew().set(fn)
}
export class ErrorCallback extends c.Function<[type: ErrorType, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (type: ErrorType, message: StringView, userdata: c.Pointer<any>) => void) => (type: any, message: any, userdata: any) => void fn(new ErrorType().setNative(type), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (type: ErrorType, message: StringView, userdata: c.Pointer<any>) => void) => new ErrorCallback().set(fn)
}
export class LoggingCallback extends c.Function<[type: LoggingType, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (type: LoggingType, message: StringView, userdata: c.Pointer<any>) => void) => (type: any, message: any, userdata: any) => void fn(new LoggingType().setNative(type), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (type: LoggingType, message: StringView, userdata: c.Pointer<any>) => void) => new LoggingCallback().set(fn)
}
export class PopErrorScopeCallback extends c.Function<[status: PopErrorScopeStatus, type: ErrorType, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'u32', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (status: PopErrorScopeStatus, type: ErrorType, message: StringView, userdata: c.Pointer<any>) => void) => (status: any, type: any, message: any, userdata: any) => void fn(new PopErrorScopeStatus().setNative(status), new ErrorType().setNative(type), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: PopErrorScopeStatus, type: ErrorType, message: StringView, userdata: c.Pointer<any>) => void) => new PopErrorScopeCallback().set(fn)
}
export class Proc extends c.Function<[]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, [])
  }
  protected override _fn = (fn: () => void) => () => void fn()
  static new = (fn: () => void) => new Proc().set(fn)
}
export class QueueWorkDoneCallback extends c.Function<[status: QueueWorkDoneStatus, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer'])
  }
  protected override _fn = (fn: (status: QueueWorkDoneStatus, userdata: c.Pointer<any>) => void) => (status: any, userdata: any) => void fn(new QueueWorkDoneStatus().setNative(status), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: QueueWorkDoneStatus, userdata: c.Pointer<any>) => void) => new QueueWorkDoneCallback().set(fn)
}
export class RequestAdapterCallback extends c.Function<[status: RequestAdapterStatus, adapter: Adapter, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (status: RequestAdapterStatus, adapter: Adapter, message: StringView, userdata: c.Pointer<any>) => void) => (status: any, adapter: any, message: any, userdata: any) => void fn(new RequestAdapterStatus().setNative(status), new Adapter().setNative(adapter), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: RequestAdapterStatus, adapter: Adapter, message: StringView, userdata: c.Pointer<any>) => void) => new RequestAdapterCallback().set(fn)
}
export class RequestDeviceCallback extends c.Function<[status: RequestDeviceStatus, device: Device, message: StringView, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer'])
  }
  protected override _fn = (fn: (status: RequestDeviceStatus, device: Device, message: StringView, userdata: c.Pointer<any>) => void) => (status: any, device: any, message: any, userdata: any) => void fn(new RequestDeviceStatus().setNative(status), new Device().setNative(device), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (status: RequestDeviceStatus, device: Device, message: StringView, userdata: c.Pointer<any>) => void) => new RequestDeviceCallback().set(fn)
}
export class BufferMapCallback2 extends c.Function<[status: MapAsyncStatus, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: MapAsyncStatus, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, message: any, userdata1: any, userdata2: any) => void fn(new MapAsyncStatus().setNative(status), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: MapAsyncStatus, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new BufferMapCallback2().set(fn)
}
export class CompilationInfoCallback2 extends c.Function<[status: CompilationInfoRequestStatus, compilationInfo: c.Pointer<CompilationInfo>, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: CompilationInfoRequestStatus, compilationInfo: c.Pointer<CompilationInfo>, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, compilationInfo: any, userdata1: any, userdata2: any) => void fn(new CompilationInfoRequestStatus().setNative(status), new c.Pointer<CompilationInfo>().setNative(compilationInfo), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: CompilationInfoRequestStatus, compilationInfo: c.Pointer<CompilationInfo>, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new CompilationInfoCallback2().set(fn)
}
export class CreateComputePipelineAsyncCallback2 extends c.Function<[status: CreatePipelineAsyncStatus, pipeline: ComputePipeline, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: CreatePipelineAsyncStatus, pipeline: ComputePipeline, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, pipeline: any, message: any, userdata1: any, userdata2: any) => void fn(new CreatePipelineAsyncStatus().setNative(status), new ComputePipeline().setNative(pipeline), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: CreatePipelineAsyncStatus, pipeline: ComputePipeline, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new CreateComputePipelineAsyncCallback2().set(fn)
}
export class CreateRenderPipelineAsyncCallback2 extends c.Function<[status: CreatePipelineAsyncStatus, pipeline: RenderPipeline, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: CreatePipelineAsyncStatus, pipeline: RenderPipeline, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, pipeline: any, message: any, userdata1: any, userdata2: any) => void fn(new CreatePipelineAsyncStatus().setNative(status), new RenderPipeline().setNative(pipeline), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: CreatePipelineAsyncStatus, pipeline: RenderPipeline, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new CreateRenderPipelineAsyncCallback2().set(fn)
}
export class DeviceLostCallback2 extends c.Function<[device: c.Pointer<Device>, reason: DeviceLostReason, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: c.Pointer<Device>, reason: DeviceLostReason, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (device: any, reason: any, message: any, userdata1: any, userdata2: any) => void fn(new c.Pointer<Device>().setNative(device), new DeviceLostReason().setNative(reason), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (device: c.Pointer<Device>, reason: DeviceLostReason, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new DeviceLostCallback2().set(fn)
}
export class PopErrorScopeCallback2 extends c.Function<[status: PopErrorScopeStatus, type: ErrorType, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'u32', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: PopErrorScopeStatus, type: ErrorType, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, type: any, message: any, userdata1: any, userdata2: any) => void fn(new PopErrorScopeStatus().setNative(status), new ErrorType().setNative(type), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: PopErrorScopeStatus, type: ErrorType, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new PopErrorScopeCallback2().set(fn)
}
export class QueueWorkDoneCallback2 extends c.Function<[status: QueueWorkDoneStatus, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: QueueWorkDoneStatus, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, userdata1: any, userdata2: any) => void fn(new QueueWorkDoneStatus().setNative(status), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: QueueWorkDoneStatus, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new QueueWorkDoneCallback2().set(fn)
}
export class RequestAdapterCallback2 extends c.Function<[status: RequestAdapterStatus, adapter: Adapter, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: RequestAdapterStatus, adapter: Adapter, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, adapter: any, message: any, userdata1: any, userdata2: any) => void fn(new RequestAdapterStatus().setNative(status), new Adapter().setNative(adapter), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: RequestAdapterStatus, adapter: Adapter, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new RequestAdapterCallback2().set(fn)
}
export class RequestDeviceCallback2 extends c.Function<[status: RequestDeviceStatus, device: Device, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['u32', 'pointer', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (status: RequestDeviceStatus, device: Device, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (status: any, device: any, message: any, userdata1: any, userdata2: any) => void fn(new RequestDeviceStatus().setNative(status), new Device().setNative(device), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (status: RequestDeviceStatus, device: Device, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new RequestDeviceCallback2().set(fn)
}
export class UncapturedErrorCallback extends c.Function<[device: c.Pointer<Device>, type: ErrorType, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'buffer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: c.Pointer<Device>, type: ErrorType, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => (device: any, type: any, message: any, userdata1: any, userdata2: any) => void fn(new c.Pointer<Device>().setNative(device), new ErrorType().setNative(type), new StringView().setNative(message), new c.Pointer<any>().setNative(userdata1), new c.Pointer<any>().setNative(userdata2))
  static new = (fn: (device: c.Pointer<Device>, type: ErrorType, message: StringView, userdata1: c.Pointer<any>, userdata2: c.Pointer<any>) => void) => new UncapturedErrorCallback().set(fn)
}
export class ProcAdapterInfoFreeMembers extends c.Function<[value: AdapterInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: AdapterInfo) => void) => (value: any) => void fn(new AdapterInfo().setNative(value))
  static new = (fn: (value: AdapterInfo) => void) => new ProcAdapterInfoFreeMembers().set(fn)
}
export class ProcAdapterPropertiesMemoryHeapsFreeMembers extends c.Function<[value: AdapterPropertiesMemoryHeaps]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: AdapterPropertiesMemoryHeaps) => void) => (value: any) => void fn(new AdapterPropertiesMemoryHeaps().setNative(value))
  static new = (fn: (value: AdapterPropertiesMemoryHeaps) => void) => new ProcAdapterPropertiesMemoryHeapsFreeMembers().set(fn)
}
export class ProcCreateInstance extends c.Function<[descriptor: c.Pointer<InstanceDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (descriptor: c.Pointer<InstanceDescriptor>) => void) => (descriptor: any) => void fn(new c.Pointer<InstanceDescriptor>().setNative(descriptor))
  static new = (fn: (descriptor: c.Pointer<InstanceDescriptor>) => void) => new ProcCreateInstance().set(fn)
}
export class ProcDrmFormatCapabilitiesFreeMembers extends c.Function<[value: DrmFormatCapabilities]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: DrmFormatCapabilities) => void) => (value: any) => void fn(new DrmFormatCapabilities().setNative(value))
  static new = (fn: (value: DrmFormatCapabilities) => void) => new ProcDrmFormatCapabilitiesFreeMembers().set(fn)
}
export class ProcGetInstanceFeatures extends c.Function<[features: c.Pointer<InstanceFeatures>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (features: c.Pointer<InstanceFeatures>) => void) => (features: any) => void fn(new c.Pointer<InstanceFeatures>().setNative(features))
  static new = (fn: (features: c.Pointer<InstanceFeatures>) => void) => new ProcGetInstanceFeatures().set(fn)
}
export class ProcGetProcAddress extends c.Function<[procName: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (procName: StringView) => void) => (procName: any) => void fn(new StringView().setNative(procName))
  static new = (fn: (procName: StringView) => void) => new ProcGetProcAddress().set(fn)
}
export class ProcSharedBufferMemoryEndAccessStateFreeMembers extends c.Function<[value: SharedBufferMemoryEndAccessState]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: SharedBufferMemoryEndAccessState) => void) => (value: any) => void fn(new SharedBufferMemoryEndAccessState().setNative(value))
  static new = (fn: (value: SharedBufferMemoryEndAccessState) => void) => new ProcSharedBufferMemoryEndAccessStateFreeMembers().set(fn)
}
export class ProcSharedTextureMemoryEndAccessStateFreeMembers extends c.Function<[value: SharedTextureMemoryEndAccessState]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: SharedTextureMemoryEndAccessState) => void) => (value: any) => void fn(new SharedTextureMemoryEndAccessState().setNative(value))
  static new = (fn: (value: SharedTextureMemoryEndAccessState) => void) => new ProcSharedTextureMemoryEndAccessStateFreeMembers().set(fn)
}
export class ProcSupportedFeaturesFreeMembers extends c.Function<[value: SupportedFeatures]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: SupportedFeatures) => void) => (value: any) => void fn(new SupportedFeatures().setNative(value))
  static new = (fn: (value: SupportedFeatures) => void) => new ProcSupportedFeaturesFreeMembers().set(fn)
}
export class ProcSurfaceCapabilitiesFreeMembers extends c.Function<[value: SurfaceCapabilities]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['buffer'])
  }
  protected override _fn = (fn: (value: SurfaceCapabilities) => void) => (value: any) => void fn(new SurfaceCapabilities().setNative(value))
  static new = (fn: (value: SurfaceCapabilities) => void) => new ProcSurfaceCapabilitiesFreeMembers().set(fn)
}
export class ProcAdapterCreateDevice extends c.Function<[adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>) => void) => (adapter: any, descriptor: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<DeviceDescriptor>().setNative(descriptor))
  static new = (fn: (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>) => void) => new ProcAdapterCreateDevice().set(fn)
}
export class ProcAdapterGetFeatures extends c.Function<[adapter: Adapter, features: c.Pointer<SupportedFeatures>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter, features: c.Pointer<SupportedFeatures>) => void) => (adapter: any, features: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<SupportedFeatures>().setNative(features))
  static new = (fn: (adapter: Adapter, features: c.Pointer<SupportedFeatures>) => void) => new ProcAdapterGetFeatures().set(fn)
}
export class ProcAdapterGetFormatCapabilities extends c.Function<[adapter: Adapter, format: TextureFormat, capabilities: c.Pointer<FormatCapabilities>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter, format: TextureFormat, capabilities: c.Pointer<FormatCapabilities>) => void) => (adapter: any, format: any, capabilities: any) => void fn(new Adapter().setNative(adapter), new TextureFormat().setNative(format), new c.Pointer<FormatCapabilities>().setNative(capabilities))
  static new = (fn: (adapter: Adapter, format: TextureFormat, capabilities: c.Pointer<FormatCapabilities>) => void) => new ProcAdapterGetFormatCapabilities().set(fn)
}
export class ProcAdapterGetInfo extends c.Function<[adapter: Adapter, info: c.Pointer<AdapterInfo>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter, info: c.Pointer<AdapterInfo>) => void) => (adapter: any, info: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<AdapterInfo>().setNative(info))
  static new = (fn: (adapter: Adapter, info: c.Pointer<AdapterInfo>) => void) => new ProcAdapterGetInfo().set(fn)
}
export class ProcAdapterGetInstance extends c.Function<[adapter: Adapter]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter) => void) => (adapter: any) => void fn(new Adapter().setNative(adapter))
  static new = (fn: (adapter: Adapter) => void) => new ProcAdapterGetInstance().set(fn)
}
export class ProcAdapterGetLimits extends c.Function<[adapter: Adapter, limits: c.Pointer<SupportedLimits>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter, limits: c.Pointer<SupportedLimits>) => void) => (adapter: any, limits: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<SupportedLimits>().setNative(limits))
  static new = (fn: (adapter: Adapter, limits: c.Pointer<SupportedLimits>) => void) => new ProcAdapterGetLimits().set(fn)
}
export class ProcAdapterHasFeature extends c.Function<[adapter: Adapter, feature: FeatureName]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (adapter: Adapter, feature: FeatureName) => void) => (adapter: any, feature: any) => void fn(new Adapter().setNative(adapter), new FeatureName().setNative(feature))
  static new = (fn: (adapter: Adapter, feature: FeatureName) => void) => new ProcAdapterHasFeature().set(fn)
}
export class ProcAdapterRequestDevice extends c.Function<[adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>, callback: RequestDeviceCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>, callback: RequestDeviceCallback, userdata: c.Pointer<any>) => void) => (adapter: any, descriptor: any, callback: any, userdata: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<DeviceDescriptor>().setNative(descriptor), new RequestDeviceCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>, callback: RequestDeviceCallback, userdata: c.Pointer<any>) => void) => new ProcAdapterRequestDevice().set(fn)
}
export class ProcAdapterRequestDevice2 extends c.Function<[adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo2) => void) => (adapter: any, options: any, callbackInfo: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<DeviceDescriptor>().setNative(options), new RequestDeviceCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo2) => void) => new ProcAdapterRequestDevice2().set(fn)
}
export class ProcAdapterRequestDeviceF extends c.Function<[adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo) => void) => (adapter: any, options: any, callbackInfo: any) => void fn(new Adapter().setNative(adapter), new c.Pointer<DeviceDescriptor>().setNative(options), new RequestDeviceCallbackInfo().setNative(callbackInfo))
  static new = (fn: (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo) => void) => new ProcAdapterRequestDeviceF().set(fn)
}
export class ProcAdapterAddRef extends c.Function<[adapter: Adapter]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter) => void) => (adapter: any) => void fn(new Adapter().setNative(adapter))
  static new = (fn: (adapter: Adapter) => void) => new ProcAdapterAddRef().set(fn)
}
export class ProcAdapterRelease extends c.Function<[adapter: Adapter]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (adapter: Adapter) => void) => (adapter: any) => void fn(new Adapter().setNative(adapter))
  static new = (fn: (adapter: Adapter) => void) => new ProcAdapterRelease().set(fn)
}
export class ProcBindGroupSetLabel extends c.Function<[bindGroup: BindGroup, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (bindGroup: BindGroup, label: StringView) => void) => (bindGroup: any, label: any) => void fn(new BindGroup().setNative(bindGroup), new StringView().setNative(label))
  static new = (fn: (bindGroup: BindGroup, label: StringView) => void) => new ProcBindGroupSetLabel().set(fn)
}
export class ProcBindGroupAddRef extends c.Function<[bindGroup: BindGroup]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (bindGroup: BindGroup) => void) => (bindGroup: any) => void fn(new BindGroup().setNative(bindGroup))
  static new = (fn: (bindGroup: BindGroup) => void) => new ProcBindGroupAddRef().set(fn)
}
export class ProcBindGroupRelease extends c.Function<[bindGroup: BindGroup]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (bindGroup: BindGroup) => void) => (bindGroup: any) => void fn(new BindGroup().setNative(bindGroup))
  static new = (fn: (bindGroup: BindGroup) => void) => new ProcBindGroupRelease().set(fn)
}
export class ProcBindGroupLayoutSetLabel extends c.Function<[bindGroupLayout: BindGroupLayout, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (bindGroupLayout: BindGroupLayout, label: StringView) => void) => (bindGroupLayout: any, label: any) => void fn(new BindGroupLayout().setNative(bindGroupLayout), new StringView().setNative(label))
  static new = (fn: (bindGroupLayout: BindGroupLayout, label: StringView) => void) => new ProcBindGroupLayoutSetLabel().set(fn)
}
export class ProcBindGroupLayoutAddRef extends c.Function<[bindGroupLayout: BindGroupLayout]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (bindGroupLayout: BindGroupLayout) => void) => (bindGroupLayout: any) => void fn(new BindGroupLayout().setNative(bindGroupLayout))
  static new = (fn: (bindGroupLayout: BindGroupLayout) => void) => new ProcBindGroupLayoutAddRef().set(fn)
}
export class ProcBindGroupLayoutRelease extends c.Function<[bindGroupLayout: BindGroupLayout]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (bindGroupLayout: BindGroupLayout) => void) => (bindGroupLayout: any) => void fn(new BindGroupLayout().setNative(bindGroupLayout))
  static new = (fn: (bindGroupLayout: BindGroupLayout) => void) => new ProcBindGroupLayoutRelease().set(fn)
}
export class ProcBufferDestroy extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferDestroy().set(fn)
}
export class ProcBufferGetConstMappedRange extends c.Function<[buffer: Buffer, offset: c.Size, size: c.Size]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'usize'])
  }
  protected override _fn = (fn: (buffer: Buffer, offset: c.Size, size: c.Size) => void) => (buffer: any, offset: any, size: any) => void fn(new Buffer().setNative(buffer), new c.Size().setNative(offset), new c.Size().setNative(size))
  static new = (fn: (buffer: Buffer, offset: c.Size, size: c.Size) => void) => new ProcBufferGetConstMappedRange().set(fn)
}
export class ProcBufferGetMapState extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferGetMapState().set(fn)
}
export class ProcBufferGetMappedRange extends c.Function<[buffer: Buffer, offset: c.Size, size: c.Size]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'usize'])
  }
  protected override _fn = (fn: (buffer: Buffer, offset: c.Size, size: c.Size) => void) => (buffer: any, offset: any, size: any) => void fn(new Buffer().setNative(buffer), new c.Size().setNative(offset), new c.Size().setNative(size))
  static new = (fn: (buffer: Buffer, offset: c.Size, size: c.Size) => void) => new ProcBufferGetMappedRange().set(fn)
}
export class ProcBufferGetSize extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferGetSize().set(fn)
}
export class ProcBufferGetUsage extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferGetUsage().set(fn)
}
export class ProcBufferMapAsync extends c.Function<[buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callback: BufferMapCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u64', 'usize', 'usize', 'function', 'pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callback: BufferMapCallback, userdata: c.Pointer<any>) => void) => (buffer: any, mode: any, offset: any, size: any, callback: any, userdata: any) => void fn(new Buffer().setNative(buffer), new MapMode().setNative(mode), new c.Size().setNative(offset), new c.Size().setNative(size), new BufferMapCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callback: BufferMapCallback, userdata: c.Pointer<any>) => void) => new ProcBufferMapAsync().set(fn)
}
export class ProcBufferMapAsync2 extends c.Function<[buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u64', 'usize', 'usize', 'buffer'])
  }
  protected override _fn = (fn: (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo2) => void) => (buffer: any, mode: any, offset: any, size: any, callbackInfo: any) => void fn(new Buffer().setNative(buffer), new MapMode().setNative(mode), new c.Size().setNative(offset), new c.Size().setNative(size), new BufferMapCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo2) => void) => new ProcBufferMapAsync2().set(fn)
}
export class ProcBufferMapAsyncF extends c.Function<[buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u64', 'usize', 'usize', 'buffer'])
  }
  protected override _fn = (fn: (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo) => void) => (buffer: any, mode: any, offset: any, size: any, callbackInfo: any) => void fn(new Buffer().setNative(buffer), new MapMode().setNative(mode), new c.Size().setNative(offset), new c.Size().setNative(size), new BufferMapCallbackInfo().setNative(callbackInfo))
  static new = (fn: (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo) => void) => new ProcBufferMapAsyncF().set(fn)
}
export class ProcBufferSetLabel extends c.Function<[buffer: Buffer, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (buffer: Buffer, label: StringView) => void) => (buffer: any, label: any) => void fn(new Buffer().setNative(buffer), new StringView().setNative(label))
  static new = (fn: (buffer: Buffer, label: StringView) => void) => new ProcBufferSetLabel().set(fn)
}
export class ProcBufferUnmap extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferUnmap().set(fn)
}
export class ProcBufferAddRef extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferAddRef().set(fn)
}
export class ProcBufferRelease extends c.Function<[buffer: Buffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (buffer: Buffer) => void) => (buffer: any) => void fn(new Buffer().setNative(buffer))
  static new = (fn: (buffer: Buffer) => void) => new ProcBufferRelease().set(fn)
}
export class ProcCommandBufferSetLabel extends c.Function<[commandBuffer: CommandBuffer, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (commandBuffer: CommandBuffer, label: StringView) => void) => (commandBuffer: any, label: any) => void fn(new CommandBuffer().setNative(commandBuffer), new StringView().setNative(label))
  static new = (fn: (commandBuffer: CommandBuffer, label: StringView) => void) => new ProcCommandBufferSetLabel().set(fn)
}
export class ProcCommandBufferAddRef extends c.Function<[commandBuffer: CommandBuffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (commandBuffer: CommandBuffer) => void) => (commandBuffer: any) => void fn(new CommandBuffer().setNative(commandBuffer))
  static new = (fn: (commandBuffer: CommandBuffer) => void) => new ProcCommandBufferAddRef().set(fn)
}
export class ProcCommandBufferRelease extends c.Function<[commandBuffer: CommandBuffer]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (commandBuffer: CommandBuffer) => void) => (commandBuffer: any) => void fn(new CommandBuffer().setNative(commandBuffer))
  static new = (fn: (commandBuffer: CommandBuffer) => void) => new ProcCommandBufferRelease().set(fn)
}
export class ProcCommandEncoderBeginComputePass extends c.Function<[commandEncoder: CommandEncoder, descriptor: c.Pointer<ComputePassDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, descriptor: c.Pointer<ComputePassDescriptor>) => void) => (commandEncoder: any, descriptor: any) => void fn(new CommandEncoder().setNative(commandEncoder), new c.Pointer<ComputePassDescriptor>().setNative(descriptor))
  static new = (fn: (commandEncoder: CommandEncoder, descriptor: c.Pointer<ComputePassDescriptor>) => void) => new ProcCommandEncoderBeginComputePass().set(fn)
}
export class ProcCommandEncoderBeginRenderPass extends c.Function<[commandEncoder: CommandEncoder, descriptor: c.Pointer<RenderPassDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, descriptor: c.Pointer<RenderPassDescriptor>) => void) => (commandEncoder: any, descriptor: any) => void fn(new CommandEncoder().setNative(commandEncoder), new c.Pointer<RenderPassDescriptor>().setNative(descriptor))
  static new = (fn: (commandEncoder: CommandEncoder, descriptor: c.Pointer<RenderPassDescriptor>) => void) => new ProcCommandEncoderBeginRenderPass().set(fn)
}
export class ProcCommandEncoderClearBuffer extends c.Function<[commandEncoder: CommandEncoder, buffer: Buffer, offset: c.U64, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64', 'u64'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, buffer: Buffer, offset: c.U64, size: c.U64) => void) => (commandEncoder: any, buffer: any, offset: any, size: any) => void fn(new CommandEncoder().setNative(commandEncoder), new Buffer().setNative(buffer), new c.U64().setNative(offset), new c.U64().setNative(size))
  static new = (fn: (commandEncoder: CommandEncoder, buffer: Buffer, offset: c.U64, size: c.U64) => void) => new ProcCommandEncoderClearBuffer().set(fn)
}
export class ProcCommandEncoderCopyBufferToBuffer extends c.Function<[commandEncoder: CommandEncoder, source: Buffer, sourceOffset: c.U64, destination: Buffer, destinationOffset: c.U64, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64', 'pointer', 'u64', 'u64'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, source: Buffer, sourceOffset: c.U64, destination: Buffer, destinationOffset: c.U64, size: c.U64) => void) => (commandEncoder: any, source: any, sourceOffset: any, destination: any, destinationOffset: any, size: any) => void fn(new CommandEncoder().setNative(commandEncoder), new Buffer().setNative(source), new c.U64().setNative(sourceOffset), new Buffer().setNative(destination), new c.U64().setNative(destinationOffset), new c.U64().setNative(size))
  static new = (fn: (commandEncoder: CommandEncoder, source: Buffer, sourceOffset: c.U64, destination: Buffer, destinationOffset: c.U64, size: c.U64) => void) => new ProcCommandEncoderCopyBufferToBuffer().set(fn)
}
export class ProcCommandEncoderCopyBufferToTexture extends c.Function<[commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyBuffer>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyBuffer>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>) => void) => (commandEncoder: any, source: any, destination: any, copySize: any) => void fn(new CommandEncoder().setNative(commandEncoder), new c.Pointer<ImageCopyBuffer>().setNative(source), new c.Pointer<ImageCopyTexture>().setNative(destination), new c.Pointer<Extent3D>().setNative(copySize))
  static new = (fn: (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyBuffer>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>) => void) => new ProcCommandEncoderCopyBufferToTexture().set(fn)
}
export class ProcCommandEncoderCopyTextureToBuffer extends c.Function<[commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyBuffer>, copySize: c.Pointer<Extent3D>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyBuffer>, copySize: c.Pointer<Extent3D>) => void) => (commandEncoder: any, source: any, destination: any, copySize: any) => void fn(new CommandEncoder().setNative(commandEncoder), new c.Pointer<ImageCopyTexture>().setNative(source), new c.Pointer<ImageCopyBuffer>().setNative(destination), new c.Pointer<Extent3D>().setNative(copySize))
  static new = (fn: (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyBuffer>, copySize: c.Pointer<Extent3D>) => void) => new ProcCommandEncoderCopyTextureToBuffer().set(fn)
}
export class ProcCommandEncoderCopyTextureToTexture extends c.Function<[commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>) => void) => (commandEncoder: any, source: any, destination: any, copySize: any) => void fn(new CommandEncoder().setNative(commandEncoder), new c.Pointer<ImageCopyTexture>().setNative(source), new c.Pointer<ImageCopyTexture>().setNative(destination), new c.Pointer<Extent3D>().setNative(copySize))
  static new = (fn: (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>) => void) => new ProcCommandEncoderCopyTextureToTexture().set(fn)
}
export class ProcCommandEncoderFinish extends c.Function<[commandEncoder: CommandEncoder, descriptor: c.Pointer<CommandBufferDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, descriptor: c.Pointer<CommandBufferDescriptor>) => void) => (commandEncoder: any, descriptor: any) => void fn(new CommandEncoder().setNative(commandEncoder), new c.Pointer<CommandBufferDescriptor>().setNative(descriptor))
  static new = (fn: (commandEncoder: CommandEncoder, descriptor: c.Pointer<CommandBufferDescriptor>) => void) => new ProcCommandEncoderFinish().set(fn)
}
export class ProcCommandEncoderInjectValidationError extends c.Function<[commandEncoder: CommandEncoder, message: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, message: StringView) => void) => (commandEncoder: any, message: any) => void fn(new CommandEncoder().setNative(commandEncoder), new StringView().setNative(message))
  static new = (fn: (commandEncoder: CommandEncoder, message: StringView) => void) => new ProcCommandEncoderInjectValidationError().set(fn)
}
export class ProcCommandEncoderInsertDebugMarker extends c.Function<[commandEncoder: CommandEncoder, markerLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, markerLabel: StringView) => void) => (commandEncoder: any, markerLabel: any) => void fn(new CommandEncoder().setNative(commandEncoder), new StringView().setNative(markerLabel))
  static new = (fn: (commandEncoder: CommandEncoder, markerLabel: StringView) => void) => new ProcCommandEncoderInsertDebugMarker().set(fn)
}
export class ProcCommandEncoderPopDebugGroup extends c.Function<[commandEncoder: CommandEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder) => void) => (commandEncoder: any) => void fn(new CommandEncoder().setNative(commandEncoder))
  static new = (fn: (commandEncoder: CommandEncoder) => void) => new ProcCommandEncoderPopDebugGroup().set(fn)
}
export class ProcCommandEncoderPushDebugGroup extends c.Function<[commandEncoder: CommandEncoder, groupLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, groupLabel: StringView) => void) => (commandEncoder: any, groupLabel: any) => void fn(new CommandEncoder().setNative(commandEncoder), new StringView().setNative(groupLabel))
  static new = (fn: (commandEncoder: CommandEncoder, groupLabel: StringView) => void) => new ProcCommandEncoderPushDebugGroup().set(fn)
}
export class ProcCommandEncoderResolveQuerySet extends c.Function<[commandEncoder: CommandEncoder, querySet: QuerySet, firstQuery: c.U32, queryCount: c.U32, destination: Buffer, destinationOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u32', 'u32', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, querySet: QuerySet, firstQuery: c.U32, queryCount: c.U32, destination: Buffer, destinationOffset: c.U64) => void) => (commandEncoder: any, querySet: any, firstQuery: any, queryCount: any, destination: any, destinationOffset: any) => void fn(new CommandEncoder().setNative(commandEncoder), new QuerySet().setNative(querySet), new c.U32().setNative(firstQuery), new c.U32().setNative(queryCount), new Buffer().setNative(destination), new c.U64().setNative(destinationOffset))
  static new = (fn: (commandEncoder: CommandEncoder, querySet: QuerySet, firstQuery: c.U32, queryCount: c.U32, destination: Buffer, destinationOffset: c.U64) => void) => new ProcCommandEncoderResolveQuerySet().set(fn)
}
export class ProcCommandEncoderSetLabel extends c.Function<[commandEncoder: CommandEncoder, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, label: StringView) => void) => (commandEncoder: any, label: any) => void fn(new CommandEncoder().setNative(commandEncoder), new StringView().setNative(label))
  static new = (fn: (commandEncoder: CommandEncoder, label: StringView) => void) => new ProcCommandEncoderSetLabel().set(fn)
}
export class ProcCommandEncoderWriteBuffer extends c.Function<[commandEncoder: CommandEncoder, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.U8>, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.U8>, size: c.U64) => void) => (commandEncoder: any, buffer: any, bufferOffset: any, data: any, size: any) => void fn(new CommandEncoder().setNative(commandEncoder), new Buffer().setNative(buffer), new c.U64().setNative(bufferOffset), new c.Pointer<c.U8>().setNative(data), new c.U64().setNative(size))
  static new = (fn: (commandEncoder: CommandEncoder, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.U8>, size: c.U64) => void) => new ProcCommandEncoderWriteBuffer().set(fn)
}
export class ProcCommandEncoderWriteTimestamp extends c.Function<[commandEncoder: CommandEncoder, querySet: QuerySet, queryIndex: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u32'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder, querySet: QuerySet, queryIndex: c.U32) => void) => (commandEncoder: any, querySet: any, queryIndex: any) => void fn(new CommandEncoder().setNative(commandEncoder), new QuerySet().setNative(querySet), new c.U32().setNative(queryIndex))
  static new = (fn: (commandEncoder: CommandEncoder, querySet: QuerySet, queryIndex: c.U32) => void) => new ProcCommandEncoderWriteTimestamp().set(fn)
}
export class ProcCommandEncoderAddRef extends c.Function<[commandEncoder: CommandEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder) => void) => (commandEncoder: any) => void fn(new CommandEncoder().setNative(commandEncoder))
  static new = (fn: (commandEncoder: CommandEncoder) => void) => new ProcCommandEncoderAddRef().set(fn)
}
export class ProcCommandEncoderRelease extends c.Function<[commandEncoder: CommandEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (commandEncoder: CommandEncoder) => void) => (commandEncoder: any) => void fn(new CommandEncoder().setNative(commandEncoder))
  static new = (fn: (commandEncoder: CommandEncoder) => void) => new ProcCommandEncoderRelease().set(fn)
}
export class ProcComputePassEncoderDispatchWorkgroups extends c.Function<[computePassEncoder: ComputePassEncoder, workgroupCountX: c.U32, workgroupCountY: c.U32, workgroupCountZ: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'u32', 'u32'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, workgroupCountX: c.U32, workgroupCountY: c.U32, workgroupCountZ: c.U32) => void) => (computePassEncoder: any, workgroupCountX: any, workgroupCountY: any, workgroupCountZ: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new c.U32().setNative(workgroupCountX), new c.U32().setNative(workgroupCountY), new c.U32().setNative(workgroupCountZ))
  static new = (fn: (computePassEncoder: ComputePassEncoder, workgroupCountX: c.U32, workgroupCountY: c.U32, workgroupCountZ: c.U32) => void) => new ProcComputePassEncoderDispatchWorkgroups().set(fn)
}
export class ProcComputePassEncoderDispatchWorkgroupsIndirect extends c.Function<[computePassEncoder: ComputePassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => (computePassEncoder: any, indirectBuffer: any, indirectOffset: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset))
  static new = (fn: (computePassEncoder: ComputePassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => new ProcComputePassEncoderDispatchWorkgroupsIndirect().set(fn)
}
export class ProcComputePassEncoderEnd extends c.Function<[computePassEncoder: ComputePassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder) => void) => (computePassEncoder: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder))
  static new = (fn: (computePassEncoder: ComputePassEncoder) => void) => new ProcComputePassEncoderEnd().set(fn)
}
export class ProcComputePassEncoderInsertDebugMarker extends c.Function<[computePassEncoder: ComputePassEncoder, markerLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, markerLabel: StringView) => void) => (computePassEncoder: any, markerLabel: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new StringView().setNative(markerLabel))
  static new = (fn: (computePassEncoder: ComputePassEncoder, markerLabel: StringView) => void) => new ProcComputePassEncoderInsertDebugMarker().set(fn)
}
export class ProcComputePassEncoderPopDebugGroup extends c.Function<[computePassEncoder: ComputePassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder) => void) => (computePassEncoder: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder))
  static new = (fn: (computePassEncoder: ComputePassEncoder) => void) => new ProcComputePassEncoderPopDebugGroup().set(fn)
}
export class ProcComputePassEncoderPushDebugGroup extends c.Function<[computePassEncoder: ComputePassEncoder, groupLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, groupLabel: StringView) => void) => (computePassEncoder: any, groupLabel: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new StringView().setNative(groupLabel))
  static new = (fn: (computePassEncoder: ComputePassEncoder, groupLabel: StringView) => void) => new ProcComputePassEncoderPushDebugGroup().set(fn)
}
export class ProcComputePassEncoderSetBindGroup extends c.Function<[computePassEncoder: ComputePassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>) => void) => (computePassEncoder: any, groupIndex: any, group: any, dynamicOffsetCount: any, dynamicOffsets: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new c.U32().setNative(groupIndex), new BindGroup().setNative(group), new c.Size().setNative(dynamicOffsetCount), new c.Pointer<c.U32>().setNative(dynamicOffsets))
  static new = (fn: (computePassEncoder: ComputePassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>) => void) => new ProcComputePassEncoderSetBindGroup().set(fn)
}
export class ProcComputePassEncoderSetLabel extends c.Function<[computePassEncoder: ComputePassEncoder, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, label: StringView) => void) => (computePassEncoder: any, label: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new StringView().setNative(label))
  static new = (fn: (computePassEncoder: ComputePassEncoder, label: StringView) => void) => new ProcComputePassEncoderSetLabel().set(fn)
}
export class ProcComputePassEncoderSetPipeline extends c.Function<[computePassEncoder: ComputePassEncoder, pipeline: ComputePipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, pipeline: ComputePipeline) => void) => (computePassEncoder: any, pipeline: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new ComputePipeline().setNative(pipeline))
  static new = (fn: (computePassEncoder: ComputePassEncoder, pipeline: ComputePipeline) => void) => new ProcComputePassEncoderSetPipeline().set(fn)
}
export class ProcComputePassEncoderWriteTimestamp extends c.Function<[computePassEncoder: ComputePassEncoder, querySet: QuerySet, queryIndex: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u32'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder, querySet: QuerySet, queryIndex: c.U32) => void) => (computePassEncoder: any, querySet: any, queryIndex: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder), new QuerySet().setNative(querySet), new c.U32().setNative(queryIndex))
  static new = (fn: (computePassEncoder: ComputePassEncoder, querySet: QuerySet, queryIndex: c.U32) => void) => new ProcComputePassEncoderWriteTimestamp().set(fn)
}
export class ProcComputePassEncoderAddRef extends c.Function<[computePassEncoder: ComputePassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder) => void) => (computePassEncoder: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder))
  static new = (fn: (computePassEncoder: ComputePassEncoder) => void) => new ProcComputePassEncoderAddRef().set(fn)
}
export class ProcComputePassEncoderRelease extends c.Function<[computePassEncoder: ComputePassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (computePassEncoder: ComputePassEncoder) => void) => (computePassEncoder: any) => void fn(new ComputePassEncoder().setNative(computePassEncoder))
  static new = (fn: (computePassEncoder: ComputePassEncoder) => void) => new ProcComputePassEncoderRelease().set(fn)
}
export class ProcComputePipelineGetBindGroupLayout extends c.Function<[computePipeline: ComputePipeline, groupIndex: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (computePipeline: ComputePipeline, groupIndex: c.U32) => void) => (computePipeline: any, groupIndex: any) => void fn(new ComputePipeline().setNative(computePipeline), new c.U32().setNative(groupIndex))
  static new = (fn: (computePipeline: ComputePipeline, groupIndex: c.U32) => void) => new ProcComputePipelineGetBindGroupLayout().set(fn)
}
export class ProcComputePipelineSetLabel extends c.Function<[computePipeline: ComputePipeline, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (computePipeline: ComputePipeline, label: StringView) => void) => (computePipeline: any, label: any) => void fn(new ComputePipeline().setNative(computePipeline), new StringView().setNative(label))
  static new = (fn: (computePipeline: ComputePipeline, label: StringView) => void) => new ProcComputePipelineSetLabel().set(fn)
}
export class ProcComputePipelineAddRef extends c.Function<[computePipeline: ComputePipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (computePipeline: ComputePipeline) => void) => (computePipeline: any) => void fn(new ComputePipeline().setNative(computePipeline))
  static new = (fn: (computePipeline: ComputePipeline) => void) => new ProcComputePipelineAddRef().set(fn)
}
export class ProcComputePipelineRelease extends c.Function<[computePipeline: ComputePipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (computePipeline: ComputePipeline) => void) => (computePipeline: any) => void fn(new ComputePipeline().setNative(computePipeline))
  static new = (fn: (computePipeline: ComputePipeline) => void) => new ProcComputePipelineRelease().set(fn)
}
export class ProcDeviceCreateBindGroup extends c.Function<[device: Device, descriptor: c.Pointer<BindGroupDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<BindGroupDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<BindGroupDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<BindGroupDescriptor>) => void) => new ProcDeviceCreateBindGroup().set(fn)
}
export class ProcDeviceCreateBindGroupLayout extends c.Function<[device: Device, descriptor: c.Pointer<BindGroupLayoutDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<BindGroupLayoutDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<BindGroupLayoutDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<BindGroupLayoutDescriptor>) => void) => new ProcDeviceCreateBindGroupLayout().set(fn)
}
export class ProcDeviceCreateBuffer extends c.Function<[device: Device, descriptor: c.Pointer<BufferDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<BufferDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<BufferDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<BufferDescriptor>) => void) => new ProcDeviceCreateBuffer().set(fn)
}
export class ProcDeviceCreateCommandEncoder extends c.Function<[device: Device, descriptor: c.Pointer<CommandEncoderDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<CommandEncoderDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<CommandEncoderDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<CommandEncoderDescriptor>) => void) => new ProcDeviceCreateCommandEncoder().set(fn)
}
export class ProcDeviceCreateComputePipeline extends c.Function<[device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<ComputePipelineDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>) => void) => new ProcDeviceCreateComputePipeline().set(fn)
}
export class ProcDeviceCreateComputePipelineAsync extends c.Function<[device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callback: CreateComputePipelineAsyncCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callback: CreateComputePipelineAsyncCallback, userdata: c.Pointer<any>) => void) => (device: any, descriptor: any, callback: any, userdata: any) => void fn(new Device().setNative(device), new c.Pointer<ComputePipelineDescriptor>().setNative(descriptor), new CreateComputePipelineAsyncCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callback: CreateComputePipelineAsyncCallback, userdata: c.Pointer<any>) => void) => new ProcDeviceCreateComputePipelineAsync().set(fn)
}
export class ProcDeviceCreateComputePipelineAsync2 extends c.Function<[device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo2) => void) => (device: any, descriptor: any, callbackInfo: any) => void fn(new Device().setNative(device), new c.Pointer<ComputePipelineDescriptor>().setNative(descriptor), new CreateComputePipelineAsyncCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo2) => void) => new ProcDeviceCreateComputePipelineAsync2().set(fn)
}
export class ProcDeviceCreateComputePipelineAsyncF extends c.Function<[device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo) => void) => (device: any, descriptor: any, callbackInfo: any) => void fn(new Device().setNative(device), new c.Pointer<ComputePipelineDescriptor>().setNative(descriptor), new CreateComputePipelineAsyncCallbackInfo().setNative(callbackInfo))
  static new = (fn: (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo) => void) => new ProcDeviceCreateComputePipelineAsyncF().set(fn)
}
export class ProcDeviceCreateErrorBuffer extends c.Function<[device: Device, descriptor: c.Pointer<BufferDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<BufferDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<BufferDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<BufferDescriptor>) => void) => new ProcDeviceCreateErrorBuffer().set(fn)
}
export class ProcDeviceCreateErrorExternalTexture extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceCreateErrorExternalTexture().set(fn)
}
export class ProcDeviceCreateErrorShaderModule extends c.Function<[device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>, errorMessage: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>, errorMessage: StringView) => void) => (device: any, descriptor: any, errorMessage: any) => void fn(new Device().setNative(device), new c.Pointer<ShaderModuleDescriptor>().setNative(descriptor), new StringView().setNative(errorMessage))
  static new = (fn: (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>, errorMessage: StringView) => void) => new ProcDeviceCreateErrorShaderModule().set(fn)
}
export class ProcDeviceCreateErrorTexture extends c.Function<[device: Device, descriptor: c.Pointer<TextureDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<TextureDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<TextureDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<TextureDescriptor>) => void) => new ProcDeviceCreateErrorTexture().set(fn)
}
export class ProcDeviceCreateExternalTexture extends c.Function<[device: Device, externalTextureDescriptor: c.Pointer<ExternalTextureDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, externalTextureDescriptor: c.Pointer<ExternalTextureDescriptor>) => void) => (device: any, externalTextureDescriptor: any) => void fn(new Device().setNative(device), new c.Pointer<ExternalTextureDescriptor>().setNative(externalTextureDescriptor))
  static new = (fn: (device: Device, externalTextureDescriptor: c.Pointer<ExternalTextureDescriptor>) => void) => new ProcDeviceCreateExternalTexture().set(fn)
}
export class ProcDeviceCreatePipelineLayout extends c.Function<[device: Device, descriptor: c.Pointer<PipelineLayoutDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<PipelineLayoutDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<PipelineLayoutDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<PipelineLayoutDescriptor>) => void) => new ProcDeviceCreatePipelineLayout().set(fn)
}
export class ProcDeviceCreateQuerySet extends c.Function<[device: Device, descriptor: c.Pointer<QuerySetDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<QuerySetDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<QuerySetDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<QuerySetDescriptor>) => void) => new ProcDeviceCreateQuerySet().set(fn)
}
export class ProcDeviceCreateRenderBundleEncoder extends c.Function<[device: Device, descriptor: c.Pointer<RenderBundleEncoderDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<RenderBundleEncoderDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<RenderBundleEncoderDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<RenderBundleEncoderDescriptor>) => void) => new ProcDeviceCreateRenderBundleEncoder().set(fn)
}
export class ProcDeviceCreateRenderPipeline extends c.Function<[device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<RenderPipelineDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>) => void) => new ProcDeviceCreateRenderPipeline().set(fn)
}
export class ProcDeviceCreateRenderPipelineAsync extends c.Function<[device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callback: CreateRenderPipelineAsyncCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callback: CreateRenderPipelineAsyncCallback, userdata: c.Pointer<any>) => void) => (device: any, descriptor: any, callback: any, userdata: any) => void fn(new Device().setNative(device), new c.Pointer<RenderPipelineDescriptor>().setNative(descriptor), new CreateRenderPipelineAsyncCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callback: CreateRenderPipelineAsyncCallback, userdata: c.Pointer<any>) => void) => new ProcDeviceCreateRenderPipelineAsync().set(fn)
}
export class ProcDeviceCreateRenderPipelineAsync2 extends c.Function<[device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo2) => void) => (device: any, descriptor: any, callbackInfo: any) => void fn(new Device().setNative(device), new c.Pointer<RenderPipelineDescriptor>().setNative(descriptor), new CreateRenderPipelineAsyncCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo2) => void) => new ProcDeviceCreateRenderPipelineAsync2().set(fn)
}
export class ProcDeviceCreateRenderPipelineAsyncF extends c.Function<[device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo) => void) => (device: any, descriptor: any, callbackInfo: any) => void fn(new Device().setNative(device), new c.Pointer<RenderPipelineDescriptor>().setNative(descriptor), new CreateRenderPipelineAsyncCallbackInfo().setNative(callbackInfo))
  static new = (fn: (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo) => void) => new ProcDeviceCreateRenderPipelineAsyncF().set(fn)
}
export class ProcDeviceCreateSampler extends c.Function<[device: Device, descriptor: c.Pointer<SamplerDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<SamplerDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<SamplerDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<SamplerDescriptor>) => void) => new ProcDeviceCreateSampler().set(fn)
}
export class ProcDeviceCreateShaderModule extends c.Function<[device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<ShaderModuleDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>) => void) => new ProcDeviceCreateShaderModule().set(fn)
}
export class ProcDeviceCreateTexture extends c.Function<[device: Device, descriptor: c.Pointer<TextureDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<TextureDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<TextureDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<TextureDescriptor>) => void) => new ProcDeviceCreateTexture().set(fn)
}
export class ProcDeviceDestroy extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceDestroy().set(fn)
}
export class ProcDeviceForceLoss extends c.Function<[device: Device, type: DeviceLostReason, message: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, type: DeviceLostReason, message: StringView) => void) => (device: any, type: any, message: any) => void fn(new Device().setNative(device), new DeviceLostReason().setNative(type), new StringView().setNative(message))
  static new = (fn: (device: Device, type: DeviceLostReason, message: StringView) => void) => new ProcDeviceForceLoss().set(fn)
}
export class ProcDeviceGetAHardwareBufferProperties extends c.Function<[device: Device, handle: c.Pointer<any>, properties: c.Pointer<AHardwareBufferProperties>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, handle: c.Pointer<any>, properties: c.Pointer<AHardwareBufferProperties>) => void) => (device: any, handle: any, properties: any) => void fn(new Device().setNative(device), new c.Pointer<any>().setNative(handle), new c.Pointer<AHardwareBufferProperties>().setNative(properties))
  static new = (fn: (device: Device, handle: c.Pointer<any>, properties: c.Pointer<AHardwareBufferProperties>) => void) => new ProcDeviceGetAHardwareBufferProperties().set(fn)
}
export class ProcDeviceGetAdapter extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceGetAdapter().set(fn)
}
export class ProcDeviceGetAdapterInfo extends c.Function<[device: Device, adapterInfo: c.Pointer<AdapterInfo>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, adapterInfo: c.Pointer<AdapterInfo>) => void) => (device: any, adapterInfo: any) => void fn(new Device().setNative(device), new c.Pointer<AdapterInfo>().setNative(adapterInfo))
  static new = (fn: (device: Device, adapterInfo: c.Pointer<AdapterInfo>) => void) => new ProcDeviceGetAdapterInfo().set(fn)
}
export class ProcDeviceGetFeatures extends c.Function<[device: Device, features: c.Pointer<SupportedFeatures>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, features: c.Pointer<SupportedFeatures>) => void) => (device: any, features: any) => void fn(new Device().setNative(device), new c.Pointer<SupportedFeatures>().setNative(features))
  static new = (fn: (device: Device, features: c.Pointer<SupportedFeatures>) => void) => new ProcDeviceGetFeatures().set(fn)
}
export class ProcDeviceGetLimits extends c.Function<[device: Device, limits: c.Pointer<SupportedLimits>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, limits: c.Pointer<SupportedLimits>) => void) => (device: any, limits: any) => void fn(new Device().setNative(device), new c.Pointer<SupportedLimits>().setNative(limits))
  static new = (fn: (device: Device, limits: c.Pointer<SupportedLimits>) => void) => new ProcDeviceGetLimits().set(fn)
}
export class ProcDeviceGetLostFuture extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceGetLostFuture().set(fn)
}
export class ProcDeviceGetQueue extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceGetQueue().set(fn)
}
export class ProcDeviceHasFeature extends c.Function<[device: Device, feature: FeatureName]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (device: Device, feature: FeatureName) => void) => (device: any, feature: any) => void fn(new Device().setNative(device), new FeatureName().setNative(feature))
  static new = (fn: (device: Device, feature: FeatureName) => void) => new ProcDeviceHasFeature().set(fn)
}
export class ProcDeviceImportSharedBufferMemory extends c.Function<[device: Device, descriptor: c.Pointer<SharedBufferMemoryDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<SharedBufferMemoryDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<SharedBufferMemoryDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<SharedBufferMemoryDescriptor>) => void) => new ProcDeviceImportSharedBufferMemory().set(fn)
}
export class ProcDeviceImportSharedFence extends c.Function<[device: Device, descriptor: c.Pointer<SharedFenceDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<SharedFenceDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<SharedFenceDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<SharedFenceDescriptor>) => void) => new ProcDeviceImportSharedFence().set(fn)
}
export class ProcDeviceImportSharedTextureMemory extends c.Function<[device: Device, descriptor: c.Pointer<SharedTextureMemoryDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<SharedTextureMemoryDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<SharedTextureMemoryDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<SharedTextureMemoryDescriptor>) => void) => new ProcDeviceImportSharedTextureMemory().set(fn)
}
export class ProcDeviceInjectError extends c.Function<[device: Device, type: ErrorType, message: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, type: ErrorType, message: StringView) => void) => (device: any, type: any, message: any) => void fn(new Device().setNative(device), new ErrorType().setNative(type), new StringView().setNative(message))
  static new = (fn: (device: Device, type: ErrorType, message: StringView) => void) => new ProcDeviceInjectError().set(fn)
}
export class ProcDevicePopErrorScope extends c.Function<[device: Device, oldCallback: ErrorCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, oldCallback: ErrorCallback, userdata: c.Pointer<any>) => void) => (device: any, oldCallback: any, userdata: any) => void fn(new Device().setNative(device), new ErrorCallback().setNative(oldCallback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (device: Device, oldCallback: ErrorCallback, userdata: c.Pointer<any>) => void) => new ProcDevicePopErrorScope().set(fn)
}
export class ProcDevicePopErrorScope2 extends c.Function<[device: Device, callbackInfo: PopErrorScopeCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, callbackInfo: PopErrorScopeCallbackInfo2) => void) => (device: any, callbackInfo: any) => void fn(new Device().setNative(device), new PopErrorScopeCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (device: Device, callbackInfo: PopErrorScopeCallbackInfo2) => void) => new ProcDevicePopErrorScope2().set(fn)
}
export class ProcDevicePopErrorScopeF extends c.Function<[device: Device, callbackInfo: PopErrorScopeCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, callbackInfo: PopErrorScopeCallbackInfo) => void) => (device: any, callbackInfo: any) => void fn(new Device().setNative(device), new PopErrorScopeCallbackInfo().setNative(callbackInfo))
  static new = (fn: (device: Device, callbackInfo: PopErrorScopeCallbackInfo) => void) => new ProcDevicePopErrorScopeF().set(fn)
}
export class ProcDevicePushErrorScope extends c.Function<[device: Device, filter: ErrorFilter]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (device: Device, filter: ErrorFilter) => void) => (device: any, filter: any) => void fn(new Device().setNative(device), new ErrorFilter().setNative(filter))
  static new = (fn: (device: Device, filter: ErrorFilter) => void) => new ProcDevicePushErrorScope().set(fn)
}
export class ProcDeviceSetLabel extends c.Function<[device: Device, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (device: Device, label: StringView) => void) => (device: any, label: any) => void fn(new Device().setNative(device), new StringView().setNative(label))
  static new = (fn: (device: Device, label: StringView) => void) => new ProcDeviceSetLabel().set(fn)
}
export class ProcDeviceSetLoggingCallback extends c.Function<[device: Device, callback: LoggingCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, callback: LoggingCallback, userdata: c.Pointer<any>) => void) => (device: any, callback: any, userdata: any) => void fn(new Device().setNative(device), new LoggingCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (device: Device, callback: LoggingCallback, userdata: c.Pointer<any>) => void) => new ProcDeviceSetLoggingCallback().set(fn)
}
export class ProcDeviceTick extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceTick().set(fn)
}
export class ProcDeviceValidateTextureDescriptor extends c.Function<[device: Device, descriptor: c.Pointer<TextureDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (device: Device, descriptor: c.Pointer<TextureDescriptor>) => void) => (device: any, descriptor: any) => void fn(new Device().setNative(device), new c.Pointer<TextureDescriptor>().setNative(descriptor))
  static new = (fn: (device: Device, descriptor: c.Pointer<TextureDescriptor>) => void) => new ProcDeviceValidateTextureDescriptor().set(fn)
}
export class ProcDeviceAddRef extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceAddRef().set(fn)
}
export class ProcDeviceRelease extends c.Function<[device: Device]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (device: Device) => void) => (device: any) => void fn(new Device().setNative(device))
  static new = (fn: (device: Device) => void) => new ProcDeviceRelease().set(fn)
}
export class ProcExternalTextureDestroy extends c.Function<[externalTexture: ExternalTexture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (externalTexture: ExternalTexture) => void) => (externalTexture: any) => void fn(new ExternalTexture().setNative(externalTexture))
  static new = (fn: (externalTexture: ExternalTexture) => void) => new ProcExternalTextureDestroy().set(fn)
}
export class ProcExternalTextureExpire extends c.Function<[externalTexture: ExternalTexture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (externalTexture: ExternalTexture) => void) => (externalTexture: any) => void fn(new ExternalTexture().setNative(externalTexture))
  static new = (fn: (externalTexture: ExternalTexture) => void) => new ProcExternalTextureExpire().set(fn)
}
export class ProcExternalTextureRefresh extends c.Function<[externalTexture: ExternalTexture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (externalTexture: ExternalTexture) => void) => (externalTexture: any) => void fn(new ExternalTexture().setNative(externalTexture))
  static new = (fn: (externalTexture: ExternalTexture) => void) => new ProcExternalTextureRefresh().set(fn)
}
export class ProcExternalTextureSetLabel extends c.Function<[externalTexture: ExternalTexture, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (externalTexture: ExternalTexture, label: StringView) => void) => (externalTexture: any, label: any) => void fn(new ExternalTexture().setNative(externalTexture), new StringView().setNative(label))
  static new = (fn: (externalTexture: ExternalTexture, label: StringView) => void) => new ProcExternalTextureSetLabel().set(fn)
}
export class ProcExternalTextureAddRef extends c.Function<[externalTexture: ExternalTexture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (externalTexture: ExternalTexture) => void) => (externalTexture: any) => void fn(new ExternalTexture().setNative(externalTexture))
  static new = (fn: (externalTexture: ExternalTexture) => void) => new ProcExternalTextureAddRef().set(fn)
}
export class ProcExternalTextureRelease extends c.Function<[externalTexture: ExternalTexture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (externalTexture: ExternalTexture) => void) => (externalTexture: any) => void fn(new ExternalTexture().setNative(externalTexture))
  static new = (fn: (externalTexture: ExternalTexture) => void) => new ProcExternalTextureRelease().set(fn)
}
export class ProcInstanceCreateSurface extends c.Function<[instance: Instance, descriptor: c.Pointer<SurfaceDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (instance: Instance, descriptor: c.Pointer<SurfaceDescriptor>) => void) => (instance: any, descriptor: any) => void fn(new Instance().setNative(instance), new c.Pointer<SurfaceDescriptor>().setNative(descriptor))
  static new = (fn: (instance: Instance, descriptor: c.Pointer<SurfaceDescriptor>) => void) => new ProcInstanceCreateSurface().set(fn)
}
export class ProcInstanceEnumerateWGSLLanguageFeatures extends c.Function<[instance: Instance, features: c.Pointer<WGSLFeatureName>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (instance: Instance, features: c.Pointer<WGSLFeatureName>) => void) => (instance: any, features: any) => void fn(new Instance().setNative(instance), new c.Pointer<WGSLFeatureName>().setNative(features))
  static new = (fn: (instance: Instance, features: c.Pointer<WGSLFeatureName>) => void) => new ProcInstanceEnumerateWGSLLanguageFeatures().set(fn)
}
export class ProcInstanceHasWGSLLanguageFeature extends c.Function<[instance: Instance, feature: WGSLFeatureName]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (instance: Instance, feature: WGSLFeatureName) => void) => (instance: any, feature: any) => void fn(new Instance().setNative(instance), new WGSLFeatureName().setNative(feature))
  static new = (fn: (instance: Instance, feature: WGSLFeatureName) => void) => new ProcInstanceHasWGSLLanguageFeature().set(fn)
}
export class ProcInstanceProcessEvents extends c.Function<[instance: Instance]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (instance: Instance) => void) => (instance: any) => void fn(new Instance().setNative(instance))
  static new = (fn: (instance: Instance) => void) => new ProcInstanceProcessEvents().set(fn)
}
export class ProcInstanceRequestAdapter extends c.Function<[instance: Instance, options: c.Pointer<RequestAdapterOptions>, callback: RequestAdapterCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callback: RequestAdapterCallback, userdata: c.Pointer<any>) => void) => (instance: any, options: any, callback: any, userdata: any) => void fn(new Instance().setNative(instance), new c.Pointer<RequestAdapterOptions>().setNative(options), new RequestAdapterCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callback: RequestAdapterCallback, userdata: c.Pointer<any>) => void) => new ProcInstanceRequestAdapter().set(fn)
}
export class ProcInstanceRequestAdapter2 extends c.Function<[instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo2) => void) => (instance: any, options: any, callbackInfo: any) => void fn(new Instance().setNative(instance), new c.Pointer<RequestAdapterOptions>().setNative(options), new RequestAdapterCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo2) => void) => new ProcInstanceRequestAdapter2().set(fn)
}
export class ProcInstanceRequestAdapterF extends c.Function<[instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'buffer'])
  }
  protected override _fn = (fn: (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo) => void) => (instance: any, options: any, callbackInfo: any) => void fn(new Instance().setNative(instance), new c.Pointer<RequestAdapterOptions>().setNative(options), new RequestAdapterCallbackInfo().setNative(callbackInfo))
  static new = (fn: (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo) => void) => new ProcInstanceRequestAdapterF().set(fn)
}
export class ProcInstanceWaitAny extends c.Function<[instance: Instance, futureCount: c.Size, futures: c.Pointer<FutureWaitInfo>, timeoutNS: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (instance: Instance, futureCount: c.Size, futures: c.Pointer<FutureWaitInfo>, timeoutNS: c.U64) => void) => (instance: any, futureCount: any, futures: any, timeoutNS: any) => void fn(new Instance().setNative(instance), new c.Size().setNative(futureCount), new c.Pointer<FutureWaitInfo>().setNative(futures), new c.U64().setNative(timeoutNS))
  static new = (fn: (instance: Instance, futureCount: c.Size, futures: c.Pointer<FutureWaitInfo>, timeoutNS: c.U64) => void) => new ProcInstanceWaitAny().set(fn)
}
export class ProcInstanceAddRef extends c.Function<[instance: Instance]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (instance: Instance) => void) => (instance: any) => void fn(new Instance().setNative(instance))
  static new = (fn: (instance: Instance) => void) => new ProcInstanceAddRef().set(fn)
}
export class ProcInstanceRelease extends c.Function<[instance: Instance]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (instance: Instance) => void) => (instance: any) => void fn(new Instance().setNative(instance))
  static new = (fn: (instance: Instance) => void) => new ProcInstanceRelease().set(fn)
}
export class ProcPipelineLayoutSetLabel extends c.Function<[pipelineLayout: PipelineLayout, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (pipelineLayout: PipelineLayout, label: StringView) => void) => (pipelineLayout: any, label: any) => void fn(new PipelineLayout().setNative(pipelineLayout), new StringView().setNative(label))
  static new = (fn: (pipelineLayout: PipelineLayout, label: StringView) => void) => new ProcPipelineLayoutSetLabel().set(fn)
}
export class ProcPipelineLayoutAddRef extends c.Function<[pipelineLayout: PipelineLayout]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (pipelineLayout: PipelineLayout) => void) => (pipelineLayout: any) => void fn(new PipelineLayout().setNative(pipelineLayout))
  static new = (fn: (pipelineLayout: PipelineLayout) => void) => new ProcPipelineLayoutAddRef().set(fn)
}
export class ProcPipelineLayoutRelease extends c.Function<[pipelineLayout: PipelineLayout]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (pipelineLayout: PipelineLayout) => void) => (pipelineLayout: any) => void fn(new PipelineLayout().setNative(pipelineLayout))
  static new = (fn: (pipelineLayout: PipelineLayout) => void) => new ProcPipelineLayoutRelease().set(fn)
}
export class ProcQuerySetDestroy extends c.Function<[querySet: QuerySet]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (querySet: QuerySet) => void) => (querySet: any) => void fn(new QuerySet().setNative(querySet))
  static new = (fn: (querySet: QuerySet) => void) => new ProcQuerySetDestroy().set(fn)
}
export class ProcQuerySetGetCount extends c.Function<[querySet: QuerySet]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (querySet: QuerySet) => void) => (querySet: any) => void fn(new QuerySet().setNative(querySet))
  static new = (fn: (querySet: QuerySet) => void) => new ProcQuerySetGetCount().set(fn)
}
export class ProcQuerySetGetType extends c.Function<[querySet: QuerySet]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (querySet: QuerySet) => void) => (querySet: any) => void fn(new QuerySet().setNative(querySet))
  static new = (fn: (querySet: QuerySet) => void) => new ProcQuerySetGetType().set(fn)
}
export class ProcQuerySetSetLabel extends c.Function<[querySet: QuerySet, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (querySet: QuerySet, label: StringView) => void) => (querySet: any, label: any) => void fn(new QuerySet().setNative(querySet), new StringView().setNative(label))
  static new = (fn: (querySet: QuerySet, label: StringView) => void) => new ProcQuerySetSetLabel().set(fn)
}
export class ProcQuerySetAddRef extends c.Function<[querySet: QuerySet]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (querySet: QuerySet) => void) => (querySet: any) => void fn(new QuerySet().setNative(querySet))
  static new = (fn: (querySet: QuerySet) => void) => new ProcQuerySetAddRef().set(fn)
}
export class ProcQuerySetRelease extends c.Function<[querySet: QuerySet]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (querySet: QuerySet) => void) => (querySet: any) => void fn(new QuerySet().setNative(querySet))
  static new = (fn: (querySet: QuerySet) => void) => new ProcQuerySetRelease().set(fn)
}
export class ProcQueueCopyExternalTextureForBrowser extends c.Function<[queue: Queue, source: c.Pointer<ImageCopyExternalTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (queue: Queue, source: c.Pointer<ImageCopyExternalTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>) => void) => (queue: any, source: any, destination: any, copySize: any, options: any) => void fn(new Queue().setNative(queue), new c.Pointer<ImageCopyExternalTexture>().setNative(source), new c.Pointer<ImageCopyTexture>().setNative(destination), new c.Pointer<Extent3D>().setNative(copySize), new c.Pointer<CopyTextureForBrowserOptions>().setNative(options))
  static new = (fn: (queue: Queue, source: c.Pointer<ImageCopyExternalTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>) => void) => new ProcQueueCopyExternalTextureForBrowser().set(fn)
}
export class ProcQueueCopyTextureForBrowser extends c.Function<[queue: Queue, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (queue: Queue, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>) => void) => (queue: any, source: any, destination: any, copySize: any, options: any) => void fn(new Queue().setNative(queue), new c.Pointer<ImageCopyTexture>().setNative(source), new c.Pointer<ImageCopyTexture>().setNative(destination), new c.Pointer<Extent3D>().setNative(copySize), new c.Pointer<CopyTextureForBrowserOptions>().setNative(options))
  static new = (fn: (queue: Queue, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>) => void) => new ProcQueueCopyTextureForBrowser().set(fn)
}
export class ProcQueueOnSubmittedWorkDone extends c.Function<[queue: Queue, callback: QueueWorkDoneCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (queue: Queue, callback: QueueWorkDoneCallback, userdata: c.Pointer<any>) => void) => (queue: any, callback: any, userdata: any) => void fn(new Queue().setNative(queue), new QueueWorkDoneCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (queue: Queue, callback: QueueWorkDoneCallback, userdata: c.Pointer<any>) => void) => new ProcQueueOnSubmittedWorkDone().set(fn)
}
export class ProcQueueOnSubmittedWorkDone2 extends c.Function<[queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo2) => void) => (queue: any, callbackInfo: any) => void fn(new Queue().setNative(queue), new QueueWorkDoneCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo2) => void) => new ProcQueueOnSubmittedWorkDone2().set(fn)
}
export class ProcQueueOnSubmittedWorkDoneF extends c.Function<[queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo) => void) => (queue: any, callbackInfo: any) => void fn(new Queue().setNative(queue), new QueueWorkDoneCallbackInfo().setNative(callbackInfo))
  static new = (fn: (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo) => void) => new ProcQueueOnSubmittedWorkDoneF().set(fn)
}
export class ProcQueueSetLabel extends c.Function<[queue: Queue, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (queue: Queue, label: StringView) => void) => (queue: any, label: any) => void fn(new Queue().setNative(queue), new StringView().setNative(label))
  static new = (fn: (queue: Queue, label: StringView) => void) => new ProcQueueSetLabel().set(fn)
}
export class ProcQueueSubmit extends c.Function<[queue: Queue, commandCount: c.Size, commands: c.Pointer<CommandBuffer>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (queue: Queue, commandCount: c.Size, commands: c.Pointer<CommandBuffer>) => void) => (queue: any, commandCount: any, commands: any) => void fn(new Queue().setNative(queue), new c.Size().setNative(commandCount), new c.Pointer<CommandBuffer>().setNative(commands))
  static new = (fn: (queue: Queue, commandCount: c.Size, commands: c.Pointer<CommandBuffer>) => void) => new ProcQueueSubmit().set(fn)
}
export class ProcQueueWriteBuffer extends c.Function<[queue: Queue, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<any>, size: c.Size]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64', 'pointer', 'usize'])
  }
  protected override _fn = (fn: (queue: Queue, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<any>, size: c.Size) => void) => (queue: any, buffer: any, bufferOffset: any, data: any, size: any) => void fn(new Queue().setNative(queue), new Buffer().setNative(buffer), new c.U64().setNative(bufferOffset), new c.Pointer<any>().setNative(data), new c.Size().setNative(size))
  static new = (fn: (queue: Queue, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<any>, size: c.Size) => void) => new ProcQueueWriteBuffer().set(fn)
}
export class ProcQueueWriteTexture extends c.Function<[queue: Queue, destination: c.Pointer<ImageCopyTexture>, data: c.Pointer<any>, dataSize: c.Size, dataLayout: c.Pointer<TextureDataLayout>, writeSize: c.Pointer<Extent3D>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer', 'usize', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (queue: Queue, destination: c.Pointer<ImageCopyTexture>, data: c.Pointer<any>, dataSize: c.Size, dataLayout: c.Pointer<TextureDataLayout>, writeSize: c.Pointer<Extent3D>) => void) => (queue: any, destination: any, data: any, dataSize: any, dataLayout: any, writeSize: any) => void fn(new Queue().setNative(queue), new c.Pointer<ImageCopyTexture>().setNative(destination), new c.Pointer<any>().setNative(data), new c.Size().setNative(dataSize), new c.Pointer<TextureDataLayout>().setNative(dataLayout), new c.Pointer<Extent3D>().setNative(writeSize))
  static new = (fn: (queue: Queue, destination: c.Pointer<ImageCopyTexture>, data: c.Pointer<any>, dataSize: c.Size, dataLayout: c.Pointer<TextureDataLayout>, writeSize: c.Pointer<Extent3D>) => void) => new ProcQueueWriteTexture().set(fn)
}
export class ProcQueueAddRef extends c.Function<[queue: Queue]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (queue: Queue) => void) => (queue: any) => void fn(new Queue().setNative(queue))
  static new = (fn: (queue: Queue) => void) => new ProcQueueAddRef().set(fn)
}
export class ProcQueueRelease extends c.Function<[queue: Queue]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (queue: Queue) => void) => (queue: any) => void fn(new Queue().setNative(queue))
  static new = (fn: (queue: Queue) => void) => new ProcQueueRelease().set(fn)
}
export class ProcRenderBundleSetLabel extends c.Function<[renderBundle: RenderBundle, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderBundle: RenderBundle, label: StringView) => void) => (renderBundle: any, label: any) => void fn(new RenderBundle().setNative(renderBundle), new StringView().setNative(label))
  static new = (fn: (renderBundle: RenderBundle, label: StringView) => void) => new ProcRenderBundleSetLabel().set(fn)
}
export class ProcRenderBundleAddRef extends c.Function<[renderBundle: RenderBundle]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderBundle: RenderBundle) => void) => (renderBundle: any) => void fn(new RenderBundle().setNative(renderBundle))
  static new = (fn: (renderBundle: RenderBundle) => void) => new ProcRenderBundleAddRef().set(fn)
}
export class ProcRenderBundleRelease extends c.Function<[renderBundle: RenderBundle]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderBundle: RenderBundle) => void) => (renderBundle: any) => void fn(new RenderBundle().setNative(renderBundle))
  static new = (fn: (renderBundle: RenderBundle) => void) => new ProcRenderBundleRelease().set(fn)
}
export class ProcRenderBundleEncoderDraw extends c.Function<[renderBundleEncoder: RenderBundleEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'u32', 'u32', 'u32'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32) => void) => (renderBundleEncoder: any, vertexCount: any, instanceCount: any, firstVertex: any, firstInstance: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new c.U32().setNative(vertexCount), new c.U32().setNative(instanceCount), new c.U32().setNative(firstVertex), new c.U32().setNative(firstInstance))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32) => void) => new ProcRenderBundleEncoderDraw().set(fn)
}
export class ProcRenderBundleEncoderDrawIndexed extends c.Function<[renderBundleEncoder: RenderBundleEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'u32', 'u32', 'i32', 'u32'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32) => void) => (renderBundleEncoder: any, indexCount: any, instanceCount: any, firstIndex: any, baseVertex: any, firstInstance: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new c.U32().setNative(indexCount), new c.U32().setNative(instanceCount), new c.U32().setNative(firstIndex), new c.I32().setNative(baseVertex), new c.U32().setNative(firstInstance))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32) => void) => new ProcRenderBundleEncoderDrawIndexed().set(fn)
}
export class ProcRenderBundleEncoderDrawIndexedIndirect extends c.Function<[renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => (renderBundleEncoder: any, indirectBuffer: any, indirectOffset: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => new ProcRenderBundleEncoderDrawIndexedIndirect().set(fn)
}
export class ProcRenderBundleEncoderDrawIndirect extends c.Function<[renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => (renderBundleEncoder: any, indirectBuffer: any, indirectOffset: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => new ProcRenderBundleEncoderDrawIndirect().set(fn)
}
export class ProcRenderBundleEncoderFinish extends c.Function<[renderBundleEncoder: RenderBundleEncoder, descriptor: c.Pointer<RenderBundleDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, descriptor: c.Pointer<RenderBundleDescriptor>) => void) => (renderBundleEncoder: any, descriptor: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new c.Pointer<RenderBundleDescriptor>().setNative(descriptor))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, descriptor: c.Pointer<RenderBundleDescriptor>) => void) => new ProcRenderBundleEncoderFinish().set(fn)
}
export class ProcRenderBundleEncoderInsertDebugMarker extends c.Function<[renderBundleEncoder: RenderBundleEncoder, markerLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, markerLabel: StringView) => void) => (renderBundleEncoder: any, markerLabel: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new StringView().setNative(markerLabel))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, markerLabel: StringView) => void) => new ProcRenderBundleEncoderInsertDebugMarker().set(fn)
}
export class ProcRenderBundleEncoderPopDebugGroup extends c.Function<[renderBundleEncoder: RenderBundleEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder) => void) => (renderBundleEncoder: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder) => void) => new ProcRenderBundleEncoderPopDebugGroup().set(fn)
}
export class ProcRenderBundleEncoderPushDebugGroup extends c.Function<[renderBundleEncoder: RenderBundleEncoder, groupLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, groupLabel: StringView) => void) => (renderBundleEncoder: any, groupLabel: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new StringView().setNative(groupLabel))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, groupLabel: StringView) => void) => new ProcRenderBundleEncoderPushDebugGroup().set(fn)
}
export class ProcRenderBundleEncoderSetBindGroup extends c.Function<[renderBundleEncoder: RenderBundleEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>) => void) => (renderBundleEncoder: any, groupIndex: any, group: any, dynamicOffsetCount: any, dynamicOffsets: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new c.U32().setNative(groupIndex), new BindGroup().setNative(group), new c.Size().setNative(dynamicOffsetCount), new c.Pointer<c.U32>().setNative(dynamicOffsets))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>) => void) => new ProcRenderBundleEncoderSetBindGroup().set(fn)
}
export class ProcRenderBundleEncoderSetIndexBuffer extends c.Function<[renderBundleEncoder: RenderBundleEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u32', 'u64', 'u64'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64) => void) => (renderBundleEncoder: any, buffer: any, format: any, offset: any, size: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new Buffer().setNative(buffer), new IndexFormat().setNative(format), new c.U64().setNative(offset), new c.U64().setNative(size))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64) => void) => new ProcRenderBundleEncoderSetIndexBuffer().set(fn)
}
export class ProcRenderBundleEncoderSetLabel extends c.Function<[renderBundleEncoder: RenderBundleEncoder, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, label: StringView) => void) => (renderBundleEncoder: any, label: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new StringView().setNative(label))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, label: StringView) => void) => new ProcRenderBundleEncoderSetLabel().set(fn)
}
export class ProcRenderBundleEncoderSetPipeline extends c.Function<[renderBundleEncoder: RenderBundleEncoder, pipeline: RenderPipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, pipeline: RenderPipeline) => void) => (renderBundleEncoder: any, pipeline: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new RenderPipeline().setNative(pipeline))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, pipeline: RenderPipeline) => void) => new ProcRenderBundleEncoderSetPipeline().set(fn)
}
export class ProcRenderBundleEncoderSetVertexBuffer extends c.Function<[renderBundleEncoder: RenderBundleEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'pointer', 'u64', 'u64'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64) => void) => (renderBundleEncoder: any, slot: any, buffer: any, offset: any, size: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder), new c.U32().setNative(slot), new Buffer().setNative(buffer), new c.U64().setNative(offset), new c.U64().setNative(size))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64) => void) => new ProcRenderBundleEncoderSetVertexBuffer().set(fn)
}
export class ProcRenderBundleEncoderAddRef extends c.Function<[renderBundleEncoder: RenderBundleEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder) => void) => (renderBundleEncoder: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder) => void) => new ProcRenderBundleEncoderAddRef().set(fn)
}
export class ProcRenderBundleEncoderRelease extends c.Function<[renderBundleEncoder: RenderBundleEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderBundleEncoder: RenderBundleEncoder) => void) => (renderBundleEncoder: any) => void fn(new RenderBundleEncoder().setNative(renderBundleEncoder))
  static new = (fn: (renderBundleEncoder: RenderBundleEncoder) => void) => new ProcRenderBundleEncoderRelease().set(fn)
}
export class ProcRenderPassEncoderBeginOcclusionQuery extends c.Function<[renderPassEncoder: RenderPassEncoder, queryIndex: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, queryIndex: c.U32) => void) => (renderPassEncoder: any, queryIndex: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(queryIndex))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, queryIndex: c.U32) => void) => new ProcRenderPassEncoderBeginOcclusionQuery().set(fn)
}
export class ProcRenderPassEncoderDraw extends c.Function<[renderPassEncoder: RenderPassEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'u32', 'u32', 'u32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32) => void) => (renderPassEncoder: any, vertexCount: any, instanceCount: any, firstVertex: any, firstInstance: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(vertexCount), new c.U32().setNative(instanceCount), new c.U32().setNative(firstVertex), new c.U32().setNative(firstInstance))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32) => void) => new ProcRenderPassEncoderDraw().set(fn)
}
export class ProcRenderPassEncoderDrawIndexed extends c.Function<[renderPassEncoder: RenderPassEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'u32', 'u32', 'i32', 'u32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32) => void) => (renderPassEncoder: any, indexCount: any, instanceCount: any, firstIndex: any, baseVertex: any, firstInstance: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(indexCount), new c.U32().setNative(instanceCount), new c.U32().setNative(firstIndex), new c.I32().setNative(baseVertex), new c.U32().setNative(firstInstance))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32) => void) => new ProcRenderPassEncoderDrawIndexed().set(fn)
}
export class ProcRenderPassEncoderDrawIndexedIndirect extends c.Function<[renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => (renderPassEncoder: any, indirectBuffer: any, indirectOffset: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => new ProcRenderPassEncoderDrawIndexedIndirect().set(fn)
}
export class ProcRenderPassEncoderDrawIndirect extends c.Function<[renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => (renderPassEncoder: any, indirectBuffer: any, indirectOffset: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64) => void) => new ProcRenderPassEncoderDrawIndirect().set(fn)
}
export class ProcRenderPassEncoderEnd extends c.Function<[renderPassEncoder: RenderPassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder) => void) => (renderPassEncoder: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder))
  static new = (fn: (renderPassEncoder: RenderPassEncoder) => void) => new ProcRenderPassEncoderEnd().set(fn)
}
export class ProcRenderPassEncoderEndOcclusionQuery extends c.Function<[renderPassEncoder: RenderPassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder) => void) => (renderPassEncoder: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder))
  static new = (fn: (renderPassEncoder: RenderPassEncoder) => void) => new ProcRenderPassEncoderEndOcclusionQuery().set(fn)
}
export class ProcRenderPassEncoderExecuteBundles extends c.Function<[renderPassEncoder: RenderPassEncoder, bundleCount: c.Size, bundles: c.Pointer<RenderBundle>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, bundleCount: c.Size, bundles: c.Pointer<RenderBundle>) => void) => (renderPassEncoder: any, bundleCount: any, bundles: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.Size().setNative(bundleCount), new c.Pointer<RenderBundle>().setNative(bundles))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, bundleCount: c.Size, bundles: c.Pointer<RenderBundle>) => void) => new ProcRenderPassEncoderExecuteBundles().set(fn)
}
export class ProcRenderPassEncoderInsertDebugMarker extends c.Function<[renderPassEncoder: RenderPassEncoder, markerLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, markerLabel: StringView) => void) => (renderPassEncoder: any, markerLabel: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new StringView().setNative(markerLabel))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, markerLabel: StringView) => void) => new ProcRenderPassEncoderInsertDebugMarker().set(fn)
}
export class ProcRenderPassEncoderMultiDrawIndexedIndirect extends c.Function<[renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64', 'u32', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64) => void) => (renderPassEncoder: any, indirectBuffer: any, indirectOffset: any, maxDrawCount: any, drawCountBuffer: any, drawCountBufferOffset: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset), new c.U32().setNative(maxDrawCount), new Buffer().setNative(drawCountBuffer), new c.U64().setNative(drawCountBufferOffset))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64) => void) => new ProcRenderPassEncoderMultiDrawIndexedIndirect().set(fn)
}
export class ProcRenderPassEncoderMultiDrawIndirect extends c.Function<[renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u64', 'u32', 'pointer', 'u64'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64) => void) => (renderPassEncoder: any, indirectBuffer: any, indirectOffset: any, maxDrawCount: any, drawCountBuffer: any, drawCountBufferOffset: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new Buffer().setNative(indirectBuffer), new c.U64().setNative(indirectOffset), new c.U32().setNative(maxDrawCount), new Buffer().setNative(drawCountBuffer), new c.U64().setNative(drawCountBufferOffset))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64) => void) => new ProcRenderPassEncoderMultiDrawIndirect().set(fn)
}
export class ProcRenderPassEncoderPixelLocalStorageBarrier extends c.Function<[renderPassEncoder: RenderPassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder) => void) => (renderPassEncoder: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder))
  static new = (fn: (renderPassEncoder: RenderPassEncoder) => void) => new ProcRenderPassEncoderPixelLocalStorageBarrier().set(fn)
}
export class ProcRenderPassEncoderPopDebugGroup extends c.Function<[renderPassEncoder: RenderPassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder) => void) => (renderPassEncoder: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder))
  static new = (fn: (renderPassEncoder: RenderPassEncoder) => void) => new ProcRenderPassEncoderPopDebugGroup().set(fn)
}
export class ProcRenderPassEncoderPushDebugGroup extends c.Function<[renderPassEncoder: RenderPassEncoder, groupLabel: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, groupLabel: StringView) => void) => (renderPassEncoder: any, groupLabel: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new StringView().setNative(groupLabel))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, groupLabel: StringView) => void) => new ProcRenderPassEncoderPushDebugGroup().set(fn)
}
export class ProcRenderPassEncoderSetBindGroup extends c.Function<[renderPassEncoder: RenderPassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'pointer', 'usize', 'pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>) => void) => (renderPassEncoder: any, groupIndex: any, group: any, dynamicOffsetCount: any, dynamicOffsets: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(groupIndex), new BindGroup().setNative(group), new c.Size().setNative(dynamicOffsetCount), new c.Pointer<c.U32>().setNative(dynamicOffsets))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>) => void) => new ProcRenderPassEncoderSetBindGroup().set(fn)
}
export class ProcRenderPassEncoderSetBlendConstant extends c.Function<[renderPassEncoder: RenderPassEncoder, color: c.Pointer<Color>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, color: c.Pointer<Color>) => void) => (renderPassEncoder: any, color: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.Pointer<Color>().setNative(color))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, color: c.Pointer<Color>) => void) => new ProcRenderPassEncoderSetBlendConstant().set(fn)
}
export class ProcRenderPassEncoderSetIndexBuffer extends c.Function<[renderPassEncoder: RenderPassEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u32', 'u64', 'u64'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64) => void) => (renderPassEncoder: any, buffer: any, format: any, offset: any, size: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new Buffer().setNative(buffer), new IndexFormat().setNative(format), new c.U64().setNative(offset), new c.U64().setNative(size))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64) => void) => new ProcRenderPassEncoderSetIndexBuffer().set(fn)
}
export class ProcRenderPassEncoderSetLabel extends c.Function<[renderPassEncoder: RenderPassEncoder, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, label: StringView) => void) => (renderPassEncoder: any, label: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new StringView().setNative(label))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, label: StringView) => void) => new ProcRenderPassEncoderSetLabel().set(fn)
}
export class ProcRenderPassEncoderSetPipeline extends c.Function<[renderPassEncoder: RenderPassEncoder, pipeline: RenderPipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, pipeline: RenderPipeline) => void) => (renderPassEncoder: any, pipeline: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new RenderPipeline().setNative(pipeline))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, pipeline: RenderPipeline) => void) => new ProcRenderPassEncoderSetPipeline().set(fn)
}
export class ProcRenderPassEncoderSetScissorRect extends c.Function<[renderPassEncoder: RenderPassEncoder, x: c.U32, y: c.U32, width: c.U32, height: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'u32', 'u32', 'u32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, x: c.U32, y: c.U32, width: c.U32, height: c.U32) => void) => (renderPassEncoder: any, x: any, y: any, width: any, height: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(x), new c.U32().setNative(y), new c.U32().setNative(width), new c.U32().setNative(height))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, x: c.U32, y: c.U32, width: c.U32, height: c.U32) => void) => new ProcRenderPassEncoderSetScissorRect().set(fn)
}
export class ProcRenderPassEncoderSetStencilReference extends c.Function<[renderPassEncoder: RenderPassEncoder, reference: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, reference: c.U32) => void) => (renderPassEncoder: any, reference: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(reference))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, reference: c.U32) => void) => new ProcRenderPassEncoderSetStencilReference().set(fn)
}
export class ProcRenderPassEncoderSetVertexBuffer extends c.Function<[renderPassEncoder: RenderPassEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32', 'pointer', 'u64', 'u64'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64) => void) => (renderPassEncoder: any, slot: any, buffer: any, offset: any, size: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.U32().setNative(slot), new Buffer().setNative(buffer), new c.U64().setNative(offset), new c.U64().setNative(size))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64) => void) => new ProcRenderPassEncoderSetVertexBuffer().set(fn)
}
export class ProcRenderPassEncoderSetViewport extends c.Function<[renderPassEncoder: RenderPassEncoder, x: c.F32, y: c.F32, width: c.F32, height: c.F32, minDepth: c.F32, maxDepth: c.F32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'f32', 'f32', 'f32', 'f32', 'f32', 'f32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, x: c.F32, y: c.F32, width: c.F32, height: c.F32, minDepth: c.F32, maxDepth: c.F32) => void) => (renderPassEncoder: any, x: any, y: any, width: any, height: any, minDepth: any, maxDepth: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new c.F32().setNative(x), new c.F32().setNative(y), new c.F32().setNative(width), new c.F32().setNative(height), new c.F32().setNative(minDepth), new c.F32().setNative(maxDepth))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, x: c.F32, y: c.F32, width: c.F32, height: c.F32, minDepth: c.F32, maxDepth: c.F32) => void) => new ProcRenderPassEncoderSetViewport().set(fn)
}
export class ProcRenderPassEncoderWriteTimestamp extends c.Function<[renderPassEncoder: RenderPassEncoder, querySet: QuerySet, queryIndex: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'u32'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder, querySet: QuerySet, queryIndex: c.U32) => void) => (renderPassEncoder: any, querySet: any, queryIndex: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder), new QuerySet().setNative(querySet), new c.U32().setNative(queryIndex))
  static new = (fn: (renderPassEncoder: RenderPassEncoder, querySet: QuerySet, queryIndex: c.U32) => void) => new ProcRenderPassEncoderWriteTimestamp().set(fn)
}
export class ProcRenderPassEncoderAddRef extends c.Function<[renderPassEncoder: RenderPassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder) => void) => (renderPassEncoder: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder))
  static new = (fn: (renderPassEncoder: RenderPassEncoder) => void) => new ProcRenderPassEncoderAddRef().set(fn)
}
export class ProcRenderPassEncoderRelease extends c.Function<[renderPassEncoder: RenderPassEncoder]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPassEncoder: RenderPassEncoder) => void) => (renderPassEncoder: any) => void fn(new RenderPassEncoder().setNative(renderPassEncoder))
  static new = (fn: (renderPassEncoder: RenderPassEncoder) => void) => new ProcRenderPassEncoderRelease().set(fn)
}
export class ProcRenderPipelineGetBindGroupLayout extends c.Function<[renderPipeline: RenderPipeline, groupIndex: c.U32]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'u32'])
  }
  protected override _fn = (fn: (renderPipeline: RenderPipeline, groupIndex: c.U32) => void) => (renderPipeline: any, groupIndex: any) => void fn(new RenderPipeline().setNative(renderPipeline), new c.U32().setNative(groupIndex))
  static new = (fn: (renderPipeline: RenderPipeline, groupIndex: c.U32) => void) => new ProcRenderPipelineGetBindGroupLayout().set(fn)
}
export class ProcRenderPipelineSetLabel extends c.Function<[renderPipeline: RenderPipeline, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (renderPipeline: RenderPipeline, label: StringView) => void) => (renderPipeline: any, label: any) => void fn(new RenderPipeline().setNative(renderPipeline), new StringView().setNative(label))
  static new = (fn: (renderPipeline: RenderPipeline, label: StringView) => void) => new ProcRenderPipelineSetLabel().set(fn)
}
export class ProcRenderPipelineAddRef extends c.Function<[renderPipeline: RenderPipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPipeline: RenderPipeline) => void) => (renderPipeline: any) => void fn(new RenderPipeline().setNative(renderPipeline))
  static new = (fn: (renderPipeline: RenderPipeline) => void) => new ProcRenderPipelineAddRef().set(fn)
}
export class ProcRenderPipelineRelease extends c.Function<[renderPipeline: RenderPipeline]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (renderPipeline: RenderPipeline) => void) => (renderPipeline: any) => void fn(new RenderPipeline().setNative(renderPipeline))
  static new = (fn: (renderPipeline: RenderPipeline) => void) => new ProcRenderPipelineRelease().set(fn)
}
export class ProcSamplerSetLabel extends c.Function<[sampler: Sampler, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (sampler: Sampler, label: StringView) => void) => (sampler: any, label: any) => void fn(new Sampler().setNative(sampler), new StringView().setNative(label))
  static new = (fn: (sampler: Sampler, label: StringView) => void) => new ProcSamplerSetLabel().set(fn)
}
export class ProcSamplerAddRef extends c.Function<[sampler: Sampler]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sampler: Sampler) => void) => (sampler: any) => void fn(new Sampler().setNative(sampler))
  static new = (fn: (sampler: Sampler) => void) => new ProcSamplerAddRef().set(fn)
}
export class ProcSamplerRelease extends c.Function<[sampler: Sampler]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sampler: Sampler) => void) => (sampler: any) => void fn(new Sampler().setNative(sampler))
  static new = (fn: (sampler: Sampler) => void) => new ProcSamplerRelease().set(fn)
}
export class ProcShaderModuleGetCompilationInfo extends c.Function<[shaderModule: ShaderModule, callback: CompilationInfoCallback, userdata: c.Pointer<any>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'function', 'pointer'])
  }
  protected override _fn = (fn: (shaderModule: ShaderModule, callback: CompilationInfoCallback, userdata: c.Pointer<any>) => void) => (shaderModule: any, callback: any, userdata: any) => void fn(new ShaderModule().setNative(shaderModule), new CompilationInfoCallback().setNative(callback), new c.Pointer<any>().setNative(userdata))
  static new = (fn: (shaderModule: ShaderModule, callback: CompilationInfoCallback, userdata: c.Pointer<any>) => void) => new ProcShaderModuleGetCompilationInfo().set(fn)
}
export class ProcShaderModuleGetCompilationInfo2 extends c.Function<[shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo2]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo2) => void) => (shaderModule: any, callbackInfo: any) => void fn(new ShaderModule().setNative(shaderModule), new CompilationInfoCallbackInfo2().setNative(callbackInfo))
  static new = (fn: (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo2) => void) => new ProcShaderModuleGetCompilationInfo2().set(fn)
}
export class ProcShaderModuleGetCompilationInfoF extends c.Function<[shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo) => void) => (shaderModule: any, callbackInfo: any) => void fn(new ShaderModule().setNative(shaderModule), new CompilationInfoCallbackInfo().setNative(callbackInfo))
  static new = (fn: (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo) => void) => new ProcShaderModuleGetCompilationInfoF().set(fn)
}
export class ProcShaderModuleSetLabel extends c.Function<[shaderModule: ShaderModule, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (shaderModule: ShaderModule, label: StringView) => void) => (shaderModule: any, label: any) => void fn(new ShaderModule().setNative(shaderModule), new StringView().setNative(label))
  static new = (fn: (shaderModule: ShaderModule, label: StringView) => void) => new ProcShaderModuleSetLabel().set(fn)
}
export class ProcShaderModuleAddRef extends c.Function<[shaderModule: ShaderModule]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (shaderModule: ShaderModule) => void) => (shaderModule: any) => void fn(new ShaderModule().setNative(shaderModule))
  static new = (fn: (shaderModule: ShaderModule) => void) => new ProcShaderModuleAddRef().set(fn)
}
export class ProcShaderModuleRelease extends c.Function<[shaderModule: ShaderModule]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (shaderModule: ShaderModule) => void) => (shaderModule: any) => void fn(new ShaderModule().setNative(shaderModule))
  static new = (fn: (shaderModule: ShaderModule) => void) => new ProcShaderModuleRelease().set(fn)
}
export class ProcSharedBufferMemoryBeginAccess extends c.Function<[sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryBeginAccessDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryBeginAccessDescriptor>) => void) => (sharedBufferMemory: any, buffer: any, descriptor: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory), new Buffer().setNative(buffer), new c.Pointer<SharedBufferMemoryBeginAccessDescriptor>().setNative(descriptor))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryBeginAccessDescriptor>) => void) => new ProcSharedBufferMemoryBeginAccess().set(fn)
}
export class ProcSharedBufferMemoryCreateBuffer extends c.Function<[sharedBufferMemory: SharedBufferMemory, descriptor: c.Pointer<BufferDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory, descriptor: c.Pointer<BufferDescriptor>) => void) => (sharedBufferMemory: any, descriptor: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory), new c.Pointer<BufferDescriptor>().setNative(descriptor))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory, descriptor: c.Pointer<BufferDescriptor>) => void) => new ProcSharedBufferMemoryCreateBuffer().set(fn)
}
export class ProcSharedBufferMemoryEndAccess extends c.Function<[sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryEndAccessState>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryEndAccessState>) => void) => (sharedBufferMemory: any, buffer: any, descriptor: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory), new Buffer().setNative(buffer), new c.Pointer<SharedBufferMemoryEndAccessState>().setNative(descriptor))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryEndAccessState>) => void) => new ProcSharedBufferMemoryEndAccess().set(fn)
}
export class ProcSharedBufferMemoryGetProperties extends c.Function<[sharedBufferMemory: SharedBufferMemory, properties: c.Pointer<SharedBufferMemoryProperties>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory, properties: c.Pointer<SharedBufferMemoryProperties>) => void) => (sharedBufferMemory: any, properties: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory), new c.Pointer<SharedBufferMemoryProperties>().setNative(properties))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory, properties: c.Pointer<SharedBufferMemoryProperties>) => void) => new ProcSharedBufferMemoryGetProperties().set(fn)
}
export class ProcSharedBufferMemoryIsDeviceLost extends c.Function<[sharedBufferMemory: SharedBufferMemory]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory) => void) => (sharedBufferMemory: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory) => void) => new ProcSharedBufferMemoryIsDeviceLost().set(fn)
}
export class ProcSharedBufferMemorySetLabel extends c.Function<[sharedBufferMemory: SharedBufferMemory, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory, label: StringView) => void) => (sharedBufferMemory: any, label: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory), new StringView().setNative(label))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory, label: StringView) => void) => new ProcSharedBufferMemorySetLabel().set(fn)
}
export class ProcSharedBufferMemoryAddRef extends c.Function<[sharedBufferMemory: SharedBufferMemory]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory) => void) => (sharedBufferMemory: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory) => void) => new ProcSharedBufferMemoryAddRef().set(fn)
}
export class ProcSharedBufferMemoryRelease extends c.Function<[sharedBufferMemory: SharedBufferMemory]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedBufferMemory: SharedBufferMemory) => void) => (sharedBufferMemory: any) => void fn(new SharedBufferMemory().setNative(sharedBufferMemory))
  static new = (fn: (sharedBufferMemory: SharedBufferMemory) => void) => new ProcSharedBufferMemoryRelease().set(fn)
}
export class ProcSharedFenceExportInfo extends c.Function<[sharedFence: SharedFence, info: c.Pointer<SharedFenceExportInfo>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedFence: SharedFence, info: c.Pointer<SharedFenceExportInfo>) => void) => (sharedFence: any, info: any) => void fn(new SharedFence().setNative(sharedFence), new c.Pointer<SharedFenceExportInfo>().setNative(info))
  static new = (fn: (sharedFence: SharedFence, info: c.Pointer<SharedFenceExportInfo>) => void) => new ProcSharedFenceExportInfo().set(fn)
}
export class ProcSharedFenceAddRef extends c.Function<[sharedFence: SharedFence]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedFence: SharedFence) => void) => (sharedFence: any) => void fn(new SharedFence().setNative(sharedFence))
  static new = (fn: (sharedFence: SharedFence) => void) => new ProcSharedFenceAddRef().set(fn)
}
export class ProcSharedFenceRelease extends c.Function<[sharedFence: SharedFence]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedFence: SharedFence) => void) => (sharedFence: any) => void fn(new SharedFence().setNative(sharedFence))
  static new = (fn: (sharedFence: SharedFence) => void) => new ProcSharedFenceRelease().set(fn)
}
export class ProcSharedTextureMemoryBeginAccess extends c.Function<[sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryBeginAccessDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryBeginAccessDescriptor>) => void) => (sharedTextureMemory: any, texture: any, descriptor: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory), new Texture().setNative(texture), new c.Pointer<SharedTextureMemoryBeginAccessDescriptor>().setNative(descriptor))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryBeginAccessDescriptor>) => void) => new ProcSharedTextureMemoryBeginAccess().set(fn)
}
export class ProcSharedTextureMemoryCreateTexture extends c.Function<[sharedTextureMemory: SharedTextureMemory, descriptor: c.Pointer<TextureDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory, descriptor: c.Pointer<TextureDescriptor>) => void) => (sharedTextureMemory: any, descriptor: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory), new c.Pointer<TextureDescriptor>().setNative(descriptor))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory, descriptor: c.Pointer<TextureDescriptor>) => void) => new ProcSharedTextureMemoryCreateTexture().set(fn)
}
export class ProcSharedTextureMemoryEndAccess extends c.Function<[sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryEndAccessState>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryEndAccessState>) => void) => (sharedTextureMemory: any, texture: any, descriptor: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory), new Texture().setNative(texture), new c.Pointer<SharedTextureMemoryEndAccessState>().setNative(descriptor))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryEndAccessState>) => void) => new ProcSharedTextureMemoryEndAccess().set(fn)
}
export class ProcSharedTextureMemoryGetProperties extends c.Function<[sharedTextureMemory: SharedTextureMemory, properties: c.Pointer<SharedTextureMemoryProperties>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory, properties: c.Pointer<SharedTextureMemoryProperties>) => void) => (sharedTextureMemory: any, properties: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory), new c.Pointer<SharedTextureMemoryProperties>().setNative(properties))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory, properties: c.Pointer<SharedTextureMemoryProperties>) => void) => new ProcSharedTextureMemoryGetProperties().set(fn)
}
export class ProcSharedTextureMemoryIsDeviceLost extends c.Function<[sharedTextureMemory: SharedTextureMemory]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory) => void) => (sharedTextureMemory: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory) => void) => new ProcSharedTextureMemoryIsDeviceLost().set(fn)
}
export class ProcSharedTextureMemorySetLabel extends c.Function<[sharedTextureMemory: SharedTextureMemory, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory, label: StringView) => void) => (sharedTextureMemory: any, label: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory), new StringView().setNative(label))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory, label: StringView) => void) => new ProcSharedTextureMemorySetLabel().set(fn)
}
export class ProcSharedTextureMemoryAddRef extends c.Function<[sharedTextureMemory: SharedTextureMemory]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory) => void) => (sharedTextureMemory: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory) => void) => new ProcSharedTextureMemoryAddRef().set(fn)
}
export class ProcSharedTextureMemoryRelease extends c.Function<[sharedTextureMemory: SharedTextureMemory]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (sharedTextureMemory: SharedTextureMemory) => void) => (sharedTextureMemory: any) => void fn(new SharedTextureMemory().setNative(sharedTextureMemory))
  static new = (fn: (sharedTextureMemory: SharedTextureMemory) => void) => new ProcSharedTextureMemoryRelease().set(fn)
}
export class ProcSurfaceConfigure extends c.Function<[surface: Surface, config: c.Pointer<SurfaceConfiguration>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (surface: Surface, config: c.Pointer<SurfaceConfiguration>) => void) => (surface: any, config: any) => void fn(new Surface().setNative(surface), new c.Pointer<SurfaceConfiguration>().setNative(config))
  static new = (fn: (surface: Surface, config: c.Pointer<SurfaceConfiguration>) => void) => new ProcSurfaceConfigure().set(fn)
}
export class ProcSurfaceGetCapabilities extends c.Function<[surface: Surface, adapter: Adapter, capabilities: c.Pointer<SurfaceCapabilities>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer', 'pointer'])
  }
  protected override _fn = (fn: (surface: Surface, adapter: Adapter, capabilities: c.Pointer<SurfaceCapabilities>) => void) => (surface: any, adapter: any, capabilities: any) => void fn(new Surface().setNative(surface), new Adapter().setNative(adapter), new c.Pointer<SurfaceCapabilities>().setNative(capabilities))
  static new = (fn: (surface: Surface, adapter: Adapter, capabilities: c.Pointer<SurfaceCapabilities>) => void) => new ProcSurfaceGetCapabilities().set(fn)
}
export class ProcSurfaceGetCurrentTexture extends c.Function<[surface: Surface, surfaceTexture: c.Pointer<SurfaceTexture>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (surface: Surface, surfaceTexture: c.Pointer<SurfaceTexture>) => void) => (surface: any, surfaceTexture: any) => void fn(new Surface().setNative(surface), new c.Pointer<SurfaceTexture>().setNative(surfaceTexture))
  static new = (fn: (surface: Surface, surfaceTexture: c.Pointer<SurfaceTexture>) => void) => new ProcSurfaceGetCurrentTexture().set(fn)
}
export class ProcSurfacePresent extends c.Function<[surface: Surface]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (surface: Surface) => void) => (surface: any) => void fn(new Surface().setNative(surface))
  static new = (fn: (surface: Surface) => void) => new ProcSurfacePresent().set(fn)
}
export class ProcSurfaceSetLabel extends c.Function<[surface: Surface, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (surface: Surface, label: StringView) => void) => (surface: any, label: any) => void fn(new Surface().setNative(surface), new StringView().setNative(label))
  static new = (fn: (surface: Surface, label: StringView) => void) => new ProcSurfaceSetLabel().set(fn)
}
export class ProcSurfaceUnconfigure extends c.Function<[surface: Surface]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (surface: Surface) => void) => (surface: any) => void fn(new Surface().setNative(surface))
  static new = (fn: (surface: Surface) => void) => new ProcSurfaceUnconfigure().set(fn)
}
export class ProcSurfaceAddRef extends c.Function<[surface: Surface]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (surface: Surface) => void) => (surface: any) => void fn(new Surface().setNative(surface))
  static new = (fn: (surface: Surface) => void) => new ProcSurfaceAddRef().set(fn)
}
export class ProcSurfaceRelease extends c.Function<[surface: Surface]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (surface: Surface) => void) => (surface: any) => void fn(new Surface().setNative(surface))
  static new = (fn: (surface: Surface) => void) => new ProcSurfaceRelease().set(fn)
}
export class ProcTextureCreateErrorView extends c.Function<[texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>) => void) => (texture: any, descriptor: any) => void fn(new Texture().setNative(texture), new c.Pointer<TextureViewDescriptor>().setNative(descriptor))
  static new = (fn: (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>) => void) => new ProcTextureCreateErrorView().set(fn)
}
export class ProcTextureCreateView extends c.Function<[texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'pointer'])
  }
  protected override _fn = (fn: (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>) => void) => (texture: any, descriptor: any) => void fn(new Texture().setNative(texture), new c.Pointer<TextureViewDescriptor>().setNative(descriptor))
  static new = (fn: (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>) => void) => new ProcTextureCreateView().set(fn)
}
export class ProcTextureDestroy extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureDestroy().set(fn)
}
export class ProcTextureGetDepthOrArrayLayers extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetDepthOrArrayLayers().set(fn)
}
export class ProcTextureGetDimension extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetDimension().set(fn)
}
export class ProcTextureGetFormat extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetFormat().set(fn)
}
export class ProcTextureGetHeight extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetHeight().set(fn)
}
export class ProcTextureGetMipLevelCount extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetMipLevelCount().set(fn)
}
export class ProcTextureGetSampleCount extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetSampleCount().set(fn)
}
export class ProcTextureGetUsage extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetUsage().set(fn)
}
export class ProcTextureGetWidth extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureGetWidth().set(fn)
}
export class ProcTextureSetLabel extends c.Function<[texture: Texture, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (texture: Texture, label: StringView) => void) => (texture: any, label: any) => void fn(new Texture().setNative(texture), new StringView().setNative(label))
  static new = (fn: (texture: Texture, label: StringView) => void) => new ProcTextureSetLabel().set(fn)
}
export class ProcTextureAddRef extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureAddRef().set(fn)
}
export class ProcTextureRelease extends c.Function<[texture: Texture]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (texture: Texture) => void) => (texture: any) => void fn(new Texture().setNative(texture))
  static new = (fn: (texture: Texture) => void) => new ProcTextureRelease().set(fn)
}
export class ProcTextureViewSetLabel extends c.Function<[textureView: TextureView, label: StringView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer', 'buffer'])
  }
  protected override _fn = (fn: (textureView: TextureView, label: StringView) => void) => (textureView: any, label: any) => void fn(new TextureView().setNative(textureView), new StringView().setNative(label))
  static new = (fn: (textureView: TextureView, label: StringView) => void) => new ProcTextureViewSetLabel().set(fn)
}
export class ProcTextureViewAddRef extends c.Function<[textureView: TextureView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (textureView: TextureView) => void) => (textureView: any) => void fn(new TextureView().setNative(textureView))
  static new = (fn: (textureView: TextureView) => void) => new ProcTextureViewAddRef().set(fn)
}
export class ProcTextureViewRelease extends c.Function<[textureView: TextureView]> {
  constructor(buffer?: ArrayBuffer, offset?: number) {
    super(buffer, offset, ['pointer'])
  }
  protected override _fn = (fn: (textureView: TextureView) => void) => (textureView: any) => void fn(new TextureView().setNative(textureView))
  static new = (fn: (textureView: TextureView) => void) => new ProcTextureViewRelease().set(fn)
}

// functions
export const adapterInfoFreeMembers = (value: AdapterInfo): c.Void => new c.Void().setNative(lib.symbols.wgpuAdapterInfoFreeMembers(value.native))
export const adapterPropertiesMemoryHeapsFreeMembers = (value: AdapterPropertiesMemoryHeaps): c.Void => new c.Void().setNative(lib.symbols.wgpuAdapterPropertiesMemoryHeapsFreeMembers(value.native))
export const createInstance = (descriptor: c.Pointer<InstanceDescriptor>): Instance => new Instance().setNative(lib.symbols.wgpuCreateInstance(descriptor.native))
export const drmFormatCapabilitiesFreeMembers = (value: DrmFormatCapabilities): c.Void => new c.Void().setNative(lib.symbols.wgpuDrmFormatCapabilitiesFreeMembers(value.native))
export const getInstanceFeatures = (features: c.Pointer<InstanceFeatures>): Status => new Status().setNative(lib.symbols.wgpuGetInstanceFeatures(features.native))
export const getProcAddress = (procName: StringView): Proc => new Proc().setNative(lib.symbols.wgpuGetProcAddress(procName.native))
export const sharedBufferMemoryEndAccessStateFreeMembers = (value: SharedBufferMemoryEndAccessState): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedBufferMemoryEndAccessStateFreeMembers(value.native))
export const sharedTextureMemoryEndAccessStateFreeMembers = (value: SharedTextureMemoryEndAccessState): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedTextureMemoryEndAccessStateFreeMembers(value.native))
export const supportedFeaturesFreeMembers = (value: SupportedFeatures): c.Void => new c.Void().setNative(lib.symbols.wgpuSupportedFeaturesFreeMembers(value.native))
export const surfaceCapabilitiesFreeMembers = (value: SurfaceCapabilities): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceCapabilitiesFreeMembers(value.native))
export const adapterCreateDevice = (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>): Device => new Device().setNative(lib.symbols.wgpuAdapterCreateDevice(adapter.native, descriptor.native))
export const adapterGetFeatures = (adapter: Adapter, features: c.Pointer<SupportedFeatures>): c.Void => new c.Void().setNative(lib.symbols.wgpuAdapterGetFeatures(adapter.native, features.native))
export const adapterGetFormatCapabilities = (adapter: Adapter, format: TextureFormat, capabilities: c.Pointer<FormatCapabilities>): Status => new Status().setNative(lib.symbols.wgpuAdapterGetFormatCapabilities(adapter.native, format.native, capabilities.native))
export const adapterGetInfo = (adapter: Adapter, info: c.Pointer<AdapterInfo>): Status => new Status().setNative(lib.symbols.wgpuAdapterGetInfo(adapter.native, info.native))
export const adapterGetInstance = (adapter: Adapter): Instance => new Instance().setNative(lib.symbols.wgpuAdapterGetInstance(adapter.native))
export const adapterGetLimits = (adapter: Adapter, limits: c.Pointer<SupportedLimits>): Status => new Status().setNative(lib.symbols.wgpuAdapterGetLimits(adapter.native, limits.native))
export const adapterHasFeature = (adapter: Adapter, feature: FeatureName): Bool => new Bool().setNative(lib.symbols.wgpuAdapterHasFeature(adapter.native, feature.native))
export const adapterRequestDevice = (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>, callback: RequestDeviceCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuAdapterRequestDevice(adapter.native, descriptor.native, callback.native, userdata.native))
export const adapterRequestDevice2 = (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuAdapterRequestDevice2(adapter.native, options.native, callbackInfo.native))
export const adapterRequestDeviceF = (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuAdapterRequestDeviceF(adapter.native, options.native, callbackInfo.native))
export const adapterAddRef = (adapter: Adapter): c.Void => new c.Void().setNative(lib.symbols.wgpuAdapterAddRef(adapter.native))
export const adapterRelease = (adapter: Adapter): c.Void => new c.Void().setNative(lib.symbols.wgpuAdapterRelease(adapter.native))
export const bindGroupSetLabel = (bindGroup: BindGroup, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuBindGroupSetLabel(bindGroup.native, label.native))
export const bindGroupAddRef = (bindGroup: BindGroup): c.Void => new c.Void().setNative(lib.symbols.wgpuBindGroupAddRef(bindGroup.native))
export const bindGroupRelease = (bindGroup: BindGroup): c.Void => new c.Void().setNative(lib.symbols.wgpuBindGroupRelease(bindGroup.native))
export const bindGroupLayoutSetLabel = (bindGroupLayout: BindGroupLayout, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuBindGroupLayoutSetLabel(bindGroupLayout.native, label.native))
export const bindGroupLayoutAddRef = (bindGroupLayout: BindGroupLayout): c.Void => new c.Void().setNative(lib.symbols.wgpuBindGroupLayoutAddRef(bindGroupLayout.native))
export const bindGroupLayoutRelease = (bindGroupLayout: BindGroupLayout): c.Void => new c.Void().setNative(lib.symbols.wgpuBindGroupLayoutRelease(bindGroupLayout.native))
export const bufferDestroy = (buffer: Buffer): c.Void => new c.Void().setNative(lib.symbols.wgpuBufferDestroy(buffer.native))
export const bufferGetConstMappedRange = (buffer: Buffer, offset: c.Size, size: c.Size): c.Pointer<c.Void> => new c.Pointer<c.Void>().setNative(lib.symbols.wgpuBufferGetConstMappedRange(buffer.native, offset.native, size.native))
export const bufferGetMapState = (buffer: Buffer): BufferMapState => new BufferMapState().setNative(lib.symbols.wgpuBufferGetMapState(buffer.native))
export const bufferGetMappedRange = (buffer: Buffer, offset: c.Size, size: c.Size): c.Pointer<c.Void> => new c.Pointer<c.Void>().setNative(lib.symbols.wgpuBufferGetMappedRange(buffer.native, offset.native, size.native))
export const bufferGetSize = (buffer: Buffer): c.U64 => new c.U64().setNative(lib.symbols.wgpuBufferGetSize(buffer.native))
export const bufferGetUsage = (buffer: Buffer): BufferUsage => new BufferUsage().setNative(lib.symbols.wgpuBufferGetUsage(buffer.native))
export const bufferMapAsync = (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callback: BufferMapCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuBufferMapAsync(buffer.native, mode.native, offset.native, size.native, callback.native, userdata.native))
export const bufferMapAsync2 = (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuBufferMapAsync2(buffer.native, mode.native, offset.native, size.native, callbackInfo.native))
export const bufferMapAsyncF = (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuBufferMapAsyncF(buffer.native, mode.native, offset.native, size.native, callbackInfo.native))
export const bufferSetLabel = (buffer: Buffer, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuBufferSetLabel(buffer.native, label.native))
export const bufferUnmap = (buffer: Buffer): c.Void => new c.Void().setNative(lib.symbols.wgpuBufferUnmap(buffer.native))
export const bufferAddRef = (buffer: Buffer): c.Void => new c.Void().setNative(lib.symbols.wgpuBufferAddRef(buffer.native))
export const bufferRelease = (buffer: Buffer): c.Void => new c.Void().setNative(lib.symbols.wgpuBufferRelease(buffer.native))
export const commandBufferSetLabel = (commandBuffer: CommandBuffer, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandBufferSetLabel(commandBuffer.native, label.native))
export const commandBufferAddRef = (commandBuffer: CommandBuffer): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandBufferAddRef(commandBuffer.native))
export const commandBufferRelease = (commandBuffer: CommandBuffer): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandBufferRelease(commandBuffer.native))
export const commandEncoderBeginComputePass = (commandEncoder: CommandEncoder, descriptor: c.Pointer<ComputePassDescriptor>): ComputePassEncoder => new ComputePassEncoder().setNative(lib.symbols.wgpuCommandEncoderBeginComputePass(commandEncoder.native, descriptor.native))
export const commandEncoderBeginRenderPass = (commandEncoder: CommandEncoder, descriptor: c.Pointer<RenderPassDescriptor>): RenderPassEncoder => new RenderPassEncoder().setNative(lib.symbols.wgpuCommandEncoderBeginRenderPass(commandEncoder.native, descriptor.native))
export const commandEncoderClearBuffer = (commandEncoder: CommandEncoder, buffer: Buffer, offset: c.U64, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderClearBuffer(commandEncoder.native, buffer.native, offset.native, size.native))
export const commandEncoderCopyBufferToBuffer = (commandEncoder: CommandEncoder, source: Buffer, sourceOffset: c.U64, destination: Buffer, destinationOffset: c.U64, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderCopyBufferToBuffer(commandEncoder.native, source.native, sourceOffset.native, destination.native, destinationOffset.native, size.native))
export const commandEncoderCopyBufferToTexture = (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyBuffer>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderCopyBufferToTexture(commandEncoder.native, source.native, destination.native, copySize.native))
export const commandEncoderCopyTextureToBuffer = (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyBuffer>, copySize: c.Pointer<Extent3D>): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderCopyTextureToBuffer(commandEncoder.native, source.native, destination.native, copySize.native))
export const commandEncoderCopyTextureToTexture = (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderCopyTextureToTexture(commandEncoder.native, source.native, destination.native, copySize.native))
export const commandEncoderFinish = (commandEncoder: CommandEncoder, descriptor: c.Pointer<CommandBufferDescriptor>): CommandBuffer => new CommandBuffer().setNative(lib.symbols.wgpuCommandEncoderFinish(commandEncoder.native, descriptor.native))
export const commandEncoderInjectValidationError = (commandEncoder: CommandEncoder, message: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderInjectValidationError(commandEncoder.native, message.native))
export const commandEncoderInsertDebugMarker = (commandEncoder: CommandEncoder, markerLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderInsertDebugMarker(commandEncoder.native, markerLabel.native))
export const commandEncoderPopDebugGroup = (commandEncoder: CommandEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderPopDebugGroup(commandEncoder.native))
export const commandEncoderPushDebugGroup = (commandEncoder: CommandEncoder, groupLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderPushDebugGroup(commandEncoder.native, groupLabel.native))
export const commandEncoderResolveQuerySet = (commandEncoder: CommandEncoder, querySet: QuerySet, firstQuery: c.U32, queryCount: c.U32, destination: Buffer, destinationOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderResolveQuerySet(commandEncoder.native, querySet.native, firstQuery.native, queryCount.native, destination.native, destinationOffset.native))
export const commandEncoderSetLabel = (commandEncoder: CommandEncoder, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderSetLabel(commandEncoder.native, label.native))
export const commandEncoderWriteBuffer = (commandEncoder: CommandEncoder, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.U8>, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderWriteBuffer(commandEncoder.native, buffer.native, bufferOffset.native, data.native, size.native))
export const commandEncoderWriteTimestamp = (commandEncoder: CommandEncoder, querySet: QuerySet, queryIndex: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderWriteTimestamp(commandEncoder.native, querySet.native, queryIndex.native))
export const commandEncoderAddRef = (commandEncoder: CommandEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderAddRef(commandEncoder.native))
export const commandEncoderRelease = (commandEncoder: CommandEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuCommandEncoderRelease(commandEncoder.native))
export const computePassEncoderDispatchWorkgroups = (computePassEncoder: ComputePassEncoder, workgroupCountX: c.U32, workgroupCountY: c.U32, workgroupCountZ: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder.native, workgroupCountX.native, workgroupCountY.native, workgroupCountZ.native))
export const computePassEncoderDispatchWorkgroupsIndirect = (computePassEncoder: ComputePassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder.native, indirectBuffer.native, indirectOffset.native))
export const computePassEncoderEnd = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderEnd(computePassEncoder.native))
export const computePassEncoderInsertDebugMarker = (computePassEncoder: ComputePassEncoder, markerLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderInsertDebugMarker(computePassEncoder.native, markerLabel.native))
export const computePassEncoderPopDebugGroup = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderPopDebugGroup(computePassEncoder.native))
export const computePassEncoderPushDebugGroup = (computePassEncoder: ComputePassEncoder, groupLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderPushDebugGroup(computePassEncoder.native, groupLabel.native))
export const computePassEncoderSetBindGroup = (computePassEncoder: ComputePassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderSetBindGroup(computePassEncoder.native, groupIndex.native, group.native, dynamicOffsetCount.native, dynamicOffsets.native))
export const computePassEncoderSetLabel = (computePassEncoder: ComputePassEncoder, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderSetLabel(computePassEncoder.native, label.native))
export const computePassEncoderSetPipeline = (computePassEncoder: ComputePassEncoder, pipeline: ComputePipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderSetPipeline(computePassEncoder.native, pipeline.native))
export const computePassEncoderWriteTimestamp = (computePassEncoder: ComputePassEncoder, querySet: QuerySet, queryIndex: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderWriteTimestamp(computePassEncoder.native, querySet.native, queryIndex.native))
export const computePassEncoderAddRef = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderAddRef(computePassEncoder.native))
export const computePassEncoderRelease = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePassEncoderRelease(computePassEncoder.native))
export const computePipelineGetBindGroupLayout = (computePipeline: ComputePipeline, groupIndex: c.U32): BindGroupLayout => new BindGroupLayout().setNative(lib.symbols.wgpuComputePipelineGetBindGroupLayout(computePipeline.native, groupIndex.native))
export const computePipelineSetLabel = (computePipeline: ComputePipeline, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePipelineSetLabel(computePipeline.native, label.native))
export const computePipelineAddRef = (computePipeline: ComputePipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePipelineAddRef(computePipeline.native))
export const computePipelineRelease = (computePipeline: ComputePipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuComputePipelineRelease(computePipeline.native))
export const deviceCreateBindGroup = (device: Device, descriptor: c.Pointer<BindGroupDescriptor>): BindGroup => new BindGroup().setNative(lib.symbols.wgpuDeviceCreateBindGroup(device.native, descriptor.native))
export const deviceCreateBindGroupLayout = (device: Device, descriptor: c.Pointer<BindGroupLayoutDescriptor>): BindGroupLayout => new BindGroupLayout().setNative(lib.symbols.wgpuDeviceCreateBindGroupLayout(device.native, descriptor.native))
export const deviceCreateBuffer = (device: Device, descriptor: c.Pointer<BufferDescriptor>): Buffer => new Buffer().setNative(lib.symbols.wgpuDeviceCreateBuffer(device.native, descriptor.native))
export const deviceCreateCommandEncoder = (device: Device, descriptor: c.Pointer<CommandEncoderDescriptor>): CommandEncoder => new CommandEncoder().setNative(lib.symbols.wgpuDeviceCreateCommandEncoder(device.native, descriptor.native))
export const deviceCreateComputePipeline = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>): ComputePipeline => new ComputePipeline().setNative(lib.symbols.wgpuDeviceCreateComputePipeline(device.native, descriptor.native))
export const deviceCreateComputePipelineAsync = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callback: CreateComputePipelineAsyncCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceCreateComputePipelineAsync(device.native, descriptor.native, callback.native, userdata.native))
export const deviceCreateComputePipelineAsync2 = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuDeviceCreateComputePipelineAsync2(device.native, descriptor.native, callbackInfo.native))
export const deviceCreateComputePipelineAsyncF = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuDeviceCreateComputePipelineAsyncF(device.native, descriptor.native, callbackInfo.native))
export const deviceCreateErrorBuffer = (device: Device, descriptor: c.Pointer<BufferDescriptor>): Buffer => new Buffer().setNative(lib.symbols.wgpuDeviceCreateErrorBuffer(device.native, descriptor.native))
export const deviceCreateErrorExternalTexture = (device: Device): ExternalTexture => new ExternalTexture().setNative(lib.symbols.wgpuDeviceCreateErrorExternalTexture(device.native))
export const deviceCreateErrorShaderModule = (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>, errorMessage: StringView): ShaderModule => new ShaderModule().setNative(lib.symbols.wgpuDeviceCreateErrorShaderModule(device.native, descriptor.native, errorMessage.native))
export const deviceCreateErrorTexture = (device: Device, descriptor: c.Pointer<TextureDescriptor>): Texture => new Texture().setNative(lib.symbols.wgpuDeviceCreateErrorTexture(device.native, descriptor.native))
export const deviceCreateExternalTexture = (device: Device, externalTextureDescriptor: c.Pointer<ExternalTextureDescriptor>): ExternalTexture => new ExternalTexture().setNative(lib.symbols.wgpuDeviceCreateExternalTexture(device.native, externalTextureDescriptor.native))
export const deviceCreatePipelineLayout = (device: Device, descriptor: c.Pointer<PipelineLayoutDescriptor>): PipelineLayout => new PipelineLayout().setNative(lib.symbols.wgpuDeviceCreatePipelineLayout(device.native, descriptor.native))
export const deviceCreateQuerySet = (device: Device, descriptor: c.Pointer<QuerySetDescriptor>): QuerySet => new QuerySet().setNative(lib.symbols.wgpuDeviceCreateQuerySet(device.native, descriptor.native))
export const deviceCreateRenderBundleEncoder = (device: Device, descriptor: c.Pointer<RenderBundleEncoderDescriptor>): RenderBundleEncoder => new RenderBundleEncoder().setNative(lib.symbols.wgpuDeviceCreateRenderBundleEncoder(device.native, descriptor.native))
export const deviceCreateRenderPipeline = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>): RenderPipeline => new RenderPipeline().setNative(lib.symbols.wgpuDeviceCreateRenderPipeline(device.native, descriptor.native))
export const deviceCreateRenderPipelineAsync = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callback: CreateRenderPipelineAsyncCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceCreateRenderPipelineAsync(device.native, descriptor.native, callback.native, userdata.native))
export const deviceCreateRenderPipelineAsync2 = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuDeviceCreateRenderPipelineAsync2(device.native, descriptor.native, callbackInfo.native))
export const deviceCreateRenderPipelineAsyncF = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuDeviceCreateRenderPipelineAsyncF(device.native, descriptor.native, callbackInfo.native))
export const deviceCreateSampler = (device: Device, descriptor: c.Pointer<SamplerDescriptor>): Sampler => new Sampler().setNative(lib.symbols.wgpuDeviceCreateSampler(device.native, descriptor.native))
export const deviceCreateShaderModule = (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>): ShaderModule => new ShaderModule().setNative(lib.symbols.wgpuDeviceCreateShaderModule(device.native, descriptor.native))
export const deviceCreateTexture = (device: Device, descriptor: c.Pointer<TextureDescriptor>): Texture => new Texture().setNative(lib.symbols.wgpuDeviceCreateTexture(device.native, descriptor.native))
export const deviceDestroy = (device: Device): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceDestroy(device.native))
export const deviceForceLoss = (device: Device, type: DeviceLostReason, message: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceForceLoss(device.native, type.native, message.native))
export const deviceGetAHardwareBufferProperties = (device: Device, handle: c.Pointer<c.Void>, properties: c.Pointer<AHardwareBufferProperties>): Status => new Status().setNative(lib.symbols.wgpuDeviceGetAHardwareBufferProperties(device.native, handle.native, properties.native))
export const deviceGetAdapter = (device: Device): Adapter => new Adapter().setNative(lib.symbols.wgpuDeviceGetAdapter(device.native))
export const deviceGetAdapterInfo = (device: Device, adapterInfo: c.Pointer<AdapterInfo>): Status => new Status().setNative(lib.symbols.wgpuDeviceGetAdapterInfo(device.native, adapterInfo.native))
export const deviceGetFeatures = (device: Device, features: c.Pointer<SupportedFeatures>): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceGetFeatures(device.native, features.native))
export const deviceGetLimits = (device: Device, limits: c.Pointer<SupportedLimits>): Status => new Status().setNative(lib.symbols.wgpuDeviceGetLimits(device.native, limits.native))
export const deviceGetLostFuture = (device: Device): Future => new Future().setNative(lib.symbols.wgpuDeviceGetLostFuture(device.native))
export const deviceGetQueue = (device: Device): Queue => new Queue().setNative(lib.symbols.wgpuDeviceGetQueue(device.native))
export const deviceHasFeature = (device: Device, feature: FeatureName): Bool => new Bool().setNative(lib.symbols.wgpuDeviceHasFeature(device.native, feature.native))
export const deviceImportSharedBufferMemory = (device: Device, descriptor: c.Pointer<SharedBufferMemoryDescriptor>): SharedBufferMemory => new SharedBufferMemory().setNative(lib.symbols.wgpuDeviceImportSharedBufferMemory(device.native, descriptor.native))
export const deviceImportSharedFence = (device: Device, descriptor: c.Pointer<SharedFenceDescriptor>): SharedFence => new SharedFence().setNative(lib.symbols.wgpuDeviceImportSharedFence(device.native, descriptor.native))
export const deviceImportSharedTextureMemory = (device: Device, descriptor: c.Pointer<SharedTextureMemoryDescriptor>): SharedTextureMemory => new SharedTextureMemory().setNative(lib.symbols.wgpuDeviceImportSharedTextureMemory(device.native, descriptor.native))
export const deviceInjectError = (device: Device, type: ErrorType, message: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceInjectError(device.native, type.native, message.native))
export const devicePopErrorScope = (device: Device, oldCallback: ErrorCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuDevicePopErrorScope(device.native, oldCallback.native, userdata.native))
export const devicePopErrorScope2 = (device: Device, callbackInfo: PopErrorScopeCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuDevicePopErrorScope2(device.native, callbackInfo.native))
export const devicePopErrorScopeF = (device: Device, callbackInfo: PopErrorScopeCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuDevicePopErrorScopeF(device.native, callbackInfo.native))
export const devicePushErrorScope = (device: Device, filter: ErrorFilter): c.Void => new c.Void().setNative(lib.symbols.wgpuDevicePushErrorScope(device.native, filter.native))
export const deviceSetLabel = (device: Device, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceSetLabel(device.native, label.native))
export const deviceSetLoggingCallback = (device: Device, callback: LoggingCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceSetLoggingCallback(device.native, callback.native, userdata.native))
export const deviceTick = (device: Device): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceTick(device.native))
export const deviceValidateTextureDescriptor = (device: Device, descriptor: c.Pointer<TextureDescriptor>): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceValidateTextureDescriptor(device.native, descriptor.native))
export const deviceAddRef = (device: Device): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceAddRef(device.native))
export const deviceRelease = (device: Device): c.Void => new c.Void().setNative(lib.symbols.wgpuDeviceRelease(device.native))
export const externalTextureDestroy = (externalTexture: ExternalTexture): c.Void => new c.Void().setNative(lib.symbols.wgpuExternalTextureDestroy(externalTexture.native))
export const externalTextureExpire = (externalTexture: ExternalTexture): c.Void => new c.Void().setNative(lib.symbols.wgpuExternalTextureExpire(externalTexture.native))
export const externalTextureRefresh = (externalTexture: ExternalTexture): c.Void => new c.Void().setNative(lib.symbols.wgpuExternalTextureRefresh(externalTexture.native))
export const externalTextureSetLabel = (externalTexture: ExternalTexture, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuExternalTextureSetLabel(externalTexture.native, label.native))
export const externalTextureAddRef = (externalTexture: ExternalTexture): c.Void => new c.Void().setNative(lib.symbols.wgpuExternalTextureAddRef(externalTexture.native))
export const externalTextureRelease = (externalTexture: ExternalTexture): c.Void => new c.Void().setNative(lib.symbols.wgpuExternalTextureRelease(externalTexture.native))
export const instanceCreateSurface = (instance: Instance, descriptor: c.Pointer<SurfaceDescriptor>): Surface => new Surface().setNative(lib.symbols.wgpuInstanceCreateSurface(instance.native, descriptor.native))
export const instanceEnumerateWGSLLanguageFeatures = (instance: Instance, features: c.Pointer<WGSLFeatureName>): c.Size => new c.Size().setNative(lib.symbols.wgpuInstanceEnumerateWGSLLanguageFeatures(instance.native, features.native))
export const instanceHasWGSLLanguageFeature = (instance: Instance, feature: WGSLFeatureName): Bool => new Bool().setNative(lib.symbols.wgpuInstanceHasWGSLLanguageFeature(instance.native, feature.native))
export const instanceProcessEvents = (instance: Instance): c.Void => new c.Void().setNative(lib.symbols.wgpuInstanceProcessEvents(instance.native))
export const instanceRequestAdapter = (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callback: RequestAdapterCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuInstanceRequestAdapter(instance.native, options.native, callback.native, userdata.native))
export const instanceRequestAdapter2 = (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuInstanceRequestAdapter2(instance.native, options.native, callbackInfo.native))
export const instanceRequestAdapterF = (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuInstanceRequestAdapterF(instance.native, options.native, callbackInfo.native))
export const instanceWaitAny = (instance: Instance, futureCount: c.Size, futures: c.Pointer<FutureWaitInfo>, timeoutNS: c.U64): WaitStatus => new WaitStatus().setNative(lib.symbols.wgpuInstanceWaitAny(instance.native, futureCount.native, futures.native, timeoutNS.native))
export const instanceAddRef = (instance: Instance): c.Void => new c.Void().setNative(lib.symbols.wgpuInstanceAddRef(instance.native))
export const instanceRelease = (instance: Instance): c.Void => new c.Void().setNative(lib.symbols.wgpuInstanceRelease(instance.native))
export const pipelineLayoutSetLabel = (pipelineLayout: PipelineLayout, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuPipelineLayoutSetLabel(pipelineLayout.native, label.native))
export const pipelineLayoutAddRef = (pipelineLayout: PipelineLayout): c.Void => new c.Void().setNative(lib.symbols.wgpuPipelineLayoutAddRef(pipelineLayout.native))
export const pipelineLayoutRelease = (pipelineLayout: PipelineLayout): c.Void => new c.Void().setNative(lib.symbols.wgpuPipelineLayoutRelease(pipelineLayout.native))
export const querySetDestroy = (querySet: QuerySet): c.Void => new c.Void().setNative(lib.symbols.wgpuQuerySetDestroy(querySet.native))
export const querySetGetCount = (querySet: QuerySet): c.U32 => new c.U32().setNative(lib.symbols.wgpuQuerySetGetCount(querySet.native))
export const querySetGetType = (querySet: QuerySet): QueryType => new QueryType().setNative(lib.symbols.wgpuQuerySetGetType(querySet.native))
export const querySetSetLabel = (querySet: QuerySet, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuQuerySetSetLabel(querySet.native, label.native))
export const querySetAddRef = (querySet: QuerySet): c.Void => new c.Void().setNative(lib.symbols.wgpuQuerySetAddRef(querySet.native))
export const querySetRelease = (querySet: QuerySet): c.Void => new c.Void().setNative(lib.symbols.wgpuQuerySetRelease(querySet.native))
export const queueCopyExternalTextureForBrowser = (queue: Queue, source: c.Pointer<ImageCopyExternalTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueCopyExternalTextureForBrowser(queue.native, source.native, destination.native, copySize.native, options.native))
export const queueCopyTextureForBrowser = (queue: Queue, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueCopyTextureForBrowser(queue.native, source.native, destination.native, copySize.native, options.native))
export const queueOnSubmittedWorkDone = (queue: Queue, callback: QueueWorkDoneCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueOnSubmittedWorkDone(queue.native, callback.native, userdata.native))
export const queueOnSubmittedWorkDone2 = (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuQueueOnSubmittedWorkDone2(queue.native, callbackInfo.native))
export const queueOnSubmittedWorkDoneF = (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuQueueOnSubmittedWorkDoneF(queue.native, callbackInfo.native))
export const queueSetLabel = (queue: Queue, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueSetLabel(queue.native, label.native))
export const queueSubmit = (queue: Queue, commandCount: c.Size, commands: c.Pointer<CommandBuffer>): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueSubmit(queue.native, commandCount.native, commands.native))
export const queueWriteBuffer = (queue: Queue, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.Void>, size: c.Size): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueWriteBuffer(queue.native, buffer.native, bufferOffset.native, data.native, size.native))
export const queueWriteTexture = (queue: Queue, destination: c.Pointer<ImageCopyTexture>, data: c.Pointer<c.Void>, dataSize: c.Size, dataLayout: c.Pointer<TextureDataLayout>, writeSize: c.Pointer<Extent3D>): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueWriteTexture(queue.native, destination.native, data.native, dataSize.native, dataLayout.native, writeSize.native))
export const queueAddRef = (queue: Queue): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueAddRef(queue.native))
export const queueRelease = (queue: Queue): c.Void => new c.Void().setNative(lib.symbols.wgpuQueueRelease(queue.native))
export const renderBundleSetLabel = (renderBundle: RenderBundle, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleSetLabel(renderBundle.native, label.native))
export const renderBundleAddRef = (renderBundle: RenderBundle): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleAddRef(renderBundle.native))
export const renderBundleRelease = (renderBundle: RenderBundle): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleRelease(renderBundle.native))
export const renderBundleEncoderDraw = (renderBundleEncoder: RenderBundleEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderDraw(renderBundleEncoder.native, vertexCount.native, instanceCount.native, firstVertex.native, firstInstance.native))
export const renderBundleEncoderDrawIndexed = (renderBundleEncoder: RenderBundleEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder.native, indexCount.native, instanceCount.native, firstIndex.native, baseVertex.native, firstInstance.native))
export const renderBundleEncoderDrawIndexedIndirect = (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder.native, indirectBuffer.native, indirectOffset.native))
export const renderBundleEncoderDrawIndirect = (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder.native, indirectBuffer.native, indirectOffset.native))
export const renderBundleEncoderFinish = (renderBundleEncoder: RenderBundleEncoder, descriptor: c.Pointer<RenderBundleDescriptor>): RenderBundle => new RenderBundle().setNative(lib.symbols.wgpuRenderBundleEncoderFinish(renderBundleEncoder.native, descriptor.native))
export const renderBundleEncoderInsertDebugMarker = (renderBundleEncoder: RenderBundleEncoder, markerLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder.native, markerLabel.native))
export const renderBundleEncoderPopDebugGroup = (renderBundleEncoder: RenderBundleEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder.native))
export const renderBundleEncoderPushDebugGroup = (renderBundleEncoder: RenderBundleEncoder, groupLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder.native, groupLabel.native))
export const renderBundleEncoderSetBindGroup = (renderBundleEncoder: RenderBundleEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder.native, groupIndex.native, group.native, dynamicOffsetCount.native, dynamicOffsets.native))
export const renderBundleEncoderSetIndexBuffer = (renderBundleEncoder: RenderBundleEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder.native, buffer.native, format.native, offset.native, size.native))
export const renderBundleEncoderSetLabel = (renderBundleEncoder: RenderBundleEncoder, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderSetLabel(renderBundleEncoder.native, label.native))
export const renderBundleEncoderSetPipeline = (renderBundleEncoder: RenderBundleEncoder, pipeline: RenderPipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder.native, pipeline.native))
export const renderBundleEncoderSetVertexBuffer = (renderBundleEncoder: RenderBundleEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder.native, slot.native, buffer.native, offset.native, size.native))
export const renderBundleEncoderAddRef = (renderBundleEncoder: RenderBundleEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderAddRef(renderBundleEncoder.native))
export const renderBundleEncoderRelease = (renderBundleEncoder: RenderBundleEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderBundleEncoderRelease(renderBundleEncoder.native))
export const renderPassEncoderBeginOcclusionQuery = (renderPassEncoder: RenderPassEncoder, queryIndex: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder.native, queryIndex.native))
export const renderPassEncoderDraw = (renderPassEncoder: RenderPassEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderDraw(renderPassEncoder.native, vertexCount.native, instanceCount.native, firstVertex.native, firstInstance.native))
export const renderPassEncoderDrawIndexed = (renderPassEncoder: RenderPassEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderDrawIndexed(renderPassEncoder.native, indexCount.native, instanceCount.native, firstIndex.native, baseVertex.native, firstInstance.native))
export const renderPassEncoderDrawIndexedIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder.native, indirectBuffer.native, indirectOffset.native))
export const renderPassEncoderDrawIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderDrawIndirect(renderPassEncoder.native, indirectBuffer.native, indirectOffset.native))
export const renderPassEncoderEnd = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderEnd(renderPassEncoder.native))
export const renderPassEncoderEndOcclusionQuery = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder.native))
export const renderPassEncoderExecuteBundles = (renderPassEncoder: RenderPassEncoder, bundleCount: c.Size, bundles: c.Pointer<RenderBundle>): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderExecuteBundles(renderPassEncoder.native, bundleCount.native, bundles.native))
export const renderPassEncoderInsertDebugMarker = (renderPassEncoder: RenderPassEncoder, markerLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder.native, markerLabel.native))
export const renderPassEncoderMultiDrawIndexedIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder.native, indirectBuffer.native, indirectOffset.native, maxDrawCount.native, drawCountBuffer.native, drawCountBufferOffset.native))
export const renderPassEncoderMultiDrawIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder.native, indirectBuffer.native, indirectOffset.native, maxDrawCount.native, drawCountBuffer.native, drawCountBufferOffset.native))
export const renderPassEncoderPixelLocalStorageBarrier = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder.native))
export const renderPassEncoderPopDebugGroup = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder.native))
export const renderPassEncoderPushDebugGroup = (renderPassEncoder: RenderPassEncoder, groupLabel: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder.native, groupLabel.native))
export const renderPassEncoderSetBindGroup = (renderPassEncoder: RenderPassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetBindGroup(renderPassEncoder.native, groupIndex.native, group.native, dynamicOffsetCount.native, dynamicOffsets.native))
export const renderPassEncoderSetBlendConstant = (renderPassEncoder: RenderPassEncoder, color: c.Pointer<Color>): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder.native, color.native))
export const renderPassEncoderSetIndexBuffer = (renderPassEncoder: RenderPassEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder.native, buffer.native, format.native, offset.native, size.native))
export const renderPassEncoderSetLabel = (renderPassEncoder: RenderPassEncoder, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetLabel(renderPassEncoder.native, label.native))
export const renderPassEncoderSetPipeline = (renderPassEncoder: RenderPassEncoder, pipeline: RenderPipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetPipeline(renderPassEncoder.native, pipeline.native))
export const renderPassEncoderSetScissorRect = (renderPassEncoder: RenderPassEncoder, x: c.U32, y: c.U32, width: c.U32, height: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetScissorRect(renderPassEncoder.native, x.native, y.native, width.native, height.native))
export const renderPassEncoderSetStencilReference = (renderPassEncoder: RenderPassEncoder, reference: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetStencilReference(renderPassEncoder.native, reference.native))
export const renderPassEncoderSetVertexBuffer = (renderPassEncoder: RenderPassEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder.native, slot.native, buffer.native, offset.native, size.native))
export const renderPassEncoderSetViewport = (renderPassEncoder: RenderPassEncoder, x: c.F32, y: c.F32, width: c.F32, height: c.F32, minDepth: c.F32, maxDepth: c.F32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderSetViewport(renderPassEncoder.native, x.native, y.native, width.native, height.native, minDepth.native, maxDepth.native))
export const renderPassEncoderWriteTimestamp = (renderPassEncoder: RenderPassEncoder, querySet: QuerySet, queryIndex: c.U32): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder.native, querySet.native, queryIndex.native))
export const renderPassEncoderAddRef = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderAddRef(renderPassEncoder.native))
export const renderPassEncoderRelease = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPassEncoderRelease(renderPassEncoder.native))
export const renderPipelineGetBindGroupLayout = (renderPipeline: RenderPipeline, groupIndex: c.U32): BindGroupLayout => new BindGroupLayout().setNative(lib.symbols.wgpuRenderPipelineGetBindGroupLayout(renderPipeline.native, groupIndex.native))
export const renderPipelineSetLabel = (renderPipeline: RenderPipeline, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPipelineSetLabel(renderPipeline.native, label.native))
export const renderPipelineAddRef = (renderPipeline: RenderPipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPipelineAddRef(renderPipeline.native))
export const renderPipelineRelease = (renderPipeline: RenderPipeline): c.Void => new c.Void().setNative(lib.symbols.wgpuRenderPipelineRelease(renderPipeline.native))
export const samplerSetLabel = (sampler: Sampler, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuSamplerSetLabel(sampler.native, label.native))
export const samplerAddRef = (sampler: Sampler): c.Void => new c.Void().setNative(lib.symbols.wgpuSamplerAddRef(sampler.native))
export const samplerRelease = (sampler: Sampler): c.Void => new c.Void().setNative(lib.symbols.wgpuSamplerRelease(sampler.native))
export const shaderModuleGetCompilationInfo = (shaderModule: ShaderModule, callback: CompilationInfoCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void().setNative(lib.symbols.wgpuShaderModuleGetCompilationInfo(shaderModule.native, callback.native, userdata.native))
export const shaderModuleGetCompilationInfo2 = (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo2): Future => new Future().setNative(lib.symbols.wgpuShaderModuleGetCompilationInfo2(shaderModule.native, callbackInfo.native))
export const shaderModuleGetCompilationInfoF = (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo): Future => new Future().setNative(lib.symbols.wgpuShaderModuleGetCompilationInfoF(shaderModule.native, callbackInfo.native))
export const shaderModuleSetLabel = (shaderModule: ShaderModule, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuShaderModuleSetLabel(shaderModule.native, label.native))
export const shaderModuleAddRef = (shaderModule: ShaderModule): c.Void => new c.Void().setNative(lib.symbols.wgpuShaderModuleAddRef(shaderModule.native))
export const shaderModuleRelease = (shaderModule: ShaderModule): c.Void => new c.Void().setNative(lib.symbols.wgpuShaderModuleRelease(shaderModule.native))
export const sharedBufferMemoryBeginAccess = (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryBeginAccessDescriptor>): Status => new Status().setNative(lib.symbols.wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory.native, buffer.native, descriptor.native))
export const sharedBufferMemoryCreateBuffer = (sharedBufferMemory: SharedBufferMemory, descriptor: c.Pointer<BufferDescriptor>): Buffer => new Buffer().setNative(lib.symbols.wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory.native, descriptor.native))
export const sharedBufferMemoryEndAccess = (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryEndAccessState>): Status => new Status().setNative(lib.symbols.wgpuSharedBufferMemoryEndAccess(sharedBufferMemory.native, buffer.native, descriptor.native))
export const sharedBufferMemoryGetProperties = (sharedBufferMemory: SharedBufferMemory, properties: c.Pointer<SharedBufferMemoryProperties>): Status => new Status().setNative(lib.symbols.wgpuSharedBufferMemoryGetProperties(sharedBufferMemory.native, properties.native))
export const sharedBufferMemoryIsDeviceLost = (sharedBufferMemory: SharedBufferMemory): Bool => new Bool().setNative(lib.symbols.wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory.native))
export const sharedBufferMemorySetLabel = (sharedBufferMemory: SharedBufferMemory, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedBufferMemorySetLabel(sharedBufferMemory.native, label.native))
export const sharedBufferMemoryAddRef = (sharedBufferMemory: SharedBufferMemory): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedBufferMemoryAddRef(sharedBufferMemory.native))
export const sharedBufferMemoryRelease = (sharedBufferMemory: SharedBufferMemory): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedBufferMemoryRelease(sharedBufferMemory.native))
export const sharedFenceExportInfo = (sharedFence: SharedFence, info: c.Pointer<SharedFenceExportInfo>): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedFenceExportInfo(sharedFence.native, info.native))
export const sharedFenceAddRef = (sharedFence: SharedFence): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedFenceAddRef(sharedFence.native))
export const sharedFenceRelease = (sharedFence: SharedFence): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedFenceRelease(sharedFence.native))
export const sharedTextureMemoryBeginAccess = (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryBeginAccessDescriptor>): Status => new Status().setNative(lib.symbols.wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory.native, texture.native, descriptor.native))
export const sharedTextureMemoryCreateTexture = (sharedTextureMemory: SharedTextureMemory, descriptor: c.Pointer<TextureDescriptor>): Texture => new Texture().setNative(lib.symbols.wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory.native, descriptor.native))
export const sharedTextureMemoryEndAccess = (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryEndAccessState>): Status => new Status().setNative(lib.symbols.wgpuSharedTextureMemoryEndAccess(sharedTextureMemory.native, texture.native, descriptor.native))
export const sharedTextureMemoryGetProperties = (sharedTextureMemory: SharedTextureMemory, properties: c.Pointer<SharedTextureMemoryProperties>): Status => new Status().setNative(lib.symbols.wgpuSharedTextureMemoryGetProperties(sharedTextureMemory.native, properties.native))
export const sharedTextureMemoryIsDeviceLost = (sharedTextureMemory: SharedTextureMemory): Bool => new Bool().setNative(lib.symbols.wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory.native))
export const sharedTextureMemorySetLabel = (sharedTextureMemory: SharedTextureMemory, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedTextureMemorySetLabel(sharedTextureMemory.native, label.native))
export const sharedTextureMemoryAddRef = (sharedTextureMemory: SharedTextureMemory): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedTextureMemoryAddRef(sharedTextureMemory.native))
export const sharedTextureMemoryRelease = (sharedTextureMemory: SharedTextureMemory): c.Void => new c.Void().setNative(lib.symbols.wgpuSharedTextureMemoryRelease(sharedTextureMemory.native))
export const surfaceConfigure = (surface: Surface, config: c.Pointer<SurfaceConfiguration>): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceConfigure(surface.native, config.native))
export const surfaceGetCapabilities = (surface: Surface, adapter: Adapter, capabilities: c.Pointer<SurfaceCapabilities>): Status => new Status().setNative(lib.symbols.wgpuSurfaceGetCapabilities(surface.native, adapter.native, capabilities.native))
export const surfaceGetCurrentTexture = (surface: Surface, surfaceTexture: c.Pointer<SurfaceTexture>): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceGetCurrentTexture(surface.native, surfaceTexture.native))
export const surfacePresent = (surface: Surface): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfacePresent(surface.native))
export const surfaceSetLabel = (surface: Surface, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceSetLabel(surface.native, label.native))
export const surfaceUnconfigure = (surface: Surface): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceUnconfigure(surface.native))
export const surfaceAddRef = (surface: Surface): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceAddRef(surface.native))
export const surfaceRelease = (surface: Surface): c.Void => new c.Void().setNative(lib.symbols.wgpuSurfaceRelease(surface.native))
export const textureCreateErrorView = (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>): TextureView => new TextureView().setNative(lib.symbols.wgpuTextureCreateErrorView(texture.native, descriptor.native))
export const textureCreateView = (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>): TextureView => new TextureView().setNative(lib.symbols.wgpuTextureCreateView(texture.native, descriptor.native))
export const textureDestroy = (texture: Texture): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureDestroy(texture.native))
export const textureGetDepthOrArrayLayers = (texture: Texture): c.U32 => new c.U32().setNative(lib.symbols.wgpuTextureGetDepthOrArrayLayers(texture.native))
export const textureGetDimension = (texture: Texture): TextureDimension => new TextureDimension().setNative(lib.symbols.wgpuTextureGetDimension(texture.native))
export const textureGetFormat = (texture: Texture): TextureFormat => new TextureFormat().setNative(lib.symbols.wgpuTextureGetFormat(texture.native))
export const textureGetHeight = (texture: Texture): c.U32 => new c.U32().setNative(lib.symbols.wgpuTextureGetHeight(texture.native))
export const textureGetMipLevelCount = (texture: Texture): c.U32 => new c.U32().setNative(lib.symbols.wgpuTextureGetMipLevelCount(texture.native))
export const textureGetSampleCount = (texture: Texture): c.U32 => new c.U32().setNative(lib.symbols.wgpuTextureGetSampleCount(texture.native))
export const textureGetUsage = (texture: Texture): TextureUsage => new TextureUsage().setNative(lib.symbols.wgpuTextureGetUsage(texture.native))
export const textureGetWidth = (texture: Texture): c.U32 => new c.U32().setNative(lib.symbols.wgpuTextureGetWidth(texture.native))
export const textureSetLabel = (texture: Texture, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureSetLabel(texture.native, label.native))
export const textureAddRef = (texture: Texture): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureAddRef(texture.native))
export const textureRelease = (texture: Texture): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureRelease(texture.native))
export const textureViewSetLabel = (textureView: TextureView, label: StringView): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureViewSetLabel(textureView.native, label.native))
export const textureViewAddRef = (textureView: TextureView): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureViewAddRef(textureView.native))
export const textureViewRelease = (textureView: TextureView): c.Void => new c.Void().setNative(lib.symbols.wgpuTextureViewRelease(textureView.native))
