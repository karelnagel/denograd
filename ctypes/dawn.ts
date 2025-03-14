import * as c from './mod.ts'

const lib = Deno.dlopen('/opt/homebrew/Cellar/dawn/0.1.6/lib/libwebgpu_dawn.dylib', {
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
export const WGPUBufferUsage_None = new c.U32(0)
export const WGPUBufferUsage_MapRead = new c.U32(1)
export const WGPUBufferUsage_MapWrite = new c.U32(2)
export const WGPUBufferUsage_CopySrc = new c.U32(4)
export const WGPUBufferUsage_CopyDst = new c.U32(8)
export const WGPUBufferUsage_Index = new c.U32(16)
export const WGPUBufferUsage_Vertex = new c.U32(32)
export const WGPUBufferUsage_Uniform = new c.U32(64)
export const WGPUBufferUsage_Storage = new c.U32(128)
export const WGPUBufferUsage_Indirect = new c.U32(256)
export const WGPUBufferUsage_QueryResolve = new c.U32(512)
export const WGPUColorWriteMask_None = new c.U32(0)
export const WGPUColorWriteMask_Red = new c.U32(1)
export const WGPUColorWriteMask_Green = new c.U32(2)
export const WGPUColorWriteMask_Blue = new c.U32(4)
export const WGPUColorWriteMask_Alpha = new c.U32(8)
export const WGPUColorWriteMask_All = new c.U32(15)
export const WGPUHeapProperty_DeviceLocal = new c.U32(1)
export const WGPUHeapProperty_HostVisible = new c.U32(2)
export const WGPUHeapProperty_HostCoherent = new c.U32(4)
export const WGPUHeapProperty_HostUncached = new c.U32(8)
export const WGPUHeapProperty_HostCached = new c.U32(16)
export const WGPUMapMode_None = new c.U32(0)
export const WGPUMapMode_Read = new c.U32(1)
export const WGPUMapMode_Write = new c.U32(2)
export const WGPUShaderStage_None = new c.U32(0)
export const WGPUShaderStage_Vertex = new c.U32(1)
export const WGPUShaderStage_Fragment = new c.U32(2)
export const WGPUShaderStage_Compute = new c.U32(4)
export const WGPUTextureUsage_None = new c.U32(0)
export const WGPUTextureUsage_CopySrc = new c.U32(1)
export const WGPUTextureUsage_CopyDst = new c.U32(2)
export const WGPUTextureUsage_TextureBinding = new c.U32(4)
export const WGPUTextureUsage_StorageBinding = new c.U32(8)
export const WGPUTextureUsage_RenderAttachment = new c.U32(16)
export const WGPUTextureUsage_TransientAttachment = new c.U32(32)
export const WGPUTextureUsage_StorageAttachment = new c.U32(64)

// enums
export class WGPUWGSLFeatureName extends c.U32 {
  static 'ReadonlyAndReadwriteStorageTextures' = new WGPUWGSLFeatureName(1)
  static 'Packed4x8IntegerDotProduct' = new WGPUWGSLFeatureName(2)
  static 'UnrestrictedPointerParameters' = new WGPUWGSLFeatureName(3)
  static 'PointerCompositeAccess' = new WGPUWGSLFeatureName(4)
  static 'ChromiumTestingUnimplemented' = new WGPUWGSLFeatureName(327680)
  static 'ChromiumTestingUnsafeExperimental' = new WGPUWGSLFeatureName(327681)
  static 'ChromiumTestingExperimental' = new WGPUWGSLFeatureName(327682)
  static 'ChromiumTestingShippedWithKillswitch' = new WGPUWGSLFeatureName(327683)
  static 'ChromiumTestingShipped' = new WGPUWGSLFeatureName(327684)
  static 'Force32' = new WGPUWGSLFeatureName(2147483647)
}
export class WGPUAdapterType extends c.U32 {
  static 'DiscreteGPU' = new WGPUAdapterType(1)
  static 'IntegratedGPU' = new WGPUAdapterType(2)
  static 'CPU' = new WGPUAdapterType(3)
  static 'Unknown' = new WGPUAdapterType(4)
  static 'Force32' = new WGPUAdapterType(2147483647)
}
export class WGPUAddressMode extends c.U32 {
  static 'Undefined' = new WGPUAddressMode(0)
  static 'ClampToEdge' = new WGPUAddressMode(1)
  static 'Repeat' = new WGPUAddressMode(2)
  static 'MirrorRepeat' = new WGPUAddressMode(3)
  static 'Force32' = new WGPUAddressMode(2147483647)
}
export class WGPUAlphaMode extends c.U32 {
  static 'Opaque' = new WGPUAlphaMode(1)
  static 'Premultiplied' = new WGPUAlphaMode(2)
  static 'Unpremultiplied' = new WGPUAlphaMode(3)
  static 'Force32' = new WGPUAlphaMode(2147483647)
}
export class WGPUBackendType extends c.U32 {
  static 'Undefined' = new WGPUBackendType(0)
  static 'Null' = new WGPUBackendType(1)
  static 'WebGPU' = new WGPUBackendType(2)
  static 'D3D11' = new WGPUBackendType(3)
  static 'D3D12' = new WGPUBackendType(4)
  static 'Metal' = new WGPUBackendType(5)
  static 'Vulkan' = new WGPUBackendType(6)
  static 'OpenGL' = new WGPUBackendType(7)
  static 'OpenGLES' = new WGPUBackendType(8)
  static 'Force32' = new WGPUBackendType(2147483647)
}
export class WGPUBlendFactor extends c.U32 {
  static 'Undefined' = new WGPUBlendFactor(0)
  static 'Zero' = new WGPUBlendFactor(1)
  static 'One' = new WGPUBlendFactor(2)
  static 'Src' = new WGPUBlendFactor(3)
  static 'OneMinusSrc' = new WGPUBlendFactor(4)
  static 'SrcAlpha' = new WGPUBlendFactor(5)
  static 'OneMinusSrcAlpha' = new WGPUBlendFactor(6)
  static 'Dst' = new WGPUBlendFactor(7)
  static 'OneMinusDst' = new WGPUBlendFactor(8)
  static 'DstAlpha' = new WGPUBlendFactor(9)
  static 'OneMinusDstAlpha' = new WGPUBlendFactor(10)
  static 'SrcAlphaSaturated' = new WGPUBlendFactor(11)
  static 'Constant' = new WGPUBlendFactor(12)
  static 'OneMinusConstant' = new WGPUBlendFactor(13)
  static 'Src1' = new WGPUBlendFactor(14)
  static 'OneMinusSrc1' = new WGPUBlendFactor(15)
  static 'Src1Alpha' = new WGPUBlendFactor(16)
  static 'OneMinusSrc1Alpha' = new WGPUBlendFactor(17)
  static 'Force32' = new WGPUBlendFactor(2147483647)
}
export class WGPUBlendOperation extends c.U32 {
  static 'Undefined' = new WGPUBlendOperation(0)
  static 'Add' = new WGPUBlendOperation(1)
  static 'Subtract' = new WGPUBlendOperation(2)
  static 'ReverseSubtract' = new WGPUBlendOperation(3)
  static 'Min' = new WGPUBlendOperation(4)
  static 'Max' = new WGPUBlendOperation(5)
  static 'Force32' = new WGPUBlendOperation(2147483647)
}
export class WGPUBufferBindingType extends c.U32 {
  static 'BindingNotUsed' = new WGPUBufferBindingType(0)
  static 'Uniform' = new WGPUBufferBindingType(1)
  static 'Storage' = new WGPUBufferBindingType(2)
  static 'ReadOnlyStorage' = new WGPUBufferBindingType(3)
  static 'Force32' = new WGPUBufferBindingType(2147483647)
}
export class WGPUBufferMapAsyncStatus extends c.U32 {
  static 'Success' = new WGPUBufferMapAsyncStatus(1)
  static 'InstanceDropped' = new WGPUBufferMapAsyncStatus(2)
  static 'ValidationError' = new WGPUBufferMapAsyncStatus(3)
  static 'Unknown' = new WGPUBufferMapAsyncStatus(4)
  static 'DeviceLost' = new WGPUBufferMapAsyncStatus(5)
  static 'DestroyedBeforeCallback' = new WGPUBufferMapAsyncStatus(6)
  static 'UnmappedBeforeCallback' = new WGPUBufferMapAsyncStatus(7)
  static 'MappingAlreadyPending' = new WGPUBufferMapAsyncStatus(8)
  static 'OffsetOutOfRange' = new WGPUBufferMapAsyncStatus(9)
  static 'SizeOutOfRange' = new WGPUBufferMapAsyncStatus(10)
  static 'Force32' = new WGPUBufferMapAsyncStatus(2147483647)
}
export class WGPUBufferMapState extends c.U32 {
  static 'Unmapped' = new WGPUBufferMapState(1)
  static 'Pending' = new WGPUBufferMapState(2)
  static 'Mapped' = new WGPUBufferMapState(3)
  static 'Force32' = new WGPUBufferMapState(2147483647)
}
export class WGPUCallbackMode extends c.U32 {
  static 'WaitAnyOnly' = new WGPUCallbackMode(1)
  static 'AllowProcessEvents' = new WGPUCallbackMode(2)
  static 'AllowSpontaneous' = new WGPUCallbackMode(3)
  static 'Force32' = new WGPUCallbackMode(2147483647)
}
export class WGPUCompareFunction extends c.U32 {
  static 'Undefined' = new WGPUCompareFunction(0)
  static 'Never' = new WGPUCompareFunction(1)
  static 'Less' = new WGPUCompareFunction(2)
  static 'Equal' = new WGPUCompareFunction(3)
  static 'LessEqual' = new WGPUCompareFunction(4)
  static 'Greater' = new WGPUCompareFunction(5)
  static 'NotEqual' = new WGPUCompareFunction(6)
  static 'GreaterEqual' = new WGPUCompareFunction(7)
  static 'Always' = new WGPUCompareFunction(8)
  static 'Force32' = new WGPUCompareFunction(2147483647)
}
export class WGPUCompilationInfoRequestStatus extends c.U32 {
  static 'Success' = new WGPUCompilationInfoRequestStatus(1)
  static 'InstanceDropped' = new WGPUCompilationInfoRequestStatus(2)
  static 'Error' = new WGPUCompilationInfoRequestStatus(3)
  static 'DeviceLost' = new WGPUCompilationInfoRequestStatus(4)
  static 'Unknown' = new WGPUCompilationInfoRequestStatus(5)
  static 'Force32' = new WGPUCompilationInfoRequestStatus(2147483647)
}
export class WGPUCompilationMessageType extends c.U32 {
  static 'Error' = new WGPUCompilationMessageType(1)
  static 'Warning' = new WGPUCompilationMessageType(2)
  static 'Info' = new WGPUCompilationMessageType(3)
  static 'Force32' = new WGPUCompilationMessageType(2147483647)
}
export class WGPUCompositeAlphaMode extends c.U32 {
  static 'Auto' = new WGPUCompositeAlphaMode(0)
  static 'Opaque' = new WGPUCompositeAlphaMode(1)
  static 'Premultiplied' = new WGPUCompositeAlphaMode(2)
  static 'Unpremultiplied' = new WGPUCompositeAlphaMode(3)
  static 'Inherit' = new WGPUCompositeAlphaMode(4)
  static 'Force32' = new WGPUCompositeAlphaMode(2147483647)
}
export class WGPUCreatePipelineAsyncStatus extends c.U32 {
  static 'Success' = new WGPUCreatePipelineAsyncStatus(1)
  static 'InstanceDropped' = new WGPUCreatePipelineAsyncStatus(2)
  static 'ValidationError' = new WGPUCreatePipelineAsyncStatus(3)
  static 'InternalError' = new WGPUCreatePipelineAsyncStatus(4)
  static 'DeviceLost' = new WGPUCreatePipelineAsyncStatus(5)
  static 'DeviceDestroyed' = new WGPUCreatePipelineAsyncStatus(6)
  static 'Unknown' = new WGPUCreatePipelineAsyncStatus(7)
  static 'Force32' = new WGPUCreatePipelineAsyncStatus(2147483647)
}
export class WGPUCullMode extends c.U32 {
  static 'Undefined' = new WGPUCullMode(0)
  static 'None' = new WGPUCullMode(1)
  static 'Front' = new WGPUCullMode(2)
  static 'Back' = new WGPUCullMode(3)
  static 'Force32' = new WGPUCullMode(2147483647)
}
export class WGPUDeviceLostReason extends c.U32 {
  static 'Unknown' = new WGPUDeviceLostReason(1)
  static 'Destroyed' = new WGPUDeviceLostReason(2)
  static 'InstanceDropped' = new WGPUDeviceLostReason(3)
  static 'FailedCreation' = new WGPUDeviceLostReason(4)
  static 'Force32' = new WGPUDeviceLostReason(2147483647)
}
export class WGPUErrorFilter extends c.U32 {
  static 'Validation' = new WGPUErrorFilter(1)
  static 'OutOfMemory' = new WGPUErrorFilter(2)
  static 'Internal' = new WGPUErrorFilter(3)
  static 'Force32' = new WGPUErrorFilter(2147483647)
}
export class WGPUErrorType extends c.U32 {
  static 'NoError' = new WGPUErrorType(1)
  static 'Validation' = new WGPUErrorType(2)
  static 'OutOfMemory' = new WGPUErrorType(3)
  static 'Internal' = new WGPUErrorType(4)
  static 'Unknown' = new WGPUErrorType(5)
  static 'DeviceLost' = new WGPUErrorType(6)
  static 'Force32' = new WGPUErrorType(2147483647)
}
export class WGPUExternalTextureRotation extends c.U32 {
  static 'Rotate0Degrees' = new WGPUExternalTextureRotation(1)
  static 'Rotate90Degrees' = new WGPUExternalTextureRotation(2)
  static 'Rotate180Degrees' = new WGPUExternalTextureRotation(3)
  static 'Rotate270Degrees' = new WGPUExternalTextureRotation(4)
  static 'Force32' = new WGPUExternalTextureRotation(2147483647)
}
export class WGPUFeatureLevel extends c.U32 {
  static 'Undefined' = new WGPUFeatureLevel(0)
  static 'Compatibility' = new WGPUFeatureLevel(1)
  static 'Core' = new WGPUFeatureLevel(2)
  static 'Force32' = new WGPUFeatureLevel(2147483647)
}
export class WGPUFeatureName extends c.U32 {
  static 'DepthClipControl' = new WGPUFeatureName(1)
  static 'Depth32FloatStencil8' = new WGPUFeatureName(2)
  static 'TimestampQuery' = new WGPUFeatureName(3)
  static 'TextureCompressionBC' = new WGPUFeatureName(4)
  static 'TextureCompressionETC2' = new WGPUFeatureName(5)
  static 'TextureCompressionASTC' = new WGPUFeatureName(6)
  static 'IndirectFirstInstance' = new WGPUFeatureName(7)
  static 'ShaderF16' = new WGPUFeatureName(8)
  static 'RG11B10UfloatRenderable' = new WGPUFeatureName(9)
  static 'BGRA8UnormStorage' = new WGPUFeatureName(10)
  static 'Float32Filterable' = new WGPUFeatureName(11)
  static 'Float32Blendable' = new WGPUFeatureName(12)
  static 'Subgroups' = new WGPUFeatureName(13)
  static 'SubgroupsF16' = new WGPUFeatureName(14)
  static 'DawnInternalUsages' = new WGPUFeatureName(327680)
  static 'DawnMultiPlanarFormats' = new WGPUFeatureName(327681)
  static 'DawnNative' = new WGPUFeatureName(327682)
  static 'ChromiumExperimentalTimestampQueryInsidePasses' = new WGPUFeatureName(327683)
  static 'ImplicitDeviceSynchronization' = new WGPUFeatureName(327684)
  static 'ChromiumExperimentalImmediateData' = new WGPUFeatureName(327685)
  static 'TransientAttachments' = new WGPUFeatureName(327686)
  static 'MSAARenderToSingleSampled' = new WGPUFeatureName(327687)
  static 'DualSourceBlending' = new WGPUFeatureName(327688)
  static 'D3D11MultithreadProtected' = new WGPUFeatureName(327689)
  static 'ANGLETextureSharing' = new WGPUFeatureName(327690)
  static 'PixelLocalStorageCoherent' = new WGPUFeatureName(327691)
  static 'PixelLocalStorageNonCoherent' = new WGPUFeatureName(327692)
  static 'Unorm16TextureFormats' = new WGPUFeatureName(327693)
  static 'Snorm16TextureFormats' = new WGPUFeatureName(327694)
  static 'MultiPlanarFormatExtendedUsages' = new WGPUFeatureName(327695)
  static 'MultiPlanarFormatP010' = new WGPUFeatureName(327696)
  static 'HostMappedPointer' = new WGPUFeatureName(327697)
  static 'MultiPlanarRenderTargets' = new WGPUFeatureName(327698)
  static 'MultiPlanarFormatNv12a' = new WGPUFeatureName(327699)
  static 'FramebufferFetch' = new WGPUFeatureName(327700)
  static 'BufferMapExtendedUsages' = new WGPUFeatureName(327701)
  static 'AdapterPropertiesMemoryHeaps' = new WGPUFeatureName(327702)
  static 'AdapterPropertiesD3D' = new WGPUFeatureName(327703)
  static 'AdapterPropertiesVk' = new WGPUFeatureName(327704)
  static 'R8UnormStorage' = new WGPUFeatureName(327705)
  static 'FormatCapabilities' = new WGPUFeatureName(327706)
  static 'DrmFormatCapabilities' = new WGPUFeatureName(327707)
  static 'Norm16TextureFormats' = new WGPUFeatureName(327708)
  static 'MultiPlanarFormatNv16' = new WGPUFeatureName(327709)
  static 'MultiPlanarFormatNv24' = new WGPUFeatureName(327710)
  static 'MultiPlanarFormatP210' = new WGPUFeatureName(327711)
  static 'MultiPlanarFormatP410' = new WGPUFeatureName(327712)
  static 'SharedTextureMemoryVkDedicatedAllocation' = new WGPUFeatureName(327713)
  static 'SharedTextureMemoryAHardwareBuffer' = new WGPUFeatureName(327714)
  static 'SharedTextureMemoryDmaBuf' = new WGPUFeatureName(327715)
  static 'SharedTextureMemoryOpaqueFD' = new WGPUFeatureName(327716)
  static 'SharedTextureMemoryZirconHandle' = new WGPUFeatureName(327717)
  static 'SharedTextureMemoryDXGISharedHandle' = new WGPUFeatureName(327718)
  static 'SharedTextureMemoryD3D11Texture2D' = new WGPUFeatureName(327719)
  static 'SharedTextureMemoryIOSurface' = new WGPUFeatureName(327720)
  static 'SharedTextureMemoryEGLImage' = new WGPUFeatureName(327721)
  static 'SharedFenceVkSemaphoreOpaqueFD' = new WGPUFeatureName(327722)
  static 'SharedFenceSyncFD' = new WGPUFeatureName(327723)
  static 'SharedFenceVkSemaphoreZirconHandle' = new WGPUFeatureName(327724)
  static 'SharedFenceDXGISharedHandle' = new WGPUFeatureName(327725)
  static 'SharedFenceMTLSharedEvent' = new WGPUFeatureName(327726)
  static 'SharedBufferMemoryD3D12Resource' = new WGPUFeatureName(327727)
  static 'StaticSamplers' = new WGPUFeatureName(327728)
  static 'YCbCrVulkanSamplers' = new WGPUFeatureName(327729)
  static 'ShaderModuleCompilationOptions' = new WGPUFeatureName(327730)
  static 'DawnLoadResolveTexture' = new WGPUFeatureName(327731)
  static 'DawnPartialLoadResolveTexture' = new WGPUFeatureName(327732)
  static 'MultiDrawIndirect' = new WGPUFeatureName(327733)
  static 'ClipDistances' = new WGPUFeatureName(327734)
  static 'DawnTexelCopyBufferRowAlignment' = new WGPUFeatureName(327735)
  static 'FlexibleTextureViews' = new WGPUFeatureName(327736)
  static 'Force32' = new WGPUFeatureName(2147483647)
}
export class WGPUFilterMode extends c.U32 {
  static 'Undefined' = new WGPUFilterMode(0)
  static 'Nearest' = new WGPUFilterMode(1)
  static 'Linear' = new WGPUFilterMode(2)
  static 'Force32' = new WGPUFilterMode(2147483647)
}
export class WGPUFrontFace extends c.U32 {
  static 'Undefined' = new WGPUFrontFace(0)
  static 'CCW' = new WGPUFrontFace(1)
  static 'CW' = new WGPUFrontFace(2)
  static 'Force32' = new WGPUFrontFace(2147483647)
}
export class WGPUIndexFormat extends c.U32 {
  static 'Undefined' = new WGPUIndexFormat(0)
  static 'Uint16' = new WGPUIndexFormat(1)
  static 'Uint32' = new WGPUIndexFormat(2)
  static 'Force32' = new WGPUIndexFormat(2147483647)
}
export class WGPULoadOp extends c.U32 {
  static 'Undefined' = new WGPULoadOp(0)
  static 'Load' = new WGPULoadOp(1)
  static 'Clear' = new WGPULoadOp(2)
  static 'ExpandResolveTexture' = new WGPULoadOp(327683)
  static 'Force32' = new WGPULoadOp(2147483647)
}
export class WGPULoggingType extends c.U32 {
  static 'Verbose' = new WGPULoggingType(1)
  static 'Info' = new WGPULoggingType(2)
  static 'Warning' = new WGPULoggingType(3)
  static 'Error' = new WGPULoggingType(4)
  static 'Force32' = new WGPULoggingType(2147483647)
}
export class WGPUMapAsyncStatus extends c.U32 {
  static 'Success' = new WGPUMapAsyncStatus(1)
  static 'InstanceDropped' = new WGPUMapAsyncStatus(2)
  static 'Error' = new WGPUMapAsyncStatus(3)
  static 'Aborted' = new WGPUMapAsyncStatus(4)
  static 'Unknown' = new WGPUMapAsyncStatus(5)
  static 'Force32' = new WGPUMapAsyncStatus(2147483647)
}
export class WGPUMipmapFilterMode extends c.U32 {
  static 'Undefined' = new WGPUMipmapFilterMode(0)
  static 'Nearest' = new WGPUMipmapFilterMode(1)
  static 'Linear' = new WGPUMipmapFilterMode(2)
  static 'Force32' = new WGPUMipmapFilterMode(2147483647)
}
export class WGPUOptionalBool extends c.U32 {
  static 'False' = new WGPUOptionalBool(0)
  static 'True' = new WGPUOptionalBool(1)
  static 'Undefined' = new WGPUOptionalBool(2)
  static 'Force32' = new WGPUOptionalBool(2147483647)
}
export class WGPUPopErrorScopeStatus extends c.U32 {
  static 'Success' = new WGPUPopErrorScopeStatus(1)
  static 'InstanceDropped' = new WGPUPopErrorScopeStatus(2)
  static 'Force32' = new WGPUPopErrorScopeStatus(2147483647)
}
export class WGPUPowerPreference extends c.U32 {
  static 'Undefined' = new WGPUPowerPreference(0)
  static 'LowPower' = new WGPUPowerPreference(1)
  static 'HighPerformance' = new WGPUPowerPreference(2)
  static 'Force32' = new WGPUPowerPreference(2147483647)
}
export class WGPUPresentMode extends c.U32 {
  static 'Fifo' = new WGPUPresentMode(1)
  static 'FifoRelaxed' = new WGPUPresentMode(2)
  static 'Immediate' = new WGPUPresentMode(3)
  static 'Mailbox' = new WGPUPresentMode(4)
  static 'Force32' = new WGPUPresentMode(2147483647)
}
export class WGPUPrimitiveTopology extends c.U32 {
  static 'Undefined' = new WGPUPrimitiveTopology(0)
  static 'PointList' = new WGPUPrimitiveTopology(1)
  static 'LineList' = new WGPUPrimitiveTopology(2)
  static 'LineStrip' = new WGPUPrimitiveTopology(3)
  static 'TriangleList' = new WGPUPrimitiveTopology(4)
  static 'TriangleStrip' = new WGPUPrimitiveTopology(5)
  static 'Force32' = new WGPUPrimitiveTopology(2147483647)
}
export class WGPUQueryType extends c.U32 {
  static 'Occlusion' = new WGPUQueryType(1)
  static 'Timestamp' = new WGPUQueryType(2)
  static 'Force32' = new WGPUQueryType(2147483647)
}
export class WGPUQueueWorkDoneStatus extends c.U32 {
  static 'Success' = new WGPUQueueWorkDoneStatus(1)
  static 'InstanceDropped' = new WGPUQueueWorkDoneStatus(2)
  static 'Error' = new WGPUQueueWorkDoneStatus(3)
  static 'Unknown' = new WGPUQueueWorkDoneStatus(4)
  static 'DeviceLost' = new WGPUQueueWorkDoneStatus(5)
  static 'Force32' = new WGPUQueueWorkDoneStatus(2147483647)
}
export class WGPURequestAdapterStatus extends c.U32 {
  static 'Success' = new WGPURequestAdapterStatus(1)
  static 'InstanceDropped' = new WGPURequestAdapterStatus(2)
  static 'Unavailable' = new WGPURequestAdapterStatus(3)
  static 'Error' = new WGPURequestAdapterStatus(4)
  static 'Unknown' = new WGPURequestAdapterStatus(5)
  static 'Force32' = new WGPURequestAdapterStatus(2147483647)
}
export class WGPURequestDeviceStatus extends c.U32 {
  static 'Success' = new WGPURequestDeviceStatus(1)
  static 'InstanceDropped' = new WGPURequestDeviceStatus(2)
  static 'Error' = new WGPURequestDeviceStatus(3)
  static 'Unknown' = new WGPURequestDeviceStatus(4)
  static 'Force32' = new WGPURequestDeviceStatus(2147483647)
}
export class WGPUSType extends c.U32 {
  static 'ShaderSourceSPIRV' = new WGPUSType(1)
  static 'ShaderSourceWGSL' = new WGPUSType(2)
  static 'RenderPassMaxDrawCount' = new WGPUSType(3)
  static 'SurfaceSourceMetalLayer' = new WGPUSType(4)
  static 'SurfaceSourceWindowsHWND' = new WGPUSType(5)
  static 'SurfaceSourceXlibWindow' = new WGPUSType(6)
  static 'SurfaceSourceWaylandSurface' = new WGPUSType(7)
  static 'SurfaceSourceAndroidNativeWindow' = new WGPUSType(8)
  static 'SurfaceSourceXCBWindow' = new WGPUSType(9)
  static 'AdapterPropertiesSubgroups' = new WGPUSType(10)
  static 'TextureBindingViewDimensionDescriptor' = new WGPUSType(131072)
  static 'SurfaceSourceCanvasHTMLSelector_Emscripten' = new WGPUSType(262144)
  static 'SurfaceDescriptorFromWindowsCoreWindow' = new WGPUSType(327680)
  static 'ExternalTextureBindingEntry' = new WGPUSType(327681)
  static 'ExternalTextureBindingLayout' = new WGPUSType(327682)
  static 'SurfaceDescriptorFromWindowsSwapChainPanel' = new WGPUSType(327683)
  static 'DawnTextureInternalUsageDescriptor' = new WGPUSType(327684)
  static 'DawnEncoderInternalUsageDescriptor' = new WGPUSType(327685)
  static 'DawnInstanceDescriptor' = new WGPUSType(327686)
  static 'DawnCacheDeviceDescriptor' = new WGPUSType(327687)
  static 'DawnAdapterPropertiesPowerPreference' = new WGPUSType(327688)
  static 'DawnBufferDescriptorErrorInfoFromWireClient' = new WGPUSType(327689)
  static 'DawnTogglesDescriptor' = new WGPUSType(327690)
  static 'DawnShaderModuleSPIRVOptionsDescriptor' = new WGPUSType(327691)
  static 'RequestAdapterOptionsLUID' = new WGPUSType(327692)
  static 'RequestAdapterOptionsGetGLProc' = new WGPUSType(327693)
  static 'RequestAdapterOptionsD3D11Device' = new WGPUSType(327694)
  static 'DawnRenderPassColorAttachmentRenderToSingleSampled' = new WGPUSType(327695)
  static 'RenderPassPixelLocalStorage' = new WGPUSType(327696)
  static 'PipelineLayoutPixelLocalStorage' = new WGPUSType(327697)
  static 'BufferHostMappedPointer' = new WGPUSType(327698)
  static 'DawnExperimentalSubgroupLimits' = new WGPUSType(327699)
  static 'AdapterPropertiesMemoryHeaps' = new WGPUSType(327700)
  static 'AdapterPropertiesD3D' = new WGPUSType(327701)
  static 'AdapterPropertiesVk' = new WGPUSType(327702)
  static 'DawnWireWGSLControl' = new WGPUSType(327703)
  static 'DawnWGSLBlocklist' = new WGPUSType(327704)
  static 'DrmFormatCapabilities' = new WGPUSType(327705)
  static 'ShaderModuleCompilationOptions' = new WGPUSType(327706)
  static 'ColorTargetStateExpandResolveTextureDawn' = new WGPUSType(327707)
  static 'RenderPassDescriptorExpandResolveRect' = new WGPUSType(327708)
  static 'SharedTextureMemoryVkDedicatedAllocationDescriptor' = new WGPUSType(327709)
  static 'SharedTextureMemoryAHardwareBufferDescriptor' = new WGPUSType(327710)
  static 'SharedTextureMemoryDmaBufDescriptor' = new WGPUSType(327711)
  static 'SharedTextureMemoryOpaqueFDDescriptor' = new WGPUSType(327712)
  static 'SharedTextureMemoryZirconHandleDescriptor' = new WGPUSType(327713)
  static 'SharedTextureMemoryDXGISharedHandleDescriptor' = new WGPUSType(327714)
  static 'SharedTextureMemoryD3D11Texture2DDescriptor' = new WGPUSType(327715)
  static 'SharedTextureMemoryIOSurfaceDescriptor' = new WGPUSType(327716)
  static 'SharedTextureMemoryEGLImageDescriptor' = new WGPUSType(327717)
  static 'SharedTextureMemoryInitializedBeginState' = new WGPUSType(327718)
  static 'SharedTextureMemoryInitializedEndState' = new WGPUSType(327719)
  static 'SharedTextureMemoryVkImageLayoutBeginState' = new WGPUSType(327720)
  static 'SharedTextureMemoryVkImageLayoutEndState' = new WGPUSType(327721)
  static 'SharedTextureMemoryD3DSwapchainBeginState' = new WGPUSType(327722)
  static 'SharedFenceVkSemaphoreOpaqueFDDescriptor' = new WGPUSType(327723)
  static 'SharedFenceVkSemaphoreOpaqueFDExportInfo' = new WGPUSType(327724)
  static 'SharedFenceSyncFDDescriptor' = new WGPUSType(327725)
  static 'SharedFenceSyncFDExportInfo' = new WGPUSType(327726)
  static 'SharedFenceVkSemaphoreZirconHandleDescriptor' = new WGPUSType(327727)
  static 'SharedFenceVkSemaphoreZirconHandleExportInfo' = new WGPUSType(327728)
  static 'SharedFenceDXGISharedHandleDescriptor' = new WGPUSType(327729)
  static 'SharedFenceDXGISharedHandleExportInfo' = new WGPUSType(327730)
  static 'SharedFenceMTLSharedEventDescriptor' = new WGPUSType(327731)
  static 'SharedFenceMTLSharedEventExportInfo' = new WGPUSType(327732)
  static 'SharedBufferMemoryD3D12ResourceDescriptor' = new WGPUSType(327733)
  static 'StaticSamplerBindingLayout' = new WGPUSType(327734)
  static 'YCbCrVkDescriptor' = new WGPUSType(327735)
  static 'SharedTextureMemoryAHardwareBufferProperties' = new WGPUSType(327736)
  static 'AHardwareBufferProperties' = new WGPUSType(327737)
  static 'DawnExperimentalImmediateDataLimits' = new WGPUSType(327738)
  static 'DawnTexelCopyBufferRowAlignmentLimits' = new WGPUSType(327739)
  static 'Force32' = new WGPUSType(2147483647)
}
export class WGPUSamplerBindingType extends c.U32 {
  static 'BindingNotUsed' = new WGPUSamplerBindingType(0)
  static 'Filtering' = new WGPUSamplerBindingType(1)
  static 'NonFiltering' = new WGPUSamplerBindingType(2)
  static 'Comparison' = new WGPUSamplerBindingType(3)
  static 'Force32' = new WGPUSamplerBindingType(2147483647)
}
export class WGPUSharedFenceType extends c.U32 {
  static 'VkSemaphoreOpaqueFD' = new WGPUSharedFenceType(1)
  static 'SyncFD' = new WGPUSharedFenceType(2)
  static 'VkSemaphoreZirconHandle' = new WGPUSharedFenceType(3)
  static 'DXGISharedHandle' = new WGPUSharedFenceType(4)
  static 'MTLSharedEvent' = new WGPUSharedFenceType(5)
  static 'Force32' = new WGPUSharedFenceType(2147483647)
}
export class WGPUStatus extends c.U32 {
  static 'Success' = new WGPUStatus(1)
  static 'Error' = new WGPUStatus(2)
  static 'Force32' = new WGPUStatus(2147483647)
}
export class WGPUStencilOperation extends c.U32 {
  static 'Undefined' = new WGPUStencilOperation(0)
  static 'Keep' = new WGPUStencilOperation(1)
  static 'Zero' = new WGPUStencilOperation(2)
  static 'Replace' = new WGPUStencilOperation(3)
  static 'Invert' = new WGPUStencilOperation(4)
  static 'IncrementClamp' = new WGPUStencilOperation(5)
  static 'DecrementClamp' = new WGPUStencilOperation(6)
  static 'IncrementWrap' = new WGPUStencilOperation(7)
  static 'DecrementWrap' = new WGPUStencilOperation(8)
  static 'Force32' = new WGPUStencilOperation(2147483647)
}
export class WGPUStorageTextureAccess extends c.U32 {
  static 'BindingNotUsed' = new WGPUStorageTextureAccess(0)
  static 'WriteOnly' = new WGPUStorageTextureAccess(1)
  static 'ReadOnly' = new WGPUStorageTextureAccess(2)
  static 'ReadWrite' = new WGPUStorageTextureAccess(3)
  static 'Force32' = new WGPUStorageTextureAccess(2147483647)
}
export class WGPUStoreOp extends c.U32 {
  static 'Undefined' = new WGPUStoreOp(0)
  static 'Store' = new WGPUStoreOp(1)
  static 'Discard' = new WGPUStoreOp(2)
  static 'Force32' = new WGPUStoreOp(2147483647)
}
export class WGPUSurfaceGetCurrentTextureStatus extends c.U32 {
  static 'Success' = new WGPUSurfaceGetCurrentTextureStatus(1)
  static 'Timeout' = new WGPUSurfaceGetCurrentTextureStatus(2)
  static 'Outdated' = new WGPUSurfaceGetCurrentTextureStatus(3)
  static 'Lost' = new WGPUSurfaceGetCurrentTextureStatus(4)
  static 'OutOfMemory' = new WGPUSurfaceGetCurrentTextureStatus(5)
  static 'DeviceLost' = new WGPUSurfaceGetCurrentTextureStatus(6)
  static 'Error' = new WGPUSurfaceGetCurrentTextureStatus(7)
  static 'Force32' = new WGPUSurfaceGetCurrentTextureStatus(2147483647)
}
export class WGPUTextureAspect extends c.U32 {
  static 'Undefined' = new WGPUTextureAspect(0)
  static 'All' = new WGPUTextureAspect(1)
  static 'StencilOnly' = new WGPUTextureAspect(2)
  static 'DepthOnly' = new WGPUTextureAspect(3)
  static 'Plane0Only' = new WGPUTextureAspect(327680)
  static 'Plane1Only' = new WGPUTextureAspect(327681)
  static 'Plane2Only' = new WGPUTextureAspect(327682)
  static 'Force32' = new WGPUTextureAspect(2147483647)
}
export class WGPUTextureDimension extends c.U32 {
  static 'Undefined' = new WGPUTextureDimension(0)
  static '1D' = new WGPUTextureDimension(1)
  static '2D' = new WGPUTextureDimension(2)
  static '3D' = new WGPUTextureDimension(3)
  static 'Force32' = new WGPUTextureDimension(2147483647)
}
export class WGPUTextureFormat extends c.U32 {
  static 'Undefined' = new WGPUTextureFormat(0)
  static 'R8Unorm' = new WGPUTextureFormat(1)
  static 'R8Snorm' = new WGPUTextureFormat(2)
  static 'R8Uint' = new WGPUTextureFormat(3)
  static 'R8Sint' = new WGPUTextureFormat(4)
  static 'R16Uint' = new WGPUTextureFormat(5)
  static 'R16Sint' = new WGPUTextureFormat(6)
  static 'R16Float' = new WGPUTextureFormat(7)
  static 'RG8Unorm' = new WGPUTextureFormat(8)
  static 'RG8Snorm' = new WGPUTextureFormat(9)
  static 'RG8Uint' = new WGPUTextureFormat(10)
  static 'RG8Sint' = new WGPUTextureFormat(11)
  static 'R32Float' = new WGPUTextureFormat(12)
  static 'R32Uint' = new WGPUTextureFormat(13)
  static 'R32Sint' = new WGPUTextureFormat(14)
  static 'RG16Uint' = new WGPUTextureFormat(15)
  static 'RG16Sint' = new WGPUTextureFormat(16)
  static 'RG16Float' = new WGPUTextureFormat(17)
  static 'RGBA8Unorm' = new WGPUTextureFormat(18)
  static 'RGBA8UnormSrgb' = new WGPUTextureFormat(19)
  static 'RGBA8Snorm' = new WGPUTextureFormat(20)
  static 'RGBA8Uint' = new WGPUTextureFormat(21)
  static 'RGBA8Sint' = new WGPUTextureFormat(22)
  static 'BGRA8Unorm' = new WGPUTextureFormat(23)
  static 'BGRA8UnormSrgb' = new WGPUTextureFormat(24)
  static 'RGB10A2Uint' = new WGPUTextureFormat(25)
  static 'RGB10A2Unorm' = new WGPUTextureFormat(26)
  static 'RG11B10Ufloat' = new WGPUTextureFormat(27)
  static 'RGB9E5Ufloat' = new WGPUTextureFormat(28)
  static 'RG32Float' = new WGPUTextureFormat(29)
  static 'RG32Uint' = new WGPUTextureFormat(30)
  static 'RG32Sint' = new WGPUTextureFormat(31)
  static 'RGBA16Uint' = new WGPUTextureFormat(32)
  static 'RGBA16Sint' = new WGPUTextureFormat(33)
  static 'RGBA16Float' = new WGPUTextureFormat(34)
  static 'RGBA32Float' = new WGPUTextureFormat(35)
  static 'RGBA32Uint' = new WGPUTextureFormat(36)
  static 'RGBA32Sint' = new WGPUTextureFormat(37)
  static 'Stencil8' = new WGPUTextureFormat(38)
  static 'Depth16Unorm' = new WGPUTextureFormat(39)
  static 'Depth24Plus' = new WGPUTextureFormat(40)
  static 'Depth24PlusStencil8' = new WGPUTextureFormat(41)
  static 'Depth32Float' = new WGPUTextureFormat(42)
  static 'Depth32FloatStencil8' = new WGPUTextureFormat(43)
  static 'BC1RGBAUnorm' = new WGPUTextureFormat(44)
  static 'BC1RGBAUnormSrgb' = new WGPUTextureFormat(45)
  static 'BC2RGBAUnorm' = new WGPUTextureFormat(46)
  static 'BC2RGBAUnormSrgb' = new WGPUTextureFormat(47)
  static 'BC3RGBAUnorm' = new WGPUTextureFormat(48)
  static 'BC3RGBAUnormSrgb' = new WGPUTextureFormat(49)
  static 'BC4RUnorm' = new WGPUTextureFormat(50)
  static 'BC4RSnorm' = new WGPUTextureFormat(51)
  static 'BC5RGUnorm' = new WGPUTextureFormat(52)
  static 'BC5RGSnorm' = new WGPUTextureFormat(53)
  static 'BC6HRGBUfloat' = new WGPUTextureFormat(54)
  static 'BC6HRGBFloat' = new WGPUTextureFormat(55)
  static 'BC7RGBAUnorm' = new WGPUTextureFormat(56)
  static 'BC7RGBAUnormSrgb' = new WGPUTextureFormat(57)
  static 'ETC2RGB8Unorm' = new WGPUTextureFormat(58)
  static 'ETC2RGB8UnormSrgb' = new WGPUTextureFormat(59)
  static 'ETC2RGB8A1Unorm' = new WGPUTextureFormat(60)
  static 'ETC2RGB8A1UnormSrgb' = new WGPUTextureFormat(61)
  static 'ETC2RGBA8Unorm' = new WGPUTextureFormat(62)
  static 'ETC2RGBA8UnormSrgb' = new WGPUTextureFormat(63)
  static 'EACR11Unorm' = new WGPUTextureFormat(64)
  static 'EACR11Snorm' = new WGPUTextureFormat(65)
  static 'EACRG11Unorm' = new WGPUTextureFormat(66)
  static 'EACRG11Snorm' = new WGPUTextureFormat(67)
  static 'ASTC4x4Unorm' = new WGPUTextureFormat(68)
  static 'ASTC4x4UnormSrgb' = new WGPUTextureFormat(69)
  static 'ASTC5x4Unorm' = new WGPUTextureFormat(70)
  static 'ASTC5x4UnormSrgb' = new WGPUTextureFormat(71)
  static 'ASTC5x5Unorm' = new WGPUTextureFormat(72)
  static 'ASTC5x5UnormSrgb' = new WGPUTextureFormat(73)
  static 'ASTC6x5Unorm' = new WGPUTextureFormat(74)
  static 'ASTC6x5UnormSrgb' = new WGPUTextureFormat(75)
  static 'ASTC6x6Unorm' = new WGPUTextureFormat(76)
  static 'ASTC6x6UnormSrgb' = new WGPUTextureFormat(77)
  static 'ASTC8x5Unorm' = new WGPUTextureFormat(78)
  static 'ASTC8x5UnormSrgb' = new WGPUTextureFormat(79)
  static 'ASTC8x6Unorm' = new WGPUTextureFormat(80)
  static 'ASTC8x6UnormSrgb' = new WGPUTextureFormat(81)
  static 'ASTC8x8Unorm' = new WGPUTextureFormat(82)
  static 'ASTC8x8UnormSrgb' = new WGPUTextureFormat(83)
  static 'ASTC10x5Unorm' = new WGPUTextureFormat(84)
  static 'ASTC10x5UnormSrgb' = new WGPUTextureFormat(85)
  static 'ASTC10x6Unorm' = new WGPUTextureFormat(86)
  static 'ASTC10x6UnormSrgb' = new WGPUTextureFormat(87)
  static 'ASTC10x8Unorm' = new WGPUTextureFormat(88)
  static 'ASTC10x8UnormSrgb' = new WGPUTextureFormat(89)
  static 'ASTC10x10Unorm' = new WGPUTextureFormat(90)
  static 'ASTC10x10UnormSrgb' = new WGPUTextureFormat(91)
  static 'ASTC12x10Unorm' = new WGPUTextureFormat(92)
  static 'ASTC12x10UnormSrgb' = new WGPUTextureFormat(93)
  static 'ASTC12x12Unorm' = new WGPUTextureFormat(94)
  static 'ASTC12x12UnormSrgb' = new WGPUTextureFormat(95)
  static 'R16Unorm' = new WGPUTextureFormat(327680)
  static 'RG16Unorm' = new WGPUTextureFormat(327681)
  static 'RGBA16Unorm' = new WGPUTextureFormat(327682)
  static 'R16Snorm' = new WGPUTextureFormat(327683)
  static 'RG16Snorm' = new WGPUTextureFormat(327684)
  static 'RGBA16Snorm' = new WGPUTextureFormat(327685)
  static 'R8BG8Biplanar420Unorm' = new WGPUTextureFormat(327686)
  static 'R10X6BG10X6Biplanar420Unorm' = new WGPUTextureFormat(327687)
  static 'R8BG8A8Triplanar420Unorm' = new WGPUTextureFormat(327688)
  static 'R8BG8Biplanar422Unorm' = new WGPUTextureFormat(327689)
  static 'R8BG8Biplanar444Unorm' = new WGPUTextureFormat(327690)
  static 'R10X6BG10X6Biplanar422Unorm' = new WGPUTextureFormat(327691)
  static 'R10X6BG10X6Biplanar444Unorm' = new WGPUTextureFormat(327692)
  static 'External' = new WGPUTextureFormat(327693)
  static 'Force32' = new WGPUTextureFormat(2147483647)
}
export class WGPUTextureSampleType extends c.U32 {
  static 'BindingNotUsed' = new WGPUTextureSampleType(0)
  static 'Float' = new WGPUTextureSampleType(1)
  static 'UnfilterableFloat' = new WGPUTextureSampleType(2)
  static 'Depth' = new WGPUTextureSampleType(3)
  static 'Sint' = new WGPUTextureSampleType(4)
  static 'Uint' = new WGPUTextureSampleType(5)
  static 'Force32' = new WGPUTextureSampleType(2147483647)
}
export class WGPUTextureViewDimension extends c.U32 {
  static 'Undefined' = new WGPUTextureViewDimension(0)
  static '1D' = new WGPUTextureViewDimension(1)
  static '2D' = new WGPUTextureViewDimension(2)
  static '2DArray' = new WGPUTextureViewDimension(3)
  static 'Cube' = new WGPUTextureViewDimension(4)
  static 'CubeArray' = new WGPUTextureViewDimension(5)
  static '3D' = new WGPUTextureViewDimension(6)
  static 'Force32' = new WGPUTextureViewDimension(2147483647)
}
export class WGPUVertexFormat extends c.U32 {
  static 'Uint8' = new WGPUVertexFormat(1)
  static 'Uint8x2' = new WGPUVertexFormat(2)
  static 'Uint8x4' = new WGPUVertexFormat(3)
  static 'Sint8' = new WGPUVertexFormat(4)
  static 'Sint8x2' = new WGPUVertexFormat(5)
  static 'Sint8x4' = new WGPUVertexFormat(6)
  static 'Unorm8' = new WGPUVertexFormat(7)
  static 'Unorm8x2' = new WGPUVertexFormat(8)
  static 'Unorm8x4' = new WGPUVertexFormat(9)
  static 'Snorm8' = new WGPUVertexFormat(10)
  static 'Snorm8x2' = new WGPUVertexFormat(11)
  static 'Snorm8x4' = new WGPUVertexFormat(12)
  static 'Uint16' = new WGPUVertexFormat(13)
  static 'Uint16x2' = new WGPUVertexFormat(14)
  static 'Uint16x4' = new WGPUVertexFormat(15)
  static 'Sint16' = new WGPUVertexFormat(16)
  static 'Sint16x2' = new WGPUVertexFormat(17)
  static 'Sint16x4' = new WGPUVertexFormat(18)
  static 'Unorm16' = new WGPUVertexFormat(19)
  static 'Unorm16x2' = new WGPUVertexFormat(20)
  static 'Unorm16x4' = new WGPUVertexFormat(21)
  static 'Snorm16' = new WGPUVertexFormat(22)
  static 'Snorm16x2' = new WGPUVertexFormat(23)
  static 'Snorm16x4' = new WGPUVertexFormat(24)
  static 'Float16' = new WGPUVertexFormat(25)
  static 'Float16x2' = new WGPUVertexFormat(26)
  static 'Float16x4' = new WGPUVertexFormat(27)
  static 'Float32' = new WGPUVertexFormat(28)
  static 'Float32x2' = new WGPUVertexFormat(29)
  static 'Float32x3' = new WGPUVertexFormat(30)
  static 'Float32x4' = new WGPUVertexFormat(31)
  static 'Uint32' = new WGPUVertexFormat(32)
  static 'Uint32x2' = new WGPUVertexFormat(33)
  static 'Uint32x3' = new WGPUVertexFormat(34)
  static 'Uint32x4' = new WGPUVertexFormat(35)
  static 'Sint32' = new WGPUVertexFormat(36)
  static 'Sint32x2' = new WGPUVertexFormat(37)
  static 'Sint32x3' = new WGPUVertexFormat(38)
  static 'Sint32x4' = new WGPUVertexFormat(39)
  static 'Unorm10_10_10_2' = new WGPUVertexFormat(40)
  static 'Unorm8x4BGRA' = new WGPUVertexFormat(41)
  static 'Force32' = new WGPUVertexFormat(2147483647)
}
export class WGPUVertexStepMode extends c.U32 {
  static 'Undefined' = new WGPUVertexStepMode(0)
  static 'Vertex' = new WGPUVertexStepMode(1)
  static 'Instance' = new WGPUVertexStepMode(2)
  static 'Force32' = new WGPUVertexStepMode(2147483647)
}
export class WGPUWaitStatus extends c.U32 {
  static 'Success' = new WGPUWaitStatus(1)
  static 'TimedOut' = new WGPUWaitStatus(2)
  static 'UnsupportedTimeout' = new WGPUWaitStatus(3)
  static 'UnsupportedCount' = new WGPUWaitStatus(4)
  static 'UnsupportedMixedSources' = new WGPUWaitStatus(5)
  static 'Unknown' = new WGPUWaitStatus(6)
  static 'Force32' = new WGPUWaitStatus(2147483647)
}

// structs
export class WGPUAdapterImpl extends c.Struct<[]> {}
export class WGPUBindGroupImpl extends c.Struct<[]> {}
export class WGPUBindGroupLayoutImpl extends c.Struct<[]> {}
export class WGPUBufferImpl extends c.Struct<[]> {}
export class WGPUCommandBufferImpl extends c.Struct<[]> {}
export class WGPUCommandEncoderImpl extends c.Struct<[]> {}
export class WGPUComputePassEncoderImpl extends c.Struct<[]> {}
export class WGPUComputePipelineImpl extends c.Struct<[]> {}
export class WGPUDeviceImpl extends c.Struct<[]> {}
export class WGPUExternalTextureImpl extends c.Struct<[]> {}
export class WGPUInstanceImpl extends c.Struct<[]> {}
export class WGPUPipelineLayoutImpl extends c.Struct<[]> {}
export class WGPUQuerySetImpl extends c.Struct<[]> {}
export class WGPUQueueImpl extends c.Struct<[]> {}
export class WGPURenderBundleImpl extends c.Struct<[]> {}
export class WGPURenderBundleEncoderImpl extends c.Struct<[]> {}
export class WGPURenderPassEncoderImpl extends c.Struct<[]> {}
export class WGPURenderPipelineImpl extends c.Struct<[]> {}
export class WGPUSamplerImpl extends c.Struct<[]> {}
export class WGPUShaderModuleImpl extends c.Struct<[]> {}
export class WGPUSharedBufferMemoryImpl extends c.Struct<[]> {}
export class WGPUSharedFenceImpl extends c.Struct<[]> {}
export class WGPUSharedTextureMemoryImpl extends c.Struct<[]> {}
export class WGPUSurfaceImpl extends c.Struct<[]> {}
export class WGPUTextureImpl extends c.Struct<[]> {}
export class WGPUTextureViewImpl extends c.Struct<[]> {}
export class WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER extends c.Struct<[unused: WGPUBool]> {}
export class WGPUAdapterPropertiesD3D extends c.Struct<[chain: WGPUChainedStructOut, shaderModel: c.U32]> {}
export class WGPUAdapterPropertiesSubgroups extends c.Struct<[chain: WGPUChainedStructOut, subgroupMinSize: c.U32, subgroupMaxSize: c.U32]> {}
export class WGPUAdapterPropertiesVk extends c.Struct<[chain: WGPUChainedStructOut, driverVersion: c.U32]> {}
export class WGPUBindGroupEntry extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, binding: c.U32, buffer: WGPUBuffer, offset: c.U64, size: c.U64, sampler: WGPUSampler, textureView: WGPUTextureView]> {}
export class WGPUBlendComponent extends c.Struct<[operation: WGPUBlendOperation, srcFactor: WGPUBlendFactor, dstFactor: WGPUBlendFactor]> {}
export class WGPUBufferBindingLayout extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, type: WGPUBufferBindingType, hasDynamicOffset: WGPUBool, minBindingSize: c.U64]> {}
export class WGPUBufferHostMappedPointer extends c.Struct<[chain: WGPUChainedStruct, pointer: c.Pointer<c.Void>, disposeCallback: WGPUCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUBufferMapCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUBufferMapCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUColor extends c.Struct<[r: c.F64, g: c.F64, b: c.F64, a: c.F64]> {}
export class WGPUColorTargetStateExpandResolveTextureDawn extends c.Struct<[chain: WGPUChainedStruct, enabled: WGPUBool]> {}
export class WGPUCompilationInfoCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUCompilationInfoCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUComputePassTimestampWrites extends c.Struct<[querySet: WGPUQuerySet, beginningOfPassWriteIndex: c.U32, endOfPassWriteIndex: c.U32]> {}
export class WGPUCopyTextureForBrowserOptions extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, flipY: WGPUBool, needsColorSpaceConversion: WGPUBool, srcAlphaMode: WGPUAlphaMode, srcTransferFunctionParameters: c.Pointer<c.F32>, conversionMatrix: c.Pointer<c.F32>, dstTransferFunctionParameters: c.Pointer<c.F32>, dstAlphaMode: WGPUAlphaMode, internalUsage: WGPUBool]> {}
export class WGPUCreateComputePipelineAsyncCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUCreateComputePipelineAsyncCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUCreateRenderPipelineAsyncCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUCreateRenderPipelineAsyncCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUDawnWGSLBlocklist extends c.Struct<[chain: WGPUChainedStruct, blocklistedFeatureCount: c.Size, blocklistedFeatures: c.Pointer<c.Pointer<c.U8>>]> {}
export class WGPUDawnAdapterPropertiesPowerPreference extends c.Struct<[chain: WGPUChainedStructOut, powerPreference: WGPUPowerPreference]> {}
export class WGPUDawnBufferDescriptorErrorInfoFromWireClient extends c.Struct<[chain: WGPUChainedStruct, outOfMemory: WGPUBool]> {}
export class WGPUDawnEncoderInternalUsageDescriptor extends c.Struct<[chain: WGPUChainedStruct, useInternalUsages: WGPUBool]> {}
export class WGPUDawnExperimentalImmediateDataLimits extends c.Struct<[chain: WGPUChainedStructOut, maxImmediateDataRangeByteSize: c.U32]> {}
export class WGPUDawnExperimentalSubgroupLimits extends c.Struct<[chain: WGPUChainedStructOut, minSubgroupSize: c.U32, maxSubgroupSize: c.U32]> {}
export class WGPUDawnRenderPassColorAttachmentRenderToSingleSampled extends c.Struct<[chain: WGPUChainedStruct, implicitSampleCount: c.U32]> {}
export class WGPUDawnShaderModuleSPIRVOptionsDescriptor extends c.Struct<[chain: WGPUChainedStruct, allowNonUniformDerivatives: WGPUBool]> {}
export class WGPUDawnTexelCopyBufferRowAlignmentLimits extends c.Struct<[chain: WGPUChainedStructOut, minTexelCopyBufferRowAlignment: c.U32]> {}
export class WGPUDawnTextureInternalUsageDescriptor extends c.Struct<[chain: WGPUChainedStruct, internalUsage: WGPUTextureUsage]> {}
export class WGPUDawnTogglesDescriptor extends c.Struct<[chain: WGPUChainedStruct, enabledToggleCount: c.Size, enabledToggles: c.Pointer<c.Pointer<c.U8>>, disabledToggleCount: c.Size, disabledToggles: c.Pointer<c.Pointer<c.U8>>]> {}
export class WGPUDawnWireWGSLControl extends c.Struct<[chain: WGPUChainedStruct, enableExperimental: WGPUBool, enableUnsafe: WGPUBool, enableTesting: WGPUBool]> {}
export class WGPUDeviceLostCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUDeviceLostCallbackNew, userdata: c.Pointer<c.Void>]> {}
export class WGPUDrmFormatProperties extends c.Struct<[modifier: c.U64, modifierPlaneCount: c.U32]> {}
export class WGPUExtent2D extends c.Struct<[width: c.U32, height: c.U32]> {}
export class WGPUExtent3D extends c.Struct<[width: c.U32, height: c.U32, depthOrArrayLayers: c.U32]> {}
export class WGPUExternalTextureBindingEntry extends c.Struct<[chain: WGPUChainedStruct, externalTexture: WGPUExternalTexture]> {}
export class WGPUExternalTextureBindingLayout extends c.Struct<[chain: WGPUChainedStruct]> {}
export class WGPUFormatCapabilities extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>]> {}
export class WGPUFuture extends c.Struct<[id: c.U64]> {}
export class WGPUInstanceFeatures extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, timedWaitAnyEnable: WGPUBool, timedWaitAnyMaxCount: c.Size]> {}
export class WGPULimits extends c.Struct<[maxTextureDimension1D: c.U32, maxTextureDimension2D: c.U32, maxTextureDimension3D: c.U32, maxTextureArrayLayers: c.U32, maxBindGroups: c.U32, maxBindGroupsPlusVertexBuffers: c.U32, maxBindingsPerBindGroup: c.U32, maxDynamicUniformBuffersPerPipelineLayout: c.U32, maxDynamicStorageBuffersPerPipelineLayout: c.U32, maxSampledTexturesPerShaderStage: c.U32, maxSamplersPerShaderStage: c.U32, maxStorageBuffersPerShaderStage: c.U32, maxStorageTexturesPerShaderStage: c.U32, maxUniformBuffersPerShaderStage: c.U32, maxUniformBufferBindingSize: c.U64, maxStorageBufferBindingSize: c.U64, minUniformBufferOffsetAlignment: c.U32, minStorageBufferOffsetAlignment: c.U32, maxVertexBuffers: c.U32, maxBufferSize: c.U64, maxVertexAttributes: c.U32, maxVertexBufferArrayStride: c.U32, maxInterStageShaderComponents: c.U32, maxInterStageShaderVariables: c.U32, maxColorAttachments: c.U32, maxColorAttachmentBytesPerSample: c.U32, maxComputeWorkgroupStorageSize: c.U32, maxComputeInvocationsPerWorkgroup: c.U32, maxComputeWorkgroupSizeX: c.U32, maxComputeWorkgroupSizeY: c.U32, maxComputeWorkgroupSizeZ: c.U32, maxComputeWorkgroupsPerDimension: c.U32, maxStorageBuffersInVertexStage: c.U32, maxStorageTexturesInVertexStage: c.U32, maxStorageBuffersInFragmentStage: c.U32, maxStorageTexturesInFragmentStage: c.U32]> {}
export class WGPUMemoryHeapInfo extends c.Struct<[properties: WGPUHeapProperty, size: c.U64]> {}
export class WGPUMultisampleState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, count: c.U32, mask: c.U32, alphaToCoverageEnabled: WGPUBool]> {}
export class WGPUOrigin2D extends c.Struct<[x: c.U32, y: c.U32]> {}
export class WGPUOrigin3D extends c.Struct<[x: c.U32, y: c.U32, z: c.U32]> {}
export class WGPUPipelineLayoutStorageAttachment extends c.Struct<[offset: c.U64, format: WGPUTextureFormat]> {}
export class WGPUPopErrorScopeCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUPopErrorScopeCallback, oldCallback: WGPUErrorCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUPrimitiveState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, topology: WGPUPrimitiveTopology, stripIndexFormat: WGPUIndexFormat, frontFace: WGPUFrontFace, cullMode: WGPUCullMode, unclippedDepth: WGPUBool]> {}
export class WGPUQueueWorkDoneCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUQueueWorkDoneCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPURenderPassDepthStencilAttachment extends c.Struct<[view: WGPUTextureView, depthLoadOp: WGPULoadOp, depthStoreOp: WGPUStoreOp, depthClearValue: c.F32, depthReadOnly: WGPUBool, stencilLoadOp: WGPULoadOp, stencilStoreOp: WGPUStoreOp, stencilClearValue: c.U32, stencilReadOnly: WGPUBool]> {}
export class WGPURenderPassDescriptorExpandResolveRect extends c.Struct<[chain: WGPUChainedStruct, x: c.U32, y: c.U32, width: c.U32, height: c.U32]> {}
export class WGPURenderPassMaxDrawCount extends c.Struct<[chain: WGPUChainedStruct, maxDrawCount: c.U64]> {}
export class WGPURenderPassTimestampWrites extends c.Struct<[querySet: WGPUQuerySet, beginningOfPassWriteIndex: c.U32, endOfPassWriteIndex: c.U32]> {}
export class WGPURequestAdapterCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPURequestAdapterCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPURequestAdapterOptions extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, compatibleSurface: WGPUSurface, featureLevel: WGPUFeatureLevel, powerPreference: WGPUPowerPreference, backendType: WGPUBackendType, forceFallbackAdapter: WGPUBool, compatibilityMode: WGPUBool]> {}
export class WGPURequestDeviceCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPURequestDeviceCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUSamplerBindingLayout extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, type: WGPUSamplerBindingType]> {}
export class WGPUShaderModuleCompilationOptions extends c.Struct<[chain: WGPUChainedStruct, strictMath: WGPUBool]> {}
export class WGPUShaderSourceSPIRV extends c.Struct<[chain: WGPUChainedStruct, codeSize: c.U32, code: c.Pointer<c.U32>]> {}
export class WGPUSharedBufferMemoryBeginAccessDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, initialized: WGPUBool, fenceCount: c.Size, fences: c.Pointer<WGPUSharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class WGPUSharedBufferMemoryEndAccessState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, initialized: WGPUBool, fenceCount: c.Size, fences: c.Pointer<WGPUSharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class WGPUSharedBufferMemoryProperties extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, usage: WGPUBufferUsage, size: c.U64]> {}
export class WGPUSharedFenceDXGISharedHandleDescriptor extends c.Struct<[chain: WGPUChainedStruct, handle: c.Pointer<c.Void>]> {}
export class WGPUSharedFenceDXGISharedHandleExportInfo extends c.Struct<[chain: WGPUChainedStructOut, handle: c.Pointer<c.Void>]> {}
export class WGPUSharedFenceMTLSharedEventDescriptor extends c.Struct<[chain: WGPUChainedStruct, sharedEvent: c.Pointer<c.Void>]> {}
export class WGPUSharedFenceMTLSharedEventExportInfo extends c.Struct<[chain: WGPUChainedStructOut, sharedEvent: c.Pointer<c.Void>]> {}
export class WGPUSharedFenceExportInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, type: WGPUSharedFenceType]> {}
export class WGPUSharedFenceSyncFDDescriptor extends c.Struct<[chain: WGPUChainedStruct, handle: c.I32]> {}
export class WGPUSharedFenceSyncFDExportInfo extends c.Struct<[chain: WGPUChainedStructOut, handle: c.I32]> {}
export class WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor extends c.Struct<[chain: WGPUChainedStruct, handle: c.I32]> {}
export class WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo extends c.Struct<[chain: WGPUChainedStructOut, handle: c.I32]> {}
export class WGPUSharedFenceVkSemaphoreZirconHandleDescriptor extends c.Struct<[chain: WGPUChainedStruct, handle: c.U32]> {}
export class WGPUSharedFenceVkSemaphoreZirconHandleExportInfo extends c.Struct<[chain: WGPUChainedStructOut, handle: c.U32]> {}
export class WGPUSharedTextureMemoryD3DSwapchainBeginState extends c.Struct<[chain: WGPUChainedStruct, isSwapchain: WGPUBool]> {}
export class WGPUSharedTextureMemoryDXGISharedHandleDescriptor extends c.Struct<[chain: WGPUChainedStruct, handle: c.Pointer<c.Void>, useKeyedMutex: WGPUBool]> {}
export class WGPUSharedTextureMemoryEGLImageDescriptor extends c.Struct<[chain: WGPUChainedStruct, image: c.Pointer<c.Void>]> {}
export class WGPUSharedTextureMemoryIOSurfaceDescriptor extends c.Struct<[chain: WGPUChainedStruct, ioSurface: c.Pointer<c.Void>]> {}
export class WGPUSharedTextureMemoryAHardwareBufferDescriptor extends c.Struct<[chain: WGPUChainedStruct, handle: c.Pointer<c.Void>, useExternalFormat: WGPUBool]> {}
export class WGPUSharedTextureMemoryBeginAccessDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, concurrentRead: WGPUBool, initialized: WGPUBool, fenceCount: c.Size, fences: c.Pointer<WGPUSharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class WGPUSharedTextureMemoryDmaBufPlane extends c.Struct<[fd: c.I32, offset: c.U64, stride: c.U32]> {}
export class WGPUSharedTextureMemoryEndAccessState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, initialized: WGPUBool, fenceCount: c.Size, fences: c.Pointer<WGPUSharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class WGPUSharedTextureMemoryOpaqueFDDescriptor extends c.Struct<[chain: WGPUChainedStruct, vkImageCreateInfo: c.Pointer<c.Void>, memoryFD: c.I32, memoryTypeIndex: c.U32, allocationSize: c.U64, dedicatedAllocation: WGPUBool]> {}
export class WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor extends c.Struct<[chain: WGPUChainedStruct, dedicatedAllocation: WGPUBool]> {}
export class WGPUSharedTextureMemoryVkImageLayoutBeginState extends c.Struct<[chain: WGPUChainedStruct, oldLayout: c.I32, newLayout: c.I32]> {}
export class WGPUSharedTextureMemoryVkImageLayoutEndState extends c.Struct<[chain: WGPUChainedStructOut, oldLayout: c.I32, newLayout: c.I32]> {}
export class WGPUSharedTextureMemoryZirconHandleDescriptor extends c.Struct<[chain: WGPUChainedStruct, memoryFD: c.U32, allocationSize: c.U64]> {}
export class WGPUStaticSamplerBindingLayout extends c.Struct<[chain: WGPUChainedStruct, sampler: WGPUSampler, sampledTextureBinding: c.U32]> {}
export class WGPUStencilFaceState extends c.Struct<[compare: WGPUCompareFunction, failOp: WGPUStencilOperation, depthFailOp: WGPUStencilOperation, passOp: WGPUStencilOperation]> {}
export class WGPUStorageTextureBindingLayout extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, access: WGPUStorageTextureAccess, format: WGPUTextureFormat, viewDimension: WGPUTextureViewDimension]> {}
export class WGPUStringView extends c.Struct<[data: c.Pointer<c.U8>, length: c.Size]> {}
export class WGPUSupportedFeatures extends c.Struct<[featureCount: c.Size, features: c.Pointer<WGPUFeatureName>]> {}
export class WGPUSurfaceCapabilities extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, usages: WGPUTextureUsage, formatCount: c.Size, formats: c.Pointer<WGPUTextureFormat>, presentModeCount: c.Size, presentModes: c.Pointer<WGPUPresentMode>, alphaModeCount: c.Size, alphaModes: c.Pointer<WGPUCompositeAlphaMode>]> {}
export class WGPUSurfaceConfiguration extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, device: WGPUDevice, format: WGPUTextureFormat, usage: WGPUTextureUsage, viewFormatCount: c.Size, viewFormats: c.Pointer<WGPUTextureFormat>, alphaMode: WGPUCompositeAlphaMode, width: c.U32, height: c.U32, presentMode: WGPUPresentMode]> {}
export class WGPUSurfaceDescriptorFromWindowsCoreWindow extends c.Struct<[chain: WGPUChainedStruct, coreWindow: c.Pointer<c.Void>]> {}
export class WGPUSurfaceDescriptorFromWindowsSwapChainPanel extends c.Struct<[chain: WGPUChainedStruct, swapChainPanel: c.Pointer<c.Void>]> {}
export class WGPUSurfaceSourceXCBWindow extends c.Struct<[chain: WGPUChainedStruct, connection: c.Pointer<c.Void>, window: c.U32]> {}
export class WGPUSurfaceSourceAndroidNativeWindow extends c.Struct<[chain: WGPUChainedStruct, window: c.Pointer<c.Void>]> {}
export class WGPUSurfaceSourceMetalLayer extends c.Struct<[chain: WGPUChainedStruct, layer: c.Pointer<c.Void>]> {}
export class WGPUSurfaceSourceWaylandSurface extends c.Struct<[chain: WGPUChainedStruct, display: c.Pointer<c.Void>, surface: c.Pointer<c.Void>]> {}
export class WGPUSurfaceSourceWindowsHWND extends c.Struct<[chain: WGPUChainedStruct, hinstance: c.Pointer<c.Void>, hwnd: c.Pointer<c.Void>]> {}
export class WGPUSurfaceSourceXlibWindow extends c.Struct<[chain: WGPUChainedStruct, display: c.Pointer<c.Void>, window: c.U64]> {}
export class WGPUSurfaceTexture extends c.Struct<[texture: WGPUTexture, suboptimal: WGPUBool, status: WGPUSurfaceGetCurrentTextureStatus]> {}
export class WGPUTextureBindingLayout extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, sampleType: WGPUTextureSampleType, viewDimension: WGPUTextureViewDimension, multisampled: WGPUBool]> {}
export class WGPUTextureBindingViewDimensionDescriptor extends c.Struct<[chain: WGPUChainedStruct, textureBindingViewDimension: WGPUTextureViewDimension]> {}
export class WGPUTextureDataLayout extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, offset: c.U64, bytesPerRow: c.U32, rowsPerImage: c.U32]> {}
export class WGPUUncapturedErrorCallbackInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, callback: WGPUErrorCallback, userdata: c.Pointer<c.Void>]> {}
export class WGPUVertexAttribute extends c.Struct<[format: WGPUVertexFormat, offset: c.U64, shaderLocation: c.U32]> {}
export class WGPUYCbCrVkDescriptor extends c.Struct<[chain: WGPUChainedStruct, vkFormat: c.U32, vkYCbCrModel: c.U32, vkYCbCrRange: c.U32, vkComponentSwizzleRed: c.U32, vkComponentSwizzleGreen: c.U32, vkComponentSwizzleBlue: c.U32, vkComponentSwizzleAlpha: c.U32, vkXChromaOffset: c.U32, vkYChromaOffset: c.U32, vkChromaFilter: WGPUFilterMode, forceExplicitReconstruction: WGPUBool, externalFormat: c.U64]> {}
export class WGPUAHardwareBufferProperties extends c.Struct<[yCbCrInfo: WGPUYCbCrVkDescriptor]> {}
export class WGPUAdapterInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, vendor: WGPUStringView, architecture: WGPUStringView, device: WGPUStringView, description: WGPUStringView, backendType: WGPUBackendType, adapterType: WGPUAdapterType, vendorID: c.U32, deviceID: c.U32, compatibilityMode: WGPUBool]> {}
export class WGPUAdapterPropertiesMemoryHeaps extends c.Struct<[chain: WGPUChainedStructOut, heapCount: c.Size, heapInfo: c.Pointer<WGPUMemoryHeapInfo>]> {}
export class WGPUBindGroupDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, layout: WGPUBindGroupLayout, entryCount: c.Size, entries: c.Pointer<WGPUBindGroupEntry>]> {}
export class WGPUBindGroupLayoutEntry extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, binding: c.U32, visibility: WGPUShaderStage, buffer: WGPUBufferBindingLayout, sampler: WGPUSamplerBindingLayout, texture: WGPUTextureBindingLayout, storageTexture: WGPUStorageTextureBindingLayout]> {}
export class WGPUBlendState extends c.Struct<[color: WGPUBlendComponent, alpha: WGPUBlendComponent]> {}
export class WGPUBufferDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, usage: WGPUBufferUsage, size: c.U64, mappedAtCreation: WGPUBool]> {}
export class WGPUCommandBufferDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUCommandEncoderDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUCompilationMessage extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, message: WGPUStringView, type: WGPUCompilationMessageType, lineNum: c.U64, linePos: c.U64, offset: c.U64, length: c.U64, utf16LinePos: c.U64, utf16Offset: c.U64, utf16Length: c.U64]> {}
export class WGPUComputePassDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, timestampWrites: c.Pointer<WGPUComputePassTimestampWrites>]> {}
export class WGPUConstantEntry extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, key: WGPUStringView, value: c.F64]> {}
export class WGPUDawnCacheDeviceDescriptor extends c.Struct<[chain: WGPUChainedStruct, isolationKey: WGPUStringView, loadDataFunction: WGPUDawnLoadCacheDataFunction, storeDataFunction: WGPUDawnStoreCacheDataFunction, functionUserdata: c.Pointer<c.Void>]> {}
export class WGPUDepthStencilState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, format: WGPUTextureFormat, depthWriteEnabled: WGPUOptionalBool, depthCompare: WGPUCompareFunction, stencilFront: WGPUStencilFaceState, stencilBack: WGPUStencilFaceState, stencilReadMask: c.U32, stencilWriteMask: c.U32, depthBias: c.I32, depthBiasSlopeScale: c.F32, depthBiasClamp: c.F32]> {}
export class WGPUDrmFormatCapabilities extends c.Struct<[chain: WGPUChainedStructOut, propertiesCount: c.Size, properties: c.Pointer<WGPUDrmFormatProperties>]> {}
export class WGPUExternalTextureDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, plane0: WGPUTextureView, plane1: WGPUTextureView, cropOrigin: WGPUOrigin2D, cropSize: WGPUExtent2D, apparentSize: WGPUExtent2D, doYuvToRgbConversionOnly: WGPUBool, yuvToRgbConversionMatrix: c.Pointer<c.F32>, srcTransferFunctionParameters: c.Pointer<c.F32>, dstTransferFunctionParameters: c.Pointer<c.F32>, gamutConversionMatrix: c.Pointer<c.F32>, mirrored: WGPUBool, rotation: WGPUExternalTextureRotation]> {}
export class WGPUFutureWaitInfo extends c.Struct<[future: WGPUFuture, completed: WGPUBool]> {}
export class WGPUImageCopyBuffer extends c.Struct<[layout: WGPUTextureDataLayout, buffer: WGPUBuffer]> {}
export class WGPUImageCopyExternalTexture extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, externalTexture: WGPUExternalTexture, origin: WGPUOrigin3D, naturalSize: WGPUExtent2D]> {}
export class WGPUImageCopyTexture extends c.Struct<[texture: WGPUTexture, mipLevel: c.U32, origin: WGPUOrigin3D, aspect: WGPUTextureAspect]> {}
export class WGPUInstanceDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, features: WGPUInstanceFeatures]> {}
export class WGPUPipelineLayoutDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, bindGroupLayoutCount: c.Size, bindGroupLayouts: c.Pointer<WGPUBindGroupLayout>, immediateDataRangeByteSize: c.U32]> {}
export class WGPUPipelineLayoutPixelLocalStorage extends c.Struct<[chain: WGPUChainedStruct, totalPixelLocalStorageSize: c.U64, storageAttachmentCount: c.Size, storageAttachments: c.Pointer<WGPUPipelineLayoutStorageAttachment>]> {}
export class WGPUQuerySetDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, type: WGPUQueryType, count: c.U32]> {}
export class WGPUQueueDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPURenderBundleDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPURenderBundleEncoderDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, colorFormatCount: c.Size, colorFormats: c.Pointer<WGPUTextureFormat>, depthStencilFormat: WGPUTextureFormat, sampleCount: c.U32, depthReadOnly: WGPUBool, stencilReadOnly: WGPUBool]> {}
export class WGPURenderPassColorAttachment extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, view: WGPUTextureView, depthSlice: c.U32, resolveTarget: WGPUTextureView, loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearValue: WGPUColor]> {}
export class WGPURenderPassStorageAttachment extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, offset: c.U64, storage: WGPUTextureView, loadOp: WGPULoadOp, storeOp: WGPUStoreOp, clearValue: WGPUColor]> {}
export class WGPURequiredLimits extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, limits: WGPULimits]> {}
export class WGPUSamplerDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, addressModeU: WGPUAddressMode, addressModeV: WGPUAddressMode, addressModeW: WGPUAddressMode, magFilter: WGPUFilterMode, minFilter: WGPUFilterMode, mipmapFilter: WGPUMipmapFilterMode, lodMinClamp: c.F32, lodMaxClamp: c.F32, compare: WGPUCompareFunction, maxAnisotropy: c.U16]> {}
export class WGPUShaderModuleDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUShaderSourceWGSL extends c.Struct<[chain: WGPUChainedStruct, code: WGPUStringView]> {}
export class WGPUSharedBufferMemoryDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUSharedFenceDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUSharedTextureMemoryAHardwareBufferProperties extends c.Struct<[chain: WGPUChainedStructOut, yCbCrInfo: WGPUYCbCrVkDescriptor]> {}
export class WGPUSharedTextureMemoryDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUSharedTextureMemoryDmaBufDescriptor extends c.Struct<[chain: WGPUChainedStruct, size: WGPUExtent3D, drmFormat: c.U32, drmModifier: c.U64, planeCount: c.Size, planes: c.Pointer<WGPUSharedTextureMemoryDmaBufPlane>]> {}
export class WGPUSharedTextureMemoryProperties extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, usage: WGPUTextureUsage, size: WGPUExtent3D, format: WGPUTextureFormat]> {}
export class WGPUSupportedLimits extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStructOut>, limits: WGPULimits]> {}
export class WGPUSurfaceDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView]> {}
export class WGPUSurfaceSourceCanvasHTMLSelector_Emscripten extends c.Struct<[chain: WGPUChainedStruct, selector: WGPUStringView]> {}
export class WGPUTextureDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, usage: WGPUTextureUsage, dimension: WGPUTextureDimension, size: WGPUExtent3D, format: WGPUTextureFormat, mipLevelCount: c.U32, sampleCount: c.U32, viewFormatCount: c.Size, viewFormats: c.Pointer<WGPUTextureFormat>]> {}
export class WGPUTextureViewDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, format: WGPUTextureFormat, dimension: WGPUTextureViewDimension, baseMipLevel: c.U32, mipLevelCount: c.U32, baseArrayLayer: c.U32, arrayLayerCount: c.U32, aspect: WGPUTextureAspect, usage: WGPUTextureUsage]> {}
export class WGPUVertexBufferLayout extends c.Struct<[arrayStride: c.U64, stepMode: WGPUVertexStepMode, attributeCount: c.Size, attributes: c.Pointer<WGPUVertexAttribute>]> {}
export class WGPUBindGroupLayoutDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, entryCount: c.Size, entries: c.Pointer<WGPUBindGroupLayoutEntry>]> {}
export class WGPUColorTargetState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, format: WGPUTextureFormat, blend: c.Pointer<WGPUBlendState>, writeMask: WGPUColorWriteMask]> {}
export class WGPUCompilationInfo extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, messageCount: c.Size, messages: c.Pointer<WGPUCompilationMessage>]> {}
export class WGPUComputeState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, module: WGPUShaderModule, entryPoint: WGPUStringView, constantCount: c.Size, constants: c.Pointer<WGPUConstantEntry>]> {}
export class WGPUDeviceDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, requiredFeatureCount: c.Size, requiredFeatures: c.Pointer<WGPUFeatureName>, requiredLimits: c.Pointer<WGPURequiredLimits>, defaultQueue: WGPUQueueDescriptor, deviceLostCallbackInfo2: WGPUDeviceLostCallbackInfo2, uncapturedErrorCallbackInfo2: WGPUUncapturedErrorCallbackInfo2]> {}
export class WGPURenderPassDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, colorAttachmentCount: c.Size, colorAttachments: c.Pointer<WGPURenderPassColorAttachment>, depthStencilAttachment: c.Pointer<WGPURenderPassDepthStencilAttachment>, occlusionQuerySet: WGPUQuerySet, timestampWrites: c.Pointer<WGPURenderPassTimestampWrites>]> {}
export class WGPURenderPassPixelLocalStorage extends c.Struct<[chain: WGPUChainedStruct, totalPixelLocalStorageSize: c.U64, storageAttachmentCount: c.Size, storageAttachments: c.Pointer<WGPURenderPassStorageAttachment>]> {}
export class WGPUVertexState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, module: WGPUShaderModule, entryPoint: WGPUStringView, constantCount: c.Size, constants: c.Pointer<WGPUConstantEntry>, bufferCount: c.Size, buffers: c.Pointer<WGPUVertexBufferLayout>]> {}
export class WGPUComputePipelineDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, layout: WGPUPipelineLayout, compute: WGPUComputeState]> {}
export class WGPUFragmentState extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, module: WGPUShaderModule, entryPoint: WGPUStringView, constantCount: c.Size, constants: c.Pointer<WGPUConstantEntry>, targetCount: c.Size, targets: c.Pointer<WGPUColorTargetState>]> {}
export class WGPURenderPipelineDescriptor extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, label: WGPUStringView, layout: WGPUPipelineLayout, vertex: WGPUVertexState, primitive: WGPUPrimitiveState, depthStencil: c.Pointer<WGPUDepthStencilState>, multisample: WGPUMultisampleState, fragment: c.Pointer<WGPUFragmentState>]> {}
export class WGPUChainedStruct extends c.Struct<[next: c.Pointer<WGPUChainedStruct>, sType: WGPUSType]> {}
export class WGPUChainedStructOut extends c.Struct<[next: c.Pointer<WGPUChainedStructOut>, sType: WGPUSType]> {}
export class WGPUBufferMapCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUBufferMapCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUCompilationInfoCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUCompilationInfoCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUCreateComputePipelineAsyncCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUCreateComputePipelineAsyncCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUCreateRenderPipelineAsyncCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUCreateRenderPipelineAsyncCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUDeviceLostCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUDeviceLostCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUPopErrorScopeCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUPopErrorScopeCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUQueueWorkDoneCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPUQueueWorkDoneCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPURequestAdapterCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPURequestAdapterCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPURequestDeviceCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, mode: WGPUCallbackMode, callback: WGPURequestDeviceCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class WGPUUncapturedErrorCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<WGPUChainedStruct>, callback: WGPUUncapturedErrorCallback, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}

// types
export class WGPUFlags extends c.U64 {}
export class WGPUBool extends c.U32 {}
export class WGPUAdapter extends c.Pointer<WGPUAdapterImpl> {}
export class WGPUBindGroup extends c.Pointer<WGPUBindGroupImpl> {}
export class WGPUBindGroupLayout extends c.Pointer<WGPUBindGroupLayoutImpl> {}
export class WGPUBuffer extends c.Pointer<WGPUBufferImpl> {}
export class WGPUCommandBuffer extends c.Pointer<WGPUCommandBufferImpl> {}
export class WGPUCommandEncoder extends c.Pointer<WGPUCommandEncoderImpl> {}
export class WGPUComputePassEncoder extends c.Pointer<WGPUComputePassEncoderImpl> {}
export class WGPUComputePipeline extends c.Pointer<WGPUComputePipelineImpl> {}
export class WGPUDevice extends c.Pointer<WGPUDeviceImpl> {}
export class WGPUExternalTexture extends c.Pointer<WGPUExternalTextureImpl> {}
export class WGPUInstance extends c.Pointer<WGPUInstanceImpl> {}
export class WGPUPipelineLayout extends c.Pointer<WGPUPipelineLayoutImpl> {}
export class WGPUQuerySet extends c.Pointer<WGPUQuerySetImpl> {}
export class WGPUQueue extends c.Pointer<WGPUQueueImpl> {}
export class WGPURenderBundle extends c.Pointer<WGPURenderBundleImpl> {}
export class WGPURenderBundleEncoder extends c.Pointer<WGPURenderBundleEncoderImpl> {}
export class WGPURenderPassEncoder extends c.Pointer<WGPURenderPassEncoderImpl> {}
export class WGPURenderPipeline extends c.Pointer<WGPURenderPipelineImpl> {}
export class WGPUSampler extends c.Pointer<WGPUSamplerImpl> {}
export class WGPUShaderModule extends c.Pointer<WGPUShaderModuleImpl> {}
export class WGPUSharedBufferMemory extends c.Pointer<WGPUSharedBufferMemoryImpl> {}
export class WGPUSharedFence extends c.Pointer<WGPUSharedFenceImpl> {}
export class WGPUSharedTextureMemory extends c.Pointer<WGPUSharedTextureMemoryImpl> {}
export class WGPUSurface extends c.Pointer<WGPUSurfaceImpl> {}
export class WGPUTexture extends c.Pointer<WGPUTextureImpl> {}
export class WGPUTextureView extends c.Pointer<WGPUTextureViewImpl> {}
export class WGPUBufferUsage extends WGPUFlags {}
export class WGPUColorWriteMask extends WGPUFlags {}
export class WGPUHeapProperty extends WGPUFlags {}
export class WGPUMapMode extends WGPUFlags {}
export class WGPUShaderStage extends WGPUFlags {}
export class WGPUTextureUsage extends WGPUFlags {}
export class WGPUBufferMapCallback extends c.Function {}
export class WGPUCallback extends c.Function {}
export class WGPUCompilationInfoCallback extends c.Function {}
export class WGPUCreateComputePipelineAsyncCallback extends c.Function {}
export class WGPUCreateRenderPipelineAsyncCallback extends c.Function {}
export class WGPUDawnLoadCacheDataFunction extends c.Function {}
export class WGPUDawnStoreCacheDataFunction extends c.Function {}
export class WGPUDeviceLostCallback extends c.Function {}
export class WGPUDeviceLostCallbackNew extends c.Function {}
export class WGPUErrorCallback extends c.Function {}
export class WGPULoggingCallback extends c.Function {}
export class WGPUPopErrorScopeCallback extends c.Function {}
export class WGPUProc extends c.Function {}
export class WGPUQueueWorkDoneCallback extends c.Function {}
export class WGPURequestAdapterCallback extends c.Function {}
export class WGPURequestDeviceCallback extends c.Function {}
export class WGPUBufferMapCallback2 extends c.Function {}
export class WGPUCompilationInfoCallback2 extends c.Function {}
export class WGPUCreateComputePipelineAsyncCallback2 extends c.Function {}
export class WGPUCreateRenderPipelineAsyncCallback2 extends c.Function {}
export class WGPUDeviceLostCallback2 extends c.Function {}
export class WGPUPopErrorScopeCallback2 extends c.Function {}
export class WGPUQueueWorkDoneCallback2 extends c.Function {}
export class WGPURequestAdapterCallback2 extends c.Function {}
export class WGPURequestDeviceCallback2 extends c.Function {}
export class WGPUUncapturedErrorCallback extends c.Function {}
export class WGPURenderPassDescriptorMaxDrawCount extends WGPURenderPassMaxDrawCount {}
export class WGPUShaderModuleSPIRVDescriptor extends WGPUShaderSourceSPIRV {}
export class WGPUShaderModuleWGSLDescriptor extends WGPUShaderSourceWGSL {}
export class WGPUSurfaceDescriptorFromAndroidNativeWindow extends WGPUSurfaceSourceAndroidNativeWindow {}
export class WGPUSurfaceDescriptorFromCanvasHTMLSelector extends WGPUSurfaceSourceCanvasHTMLSelector_Emscripten {}
export class WGPUSurfaceDescriptorFromMetalLayer extends WGPUSurfaceSourceMetalLayer {}
export class WGPUSurfaceDescriptorFromWaylandSurface extends WGPUSurfaceSourceWaylandSurface {}
export class WGPUSurfaceDescriptorFromWindowsHWND extends WGPUSurfaceSourceWindowsHWND {}
export class WGPUSurfaceDescriptorFromXcbWindow extends WGPUSurfaceSourceXCBWindow {}
export class WGPUSurfaceDescriptorFromXlibWindow extends WGPUSurfaceSourceXlibWindow {}
export class WGPUProcAdapterInfoFreeMembers extends c.Function {}
export class WGPUProcAdapterPropertiesMemoryHeapsFreeMembers extends c.Function {}
export class WGPUProcCreateInstance extends c.Function {}
export class WGPUProcDrmFormatCapabilitiesFreeMembers extends c.Function {}
export class WGPUProcGetInstanceFeatures extends c.Function {}
export class WGPUProcGetProcAddress extends c.Function {}
export class WGPUProcSharedBufferMemoryEndAccessStateFreeMembers extends c.Function {}
export class WGPUProcSharedTextureMemoryEndAccessStateFreeMembers extends c.Function {}
export class WGPUProcSupportedFeaturesFreeMembers extends c.Function {}
export class WGPUProcSurfaceCapabilitiesFreeMembers extends c.Function {}
export class WGPUProcAdapterCreateDevice extends c.Function {}
export class WGPUProcAdapterGetFeatures extends c.Function {}
export class WGPUProcAdapterGetFormatCapabilities extends c.Function {}
export class WGPUProcAdapterGetInfo extends c.Function {}
export class WGPUProcAdapterGetInstance extends c.Function {}
export class WGPUProcAdapterGetLimits extends c.Function {}
export class WGPUProcAdapterHasFeature extends c.Function {}
export class WGPUProcAdapterRequestDevice extends c.Function {}
export class WGPUProcAdapterRequestDevice2 extends c.Function {}
export class WGPUProcAdapterRequestDeviceF extends c.Function {}
export class WGPUProcAdapterAddRef extends c.Function {}
export class WGPUProcAdapterRelease extends c.Function {}
export class WGPUProcBindGroupSetLabel extends c.Function {}
export class WGPUProcBindGroupAddRef extends c.Function {}
export class WGPUProcBindGroupRelease extends c.Function {}
export class WGPUProcBindGroupLayoutSetLabel extends c.Function {}
export class WGPUProcBindGroupLayoutAddRef extends c.Function {}
export class WGPUProcBindGroupLayoutRelease extends c.Function {}
export class WGPUProcBufferDestroy extends c.Function {}
export class WGPUProcBufferGetConstMappedRange extends c.Function {}
export class WGPUProcBufferGetMapState extends c.Function {}
export class WGPUProcBufferGetMappedRange extends c.Function {}
export class WGPUProcBufferGetSize extends c.Function {}
export class WGPUProcBufferGetUsage extends c.Function {}
export class WGPUProcBufferMapAsync extends c.Function {}
export class WGPUProcBufferMapAsync2 extends c.Function {}
export class WGPUProcBufferMapAsyncF extends c.Function {}
export class WGPUProcBufferSetLabel extends c.Function {}
export class WGPUProcBufferUnmap extends c.Function {}
export class WGPUProcBufferAddRef extends c.Function {}
export class WGPUProcBufferRelease extends c.Function {}
export class WGPUProcCommandBufferSetLabel extends c.Function {}
export class WGPUProcCommandBufferAddRef extends c.Function {}
export class WGPUProcCommandBufferRelease extends c.Function {}
export class WGPUProcCommandEncoderBeginComputePass extends c.Function {}
export class WGPUProcCommandEncoderBeginRenderPass extends c.Function {}
export class WGPUProcCommandEncoderClearBuffer extends c.Function {}
export class WGPUProcCommandEncoderCopyBufferToBuffer extends c.Function {}
export class WGPUProcCommandEncoderCopyBufferToTexture extends c.Function {}
export class WGPUProcCommandEncoderCopyTextureToBuffer extends c.Function {}
export class WGPUProcCommandEncoderCopyTextureToTexture extends c.Function {}
export class WGPUProcCommandEncoderFinish extends c.Function {}
export class WGPUProcCommandEncoderInjectValidationError extends c.Function {}
export class WGPUProcCommandEncoderInsertDebugMarker extends c.Function {}
export class WGPUProcCommandEncoderPopDebugGroup extends c.Function {}
export class WGPUProcCommandEncoderPushDebugGroup extends c.Function {}
export class WGPUProcCommandEncoderResolveQuerySet extends c.Function {}
export class WGPUProcCommandEncoderSetLabel extends c.Function {}
export class WGPUProcCommandEncoderWriteBuffer extends c.Function {}
export class WGPUProcCommandEncoderWriteTimestamp extends c.Function {}
export class WGPUProcCommandEncoderAddRef extends c.Function {}
export class WGPUProcCommandEncoderRelease extends c.Function {}
export class WGPUProcComputePassEncoderDispatchWorkgroups extends c.Function {}
export class WGPUProcComputePassEncoderDispatchWorkgroupsIndirect extends c.Function {}
export class WGPUProcComputePassEncoderEnd extends c.Function {}
export class WGPUProcComputePassEncoderInsertDebugMarker extends c.Function {}
export class WGPUProcComputePassEncoderPopDebugGroup extends c.Function {}
export class WGPUProcComputePassEncoderPushDebugGroup extends c.Function {}
export class WGPUProcComputePassEncoderSetBindGroup extends c.Function {}
export class WGPUProcComputePassEncoderSetLabel extends c.Function {}
export class WGPUProcComputePassEncoderSetPipeline extends c.Function {}
export class WGPUProcComputePassEncoderWriteTimestamp extends c.Function {}
export class WGPUProcComputePassEncoderAddRef extends c.Function {}
export class WGPUProcComputePassEncoderRelease extends c.Function {}
export class WGPUProcComputePipelineGetBindGroupLayout extends c.Function {}
export class WGPUProcComputePipelineSetLabel extends c.Function {}
export class WGPUProcComputePipelineAddRef extends c.Function {}
export class WGPUProcComputePipelineRelease extends c.Function {}
export class WGPUProcDeviceCreateBindGroup extends c.Function {}
export class WGPUProcDeviceCreateBindGroupLayout extends c.Function {}
export class WGPUProcDeviceCreateBuffer extends c.Function {}
export class WGPUProcDeviceCreateCommandEncoder extends c.Function {}
export class WGPUProcDeviceCreateComputePipeline extends c.Function {}
export class WGPUProcDeviceCreateComputePipelineAsync extends c.Function {}
export class WGPUProcDeviceCreateComputePipelineAsync2 extends c.Function {}
export class WGPUProcDeviceCreateComputePipelineAsyncF extends c.Function {}
export class WGPUProcDeviceCreateErrorBuffer extends c.Function {}
export class WGPUProcDeviceCreateErrorExternalTexture extends c.Function {}
export class WGPUProcDeviceCreateErrorShaderModule extends c.Function {}
export class WGPUProcDeviceCreateErrorTexture extends c.Function {}
export class WGPUProcDeviceCreateExternalTexture extends c.Function {}
export class WGPUProcDeviceCreatePipelineLayout extends c.Function {}
export class WGPUProcDeviceCreateQuerySet extends c.Function {}
export class WGPUProcDeviceCreateRenderBundleEncoder extends c.Function {}
export class WGPUProcDeviceCreateRenderPipeline extends c.Function {}
export class WGPUProcDeviceCreateRenderPipelineAsync extends c.Function {}
export class WGPUProcDeviceCreateRenderPipelineAsync2 extends c.Function {}
export class WGPUProcDeviceCreateRenderPipelineAsyncF extends c.Function {}
export class WGPUProcDeviceCreateSampler extends c.Function {}
export class WGPUProcDeviceCreateShaderModule extends c.Function {}
export class WGPUProcDeviceCreateTexture extends c.Function {}
export class WGPUProcDeviceDestroy extends c.Function {}
export class WGPUProcDeviceForceLoss extends c.Function {}
export class WGPUProcDeviceGetAHardwareBufferProperties extends c.Function {}
export class WGPUProcDeviceGetAdapter extends c.Function {}
export class WGPUProcDeviceGetAdapterInfo extends c.Function {}
export class WGPUProcDeviceGetFeatures extends c.Function {}
export class WGPUProcDeviceGetLimits extends c.Function {}
export class WGPUProcDeviceGetLostFuture extends c.Function {}
export class WGPUProcDeviceGetQueue extends c.Function {}
export class WGPUProcDeviceHasFeature extends c.Function {}
export class WGPUProcDeviceImportSharedBufferMemory extends c.Function {}
export class WGPUProcDeviceImportSharedFence extends c.Function {}
export class WGPUProcDeviceImportSharedTextureMemory extends c.Function {}
export class WGPUProcDeviceInjectError extends c.Function {}
export class WGPUProcDevicePopErrorScope extends c.Function {}
export class WGPUProcDevicePopErrorScope2 extends c.Function {}
export class WGPUProcDevicePopErrorScopeF extends c.Function {}
export class WGPUProcDevicePushErrorScope extends c.Function {}
export class WGPUProcDeviceSetLabel extends c.Function {}
export class WGPUProcDeviceSetLoggingCallback extends c.Function {}
export class WGPUProcDeviceTick extends c.Function {}
export class WGPUProcDeviceValidateTextureDescriptor extends c.Function {}
export class WGPUProcDeviceAddRef extends c.Function {}
export class WGPUProcDeviceRelease extends c.Function {}
export class WGPUProcExternalTextureDestroy extends c.Function {}
export class WGPUProcExternalTextureExpire extends c.Function {}
export class WGPUProcExternalTextureRefresh extends c.Function {}
export class WGPUProcExternalTextureSetLabel extends c.Function {}
export class WGPUProcExternalTextureAddRef extends c.Function {}
export class WGPUProcExternalTextureRelease extends c.Function {}
export class WGPUProcInstanceCreateSurface extends c.Function {}
export class WGPUProcInstanceEnumerateWGSLLanguageFeatures extends c.Function {}
export class WGPUProcInstanceHasWGSLLanguageFeature extends c.Function {}
export class WGPUProcInstanceProcessEvents extends c.Function {}
export class WGPUProcInstanceRequestAdapter extends c.Function {}
export class WGPUProcInstanceRequestAdapter2 extends c.Function {}
export class WGPUProcInstanceRequestAdapterF extends c.Function {}
export class WGPUProcInstanceWaitAny extends c.Function {}
export class WGPUProcInstanceAddRef extends c.Function {}
export class WGPUProcInstanceRelease extends c.Function {}
export class WGPUProcPipelineLayoutSetLabel extends c.Function {}
export class WGPUProcPipelineLayoutAddRef extends c.Function {}
export class WGPUProcPipelineLayoutRelease extends c.Function {}
export class WGPUProcQuerySetDestroy extends c.Function {}
export class WGPUProcQuerySetGetCount extends c.Function {}
export class WGPUProcQuerySetGetType extends c.Function {}
export class WGPUProcQuerySetSetLabel extends c.Function {}
export class WGPUProcQuerySetAddRef extends c.Function {}
export class WGPUProcQuerySetRelease extends c.Function {}
export class WGPUProcQueueCopyExternalTextureForBrowser extends c.Function {}
export class WGPUProcQueueCopyTextureForBrowser extends c.Function {}
export class WGPUProcQueueOnSubmittedWorkDone extends c.Function {}
export class WGPUProcQueueOnSubmittedWorkDone2 extends c.Function {}
export class WGPUProcQueueOnSubmittedWorkDoneF extends c.Function {}
export class WGPUProcQueueSetLabel extends c.Function {}
export class WGPUProcQueueSubmit extends c.Function {}
export class WGPUProcQueueWriteBuffer extends c.Function {}
export class WGPUProcQueueWriteTexture extends c.Function {}
export class WGPUProcQueueAddRef extends c.Function {}
export class WGPUProcQueueRelease extends c.Function {}
export class WGPUProcRenderBundleSetLabel extends c.Function {}
export class WGPUProcRenderBundleAddRef extends c.Function {}
export class WGPUProcRenderBundleRelease extends c.Function {}
export class WGPUProcRenderBundleEncoderDraw extends c.Function {}
export class WGPUProcRenderBundleEncoderDrawIndexed extends c.Function {}
export class WGPUProcRenderBundleEncoderDrawIndexedIndirect extends c.Function {}
export class WGPUProcRenderBundleEncoderDrawIndirect extends c.Function {}
export class WGPUProcRenderBundleEncoderFinish extends c.Function {}
export class WGPUProcRenderBundleEncoderInsertDebugMarker extends c.Function {}
export class WGPUProcRenderBundleEncoderPopDebugGroup extends c.Function {}
export class WGPUProcRenderBundleEncoderPushDebugGroup extends c.Function {}
export class WGPUProcRenderBundleEncoderSetBindGroup extends c.Function {}
export class WGPUProcRenderBundleEncoderSetIndexBuffer extends c.Function {}
export class WGPUProcRenderBundleEncoderSetLabel extends c.Function {}
export class WGPUProcRenderBundleEncoderSetPipeline extends c.Function {}
export class WGPUProcRenderBundleEncoderSetVertexBuffer extends c.Function {}
export class WGPUProcRenderBundleEncoderAddRef extends c.Function {}
export class WGPUProcRenderBundleEncoderRelease extends c.Function {}
export class WGPUProcRenderPassEncoderBeginOcclusionQuery extends c.Function {}
export class WGPUProcRenderPassEncoderDraw extends c.Function {}
export class WGPUProcRenderPassEncoderDrawIndexed extends c.Function {}
export class WGPUProcRenderPassEncoderDrawIndexedIndirect extends c.Function {}
export class WGPUProcRenderPassEncoderDrawIndirect extends c.Function {}
export class WGPUProcRenderPassEncoderEnd extends c.Function {}
export class WGPUProcRenderPassEncoderEndOcclusionQuery extends c.Function {}
export class WGPUProcRenderPassEncoderExecuteBundles extends c.Function {}
export class WGPUProcRenderPassEncoderInsertDebugMarker extends c.Function {}
export class WGPUProcRenderPassEncoderMultiDrawIndexedIndirect extends c.Function {}
export class WGPUProcRenderPassEncoderMultiDrawIndirect extends c.Function {}
export class WGPUProcRenderPassEncoderPixelLocalStorageBarrier extends c.Function {}
export class WGPUProcRenderPassEncoderPopDebugGroup extends c.Function {}
export class WGPUProcRenderPassEncoderPushDebugGroup extends c.Function {}
export class WGPUProcRenderPassEncoderSetBindGroup extends c.Function {}
export class WGPUProcRenderPassEncoderSetBlendConstant extends c.Function {}
export class WGPUProcRenderPassEncoderSetIndexBuffer extends c.Function {}
export class WGPUProcRenderPassEncoderSetLabel extends c.Function {}
export class WGPUProcRenderPassEncoderSetPipeline extends c.Function {}
export class WGPUProcRenderPassEncoderSetScissorRect extends c.Function {}
export class WGPUProcRenderPassEncoderSetStencilReference extends c.Function {}
export class WGPUProcRenderPassEncoderSetVertexBuffer extends c.Function {}
export class WGPUProcRenderPassEncoderSetViewport extends c.Function {}
export class WGPUProcRenderPassEncoderWriteTimestamp extends c.Function {}
export class WGPUProcRenderPassEncoderAddRef extends c.Function {}
export class WGPUProcRenderPassEncoderRelease extends c.Function {}
export class WGPUProcRenderPipelineGetBindGroupLayout extends c.Function {}
export class WGPUProcRenderPipelineSetLabel extends c.Function {}
export class WGPUProcRenderPipelineAddRef extends c.Function {}
export class WGPUProcRenderPipelineRelease extends c.Function {}
export class WGPUProcSamplerSetLabel extends c.Function {}
export class WGPUProcSamplerAddRef extends c.Function {}
export class WGPUProcSamplerRelease extends c.Function {}
export class WGPUProcShaderModuleGetCompilationInfo extends c.Function {}
export class WGPUProcShaderModuleGetCompilationInfo2 extends c.Function {}
export class WGPUProcShaderModuleGetCompilationInfoF extends c.Function {}
export class WGPUProcShaderModuleSetLabel extends c.Function {}
export class WGPUProcShaderModuleAddRef extends c.Function {}
export class WGPUProcShaderModuleRelease extends c.Function {}
export class WGPUProcSharedBufferMemoryBeginAccess extends c.Function {}
export class WGPUProcSharedBufferMemoryCreateBuffer extends c.Function {}
export class WGPUProcSharedBufferMemoryEndAccess extends c.Function {}
export class WGPUProcSharedBufferMemoryGetProperties extends c.Function {}
export class WGPUProcSharedBufferMemoryIsDeviceLost extends c.Function {}
export class WGPUProcSharedBufferMemorySetLabel extends c.Function {}
export class WGPUProcSharedBufferMemoryAddRef extends c.Function {}
export class WGPUProcSharedBufferMemoryRelease extends c.Function {}
export class WGPUProcSharedFenceExportInfo extends c.Function {}
export class WGPUProcSharedFenceAddRef extends c.Function {}
export class WGPUProcSharedFenceRelease extends c.Function {}
export class WGPUProcSharedTextureMemoryBeginAccess extends c.Function {}
export class WGPUProcSharedTextureMemoryCreateTexture extends c.Function {}
export class WGPUProcSharedTextureMemoryEndAccess extends c.Function {}
export class WGPUProcSharedTextureMemoryGetProperties extends c.Function {}
export class WGPUProcSharedTextureMemoryIsDeviceLost extends c.Function {}
export class WGPUProcSharedTextureMemorySetLabel extends c.Function {}
export class WGPUProcSharedTextureMemoryAddRef extends c.Function {}
export class WGPUProcSharedTextureMemoryRelease extends c.Function {}
export class WGPUProcSurfaceConfigure extends c.Function {}
export class WGPUProcSurfaceGetCapabilities extends c.Function {}
export class WGPUProcSurfaceGetCurrentTexture extends c.Function {}
export class WGPUProcSurfacePresent extends c.Function {}
export class WGPUProcSurfaceSetLabel extends c.Function {}
export class WGPUProcSurfaceUnconfigure extends c.Function {}
export class WGPUProcSurfaceAddRef extends c.Function {}
export class WGPUProcSurfaceRelease extends c.Function {}
export class WGPUProcTextureCreateErrorView extends c.Function {}
export class WGPUProcTextureCreateView extends c.Function {}
export class WGPUProcTextureDestroy extends c.Function {}
export class WGPUProcTextureGetDepthOrArrayLayers extends c.Function {}
export class WGPUProcTextureGetDimension extends c.Function {}
export class WGPUProcTextureGetFormat extends c.Function {}
export class WGPUProcTextureGetHeight extends c.Function {}
export class WGPUProcTextureGetMipLevelCount extends c.Function {}
export class WGPUProcTextureGetSampleCount extends c.Function {}
export class WGPUProcTextureGetUsage extends c.Function {}
export class WGPUProcTextureGetWidth extends c.Function {}
export class WGPUProcTextureSetLabel extends c.Function {}
export class WGPUProcTextureAddRef extends c.Function {}
export class WGPUProcTextureRelease extends c.Function {}
export class WGPUProcTextureViewSetLabel extends c.Function {}
export class WGPUProcTextureViewAddRef extends c.Function {}
export class WGPUProcTextureViewRelease extends c.Function {}

// functions
export const adapterInfoFreeMembers = (value: WGPUAdapterInfo): c.Void => new c.Void(lib.symbols.wgpuAdapterInfoFreeMembers(value.value))
export const adapterPropertiesMemoryHeapsFreeMembers = (value: WGPUAdapterPropertiesMemoryHeaps): c.Void => new c.Void(lib.symbols.wgpuAdapterPropertiesMemoryHeapsFreeMembers(value.value))
export const createInstance = (descriptor: c.Pointer<WGPUInstanceDescriptor>): WGPUInstance => new WGPUInstance(lib.symbols.wgpuCreateInstance(descriptor.value))
export const drmFormatCapabilitiesFreeMembers = (value: WGPUDrmFormatCapabilities): c.Void => new c.Void(lib.symbols.wgpuDrmFormatCapabilitiesFreeMembers(value.value))
export const getInstanceFeatures = (features: c.Pointer<WGPUInstanceFeatures>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuGetInstanceFeatures(features.value))
export const getProcAddress = (procName: WGPUStringView): WGPUProc => new WGPUProc(lib.symbols.wgpuGetProcAddress(procName.value))
export const sharedBufferMemoryEndAccessStateFreeMembers = (value: WGPUSharedBufferMemoryEndAccessState): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemoryEndAccessStateFreeMembers(value.value))
export const sharedTextureMemoryEndAccessStateFreeMembers = (value: WGPUSharedTextureMemoryEndAccessState): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemoryEndAccessStateFreeMembers(value.value))
export const supportedFeaturesFreeMembers = (value: WGPUSupportedFeatures): c.Void => new c.Void(lib.symbols.wgpuSupportedFeaturesFreeMembers(value.value))
export const surfaceCapabilitiesFreeMembers = (value: WGPUSurfaceCapabilities): c.Void => new c.Void(lib.symbols.wgpuSurfaceCapabilitiesFreeMembers(value.value))
export const adapterCreateDevice = (adapter: WGPUAdapter, descriptor: c.Pointer<WGPUDeviceDescriptor>): WGPUDevice => new WGPUDevice(lib.symbols.wgpuAdapterCreateDevice(adapter.value, descriptor.value))
export const adapterGetFeatures = (adapter: WGPUAdapter, features: c.Pointer<WGPUSupportedFeatures>): c.Void => new c.Void(lib.symbols.wgpuAdapterGetFeatures(adapter.value, features.value))
export const adapterGetFormatCapabilities = (adapter: WGPUAdapter, format: WGPUTextureFormat, capabilities: c.Pointer<WGPUFormatCapabilities>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuAdapterGetFormatCapabilities(adapter.value, format.value, capabilities.value))
export const adapterGetInfo = (adapter: WGPUAdapter, info: c.Pointer<WGPUAdapterInfo>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuAdapterGetInfo(adapter.value, info.value))
export const adapterGetInstance = (adapter: WGPUAdapter): WGPUInstance => new WGPUInstance(lib.symbols.wgpuAdapterGetInstance(adapter.value))
export const adapterGetLimits = (adapter: WGPUAdapter, limits: c.Pointer<WGPUSupportedLimits>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuAdapterGetLimits(adapter.value, limits.value))
export const adapterHasFeature = (adapter: WGPUAdapter, feature: WGPUFeatureName): WGPUBool => new WGPUBool(lib.symbols.wgpuAdapterHasFeature(adapter.value, feature.value))
export const adapterRequestDevice = (adapter: WGPUAdapter, descriptor: c.Pointer<WGPUDeviceDescriptor>, callback: WGPURequestDeviceCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuAdapterRequestDevice(adapter.value, descriptor.value, callback.value, userdata.value))
export const adapterRequestDevice2 = (adapter: WGPUAdapter, options: c.Pointer<WGPUDeviceDescriptor>, callbackInfo: WGPURequestDeviceCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuAdapterRequestDevice2(adapter.value, options.value, callbackInfo.value))
export const adapterRequestDeviceF = (adapter: WGPUAdapter, options: c.Pointer<WGPUDeviceDescriptor>, callbackInfo: WGPURequestDeviceCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuAdapterRequestDeviceF(adapter.value, options.value, callbackInfo.value))
export const adapterAddRef = (adapter: WGPUAdapter): c.Void => new c.Void(lib.symbols.wgpuAdapterAddRef(adapter.value))
export const adapterRelease = (adapter: WGPUAdapter): c.Void => new c.Void(lib.symbols.wgpuAdapterRelease(adapter.value))
export const bindGroupSetLabel = (bindGroup: WGPUBindGroup, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuBindGroupSetLabel(bindGroup.value, label.value))
export const bindGroupAddRef = (bindGroup: WGPUBindGroup): c.Void => new c.Void(lib.symbols.wgpuBindGroupAddRef(bindGroup.value))
export const bindGroupRelease = (bindGroup: WGPUBindGroup): c.Void => new c.Void(lib.symbols.wgpuBindGroupRelease(bindGroup.value))
export const bindGroupLayoutSetLabel = (bindGroupLayout: WGPUBindGroupLayout, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuBindGroupLayoutSetLabel(bindGroupLayout.value, label.value))
export const bindGroupLayoutAddRef = (bindGroupLayout: WGPUBindGroupLayout): c.Void => new c.Void(lib.symbols.wgpuBindGroupLayoutAddRef(bindGroupLayout.value))
export const bindGroupLayoutRelease = (bindGroupLayout: WGPUBindGroupLayout): c.Void => new c.Void(lib.symbols.wgpuBindGroupLayoutRelease(bindGroupLayout.value))
export const bufferDestroy = (buffer: WGPUBuffer): c.Void => new c.Void(lib.symbols.wgpuBufferDestroy(buffer.value))
export const bufferGetConstMappedRange = (buffer: WGPUBuffer, offset: c.Size, size: c.Size): c.Pointer<c.Void> => new c.Pointer<c.Void>(lib.symbols.wgpuBufferGetConstMappedRange(buffer.value, offset.value, size.value))
export const bufferGetMapState = (buffer: WGPUBuffer): WGPUBufferMapState => new WGPUBufferMapState(lib.symbols.wgpuBufferGetMapState(buffer.value))
export const bufferGetMappedRange = (buffer: WGPUBuffer, offset: c.Size, size: c.Size): c.Pointer<c.Void> => new c.Pointer<c.Void>(lib.symbols.wgpuBufferGetMappedRange(buffer.value, offset.value, size.value))
export const bufferGetSize = (buffer: WGPUBuffer): c.U64 => new c.U64(lib.symbols.wgpuBufferGetSize(buffer.value))
export const bufferGetUsage = (buffer: WGPUBuffer): WGPUBufferUsage => new WGPUBufferUsage(lib.symbols.wgpuBufferGetUsage(buffer.value))
export const bufferMapAsync = (buffer: WGPUBuffer, mode: WGPUMapMode, offset: c.Size, size: c.Size, callback: WGPUBufferMapCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuBufferMapAsync(buffer.value, mode.value, offset.value, size.value, callback.value, userdata.value))
export const bufferMapAsync2 = (buffer: WGPUBuffer, mode: WGPUMapMode, offset: c.Size, size: c.Size, callbackInfo: WGPUBufferMapCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuBufferMapAsync2(buffer.value, mode.value, offset.value, size.value, callbackInfo.value))
export const bufferMapAsyncF = (buffer: WGPUBuffer, mode: WGPUMapMode, offset: c.Size, size: c.Size, callbackInfo: WGPUBufferMapCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuBufferMapAsyncF(buffer.value, mode.value, offset.value, size.value, callbackInfo.value))
export const bufferSetLabel = (buffer: WGPUBuffer, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuBufferSetLabel(buffer.value, label.value))
export const bufferUnmap = (buffer: WGPUBuffer): c.Void => new c.Void(lib.symbols.wgpuBufferUnmap(buffer.value))
export const bufferAddRef = (buffer: WGPUBuffer): c.Void => new c.Void(lib.symbols.wgpuBufferAddRef(buffer.value))
export const bufferRelease = (buffer: WGPUBuffer): c.Void => new c.Void(lib.symbols.wgpuBufferRelease(buffer.value))
export const commandBufferSetLabel = (commandBuffer: WGPUCommandBuffer, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuCommandBufferSetLabel(commandBuffer.value, label.value))
export const commandBufferAddRef = (commandBuffer: WGPUCommandBuffer): c.Void => new c.Void(lib.symbols.wgpuCommandBufferAddRef(commandBuffer.value))
export const commandBufferRelease = (commandBuffer: WGPUCommandBuffer): c.Void => new c.Void(lib.symbols.wgpuCommandBufferRelease(commandBuffer.value))
export const commandEncoderBeginComputePass = (commandEncoder: WGPUCommandEncoder, descriptor: c.Pointer<WGPUComputePassDescriptor>): WGPUComputePassEncoder => new WGPUComputePassEncoder(lib.symbols.wgpuCommandEncoderBeginComputePass(commandEncoder.value, descriptor.value))
export const commandEncoderBeginRenderPass = (commandEncoder: WGPUCommandEncoder, descriptor: c.Pointer<WGPURenderPassDescriptor>): WGPURenderPassEncoder => new WGPURenderPassEncoder(lib.symbols.wgpuCommandEncoderBeginRenderPass(commandEncoder.value, descriptor.value))
export const commandEncoderClearBuffer = (commandEncoder: WGPUCommandEncoder, buffer: WGPUBuffer, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderClearBuffer(commandEncoder.value, buffer.value, offset.value, size.value))
export const commandEncoderCopyBufferToBuffer = (commandEncoder: WGPUCommandEncoder, source: WGPUBuffer, sourceOffset: c.U64, destination: WGPUBuffer, destinationOffset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyBufferToBuffer(commandEncoder.value, source.value, sourceOffset.value, destination.value, destinationOffset.value, size.value))
export const commandEncoderCopyBufferToTexture = (commandEncoder: WGPUCommandEncoder, source: c.Pointer<WGPUImageCopyBuffer>, destination: c.Pointer<WGPUImageCopyTexture>, copySize: c.Pointer<WGPUExtent3D>): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyBufferToTexture(commandEncoder.value, source.value, destination.value, copySize.value))
export const commandEncoderCopyTextureToBuffer = (commandEncoder: WGPUCommandEncoder, source: c.Pointer<WGPUImageCopyTexture>, destination: c.Pointer<WGPUImageCopyBuffer>, copySize: c.Pointer<WGPUExtent3D>): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyTextureToBuffer(commandEncoder.value, source.value, destination.value, copySize.value))
export const commandEncoderCopyTextureToTexture = (commandEncoder: WGPUCommandEncoder, source: c.Pointer<WGPUImageCopyTexture>, destination: c.Pointer<WGPUImageCopyTexture>, copySize: c.Pointer<WGPUExtent3D>): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyTextureToTexture(commandEncoder.value, source.value, destination.value, copySize.value))
export const commandEncoderFinish = (commandEncoder: WGPUCommandEncoder, descriptor: c.Pointer<WGPUCommandBufferDescriptor>): WGPUCommandBuffer => new WGPUCommandBuffer(lib.symbols.wgpuCommandEncoderFinish(commandEncoder.value, descriptor.value))
export const commandEncoderInjectValidationError = (commandEncoder: WGPUCommandEncoder, message: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderInjectValidationError(commandEncoder.value, message.value))
export const commandEncoderInsertDebugMarker = (commandEncoder: WGPUCommandEncoder, markerLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderInsertDebugMarker(commandEncoder.value, markerLabel.value))
export const commandEncoderPopDebugGroup = (commandEncoder: WGPUCommandEncoder): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderPopDebugGroup(commandEncoder.value))
export const commandEncoderPushDebugGroup = (commandEncoder: WGPUCommandEncoder, groupLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderPushDebugGroup(commandEncoder.value, groupLabel.value))
export const commandEncoderResolveQuerySet = (commandEncoder: WGPUCommandEncoder, querySet: WGPUQuerySet, firstQuery: c.U32, queryCount: c.U32, destination: WGPUBuffer, destinationOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderResolveQuerySet(commandEncoder.value, querySet.value, firstQuery.value, queryCount.value, destination.value, destinationOffset.value))
export const commandEncoderSetLabel = (commandEncoder: WGPUCommandEncoder, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderSetLabel(commandEncoder.value, label.value))
export const commandEncoderWriteBuffer = (commandEncoder: WGPUCommandEncoder, buffer: WGPUBuffer, bufferOffset: c.U64, data: c.Pointer<c.U8>, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderWriteBuffer(commandEncoder.value, buffer.value, bufferOffset.value, data.value, size.value))
export const commandEncoderWriteTimestamp = (commandEncoder: WGPUCommandEncoder, querySet: WGPUQuerySet, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderWriteTimestamp(commandEncoder.value, querySet.value, queryIndex.value))
export const commandEncoderAddRef = (commandEncoder: WGPUCommandEncoder): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderAddRef(commandEncoder.value))
export const commandEncoderRelease = (commandEncoder: WGPUCommandEncoder): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderRelease(commandEncoder.value))
export const computePassEncoderDispatchWorkgroups = (computePassEncoder: WGPUComputePassEncoder, workgroupCountX: c.U32, workgroupCountY: c.U32, workgroupCountZ: c.U32): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder.value, workgroupCountX.value, workgroupCountY.value, workgroupCountZ.value))
export const computePassEncoderDispatchWorkgroupsIndirect = (computePassEncoder: WGPUComputePassEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder.value, indirectBuffer.value, indirectOffset.value))
export const computePassEncoderEnd = (computePassEncoder: WGPUComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderEnd(computePassEncoder.value))
export const computePassEncoderInsertDebugMarker = (computePassEncoder: WGPUComputePassEncoder, markerLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderInsertDebugMarker(computePassEncoder.value, markerLabel.value))
export const computePassEncoderPopDebugGroup = (computePassEncoder: WGPUComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderPopDebugGroup(computePassEncoder.value))
export const computePassEncoderPushDebugGroup = (computePassEncoder: WGPUComputePassEncoder, groupLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderPushDebugGroup(computePassEncoder.value, groupLabel.value))
export const computePassEncoderSetBindGroup = (computePassEncoder: WGPUComputePassEncoder, groupIndex: c.U32, group: WGPUBindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderSetBindGroup(computePassEncoder.value, groupIndex.value, group.value, dynamicOffsetCount.value, dynamicOffsets.value))
export const computePassEncoderSetLabel = (computePassEncoder: WGPUComputePassEncoder, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderSetLabel(computePassEncoder.value, label.value))
export const computePassEncoderSetPipeline = (computePassEncoder: WGPUComputePassEncoder, pipeline: WGPUComputePipeline): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderSetPipeline(computePassEncoder.value, pipeline.value))
export const computePassEncoderWriteTimestamp = (computePassEncoder: WGPUComputePassEncoder, querySet: WGPUQuerySet, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderWriteTimestamp(computePassEncoder.value, querySet.value, queryIndex.value))
export const computePassEncoderAddRef = (computePassEncoder: WGPUComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderAddRef(computePassEncoder.value))
export const computePassEncoderRelease = (computePassEncoder: WGPUComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderRelease(computePassEncoder.value))
export const computePipelineGetBindGroupLayout = (computePipeline: WGPUComputePipeline, groupIndex: c.U32): WGPUBindGroupLayout => new WGPUBindGroupLayout(lib.symbols.wgpuComputePipelineGetBindGroupLayout(computePipeline.value, groupIndex.value))
export const computePipelineSetLabel = (computePipeline: WGPUComputePipeline, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuComputePipelineSetLabel(computePipeline.value, label.value))
export const computePipelineAddRef = (computePipeline: WGPUComputePipeline): c.Void => new c.Void(lib.symbols.wgpuComputePipelineAddRef(computePipeline.value))
export const computePipelineRelease = (computePipeline: WGPUComputePipeline): c.Void => new c.Void(lib.symbols.wgpuComputePipelineRelease(computePipeline.value))
export const deviceCreateBindGroup = (device: WGPUDevice, descriptor: c.Pointer<WGPUBindGroupDescriptor>): WGPUBindGroup => new WGPUBindGroup(lib.symbols.wgpuDeviceCreateBindGroup(device.value, descriptor.value))
export const deviceCreateBindGroupLayout = (device: WGPUDevice, descriptor: c.Pointer<WGPUBindGroupLayoutDescriptor>): WGPUBindGroupLayout => new WGPUBindGroupLayout(lib.symbols.wgpuDeviceCreateBindGroupLayout(device.value, descriptor.value))
export const deviceCreateBuffer = (device: WGPUDevice, descriptor: c.Pointer<WGPUBufferDescriptor>): WGPUBuffer => new WGPUBuffer(lib.symbols.wgpuDeviceCreateBuffer(device.value, descriptor.value))
export const deviceCreateCommandEncoder = (device: WGPUDevice, descriptor: c.Pointer<WGPUCommandEncoderDescriptor>): WGPUCommandEncoder => new WGPUCommandEncoder(lib.symbols.wgpuDeviceCreateCommandEncoder(device.value, descriptor.value))
export const deviceCreateComputePipeline = (device: WGPUDevice, descriptor: c.Pointer<WGPUComputePipelineDescriptor>): WGPUComputePipeline => new WGPUComputePipeline(lib.symbols.wgpuDeviceCreateComputePipeline(device.value, descriptor.value))
export const deviceCreateComputePipelineAsync = (device: WGPUDevice, descriptor: c.Pointer<WGPUComputePipelineDescriptor>, callback: WGPUCreateComputePipelineAsyncCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDeviceCreateComputePipelineAsync(device.value, descriptor.value, callback.value, userdata.value))
export const deviceCreateComputePipelineAsync2 = (device: WGPUDevice, descriptor: c.Pointer<WGPUComputePipelineDescriptor>, callbackInfo: WGPUCreateComputePipelineAsyncCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDeviceCreateComputePipelineAsync2(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateComputePipelineAsyncF = (device: WGPUDevice, descriptor: c.Pointer<WGPUComputePipelineDescriptor>, callbackInfo: WGPUCreateComputePipelineAsyncCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDeviceCreateComputePipelineAsyncF(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateErrorBuffer = (device: WGPUDevice, descriptor: c.Pointer<WGPUBufferDescriptor>): WGPUBuffer => new WGPUBuffer(lib.symbols.wgpuDeviceCreateErrorBuffer(device.value, descriptor.value))
export const deviceCreateErrorExternalTexture = (device: WGPUDevice): WGPUExternalTexture => new WGPUExternalTexture(lib.symbols.wgpuDeviceCreateErrorExternalTexture(device.value))
export const deviceCreateErrorShaderModule = (device: WGPUDevice, descriptor: c.Pointer<WGPUShaderModuleDescriptor>, errorMessage: WGPUStringView): WGPUShaderModule => new WGPUShaderModule(lib.symbols.wgpuDeviceCreateErrorShaderModule(device.value, descriptor.value, errorMessage.value))
export const deviceCreateErrorTexture = (device: WGPUDevice, descriptor: c.Pointer<WGPUTextureDescriptor>): WGPUTexture => new WGPUTexture(lib.symbols.wgpuDeviceCreateErrorTexture(device.value, descriptor.value))
export const deviceCreateExternalTexture = (device: WGPUDevice, externalTextureDescriptor: c.Pointer<WGPUExternalTextureDescriptor>): WGPUExternalTexture => new WGPUExternalTexture(lib.symbols.wgpuDeviceCreateExternalTexture(device.value, externalTextureDescriptor.value))
export const deviceCreatePipelineLayout = (device: WGPUDevice, descriptor: c.Pointer<WGPUPipelineLayoutDescriptor>): WGPUPipelineLayout => new WGPUPipelineLayout(lib.symbols.wgpuDeviceCreatePipelineLayout(device.value, descriptor.value))
export const deviceCreateQuerySet = (device: WGPUDevice, descriptor: c.Pointer<WGPUQuerySetDescriptor>): WGPUQuerySet => new WGPUQuerySet(lib.symbols.wgpuDeviceCreateQuerySet(device.value, descriptor.value))
export const deviceCreateRenderBundleEncoder = (device: WGPUDevice, descriptor: c.Pointer<WGPURenderBundleEncoderDescriptor>): WGPURenderBundleEncoder => new WGPURenderBundleEncoder(lib.symbols.wgpuDeviceCreateRenderBundleEncoder(device.value, descriptor.value))
export const deviceCreateRenderPipeline = (device: WGPUDevice, descriptor: c.Pointer<WGPURenderPipelineDescriptor>): WGPURenderPipeline => new WGPURenderPipeline(lib.symbols.wgpuDeviceCreateRenderPipeline(device.value, descriptor.value))
export const deviceCreateRenderPipelineAsync = (device: WGPUDevice, descriptor: c.Pointer<WGPURenderPipelineDescriptor>, callback: WGPUCreateRenderPipelineAsyncCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDeviceCreateRenderPipelineAsync(device.value, descriptor.value, callback.value, userdata.value))
export const deviceCreateRenderPipelineAsync2 = (device: WGPUDevice, descriptor: c.Pointer<WGPURenderPipelineDescriptor>, callbackInfo: WGPUCreateRenderPipelineAsyncCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDeviceCreateRenderPipelineAsync2(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateRenderPipelineAsyncF = (device: WGPUDevice, descriptor: c.Pointer<WGPURenderPipelineDescriptor>, callbackInfo: WGPUCreateRenderPipelineAsyncCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDeviceCreateRenderPipelineAsyncF(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateSampler = (device: WGPUDevice, descriptor: c.Pointer<WGPUSamplerDescriptor>): WGPUSampler => new WGPUSampler(lib.symbols.wgpuDeviceCreateSampler(device.value, descriptor.value))
export const deviceCreateShaderModule = (device: WGPUDevice, descriptor: c.Pointer<WGPUShaderModuleDescriptor>): WGPUShaderModule => new WGPUShaderModule(lib.symbols.wgpuDeviceCreateShaderModule(device.value, descriptor.value))
export const deviceCreateTexture = (device: WGPUDevice, descriptor: c.Pointer<WGPUTextureDescriptor>): WGPUTexture => new WGPUTexture(lib.symbols.wgpuDeviceCreateTexture(device.value, descriptor.value))
export const deviceDestroy = (device: WGPUDevice): c.Void => new c.Void(lib.symbols.wgpuDeviceDestroy(device.value))
export const deviceForceLoss = (device: WGPUDevice, type: WGPUDeviceLostReason, message: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuDeviceForceLoss(device.value, type.value, message.value))
export const deviceGetAHardwareBufferProperties = (device: WGPUDevice, handle: c.Pointer<c.Void>, properties: c.Pointer<WGPUAHardwareBufferProperties>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuDeviceGetAHardwareBufferProperties(device.value, handle.value, properties.value))
export const deviceGetAdapter = (device: WGPUDevice): WGPUAdapter => new WGPUAdapter(lib.symbols.wgpuDeviceGetAdapter(device.value))
export const deviceGetAdapterInfo = (device: WGPUDevice, adapterInfo: c.Pointer<WGPUAdapterInfo>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuDeviceGetAdapterInfo(device.value, adapterInfo.value))
export const deviceGetFeatures = (device: WGPUDevice, features: c.Pointer<WGPUSupportedFeatures>): c.Void => new c.Void(lib.symbols.wgpuDeviceGetFeatures(device.value, features.value))
export const deviceGetLimits = (device: WGPUDevice, limits: c.Pointer<WGPUSupportedLimits>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuDeviceGetLimits(device.value, limits.value))
export const deviceGetLostFuture = (device: WGPUDevice): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDeviceGetLostFuture(device.value))
export const deviceGetQueue = (device: WGPUDevice): WGPUQueue => new WGPUQueue(lib.symbols.wgpuDeviceGetQueue(device.value))
export const deviceHasFeature = (device: WGPUDevice, feature: WGPUFeatureName): WGPUBool => new WGPUBool(lib.symbols.wgpuDeviceHasFeature(device.value, feature.value))
export const deviceImportSharedBufferMemory = (device: WGPUDevice, descriptor: c.Pointer<WGPUSharedBufferMemoryDescriptor>): WGPUSharedBufferMemory => new WGPUSharedBufferMemory(lib.symbols.wgpuDeviceImportSharedBufferMemory(device.value, descriptor.value))
export const deviceImportSharedFence = (device: WGPUDevice, descriptor: c.Pointer<WGPUSharedFenceDescriptor>): WGPUSharedFence => new WGPUSharedFence(lib.symbols.wgpuDeviceImportSharedFence(device.value, descriptor.value))
export const deviceImportSharedTextureMemory = (device: WGPUDevice, descriptor: c.Pointer<WGPUSharedTextureMemoryDescriptor>): WGPUSharedTextureMemory => new WGPUSharedTextureMemory(lib.symbols.wgpuDeviceImportSharedTextureMemory(device.value, descriptor.value))
export const deviceInjectError = (device: WGPUDevice, type: WGPUErrorType, message: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuDeviceInjectError(device.value, type.value, message.value))
export const devicePopErrorScope = (device: WGPUDevice, oldCallback: WGPUErrorCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDevicePopErrorScope(device.value, oldCallback.value, userdata.value))
export const devicePopErrorScope2 = (device: WGPUDevice, callbackInfo: WGPUPopErrorScopeCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDevicePopErrorScope2(device.value, callbackInfo.value))
export const devicePopErrorScopeF = (device: WGPUDevice, callbackInfo: WGPUPopErrorScopeCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuDevicePopErrorScopeF(device.value, callbackInfo.value))
export const devicePushErrorScope = (device: WGPUDevice, filter: WGPUErrorFilter): c.Void => new c.Void(lib.symbols.wgpuDevicePushErrorScope(device.value, filter.value))
export const deviceSetLabel = (device: WGPUDevice, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuDeviceSetLabel(device.value, label.value))
export const deviceSetLoggingCallback = (device: WGPUDevice, callback: WGPULoggingCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDeviceSetLoggingCallback(device.value, callback.value, userdata.value))
export const deviceTick = (device: WGPUDevice): c.Void => new c.Void(lib.symbols.wgpuDeviceTick(device.value))
export const deviceValidateTextureDescriptor = (device: WGPUDevice, descriptor: c.Pointer<WGPUTextureDescriptor>): c.Void => new c.Void(lib.symbols.wgpuDeviceValidateTextureDescriptor(device.value, descriptor.value))
export const deviceAddRef = (device: WGPUDevice): c.Void => new c.Void(lib.symbols.wgpuDeviceAddRef(device.value))
export const deviceRelease = (device: WGPUDevice): c.Void => new c.Void(lib.symbols.wgpuDeviceRelease(device.value))
export const externalTextureDestroy = (externalTexture: WGPUExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureDestroy(externalTexture.value))
export const externalTextureExpire = (externalTexture: WGPUExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureExpire(externalTexture.value))
export const externalTextureRefresh = (externalTexture: WGPUExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureRefresh(externalTexture.value))
export const externalTextureSetLabel = (externalTexture: WGPUExternalTexture, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuExternalTextureSetLabel(externalTexture.value, label.value))
export const externalTextureAddRef = (externalTexture: WGPUExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureAddRef(externalTexture.value))
export const externalTextureRelease = (externalTexture: WGPUExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureRelease(externalTexture.value))
export const instanceCreateSurface = (instance: WGPUInstance, descriptor: c.Pointer<WGPUSurfaceDescriptor>): WGPUSurface => new WGPUSurface(lib.symbols.wgpuInstanceCreateSurface(instance.value, descriptor.value))
export const instanceEnumerateWGSLLanguageFeatures = (instance: WGPUInstance, features: c.Pointer<WGPUWGSLFeatureName>): c.Size => new c.Size(lib.symbols.wgpuInstanceEnumerateWGSLLanguageFeatures(instance.value, features.value))
export const instanceHasWGSLLanguageFeature = (instance: WGPUInstance, feature: WGPUWGSLFeatureName): WGPUBool => new WGPUBool(lib.symbols.wgpuInstanceHasWGSLLanguageFeature(instance.value, feature.value))
export const instanceProcessEvents = (instance: WGPUInstance): c.Void => new c.Void(lib.symbols.wgpuInstanceProcessEvents(instance.value))
export const instanceRequestAdapter = (instance: WGPUInstance, options: c.Pointer<WGPURequestAdapterOptions>, callback: WGPURequestAdapterCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuInstanceRequestAdapter(instance.value, options.value, callback.value, userdata.value))
export const instanceRequestAdapter2 = (instance: WGPUInstance, options: c.Pointer<WGPURequestAdapterOptions>, callbackInfo: WGPURequestAdapterCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuInstanceRequestAdapter2(instance.value, options.value, callbackInfo.value))
export const instanceRequestAdapterF = (instance: WGPUInstance, options: c.Pointer<WGPURequestAdapterOptions>, callbackInfo: WGPURequestAdapterCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuInstanceRequestAdapterF(instance.value, options.value, callbackInfo.value))
export const instanceWaitAny = (instance: WGPUInstance, futureCount: c.Size, futures: c.Pointer<WGPUFutureWaitInfo>, timeoutNS: c.U64): WGPUWaitStatus => new WGPUWaitStatus(lib.symbols.wgpuInstanceWaitAny(instance.value, futureCount.value, futures.value, timeoutNS.value))
export const instanceAddRef = (instance: WGPUInstance): c.Void => new c.Void(lib.symbols.wgpuInstanceAddRef(instance.value))
export const instanceRelease = (instance: WGPUInstance): c.Void => new c.Void(lib.symbols.wgpuInstanceRelease(instance.value))
export const pipelineLayoutSetLabel = (pipelineLayout: WGPUPipelineLayout, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuPipelineLayoutSetLabel(pipelineLayout.value, label.value))
export const pipelineLayoutAddRef = (pipelineLayout: WGPUPipelineLayout): c.Void => new c.Void(lib.symbols.wgpuPipelineLayoutAddRef(pipelineLayout.value))
export const pipelineLayoutRelease = (pipelineLayout: WGPUPipelineLayout): c.Void => new c.Void(lib.symbols.wgpuPipelineLayoutRelease(pipelineLayout.value))
export const querySetDestroy = (querySet: WGPUQuerySet): c.Void => new c.Void(lib.symbols.wgpuQuerySetDestroy(querySet.value))
export const querySetGetCount = (querySet: WGPUQuerySet): c.U32 => new c.U32(lib.symbols.wgpuQuerySetGetCount(querySet.value))
export const querySetGetType = (querySet: WGPUQuerySet): WGPUQueryType => new WGPUQueryType(lib.symbols.wgpuQuerySetGetType(querySet.value))
export const querySetSetLabel = (querySet: WGPUQuerySet, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuQuerySetSetLabel(querySet.value, label.value))
export const querySetAddRef = (querySet: WGPUQuerySet): c.Void => new c.Void(lib.symbols.wgpuQuerySetAddRef(querySet.value))
export const querySetRelease = (querySet: WGPUQuerySet): c.Void => new c.Void(lib.symbols.wgpuQuerySetRelease(querySet.value))
export const queueCopyExternalTextureForBrowser = (queue: WGPUQueue, source: c.Pointer<WGPUImageCopyExternalTexture>, destination: c.Pointer<WGPUImageCopyTexture>, copySize: c.Pointer<WGPUExtent3D>, options: c.Pointer<WGPUCopyTextureForBrowserOptions>): c.Void => new c.Void(lib.symbols.wgpuQueueCopyExternalTextureForBrowser(queue.value, source.value, destination.value, copySize.value, options.value))
export const queueCopyTextureForBrowser = (queue: WGPUQueue, source: c.Pointer<WGPUImageCopyTexture>, destination: c.Pointer<WGPUImageCopyTexture>, copySize: c.Pointer<WGPUExtent3D>, options: c.Pointer<WGPUCopyTextureForBrowserOptions>): c.Void => new c.Void(lib.symbols.wgpuQueueCopyTextureForBrowser(queue.value, source.value, destination.value, copySize.value, options.value))
export const queueOnSubmittedWorkDone = (queue: WGPUQueue, callback: WGPUQueueWorkDoneCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuQueueOnSubmittedWorkDone(queue.value, callback.value, userdata.value))
export const queueOnSubmittedWorkDone2 = (queue: WGPUQueue, callbackInfo: WGPUQueueWorkDoneCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuQueueOnSubmittedWorkDone2(queue.value, callbackInfo.value))
export const queueOnSubmittedWorkDoneF = (queue: WGPUQueue, callbackInfo: WGPUQueueWorkDoneCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuQueueOnSubmittedWorkDoneF(queue.value, callbackInfo.value))
export const queueSetLabel = (queue: WGPUQueue, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuQueueSetLabel(queue.value, label.value))
export const queueSubmit = (queue: WGPUQueue, commandCount: c.Size, commands: c.Pointer<WGPUCommandBuffer>): c.Void => new c.Void(lib.symbols.wgpuQueueSubmit(queue.value, commandCount.value, commands.value))
export const queueWriteBuffer = (queue: WGPUQueue, buffer: WGPUBuffer, bufferOffset: c.U64, data: c.Pointer<c.Void>, size: c.Size): c.Void => new c.Void(lib.symbols.wgpuQueueWriteBuffer(queue.value, buffer.value, bufferOffset.value, data.value, size.value))
export const queueWriteTexture = (queue: WGPUQueue, destination: c.Pointer<WGPUImageCopyTexture>, data: c.Pointer<c.Void>, dataSize: c.Size, dataLayout: c.Pointer<WGPUTextureDataLayout>, writeSize: c.Pointer<WGPUExtent3D>): c.Void => new c.Void(lib.symbols.wgpuQueueWriteTexture(queue.value, destination.value, data.value, dataSize.value, dataLayout.value, writeSize.value))
export const queueAddRef = (queue: WGPUQueue): c.Void => new c.Void(lib.symbols.wgpuQueueAddRef(queue.value))
export const queueRelease = (queue: WGPUQueue): c.Void => new c.Void(lib.symbols.wgpuQueueRelease(queue.value))
export const renderBundleSetLabel = (renderBundle: WGPURenderBundle, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleSetLabel(renderBundle.value, label.value))
export const renderBundleAddRef = (renderBundle: WGPURenderBundle): c.Void => new c.Void(lib.symbols.wgpuRenderBundleAddRef(renderBundle.value))
export const renderBundleRelease = (renderBundle: WGPURenderBundle): c.Void => new c.Void(lib.symbols.wgpuRenderBundleRelease(renderBundle.value))
export const renderBundleEncoderDraw = (renderBundleEncoder: WGPURenderBundleEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDraw(renderBundleEncoder.value, vertexCount.value, instanceCount.value, firstVertex.value, firstInstance.value))
export const renderBundleEncoderDrawIndexed = (renderBundleEncoder: WGPURenderBundleEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder.value, indexCount.value, instanceCount.value, firstIndex.value, baseVertex.value, firstInstance.value))
export const renderBundleEncoderDrawIndexedIndirect = (renderBundleEncoder: WGPURenderBundleEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderBundleEncoderDrawIndirect = (renderBundleEncoder: WGPURenderBundleEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderBundleEncoderFinish = (renderBundleEncoder: WGPURenderBundleEncoder, descriptor: c.Pointer<WGPURenderBundleDescriptor>): WGPURenderBundle => new WGPURenderBundle(lib.symbols.wgpuRenderBundleEncoderFinish(renderBundleEncoder.value, descriptor.value))
export const renderBundleEncoderInsertDebugMarker = (renderBundleEncoder: WGPURenderBundleEncoder, markerLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder.value, markerLabel.value))
export const renderBundleEncoderPopDebugGroup = (renderBundleEncoder: WGPURenderBundleEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder.value))
export const renderBundleEncoderPushDebugGroup = (renderBundleEncoder: WGPURenderBundleEncoder, groupLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder.value, groupLabel.value))
export const renderBundleEncoderSetBindGroup = (renderBundleEncoder: WGPURenderBundleEncoder, groupIndex: c.U32, group: WGPUBindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder.value, groupIndex.value, group.value, dynamicOffsetCount.value, dynamicOffsets.value))
export const renderBundleEncoderSetIndexBuffer = (renderBundleEncoder: WGPURenderBundleEncoder, buffer: WGPUBuffer, format: WGPUIndexFormat, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder.value, buffer.value, format.value, offset.value, size.value))
export const renderBundleEncoderSetLabel = (renderBundleEncoder: WGPURenderBundleEncoder, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetLabel(renderBundleEncoder.value, label.value))
export const renderBundleEncoderSetPipeline = (renderBundleEncoder: WGPURenderBundleEncoder, pipeline: WGPURenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder.value, pipeline.value))
export const renderBundleEncoderSetVertexBuffer = (renderBundleEncoder: WGPURenderBundleEncoder, slot: c.U32, buffer: WGPUBuffer, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder.value, slot.value, buffer.value, offset.value, size.value))
export const renderBundleEncoderAddRef = (renderBundleEncoder: WGPURenderBundleEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderAddRef(renderBundleEncoder.value))
export const renderBundleEncoderRelease = (renderBundleEncoder: WGPURenderBundleEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderRelease(renderBundleEncoder.value))
export const renderPassEncoderBeginOcclusionQuery = (renderPassEncoder: WGPURenderPassEncoder, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder.value, queryIndex.value))
export const renderPassEncoderDraw = (renderPassEncoder: WGPURenderPassEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDraw(renderPassEncoder.value, vertexCount.value, instanceCount.value, firstVertex.value, firstInstance.value))
export const renderPassEncoderDrawIndexed = (renderPassEncoder: WGPURenderPassEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDrawIndexed(renderPassEncoder.value, indexCount.value, instanceCount.value, firstIndex.value, baseVertex.value, firstInstance.value))
export const renderPassEncoderDrawIndexedIndirect = (renderPassEncoder: WGPURenderPassEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderPassEncoderDrawIndirect = (renderPassEncoder: WGPURenderPassEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDrawIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderPassEncoderEnd = (renderPassEncoder: WGPURenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderEnd(renderPassEncoder.value))
export const renderPassEncoderEndOcclusionQuery = (renderPassEncoder: WGPURenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder.value))
export const renderPassEncoderExecuteBundles = (renderPassEncoder: WGPURenderPassEncoder, bundleCount: c.Size, bundles: c.Pointer<WGPURenderBundle>): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderExecuteBundles(renderPassEncoder.value, bundleCount.value, bundles.value))
export const renderPassEncoderInsertDebugMarker = (renderPassEncoder: WGPURenderPassEncoder, markerLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder.value, markerLabel.value))
export const renderPassEncoderMultiDrawIndexedIndirect = (renderPassEncoder: WGPURenderPassEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: WGPUBuffer, drawCountBufferOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value, maxDrawCount.value, drawCountBuffer.value, drawCountBufferOffset.value))
export const renderPassEncoderMultiDrawIndirect = (renderPassEncoder: WGPURenderPassEncoder, indirectBuffer: WGPUBuffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: WGPUBuffer, drawCountBufferOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value, maxDrawCount.value, drawCountBuffer.value, drawCountBufferOffset.value))
export const renderPassEncoderPixelLocalStorageBarrier = (renderPassEncoder: WGPURenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder.value))
export const renderPassEncoderPopDebugGroup = (renderPassEncoder: WGPURenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder.value))
export const renderPassEncoderPushDebugGroup = (renderPassEncoder: WGPURenderPassEncoder, groupLabel: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder.value, groupLabel.value))
export const renderPassEncoderSetBindGroup = (renderPassEncoder: WGPURenderPassEncoder, groupIndex: c.U32, group: WGPUBindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetBindGroup(renderPassEncoder.value, groupIndex.value, group.value, dynamicOffsetCount.value, dynamicOffsets.value))
export const renderPassEncoderSetBlendConstant = (renderPassEncoder: WGPURenderPassEncoder, color: c.Pointer<WGPUColor>): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder.value, color.value))
export const renderPassEncoderSetIndexBuffer = (renderPassEncoder: WGPURenderPassEncoder, buffer: WGPUBuffer, format: WGPUIndexFormat, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder.value, buffer.value, format.value, offset.value, size.value))
export const renderPassEncoderSetLabel = (renderPassEncoder: WGPURenderPassEncoder, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetLabel(renderPassEncoder.value, label.value))
export const renderPassEncoderSetPipeline = (renderPassEncoder: WGPURenderPassEncoder, pipeline: WGPURenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetPipeline(renderPassEncoder.value, pipeline.value))
export const renderPassEncoderSetScissorRect = (renderPassEncoder: WGPURenderPassEncoder, x: c.U32, y: c.U32, width: c.U32, height: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetScissorRect(renderPassEncoder.value, x.value, y.value, width.value, height.value))
export const renderPassEncoderSetStencilReference = (renderPassEncoder: WGPURenderPassEncoder, reference: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetStencilReference(renderPassEncoder.value, reference.value))
export const renderPassEncoderSetVertexBuffer = (renderPassEncoder: WGPURenderPassEncoder, slot: c.U32, buffer: WGPUBuffer, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder.value, slot.value, buffer.value, offset.value, size.value))
export const renderPassEncoderSetViewport = (renderPassEncoder: WGPURenderPassEncoder, x: c.F32, y: c.F32, width: c.F32, height: c.F32, minDepth: c.F32, maxDepth: c.F32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetViewport(renderPassEncoder.value, x.value, y.value, width.value, height.value, minDepth.value, maxDepth.value))
export const renderPassEncoderWriteTimestamp = (renderPassEncoder: WGPURenderPassEncoder, querySet: WGPUQuerySet, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder.value, querySet.value, queryIndex.value))
export const renderPassEncoderAddRef = (renderPassEncoder: WGPURenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderAddRef(renderPassEncoder.value))
export const renderPassEncoderRelease = (renderPassEncoder: WGPURenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderRelease(renderPassEncoder.value))
export const renderPipelineGetBindGroupLayout = (renderPipeline: WGPURenderPipeline, groupIndex: c.U32): WGPUBindGroupLayout => new WGPUBindGroupLayout(lib.symbols.wgpuRenderPipelineGetBindGroupLayout(renderPipeline.value, groupIndex.value))
export const renderPipelineSetLabel = (renderPipeline: WGPURenderPipeline, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuRenderPipelineSetLabel(renderPipeline.value, label.value))
export const renderPipelineAddRef = (renderPipeline: WGPURenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderPipelineAddRef(renderPipeline.value))
export const renderPipelineRelease = (renderPipeline: WGPURenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderPipelineRelease(renderPipeline.value))
export const samplerSetLabel = (sampler: WGPUSampler, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuSamplerSetLabel(sampler.value, label.value))
export const samplerAddRef = (sampler: WGPUSampler): c.Void => new c.Void(lib.symbols.wgpuSamplerAddRef(sampler.value))
export const samplerRelease = (sampler: WGPUSampler): c.Void => new c.Void(lib.symbols.wgpuSamplerRelease(sampler.value))
export const shaderModuleGetCompilationInfo = (shaderModule: WGPUShaderModule, callback: WGPUCompilationInfoCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuShaderModuleGetCompilationInfo(shaderModule.value, callback.value, userdata.value))
export const shaderModuleGetCompilationInfo2 = (shaderModule: WGPUShaderModule, callbackInfo: WGPUCompilationInfoCallbackInfo2): WGPUFuture => new WGPUFuture(lib.symbols.wgpuShaderModuleGetCompilationInfo2(shaderModule.value, callbackInfo.value))
export const shaderModuleGetCompilationInfoF = (shaderModule: WGPUShaderModule, callbackInfo: WGPUCompilationInfoCallbackInfo): WGPUFuture => new WGPUFuture(lib.symbols.wgpuShaderModuleGetCompilationInfoF(shaderModule.value, callbackInfo.value))
export const shaderModuleSetLabel = (shaderModule: WGPUShaderModule, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuShaderModuleSetLabel(shaderModule.value, label.value))
export const shaderModuleAddRef = (shaderModule: WGPUShaderModule): c.Void => new c.Void(lib.symbols.wgpuShaderModuleAddRef(shaderModule.value))
export const shaderModuleRelease = (shaderModule: WGPUShaderModule): c.Void => new c.Void(lib.symbols.wgpuShaderModuleRelease(shaderModule.value))
export const sharedBufferMemoryBeginAccess = (sharedBufferMemory: WGPUSharedBufferMemory, buffer: WGPUBuffer, descriptor: c.Pointer<WGPUSharedBufferMemoryBeginAccessDescriptor>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory.value, buffer.value, descriptor.value))
export const sharedBufferMemoryCreateBuffer = (sharedBufferMemory: WGPUSharedBufferMemory, descriptor: c.Pointer<WGPUBufferDescriptor>): WGPUBuffer => new WGPUBuffer(lib.symbols.wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory.value, descriptor.value))
export const sharedBufferMemoryEndAccess = (sharedBufferMemory: WGPUSharedBufferMemory, buffer: WGPUBuffer, descriptor: c.Pointer<WGPUSharedBufferMemoryEndAccessState>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSharedBufferMemoryEndAccess(sharedBufferMemory.value, buffer.value, descriptor.value))
export const sharedBufferMemoryGetProperties = (sharedBufferMemory: WGPUSharedBufferMemory, properties: c.Pointer<WGPUSharedBufferMemoryProperties>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSharedBufferMemoryGetProperties(sharedBufferMemory.value, properties.value))
export const sharedBufferMemoryIsDeviceLost = (sharedBufferMemory: WGPUSharedBufferMemory): WGPUBool => new WGPUBool(lib.symbols.wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory.value))
export const sharedBufferMemorySetLabel = (sharedBufferMemory: WGPUSharedBufferMemory, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemorySetLabel(sharedBufferMemory.value, label.value))
export const sharedBufferMemoryAddRef = (sharedBufferMemory: WGPUSharedBufferMemory): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemoryAddRef(sharedBufferMemory.value))
export const sharedBufferMemoryRelease = (sharedBufferMemory: WGPUSharedBufferMemory): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemoryRelease(sharedBufferMemory.value))
export const sharedFenceExportInfo = (sharedFence: WGPUSharedFence, info: c.Pointer<WGPUSharedFenceExportInfo>): c.Void => new c.Void(lib.symbols.wgpuSharedFenceExportInfo(sharedFence.value, info.value))
export const sharedFenceAddRef = (sharedFence: WGPUSharedFence): c.Void => new c.Void(lib.symbols.wgpuSharedFenceAddRef(sharedFence.value))
export const sharedFenceRelease = (sharedFence: WGPUSharedFence): c.Void => new c.Void(lib.symbols.wgpuSharedFenceRelease(sharedFence.value))
export const sharedTextureMemoryBeginAccess = (sharedTextureMemory: WGPUSharedTextureMemory, texture: WGPUTexture, descriptor: c.Pointer<WGPUSharedTextureMemoryBeginAccessDescriptor>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory.value, texture.value, descriptor.value))
export const sharedTextureMemoryCreateTexture = (sharedTextureMemory: WGPUSharedTextureMemory, descriptor: c.Pointer<WGPUTextureDescriptor>): WGPUTexture => new WGPUTexture(lib.symbols.wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory.value, descriptor.value))
export const sharedTextureMemoryEndAccess = (sharedTextureMemory: WGPUSharedTextureMemory, texture: WGPUTexture, descriptor: c.Pointer<WGPUSharedTextureMemoryEndAccessState>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSharedTextureMemoryEndAccess(sharedTextureMemory.value, texture.value, descriptor.value))
export const sharedTextureMemoryGetProperties = (sharedTextureMemory: WGPUSharedTextureMemory, properties: c.Pointer<WGPUSharedTextureMemoryProperties>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSharedTextureMemoryGetProperties(sharedTextureMemory.value, properties.value))
export const sharedTextureMemoryIsDeviceLost = (sharedTextureMemory: WGPUSharedTextureMemory): WGPUBool => new WGPUBool(lib.symbols.wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory.value))
export const sharedTextureMemorySetLabel = (sharedTextureMemory: WGPUSharedTextureMemory, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemorySetLabel(sharedTextureMemory.value, label.value))
export const sharedTextureMemoryAddRef = (sharedTextureMemory: WGPUSharedTextureMemory): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemoryAddRef(sharedTextureMemory.value))
export const sharedTextureMemoryRelease = (sharedTextureMemory: WGPUSharedTextureMemory): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemoryRelease(sharedTextureMemory.value))
export const surfaceConfigure = (surface: WGPUSurface, config: c.Pointer<WGPUSurfaceConfiguration>): c.Void => new c.Void(lib.symbols.wgpuSurfaceConfigure(surface.value, config.value))
export const surfaceGetCapabilities = (surface: WGPUSurface, adapter: WGPUAdapter, capabilities: c.Pointer<WGPUSurfaceCapabilities>): WGPUStatus => new WGPUStatus(lib.symbols.wgpuSurfaceGetCapabilities(surface.value, adapter.value, capabilities.value))
export const surfaceGetCurrentTexture = (surface: WGPUSurface, surfaceTexture: c.Pointer<WGPUSurfaceTexture>): c.Void => new c.Void(lib.symbols.wgpuSurfaceGetCurrentTexture(surface.value, surfaceTexture.value))
export const surfacePresent = (surface: WGPUSurface): c.Void => new c.Void(lib.symbols.wgpuSurfacePresent(surface.value))
export const surfaceSetLabel = (surface: WGPUSurface, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuSurfaceSetLabel(surface.value, label.value))
export const surfaceUnconfigure = (surface: WGPUSurface): c.Void => new c.Void(lib.symbols.wgpuSurfaceUnconfigure(surface.value))
export const surfaceAddRef = (surface: WGPUSurface): c.Void => new c.Void(lib.symbols.wgpuSurfaceAddRef(surface.value))
export const surfaceRelease = (surface: WGPUSurface): c.Void => new c.Void(lib.symbols.wgpuSurfaceRelease(surface.value))
export const textureCreateErrorView = (texture: WGPUTexture, descriptor: c.Pointer<WGPUTextureViewDescriptor>): WGPUTextureView => new WGPUTextureView(lib.symbols.wgpuTextureCreateErrorView(texture.value, descriptor.value))
export const textureCreateView = (texture: WGPUTexture, descriptor: c.Pointer<WGPUTextureViewDescriptor>): WGPUTextureView => new WGPUTextureView(lib.symbols.wgpuTextureCreateView(texture.value, descriptor.value))
export const textureDestroy = (texture: WGPUTexture): c.Void => new c.Void(lib.symbols.wgpuTextureDestroy(texture.value))
export const textureGetDepthOrArrayLayers = (texture: WGPUTexture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetDepthOrArrayLayers(texture.value))
export const textureGetDimension = (texture: WGPUTexture): WGPUTextureDimension => new WGPUTextureDimension(lib.symbols.wgpuTextureGetDimension(texture.value))
export const textureGetFormat = (texture: WGPUTexture): WGPUTextureFormat => new WGPUTextureFormat(lib.symbols.wgpuTextureGetFormat(texture.value))
export const textureGetHeight = (texture: WGPUTexture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetHeight(texture.value))
export const textureGetMipLevelCount = (texture: WGPUTexture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetMipLevelCount(texture.value))
export const textureGetSampleCount = (texture: WGPUTexture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetSampleCount(texture.value))
export const textureGetUsage = (texture: WGPUTexture): WGPUTextureUsage => new WGPUTextureUsage(lib.symbols.wgpuTextureGetUsage(texture.value))
export const textureGetWidth = (texture: WGPUTexture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetWidth(texture.value))
export const textureSetLabel = (texture: WGPUTexture, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuTextureSetLabel(texture.value, label.value))
export const textureAddRef = (texture: WGPUTexture): c.Void => new c.Void(lib.symbols.wgpuTextureAddRef(texture.value))
export const textureRelease = (texture: WGPUTexture): c.Void => new c.Void(lib.symbols.wgpuTextureRelease(texture.value))
export const textureViewSetLabel = (textureView: WGPUTextureView, label: WGPUStringView): c.Void => new c.Void(lib.symbols.wgpuTextureViewSetLabel(textureView.value, label.value))
export const textureViewAddRef = (textureView: WGPUTextureView): c.Void => new c.Void(lib.symbols.wgpuTextureViewAddRef(textureView.value))
export const textureViewRelease = (textureView: WGPUTextureView): c.Void => new c.Void(lib.symbols.wgpuTextureViewRelease(textureView.value))
