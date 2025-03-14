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
export const BufferUsage_None = new c.U32(0)
export const BufferUsage_MapRead = new c.U32(1)
export const BufferUsage_MapWrite = new c.U32(2)
export const BufferUsage_CopySrc = new c.U32(4)
export const BufferUsage_CopyDst = new c.U32(8)
export const BufferUsage_Index = new c.U32(16)
export const BufferUsage_Vertex = new c.U32(32)
export const BufferUsage_Uniform = new c.U32(64)
export const BufferUsage_Storage = new c.U32(128)
export const BufferUsage_Indirect = new c.U32(256)
export const BufferUsage_QueryResolve = new c.U32(512)
export const ColorWriteMask_None = new c.U32(0)
export const ColorWriteMask_Red = new c.U32(1)
export const ColorWriteMask_Green = new c.U32(2)
export const ColorWriteMask_Blue = new c.U32(4)
export const ColorWriteMask_Alpha = new c.U32(8)
export const ColorWriteMask_All = new c.U32(15)
export const HeapProperty_DeviceLocal = new c.U32(1)
export const HeapProperty_HostVisible = new c.U32(2)
export const HeapProperty_HostCoherent = new c.U32(4)
export const HeapProperty_HostUncached = new c.U32(8)
export const HeapProperty_HostCached = new c.U32(16)
export const MapMode_None = new c.U32(0)
export const MapMode_Read = new c.U32(1)
export const MapMode_Write = new c.U32(2)
export const ShaderStage_None = new c.U32(0)
export const ShaderStage_Vertex = new c.U32(1)
export const ShaderStage_Fragment = new c.U32(2)
export const ShaderStage_Compute = new c.U32(4)
export const TextureUsage_None = new c.U32(0)
export const TextureUsage_CopySrc = new c.U32(1)
export const TextureUsage_CopyDst = new c.U32(2)
export const TextureUsage_TextureBinding = new c.U32(4)
export const TextureUsage_StorageBinding = new c.U32(8)
export const TextureUsage_RenderAttachment = new c.U32(16)
export const TextureUsage_TransientAttachment = new c.U32(32)
export const TextureUsage_StorageAttachment = new c.U32(64)

// enums
export class WGSLFeatureName extends c.U32 {
  static 'ReadonlyAndReadwriteStorageTextures' = new WGSLFeatureName(1)
  static 'Packed4x8IntegerDotProduct' = new WGSLFeatureName(2)
  static 'UnrestrictedPointerParameters' = new WGSLFeatureName(3)
  static 'PointerCompositeAccess' = new WGSLFeatureName(4)
  static 'ChromiumTestingUnimplemented' = new WGSLFeatureName(327680)
  static 'ChromiumTestingUnsafeExperimental' = new WGSLFeatureName(327681)
  static 'ChromiumTestingExperimental' = new WGSLFeatureName(327682)
  static 'ChromiumTestingShippedWithKillswitch' = new WGSLFeatureName(327683)
  static 'ChromiumTestingShipped' = new WGSLFeatureName(327684)
  static 'Force32' = new WGSLFeatureName(2147483647)
}
export class AdapterType extends c.U32 {
  static 'DiscreteGPU' = new AdapterType(1)
  static 'IntegratedGPU' = new AdapterType(2)
  static 'CPU' = new AdapterType(3)
  static 'Unknown' = new AdapterType(4)
  static 'Force32' = new AdapterType(2147483647)
}
export class AddressMode extends c.U32 {
  static 'Undefined' = new AddressMode(0)
  static 'ClampToEdge' = new AddressMode(1)
  static 'Repeat' = new AddressMode(2)
  static 'MirrorRepeat' = new AddressMode(3)
  static 'Force32' = new AddressMode(2147483647)
}
export class AlphaMode extends c.U32 {
  static 'Opaque' = new AlphaMode(1)
  static 'Premultiplied' = new AlphaMode(2)
  static 'Unpremultiplied' = new AlphaMode(3)
  static 'Force32' = new AlphaMode(2147483647)
}
export class BackendType extends c.U32 {
  static 'Undefined' = new BackendType(0)
  static 'Null' = new BackendType(1)
  static 'WebGPU' = new BackendType(2)
  static 'D3D11' = new BackendType(3)
  static 'D3D12' = new BackendType(4)
  static 'Metal' = new BackendType(5)
  static 'Vulkan' = new BackendType(6)
  static 'OpenGL' = new BackendType(7)
  static 'OpenGLES' = new BackendType(8)
  static 'Force32' = new BackendType(2147483647)
}
export class BlendFactor extends c.U32 {
  static 'Undefined' = new BlendFactor(0)
  static 'Zero' = new BlendFactor(1)
  static 'One' = new BlendFactor(2)
  static 'Src' = new BlendFactor(3)
  static 'OneMinusSrc' = new BlendFactor(4)
  static 'SrcAlpha' = new BlendFactor(5)
  static 'OneMinusSrcAlpha' = new BlendFactor(6)
  static 'Dst' = new BlendFactor(7)
  static 'OneMinusDst' = new BlendFactor(8)
  static 'DstAlpha' = new BlendFactor(9)
  static 'OneMinusDstAlpha' = new BlendFactor(10)
  static 'SrcAlphaSaturated' = new BlendFactor(11)
  static 'Constant' = new BlendFactor(12)
  static 'OneMinusConstant' = new BlendFactor(13)
  static 'Src1' = new BlendFactor(14)
  static 'OneMinusSrc1' = new BlendFactor(15)
  static 'Src1Alpha' = new BlendFactor(16)
  static 'OneMinusSrc1Alpha' = new BlendFactor(17)
  static 'Force32' = new BlendFactor(2147483647)
}
export class BlendOperation extends c.U32 {
  static 'Undefined' = new BlendOperation(0)
  static 'Add' = new BlendOperation(1)
  static 'Subtract' = new BlendOperation(2)
  static 'ReverseSubtract' = new BlendOperation(3)
  static 'Min' = new BlendOperation(4)
  static 'Max' = new BlendOperation(5)
  static 'Force32' = new BlendOperation(2147483647)
}
export class BufferBindingType extends c.U32 {
  static 'BindingNotUsed' = new BufferBindingType(0)
  static 'Uniform' = new BufferBindingType(1)
  static 'Storage' = new BufferBindingType(2)
  static 'ReadOnlyStorage' = new BufferBindingType(3)
  static 'Force32' = new BufferBindingType(2147483647)
}
export class BufferMapAsyncStatus extends c.U32 {
  static 'Success' = new BufferMapAsyncStatus(1)
  static 'InstanceDropped' = new BufferMapAsyncStatus(2)
  static 'ValidationError' = new BufferMapAsyncStatus(3)
  static 'Unknown' = new BufferMapAsyncStatus(4)
  static 'DeviceLost' = new BufferMapAsyncStatus(5)
  static 'DestroyedBeforeCallback' = new BufferMapAsyncStatus(6)
  static 'UnmappedBeforeCallback' = new BufferMapAsyncStatus(7)
  static 'MappingAlreadyPending' = new BufferMapAsyncStatus(8)
  static 'OffsetOutOfRange' = new BufferMapAsyncStatus(9)
  static 'SizeOutOfRange' = new BufferMapAsyncStatus(10)
  static 'Force32' = new BufferMapAsyncStatus(2147483647)
}
export class BufferMapState extends c.U32 {
  static 'Unmapped' = new BufferMapState(1)
  static 'Pending' = new BufferMapState(2)
  static 'Mapped' = new BufferMapState(3)
  static 'Force32' = new BufferMapState(2147483647)
}
export class CallbackMode extends c.U32 {
  static 'WaitAnyOnly' = new CallbackMode(1)
  static 'AllowProcessEvents' = new CallbackMode(2)
  static 'AllowSpontaneous' = new CallbackMode(3)
  static 'Force32' = new CallbackMode(2147483647)
}
export class CompareFunction extends c.U32 {
  static 'Undefined' = new CompareFunction(0)
  static 'Never' = new CompareFunction(1)
  static 'Less' = new CompareFunction(2)
  static 'Equal' = new CompareFunction(3)
  static 'LessEqual' = new CompareFunction(4)
  static 'Greater' = new CompareFunction(5)
  static 'NotEqual' = new CompareFunction(6)
  static 'GreaterEqual' = new CompareFunction(7)
  static 'Always' = new CompareFunction(8)
  static 'Force32' = new CompareFunction(2147483647)
}
export class CompilationInfoRequestStatus extends c.U32 {
  static 'Success' = new CompilationInfoRequestStatus(1)
  static 'InstanceDropped' = new CompilationInfoRequestStatus(2)
  static 'Error' = new CompilationInfoRequestStatus(3)
  static 'DeviceLost' = new CompilationInfoRequestStatus(4)
  static 'Unknown' = new CompilationInfoRequestStatus(5)
  static 'Force32' = new CompilationInfoRequestStatus(2147483647)
}
export class CompilationMessageType extends c.U32 {
  static 'Error' = new CompilationMessageType(1)
  static 'Warning' = new CompilationMessageType(2)
  static 'Info' = new CompilationMessageType(3)
  static 'Force32' = new CompilationMessageType(2147483647)
}
export class CompositeAlphaMode extends c.U32 {
  static 'Auto' = new CompositeAlphaMode(0)
  static 'Opaque' = new CompositeAlphaMode(1)
  static 'Premultiplied' = new CompositeAlphaMode(2)
  static 'Unpremultiplied' = new CompositeAlphaMode(3)
  static 'Inherit' = new CompositeAlphaMode(4)
  static 'Force32' = new CompositeAlphaMode(2147483647)
}
export class CreatePipelineAsyncStatus extends c.U32 {
  static 'Success' = new CreatePipelineAsyncStatus(1)
  static 'InstanceDropped' = new CreatePipelineAsyncStatus(2)
  static 'ValidationError' = new CreatePipelineAsyncStatus(3)
  static 'InternalError' = new CreatePipelineAsyncStatus(4)
  static 'DeviceLost' = new CreatePipelineAsyncStatus(5)
  static 'DeviceDestroyed' = new CreatePipelineAsyncStatus(6)
  static 'Unknown' = new CreatePipelineAsyncStatus(7)
  static 'Force32' = new CreatePipelineAsyncStatus(2147483647)
}
export class CullMode extends c.U32 {
  static 'Undefined' = new CullMode(0)
  static 'None' = new CullMode(1)
  static 'Front' = new CullMode(2)
  static 'Back' = new CullMode(3)
  static 'Force32' = new CullMode(2147483647)
}
export class DeviceLostReason extends c.U32 {
  static 'Unknown' = new DeviceLostReason(1)
  static 'Destroyed' = new DeviceLostReason(2)
  static 'InstanceDropped' = new DeviceLostReason(3)
  static 'FailedCreation' = new DeviceLostReason(4)
  static 'Force32' = new DeviceLostReason(2147483647)
}
export class ErrorFilter extends c.U32 {
  static 'Validation' = new ErrorFilter(1)
  static 'OutOfMemory' = new ErrorFilter(2)
  static 'Internal' = new ErrorFilter(3)
  static 'Force32' = new ErrorFilter(2147483647)
}
export class ErrorType extends c.U32 {
  static 'NoError' = new ErrorType(1)
  static 'Validation' = new ErrorType(2)
  static 'OutOfMemory' = new ErrorType(3)
  static 'Internal' = new ErrorType(4)
  static 'Unknown' = new ErrorType(5)
  static 'DeviceLost' = new ErrorType(6)
  static 'Force32' = new ErrorType(2147483647)
}
export class ExternalTextureRotation extends c.U32 {
  static 'Rotate0Degrees' = new ExternalTextureRotation(1)
  static 'Rotate90Degrees' = new ExternalTextureRotation(2)
  static 'Rotate180Degrees' = new ExternalTextureRotation(3)
  static 'Rotate270Degrees' = new ExternalTextureRotation(4)
  static 'Force32' = new ExternalTextureRotation(2147483647)
}
export class FeatureLevel extends c.U32 {
  static 'Undefined' = new FeatureLevel(0)
  static 'Compatibility' = new FeatureLevel(1)
  static 'Core' = new FeatureLevel(2)
  static 'Force32' = new FeatureLevel(2147483647)
}
export class FeatureName extends c.U32 {
  static 'DepthClipControl' = new FeatureName(1)
  static 'Depth32FloatStencil8' = new FeatureName(2)
  static 'TimestampQuery' = new FeatureName(3)
  static 'TextureCompressionBC' = new FeatureName(4)
  static 'TextureCompressionETC2' = new FeatureName(5)
  static 'TextureCompressionASTC' = new FeatureName(6)
  static 'IndirectFirstInstance' = new FeatureName(7)
  static 'ShaderF16' = new FeatureName(8)
  static 'RG11B10UfloatRenderable' = new FeatureName(9)
  static 'BGRA8UnormStorage' = new FeatureName(10)
  static 'Float32Filterable' = new FeatureName(11)
  static 'Float32Blendable' = new FeatureName(12)
  static 'Subgroups' = new FeatureName(13)
  static 'SubgroupsF16' = new FeatureName(14)
  static 'DawnInternalUsages' = new FeatureName(327680)
  static 'DawnMultiPlanarFormats' = new FeatureName(327681)
  static 'DawnNative' = new FeatureName(327682)
  static 'ChromiumExperimentalTimestampQueryInsidePasses' = new FeatureName(327683)
  static 'ImplicitDeviceSynchronization' = new FeatureName(327684)
  static 'ChromiumExperimentalImmediateData' = new FeatureName(327685)
  static 'TransientAttachments' = new FeatureName(327686)
  static 'MSAARenderToSingleSampled' = new FeatureName(327687)
  static 'DualSourceBlending' = new FeatureName(327688)
  static 'D3D11MultithreadProtected' = new FeatureName(327689)
  static 'ANGLETextureSharing' = new FeatureName(327690)
  static 'PixelLocalStorageCoherent' = new FeatureName(327691)
  static 'PixelLocalStorageNonCoherent' = new FeatureName(327692)
  static 'Unorm16TextureFormats' = new FeatureName(327693)
  static 'Snorm16TextureFormats' = new FeatureName(327694)
  static 'MultiPlanarFormatExtendedUsages' = new FeatureName(327695)
  static 'MultiPlanarFormatP010' = new FeatureName(327696)
  static 'HostMappedPointer' = new FeatureName(327697)
  static 'MultiPlanarRenderTargets' = new FeatureName(327698)
  static 'MultiPlanarFormatNv12a' = new FeatureName(327699)
  static 'FramebufferFetch' = new FeatureName(327700)
  static 'BufferMapExtendedUsages' = new FeatureName(327701)
  static 'AdapterPropertiesMemoryHeaps' = new FeatureName(327702)
  static 'AdapterPropertiesD3D' = new FeatureName(327703)
  static 'AdapterPropertiesVk' = new FeatureName(327704)
  static 'R8UnormStorage' = new FeatureName(327705)
  static 'FormatCapabilities' = new FeatureName(327706)
  static 'DrmFormatCapabilities' = new FeatureName(327707)
  static 'Norm16TextureFormats' = new FeatureName(327708)
  static 'MultiPlanarFormatNv16' = new FeatureName(327709)
  static 'MultiPlanarFormatNv24' = new FeatureName(327710)
  static 'MultiPlanarFormatP210' = new FeatureName(327711)
  static 'MultiPlanarFormatP410' = new FeatureName(327712)
  static 'SharedTextureMemoryVkDedicatedAllocation' = new FeatureName(327713)
  static 'SharedTextureMemoryAHardwareBuffer' = new FeatureName(327714)
  static 'SharedTextureMemoryDmaBuf' = new FeatureName(327715)
  static 'SharedTextureMemoryOpaqueFD' = new FeatureName(327716)
  static 'SharedTextureMemoryZirconHandle' = new FeatureName(327717)
  static 'SharedTextureMemoryDXGISharedHandle' = new FeatureName(327718)
  static 'SharedTextureMemoryD3D11Texture2D' = new FeatureName(327719)
  static 'SharedTextureMemoryIOSurface' = new FeatureName(327720)
  static 'SharedTextureMemoryEGLImage' = new FeatureName(327721)
  static 'SharedFenceVkSemaphoreOpaqueFD' = new FeatureName(327722)
  static 'SharedFenceSyncFD' = new FeatureName(327723)
  static 'SharedFenceVkSemaphoreZirconHandle' = new FeatureName(327724)
  static 'SharedFenceDXGISharedHandle' = new FeatureName(327725)
  static 'SharedFenceMTLSharedEvent' = new FeatureName(327726)
  static 'SharedBufferMemoryD3D12Resource' = new FeatureName(327727)
  static 'StaticSamplers' = new FeatureName(327728)
  static 'YCbCrVulkanSamplers' = new FeatureName(327729)
  static 'ShaderModuleCompilationOptions' = new FeatureName(327730)
  static 'DawnLoadResolveTexture' = new FeatureName(327731)
  static 'DawnPartialLoadResolveTexture' = new FeatureName(327732)
  static 'MultiDrawIndirect' = new FeatureName(327733)
  static 'ClipDistances' = new FeatureName(327734)
  static 'DawnTexelCopyBufferRowAlignment' = new FeatureName(327735)
  static 'FlexibleTextureViews' = new FeatureName(327736)
  static 'Force32' = new FeatureName(2147483647)
}
export class FilterMode extends c.U32 {
  static 'Undefined' = new FilterMode(0)
  static 'Nearest' = new FilterMode(1)
  static 'Linear' = new FilterMode(2)
  static 'Force32' = new FilterMode(2147483647)
}
export class FrontFace extends c.U32 {
  static 'Undefined' = new FrontFace(0)
  static 'CCW' = new FrontFace(1)
  static 'CW' = new FrontFace(2)
  static 'Force32' = new FrontFace(2147483647)
}
export class IndexFormat extends c.U32 {
  static 'Undefined' = new IndexFormat(0)
  static 'Uint16' = new IndexFormat(1)
  static 'Uint32' = new IndexFormat(2)
  static 'Force32' = new IndexFormat(2147483647)
}
export class LoadOp extends c.U32 {
  static 'Undefined' = new LoadOp(0)
  static 'Load' = new LoadOp(1)
  static 'Clear' = new LoadOp(2)
  static 'ExpandResolveTexture' = new LoadOp(327683)
  static 'Force32' = new LoadOp(2147483647)
}
export class LoggingType extends c.U32 {
  static 'Verbose' = new LoggingType(1)
  static 'Info' = new LoggingType(2)
  static 'Warning' = new LoggingType(3)
  static 'Error' = new LoggingType(4)
  static 'Force32' = new LoggingType(2147483647)
}
export class MapAsyncStatus extends c.U32 {
  static 'Success' = new MapAsyncStatus(1)
  static 'InstanceDropped' = new MapAsyncStatus(2)
  static 'Error' = new MapAsyncStatus(3)
  static 'Aborted' = new MapAsyncStatus(4)
  static 'Unknown' = new MapAsyncStatus(5)
  static 'Force32' = new MapAsyncStatus(2147483647)
}
export class MipmapFilterMode extends c.U32 {
  static 'Undefined' = new MipmapFilterMode(0)
  static 'Nearest' = new MipmapFilterMode(1)
  static 'Linear' = new MipmapFilterMode(2)
  static 'Force32' = new MipmapFilterMode(2147483647)
}
export class OptionalBool extends c.U32 {
  static 'False' = new OptionalBool(0)
  static 'True' = new OptionalBool(1)
  static 'Undefined' = new OptionalBool(2)
  static 'Force32' = new OptionalBool(2147483647)
}
export class PopErrorScopeStatus extends c.U32 {
  static 'Success' = new PopErrorScopeStatus(1)
  static 'InstanceDropped' = new PopErrorScopeStatus(2)
  static 'Force32' = new PopErrorScopeStatus(2147483647)
}
export class PowerPreference extends c.U32 {
  static 'Undefined' = new PowerPreference(0)
  static 'LowPower' = new PowerPreference(1)
  static 'HighPerformance' = new PowerPreference(2)
  static 'Force32' = new PowerPreference(2147483647)
}
export class PresentMode extends c.U32 {
  static 'Fifo' = new PresentMode(1)
  static 'FifoRelaxed' = new PresentMode(2)
  static 'Immediate' = new PresentMode(3)
  static 'Mailbox' = new PresentMode(4)
  static 'Force32' = new PresentMode(2147483647)
}
export class PrimitiveTopology extends c.U32 {
  static 'Undefined' = new PrimitiveTopology(0)
  static 'PointList' = new PrimitiveTopology(1)
  static 'LineList' = new PrimitiveTopology(2)
  static 'LineStrip' = new PrimitiveTopology(3)
  static 'TriangleList' = new PrimitiveTopology(4)
  static 'TriangleStrip' = new PrimitiveTopology(5)
  static 'Force32' = new PrimitiveTopology(2147483647)
}
export class QueryType extends c.U32 {
  static 'Occlusion' = new QueryType(1)
  static 'Timestamp' = new QueryType(2)
  static 'Force32' = new QueryType(2147483647)
}
export class QueueWorkDoneStatus extends c.U32 {
  static 'Success' = new QueueWorkDoneStatus(1)
  static 'InstanceDropped' = new QueueWorkDoneStatus(2)
  static 'Error' = new QueueWorkDoneStatus(3)
  static 'Unknown' = new QueueWorkDoneStatus(4)
  static 'DeviceLost' = new QueueWorkDoneStatus(5)
  static 'Force32' = new QueueWorkDoneStatus(2147483647)
}
export class RequestAdapterStatus extends c.U32 {
  static 'Success' = new RequestAdapterStatus(1)
  static 'InstanceDropped' = new RequestAdapterStatus(2)
  static 'Unavailable' = new RequestAdapterStatus(3)
  static 'Error' = new RequestAdapterStatus(4)
  static 'Unknown' = new RequestAdapterStatus(5)
  static 'Force32' = new RequestAdapterStatus(2147483647)
}
export class RequestDeviceStatus extends c.U32 {
  static 'Success' = new RequestDeviceStatus(1)
  static 'InstanceDropped' = new RequestDeviceStatus(2)
  static 'Error' = new RequestDeviceStatus(3)
  static 'Unknown' = new RequestDeviceStatus(4)
  static 'Force32' = new RequestDeviceStatus(2147483647)
}
export class SType extends c.U32 {
  static 'ShaderSourceSPIRV' = new SType(1)
  static 'ShaderSourceWGSL' = new SType(2)
  static 'RenderPassMaxDrawCount' = new SType(3)
  static 'SurfaceSourceMetalLayer' = new SType(4)
  static 'SurfaceSourceWindowsHWND' = new SType(5)
  static 'SurfaceSourceXlibWindow' = new SType(6)
  static 'SurfaceSourceWaylandSurface' = new SType(7)
  static 'SurfaceSourceAndroidNativeWindow' = new SType(8)
  static 'SurfaceSourceXCBWindow' = new SType(9)
  static 'AdapterPropertiesSubgroups' = new SType(10)
  static 'TextureBindingViewDimensionDescriptor' = new SType(131072)
  static 'SurfaceSourceCanvasHTMLSelector_Emscripten' = new SType(262144)
  static 'SurfaceDescriptorFromWindowsCoreWindow' = new SType(327680)
  static 'ExternalTextureBindingEntry' = new SType(327681)
  static 'ExternalTextureBindingLayout' = new SType(327682)
  static 'SurfaceDescriptorFromWindowsSwapChainPanel' = new SType(327683)
  static 'DawnTextureInternalUsageDescriptor' = new SType(327684)
  static 'DawnEncoderInternalUsageDescriptor' = new SType(327685)
  static 'DawnInstanceDescriptor' = new SType(327686)
  static 'DawnCacheDeviceDescriptor' = new SType(327687)
  static 'DawnAdapterPropertiesPowerPreference' = new SType(327688)
  static 'DawnBufferDescriptorErrorInfoFromWireClient' = new SType(327689)
  static 'DawnTogglesDescriptor' = new SType(327690)
  static 'DawnShaderModuleSPIRVOptionsDescriptor' = new SType(327691)
  static 'RequestAdapterOptionsLUID' = new SType(327692)
  static 'RequestAdapterOptionsGetGLProc' = new SType(327693)
  static 'RequestAdapterOptionsD3D11Device' = new SType(327694)
  static 'DawnRenderPassColorAttachmentRenderToSingleSampled' = new SType(327695)
  static 'RenderPassPixelLocalStorage' = new SType(327696)
  static 'PipelineLayoutPixelLocalStorage' = new SType(327697)
  static 'BufferHostMappedPointer' = new SType(327698)
  static 'DawnExperimentalSubgroupLimits' = new SType(327699)
  static 'AdapterPropertiesMemoryHeaps' = new SType(327700)
  static 'AdapterPropertiesD3D' = new SType(327701)
  static 'AdapterPropertiesVk' = new SType(327702)
  static 'DawnWireWGSLControl' = new SType(327703)
  static 'DawnWGSLBlocklist' = new SType(327704)
  static 'DrmFormatCapabilities' = new SType(327705)
  static 'ShaderModuleCompilationOptions' = new SType(327706)
  static 'ColorTargetStateExpandResolveTextureDawn' = new SType(327707)
  static 'RenderPassDescriptorExpandResolveRect' = new SType(327708)
  static 'SharedTextureMemoryVkDedicatedAllocationDescriptor' = new SType(327709)
  static 'SharedTextureMemoryAHardwareBufferDescriptor' = new SType(327710)
  static 'SharedTextureMemoryDmaBufDescriptor' = new SType(327711)
  static 'SharedTextureMemoryOpaqueFDDescriptor' = new SType(327712)
  static 'SharedTextureMemoryZirconHandleDescriptor' = new SType(327713)
  static 'SharedTextureMemoryDXGISharedHandleDescriptor' = new SType(327714)
  static 'SharedTextureMemoryD3D11Texture2DDescriptor' = new SType(327715)
  static 'SharedTextureMemoryIOSurfaceDescriptor' = new SType(327716)
  static 'SharedTextureMemoryEGLImageDescriptor' = new SType(327717)
  static 'SharedTextureMemoryInitializedBeginState' = new SType(327718)
  static 'SharedTextureMemoryInitializedEndState' = new SType(327719)
  static 'SharedTextureMemoryVkImageLayoutBeginState' = new SType(327720)
  static 'SharedTextureMemoryVkImageLayoutEndState' = new SType(327721)
  static 'SharedTextureMemoryD3DSwapchainBeginState' = new SType(327722)
  static 'SharedFenceVkSemaphoreOpaqueFDDescriptor' = new SType(327723)
  static 'SharedFenceVkSemaphoreOpaqueFDExportInfo' = new SType(327724)
  static 'SharedFenceSyncFDDescriptor' = new SType(327725)
  static 'SharedFenceSyncFDExportInfo' = new SType(327726)
  static 'SharedFenceVkSemaphoreZirconHandleDescriptor' = new SType(327727)
  static 'SharedFenceVkSemaphoreZirconHandleExportInfo' = new SType(327728)
  static 'SharedFenceDXGISharedHandleDescriptor' = new SType(327729)
  static 'SharedFenceDXGISharedHandleExportInfo' = new SType(327730)
  static 'SharedFenceMTLSharedEventDescriptor' = new SType(327731)
  static 'SharedFenceMTLSharedEventExportInfo' = new SType(327732)
  static 'SharedBufferMemoryD3D12ResourceDescriptor' = new SType(327733)
  static 'StaticSamplerBindingLayout' = new SType(327734)
  static 'YCbCrVkDescriptor' = new SType(327735)
  static 'SharedTextureMemoryAHardwareBufferProperties' = new SType(327736)
  static 'AHardwareBufferProperties' = new SType(327737)
  static 'DawnExperimentalImmediateDataLimits' = new SType(327738)
  static 'DawnTexelCopyBufferRowAlignmentLimits' = new SType(327739)
  static 'Force32' = new SType(2147483647)
}
export class SamplerBindingType extends c.U32 {
  static 'BindingNotUsed' = new SamplerBindingType(0)
  static 'Filtering' = new SamplerBindingType(1)
  static 'NonFiltering' = new SamplerBindingType(2)
  static 'Comparison' = new SamplerBindingType(3)
  static 'Force32' = new SamplerBindingType(2147483647)
}
export class SharedFenceType extends c.U32 {
  static 'VkSemaphoreOpaqueFD' = new SharedFenceType(1)
  static 'SyncFD' = new SharedFenceType(2)
  static 'VkSemaphoreZirconHandle' = new SharedFenceType(3)
  static 'DXGISharedHandle' = new SharedFenceType(4)
  static 'MTLSharedEvent' = new SharedFenceType(5)
  static 'Force32' = new SharedFenceType(2147483647)
}
export class Status extends c.U32 {
  static 'Success' = new Status(1)
  static 'Error' = new Status(2)
  static 'Force32' = new Status(2147483647)
}
export class StencilOperation extends c.U32 {
  static 'Undefined' = new StencilOperation(0)
  static 'Keep' = new StencilOperation(1)
  static 'Zero' = new StencilOperation(2)
  static 'Replace' = new StencilOperation(3)
  static 'Invert' = new StencilOperation(4)
  static 'IncrementClamp' = new StencilOperation(5)
  static 'DecrementClamp' = new StencilOperation(6)
  static 'IncrementWrap' = new StencilOperation(7)
  static 'DecrementWrap' = new StencilOperation(8)
  static 'Force32' = new StencilOperation(2147483647)
}
export class StorageTextureAccess extends c.U32 {
  static 'BindingNotUsed' = new StorageTextureAccess(0)
  static 'WriteOnly' = new StorageTextureAccess(1)
  static 'ReadOnly' = new StorageTextureAccess(2)
  static 'ReadWrite' = new StorageTextureAccess(3)
  static 'Force32' = new StorageTextureAccess(2147483647)
}
export class StoreOp extends c.U32 {
  static 'Undefined' = new StoreOp(0)
  static 'Store' = new StoreOp(1)
  static 'Discard' = new StoreOp(2)
  static 'Force32' = new StoreOp(2147483647)
}
export class SurfaceGetCurrentTextureStatus extends c.U32 {
  static 'Success' = new SurfaceGetCurrentTextureStatus(1)
  static 'Timeout' = new SurfaceGetCurrentTextureStatus(2)
  static 'Outdated' = new SurfaceGetCurrentTextureStatus(3)
  static 'Lost' = new SurfaceGetCurrentTextureStatus(4)
  static 'OutOfMemory' = new SurfaceGetCurrentTextureStatus(5)
  static 'DeviceLost' = new SurfaceGetCurrentTextureStatus(6)
  static 'Error' = new SurfaceGetCurrentTextureStatus(7)
  static 'Force32' = new SurfaceGetCurrentTextureStatus(2147483647)
}
export class TextureAspect extends c.U32 {
  static 'Undefined' = new TextureAspect(0)
  static 'All' = new TextureAspect(1)
  static 'StencilOnly' = new TextureAspect(2)
  static 'DepthOnly' = new TextureAspect(3)
  static 'Plane0Only' = new TextureAspect(327680)
  static 'Plane1Only' = new TextureAspect(327681)
  static 'Plane2Only' = new TextureAspect(327682)
  static 'Force32' = new TextureAspect(2147483647)
}
export class TextureDimension extends c.U32 {
  static 'Undefined' = new TextureDimension(0)
  static '1D' = new TextureDimension(1)
  static '2D' = new TextureDimension(2)
  static '3D' = new TextureDimension(3)
  static 'Force32' = new TextureDimension(2147483647)
}
export class TextureFormat extends c.U32 {
  static 'Undefined' = new TextureFormat(0)
  static 'R8Unorm' = new TextureFormat(1)
  static 'R8Snorm' = new TextureFormat(2)
  static 'R8Uint' = new TextureFormat(3)
  static 'R8Sint' = new TextureFormat(4)
  static 'R16Uint' = new TextureFormat(5)
  static 'R16Sint' = new TextureFormat(6)
  static 'R16Float' = new TextureFormat(7)
  static 'RG8Unorm' = new TextureFormat(8)
  static 'RG8Snorm' = new TextureFormat(9)
  static 'RG8Uint' = new TextureFormat(10)
  static 'RG8Sint' = new TextureFormat(11)
  static 'R32Float' = new TextureFormat(12)
  static 'R32Uint' = new TextureFormat(13)
  static 'R32Sint' = new TextureFormat(14)
  static 'RG16Uint' = new TextureFormat(15)
  static 'RG16Sint' = new TextureFormat(16)
  static 'RG16Float' = new TextureFormat(17)
  static 'RGBA8Unorm' = new TextureFormat(18)
  static 'RGBA8UnormSrgb' = new TextureFormat(19)
  static 'RGBA8Snorm' = new TextureFormat(20)
  static 'RGBA8Uint' = new TextureFormat(21)
  static 'RGBA8Sint' = new TextureFormat(22)
  static 'BGRA8Unorm' = new TextureFormat(23)
  static 'BGRA8UnormSrgb' = new TextureFormat(24)
  static 'RGB10A2Uint' = new TextureFormat(25)
  static 'RGB10A2Unorm' = new TextureFormat(26)
  static 'RG11B10Ufloat' = new TextureFormat(27)
  static 'RGB9E5Ufloat' = new TextureFormat(28)
  static 'RG32Float' = new TextureFormat(29)
  static 'RG32Uint' = new TextureFormat(30)
  static 'RG32Sint' = new TextureFormat(31)
  static 'RGBA16Uint' = new TextureFormat(32)
  static 'RGBA16Sint' = new TextureFormat(33)
  static 'RGBA16Float' = new TextureFormat(34)
  static 'RGBA32Float' = new TextureFormat(35)
  static 'RGBA32Uint' = new TextureFormat(36)
  static 'RGBA32Sint' = new TextureFormat(37)
  static 'Stencil8' = new TextureFormat(38)
  static 'Depth16Unorm' = new TextureFormat(39)
  static 'Depth24Plus' = new TextureFormat(40)
  static 'Depth24PlusStencil8' = new TextureFormat(41)
  static 'Depth32Float' = new TextureFormat(42)
  static 'Depth32FloatStencil8' = new TextureFormat(43)
  static 'BC1RGBAUnorm' = new TextureFormat(44)
  static 'BC1RGBAUnormSrgb' = new TextureFormat(45)
  static 'BC2RGBAUnorm' = new TextureFormat(46)
  static 'BC2RGBAUnormSrgb' = new TextureFormat(47)
  static 'BC3RGBAUnorm' = new TextureFormat(48)
  static 'BC3RGBAUnormSrgb' = new TextureFormat(49)
  static 'BC4RUnorm' = new TextureFormat(50)
  static 'BC4RSnorm' = new TextureFormat(51)
  static 'BC5RGUnorm' = new TextureFormat(52)
  static 'BC5RGSnorm' = new TextureFormat(53)
  static 'BC6HRGBUfloat' = new TextureFormat(54)
  static 'BC6HRGBFloat' = new TextureFormat(55)
  static 'BC7RGBAUnorm' = new TextureFormat(56)
  static 'BC7RGBAUnormSrgb' = new TextureFormat(57)
  static 'ETC2RGB8Unorm' = new TextureFormat(58)
  static 'ETC2RGB8UnormSrgb' = new TextureFormat(59)
  static 'ETC2RGB8A1Unorm' = new TextureFormat(60)
  static 'ETC2RGB8A1UnormSrgb' = new TextureFormat(61)
  static 'ETC2RGBA8Unorm' = new TextureFormat(62)
  static 'ETC2RGBA8UnormSrgb' = new TextureFormat(63)
  static 'EACR11Unorm' = new TextureFormat(64)
  static 'EACR11Snorm' = new TextureFormat(65)
  static 'EACRG11Unorm' = new TextureFormat(66)
  static 'EACRG11Snorm' = new TextureFormat(67)
  static 'ASTC4x4Unorm' = new TextureFormat(68)
  static 'ASTC4x4UnormSrgb' = new TextureFormat(69)
  static 'ASTC5x4Unorm' = new TextureFormat(70)
  static 'ASTC5x4UnormSrgb' = new TextureFormat(71)
  static 'ASTC5x5Unorm' = new TextureFormat(72)
  static 'ASTC5x5UnormSrgb' = new TextureFormat(73)
  static 'ASTC6x5Unorm' = new TextureFormat(74)
  static 'ASTC6x5UnormSrgb' = new TextureFormat(75)
  static 'ASTC6x6Unorm' = new TextureFormat(76)
  static 'ASTC6x6UnormSrgb' = new TextureFormat(77)
  static 'ASTC8x5Unorm' = new TextureFormat(78)
  static 'ASTC8x5UnormSrgb' = new TextureFormat(79)
  static 'ASTC8x6Unorm' = new TextureFormat(80)
  static 'ASTC8x6UnormSrgb' = new TextureFormat(81)
  static 'ASTC8x8Unorm' = new TextureFormat(82)
  static 'ASTC8x8UnormSrgb' = new TextureFormat(83)
  static 'ASTC10x5Unorm' = new TextureFormat(84)
  static 'ASTC10x5UnormSrgb' = new TextureFormat(85)
  static 'ASTC10x6Unorm' = new TextureFormat(86)
  static 'ASTC10x6UnormSrgb' = new TextureFormat(87)
  static 'ASTC10x8Unorm' = new TextureFormat(88)
  static 'ASTC10x8UnormSrgb' = new TextureFormat(89)
  static 'ASTC10x10Unorm' = new TextureFormat(90)
  static 'ASTC10x10UnormSrgb' = new TextureFormat(91)
  static 'ASTC12x10Unorm' = new TextureFormat(92)
  static 'ASTC12x10UnormSrgb' = new TextureFormat(93)
  static 'ASTC12x12Unorm' = new TextureFormat(94)
  static 'ASTC12x12UnormSrgb' = new TextureFormat(95)
  static 'R16Unorm' = new TextureFormat(327680)
  static 'RG16Unorm' = new TextureFormat(327681)
  static 'RGBA16Unorm' = new TextureFormat(327682)
  static 'R16Snorm' = new TextureFormat(327683)
  static 'RG16Snorm' = new TextureFormat(327684)
  static 'RGBA16Snorm' = new TextureFormat(327685)
  static 'R8BG8Biplanar420Unorm' = new TextureFormat(327686)
  static 'R10X6BG10X6Biplanar420Unorm' = new TextureFormat(327687)
  static 'R8BG8A8Triplanar420Unorm' = new TextureFormat(327688)
  static 'R8BG8Biplanar422Unorm' = new TextureFormat(327689)
  static 'R8BG8Biplanar444Unorm' = new TextureFormat(327690)
  static 'R10X6BG10X6Biplanar422Unorm' = new TextureFormat(327691)
  static 'R10X6BG10X6Biplanar444Unorm' = new TextureFormat(327692)
  static 'External' = new TextureFormat(327693)
  static 'Force32' = new TextureFormat(2147483647)
}
export class TextureSampleType extends c.U32 {
  static 'BindingNotUsed' = new TextureSampleType(0)
  static 'Float' = new TextureSampleType(1)
  static 'UnfilterableFloat' = new TextureSampleType(2)
  static 'Depth' = new TextureSampleType(3)
  static 'Sint' = new TextureSampleType(4)
  static 'Uint' = new TextureSampleType(5)
  static 'Force32' = new TextureSampleType(2147483647)
}
export class TextureViewDimension extends c.U32 {
  static 'Undefined' = new TextureViewDimension(0)
  static '1D' = new TextureViewDimension(1)
  static '2D' = new TextureViewDimension(2)
  static '2DArray' = new TextureViewDimension(3)
  static 'Cube' = new TextureViewDimension(4)
  static 'CubeArray' = new TextureViewDimension(5)
  static '3D' = new TextureViewDimension(6)
  static 'Force32' = new TextureViewDimension(2147483647)
}
export class VertexFormat extends c.U32 {
  static 'Uint8' = new VertexFormat(1)
  static 'Uint8x2' = new VertexFormat(2)
  static 'Uint8x4' = new VertexFormat(3)
  static 'Sint8' = new VertexFormat(4)
  static 'Sint8x2' = new VertexFormat(5)
  static 'Sint8x4' = new VertexFormat(6)
  static 'Unorm8' = new VertexFormat(7)
  static 'Unorm8x2' = new VertexFormat(8)
  static 'Unorm8x4' = new VertexFormat(9)
  static 'Snorm8' = new VertexFormat(10)
  static 'Snorm8x2' = new VertexFormat(11)
  static 'Snorm8x4' = new VertexFormat(12)
  static 'Uint16' = new VertexFormat(13)
  static 'Uint16x2' = new VertexFormat(14)
  static 'Uint16x4' = new VertexFormat(15)
  static 'Sint16' = new VertexFormat(16)
  static 'Sint16x2' = new VertexFormat(17)
  static 'Sint16x4' = new VertexFormat(18)
  static 'Unorm16' = new VertexFormat(19)
  static 'Unorm16x2' = new VertexFormat(20)
  static 'Unorm16x4' = new VertexFormat(21)
  static 'Snorm16' = new VertexFormat(22)
  static 'Snorm16x2' = new VertexFormat(23)
  static 'Snorm16x4' = new VertexFormat(24)
  static 'Float16' = new VertexFormat(25)
  static 'Float16x2' = new VertexFormat(26)
  static 'Float16x4' = new VertexFormat(27)
  static 'Float32' = new VertexFormat(28)
  static 'Float32x2' = new VertexFormat(29)
  static 'Float32x3' = new VertexFormat(30)
  static 'Float32x4' = new VertexFormat(31)
  static 'Uint32' = new VertexFormat(32)
  static 'Uint32x2' = new VertexFormat(33)
  static 'Uint32x3' = new VertexFormat(34)
  static 'Uint32x4' = new VertexFormat(35)
  static 'Sint32' = new VertexFormat(36)
  static 'Sint32x2' = new VertexFormat(37)
  static 'Sint32x3' = new VertexFormat(38)
  static 'Sint32x4' = new VertexFormat(39)
  static 'Unorm10_10_10_2' = new VertexFormat(40)
  static 'Unorm8x4BGRA' = new VertexFormat(41)
  static 'Force32' = new VertexFormat(2147483647)
}
export class VertexStepMode extends c.U32 {
  static 'Undefined' = new VertexStepMode(0)
  static 'Vertex' = new VertexStepMode(1)
  static 'Instance' = new VertexStepMode(2)
  static 'Force32' = new VertexStepMode(2147483647)
}
export class WaitStatus extends c.U32 {
  static 'Success' = new WaitStatus(1)
  static 'TimedOut' = new WaitStatus(2)
  static 'UnsupportedTimeout' = new WaitStatus(3)
  static 'UnsupportedCount' = new WaitStatus(4)
  static 'UnsupportedMixedSources' = new WaitStatus(5)
  static 'Unknown' = new WaitStatus(6)
  static 'Force32' = new WaitStatus(2147483647)
}

// structs
export class AdapterImpl extends c.Struct<[]> {}
export class BindGroupImpl extends c.Struct<[]> {}
export class BindGroupLayoutImpl extends c.Struct<[]> {}
export class BufferImpl extends c.Struct<[]> {}
export class CommandBufferImpl extends c.Struct<[]> {}
export class CommandEncoderImpl extends c.Struct<[]> {}
export class ComputePassEncoderImpl extends c.Struct<[]> {}
export class ComputePipelineImpl extends c.Struct<[]> {}
export class DeviceImpl extends c.Struct<[]> {}
export class ExternalTextureImpl extends c.Struct<[]> {}
export class InstanceImpl extends c.Struct<[]> {}
export class PipelineLayoutImpl extends c.Struct<[]> {}
export class QuerySetImpl extends c.Struct<[]> {}
export class QueueImpl extends c.Struct<[]> {}
export class RenderBundleImpl extends c.Struct<[]> {}
export class RenderBundleEncoderImpl extends c.Struct<[]> {}
export class RenderPassEncoderImpl extends c.Struct<[]> {}
export class RenderPipelineImpl extends c.Struct<[]> {}
export class SamplerImpl extends c.Struct<[]> {}
export class ShaderModuleImpl extends c.Struct<[]> {}
export class SharedBufferMemoryImpl extends c.Struct<[]> {}
export class SharedFenceImpl extends c.Struct<[]> {}
export class SharedTextureMemoryImpl extends c.Struct<[]> {}
export class SurfaceImpl extends c.Struct<[]> {}
export class TextureImpl extends c.Struct<[]> {}
export class TextureViewImpl extends c.Struct<[]> {}
export class INTERNAL__HAVE_EMDAWNWEBGPU_HEADER extends c.Struct<[unused: Bool]> {}
export class AdapterPropertiesD3D extends c.Struct<[chain: ChainedStructOut, shaderModel: c.U32]> {}
export class AdapterPropertiesSubgroups extends c.Struct<[chain: ChainedStructOut, subgroupMinSize: c.U32, subgroupMaxSize: c.U32]> {}
export class AdapterPropertiesVk extends c.Struct<[chain: ChainedStructOut, driverVersion: c.U32]> {}
export class BindGroupEntry extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, binding: c.U32, buffer: Buffer, offset: c.U64, size: c.U64, sampler: Sampler, textureView: TextureView]> {}
export class BlendComponent extends c.Struct<[operation: BlendOperation, srcFactor: BlendFactor, dstFactor: BlendFactor]> {}
export class BufferBindingLayout extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, type: BufferBindingType, hasDynamicOffset: Bool, minBindingSize: c.U64]> {}
export class BufferHostMappedPointer extends c.Struct<[chain: ChainedStruct, pointer: c.Pointer<c.Void>, disposeCallback: Callback, userdata: c.Pointer<c.Void>]> {}
export class BufferMapCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: BufferMapCallback, userdata: c.Pointer<c.Void>]> {}
export class Color extends c.Struct<[r: c.F64, g: c.F64, b: c.F64, a: c.F64]> {}
export class ColorTargetStateExpandResolveTextureDawn extends c.Struct<[chain: ChainedStruct, enabled: Bool]> {}
export class CompilationInfoCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: CompilationInfoCallback, userdata: c.Pointer<c.Void>]> {}
export class ComputePassTimestampWrites extends c.Struct<[querySet: QuerySet, beginningOfPassWriteIndex: c.U32, endOfPassWriteIndex: c.U32]> {}
export class CopyTextureForBrowserOptions extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, flipY: Bool, needsColorSpaceConversion: Bool, srcAlphaMode: AlphaMode, srcTransferFunctionParameters: c.Pointer<c.F32>, conversionMatrix: c.Pointer<c.F32>, dstTransferFunctionParameters: c.Pointer<c.F32>, dstAlphaMode: AlphaMode, internalUsage: Bool]> {}
export class CreateComputePipelineAsyncCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: CreateComputePipelineAsyncCallback, userdata: c.Pointer<c.Void>]> {}
export class CreateRenderPipelineAsyncCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: CreateRenderPipelineAsyncCallback, userdata: c.Pointer<c.Void>]> {}
export class DawnWGSLBlocklist extends c.Struct<[chain: ChainedStruct, blocklistedFeatureCount: c.Size, blocklistedFeatures: c.Pointer<c.Pointer<c.U8>>]> {}
export class DawnAdapterPropertiesPowerPreference extends c.Struct<[chain: ChainedStructOut, powerPreference: PowerPreference]> {}
export class DawnBufferDescriptorErrorInfoFromWireClient extends c.Struct<[chain: ChainedStruct, outOfMemory: Bool]> {}
export class DawnEncoderInternalUsageDescriptor extends c.Struct<[chain: ChainedStruct, useInternalUsages: Bool]> {}
export class DawnExperimentalImmediateDataLimits extends c.Struct<[chain: ChainedStructOut, maxImmediateDataRangeByteSize: c.U32]> {}
export class DawnExperimentalSubgroupLimits extends c.Struct<[chain: ChainedStructOut, minSubgroupSize: c.U32, maxSubgroupSize: c.U32]> {}
export class DawnRenderPassColorAttachmentRenderToSingleSampled extends c.Struct<[chain: ChainedStruct, implicitSampleCount: c.U32]> {}
export class DawnShaderModuleSPIRVOptionsDescriptor extends c.Struct<[chain: ChainedStruct, allowNonUniformDerivatives: Bool]> {}
export class DawnTexelCopyBufferRowAlignmentLimits extends c.Struct<[chain: ChainedStructOut, minTexelCopyBufferRowAlignment: c.U32]> {}
export class DawnTextureInternalUsageDescriptor extends c.Struct<[chain: ChainedStruct, internalUsage: TextureUsage]> {}
export class DawnTogglesDescriptor extends c.Struct<[chain: ChainedStruct, enabledToggleCount: c.Size, enabledToggles: c.Pointer<c.Pointer<c.U8>>, disabledToggleCount: c.Size, disabledToggles: c.Pointer<c.Pointer<c.U8>>]> {}
export class DawnWireWGSLControl extends c.Struct<[chain: ChainedStruct, enableExperimental: Bool, enableUnsafe: Bool, enableTesting: Bool]> {}
export class DeviceLostCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: DeviceLostCallbackNew, userdata: c.Pointer<c.Void>]> {}
export class DrmFormatProperties extends c.Struct<[modifier: c.U64, modifierPlaneCount: c.U32]> {}
export class Extent2D extends c.Struct<[width: c.U32, height: c.U32]> {}
export class Extent3D extends c.Struct<[width: c.U32, height: c.U32, depthOrArrayLayers: c.U32]> {}
export class ExternalTextureBindingEntry extends c.Struct<[chain: ChainedStruct, externalTexture: ExternalTexture]> {}
export class ExternalTextureBindingLayout extends c.Struct<[chain: ChainedStruct]> {}
export class FormatCapabilities extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>]> {}
export class Future extends c.Struct<[id: c.U64]> {}
export class InstanceFeatures extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, timedWaitAnyEnable: Bool, timedWaitAnyMaxCount: c.Size]> {}
export class Limits extends c.Struct<[maxTextureDimension1D: c.U32, maxTextureDimension2D: c.U32, maxTextureDimension3D: c.U32, maxTextureArrayLayers: c.U32, maxBindGroups: c.U32, maxBindGroupsPlusVertexBuffers: c.U32, maxBindingsPerBindGroup: c.U32, maxDynamicUniformBuffersPerPipelineLayout: c.U32, maxDynamicStorageBuffersPerPipelineLayout: c.U32, maxSampledTexturesPerShaderStage: c.U32, maxSamplersPerShaderStage: c.U32, maxStorageBuffersPerShaderStage: c.U32, maxStorageTexturesPerShaderStage: c.U32, maxUniformBuffersPerShaderStage: c.U32, maxUniformBufferBindingSize: c.U64, maxStorageBufferBindingSize: c.U64, minUniformBufferOffsetAlignment: c.U32, minStorageBufferOffsetAlignment: c.U32, maxVertexBuffers: c.U32, maxBufferSize: c.U64, maxVertexAttributes: c.U32, maxVertexBufferArrayStride: c.U32, maxInterStageShaderComponents: c.U32, maxInterStageShaderVariables: c.U32, maxColorAttachments: c.U32, maxColorAttachmentBytesPerSample: c.U32, maxComputeWorkgroupStorageSize: c.U32, maxComputeInvocationsPerWorkgroup: c.U32, maxComputeWorkgroupSizeX: c.U32, maxComputeWorkgroupSizeY: c.U32, maxComputeWorkgroupSizeZ: c.U32, maxComputeWorkgroupsPerDimension: c.U32, maxStorageBuffersInVertexStage: c.U32, maxStorageTexturesInVertexStage: c.U32, maxStorageBuffersInFragmentStage: c.U32, maxStorageTexturesInFragmentStage: c.U32]> {}
export class MemoryHeapInfo extends c.Struct<[properties: HeapProperty, size: c.U64]> {}
export class MultisampleState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, count: c.U32, mask: c.U32, alphaToCoverageEnabled: Bool]> {}
export class Origin2D extends c.Struct<[x: c.U32, y: c.U32]> {}
export class Origin3D extends c.Struct<[x: c.U32, y: c.U32, z: c.U32]> {}
export class PipelineLayoutStorageAttachment extends c.Struct<[offset: c.U64, format: TextureFormat]> {}
export class PopErrorScopeCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: PopErrorScopeCallback, oldCallback: ErrorCallback, userdata: c.Pointer<c.Void>]> {}
export class PrimitiveState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, topology: PrimitiveTopology, stripIndexFormat: IndexFormat, frontFace: FrontFace, cullMode: CullMode, unclippedDepth: Bool]> {}
export class QueueWorkDoneCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: QueueWorkDoneCallback, userdata: c.Pointer<c.Void>]> {}
export class RenderPassDepthStencilAttachment extends c.Struct<[view: TextureView, depthLoadOp: LoadOp, depthStoreOp: StoreOp, depthClearValue: c.F32, depthReadOnly: Bool, stencilLoadOp: LoadOp, stencilStoreOp: StoreOp, stencilClearValue: c.U32, stencilReadOnly: Bool]> {}
export class RenderPassDescriptorExpandResolveRect extends c.Struct<[chain: ChainedStruct, x: c.U32, y: c.U32, width: c.U32, height: c.U32]> {}
export class RenderPassMaxDrawCount extends c.Struct<[chain: ChainedStruct, maxDrawCount: c.U64]> {}
export class RenderPassTimestampWrites extends c.Struct<[querySet: QuerySet, beginningOfPassWriteIndex: c.U32, endOfPassWriteIndex: c.U32]> {}
export class RequestAdapterCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: RequestAdapterCallback, userdata: c.Pointer<c.Void>]> {}
export class RequestAdapterOptions extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, compatibleSurface: Surface, featureLevel: FeatureLevel, powerPreference: PowerPreference, backendType: BackendType, forceFallbackAdapter: Bool, compatibilityMode: Bool]> {}
export class RequestDeviceCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: RequestDeviceCallback, userdata: c.Pointer<c.Void>]> {}
export class SamplerBindingLayout extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, type: SamplerBindingType]> {}
export class ShaderModuleCompilationOptions extends c.Struct<[chain: ChainedStruct, strictMath: Bool]> {}
export class ShaderSourceSPIRV extends c.Struct<[chain: ChainedStruct, codeSize: c.U32, code: c.Pointer<c.U32>]> {}
export class SharedBufferMemoryBeginAccessDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, initialized: Bool, fenceCount: c.Size, fences: c.Pointer<SharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class SharedBufferMemoryEndAccessState extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, initialized: Bool, fenceCount: c.Size, fences: c.Pointer<SharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class SharedBufferMemoryProperties extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, usage: BufferUsage, size: c.U64]> {}
export class SharedFenceDXGISharedHandleDescriptor extends c.Struct<[chain: ChainedStruct, handle: c.Pointer<c.Void>]> {}
export class SharedFenceDXGISharedHandleExportInfo extends c.Struct<[chain: ChainedStructOut, handle: c.Pointer<c.Void>]> {}
export class SharedFenceMTLSharedEventDescriptor extends c.Struct<[chain: ChainedStruct, sharedEvent: c.Pointer<c.Void>]> {}
export class SharedFenceMTLSharedEventExportInfo extends c.Struct<[chain: ChainedStructOut, sharedEvent: c.Pointer<c.Void>]> {}
export class SharedFenceExportInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, type: SharedFenceType]> {}
export class SharedFenceSyncFDDescriptor extends c.Struct<[chain: ChainedStruct, handle: c.I32]> {}
export class SharedFenceSyncFDExportInfo extends c.Struct<[chain: ChainedStructOut, handle: c.I32]> {}
export class SharedFenceVkSemaphoreOpaqueFDDescriptor extends c.Struct<[chain: ChainedStruct, handle: c.I32]> {}
export class SharedFenceVkSemaphoreOpaqueFDExportInfo extends c.Struct<[chain: ChainedStructOut, handle: c.I32]> {}
export class SharedFenceVkSemaphoreZirconHandleDescriptor extends c.Struct<[chain: ChainedStruct, handle: c.U32]> {}
export class SharedFenceVkSemaphoreZirconHandleExportInfo extends c.Struct<[chain: ChainedStructOut, handle: c.U32]> {}
export class SharedTextureMemoryD3DSwapchainBeginState extends c.Struct<[chain: ChainedStruct, isSwapchain: Bool]> {}
export class SharedTextureMemoryDXGISharedHandleDescriptor extends c.Struct<[chain: ChainedStruct, handle: c.Pointer<c.Void>, useKeyedMutex: Bool]> {}
export class SharedTextureMemoryEGLImageDescriptor extends c.Struct<[chain: ChainedStruct, image: c.Pointer<c.Void>]> {}
export class SharedTextureMemoryIOSurfaceDescriptor extends c.Struct<[chain: ChainedStruct, ioSurface: c.Pointer<c.Void>]> {}
export class SharedTextureMemoryAHardwareBufferDescriptor extends c.Struct<[chain: ChainedStruct, handle: c.Pointer<c.Void>, useExternalFormat: Bool]> {}
export class SharedTextureMemoryBeginAccessDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, concurrentRead: Bool, initialized: Bool, fenceCount: c.Size, fences: c.Pointer<SharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class SharedTextureMemoryDmaBufPlane extends c.Struct<[fd: c.I32, offset: c.U64, stride: c.U32]> {}
export class SharedTextureMemoryEndAccessState extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, initialized: Bool, fenceCount: c.Size, fences: c.Pointer<SharedFence>, signaledValues: c.Pointer<c.U64>]> {}
export class SharedTextureMemoryOpaqueFDDescriptor extends c.Struct<[chain: ChainedStruct, vkImageCreateInfo: c.Pointer<c.Void>, memoryFD: c.I32, memoryTypeIndex: c.U32, allocationSize: c.U64, dedicatedAllocation: Bool]> {}
export class SharedTextureMemoryVkDedicatedAllocationDescriptor extends c.Struct<[chain: ChainedStruct, dedicatedAllocation: Bool]> {}
export class SharedTextureMemoryVkImageLayoutBeginState extends c.Struct<[chain: ChainedStruct, oldLayout: c.I32, newLayout: c.I32]> {}
export class SharedTextureMemoryVkImageLayoutEndState extends c.Struct<[chain: ChainedStructOut, oldLayout: c.I32, newLayout: c.I32]> {}
export class SharedTextureMemoryZirconHandleDescriptor extends c.Struct<[chain: ChainedStruct, memoryFD: c.U32, allocationSize: c.U64]> {}
export class StaticSamplerBindingLayout extends c.Struct<[chain: ChainedStruct, sampler: Sampler, sampledTextureBinding: c.U32]> {}
export class StencilFaceState extends c.Struct<[compare: CompareFunction, failOp: StencilOperation, depthFailOp: StencilOperation, passOp: StencilOperation]> {}
export class StorageTextureBindingLayout extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, access: StorageTextureAccess, format: TextureFormat, viewDimension: TextureViewDimension]> {}
export class StringView extends c.Struct<[data: c.Pointer<c.U8>, length: c.Size]> {}
export class SupportedFeatures extends c.Struct<[featureCount: c.Size, features: c.Pointer<FeatureName>]> {}
export class SurfaceCapabilities extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, usages: TextureUsage, formatCount: c.Size, formats: c.Pointer<TextureFormat>, presentModeCount: c.Size, presentModes: c.Pointer<PresentMode>, alphaModeCount: c.Size, alphaModes: c.Pointer<CompositeAlphaMode>]> {}
export class SurfaceConfiguration extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, device: Device, format: TextureFormat, usage: TextureUsage, viewFormatCount: c.Size, viewFormats: c.Pointer<TextureFormat>, alphaMode: CompositeAlphaMode, width: c.U32, height: c.U32, presentMode: PresentMode]> {}
export class SurfaceDescriptorFromWindowsCoreWindow extends c.Struct<[chain: ChainedStruct, coreWindow: c.Pointer<c.Void>]> {}
export class SurfaceDescriptorFromWindowsSwapChainPanel extends c.Struct<[chain: ChainedStruct, swapChainPanel: c.Pointer<c.Void>]> {}
export class SurfaceSourceXCBWindow extends c.Struct<[chain: ChainedStruct, connection: c.Pointer<c.Void>, window: c.U32]> {}
export class SurfaceSourceAndroidNativeWindow extends c.Struct<[chain: ChainedStruct, window: c.Pointer<c.Void>]> {}
export class SurfaceSourceMetalLayer extends c.Struct<[chain: ChainedStruct, layer: c.Pointer<c.Void>]> {}
export class SurfaceSourceWaylandSurface extends c.Struct<[chain: ChainedStruct, display: c.Pointer<c.Void>, surface: c.Pointer<c.Void>]> {}
export class SurfaceSourceWindowsHWND extends c.Struct<[chain: ChainedStruct, hinstance: c.Pointer<c.Void>, hwnd: c.Pointer<c.Void>]> {}
export class SurfaceSourceXlibWindow extends c.Struct<[chain: ChainedStruct, display: c.Pointer<c.Void>, window: c.U64]> {}
export class SurfaceTexture extends c.Struct<[texture: Texture, suboptimal: Bool, status: SurfaceGetCurrentTextureStatus]> {}
export class TextureBindingLayout extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, sampleType: TextureSampleType, viewDimension: TextureViewDimension, multisampled: Bool]> {}
export class TextureBindingViewDimensionDescriptor extends c.Struct<[chain: ChainedStruct, textureBindingViewDimension: TextureViewDimension]> {}
export class TextureDataLayout extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, offset: c.U64, bytesPerRow: c.U32, rowsPerImage: c.U32]> {}
export class UncapturedErrorCallbackInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, callback: ErrorCallback, userdata: c.Pointer<c.Void>]> {}
export class VertexAttribute extends c.Struct<[format: VertexFormat, offset: c.U64, shaderLocation: c.U32]> {}
export class YCbCrVkDescriptor extends c.Struct<[chain: ChainedStruct, vkFormat: c.U32, vkYCbCrModel: c.U32, vkYCbCrRange: c.U32, vkComponentSwizzleRed: c.U32, vkComponentSwizzleGreen: c.U32, vkComponentSwizzleBlue: c.U32, vkComponentSwizzleAlpha: c.U32, vkXChromaOffset: c.U32, vkYChromaOffset: c.U32, vkChromaFilter: FilterMode, forceExplicitReconstruction: Bool, externalFormat: c.U64]> {}
export class AHardwareBufferProperties extends c.Struct<[yCbCrInfo: YCbCrVkDescriptor]> {}
export class AdapterInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, vendor: StringView, architecture: StringView, device: StringView, description: StringView, backendType: BackendType, adapterType: AdapterType, vendorID: c.U32, deviceID: c.U32, compatibilityMode: Bool]> {}
export class AdapterPropertiesMemoryHeaps extends c.Struct<[chain: ChainedStructOut, heapCount: c.Size, heapInfo: c.Pointer<MemoryHeapInfo>]> {}
export class BindGroupDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, layout: BindGroupLayout, entryCount: c.Size, entries: c.Pointer<BindGroupEntry>]> {}
export class BindGroupLayoutEntry extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, binding: c.U32, visibility: ShaderStage, buffer: BufferBindingLayout, sampler: SamplerBindingLayout, texture: TextureBindingLayout, storageTexture: StorageTextureBindingLayout]> {}
export class BlendState extends c.Struct<[color: BlendComponent, alpha: BlendComponent]> {}
export class BufferDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, usage: BufferUsage, size: c.U64, mappedAtCreation: Bool]> {}
export class CommandBufferDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class CommandEncoderDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class CompilationMessage extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, message: StringView, type: CompilationMessageType, lineNum: c.U64, linePos: c.U64, offset: c.U64, length: c.U64, utf16LinePos: c.U64, utf16Offset: c.U64, utf16Length: c.U64]> {}
export class ComputePassDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, timestampWrites: c.Pointer<ComputePassTimestampWrites>]> {}
export class ConstantEntry extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, key: StringView, value: c.F64]> {}
export class DawnCacheDeviceDescriptor extends c.Struct<[chain: ChainedStruct, isolationKey: StringView, loadDataFunction: DawnLoadCacheDataFunction, storeDataFunction: DawnStoreCacheDataFunction, functionUserdata: c.Pointer<c.Void>]> {}
export class DepthStencilState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, format: TextureFormat, depthWriteEnabled: OptionalBool, depthCompare: CompareFunction, stencilFront: StencilFaceState, stencilBack: StencilFaceState, stencilReadMask: c.U32, stencilWriteMask: c.U32, depthBias: c.I32, depthBiasSlopeScale: c.F32, depthBiasClamp: c.F32]> {}
export class DrmFormatCapabilities extends c.Struct<[chain: ChainedStructOut, propertiesCount: c.Size, properties: c.Pointer<DrmFormatProperties>]> {}
export class ExternalTextureDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, plane0: TextureView, plane1: TextureView, cropOrigin: Origin2D, cropSize: Extent2D, apparentSize: Extent2D, doYuvToRgbConversionOnly: Bool, yuvToRgbConversionMatrix: c.Pointer<c.F32>, srcTransferFunctionParameters: c.Pointer<c.F32>, dstTransferFunctionParameters: c.Pointer<c.F32>, gamutConversionMatrix: c.Pointer<c.F32>, mirrored: Bool, rotation: ExternalTextureRotation]> {}
export class FutureWaitInfo extends c.Struct<[future: Future, completed: Bool]> {}
export class ImageCopyBuffer extends c.Struct<[layout: TextureDataLayout, buffer: Buffer]> {}
export class ImageCopyExternalTexture extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, externalTexture: ExternalTexture, origin: Origin3D, naturalSize: Extent2D]> {}
export class ImageCopyTexture extends c.Struct<[texture: Texture, mipLevel: c.U32, origin: Origin3D, aspect: TextureAspect]> {}
export class InstanceDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, features: InstanceFeatures]> {}
export class PipelineLayoutDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, bindGroupLayoutCount: c.Size, bindGroupLayouts: c.Pointer<BindGroupLayout>, immediateDataRangeByteSize: c.U32]> {}
export class PipelineLayoutPixelLocalStorage extends c.Struct<[chain: ChainedStruct, totalPixelLocalStorageSize: c.U64, storageAttachmentCount: c.Size, storageAttachments: c.Pointer<PipelineLayoutStorageAttachment>]> {}
export class QuerySetDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, type: QueryType, count: c.U32]> {}
export class QueueDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class RenderBundleDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class RenderBundleEncoderDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, colorFormatCount: c.Size, colorFormats: c.Pointer<TextureFormat>, depthStencilFormat: TextureFormat, sampleCount: c.U32, depthReadOnly: Bool, stencilReadOnly: Bool]> {}
export class RenderPassColorAttachment extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, view: TextureView, depthSlice: c.U32, resolveTarget: TextureView, loadOp: LoadOp, storeOp: StoreOp, clearValue: Color]> {}
export class RenderPassStorageAttachment extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, offset: c.U64, storage: TextureView, loadOp: LoadOp, storeOp: StoreOp, clearValue: Color]> {}
export class RequiredLimits extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, limits: Limits]> {}
export class SamplerDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, addressModeU: AddressMode, addressModeV: AddressMode, addressModeW: AddressMode, magFilter: FilterMode, minFilter: FilterMode, mipmapFilter: MipmapFilterMode, lodMinClamp: c.F32, lodMaxClamp: c.F32, compare: CompareFunction, maxAnisotropy: c.U16]> {}
export class ShaderModuleDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class ShaderSourceWGSL extends c.Struct<[chain: ChainedStruct, code: StringView]> {}
export class SharedBufferMemoryDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class SharedFenceDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class SharedTextureMemoryAHardwareBufferProperties extends c.Struct<[chain: ChainedStructOut, yCbCrInfo: YCbCrVkDescriptor]> {}
export class SharedTextureMemoryDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class SharedTextureMemoryDmaBufDescriptor extends c.Struct<[chain: ChainedStruct, size: Extent3D, drmFormat: c.U32, drmModifier: c.U64, planeCount: c.Size, planes: c.Pointer<SharedTextureMemoryDmaBufPlane>]> {}
export class SharedTextureMemoryProperties extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, usage: TextureUsage, size: Extent3D, format: TextureFormat]> {}
export class SupportedLimits extends c.Struct<[nextInChain: c.Pointer<ChainedStructOut>, limits: Limits]> {}
export class SurfaceDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView]> {}
export class SurfaceSourceCanvasHTMLSelector_Emscripten extends c.Struct<[chain: ChainedStruct, selector: StringView]> {}
export class TextureDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, usage: TextureUsage, dimension: TextureDimension, size: Extent3D, format: TextureFormat, mipLevelCount: c.U32, sampleCount: c.U32, viewFormatCount: c.Size, viewFormats: c.Pointer<TextureFormat>]> {}
export class TextureViewDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, format: TextureFormat, dimension: TextureViewDimension, baseMipLevel: c.U32, mipLevelCount: c.U32, baseArrayLayer: c.U32, arrayLayerCount: c.U32, aspect: TextureAspect, usage: TextureUsage]> {}
export class VertexBufferLayout extends c.Struct<[arrayStride: c.U64, stepMode: VertexStepMode, attributeCount: c.Size, attributes: c.Pointer<VertexAttribute>]> {}
export class BindGroupLayoutDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, entryCount: c.Size, entries: c.Pointer<BindGroupLayoutEntry>]> {}
export class ColorTargetState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, format: TextureFormat, blend: c.Pointer<BlendState>, writeMask: ColorWriteMask]> {}
export class CompilationInfo extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, messageCount: c.Size, messages: c.Pointer<CompilationMessage>]> {}
export class ComputeState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, module: ShaderModule, entryPoint: StringView, constantCount: c.Size, constants: c.Pointer<ConstantEntry>]> {}
export class DeviceDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, requiredFeatureCount: c.Size, requiredFeatures: c.Pointer<FeatureName>, requiredLimits: c.Pointer<RequiredLimits>, defaultQueue: QueueDescriptor, deviceLostCallbackInfo2: DeviceLostCallbackInfo2, uncapturedErrorCallbackInfo2: UncapturedErrorCallbackInfo2]> {}
export class RenderPassDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, colorAttachmentCount: c.Size, colorAttachments: c.Pointer<RenderPassColorAttachment>, depthStencilAttachment: c.Pointer<RenderPassDepthStencilAttachment>, occlusionQuerySet: QuerySet, timestampWrites: c.Pointer<RenderPassTimestampWrites>]> {}
export class RenderPassPixelLocalStorage extends c.Struct<[chain: ChainedStruct, totalPixelLocalStorageSize: c.U64, storageAttachmentCount: c.Size, storageAttachments: c.Pointer<RenderPassStorageAttachment>]> {}
export class VertexState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, module: ShaderModule, entryPoint: StringView, constantCount: c.Size, constants: c.Pointer<ConstantEntry>, bufferCount: c.Size, buffers: c.Pointer<VertexBufferLayout>]> {}
export class ComputePipelineDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, layout: PipelineLayout, compute: ComputeState]> {}
export class FragmentState extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, module: ShaderModule, entryPoint: StringView, constantCount: c.Size, constants: c.Pointer<ConstantEntry>, targetCount: c.Size, targets: c.Pointer<ColorTargetState>]> {}
export class RenderPipelineDescriptor extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, label: StringView, layout: PipelineLayout, vertex: VertexState, primitive: PrimitiveState, depthStencil: c.Pointer<DepthStencilState>, multisample: MultisampleState, fragment: c.Pointer<FragmentState>]> {}
export class ChainedStruct extends c.Struct<[next: c.Pointer<ChainedStruct>, sType: SType]> {}
export class ChainedStructOut extends c.Struct<[next: c.Pointer<ChainedStructOut>, sType: SType]> {}
export class BufferMapCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: BufferMapCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class CompilationInfoCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: CompilationInfoCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class CreateComputePipelineAsyncCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: CreateComputePipelineAsyncCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class CreateRenderPipelineAsyncCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: CreateRenderPipelineAsyncCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class DeviceLostCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: DeviceLostCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class PopErrorScopeCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: PopErrorScopeCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class QueueWorkDoneCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: QueueWorkDoneCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class RequestAdapterCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: RequestAdapterCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class RequestDeviceCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, mode: CallbackMode, callback: RequestDeviceCallback2, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}
export class UncapturedErrorCallbackInfo2 extends c.Struct<[nextInChain: c.Pointer<ChainedStruct>, callback: UncapturedErrorCallback, userdata1: c.Pointer<c.Void>, userdata2: c.Pointer<c.Void>]> {}

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
export class BufferMapCallback extends c.Function {}
export class Callback extends c.Function {}
export class CompilationInfoCallback extends c.Function {}
export class CreateComputePipelineAsyncCallback extends c.Function {}
export class CreateRenderPipelineAsyncCallback extends c.Function {}
export class DawnLoadCacheDataFunction extends c.Function {}
export class DawnStoreCacheDataFunction extends c.Function {}
export class DeviceLostCallback extends c.Function {}
export class DeviceLostCallbackNew extends c.Function {}
export class ErrorCallback extends c.Function {}
export class LoggingCallback extends c.Function {}
export class PopErrorScopeCallback extends c.Function {}
export class Proc extends c.Function {}
export class QueueWorkDoneCallback extends c.Function {}
export class RequestAdapterCallback extends c.Function {}
export class RequestDeviceCallback extends c.Function {}
export class BufferMapCallback2 extends c.Function {}
export class CompilationInfoCallback2 extends c.Function {}
export class CreateComputePipelineAsyncCallback2 extends c.Function {}
export class CreateRenderPipelineAsyncCallback2 extends c.Function {}
export class DeviceLostCallback2 extends c.Function {}
export class PopErrorScopeCallback2 extends c.Function {}
export class QueueWorkDoneCallback2 extends c.Function {}
export class RequestAdapterCallback2 extends c.Function {}
export class RequestDeviceCallback2 extends c.Function {}
export class UncapturedErrorCallback extends c.Function {}
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
export class ProcAdapterInfoFreeMembers extends c.Function {}
export class ProcAdapterPropertiesMemoryHeapsFreeMembers extends c.Function {}
export class ProcCreateInstance extends c.Function {}
export class ProcDrmFormatCapabilitiesFreeMembers extends c.Function {}
export class ProcGetInstanceFeatures extends c.Function {}
export class ProcGetProcAddress extends c.Function {}
export class ProcSharedBufferMemoryEndAccessStateFreeMembers extends c.Function {}
export class ProcSharedTextureMemoryEndAccessStateFreeMembers extends c.Function {}
export class ProcSupportedFeaturesFreeMembers extends c.Function {}
export class ProcSurfaceCapabilitiesFreeMembers extends c.Function {}
export class ProcAdapterCreateDevice extends c.Function {}
export class ProcAdapterGetFeatures extends c.Function {}
export class ProcAdapterGetFormatCapabilities extends c.Function {}
export class ProcAdapterGetInfo extends c.Function {}
export class ProcAdapterGetInstance extends c.Function {}
export class ProcAdapterGetLimits extends c.Function {}
export class ProcAdapterHasFeature extends c.Function {}
export class ProcAdapterRequestDevice extends c.Function {}
export class ProcAdapterRequestDevice2 extends c.Function {}
export class ProcAdapterRequestDeviceF extends c.Function {}
export class ProcAdapterAddRef extends c.Function {}
export class ProcAdapterRelease extends c.Function {}
export class ProcBindGroupSetLabel extends c.Function {}
export class ProcBindGroupAddRef extends c.Function {}
export class ProcBindGroupRelease extends c.Function {}
export class ProcBindGroupLayoutSetLabel extends c.Function {}
export class ProcBindGroupLayoutAddRef extends c.Function {}
export class ProcBindGroupLayoutRelease extends c.Function {}
export class ProcBufferDestroy extends c.Function {}
export class ProcBufferGetConstMappedRange extends c.Function {}
export class ProcBufferGetMapState extends c.Function {}
export class ProcBufferGetMappedRange extends c.Function {}
export class ProcBufferGetSize extends c.Function {}
export class ProcBufferGetUsage extends c.Function {}
export class ProcBufferMapAsync extends c.Function {}
export class ProcBufferMapAsync2 extends c.Function {}
export class ProcBufferMapAsyncF extends c.Function {}
export class ProcBufferSetLabel extends c.Function {}
export class ProcBufferUnmap extends c.Function {}
export class ProcBufferAddRef extends c.Function {}
export class ProcBufferRelease extends c.Function {}
export class ProcCommandBufferSetLabel extends c.Function {}
export class ProcCommandBufferAddRef extends c.Function {}
export class ProcCommandBufferRelease extends c.Function {}
export class ProcCommandEncoderBeginComputePass extends c.Function {}
export class ProcCommandEncoderBeginRenderPass extends c.Function {}
export class ProcCommandEncoderClearBuffer extends c.Function {}
export class ProcCommandEncoderCopyBufferToBuffer extends c.Function {}
export class ProcCommandEncoderCopyBufferToTexture extends c.Function {}
export class ProcCommandEncoderCopyTextureToBuffer extends c.Function {}
export class ProcCommandEncoderCopyTextureToTexture extends c.Function {}
export class ProcCommandEncoderFinish extends c.Function {}
export class ProcCommandEncoderInjectValidationError extends c.Function {}
export class ProcCommandEncoderInsertDebugMarker extends c.Function {}
export class ProcCommandEncoderPopDebugGroup extends c.Function {}
export class ProcCommandEncoderPushDebugGroup extends c.Function {}
export class ProcCommandEncoderResolveQuerySet extends c.Function {}
export class ProcCommandEncoderSetLabel extends c.Function {}
export class ProcCommandEncoderWriteBuffer extends c.Function {}
export class ProcCommandEncoderWriteTimestamp extends c.Function {}
export class ProcCommandEncoderAddRef extends c.Function {}
export class ProcCommandEncoderRelease extends c.Function {}
export class ProcComputePassEncoderDispatchWorkgroups extends c.Function {}
export class ProcComputePassEncoderDispatchWorkgroupsIndirect extends c.Function {}
export class ProcComputePassEncoderEnd extends c.Function {}
export class ProcComputePassEncoderInsertDebugMarker extends c.Function {}
export class ProcComputePassEncoderPopDebugGroup extends c.Function {}
export class ProcComputePassEncoderPushDebugGroup extends c.Function {}
export class ProcComputePassEncoderSetBindGroup extends c.Function {}
export class ProcComputePassEncoderSetLabel extends c.Function {}
export class ProcComputePassEncoderSetPipeline extends c.Function {}
export class ProcComputePassEncoderWriteTimestamp extends c.Function {}
export class ProcComputePassEncoderAddRef extends c.Function {}
export class ProcComputePassEncoderRelease extends c.Function {}
export class ProcComputePipelineGetBindGroupLayout extends c.Function {}
export class ProcComputePipelineSetLabel extends c.Function {}
export class ProcComputePipelineAddRef extends c.Function {}
export class ProcComputePipelineRelease extends c.Function {}
export class ProcDeviceCreateBindGroup extends c.Function {}
export class ProcDeviceCreateBindGroupLayout extends c.Function {}
export class ProcDeviceCreateBuffer extends c.Function {}
export class ProcDeviceCreateCommandEncoder extends c.Function {}
export class ProcDeviceCreateComputePipeline extends c.Function {}
export class ProcDeviceCreateComputePipelineAsync extends c.Function {}
export class ProcDeviceCreateComputePipelineAsync2 extends c.Function {}
export class ProcDeviceCreateComputePipelineAsyncF extends c.Function {}
export class ProcDeviceCreateErrorBuffer extends c.Function {}
export class ProcDeviceCreateErrorExternalTexture extends c.Function {}
export class ProcDeviceCreateErrorShaderModule extends c.Function {}
export class ProcDeviceCreateErrorTexture extends c.Function {}
export class ProcDeviceCreateExternalTexture extends c.Function {}
export class ProcDeviceCreatePipelineLayout extends c.Function {}
export class ProcDeviceCreateQuerySet extends c.Function {}
export class ProcDeviceCreateRenderBundleEncoder extends c.Function {}
export class ProcDeviceCreateRenderPipeline extends c.Function {}
export class ProcDeviceCreateRenderPipelineAsync extends c.Function {}
export class ProcDeviceCreateRenderPipelineAsync2 extends c.Function {}
export class ProcDeviceCreateRenderPipelineAsyncF extends c.Function {}
export class ProcDeviceCreateSampler extends c.Function {}
export class ProcDeviceCreateShaderModule extends c.Function {}
export class ProcDeviceCreateTexture extends c.Function {}
export class ProcDeviceDestroy extends c.Function {}
export class ProcDeviceForceLoss extends c.Function {}
export class ProcDeviceGetAHardwareBufferProperties extends c.Function {}
export class ProcDeviceGetAdapter extends c.Function {}
export class ProcDeviceGetAdapterInfo extends c.Function {}
export class ProcDeviceGetFeatures extends c.Function {}
export class ProcDeviceGetLimits extends c.Function {}
export class ProcDeviceGetLostFuture extends c.Function {}
export class ProcDeviceGetQueue extends c.Function {}
export class ProcDeviceHasFeature extends c.Function {}
export class ProcDeviceImportSharedBufferMemory extends c.Function {}
export class ProcDeviceImportSharedFence extends c.Function {}
export class ProcDeviceImportSharedTextureMemory extends c.Function {}
export class ProcDeviceInjectError extends c.Function {}
export class ProcDevicePopErrorScope extends c.Function {}
export class ProcDevicePopErrorScope2 extends c.Function {}
export class ProcDevicePopErrorScopeF extends c.Function {}
export class ProcDevicePushErrorScope extends c.Function {}
export class ProcDeviceSetLabel extends c.Function {}
export class ProcDeviceSetLoggingCallback extends c.Function {}
export class ProcDeviceTick extends c.Function {}
export class ProcDeviceValidateTextureDescriptor extends c.Function {}
export class ProcDeviceAddRef extends c.Function {}
export class ProcDeviceRelease extends c.Function {}
export class ProcExternalTextureDestroy extends c.Function {}
export class ProcExternalTextureExpire extends c.Function {}
export class ProcExternalTextureRefresh extends c.Function {}
export class ProcExternalTextureSetLabel extends c.Function {}
export class ProcExternalTextureAddRef extends c.Function {}
export class ProcExternalTextureRelease extends c.Function {}
export class ProcInstanceCreateSurface extends c.Function {}
export class ProcInstanceEnumerateWGSLLanguageFeatures extends c.Function {}
export class ProcInstanceHasWGSLLanguageFeature extends c.Function {}
export class ProcInstanceProcessEvents extends c.Function {}
export class ProcInstanceRequestAdapter extends c.Function {}
export class ProcInstanceRequestAdapter2 extends c.Function {}
export class ProcInstanceRequestAdapterF extends c.Function {}
export class ProcInstanceWaitAny extends c.Function {}
export class ProcInstanceAddRef extends c.Function {}
export class ProcInstanceRelease extends c.Function {}
export class ProcPipelineLayoutSetLabel extends c.Function {}
export class ProcPipelineLayoutAddRef extends c.Function {}
export class ProcPipelineLayoutRelease extends c.Function {}
export class ProcQuerySetDestroy extends c.Function {}
export class ProcQuerySetGetCount extends c.Function {}
export class ProcQuerySetGetType extends c.Function {}
export class ProcQuerySetSetLabel extends c.Function {}
export class ProcQuerySetAddRef extends c.Function {}
export class ProcQuerySetRelease extends c.Function {}
export class ProcQueueCopyExternalTextureForBrowser extends c.Function {}
export class ProcQueueCopyTextureForBrowser extends c.Function {}
export class ProcQueueOnSubmittedWorkDone extends c.Function {}
export class ProcQueueOnSubmittedWorkDone2 extends c.Function {}
export class ProcQueueOnSubmittedWorkDoneF extends c.Function {}
export class ProcQueueSetLabel extends c.Function {}
export class ProcQueueSubmit extends c.Function {}
export class ProcQueueWriteBuffer extends c.Function {}
export class ProcQueueWriteTexture extends c.Function {}
export class ProcQueueAddRef extends c.Function {}
export class ProcQueueRelease extends c.Function {}
export class ProcRenderBundleSetLabel extends c.Function {}
export class ProcRenderBundleAddRef extends c.Function {}
export class ProcRenderBundleRelease extends c.Function {}
export class ProcRenderBundleEncoderDraw extends c.Function {}
export class ProcRenderBundleEncoderDrawIndexed extends c.Function {}
export class ProcRenderBundleEncoderDrawIndexedIndirect extends c.Function {}
export class ProcRenderBundleEncoderDrawIndirect extends c.Function {}
export class ProcRenderBundleEncoderFinish extends c.Function {}
export class ProcRenderBundleEncoderInsertDebugMarker extends c.Function {}
export class ProcRenderBundleEncoderPopDebugGroup extends c.Function {}
export class ProcRenderBundleEncoderPushDebugGroup extends c.Function {}
export class ProcRenderBundleEncoderSetBindGroup extends c.Function {}
export class ProcRenderBundleEncoderSetIndexBuffer extends c.Function {}
export class ProcRenderBundleEncoderSetLabel extends c.Function {}
export class ProcRenderBundleEncoderSetPipeline extends c.Function {}
export class ProcRenderBundleEncoderSetVertexBuffer extends c.Function {}
export class ProcRenderBundleEncoderAddRef extends c.Function {}
export class ProcRenderBundleEncoderRelease extends c.Function {}
export class ProcRenderPassEncoderBeginOcclusionQuery extends c.Function {}
export class ProcRenderPassEncoderDraw extends c.Function {}
export class ProcRenderPassEncoderDrawIndexed extends c.Function {}
export class ProcRenderPassEncoderDrawIndexedIndirect extends c.Function {}
export class ProcRenderPassEncoderDrawIndirect extends c.Function {}
export class ProcRenderPassEncoderEnd extends c.Function {}
export class ProcRenderPassEncoderEndOcclusionQuery extends c.Function {}
export class ProcRenderPassEncoderExecuteBundles extends c.Function {}
export class ProcRenderPassEncoderInsertDebugMarker extends c.Function {}
export class ProcRenderPassEncoderMultiDrawIndexedIndirect extends c.Function {}
export class ProcRenderPassEncoderMultiDrawIndirect extends c.Function {}
export class ProcRenderPassEncoderPixelLocalStorageBarrier extends c.Function {}
export class ProcRenderPassEncoderPopDebugGroup extends c.Function {}
export class ProcRenderPassEncoderPushDebugGroup extends c.Function {}
export class ProcRenderPassEncoderSetBindGroup extends c.Function {}
export class ProcRenderPassEncoderSetBlendConstant extends c.Function {}
export class ProcRenderPassEncoderSetIndexBuffer extends c.Function {}
export class ProcRenderPassEncoderSetLabel extends c.Function {}
export class ProcRenderPassEncoderSetPipeline extends c.Function {}
export class ProcRenderPassEncoderSetScissorRect extends c.Function {}
export class ProcRenderPassEncoderSetStencilReference extends c.Function {}
export class ProcRenderPassEncoderSetVertexBuffer extends c.Function {}
export class ProcRenderPassEncoderSetViewport extends c.Function {}
export class ProcRenderPassEncoderWriteTimestamp extends c.Function {}
export class ProcRenderPassEncoderAddRef extends c.Function {}
export class ProcRenderPassEncoderRelease extends c.Function {}
export class ProcRenderPipelineGetBindGroupLayout extends c.Function {}
export class ProcRenderPipelineSetLabel extends c.Function {}
export class ProcRenderPipelineAddRef extends c.Function {}
export class ProcRenderPipelineRelease extends c.Function {}
export class ProcSamplerSetLabel extends c.Function {}
export class ProcSamplerAddRef extends c.Function {}
export class ProcSamplerRelease extends c.Function {}
export class ProcShaderModuleGetCompilationInfo extends c.Function {}
export class ProcShaderModuleGetCompilationInfo2 extends c.Function {}
export class ProcShaderModuleGetCompilationInfoF extends c.Function {}
export class ProcShaderModuleSetLabel extends c.Function {}
export class ProcShaderModuleAddRef extends c.Function {}
export class ProcShaderModuleRelease extends c.Function {}
export class ProcSharedBufferMemoryBeginAccess extends c.Function {}
export class ProcSharedBufferMemoryCreateBuffer extends c.Function {}
export class ProcSharedBufferMemoryEndAccess extends c.Function {}
export class ProcSharedBufferMemoryGetProperties extends c.Function {}
export class ProcSharedBufferMemoryIsDeviceLost extends c.Function {}
export class ProcSharedBufferMemorySetLabel extends c.Function {}
export class ProcSharedBufferMemoryAddRef extends c.Function {}
export class ProcSharedBufferMemoryRelease extends c.Function {}
export class ProcSharedFenceExportInfo extends c.Function {}
export class ProcSharedFenceAddRef extends c.Function {}
export class ProcSharedFenceRelease extends c.Function {}
export class ProcSharedTextureMemoryBeginAccess extends c.Function {}
export class ProcSharedTextureMemoryCreateTexture extends c.Function {}
export class ProcSharedTextureMemoryEndAccess extends c.Function {}
export class ProcSharedTextureMemoryGetProperties extends c.Function {}
export class ProcSharedTextureMemoryIsDeviceLost extends c.Function {}
export class ProcSharedTextureMemorySetLabel extends c.Function {}
export class ProcSharedTextureMemoryAddRef extends c.Function {}
export class ProcSharedTextureMemoryRelease extends c.Function {}
export class ProcSurfaceConfigure extends c.Function {}
export class ProcSurfaceGetCapabilities extends c.Function {}
export class ProcSurfaceGetCurrentTexture extends c.Function {}
export class ProcSurfacePresent extends c.Function {}
export class ProcSurfaceSetLabel extends c.Function {}
export class ProcSurfaceUnconfigure extends c.Function {}
export class ProcSurfaceAddRef extends c.Function {}
export class ProcSurfaceRelease extends c.Function {}
export class ProcTextureCreateErrorView extends c.Function {}
export class ProcTextureCreateView extends c.Function {}
export class ProcTextureDestroy extends c.Function {}
export class ProcTextureGetDepthOrArrayLayers extends c.Function {}
export class ProcTextureGetDimension extends c.Function {}
export class ProcTextureGetFormat extends c.Function {}
export class ProcTextureGetHeight extends c.Function {}
export class ProcTextureGetMipLevelCount extends c.Function {}
export class ProcTextureGetSampleCount extends c.Function {}
export class ProcTextureGetUsage extends c.Function {}
export class ProcTextureGetWidth extends c.Function {}
export class ProcTextureSetLabel extends c.Function {}
export class ProcTextureAddRef extends c.Function {}
export class ProcTextureRelease extends c.Function {}
export class ProcTextureViewSetLabel extends c.Function {}
export class ProcTextureViewAddRef extends c.Function {}
export class ProcTextureViewRelease extends c.Function {}

// functions
export const adapterInfoFreeMembers = (value: AdapterInfo): c.Void => new c.Void(lib.symbols.wgpuAdapterInfoFreeMembers(value.value))
export const adapterPropertiesMemoryHeapsFreeMembers = (value: AdapterPropertiesMemoryHeaps): c.Void => new c.Void(lib.symbols.wgpuAdapterPropertiesMemoryHeapsFreeMembers(value.value))
export const createInstance = (descriptor: c.Pointer<InstanceDescriptor>): Instance => new Instance(lib.symbols.wgpuCreateInstance(descriptor.value))
export const drmFormatCapabilitiesFreeMembers = (value: DrmFormatCapabilities): c.Void => new c.Void(lib.symbols.wgpuDrmFormatCapabilitiesFreeMembers(value.value))
export const getInstanceFeatures = (features: c.Pointer<InstanceFeatures>): Status => new Status(lib.symbols.wgpuGetInstanceFeatures(features.value))
export const getProcAddress = (procName: StringView): Proc => new Proc(lib.symbols.wgpuGetProcAddress(procName.value))
export const sharedBufferMemoryEndAccessStateFreeMembers = (value: SharedBufferMemoryEndAccessState): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemoryEndAccessStateFreeMembers(value.value))
export const sharedTextureMemoryEndAccessStateFreeMembers = (value: SharedTextureMemoryEndAccessState): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemoryEndAccessStateFreeMembers(value.value))
export const supportedFeaturesFreeMembers = (value: SupportedFeatures): c.Void => new c.Void(lib.symbols.wgpuSupportedFeaturesFreeMembers(value.value))
export const surfaceCapabilitiesFreeMembers = (value: SurfaceCapabilities): c.Void => new c.Void(lib.symbols.wgpuSurfaceCapabilitiesFreeMembers(value.value))
export const adapterCreateDevice = (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>): Device => new Device(lib.symbols.wgpuAdapterCreateDevice(adapter.value, descriptor.value))
export const adapterGetFeatures = (adapter: Adapter, features: c.Pointer<SupportedFeatures>): c.Void => new c.Void(lib.symbols.wgpuAdapterGetFeatures(adapter.value, features.value))
export const adapterGetFormatCapabilities = (adapter: Adapter, format: TextureFormat, capabilities: c.Pointer<FormatCapabilities>): Status => new Status(lib.symbols.wgpuAdapterGetFormatCapabilities(adapter.value, format.value, capabilities.value))
export const adapterGetInfo = (adapter: Adapter, info: c.Pointer<AdapterInfo>): Status => new Status(lib.symbols.wgpuAdapterGetInfo(adapter.value, info.value))
export const adapterGetInstance = (adapter: Adapter): Instance => new Instance(lib.symbols.wgpuAdapterGetInstance(adapter.value))
export const adapterGetLimits = (adapter: Adapter, limits: c.Pointer<SupportedLimits>): Status => new Status(lib.symbols.wgpuAdapterGetLimits(adapter.value, limits.value))
export const adapterHasFeature = (adapter: Adapter, feature: FeatureName): Bool => new Bool(lib.symbols.wgpuAdapterHasFeature(adapter.value, feature.value))
export const adapterRequestDevice = (adapter: Adapter, descriptor: c.Pointer<DeviceDescriptor>, callback: RequestDeviceCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuAdapterRequestDevice(adapter.value, descriptor.value, callback.value, userdata.value))
export const adapterRequestDevice2 = (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo2): Future => new Future(lib.symbols.wgpuAdapterRequestDevice2(adapter.value, options.value, callbackInfo.value))
export const adapterRequestDeviceF = (adapter: Adapter, options: c.Pointer<DeviceDescriptor>, callbackInfo: RequestDeviceCallbackInfo): Future => new Future(lib.symbols.wgpuAdapterRequestDeviceF(adapter.value, options.value, callbackInfo.value))
export const adapterAddRef = (adapter: Adapter): c.Void => new c.Void(lib.symbols.wgpuAdapterAddRef(adapter.value))
export const adapterRelease = (adapter: Adapter): c.Void => new c.Void(lib.symbols.wgpuAdapterRelease(adapter.value))
export const bindGroupSetLabel = (bindGroup: BindGroup, label: StringView): c.Void => new c.Void(lib.symbols.wgpuBindGroupSetLabel(bindGroup.value, label.value))
export const bindGroupAddRef = (bindGroup: BindGroup): c.Void => new c.Void(lib.symbols.wgpuBindGroupAddRef(bindGroup.value))
export const bindGroupRelease = (bindGroup: BindGroup): c.Void => new c.Void(lib.symbols.wgpuBindGroupRelease(bindGroup.value))
export const bindGroupLayoutSetLabel = (bindGroupLayout: BindGroupLayout, label: StringView): c.Void => new c.Void(lib.symbols.wgpuBindGroupLayoutSetLabel(bindGroupLayout.value, label.value))
export const bindGroupLayoutAddRef = (bindGroupLayout: BindGroupLayout): c.Void => new c.Void(lib.symbols.wgpuBindGroupLayoutAddRef(bindGroupLayout.value))
export const bindGroupLayoutRelease = (bindGroupLayout: BindGroupLayout): c.Void => new c.Void(lib.symbols.wgpuBindGroupLayoutRelease(bindGroupLayout.value))
export const bufferDestroy = (buffer: Buffer): c.Void => new c.Void(lib.symbols.wgpuBufferDestroy(buffer.value))
export const bufferGetConstMappedRange = (buffer: Buffer, offset: c.Size, size: c.Size): c.Pointer<c.Void> => new c.Pointer<c.Void>(lib.symbols.wgpuBufferGetConstMappedRange(buffer.value, offset.value, size.value))
export const bufferGetMapState = (buffer: Buffer): BufferMapState => new BufferMapState(lib.symbols.wgpuBufferGetMapState(buffer.value))
export const bufferGetMappedRange = (buffer: Buffer, offset: c.Size, size: c.Size): c.Pointer<c.Void> => new c.Pointer<c.Void>(lib.symbols.wgpuBufferGetMappedRange(buffer.value, offset.value, size.value))
export const bufferGetSize = (buffer: Buffer): c.U64 => new c.U64(lib.symbols.wgpuBufferGetSize(buffer.value))
export const bufferGetUsage = (buffer: Buffer): BufferUsage => new BufferUsage(lib.symbols.wgpuBufferGetUsage(buffer.value))
export const bufferMapAsync = (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callback: BufferMapCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuBufferMapAsync(buffer.value, mode.value, offset.value, size.value, callback.value, userdata.value))
export const bufferMapAsync2 = (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo2): Future => new Future(lib.symbols.wgpuBufferMapAsync2(buffer.value, mode.value, offset.value, size.value, callbackInfo.value))
export const bufferMapAsyncF = (buffer: Buffer, mode: MapMode, offset: c.Size, size: c.Size, callbackInfo: BufferMapCallbackInfo): Future => new Future(lib.symbols.wgpuBufferMapAsyncF(buffer.value, mode.value, offset.value, size.value, callbackInfo.value))
export const bufferSetLabel = (buffer: Buffer, label: StringView): c.Void => new c.Void(lib.symbols.wgpuBufferSetLabel(buffer.value, label.value))
export const bufferUnmap = (buffer: Buffer): c.Void => new c.Void(lib.symbols.wgpuBufferUnmap(buffer.value))
export const bufferAddRef = (buffer: Buffer): c.Void => new c.Void(lib.symbols.wgpuBufferAddRef(buffer.value))
export const bufferRelease = (buffer: Buffer): c.Void => new c.Void(lib.symbols.wgpuBufferRelease(buffer.value))
export const commandBufferSetLabel = (commandBuffer: CommandBuffer, label: StringView): c.Void => new c.Void(lib.symbols.wgpuCommandBufferSetLabel(commandBuffer.value, label.value))
export const commandBufferAddRef = (commandBuffer: CommandBuffer): c.Void => new c.Void(lib.symbols.wgpuCommandBufferAddRef(commandBuffer.value))
export const commandBufferRelease = (commandBuffer: CommandBuffer): c.Void => new c.Void(lib.symbols.wgpuCommandBufferRelease(commandBuffer.value))
export const commandEncoderBeginComputePass = (commandEncoder: CommandEncoder, descriptor: c.Pointer<ComputePassDescriptor>): ComputePassEncoder => new ComputePassEncoder(lib.symbols.wgpuCommandEncoderBeginComputePass(commandEncoder.value, descriptor.value))
export const commandEncoderBeginRenderPass = (commandEncoder: CommandEncoder, descriptor: c.Pointer<RenderPassDescriptor>): RenderPassEncoder => new RenderPassEncoder(lib.symbols.wgpuCommandEncoderBeginRenderPass(commandEncoder.value, descriptor.value))
export const commandEncoderClearBuffer = (commandEncoder: CommandEncoder, buffer: Buffer, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderClearBuffer(commandEncoder.value, buffer.value, offset.value, size.value))
export const commandEncoderCopyBufferToBuffer = (commandEncoder: CommandEncoder, source: Buffer, sourceOffset: c.U64, destination: Buffer, destinationOffset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyBufferToBuffer(commandEncoder.value, source.value, sourceOffset.value, destination.value, destinationOffset.value, size.value))
export const commandEncoderCopyBufferToTexture = (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyBuffer>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyBufferToTexture(commandEncoder.value, source.value, destination.value, copySize.value))
export const commandEncoderCopyTextureToBuffer = (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyBuffer>, copySize: c.Pointer<Extent3D>): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyTextureToBuffer(commandEncoder.value, source.value, destination.value, copySize.value))
export const commandEncoderCopyTextureToTexture = (commandEncoder: CommandEncoder, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderCopyTextureToTexture(commandEncoder.value, source.value, destination.value, copySize.value))
export const commandEncoderFinish = (commandEncoder: CommandEncoder, descriptor: c.Pointer<CommandBufferDescriptor>): CommandBuffer => new CommandBuffer(lib.symbols.wgpuCommandEncoderFinish(commandEncoder.value, descriptor.value))
export const commandEncoderInjectValidationError = (commandEncoder: CommandEncoder, message: StringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderInjectValidationError(commandEncoder.value, message.value))
export const commandEncoderInsertDebugMarker = (commandEncoder: CommandEncoder, markerLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderInsertDebugMarker(commandEncoder.value, markerLabel.value))
export const commandEncoderPopDebugGroup = (commandEncoder: CommandEncoder): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderPopDebugGroup(commandEncoder.value))
export const commandEncoderPushDebugGroup = (commandEncoder: CommandEncoder, groupLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderPushDebugGroup(commandEncoder.value, groupLabel.value))
export const commandEncoderResolveQuerySet = (commandEncoder: CommandEncoder, querySet: QuerySet, firstQuery: c.U32, queryCount: c.U32, destination: Buffer, destinationOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderResolveQuerySet(commandEncoder.value, querySet.value, firstQuery.value, queryCount.value, destination.value, destinationOffset.value))
export const commandEncoderSetLabel = (commandEncoder: CommandEncoder, label: StringView): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderSetLabel(commandEncoder.value, label.value))
export const commandEncoderWriteBuffer = (commandEncoder: CommandEncoder, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.U8>, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderWriteBuffer(commandEncoder.value, buffer.value, bufferOffset.value, data.value, size.value))
export const commandEncoderWriteTimestamp = (commandEncoder: CommandEncoder, querySet: QuerySet, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderWriteTimestamp(commandEncoder.value, querySet.value, queryIndex.value))
export const commandEncoderAddRef = (commandEncoder: CommandEncoder): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderAddRef(commandEncoder.value))
export const commandEncoderRelease = (commandEncoder: CommandEncoder): c.Void => new c.Void(lib.symbols.wgpuCommandEncoderRelease(commandEncoder.value))
export const computePassEncoderDispatchWorkgroups = (computePassEncoder: ComputePassEncoder, workgroupCountX: c.U32, workgroupCountY: c.U32, workgroupCountZ: c.U32): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder.value, workgroupCountX.value, workgroupCountY.value, workgroupCountZ.value))
export const computePassEncoderDispatchWorkgroupsIndirect = (computePassEncoder: ComputePassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder.value, indirectBuffer.value, indirectOffset.value))
export const computePassEncoderEnd = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderEnd(computePassEncoder.value))
export const computePassEncoderInsertDebugMarker = (computePassEncoder: ComputePassEncoder, markerLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderInsertDebugMarker(computePassEncoder.value, markerLabel.value))
export const computePassEncoderPopDebugGroup = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderPopDebugGroup(computePassEncoder.value))
export const computePassEncoderPushDebugGroup = (computePassEncoder: ComputePassEncoder, groupLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderPushDebugGroup(computePassEncoder.value, groupLabel.value))
export const computePassEncoderSetBindGroup = (computePassEncoder: ComputePassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderSetBindGroup(computePassEncoder.value, groupIndex.value, group.value, dynamicOffsetCount.value, dynamicOffsets.value))
export const computePassEncoderSetLabel = (computePassEncoder: ComputePassEncoder, label: StringView): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderSetLabel(computePassEncoder.value, label.value))
export const computePassEncoderSetPipeline = (computePassEncoder: ComputePassEncoder, pipeline: ComputePipeline): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderSetPipeline(computePassEncoder.value, pipeline.value))
export const computePassEncoderWriteTimestamp = (computePassEncoder: ComputePassEncoder, querySet: QuerySet, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderWriteTimestamp(computePassEncoder.value, querySet.value, queryIndex.value))
export const computePassEncoderAddRef = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderAddRef(computePassEncoder.value))
export const computePassEncoderRelease = (computePassEncoder: ComputePassEncoder): c.Void => new c.Void(lib.symbols.wgpuComputePassEncoderRelease(computePassEncoder.value))
export const computePipelineGetBindGroupLayout = (computePipeline: ComputePipeline, groupIndex: c.U32): BindGroupLayout => new BindGroupLayout(lib.symbols.wgpuComputePipelineGetBindGroupLayout(computePipeline.value, groupIndex.value))
export const computePipelineSetLabel = (computePipeline: ComputePipeline, label: StringView): c.Void => new c.Void(lib.symbols.wgpuComputePipelineSetLabel(computePipeline.value, label.value))
export const computePipelineAddRef = (computePipeline: ComputePipeline): c.Void => new c.Void(lib.symbols.wgpuComputePipelineAddRef(computePipeline.value))
export const computePipelineRelease = (computePipeline: ComputePipeline): c.Void => new c.Void(lib.symbols.wgpuComputePipelineRelease(computePipeline.value))
export const deviceCreateBindGroup = (device: Device, descriptor: c.Pointer<BindGroupDescriptor>): BindGroup => new BindGroup(lib.symbols.wgpuDeviceCreateBindGroup(device.value, descriptor.value))
export const deviceCreateBindGroupLayout = (device: Device, descriptor: c.Pointer<BindGroupLayoutDescriptor>): BindGroupLayout => new BindGroupLayout(lib.symbols.wgpuDeviceCreateBindGroupLayout(device.value, descriptor.value))
export const deviceCreateBuffer = (device: Device, descriptor: c.Pointer<BufferDescriptor>): Buffer => new Buffer(lib.symbols.wgpuDeviceCreateBuffer(device.value, descriptor.value))
export const deviceCreateCommandEncoder = (device: Device, descriptor: c.Pointer<CommandEncoderDescriptor>): CommandEncoder => new CommandEncoder(lib.symbols.wgpuDeviceCreateCommandEncoder(device.value, descriptor.value))
export const deviceCreateComputePipeline = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>): ComputePipeline => new ComputePipeline(lib.symbols.wgpuDeviceCreateComputePipeline(device.value, descriptor.value))
export const deviceCreateComputePipelineAsync = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callback: CreateComputePipelineAsyncCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDeviceCreateComputePipelineAsync(device.value, descriptor.value, callback.value, userdata.value))
export const deviceCreateComputePipelineAsync2 = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo2): Future => new Future(lib.symbols.wgpuDeviceCreateComputePipelineAsync2(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateComputePipelineAsyncF = (device: Device, descriptor: c.Pointer<ComputePipelineDescriptor>, callbackInfo: CreateComputePipelineAsyncCallbackInfo): Future => new Future(lib.symbols.wgpuDeviceCreateComputePipelineAsyncF(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateErrorBuffer = (device: Device, descriptor: c.Pointer<BufferDescriptor>): Buffer => new Buffer(lib.symbols.wgpuDeviceCreateErrorBuffer(device.value, descriptor.value))
export const deviceCreateErrorExternalTexture = (device: Device): ExternalTexture => new ExternalTexture(lib.symbols.wgpuDeviceCreateErrorExternalTexture(device.value))
export const deviceCreateErrorShaderModule = (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>, errorMessage: StringView): ShaderModule => new ShaderModule(lib.symbols.wgpuDeviceCreateErrorShaderModule(device.value, descriptor.value, errorMessage.value))
export const deviceCreateErrorTexture = (device: Device, descriptor: c.Pointer<TextureDescriptor>): Texture => new Texture(lib.symbols.wgpuDeviceCreateErrorTexture(device.value, descriptor.value))
export const deviceCreateExternalTexture = (device: Device, externalTextureDescriptor: c.Pointer<ExternalTextureDescriptor>): ExternalTexture => new ExternalTexture(lib.symbols.wgpuDeviceCreateExternalTexture(device.value, externalTextureDescriptor.value))
export const deviceCreatePipelineLayout = (device: Device, descriptor: c.Pointer<PipelineLayoutDescriptor>): PipelineLayout => new PipelineLayout(lib.symbols.wgpuDeviceCreatePipelineLayout(device.value, descriptor.value))
export const deviceCreateQuerySet = (device: Device, descriptor: c.Pointer<QuerySetDescriptor>): QuerySet => new QuerySet(lib.symbols.wgpuDeviceCreateQuerySet(device.value, descriptor.value))
export const deviceCreateRenderBundleEncoder = (device: Device, descriptor: c.Pointer<RenderBundleEncoderDescriptor>): RenderBundleEncoder => new RenderBundleEncoder(lib.symbols.wgpuDeviceCreateRenderBundleEncoder(device.value, descriptor.value))
export const deviceCreateRenderPipeline = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>): RenderPipeline => new RenderPipeline(lib.symbols.wgpuDeviceCreateRenderPipeline(device.value, descriptor.value))
export const deviceCreateRenderPipelineAsync = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callback: CreateRenderPipelineAsyncCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDeviceCreateRenderPipelineAsync(device.value, descriptor.value, callback.value, userdata.value))
export const deviceCreateRenderPipelineAsync2 = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo2): Future => new Future(lib.symbols.wgpuDeviceCreateRenderPipelineAsync2(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateRenderPipelineAsyncF = (device: Device, descriptor: c.Pointer<RenderPipelineDescriptor>, callbackInfo: CreateRenderPipelineAsyncCallbackInfo): Future => new Future(lib.symbols.wgpuDeviceCreateRenderPipelineAsyncF(device.value, descriptor.value, callbackInfo.value))
export const deviceCreateSampler = (device: Device, descriptor: c.Pointer<SamplerDescriptor>): Sampler => new Sampler(lib.symbols.wgpuDeviceCreateSampler(device.value, descriptor.value))
export const deviceCreateShaderModule = (device: Device, descriptor: c.Pointer<ShaderModuleDescriptor>): ShaderModule => new ShaderModule(lib.symbols.wgpuDeviceCreateShaderModule(device.value, descriptor.value))
export const deviceCreateTexture = (device: Device, descriptor: c.Pointer<TextureDescriptor>): Texture => new Texture(lib.symbols.wgpuDeviceCreateTexture(device.value, descriptor.value))
export const deviceDestroy = (device: Device): c.Void => new c.Void(lib.symbols.wgpuDeviceDestroy(device.value))
export const deviceForceLoss = (device: Device, type: DeviceLostReason, message: StringView): c.Void => new c.Void(lib.symbols.wgpuDeviceForceLoss(device.value, type.value, message.value))
export const deviceGetAHardwareBufferProperties = (device: Device, handle: c.Pointer<c.Void>, properties: c.Pointer<AHardwareBufferProperties>): Status => new Status(lib.symbols.wgpuDeviceGetAHardwareBufferProperties(device.value, handle.value, properties.value))
export const deviceGetAdapter = (device: Device): Adapter => new Adapter(lib.symbols.wgpuDeviceGetAdapter(device.value))
export const deviceGetAdapterInfo = (device: Device, adapterInfo: c.Pointer<AdapterInfo>): Status => new Status(lib.symbols.wgpuDeviceGetAdapterInfo(device.value, adapterInfo.value))
export const deviceGetFeatures = (device: Device, features: c.Pointer<SupportedFeatures>): c.Void => new c.Void(lib.symbols.wgpuDeviceGetFeatures(device.value, features.value))
export const deviceGetLimits = (device: Device, limits: c.Pointer<SupportedLimits>): Status => new Status(lib.symbols.wgpuDeviceGetLimits(device.value, limits.value))
export const deviceGetLostFuture = (device: Device): Future => new Future(lib.symbols.wgpuDeviceGetLostFuture(device.value))
export const deviceGetQueue = (device: Device): Queue => new Queue(lib.symbols.wgpuDeviceGetQueue(device.value))
export const deviceHasFeature = (device: Device, feature: FeatureName): Bool => new Bool(lib.symbols.wgpuDeviceHasFeature(device.value, feature.value))
export const deviceImportSharedBufferMemory = (device: Device, descriptor: c.Pointer<SharedBufferMemoryDescriptor>): SharedBufferMemory => new SharedBufferMemory(lib.symbols.wgpuDeviceImportSharedBufferMemory(device.value, descriptor.value))
export const deviceImportSharedFence = (device: Device, descriptor: c.Pointer<SharedFenceDescriptor>): SharedFence => new SharedFence(lib.symbols.wgpuDeviceImportSharedFence(device.value, descriptor.value))
export const deviceImportSharedTextureMemory = (device: Device, descriptor: c.Pointer<SharedTextureMemoryDescriptor>): SharedTextureMemory => new SharedTextureMemory(lib.symbols.wgpuDeviceImportSharedTextureMemory(device.value, descriptor.value))
export const deviceInjectError = (device: Device, type: ErrorType, message: StringView): c.Void => new c.Void(lib.symbols.wgpuDeviceInjectError(device.value, type.value, message.value))
export const devicePopErrorScope = (device: Device, oldCallback: ErrorCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDevicePopErrorScope(device.value, oldCallback.value, userdata.value))
export const devicePopErrorScope2 = (device: Device, callbackInfo: PopErrorScopeCallbackInfo2): Future => new Future(lib.symbols.wgpuDevicePopErrorScope2(device.value, callbackInfo.value))
export const devicePopErrorScopeF = (device: Device, callbackInfo: PopErrorScopeCallbackInfo): Future => new Future(lib.symbols.wgpuDevicePopErrorScopeF(device.value, callbackInfo.value))
export const devicePushErrorScope = (device: Device, filter: ErrorFilter): c.Void => new c.Void(lib.symbols.wgpuDevicePushErrorScope(device.value, filter.value))
export const deviceSetLabel = (device: Device, label: StringView): c.Void => new c.Void(lib.symbols.wgpuDeviceSetLabel(device.value, label.value))
export const deviceSetLoggingCallback = (device: Device, callback: LoggingCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuDeviceSetLoggingCallback(device.value, callback.value, userdata.value))
export const deviceTick = (device: Device): c.Void => new c.Void(lib.symbols.wgpuDeviceTick(device.value))
export const deviceValidateTextureDescriptor = (device: Device, descriptor: c.Pointer<TextureDescriptor>): c.Void => new c.Void(lib.symbols.wgpuDeviceValidateTextureDescriptor(device.value, descriptor.value))
export const deviceAddRef = (device: Device): c.Void => new c.Void(lib.symbols.wgpuDeviceAddRef(device.value))
export const deviceRelease = (device: Device): c.Void => new c.Void(lib.symbols.wgpuDeviceRelease(device.value))
export const externalTextureDestroy = (externalTexture: ExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureDestroy(externalTexture.value))
export const externalTextureExpire = (externalTexture: ExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureExpire(externalTexture.value))
export const externalTextureRefresh = (externalTexture: ExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureRefresh(externalTexture.value))
export const externalTextureSetLabel = (externalTexture: ExternalTexture, label: StringView): c.Void => new c.Void(lib.symbols.wgpuExternalTextureSetLabel(externalTexture.value, label.value))
export const externalTextureAddRef = (externalTexture: ExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureAddRef(externalTexture.value))
export const externalTextureRelease = (externalTexture: ExternalTexture): c.Void => new c.Void(lib.symbols.wgpuExternalTextureRelease(externalTexture.value))
export const instanceCreateSurface = (instance: Instance, descriptor: c.Pointer<SurfaceDescriptor>): Surface => new Surface(lib.symbols.wgpuInstanceCreateSurface(instance.value, descriptor.value))
export const instanceEnumerateWGSLLanguageFeatures = (instance: Instance, features: c.Pointer<WGSLFeatureName>): c.Size => new c.Size(lib.symbols.wgpuInstanceEnumerateWGSLLanguageFeatures(instance.value, features.value))
export const instanceHasWGSLLanguageFeature = (instance: Instance, feature: WGSLFeatureName): Bool => new Bool(lib.symbols.wgpuInstanceHasWGSLLanguageFeature(instance.value, feature.value))
export const instanceProcessEvents = (instance: Instance): c.Void => new c.Void(lib.symbols.wgpuInstanceProcessEvents(instance.value))
export const instanceRequestAdapter = (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callback: RequestAdapterCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuInstanceRequestAdapter(instance.value, options.value, callback.value, userdata.value))
export const instanceRequestAdapter2 = (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo2): Future => new Future(lib.symbols.wgpuInstanceRequestAdapter2(instance.value, options.value, callbackInfo.value))
export const instanceRequestAdapterF = (instance: Instance, options: c.Pointer<RequestAdapterOptions>, callbackInfo: RequestAdapterCallbackInfo): Future => new Future(lib.symbols.wgpuInstanceRequestAdapterF(instance.value, options.value, callbackInfo.value))
export const instanceWaitAny = (instance: Instance, futureCount: c.Size, futures: c.Pointer<FutureWaitInfo>, timeoutNS: c.U64): WaitStatus => new WaitStatus(lib.symbols.wgpuInstanceWaitAny(instance.value, futureCount.value, futures.value, timeoutNS.value))
export const instanceAddRef = (instance: Instance): c.Void => new c.Void(lib.symbols.wgpuInstanceAddRef(instance.value))
export const instanceRelease = (instance: Instance): c.Void => new c.Void(lib.symbols.wgpuInstanceRelease(instance.value))
export const pipelineLayoutSetLabel = (pipelineLayout: PipelineLayout, label: StringView): c.Void => new c.Void(lib.symbols.wgpuPipelineLayoutSetLabel(pipelineLayout.value, label.value))
export const pipelineLayoutAddRef = (pipelineLayout: PipelineLayout): c.Void => new c.Void(lib.symbols.wgpuPipelineLayoutAddRef(pipelineLayout.value))
export const pipelineLayoutRelease = (pipelineLayout: PipelineLayout): c.Void => new c.Void(lib.symbols.wgpuPipelineLayoutRelease(pipelineLayout.value))
export const querySetDestroy = (querySet: QuerySet): c.Void => new c.Void(lib.symbols.wgpuQuerySetDestroy(querySet.value))
export const querySetGetCount = (querySet: QuerySet): c.U32 => new c.U32(lib.symbols.wgpuQuerySetGetCount(querySet.value))
export const querySetGetType = (querySet: QuerySet): QueryType => new QueryType(lib.symbols.wgpuQuerySetGetType(querySet.value))
export const querySetSetLabel = (querySet: QuerySet, label: StringView): c.Void => new c.Void(lib.symbols.wgpuQuerySetSetLabel(querySet.value, label.value))
export const querySetAddRef = (querySet: QuerySet): c.Void => new c.Void(lib.symbols.wgpuQuerySetAddRef(querySet.value))
export const querySetRelease = (querySet: QuerySet): c.Void => new c.Void(lib.symbols.wgpuQuerySetRelease(querySet.value))
export const queueCopyExternalTextureForBrowser = (queue: Queue, source: c.Pointer<ImageCopyExternalTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>): c.Void => new c.Void(lib.symbols.wgpuQueueCopyExternalTextureForBrowser(queue.value, source.value, destination.value, copySize.value, options.value))
export const queueCopyTextureForBrowser = (queue: Queue, source: c.Pointer<ImageCopyTexture>, destination: c.Pointer<ImageCopyTexture>, copySize: c.Pointer<Extent3D>, options: c.Pointer<CopyTextureForBrowserOptions>): c.Void => new c.Void(lib.symbols.wgpuQueueCopyTextureForBrowser(queue.value, source.value, destination.value, copySize.value, options.value))
export const queueOnSubmittedWorkDone = (queue: Queue, callback: QueueWorkDoneCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuQueueOnSubmittedWorkDone(queue.value, callback.value, userdata.value))
export const queueOnSubmittedWorkDone2 = (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo2): Future => new Future(lib.symbols.wgpuQueueOnSubmittedWorkDone2(queue.value, callbackInfo.value))
export const queueOnSubmittedWorkDoneF = (queue: Queue, callbackInfo: QueueWorkDoneCallbackInfo): Future => new Future(lib.symbols.wgpuQueueOnSubmittedWorkDoneF(queue.value, callbackInfo.value))
export const queueSetLabel = (queue: Queue, label: StringView): c.Void => new c.Void(lib.symbols.wgpuQueueSetLabel(queue.value, label.value))
export const queueSubmit = (queue: Queue, commandCount: c.Size, commands: c.Pointer<CommandBuffer>): c.Void => new c.Void(lib.symbols.wgpuQueueSubmit(queue.value, commandCount.value, commands.value))
export const queueWriteBuffer = (queue: Queue, buffer: Buffer, bufferOffset: c.U64, data: c.Pointer<c.Void>, size: c.Size): c.Void => new c.Void(lib.symbols.wgpuQueueWriteBuffer(queue.value, buffer.value, bufferOffset.value, data.value, size.value))
export const queueWriteTexture = (queue: Queue, destination: c.Pointer<ImageCopyTexture>, data: c.Pointer<c.Void>, dataSize: c.Size, dataLayout: c.Pointer<TextureDataLayout>, writeSize: c.Pointer<Extent3D>): c.Void => new c.Void(lib.symbols.wgpuQueueWriteTexture(queue.value, destination.value, data.value, dataSize.value, dataLayout.value, writeSize.value))
export const queueAddRef = (queue: Queue): c.Void => new c.Void(lib.symbols.wgpuQueueAddRef(queue.value))
export const queueRelease = (queue: Queue): c.Void => new c.Void(lib.symbols.wgpuQueueRelease(queue.value))
export const renderBundleSetLabel = (renderBundle: RenderBundle, label: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleSetLabel(renderBundle.value, label.value))
export const renderBundleAddRef = (renderBundle: RenderBundle): c.Void => new c.Void(lib.symbols.wgpuRenderBundleAddRef(renderBundle.value))
export const renderBundleRelease = (renderBundle: RenderBundle): c.Void => new c.Void(lib.symbols.wgpuRenderBundleRelease(renderBundle.value))
export const renderBundleEncoderDraw = (renderBundleEncoder: RenderBundleEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDraw(renderBundleEncoder.value, vertexCount.value, instanceCount.value, firstVertex.value, firstInstance.value))
export const renderBundleEncoderDrawIndexed = (renderBundleEncoder: RenderBundleEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder.value, indexCount.value, instanceCount.value, firstIndex.value, baseVertex.value, firstInstance.value))
export const renderBundleEncoderDrawIndexedIndirect = (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderBundleEncoderDrawIndirect = (renderBundleEncoder: RenderBundleEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderBundleEncoderFinish = (renderBundleEncoder: RenderBundleEncoder, descriptor: c.Pointer<RenderBundleDescriptor>): RenderBundle => new RenderBundle(lib.symbols.wgpuRenderBundleEncoderFinish(renderBundleEncoder.value, descriptor.value))
export const renderBundleEncoderInsertDebugMarker = (renderBundleEncoder: RenderBundleEncoder, markerLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder.value, markerLabel.value))
export const renderBundleEncoderPopDebugGroup = (renderBundleEncoder: RenderBundleEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder.value))
export const renderBundleEncoderPushDebugGroup = (renderBundleEncoder: RenderBundleEncoder, groupLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder.value, groupLabel.value))
export const renderBundleEncoderSetBindGroup = (renderBundleEncoder: RenderBundleEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder.value, groupIndex.value, group.value, dynamicOffsetCount.value, dynamicOffsets.value))
export const renderBundleEncoderSetIndexBuffer = (renderBundleEncoder: RenderBundleEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder.value, buffer.value, format.value, offset.value, size.value))
export const renderBundleEncoderSetLabel = (renderBundleEncoder: RenderBundleEncoder, label: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetLabel(renderBundleEncoder.value, label.value))
export const renderBundleEncoderSetPipeline = (renderBundleEncoder: RenderBundleEncoder, pipeline: RenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder.value, pipeline.value))
export const renderBundleEncoderSetVertexBuffer = (renderBundleEncoder: RenderBundleEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder.value, slot.value, buffer.value, offset.value, size.value))
export const renderBundleEncoderAddRef = (renderBundleEncoder: RenderBundleEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderAddRef(renderBundleEncoder.value))
export const renderBundleEncoderRelease = (renderBundleEncoder: RenderBundleEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderBundleEncoderRelease(renderBundleEncoder.value))
export const renderPassEncoderBeginOcclusionQuery = (renderPassEncoder: RenderPassEncoder, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder.value, queryIndex.value))
export const renderPassEncoderDraw = (renderPassEncoder: RenderPassEncoder, vertexCount: c.U32, instanceCount: c.U32, firstVertex: c.U32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDraw(renderPassEncoder.value, vertexCount.value, instanceCount.value, firstVertex.value, firstInstance.value))
export const renderPassEncoderDrawIndexed = (renderPassEncoder: RenderPassEncoder, indexCount: c.U32, instanceCount: c.U32, firstIndex: c.U32, baseVertex: c.I32, firstInstance: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDrawIndexed(renderPassEncoder.value, indexCount.value, instanceCount.value, firstIndex.value, baseVertex.value, firstInstance.value))
export const renderPassEncoderDrawIndexedIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderPassEncoderDrawIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderDrawIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value))
export const renderPassEncoderEnd = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderEnd(renderPassEncoder.value))
export const renderPassEncoderEndOcclusionQuery = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder.value))
export const renderPassEncoderExecuteBundles = (renderPassEncoder: RenderPassEncoder, bundleCount: c.Size, bundles: c.Pointer<RenderBundle>): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderExecuteBundles(renderPassEncoder.value, bundleCount.value, bundles.value))
export const renderPassEncoderInsertDebugMarker = (renderPassEncoder: RenderPassEncoder, markerLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder.value, markerLabel.value))
export const renderPassEncoderMultiDrawIndexedIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value, maxDrawCount.value, drawCountBuffer.value, drawCountBufferOffset.value))
export const renderPassEncoderMultiDrawIndirect = (renderPassEncoder: RenderPassEncoder, indirectBuffer: Buffer, indirectOffset: c.U64, maxDrawCount: c.U32, drawCountBuffer: Buffer, drawCountBufferOffset: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder.value, indirectBuffer.value, indirectOffset.value, maxDrawCount.value, drawCountBuffer.value, drawCountBufferOffset.value))
export const renderPassEncoderPixelLocalStorageBarrier = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder.value))
export const renderPassEncoderPopDebugGroup = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder.value))
export const renderPassEncoderPushDebugGroup = (renderPassEncoder: RenderPassEncoder, groupLabel: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder.value, groupLabel.value))
export const renderPassEncoderSetBindGroup = (renderPassEncoder: RenderPassEncoder, groupIndex: c.U32, group: BindGroup, dynamicOffsetCount: c.Size, dynamicOffsets: c.Pointer<c.U32>): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetBindGroup(renderPassEncoder.value, groupIndex.value, group.value, dynamicOffsetCount.value, dynamicOffsets.value))
export const renderPassEncoderSetBlendConstant = (renderPassEncoder: RenderPassEncoder, color: c.Pointer<Color>): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder.value, color.value))
export const renderPassEncoderSetIndexBuffer = (renderPassEncoder: RenderPassEncoder, buffer: Buffer, format: IndexFormat, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder.value, buffer.value, format.value, offset.value, size.value))
export const renderPassEncoderSetLabel = (renderPassEncoder: RenderPassEncoder, label: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetLabel(renderPassEncoder.value, label.value))
export const renderPassEncoderSetPipeline = (renderPassEncoder: RenderPassEncoder, pipeline: RenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetPipeline(renderPassEncoder.value, pipeline.value))
export const renderPassEncoderSetScissorRect = (renderPassEncoder: RenderPassEncoder, x: c.U32, y: c.U32, width: c.U32, height: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetScissorRect(renderPassEncoder.value, x.value, y.value, width.value, height.value))
export const renderPassEncoderSetStencilReference = (renderPassEncoder: RenderPassEncoder, reference: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetStencilReference(renderPassEncoder.value, reference.value))
export const renderPassEncoderSetVertexBuffer = (renderPassEncoder: RenderPassEncoder, slot: c.U32, buffer: Buffer, offset: c.U64, size: c.U64): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder.value, slot.value, buffer.value, offset.value, size.value))
export const renderPassEncoderSetViewport = (renderPassEncoder: RenderPassEncoder, x: c.F32, y: c.F32, width: c.F32, height: c.F32, minDepth: c.F32, maxDepth: c.F32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderSetViewport(renderPassEncoder.value, x.value, y.value, width.value, height.value, minDepth.value, maxDepth.value))
export const renderPassEncoderWriteTimestamp = (renderPassEncoder: RenderPassEncoder, querySet: QuerySet, queryIndex: c.U32): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder.value, querySet.value, queryIndex.value))
export const renderPassEncoderAddRef = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderAddRef(renderPassEncoder.value))
export const renderPassEncoderRelease = (renderPassEncoder: RenderPassEncoder): c.Void => new c.Void(lib.symbols.wgpuRenderPassEncoderRelease(renderPassEncoder.value))
export const renderPipelineGetBindGroupLayout = (renderPipeline: RenderPipeline, groupIndex: c.U32): BindGroupLayout => new BindGroupLayout(lib.symbols.wgpuRenderPipelineGetBindGroupLayout(renderPipeline.value, groupIndex.value))
export const renderPipelineSetLabel = (renderPipeline: RenderPipeline, label: StringView): c.Void => new c.Void(lib.symbols.wgpuRenderPipelineSetLabel(renderPipeline.value, label.value))
export const renderPipelineAddRef = (renderPipeline: RenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderPipelineAddRef(renderPipeline.value))
export const renderPipelineRelease = (renderPipeline: RenderPipeline): c.Void => new c.Void(lib.symbols.wgpuRenderPipelineRelease(renderPipeline.value))
export const samplerSetLabel = (sampler: Sampler, label: StringView): c.Void => new c.Void(lib.symbols.wgpuSamplerSetLabel(sampler.value, label.value))
export const samplerAddRef = (sampler: Sampler): c.Void => new c.Void(lib.symbols.wgpuSamplerAddRef(sampler.value))
export const samplerRelease = (sampler: Sampler): c.Void => new c.Void(lib.symbols.wgpuSamplerRelease(sampler.value))
export const shaderModuleGetCompilationInfo = (shaderModule: ShaderModule, callback: CompilationInfoCallback, userdata: c.Pointer<c.Void>): c.Void => new c.Void(lib.symbols.wgpuShaderModuleGetCompilationInfo(shaderModule.value, callback.value, userdata.value))
export const shaderModuleGetCompilationInfo2 = (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo2): Future => new Future(lib.symbols.wgpuShaderModuleGetCompilationInfo2(shaderModule.value, callbackInfo.value))
export const shaderModuleGetCompilationInfoF = (shaderModule: ShaderModule, callbackInfo: CompilationInfoCallbackInfo): Future => new Future(lib.symbols.wgpuShaderModuleGetCompilationInfoF(shaderModule.value, callbackInfo.value))
export const shaderModuleSetLabel = (shaderModule: ShaderModule, label: StringView): c.Void => new c.Void(lib.symbols.wgpuShaderModuleSetLabel(shaderModule.value, label.value))
export const shaderModuleAddRef = (shaderModule: ShaderModule): c.Void => new c.Void(lib.symbols.wgpuShaderModuleAddRef(shaderModule.value))
export const shaderModuleRelease = (shaderModule: ShaderModule): c.Void => new c.Void(lib.symbols.wgpuShaderModuleRelease(shaderModule.value))
export const sharedBufferMemoryBeginAccess = (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryBeginAccessDescriptor>): Status => new Status(lib.symbols.wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory.value, buffer.value, descriptor.value))
export const sharedBufferMemoryCreateBuffer = (sharedBufferMemory: SharedBufferMemory, descriptor: c.Pointer<BufferDescriptor>): Buffer => new Buffer(lib.symbols.wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory.value, descriptor.value))
export const sharedBufferMemoryEndAccess = (sharedBufferMemory: SharedBufferMemory, buffer: Buffer, descriptor: c.Pointer<SharedBufferMemoryEndAccessState>): Status => new Status(lib.symbols.wgpuSharedBufferMemoryEndAccess(sharedBufferMemory.value, buffer.value, descriptor.value))
export const sharedBufferMemoryGetProperties = (sharedBufferMemory: SharedBufferMemory, properties: c.Pointer<SharedBufferMemoryProperties>): Status => new Status(lib.symbols.wgpuSharedBufferMemoryGetProperties(sharedBufferMemory.value, properties.value))
export const sharedBufferMemoryIsDeviceLost = (sharedBufferMemory: SharedBufferMemory): Bool => new Bool(lib.symbols.wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory.value))
export const sharedBufferMemorySetLabel = (sharedBufferMemory: SharedBufferMemory, label: StringView): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemorySetLabel(sharedBufferMemory.value, label.value))
export const sharedBufferMemoryAddRef = (sharedBufferMemory: SharedBufferMemory): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemoryAddRef(sharedBufferMemory.value))
export const sharedBufferMemoryRelease = (sharedBufferMemory: SharedBufferMemory): c.Void => new c.Void(lib.symbols.wgpuSharedBufferMemoryRelease(sharedBufferMemory.value))
export const sharedFenceExportInfo = (sharedFence: SharedFence, info: c.Pointer<SharedFenceExportInfo>): c.Void => new c.Void(lib.symbols.wgpuSharedFenceExportInfo(sharedFence.value, info.value))
export const sharedFenceAddRef = (sharedFence: SharedFence): c.Void => new c.Void(lib.symbols.wgpuSharedFenceAddRef(sharedFence.value))
export const sharedFenceRelease = (sharedFence: SharedFence): c.Void => new c.Void(lib.symbols.wgpuSharedFenceRelease(sharedFence.value))
export const sharedTextureMemoryBeginAccess = (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryBeginAccessDescriptor>): Status => new Status(lib.symbols.wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory.value, texture.value, descriptor.value))
export const sharedTextureMemoryCreateTexture = (sharedTextureMemory: SharedTextureMemory, descriptor: c.Pointer<TextureDescriptor>): Texture => new Texture(lib.symbols.wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory.value, descriptor.value))
export const sharedTextureMemoryEndAccess = (sharedTextureMemory: SharedTextureMemory, texture: Texture, descriptor: c.Pointer<SharedTextureMemoryEndAccessState>): Status => new Status(lib.symbols.wgpuSharedTextureMemoryEndAccess(sharedTextureMemory.value, texture.value, descriptor.value))
export const sharedTextureMemoryGetProperties = (sharedTextureMemory: SharedTextureMemory, properties: c.Pointer<SharedTextureMemoryProperties>): Status => new Status(lib.symbols.wgpuSharedTextureMemoryGetProperties(sharedTextureMemory.value, properties.value))
export const sharedTextureMemoryIsDeviceLost = (sharedTextureMemory: SharedTextureMemory): Bool => new Bool(lib.symbols.wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory.value))
export const sharedTextureMemorySetLabel = (sharedTextureMemory: SharedTextureMemory, label: StringView): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemorySetLabel(sharedTextureMemory.value, label.value))
export const sharedTextureMemoryAddRef = (sharedTextureMemory: SharedTextureMemory): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemoryAddRef(sharedTextureMemory.value))
export const sharedTextureMemoryRelease = (sharedTextureMemory: SharedTextureMemory): c.Void => new c.Void(lib.symbols.wgpuSharedTextureMemoryRelease(sharedTextureMemory.value))
export const surfaceConfigure = (surface: Surface, config: c.Pointer<SurfaceConfiguration>): c.Void => new c.Void(lib.symbols.wgpuSurfaceConfigure(surface.value, config.value))
export const surfaceGetCapabilities = (surface: Surface, adapter: Adapter, capabilities: c.Pointer<SurfaceCapabilities>): Status => new Status(lib.symbols.wgpuSurfaceGetCapabilities(surface.value, adapter.value, capabilities.value))
export const surfaceGetCurrentTexture = (surface: Surface, surfaceTexture: c.Pointer<SurfaceTexture>): c.Void => new c.Void(lib.symbols.wgpuSurfaceGetCurrentTexture(surface.value, surfaceTexture.value))
export const surfacePresent = (surface: Surface): c.Void => new c.Void(lib.symbols.wgpuSurfacePresent(surface.value))
export const surfaceSetLabel = (surface: Surface, label: StringView): c.Void => new c.Void(lib.symbols.wgpuSurfaceSetLabel(surface.value, label.value))
export const surfaceUnconfigure = (surface: Surface): c.Void => new c.Void(lib.symbols.wgpuSurfaceUnconfigure(surface.value))
export const surfaceAddRef = (surface: Surface): c.Void => new c.Void(lib.symbols.wgpuSurfaceAddRef(surface.value))
export const surfaceRelease = (surface: Surface): c.Void => new c.Void(lib.symbols.wgpuSurfaceRelease(surface.value))
export const textureCreateErrorView = (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>): TextureView => new TextureView(lib.symbols.wgpuTextureCreateErrorView(texture.value, descriptor.value))
export const textureCreateView = (texture: Texture, descriptor: c.Pointer<TextureViewDescriptor>): TextureView => new TextureView(lib.symbols.wgpuTextureCreateView(texture.value, descriptor.value))
export const textureDestroy = (texture: Texture): c.Void => new c.Void(lib.symbols.wgpuTextureDestroy(texture.value))
export const textureGetDepthOrArrayLayers = (texture: Texture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetDepthOrArrayLayers(texture.value))
export const textureGetDimension = (texture: Texture): TextureDimension => new TextureDimension(lib.symbols.wgpuTextureGetDimension(texture.value))
export const textureGetFormat = (texture: Texture): TextureFormat => new TextureFormat(lib.symbols.wgpuTextureGetFormat(texture.value))
export const textureGetHeight = (texture: Texture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetHeight(texture.value))
export const textureGetMipLevelCount = (texture: Texture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetMipLevelCount(texture.value))
export const textureGetSampleCount = (texture: Texture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetSampleCount(texture.value))
export const textureGetUsage = (texture: Texture): TextureUsage => new TextureUsage(lib.symbols.wgpuTextureGetUsage(texture.value))
export const textureGetWidth = (texture: Texture): c.U32 => new c.U32(lib.symbols.wgpuTextureGetWidth(texture.value))
export const textureSetLabel = (texture: Texture, label: StringView): c.Void => new c.Void(lib.symbols.wgpuTextureSetLabel(texture.value, label.value))
export const textureAddRef = (texture: Texture): c.Void => new c.Void(lib.symbols.wgpuTextureAddRef(texture.value))
export const textureRelease = (texture: Texture): c.Void => new c.Void(lib.symbols.wgpuTextureRelease(texture.value))
export const textureViewSetLabel = (textureView: TextureView, label: StringView): c.Void => new c.Void(lib.symbols.wgpuTextureViewSetLabel(textureView.value, label.value))
export const textureViewAddRef = (textureView: TextureView): c.Void => new c.Void(lib.symbols.wgpuTextureViewAddRef(textureView.value))
export const textureViewRelease = (textureView: TextureView): c.Void => new c.Void(lib.symbols.wgpuTextureViewRelease(textureView.value))
