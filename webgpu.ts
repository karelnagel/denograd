// DL
export const webgpu = Deno.dlopen('path', {
  emscripten_webgpu_get_device: { parameters: [], result: 'pointer'  },
  wgpuAdapterInfoFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuAdapterPropertiesMemoryHeapsFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuCreateInstance: { parameters: ['pointer'], result: 'pointer'  },
  wgpuDrmFormatCapabilitiesFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuGetInstanceFeatures: { parameters: ['pointer'], result: 'pointer'  },
  wgpuGetProcAddress: { parameters: ['pointer'], result: 'pointer'  },
  wgpuSharedBufferMemoryEndAccessStateFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedTextureMemoryEndAccessStateFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuSupportedFeaturesFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuSurfaceCapabilitiesFreeMembers: { parameters: ['pointer'], result: 'void'  },
  wgpuAdapterCreateDevice: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterGetFeatures: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuAdapterGetFormatCapabilities: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterGetInfo: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterGetInstance: { parameters: ['pointer'], result: 'pointer'  },
  wgpuAdapterGetLimits: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterHasFeature: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterRequestDevice: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuAdapterRequestDevice2: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterRequestDeviceF: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuAdapterAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuAdapterRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuBindGroupSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuBindGroupAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuBindGroupRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuBindGroupLayoutSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuBindGroupLayoutAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuBindGroupLayoutRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuBufferDestroy: { parameters: ['pointer'], result: 'void'  },
  wgpuBufferGetConstMappedRange: { parameters: ['pointer', 'isize', 'isize'], result: 'pointer'  },
  wgpuBufferGetMapState: { parameters: ['pointer'], result: 'pointer'  },
  wgpuBufferGetMappedRange: { parameters: ['pointer', 'isize', 'isize'], result: 'pointer'  },
  wgpuBufferGetSize: { parameters: ['pointer'], result: 'u64'  },
  wgpuBufferGetUsage: { parameters: ['pointer'], result: 'pointer'  },
  wgpuBufferMapAsync: { parameters: ['pointer', 'pointer', 'isize', 'isize', 'pointer', 'pointer'], result: 'void'  },
  wgpuBufferMapAsync2: { parameters: ['pointer', 'pointer', 'isize', 'isize', 'pointer'], result: 'pointer'  },
  wgpuBufferMapAsyncF: { parameters: ['pointer', 'pointer', 'isize', 'isize', 'pointer'], result: 'pointer'  },
  wgpuBufferSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuBufferUnmap: { parameters: ['pointer'], result: 'void'  },
  wgpuBufferAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuBufferRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuCommandBufferSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuCommandBufferAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuCommandBufferRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuCommandEncoderBeginComputePass: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuCommandEncoderBeginRenderPass: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuCommandEncoderClearBuffer: { parameters: ['pointer', 'pointer', 'u64', 'u64'], result: 'void'  },
  wgpuCommandEncoderCopyBufferToBuffer: { parameters: ['pointer', 'pointer', 'u64', 'pointer', 'u64', 'u64'], result: 'void'  },
  wgpuCommandEncoderCopyBufferToTexture: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderCopyTextureToBuffer: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderCopyTextureToTexture: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderFinish: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuCommandEncoderInjectValidationError: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderInsertDebugMarker: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void'  },
  wgpuCommandEncoderPushDebugGroup: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderResolveQuerySet: { parameters: ['pointer', 'pointer', 'u32', 'u32', 'pointer', 'u64'], result: 'void'  },
  wgpuCommandEncoderSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuCommandEncoderWriteBuffer: { parameters: ['pointer', 'pointer', 'u64', 'pointer', 'u64'], result: 'void'  },
  wgpuCommandEncoderWriteTimestamp: { parameters: ['pointer', 'pointer', 'u32'], result: 'void'  },
  wgpuCommandEncoderAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuCommandEncoderRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuComputePassEncoderDispatchWorkgroups: { parameters: ['pointer', 'u32', 'u32', 'u32'], result: 'void'  },
  wgpuComputePassEncoderDispatchWorkgroupsIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void'  },
  wgpuComputePassEncoderEnd: { parameters: ['pointer'], result: 'void'  },
  wgpuComputePassEncoderInsertDebugMarker: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuComputePassEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void'  },
  wgpuComputePassEncoderPushDebugGroup: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuComputePassEncoderSetBindGroup: { parameters: ['pointer', 'u32', 'pointer', 'isize', 'pointer'], result: 'void'  },
  wgpuComputePassEncoderSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuComputePassEncoderSetPipeline: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuComputePassEncoderWriteTimestamp: { parameters: ['pointer', 'pointer', 'u32'], result: 'void'  },
  wgpuComputePassEncoderAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuComputePassEncoderRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuComputePipelineGetBindGroupLayout: { parameters: ['pointer', 'u32'], result: 'pointer'  },
  wgpuComputePipelineSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuComputePipelineAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuComputePipelineRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuDeviceCreateBindGroup: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateBindGroupLayout: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateBuffer: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateCommandEncoder: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateComputePipeline: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateComputePipelineAsync: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuDeviceCreateComputePipelineAsync2: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateComputePipelineAsyncF: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateErrorBuffer: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateErrorExternalTexture: { parameters: ['pointer'], result: 'pointer'  },
  wgpuDeviceCreateErrorShaderModule: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateErrorTexture: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateExternalTexture: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreatePipelineLayout: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateQuerySet: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateRenderBundleEncoder: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateRenderPipeline: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateRenderPipelineAsync: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuDeviceCreateRenderPipelineAsync2: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateRenderPipelineAsyncF: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateSampler: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateShaderModule: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceCreateTexture: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceDestroy: { parameters: ['pointer'], result: 'void'  },
  wgpuDeviceForceLoss: { parameters: ['pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuDeviceGetAHardwareBufferProperties: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceGetAdapter: { parameters: ['pointer'], result: 'pointer'  },
  wgpuDeviceGetAdapterInfo: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceGetFeatures: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuDeviceGetLimits: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceGetLostFuture: { parameters: ['pointer'], result: 'pointer'  },
  wgpuDeviceGetQueue: { parameters: ['pointer'], result: 'pointer'  },
  wgpuDeviceHasFeature: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceImportSharedBufferMemory: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceImportSharedFence: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceImportSharedTextureMemory: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDeviceInjectError: { parameters: ['pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuDevicePopErrorScope: { parameters: ['pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuDevicePopErrorScope2: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDevicePopErrorScopeF: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuDevicePushErrorScope: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuDeviceSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuDeviceSetLoggingCallback: { parameters: ['pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuDeviceTick: { parameters: ['pointer'], result: 'void'  },
  wgpuDeviceValidateTextureDescriptor: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuDeviceAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuDeviceRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuExternalTextureDestroy: { parameters: ['pointer'], result: 'void'  },
  wgpuExternalTextureExpire: { parameters: ['pointer'], result: 'void'  },
  wgpuExternalTextureRefresh: { parameters: ['pointer'], result: 'void'  },
  wgpuExternalTextureSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuExternalTextureAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuExternalTextureRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuInstanceCreateSurface: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuInstanceEnumerateWGSLLanguageFeatures: { parameters: ['pointer', 'pointer'], result: 'isize'  },
  wgpuInstanceHasWGSLLanguageFeature: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuInstanceProcessEvents: { parameters: ['pointer'], result: 'void'  },
  wgpuInstanceRequestAdapter: { parameters: ['pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuInstanceRequestAdapter2: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuInstanceRequestAdapterF: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuInstanceWaitAny: { parameters: ['pointer', 'isize', 'pointer', 'u64'], result: 'pointer'  },
  wgpuInstanceAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuInstanceRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuPipelineLayoutSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuPipelineLayoutAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuPipelineLayoutRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuQuerySetDestroy: { parameters: ['pointer'], result: 'void'  },
  wgpuQuerySetGetCount: { parameters: ['pointer'], result: 'u32'  },
  wgpuQuerySetGetType: { parameters: ['pointer'], result: 'pointer'  },
  wgpuQuerySetSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuQuerySetAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuQuerySetRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuQueueCopyExternalTextureForBrowser: { parameters: ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuQueueCopyTextureForBrowser: { parameters: ['pointer', 'pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuQueueOnSubmittedWorkDone: { parameters: ['pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuQueueOnSubmittedWorkDone2: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuQueueOnSubmittedWorkDoneF: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuQueueSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuQueueSubmit: { parameters: ['pointer', 'isize', 'pointer'], result: 'void'  },
  wgpuQueueWriteBuffer: { parameters: ['pointer', 'pointer', 'u64', 'pointer', 'isize'], result: 'void'  },
  wgpuQueueWriteTexture: { parameters: ['pointer', 'pointer', 'pointer', 'isize', 'pointer', 'pointer'], result: 'void'  },
  wgpuQueueAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuQueueRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderBundleSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderBundleAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderBundleRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderBundleEncoderDraw: { parameters: ['pointer', 'u32', 'u32', 'u32', 'u32'], result: 'void'  },
  wgpuRenderBundleEncoderDrawIndexed: { parameters: ['pointer', 'u32', 'u32', 'u32', 'i32', 'u32'], result: 'void'  },
  wgpuRenderBundleEncoderDrawIndexedIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void'  },
  wgpuRenderBundleEncoderDrawIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void'  },
  wgpuRenderBundleEncoderFinish: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuRenderBundleEncoderInsertDebugMarker: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderBundleEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderBundleEncoderPushDebugGroup: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderBundleEncoderSetBindGroup: { parameters: ['pointer', 'u32', 'pointer', 'isize', 'pointer'], result: 'void'  },
  wgpuRenderBundleEncoderSetIndexBuffer: { parameters: ['pointer', 'pointer', 'pointer', 'u64', 'u64'], result: 'void'  },
  wgpuRenderBundleEncoderSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderBundleEncoderSetPipeline: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderBundleEncoderSetVertexBuffer: { parameters: ['pointer', 'u32', 'pointer', 'u64', 'u64'], result: 'void'  },
  wgpuRenderBundleEncoderAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderBundleEncoderRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPassEncoderBeginOcclusionQuery: { parameters: ['pointer', 'u32'], result: 'void'  },
  wgpuRenderPassEncoderDraw: { parameters: ['pointer', 'u32', 'u32', 'u32', 'u32'], result: 'void'  },
  wgpuRenderPassEncoderDrawIndexed: { parameters: ['pointer', 'u32', 'u32', 'u32', 'i32', 'u32'], result: 'void'  },
  wgpuRenderPassEncoderDrawIndexedIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void'  },
  wgpuRenderPassEncoderDrawIndirect: { parameters: ['pointer', 'pointer', 'u64'], result: 'void'  },
  wgpuRenderPassEncoderEnd: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPassEncoderEndOcclusionQuery: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPassEncoderExecuteBundles: { parameters: ['pointer', 'isize', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderInsertDebugMarker: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderMultiDrawIndexedIndirect: { parameters: ['pointer', 'pointer', 'u64', 'u32', 'pointer', 'u64'], result: 'void'  },
  wgpuRenderPassEncoderMultiDrawIndirect: { parameters: ['pointer', 'pointer', 'u64', 'u32', 'pointer', 'u64'], result: 'void'  },
  wgpuRenderPassEncoderPixelLocalStorageBarrier: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPassEncoderPopDebugGroup: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPassEncoderPushDebugGroup: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderSetBindGroup: { parameters: ['pointer', 'u32', 'pointer', 'isize', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderSetBlendConstant: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderSetIndexBuffer: { parameters: ['pointer', 'pointer', 'pointer', 'u64', 'u64'], result: 'void'  },
  wgpuRenderPassEncoderSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderSetPipeline: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderSetScissorRect: { parameters: ['pointer', 'u32', 'u32', 'u32', 'u32'], result: 'void'  },
  wgpuRenderPassEncoderSetStencilReference: { parameters: ['pointer', 'u32'], result: 'void'  },
  wgpuRenderPassEncoderSetVertexBuffer: { parameters: ['pointer', 'u32', 'pointer', 'u64', 'u64'], result: 'void'  },
  wgpuRenderPassEncoderSetViewport: { parameters: ['pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuRenderPassEncoderWriteTimestamp: { parameters: ['pointer', 'pointer', 'u32'], result: 'void'  },
  wgpuRenderPassEncoderAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPassEncoderRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPipelineGetBindGroupLayout: { parameters: ['pointer', 'u32'], result: 'pointer'  },
  wgpuRenderPipelineSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuRenderPipelineAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuRenderPipelineRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuSamplerSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSamplerAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuSamplerRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuShaderModuleGetCompilationInfo: { parameters: ['pointer', 'pointer', 'pointer'], result: 'void'  },
  wgpuShaderModuleGetCompilationInfo2: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuShaderModuleGetCompilationInfoF: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuShaderModuleSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuShaderModuleAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuShaderModuleRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedBufferMemoryBeginAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedBufferMemoryCreateBuffer: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedBufferMemoryEndAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedBufferMemoryGetProperties: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedBufferMemoryIsDeviceLost: { parameters: ['pointer'], result: 'pointer'  },
  wgpuSharedBufferMemorySetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSharedBufferMemoryAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedBufferMemoryRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedFenceExportInfo: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSharedFenceAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedFenceRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedTextureMemoryBeginAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedTextureMemoryCreateTexture: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedTextureMemoryEndAccess: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedTextureMemoryGetProperties: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuSharedTextureMemoryIsDeviceLost: { parameters: ['pointer'], result: 'pointer'  },
  wgpuSharedTextureMemorySetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSharedTextureMemoryAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuSharedTextureMemoryRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuSurfaceConfigure: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSurfaceGetCapabilities: { parameters: ['pointer', 'pointer', 'pointer'], result: 'pointer'  },
  wgpuSurfaceGetCurrentTexture: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSurfacePresent: { parameters: ['pointer'], result: 'void'  },
  wgpuSurfaceSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuSurfaceUnconfigure: { parameters: ['pointer'], result: 'void'  },
  wgpuSurfaceAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuSurfaceRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuTextureCreateErrorView: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuTextureCreateView: { parameters: ['pointer', 'pointer'], result: 'pointer'  },
  wgpuTextureDestroy: { parameters: ['pointer'], result: 'void'  },
  wgpuTextureGetDepthOrArrayLayers: { parameters: ['pointer'], result: 'u32'  },
  wgpuTextureGetDimension: { parameters: ['pointer'], result: 'pointer'  },
  wgpuTextureGetFormat: { parameters: ['pointer'], result: 'pointer'  },
  wgpuTextureGetHeight: { parameters: ['pointer'], result: 'u32'  },
  wgpuTextureGetMipLevelCount: { parameters: ['pointer'], result: 'u32'  },
  wgpuTextureGetSampleCount: { parameters: ['pointer'], result: 'u32'  },
  wgpuTextureGetUsage: { parameters: ['pointer'], result: 'pointer'  },
  wgpuTextureGetWidth: { parameters: ['pointer'], result: 'u32'  },
  wgpuTextureSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuTextureAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuTextureRelease: { parameters: ['pointer'], result: 'void'  },
  wgpuTextureViewSetLabel: { parameters: ['pointer', 'pointer'], result: 'void'  },
  wgpuTextureViewAddRef: { parameters: ['pointer'], result: 'void'  },
  wgpuTextureViewRelease: { parameters: ['pointer'], result: 'void'  }
})

// CONSTS
export const WGPUBufferUsage_None = 0x0000000000000000
export const WGPUBufferUsage_MapRead = 0x0000000000000001
export const WGPUBufferUsage_MapWrite = 0x0000000000000002
export const WGPUBufferUsage_CopySrc = 0x0000000000000004
export const WGPUBufferUsage_CopyDst = 0x0000000000000008
export const WGPUBufferUsage_Index = 0x0000000000000010
export const WGPUBufferUsage_Vertex = 0x0000000000000020
export const WGPUBufferUsage_Uniform = 0x0000000000000040
export const WGPUBufferUsage_Storage = 0x0000000000000080
export const WGPUBufferUsage_Indirect = 0x0000000000000100
export const WGPUBufferUsage_QueryResolve = 0x0000000000000200
export const WGPUColorWriteMask_None = 0x0000000000000000
export const WGPUColorWriteMask_Red = 0x0000000000000001
export const WGPUColorWriteMask_Green = 0x0000000000000002
export const WGPUColorWriteMask_Blue = 0x0000000000000004
export const WGPUColorWriteMask_Alpha = 0x0000000000000008
export const WGPUColorWriteMask_All = 0x000000000000000F
export const WGPUHeapProperty_DeviceLocal = 0x0000000000000001
export const WGPUHeapProperty_HostVisible = 0x0000000000000002
export const WGPUHeapProperty_HostCoherent = 0x0000000000000004
export const WGPUHeapProperty_HostUncached = 0x0000000000000008
export const WGPUHeapProperty_HostCached = 0x0000000000000010
export const WGPUMapMode_None = 0x0000000000000000
export const WGPUMapMode_Read = 0x0000000000000001
export const WGPUMapMode_Write = 0x0000000000000002
export const WGPUShaderStage_None = 0x0000000000000000
export const WGPUShaderStage_Vertex = 0x0000000000000001
export const WGPUShaderStage_Fragment = 0x0000000000000002
export const WGPUShaderStage_Compute = 0x0000000000000004
export const WGPUTextureUsage_None = 0x0000000000000000
export const WGPUTextureUsage_CopySrc = 0x0000000000000001
export const WGPUTextureUsage_CopyDst = 0x0000000000000002
export const WGPUTextureUsage_TextureBinding = 0x0000000000000004
export const WGPUTextureUsage_StorageBinding = 0x0000000000000008
export const WGPUTextureUsage_RenderAttachment = 0x0000000000000010
export const WGPUTextureUsage_TransientAttachment = 0x0000000000000020
export const WGPUTextureUsage_StorageAttachment = 0x0000000000000040

// ENUMS
export enum WGPUWGSLFeatureName {
  'ReadonlyAndReadwriteStorageTextures' = 0x00000001,
  'Packed4x8IntegerDotProduct' = 0x00000002,
  'UnrestrictedPointerParameters' = 0x00000003,
  'PointerCompositeAccess' = 0x00000004,
  'ChromiumTestingUnimplemented' = 0x00050000,
  'ChromiumTestingUnsafeExperimental' = 0x00050001,
  'ChromiumTestingExperimental' = 0x00050002,
  'ChromiumTestingShippedWithKillswitch' = 0x00050003,
  'ChromiumTestingShipped' = 0x00050004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUAdapterType {
  'DiscreteGPU' = 0x00000001,
  'IntegratedGPU' = 0x00000002,
  'CPU' = 0x00000003,
  'Unknown' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUAddressMode {
  'Undefined' = 0x00000000,
  'ClampToEdge' = 0x00000001,
  'Repeat' = 0x00000002,
  'MirrorRepeat' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUAlphaMode {
  'Opaque' = 0x00000001,
  'Premultiplied' = 0x00000002,
  'Unpremultiplied' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUBackendType {
  'Undefined' = 0x00000000,
  'Null' = 0x00000001,
  'WebGPU' = 0x00000002,
  'D3D11' = 0x00000003,
  'D3D12' = 0x00000004,
  'Metal' = 0x00000005,
  'Vulkan' = 0x00000006,
  'OpenGL' = 0x00000007,
  'OpenGLES' = 0x00000008,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUBlendFactor {
  'Undefined' = 0x00000000,
  'Zero' = 0x00000001,
  'One' = 0x00000002,
  'Src' = 0x00000003,
  'OneMinusSrc' = 0x00000004,
  'SrcAlpha' = 0x00000005,
  'OneMinusSrcAlpha' = 0x00000006,
  'Dst' = 0x00000007,
  'OneMinusDst' = 0x00000008,
  'DstAlpha' = 0x00000009,
  'OneMinusDstAlpha' = 0x0000000A,
  'SrcAlphaSaturated' = 0x0000000B,
  'Constant' = 0x0000000C,
  'OneMinusConstant' = 0x0000000D,
  'Src1' = 0x0000000E,
  'OneMinusSrc1' = 0x0000000F,
  'Src1Alpha' = 0x00000010,
  'OneMinusSrc1Alpha' = 0x00000011,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUBlendOperation {
  'Undefined' = 0x00000000,
  'Add' = 0x00000001,
  'Subtract' = 0x00000002,
  'ReverseSubtract' = 0x00000003,
  'Min' = 0x00000004,
  'Max' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUBufferBindingType {
  'BindingNotUsed' = 0x00000000,
  'Uniform' = 0x00000001,
  'Storage' = 0x00000002,
  'ReadOnlyStorage' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUBufferMapAsyncStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'ValidationError' = 0x00000003,
  'Unknown' = 0x00000004,
  'DeviceLost' = 0x00000005,
  'DestroyedBeforeCallback' = 0x00000006,
  'UnmappedBeforeCallback' = 0x00000007,
  'MappingAlreadyPending' = 0x00000008,
  'OffsetOutOfRange' = 0x00000009,
  'SizeOutOfRange' = 0x0000000A,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUBufferMapState {
  'Unmapped' = 0x00000001,
  'Pending' = 0x00000002,
  'Mapped' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCallbackMode {
  'WaitAnyOnly' = 0x00000001,
  'AllowProcessEvents' = 0x00000002,
  'AllowSpontaneous' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCompareFunction {
  'Undefined' = 0x00000000,
  'Never' = 0x00000001,
  'Less' = 0x00000002,
  'Equal' = 0x00000003,
  'LessEqual' = 0x00000004,
  'Greater' = 0x00000005,
  'NotEqual' = 0x00000006,
  'GreaterEqual' = 0x00000007,
  'Always' = 0x00000008,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCompilationInfoRequestStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'Error' = 0x00000003,
  'DeviceLost' = 0x00000004,
  'Unknown' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCompilationMessageType {
  'Error' = 0x00000001,
  'Warning' = 0x00000002,
  'Info' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCompositeAlphaMode {
  'Auto' = 0x00000000,
  'Opaque' = 0x00000001,
  'Premultiplied' = 0x00000002,
  'Unpremultiplied' = 0x00000003,
  'Inherit' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCreatePipelineAsyncStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'ValidationError' = 0x00000003,
  'InternalError' = 0x00000004,
  'DeviceLost' = 0x00000005,
  'DeviceDestroyed' = 0x00000006,
  'Unknown' = 0x00000007,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUCullMode {
  'Undefined' = 0x00000000,
  'None' = 0x00000001,
  'Front' = 0x00000002,
  'Back' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUDeviceLostReason {
  'Unknown' = 0x00000001,
  'Destroyed' = 0x00000002,
  'InstanceDropped' = 0x00000003,
  'FailedCreation' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUErrorFilter {
  'Validation' = 0x00000001,
  'OutOfMemory' = 0x00000002,
  'Internal' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUErrorType {
  'NoError' = 0x00000001,
  'Validation' = 0x00000002,
  'OutOfMemory' = 0x00000003,
  'Internal' = 0x00000004,
  'Unknown' = 0x00000005,
  'DeviceLost' = 0x00000006,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUExternalTextureRotation {
  'Rotate0Degrees' = 0x00000001,
  'Rotate90Degrees' = 0x00000002,
  'Rotate180Degrees' = 0x00000003,
  'Rotate270Degrees' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUFeatureLevel {
  'Undefined' = 0x00000000,
  'Compatibility' = 0x00000001,
  'Core' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUFeatureName {
  'DepthClipControl' = 0x00000001,
  'Depth32FloatStencil8' = 0x00000002,
  'TimestampQuery' = 0x00000003,
  'TextureCompressionBC' = 0x00000004,
  'TextureCompressionETC2' = 0x00000005,
  'TextureCompressionASTC' = 0x00000006,
  'IndirectFirstInstance' = 0x00000007,
  'ShaderF16' = 0x00000008,
  'RG11B10UfloatRenderable' = 0x00000009,
  'BGRA8UnormStorage' = 0x0000000A,
  'Float32Filterable' = 0x0000000B,
  'Float32Blendable' = 0x0000000C,
  'Subgroups' = 0x0000000D,
  'SubgroupsF16' = 0x0000000E,
  'DawnInternalUsages' = 0x00050000,
  'DawnMultiPlanarFormats' = 0x00050001,
  'DawnNative' = 0x00050002,
  'ChromiumExperimentalTimestampQueryInsidePasses' = 0x00050003,
  'ImplicitDeviceSynchronization' = 0x00050004,
  'ChromiumExperimentalImmediateData' = 0x00050005,
  'TransientAttachments' = 0x00050006,
  'MSAARenderToSingleSampled' = 0x00050007,
  'DualSourceBlending' = 0x00050008,
  'D3D11MultithreadProtected' = 0x00050009,
  'ANGLETextureSharing' = 0x0005000A,
  'PixelLocalStorageCoherent' = 0x0005000B,
  'PixelLocalStorageNonCoherent' = 0x0005000C,
  'Unorm16TextureFormats' = 0x0005000D,
  'Snorm16TextureFormats' = 0x0005000E,
  'MultiPlanarFormatExtendedUsages' = 0x0005000F,
  'MultiPlanarFormatP010' = 0x00050010,
  'HostMappedPointer' = 0x00050011,
  'MultiPlanarRenderTargets' = 0x00050012,
  'MultiPlanarFormatNv12a' = 0x00050013,
  'FramebufferFetch' = 0x00050014,
  'BufferMapExtendedUsages' = 0x00050015,
  'AdapterPropertiesMemoryHeaps' = 0x00050016,
  'AdapterPropertiesD3D' = 0x00050017,
  'AdapterPropertiesVk' = 0x00050018,
  'R8UnormStorage' = 0x00050019,
  'FormatCapabilities' = 0x0005001A,
  'DrmFormatCapabilities' = 0x0005001B,
  'Norm16TextureFormats' = 0x0005001C,
  'MultiPlanarFormatNv16' = 0x0005001D,
  'MultiPlanarFormatNv24' = 0x0005001E,
  'MultiPlanarFormatP210' = 0x0005001F,
  'MultiPlanarFormatP410' = 0x00050020,
  'SharedTextureMemoryVkDedicatedAllocation' = 0x00050021,
  'SharedTextureMemoryAHardwareBuffer' = 0x00050022,
  'SharedTextureMemoryDmaBuf' = 0x00050023,
  'SharedTextureMemoryOpaqueFD' = 0x00050024,
  'SharedTextureMemoryZirconHandle' = 0x00050025,
  'SharedTextureMemoryDXGISharedHandle' = 0x00050026,
  'SharedTextureMemoryD3D11Texture2D' = 0x00050027,
  'SharedTextureMemoryIOSurface' = 0x00050028,
  'SharedTextureMemoryEGLImage' = 0x00050029,
  'SharedFenceVkSemaphoreOpaqueFD' = 0x0005002A,
  'SharedFenceSyncFD' = 0x0005002B,
  'SharedFenceVkSemaphoreZirconHandle' = 0x0005002C,
  'SharedFenceDXGISharedHandle' = 0x0005002D,
  'SharedFenceMTLSharedEvent' = 0x0005002E,
  'SharedBufferMemoryD3D12Resource' = 0x0005002F,
  'StaticSamplers' = 0x00050030,
  'YCbCrVulkanSamplers' = 0x00050031,
  'ShaderModuleCompilationOptions' = 0x00050032,
  'DawnLoadResolveTexture' = 0x00050033,
  'DawnPartialLoadResolveTexture' = 0x00050034,
  'MultiDrawIndirect' = 0x00050035,
  'ClipDistances' = 0x00050036,
  'DawnTexelCopyBufferRowAlignment' = 0x00050037,
  'FlexibleTextureViews' = 0x00050038,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUFilterMode {
  'Undefined' = 0x00000000,
  'Nearest' = 0x00000001,
  'Linear' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUFrontFace {
  'Undefined' = 0x00000000,
  'CCW' = 0x00000001,
  'CW' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUIndexFormat {
  'Undefined' = 0x00000000,
  'Uint16' = 0x00000001,
  'Uint32' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPULoadOp {
  'Undefined' = 0x00000000,
  'Load' = 0x00000001,
  'Clear' = 0x00000002,
  'ExpandResolveTexture' = 0x00050003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPULoggingType {
  'Verbose' = 0x00000001,
  'Info' = 0x00000002,
  'Warning' = 0x00000003,
  'Error' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUMapAsyncStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'Error' = 0x00000003,
  'Aborted' = 0x00000004,
  'Unknown' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUMipmapFilterMode {
  'Undefined' = 0x00000000,
  'Nearest' = 0x00000001,
  'Linear' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUOptionalBool {
  'False' = 0x00000000,
  'True' = 0x00000001,
  'Undefined' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUPopErrorScopeStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUPowerPreference {
  'Undefined' = 0x00000000,
  'LowPower' = 0x00000001,
  'HighPerformance' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUPresentMode {
  'Fifo' = 0x00000001,
  'FifoRelaxed' = 0x00000002,
  'Immediate' = 0x00000003,
  'Mailbox' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUPrimitiveTopology {
  'Undefined' = 0x00000000,
  'PointList' = 0x00000001,
  'LineList' = 0x00000002,
  'LineStrip' = 0x00000003,
  'TriangleList' = 0x00000004,
  'TriangleStrip' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUQueryType {
  'Occlusion' = 0x00000001,
  'Timestamp' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUQueueWorkDoneStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'Error' = 0x00000003,
  'Unknown' = 0x00000004,
  'DeviceLost' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPURequestAdapterStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'Unavailable' = 0x00000003,
  'Error' = 0x00000004,
  'Unknown' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPURequestDeviceStatus {
  'Success' = 0x00000001,
  'InstanceDropped' = 0x00000002,
  'Error' = 0x00000003,
  'Unknown' = 0x00000004,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUSType {
  'ShaderSourceSPIRV' = 0x00000001,
  'ShaderSourceWGSL' = 0x00000002,
  'RenderPassMaxDrawCount' = 0x00000003,
  'SurfaceSourceMetalLayer' = 0x00000004,
  'SurfaceSourceWindowsHWND' = 0x00000005,
  'SurfaceSourceXlibWindow' = 0x00000006,
  'SurfaceSourceWaylandSurface' = 0x00000007,
  'SurfaceSourceAndroidNativeWindow' = 0x00000008,
  'SurfaceSourceXCBWindow' = 0x00000009,
  'AdapterPropertiesSubgroups' = 0x0000000A,
  'TextureBindingViewDimensionDescriptor' = 0x00020000,
  'SurfaceSourceCanvasHTMLSelector_Emscripten' = 0x00040000,
  'SurfaceDescriptorFromWindowsCoreWindow' = 0x00050000,
  'ExternalTextureBindingEntry' = 0x00050001,
  'ExternalTextureBindingLayout' = 0x00050002,
  'SurfaceDescriptorFromWindowsSwapChainPanel' = 0x00050003,
  'DawnTextureInternalUsageDescriptor' = 0x00050004,
  'DawnEncoderInternalUsageDescriptor' = 0x00050005,
  'DawnInstanceDescriptor' = 0x00050006,
  'DawnCacheDeviceDescriptor' = 0x00050007,
  'DawnAdapterPropertiesPowerPreference' = 0x00050008,
  'DawnBufferDescriptorErrorInfoFromWireClient' = 0x00050009,
  'DawnTogglesDescriptor' = 0x0005000A,
  'DawnShaderModuleSPIRVOptionsDescriptor' = 0x0005000B,
  'RequestAdapterOptionsLUID' = 0x0005000C,
  'RequestAdapterOptionsGetGLProc' = 0x0005000D,
  'RequestAdapterOptionsD3D11Device' = 0x0005000E,
  'DawnRenderPassColorAttachmentRenderToSingleSampled' = 0x0005000F,
  'RenderPassPixelLocalStorage' = 0x00050010,
  'PipelineLayoutPixelLocalStorage' = 0x00050011,
  'BufferHostMappedPointer' = 0x00050012,
  'DawnExperimentalSubgroupLimits' = 0x00050013,
  'AdapterPropertiesMemoryHeaps' = 0x00050014,
  'AdapterPropertiesD3D' = 0x00050015,
  'AdapterPropertiesVk' = 0x00050016,
  'DawnWireWGSLControl' = 0x00050017,
  'DawnWGSLBlocklist' = 0x00050018,
  'DrmFormatCapabilities' = 0x00050019,
  'ShaderModuleCompilationOptions' = 0x0005001A,
  'ColorTargetStateExpandResolveTextureDawn' = 0x0005001B,
  'RenderPassDescriptorExpandResolveRect' = 0x0005001C,
  'SharedTextureMemoryVkDedicatedAllocationDescriptor' = 0x0005001D,
  'SharedTextureMemoryAHardwareBufferDescriptor' = 0x0005001E,
  'SharedTextureMemoryDmaBufDescriptor' = 0x0005001F,
  'SharedTextureMemoryOpaqueFDDescriptor' = 0x00050020,
  'SharedTextureMemoryZirconHandleDescriptor' = 0x00050021,
  'SharedTextureMemoryDXGISharedHandleDescriptor' = 0x00050022,
  'SharedTextureMemoryD3D11Texture2DDescriptor' = 0x00050023,
  'SharedTextureMemoryIOSurfaceDescriptor' = 0x00050024,
  'SharedTextureMemoryEGLImageDescriptor' = 0x00050025,
  'SharedTextureMemoryInitializedBeginState' = 0x00050026,
  'SharedTextureMemoryInitializedEndState' = 0x00050027,
  'SharedTextureMemoryVkImageLayoutBeginState' = 0x00050028,
  'SharedTextureMemoryVkImageLayoutEndState' = 0x00050029,
  'SharedTextureMemoryD3DSwapchainBeginState' = 0x0005002A,
  'SharedFenceVkSemaphoreOpaqueFDDescriptor' = 0x0005002B,
  'SharedFenceVkSemaphoreOpaqueFDExportInfo' = 0x0005002C,
  'SharedFenceSyncFDDescriptor' = 0x0005002D,
  'SharedFenceSyncFDExportInfo' = 0x0005002E,
  'SharedFenceVkSemaphoreZirconHandleDescriptor' = 0x0005002F,
  'SharedFenceVkSemaphoreZirconHandleExportInfo' = 0x00050030,
  'SharedFenceDXGISharedHandleDescriptor' = 0x00050031,
  'SharedFenceDXGISharedHandleExportInfo' = 0x00050032,
  'SharedFenceMTLSharedEventDescriptor' = 0x00050033,
  'SharedFenceMTLSharedEventExportInfo' = 0x00050034,
  'SharedBufferMemoryD3D12ResourceDescriptor' = 0x00050035,
  'StaticSamplerBindingLayout' = 0x00050036,
  'YCbCrVkDescriptor' = 0x00050037,
  'SharedTextureMemoryAHardwareBufferProperties' = 0x00050038,
  'AHardwareBufferProperties' = 0x00050039,
  'DawnExperimentalImmediateDataLimits' = 0x0005003A,
  'DawnTexelCopyBufferRowAlignmentLimits' = 0x0005003B,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUSamplerBindingType {
  'BindingNotUsed' = 0x00000000,
  'Filtering' = 0x00000001,
  'NonFiltering' = 0x00000002,
  'Comparison' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUSharedFenceType {
  'VkSemaphoreOpaqueFD' = 0x00000001,
  'SyncFD' = 0x00000002,
  'VkSemaphoreZirconHandle' = 0x00000003,
  'DXGISharedHandle' = 0x00000004,
  'MTLSharedEvent' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUStatus {
  'Success' = 0x00000001,
  'Error' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUStencilOperation {
  'Undefined' = 0x00000000,
  'Keep' = 0x00000001,
  'Zero' = 0x00000002,
  'Replace' = 0x00000003,
  'Invert' = 0x00000004,
  'IncrementClamp' = 0x00000005,
  'DecrementClamp' = 0x00000006,
  'IncrementWrap' = 0x00000007,
  'DecrementWrap' = 0x00000008,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUStorageTextureAccess {
  'BindingNotUsed' = 0x00000000,
  'WriteOnly' = 0x00000001,
  'ReadOnly' = 0x00000002,
  'ReadWrite' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUStoreOp {
  'Undefined' = 0x00000000,
  'Store' = 0x00000001,
  'Discard' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUSurfaceGetCurrentTextureStatus {
  'Success' = 0x00000001,
  'Timeout' = 0x00000002,
  'Outdated' = 0x00000003,
  'Lost' = 0x00000004,
  'OutOfMemory' = 0x00000005,
  'DeviceLost' = 0x00000006,
  'Error' = 0x00000007,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUTextureAspect {
  'Undefined' = 0x00000000,
  'All' = 0x00000001,
  'StencilOnly' = 0x00000002,
  'DepthOnly' = 0x00000003,
  'Plane0Only' = 0x00050000,
  'Plane1Only' = 0x00050001,
  'Plane2Only' = 0x00050002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUTextureDimension {
  'Undefined' = 0x00000000,
  '1D' = 0x00000001,
  '2D' = 0x00000002,
  '3D' = 0x00000003,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUTextureFormat {
  'Undefined' = 0x00000000,
  'R8Unorm' = 0x00000001,
  'R8Snorm' = 0x00000002,
  'R8Uint' = 0x00000003,
  'R8Sint' = 0x00000004,
  'R16Uint' = 0x00000005,
  'R16Sint' = 0x00000006,
  'R16Float' = 0x00000007,
  'RG8Unorm' = 0x00000008,
  'RG8Snorm' = 0x00000009,
  'RG8Uint' = 0x0000000A,
  'RG8Sint' = 0x0000000B,
  'R32Float' = 0x0000000C,
  'R32Uint' = 0x0000000D,
  'R32Sint' = 0x0000000E,
  'RG16Uint' = 0x0000000F,
  'RG16Sint' = 0x00000010,
  'RG16Float' = 0x00000011,
  'RGBA8Unorm' = 0x00000012,
  'RGBA8UnormSrgb' = 0x00000013,
  'RGBA8Snorm' = 0x00000014,
  'RGBA8Uint' = 0x00000015,
  'RGBA8Sint' = 0x00000016,
  'BGRA8Unorm' = 0x00000017,
  'BGRA8UnormSrgb' = 0x00000018,
  'RGB10A2Uint' = 0x00000019,
  'RGB10A2Unorm' = 0x0000001A,
  'RG11B10Ufloat' = 0x0000001B,
  'RGB9E5Ufloat' = 0x0000001C,
  'RG32Float' = 0x0000001D,
  'RG32Uint' = 0x0000001E,
  'RG32Sint' = 0x0000001F,
  'RGBA16Uint' = 0x00000020,
  'RGBA16Sint' = 0x00000021,
  'RGBA16Float' = 0x00000022,
  'RGBA32Float' = 0x00000023,
  'RGBA32Uint' = 0x00000024,
  'RGBA32Sint' = 0x00000025,
  'Stencil8' = 0x00000026,
  'Depth16Unorm' = 0x00000027,
  'Depth24Plus' = 0x00000028,
  'Depth24PlusStencil8' = 0x00000029,
  'Depth32Float' = 0x0000002A,
  'Depth32FloatStencil8' = 0x0000002B,
  'BC1RGBAUnorm' = 0x0000002C,
  'BC1RGBAUnormSrgb' = 0x0000002D,
  'BC2RGBAUnorm' = 0x0000002E,
  'BC2RGBAUnormSrgb' = 0x0000002F,
  'BC3RGBAUnorm' = 0x00000030,
  'BC3RGBAUnormSrgb' = 0x00000031,
  'BC4RUnorm' = 0x00000032,
  'BC4RSnorm' = 0x00000033,
  'BC5RGUnorm' = 0x00000034,
  'BC5RGSnorm' = 0x00000035,
  'BC6HRGBUfloat' = 0x00000036,
  'BC6HRGBFloat' = 0x00000037,
  'BC7RGBAUnorm' = 0x00000038,
  'BC7RGBAUnormSrgb' = 0x00000039,
  'ETC2RGB8Unorm' = 0x0000003A,
  'ETC2RGB8UnormSrgb' = 0x0000003B,
  'ETC2RGB8A1Unorm' = 0x0000003C,
  'ETC2RGB8A1UnormSrgb' = 0x0000003D,
  'ETC2RGBA8Unorm' = 0x0000003E,
  'ETC2RGBA8UnormSrgb' = 0x0000003F,
  'EACR11Unorm' = 0x00000040,
  'EACR11Snorm' = 0x00000041,
  'EACRG11Unorm' = 0x00000042,
  'EACRG11Snorm' = 0x00000043,
  'ASTC4x4Unorm' = 0x00000044,
  'ASTC4x4UnormSrgb' = 0x00000045,
  'ASTC5x4Unorm' = 0x00000046,
  'ASTC5x4UnormSrgb' = 0x00000047,
  'ASTC5x5Unorm' = 0x00000048,
  'ASTC5x5UnormSrgb' = 0x00000049,
  'ASTC6x5Unorm' = 0x0000004A,
  'ASTC6x5UnormSrgb' = 0x0000004B,
  'ASTC6x6Unorm' = 0x0000004C,
  'ASTC6x6UnormSrgb' = 0x0000004D,
  'ASTC8x5Unorm' = 0x0000004E,
  'ASTC8x5UnormSrgb' = 0x0000004F,
  'ASTC8x6Unorm' = 0x00000050,
  'ASTC8x6UnormSrgb' = 0x00000051,
  'ASTC8x8Unorm' = 0x00000052,
  'ASTC8x8UnormSrgb' = 0x00000053,
  'ASTC10x5Unorm' = 0x00000054,
  'ASTC10x5UnormSrgb' = 0x00000055,
  'ASTC10x6Unorm' = 0x00000056,
  'ASTC10x6UnormSrgb' = 0x00000057,
  'ASTC10x8Unorm' = 0x00000058,
  'ASTC10x8UnormSrgb' = 0x00000059,
  'ASTC10x10Unorm' = 0x0000005A,
  'ASTC10x10UnormSrgb' = 0x0000005B,
  'ASTC12x10Unorm' = 0x0000005C,
  'ASTC12x10UnormSrgb' = 0x0000005D,
  'ASTC12x12Unorm' = 0x0000005E,
  'ASTC12x12UnormSrgb' = 0x0000005F,
  'R16Unorm' = 0x00050000,
  'RG16Unorm' = 0x00050001,
  'RGBA16Unorm' = 0x00050002,
  'R16Snorm' = 0x00050003,
  'RG16Snorm' = 0x00050004,
  'RGBA16Snorm' = 0x00050005,
  'R8BG8Biplanar420Unorm' = 0x00050006,
  'R10X6BG10X6Biplanar420Unorm' = 0x00050007,
  'R8BG8A8Triplanar420Unorm' = 0x00050008,
  'R8BG8Biplanar422Unorm' = 0x00050009,
  'R8BG8Biplanar444Unorm' = 0x0005000A,
  'R10X6BG10X6Biplanar422Unorm' = 0x0005000B,
  'R10X6BG10X6Biplanar444Unorm' = 0x0005000C,
  'External' = 0x0005000D,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUTextureSampleType {
  'BindingNotUsed' = 0x00000000,
  'Float' = 0x00000001,
  'UnfilterableFloat' = 0x00000002,
  'Depth' = 0x00000003,
  'Sint' = 0x00000004,
  'Uint' = 0x00000005,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUTextureViewDimension {
  'Undefined' = 0x00000000,
  '1D' = 0x00000001,
  '2D' = 0x00000002,
  '2DArray' = 0x00000003,
  'Cube' = 0x00000004,
  'CubeArray' = 0x00000005,
  '3D' = 0x00000006,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUVertexFormat {
  'Uint8' = 0x00000001,
  'Uint8x2' = 0x00000002,
  'Uint8x4' = 0x00000003,
  'Sint8' = 0x00000004,
  'Sint8x2' = 0x00000005,
  'Sint8x4' = 0x00000006,
  'Unorm8' = 0x00000007,
  'Unorm8x2' = 0x00000008,
  'Unorm8x4' = 0x00000009,
  'Snorm8' = 0x0000000A,
  'Snorm8x2' = 0x0000000B,
  'Snorm8x4' = 0x0000000C,
  'Uint16' = 0x0000000D,
  'Uint16x2' = 0x0000000E,
  'Uint16x4' = 0x0000000F,
  'Sint16' = 0x00000010,
  'Sint16x2' = 0x00000011,
  'Sint16x4' = 0x00000012,
  'Unorm16' = 0x00000013,
  'Unorm16x2' = 0x00000014,
  'Unorm16x4' = 0x00000015,
  'Snorm16' = 0x00000016,
  'Snorm16x2' = 0x00000017,
  'Snorm16x4' = 0x00000018,
  'Float16' = 0x00000019,
  'Float16x2' = 0x0000001A,
  'Float16x4' = 0x0000001B,
  'Float32' = 0x0000001C,
  'Float32x2' = 0x0000001D,
  'Float32x3' = 0x0000001E,
  'Float32x4' = 0x0000001F,
  'Uint32' = 0x00000020,
  'Uint32x2' = 0x00000021,
  'Uint32x3' = 0x00000022,
  'Uint32x4' = 0x00000023,
  'Sint32' = 0x00000024,
  'Sint32x2' = 0x00000025,
  'Sint32x3' = 0x00000026,
  'Sint32x4' = 0x00000027,
  'Unorm10_10_10_2' = 0x00000028,
  'Unorm8x4BGRA' = 0x00000029,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUVertexStepMode {
  'Undefined' = 0x00000000,
  'Vertex' = 0x00000001,
  'Instance' = 0x00000002,
  'Force32' = 0x7FFFFFFF
}
export enum WGPUWaitStatus {
  'Success' = 0x00000001,
  'TimedOut' = 0x00000002,
  'UnsupportedTimeout' = 0x00000003,
  'UnsupportedCount' = 0x00000004,
  'UnsupportedMixedSources' = 0x00000005,
  'Unknown' = 0x00000006,
  'Force32' = 0x7FFFFFFF
}

// STRUCTS
export type WGPUChainedStruct = {
  next: struct WGPUChainedStruct | null
  sType: WGPUSType
}
export type WGPUChainedStructOut = {
  next: struct WGPUChainedStructOut | null
  sType: WGPUSType
}
export type WGPUBufferMapCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUBufferMapCallback2
  userdata1: null
  userdata2: null
}
export type WGPUCompilationInfoCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUCompilationInfoCallback2
  userdata1: null
  userdata2: null
}
export type WGPUCreateComputePipelineAsyncCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUCreateComputePipelineAsyncCallback2
  userdata1: null
  userdata2: null
}
export type WGPUCreateRenderPipelineAsyncCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUCreateRenderPipelineAsyncCallback2
  userdata1: null
  userdata2: null
}
export type WGPUDeviceLostCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUDeviceLostCallback2
  userdata1: null
  userdata2: null
}
export type WGPUPopErrorScopeCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUPopErrorScopeCallback2
  userdata1: null
  userdata2: null
}
export type WGPUQueueWorkDoneCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUQueueWorkDoneCallback2
  userdata1: null
  userdata2: null
}
export type WGPURequestAdapterCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPURequestAdapterCallback2
  userdata1: null
  userdata2: null
}
export type WGPURequestDeviceCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPURequestDeviceCallback2
  userdata1: null
  userdata2: null
}
export type WGPUUncapturedErrorCallbackInfo2 = {
  nextInChain: WGPUChainedStruct | null
  callback: WGPUUncapturedErrorCallback
  userdata1: null
  userdata2: null
}
export type WGPUINTERNAL__HAVE_EMDAWNWEBGPU_HEADER = {
  unused: boolean
}
export type WGPUAdapterPropertiesD3D = {
  chain: WGPUChainedStructOut
  shaderModel: number
}
export type WGPUAdapterPropertiesSubgroups = {
  chain: WGPUChainedStructOut
  subgroupMinSize: number
  subgroupMaxSize: number
}
export type WGPUAdapterPropertiesVk = {
  chain: WGPUChainedStructOut
  driverVersion: number
}
export type WGPUBindGroupEntry = {
  nextInChain: WGPUChainedStruct | null
  binding: number
  buffer: WGPUBuffer | null
  offset: bigint
  size: bigint
  sampler: WGPUSampler | null
  textureView: WGPUTextureView | null
}
export type WGPUBlendComponent = {
  operation: WGPUBlendOperation
  srcFactor: WGPUBlendFactor
  dstFactor: WGPUBlendFactor
}
export type WGPUBufferBindingLayout = {
  nextInChain: WGPUChainedStruct | null
  type: WGPUBufferBindingType
  hasDynamicOffset: boolean
  minBindingSize: bigint
}
export type WGPUBufferHostMappedPointer = {
  chain: WGPUChainedStruct
  pointer: null
  disposeCallback: WGPUCallback
  userdata: null
}
export type WGPUBufferMapCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUBufferMapCallback
  userdata: null
}
export type WGPUColor = {
  r: number
  g: number
  b: number
  a: number
}
export type WGPUColorTargetStateExpandResolveTextureDawn = {
  chain: WGPUChainedStruct
  enabled: boolean
}
export type WGPUCompilationInfoCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUCompilationInfoCallback
  userdata: null
}
export type WGPUComputePassTimestampWrites = {
  querySet: WGPUQuerySet
  beginningOfPassWriteIndex: number
  endOfPassWriteIndex: number
}
export type WGPUCopyTextureForBrowserOptions = {
  nextInChain: WGPUChainedStruct | null
  flipY: boolean
  needsColorSpaceConversion: boolean
  srcAlphaMode: WGPUAlphaMode
  srcTransferFunctionParameters: number | null
  conversionMatrix: number | null
  dstTransferFunctionParameters: number | null
  dstAlphaMode: WGPUAlphaMode
  internalUsage: boolean
}
export type WGPUCreateComputePipelineAsyncCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUCreateComputePipelineAsyncCallback
  userdata: null
}
export type WGPUCreateRenderPipelineAsyncCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUCreateRenderPipelineAsyncCallback
  userdata: null
}
export type WGPUDawnWGSLBlocklist = {
  chain: WGPUChainedStruct
  blocklistedFeatureCount: number
  blocklistedFeatures: string | null
}
export type WGPUDawnAdapterPropertiesPowerPreference = {
  chain: WGPUChainedStructOut
  powerPreference: WGPUPowerPreference
}
export type WGPUDawnBufferDescriptorErrorInfoFromWireClient = {
  chain: WGPUChainedStruct
  outOfMemory: boolean
}
export type WGPUDawnEncoderInternalUsageDescriptor = {
  chain: WGPUChainedStruct
  useInternalUsages: boolean
}
export type WGPUDawnExperimentalImmediateDataLimits = {
  chain: WGPUChainedStructOut
  maxImmediateDataRangeByteSize: number
}
export type WGPUDawnExperimentalSubgroupLimits = {
  chain: WGPUChainedStructOut
  minSubgroupSize: number
  maxSubgroupSize: number
}
export type WGPUDawnRenderPassColorAttachmentRenderToSingleSampled = {
  chain: WGPUChainedStruct
  implicitSampleCount: number
}
export type WGPUDawnShaderModuleSPIRVOptionsDescriptor = {
  chain: WGPUChainedStruct
  allowNonUniformDerivatives: boolean
}
export type WGPUDawnTexelCopyBufferRowAlignmentLimits = {
  chain: WGPUChainedStructOut
  minTexelCopyBufferRowAlignment: number
}
export type WGPUDawnTextureInternalUsageDescriptor = {
  chain: WGPUChainedStruct
  internalUsage: WGPUTextureUsage
}
export type WGPUDawnTogglesDescriptor = {
  chain: WGPUChainedStruct
  enabledToggleCount: number
  enabledToggles: string | null
  disabledToggleCount: number
  disabledToggles: string | null
}
export type WGPUDawnWireWGSLControl = {
  chain: WGPUChainedStruct
  enableExperimental: boolean
  enableUnsafe: boolean
  enableTesting: boolean
}
export type WGPUDeviceLostCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUDeviceLostCallbackNew
  userdata: null
}
export type WGPUDrmFormatProperties = {
  modifier: bigint
  modifierPlaneCount: number
}
export type WGPUExtent2D = {
  width: number
  height: number
}
export type WGPUExtent3D = {
  width: number
  height: number
  depthOrArrayLayers: number
}
export type WGPUExternalTextureBindingEntry = {
  chain: WGPUChainedStruct
  externalTexture: WGPUExternalTexture
}
export type WGPUExternalTextureBindingLayout = {
  chain: WGPUChainedStruct
}
export type WGPUFormatCapabilities = {
  nextInChain: WGPUChainedStructOut | null
}
export type WGPUFuture = {
  id: bigint
}
export type WGPUInstanceFeatures = {
  nextInChain: WGPUChainedStruct | null
  timedWaitAnyEnable: boolean
  timedWaitAnyMaxCount: number
}
export type WGPULimits = {
  maxTextureDimension1D: number
  maxTextureDimension2D: number
  maxTextureDimension3D: number
  maxTextureArrayLayers: number
  maxBindGroups: number
  maxBindGroupsPlusVertexBuffers: number
  maxBindingsPerBindGroup: number
  maxDynamicUniformBuffersPerPipelineLayout: number
  maxDynamicStorageBuffersPerPipelineLayout: number
  maxSampledTexturesPerShaderStage: number
  maxSamplersPerShaderStage: number
  maxStorageBuffersPerShaderStage: number
  maxStorageTexturesPerShaderStage: number
  maxUniformBuffersPerShaderStage: number
  maxUniformBufferBindingSize: bigint
  maxStorageBufferBindingSize: bigint
  minUniformBufferOffsetAlignment: number
  minStorageBufferOffsetAlignment: number
  maxVertexBuffers: number
  maxBufferSize: bigint
  maxVertexAttributes: number
  maxVertexBufferArrayStride: number
  maxInterStageShaderComponents: number
  maxInterStageShaderVariables: number
  maxColorAttachments: number
  maxColorAttachmentBytesPerSample: number
  maxComputeWorkgroupStorageSize: number
  maxComputeInvocationsPerWorkgroup: number
  maxComputeWorkgroupSizeX: number
  maxComputeWorkgroupSizeY: number
  maxComputeWorkgroupSizeZ: number
  maxComputeWorkgroupsPerDimension: number
  maxStorageBuffersInVertexStage: number
  maxStorageTexturesInVertexStage: number
  maxStorageBuffersInFragmentStage: number
  maxStorageTexturesInFragmentStage: number
}
export type WGPUMemoryHeapInfo = {
  properties: WGPUHeapProperty
  size: bigint
}
export type WGPUMultisampleState = {
  nextInChain: WGPUChainedStruct | null
  count: number
  mask: number
  alphaToCoverageEnabled: boolean
}
export type WGPUOrigin2D = {
  x: number
  y: number
}
export type WGPUOrigin3D = {
  x: number
  y: number
  z: number
}
export type WGPUPipelineLayoutStorageAttachment = {
  offset: bigint
  format: WGPUTextureFormat
}
export type WGPUPopErrorScopeCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUPopErrorScopeCallback
  oldCallback: WGPUErrorCallback
  userdata: null
}
export type WGPUPrimitiveState = {
  nextInChain: WGPUChainedStruct | null
  topology: WGPUPrimitiveTopology
  stripIndexFormat: WGPUIndexFormat
  frontFace: WGPUFrontFace
  cullMode: WGPUCullMode
  unclippedDepth: boolean
}
export type WGPUQueueWorkDoneCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPUQueueWorkDoneCallback
  userdata: null
}
export type WGPURenderPassDepthStencilAttachment = {
  view: WGPUTextureView
  depthLoadOp: WGPULoadOp
  depthStoreOp: WGPUStoreOp
  depthClearValue: number
  depthReadOnly: boolean
  stencilLoadOp: WGPULoadOp
  stencilStoreOp: WGPUStoreOp
  stencilClearValue: number
  stencilReadOnly: boolean
}
export type WGPURenderPassDescriptorExpandResolveRect = {
  chain: WGPUChainedStruct
  x: number
  y: number
  width: number
  height: number
}
export type WGPURenderPassMaxDrawCount = {
  chain: WGPUChainedStruct
  maxDrawCount: bigint
}
export type WGPURenderPassTimestampWrites = {
  querySet: WGPUQuerySet
  beginningOfPassWriteIndex: number
  endOfPassWriteIndex: number
}
export type WGPURequestAdapterCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPURequestAdapterCallback
  userdata: null
}
export type WGPURequestAdapterOptions = {
  nextInChain: WGPUChainedStruct | null
  compatibleSurface: WGPUSurface | null
  featureLevel: WGPUFeatureLevel
  powerPreference: WGPUPowerPreference
  backendType: WGPUBackendType
  forceFallbackAdapter: boolean
  compatibilityMode: boolean
}
export type WGPURequestDeviceCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  mode: WGPUCallbackMode
  callback: WGPURequestDeviceCallback
  userdata: null
}
export type WGPUSamplerBindingLayout = {
  nextInChain: WGPUChainedStruct | null
  type: WGPUSamplerBindingType
}
export type WGPUShaderModuleCompilationOptions = {
  chain: WGPUChainedStruct
  strictMath: boolean
}
export type WGPUShaderSourceSPIRV = {
  chain: WGPUChainedStruct
  codeSize: number
  code: number | null
}
export type WGPUSharedBufferMemoryBeginAccessDescriptor = {
  nextInChain: WGPUChainedStruct | null
  initialized: boolean
  fenceCount: number
  fences: WGPUSharedFence | null
  signaledValues: bigint | null
}
export type WGPUSharedBufferMemoryEndAccessState = {
  nextInChain: WGPUChainedStructOut | null
  initialized: boolean
  fenceCount: number
  fences: WGPUSharedFence | null
  signaledValues: bigint | null
}
export type WGPUSharedBufferMemoryProperties = {
  nextInChain: WGPUChainedStructOut | null
  usage: WGPUBufferUsage
  size: bigint
}
export type WGPUSharedFenceDXGISharedHandleDescriptor = {
  chain: WGPUChainedStruct
  handle: null
}
export type WGPUSharedFenceDXGISharedHandleExportInfo = {
  chain: WGPUChainedStructOut
  handle: null
}
export type WGPUSharedFenceMTLSharedEventDescriptor = {
  chain: WGPUChainedStruct
  sharedEvent: null
}
export type WGPUSharedFenceMTLSharedEventExportInfo = {
  chain: WGPUChainedStructOut
  sharedEvent: null
}
export type WGPUSharedFenceExportInfo = {
  nextInChain: WGPUChainedStructOut | null
  type: WGPUSharedFenceType
}
export type WGPUSharedFenceSyncFDDescriptor = {
  chain: WGPUChainedStruct
  handle: number
}
export type WGPUSharedFenceSyncFDExportInfo = {
  chain: WGPUChainedStructOut
  handle: number
}
export type WGPUSharedFenceVkSemaphoreOpaqueFDDescriptor = {
  chain: WGPUChainedStruct
  handle: number
}
export type WGPUSharedFenceVkSemaphoreOpaqueFDExportInfo = {
  chain: WGPUChainedStructOut
  handle: number
}
export type WGPUSharedFenceVkSemaphoreZirconHandleDescriptor = {
  chain: WGPUChainedStruct
  handle: number
}
export type WGPUSharedFenceVkSemaphoreZirconHandleExportInfo = {
  chain: WGPUChainedStructOut
  handle: number
}
export type WGPUSharedTextureMemoryD3DSwapchainBeginState = {
  chain: WGPUChainedStruct
  isSwapchain: boolean
}
export type WGPUSharedTextureMemoryDXGISharedHandleDescriptor = {
  chain: WGPUChainedStruct
  handle: null
  useKeyedMutex: boolean
}
export type WGPUSharedTextureMemoryEGLImageDescriptor = {
  chain: WGPUChainedStruct
  image: null
}
export type WGPUSharedTextureMemoryIOSurfaceDescriptor = {
  chain: WGPUChainedStruct
  ioSurface: null
}
export type WGPUSharedTextureMemoryAHardwareBufferDescriptor = {
  chain: WGPUChainedStruct
  handle: null
  useExternalFormat: boolean
}
export type WGPUSharedTextureMemoryBeginAccessDescriptor = {
  nextInChain: WGPUChainedStruct | null
  concurrentRead: boolean
  initialized: boolean
  fenceCount: number
  fences: WGPUSharedFence | null
  signaledValues: bigint | null
}
export type WGPUSharedTextureMemoryDmaBufPlane = {
  fd: number
  offset: bigint
  stride: number
}
export type WGPUSharedTextureMemoryEndAccessState = {
  nextInChain: WGPUChainedStructOut | null
  initialized: boolean
  fenceCount: number
  fences: WGPUSharedFence | null
  signaledValues: bigint | null
}
export type WGPUSharedTextureMemoryOpaqueFDDescriptor = {
  chain: WGPUChainedStruct
  vkImageCreateInfo: null
  memoryFD: number
  memoryTypeIndex: number
  allocationSize: bigint
  dedicatedAllocation: boolean
}
export type WGPUSharedTextureMemoryVkDedicatedAllocationDescriptor = {
  chain: WGPUChainedStruct
  dedicatedAllocation: boolean
}
export type WGPUSharedTextureMemoryVkImageLayoutBeginState = {
  chain: WGPUChainedStruct
  oldLayout: number
  newLayout: number
}
export type WGPUSharedTextureMemoryVkImageLayoutEndState = {
  chain: WGPUChainedStructOut
  oldLayout: number
  newLayout: number
}
export type WGPUSharedTextureMemoryZirconHandleDescriptor = {
  chain: WGPUChainedStruct
  memoryFD: number
  allocationSize: bigint
}
export type WGPUStaticSamplerBindingLayout = {
  chain: WGPUChainedStruct
  sampler: WGPUSampler
  sampledTextureBinding: number
}
export type WGPUStencilFaceState = {
  compare: WGPUCompareFunction
  failOp: WGPUStencilOperation
  depthFailOp: WGPUStencilOperation
  passOp: WGPUStencilOperation
}
export type WGPUStorageTextureBindingLayout = {
  nextInChain: WGPUChainedStruct | null
  access: WGPUStorageTextureAccess
  format: WGPUTextureFormat
  viewDimension: WGPUTextureViewDimension
}
export type WGPUStringView = {
  data: number | null
  length: number
}
export type WGPUSupportedFeatures = {
  featureCount: number
  features: WGPUFeatureName | null
}
export type WGPUSurfaceCapabilities = {
  nextInChain: WGPUChainedStructOut | null
  usages: WGPUTextureUsage
  formatCount: number
  formats: WGPUTextureFormat | null
  presentModeCount: number
  presentModes: WGPUPresentMode | null
  alphaModeCount: number
  alphaModes: WGPUCompositeAlphaMode | null
}
export type WGPUSurfaceConfiguration = {
  nextInChain: WGPUChainedStruct | null
  device: WGPUDevice
  format: WGPUTextureFormat
  usage: WGPUTextureUsage
  viewFormatCount: number
  viewFormats: WGPUTextureFormat | null
  alphaMode: WGPUCompositeAlphaMode
  width: number
  height: number
  presentMode: WGPUPresentMode
}
export type WGPUSurfaceDescriptorFromWindowsCoreWindow = {
  chain: WGPUChainedStruct
  coreWindow: null
}
export type WGPUSurfaceDescriptorFromWindowsSwapChainPanel = {
  chain: WGPUChainedStruct
  swapChainPanel: null
}
export type WGPUSurfaceSourceXCBWindow = {
  chain: WGPUChainedStruct
  connection: null
  window: number
}
export type WGPUSurfaceSourceAndroidNativeWindow = {
  chain: WGPUChainedStruct
  window: null
}
export type WGPUSurfaceSourceMetalLayer = {
  chain: WGPUChainedStruct
  layer: null
}
export type WGPUSurfaceSourceWaylandSurface = {
  chain: WGPUChainedStruct
  display: null
  surface: null
}
export type WGPUSurfaceSourceWindowsHWND = {
  chain: WGPUChainedStruct
  hinstance: null
  hwnd: null
}
export type WGPUSurfaceSourceXlibWindow = {
  chain: WGPUChainedStruct
  display: null
  window: bigint
}
export type WGPUSurfaceTexture = {
  texture: WGPUTexture
  suboptimal: boolean
  status: WGPUSurfaceGetCurrentTextureStatus
}
export type WGPUTextureBindingLayout = {
  nextInChain: WGPUChainedStruct | null
  sampleType: WGPUTextureSampleType
  viewDimension: WGPUTextureViewDimension
  multisampled: boolean
}
export type WGPUTextureBindingViewDimensionDescriptor = {
  chain: WGPUChainedStruct
  textureBindingViewDimension: WGPUTextureViewDimension
}
export type WGPUTextureDataLayout = {
  nextInChain: WGPUChainedStruct | null
  offset: bigint
  bytesPerRow: number
  rowsPerImage: number
}
export type WGPUUncapturedErrorCallbackInfo = {
  nextInChain: WGPUChainedStruct | null
  callback: WGPUErrorCallback
  userdata: null
}
export type WGPUVertexAttribute = {
  format: WGPUVertexFormat
  offset: bigint
  shaderLocation: number
}
export type WGPUYCbCrVkDescriptor = {
  chain: WGPUChainedStruct
  vkFormat: number
  vkYCbCrModel: number
  vkYCbCrRange: number
  vkComponentSwizzleRed: number
  vkComponentSwizzleGreen: number
  vkComponentSwizzleBlue: number
  vkComponentSwizzleAlpha: number
  vkXChromaOffset: number
  vkYChromaOffset: number
  vkChromaFilter: WGPUFilterMode
  forceExplicitReconstruction: boolean
  externalFormat: bigint
}
export type WGPUAHardwareBufferProperties = {
  yCbCrInfo: WGPUYCbCrVkDescriptor
}
export type WGPUAdapterInfo = {
  nextInChain: WGPUChainedStructOut | null
  vendor: WGPUStringView
  architecture: WGPUStringView
  device: WGPUStringView
  description: WGPUStringView
  backendType: WGPUBackendType
  adapterType: WGPUAdapterType
  vendorID: number
  deviceID: number
  compatibilityMode: boolean
}
export type WGPUAdapterPropertiesMemoryHeaps = {
  chain: WGPUChainedStructOut
  heapCount: number
  heapInfo: WGPUMemoryHeapInfo | null
}
export type WGPUBindGroupDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  layout: WGPUBindGroupLayout
  entryCount: number
  entries: WGPUBindGroupEntry | null
}
export type WGPUBindGroupLayoutEntry = {
  nextInChain: WGPUChainedStruct | null
  binding: number
  visibility: WGPUShaderStage
  buffer: WGPUBufferBindingLayout
  sampler: WGPUSamplerBindingLayout
  texture: WGPUTextureBindingLayout
  storageTexture: WGPUStorageTextureBindingLayout
}
export type WGPUBlendState = {
  color: WGPUBlendComponent
  alpha: WGPUBlendComponent
}
export type WGPUBufferDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  usage: WGPUBufferUsage
  size: bigint
  mappedAtCreation: boolean
}
export type WGPUCommandBufferDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUCommandEncoderDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUCompilationMessage = {
  nextInChain: WGPUChainedStruct | null
  message: WGPUStringView
  type: WGPUCompilationMessageType
  lineNum: bigint
  linePos: bigint
  offset: bigint
  length: bigint
  utf16LinePos: bigint
  utf16Offset: bigint
  utf16Length: bigint
}
export type WGPUComputePassDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  timestampWrites: WGPUComputePassTimestampWrites | null
}
export type WGPUConstantEntry = {
  nextInChain: WGPUChainedStruct | null
  key: WGPUStringView
  value: number
}
export type WGPUDawnCacheDeviceDescriptor = {
  chain: WGPUChainedStruct
  isolationKey: WGPUStringView
  loadDataFunction: WGPUDawnLoadCacheDataFunction
  storeDataFunction: WGPUDawnStoreCacheDataFunction
  functionUserdata: null
}
export type WGPUDepthStencilState = {
  nextInChain: WGPUChainedStruct | null
  format: WGPUTextureFormat
  depthWriteEnabled: WGPUOptionalBool
  depthCompare: WGPUCompareFunction
  stencilFront: WGPUStencilFaceState
  stencilBack: WGPUStencilFaceState
  stencilReadMask: number
  stencilWriteMask: number
  depthBias: number
  depthBiasSlopeScale: number
  depthBiasClamp: number
}
export type WGPUDrmFormatCapabilities = {
  chain: WGPUChainedStructOut
  propertiesCount: number
  properties: WGPUDrmFormatProperties | null
}
export type WGPUExternalTextureDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  plane0: WGPUTextureView
  plane1: WGPUTextureView | null
  cropOrigin: WGPUOrigin2D
  cropSize: WGPUExtent2D
  apparentSize: WGPUExtent2D
  doYuvToRgbConversionOnly: boolean
  yuvToRgbConversionMatrix: number | null
  srcTransferFunctionParameters: number | null
  dstTransferFunctionParameters: number | null
  gamutConversionMatrix: number | null
  mirrored: boolean
  rotation: WGPUExternalTextureRotation
}
export type WGPUFutureWaitInfo = {
  future: WGPUFuture
  completed: boolean
}
export type WGPUImageCopyBuffer = {
  layout: WGPUTextureDataLayout
  buffer: WGPUBuffer
}
export type WGPUImageCopyExternalTexture = {
  nextInChain: WGPUChainedStruct | null
  externalTexture: WGPUExternalTexture
  origin: WGPUOrigin3D
  naturalSize: WGPUExtent2D
}
export type WGPUImageCopyTexture = {
  texture: WGPUTexture
  mipLevel: number
  origin: WGPUOrigin3D
  aspect: WGPUTextureAspect
}
export type WGPUInstanceDescriptor = {
  nextInChain: WGPUChainedStruct | null
  features: WGPUInstanceFeatures
}
export type WGPUPipelineLayoutDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  bindGroupLayoutCount: number
  bindGroupLayouts: WGPUBindGroupLayout | null
  immediateDataRangeByteSize: number
}
export type WGPUPipelineLayoutPixelLocalStorage = {
  chain: WGPUChainedStruct
  totalPixelLocalStorageSize: bigint
  storageAttachmentCount: number
  storageAttachments: WGPUPipelineLayoutStorageAttachment | null
}
export type WGPUQuerySetDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  type: WGPUQueryType
  count: number
}
export type WGPUQueueDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPURenderBundleDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPURenderBundleEncoderDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  colorFormatCount: number
  colorFormats: WGPUTextureFormat | null
  depthStencilFormat: WGPUTextureFormat
  sampleCount: number
  depthReadOnly: boolean
  stencilReadOnly: boolean
}
export type WGPURenderPassColorAttachment = {
  nextInChain: WGPUChainedStruct | null
  view: WGPUTextureView | null
  depthSlice: number
  resolveTarget: WGPUTextureView | null
  loadOp: WGPULoadOp
  storeOp: WGPUStoreOp
  clearValue: WGPUColor
}
export type WGPURenderPassStorageAttachment = {
  nextInChain: WGPUChainedStruct | null
  offset: bigint
  storage: WGPUTextureView
  loadOp: WGPULoadOp
  storeOp: WGPUStoreOp
  clearValue: WGPUColor
}
export type WGPURequiredLimits = {
  nextInChain: WGPUChainedStruct | null
  limits: WGPULimits
}
export type WGPUSamplerDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  addressModeU: WGPUAddressMode
  addressModeV: WGPUAddressMode
  addressModeW: WGPUAddressMode
  magFilter: WGPUFilterMode
  minFilter: WGPUFilterMode
  mipmapFilter: WGPUMipmapFilterMode
  lodMinClamp: number
  lodMaxClamp: number
  compare: WGPUCompareFunction
  maxAnisotropy: number
}
export type WGPUShaderModuleDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUShaderSourceWGSL = {
  chain: WGPUChainedStruct
  code: WGPUStringView
}
export type WGPUSharedBufferMemoryDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUSharedFenceDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUSharedTextureMemoryAHardwareBufferProperties = {
  chain: WGPUChainedStructOut
  yCbCrInfo: WGPUYCbCrVkDescriptor
}
export type WGPUSharedTextureMemoryDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUSharedTextureMemoryDmaBufDescriptor = {
  chain: WGPUChainedStruct
  size: WGPUExtent3D
  drmFormat: number
  drmModifier: bigint
  planeCount: number
  planes: WGPUSharedTextureMemoryDmaBufPlane | null
}
export type WGPUSharedTextureMemoryProperties = {
  nextInChain: WGPUChainedStructOut | null
  usage: WGPUTextureUsage
  size: WGPUExtent3D
  format: WGPUTextureFormat
}
export type WGPUSupportedLimits = {
  nextInChain: WGPUChainedStructOut | null
  limits: WGPULimits
}
export type WGPUSurfaceDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
}
export type WGPUSurfaceSourceCanvasHTMLSelector_Emscripten = {
  chain: WGPUChainedStruct
  selector: WGPUStringView
}
export type WGPUTextureDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  usage: WGPUTextureUsage
  dimension: WGPUTextureDimension
  size: WGPUExtent3D
  format: WGPUTextureFormat
  mipLevelCount: number
  sampleCount: number
  viewFormatCount: number
  viewFormats: WGPUTextureFormat | null
}
export type WGPUTextureViewDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  format: WGPUTextureFormat
  dimension: WGPUTextureViewDimension
  baseMipLevel: number
  mipLevelCount: number
  baseArrayLayer: number
  arrayLayerCount: number
  aspect: WGPUTextureAspect
  usage: WGPUTextureUsage
}
export type WGPUVertexBufferLayout = {
  arrayStride: bigint
  stepMode: WGPUVertexStepMode
  attributeCount: number
  attributes: WGPUVertexAttribute | null
}
export type WGPUBindGroupLayoutDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  entryCount: number
  entries: WGPUBindGroupLayoutEntry | null
}
export type WGPUColorTargetState = {
  nextInChain: WGPUChainedStruct | null
  format: WGPUTextureFormat
  blend: WGPUBlendState | null
  writeMask: WGPUColorWriteMask
}
export type WGPUCompilationInfo = {
  nextInChain: WGPUChainedStruct | null
  messageCount: number
  messages: WGPUCompilationMessage | null
}
export type WGPUComputeState = {
  nextInChain: WGPUChainedStruct | null
  module: WGPUShaderModule
  entryPoint: WGPUStringView
  constantCount: number
  constants: WGPUConstantEntry | null
}
export type WGPUDeviceDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  requiredFeatureCount: number
  requiredFeatures: WGPUFeatureName | null
  requiredLimits: WGPURequiredLimits | null
  defaultQueue: WGPUQueueDescriptor
  deviceLostCallbackInfo2: WGPUDeviceLostCallbackInfo2
  uncapturedErrorCallbackInfo2: WGPUUncapturedErrorCallbackInfo2
}
export type WGPURenderPassDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  colorAttachmentCount: number
  colorAttachments: WGPURenderPassColorAttachment | null
  depthStencilAttachment: WGPURenderPassDepthStencilAttachment | null
  occlusionQuerySet: WGPUQuerySet | null
  timestampWrites: WGPURenderPassTimestampWrites | null
}
export type WGPURenderPassPixelLocalStorage = {
  chain: WGPUChainedStruct
  totalPixelLocalStorageSize: bigint
  storageAttachmentCount: number
  storageAttachments: WGPURenderPassStorageAttachment | null
}
export type WGPUVertexState = {
  nextInChain: WGPUChainedStruct | null
  module: WGPUShaderModule
  entryPoint: WGPUStringView
  constantCount: number
  constants: WGPUConstantEntry | null
  bufferCount: number
  buffers: WGPUVertexBufferLayout | null
}
export type WGPUComputePipelineDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  layout: WGPUPipelineLayout | null
  compute: WGPUComputeState
}
export type WGPUFragmentState = {
  nextInChain: WGPUChainedStruct | null
  module: WGPUShaderModule
  entryPoint: WGPUStringView
  constantCount: number
  constants: WGPUConstantEntry | null
  targetCount: number
  targets: WGPUColorTargetState | null
}
export type WGPURenderPipelineDescriptor = {
  nextInChain: WGPUChainedStruct | null
  label: WGPUStringView
  layout: WGPUPipelineLayout | null
  vertex: WGPUVertexState
  primitive: WGPUPrimitiveState
  depthStencil: WGPUDepthStencilState | null
  multisample: WGPUMultisampleState
  fragment: WGPUFragmentState | null
}

// FUNCTIONS
export function emscripten_webgpu_get_device(): WGPUDevice {
 return webgpu.symbols.emscripten_webgpu_get_device() 
}
export function wgpuAdapterInfoFreeMembers(value:WGPUAdapterInfo): null {
 return webgpu.symbols.wgpuAdapterInfoFreeMembers(value) 
}
export function wgpuAdapterPropertiesMemoryHeapsFreeMembers(value:WGPUAdapterPropertiesMemoryHeaps): null {
 return webgpu.symbols.wgpuAdapterPropertiesMemoryHeapsFreeMembers(value) 
}
export function wgpuCreateInstance(descriptor:WGPUInstanceDescriptor | null): WGPUInstance {
 return webgpu.symbols.wgpuCreateInstance(descriptor) 
}
export function wgpuDrmFormatCapabilitiesFreeMembers(value:WGPUDrmFormatCapabilities): null {
 return webgpu.symbols.wgpuDrmFormatCapabilitiesFreeMembers(value) 
}
export function wgpuGetInstanceFeatures(features:WGPUInstanceFeatures | null): WGPUStatus {
 return webgpu.symbols.wgpuGetInstanceFeatures(features) 
}
export function wgpuGetProcAddress(procName:WGPUStringView): WGPUProc {
 return webgpu.symbols.wgpuGetProcAddress(procName) 
}
export function wgpuSharedBufferMemoryEndAccessStateFreeMembers(value:WGPUSharedBufferMemoryEndAccessState): null {
 return webgpu.symbols.wgpuSharedBufferMemoryEndAccessStateFreeMembers(value) 
}
export function wgpuSharedTextureMemoryEndAccessStateFreeMembers(value:WGPUSharedTextureMemoryEndAccessState): null {
 return webgpu.symbols.wgpuSharedTextureMemoryEndAccessStateFreeMembers(value) 
}
export function wgpuSupportedFeaturesFreeMembers(value:WGPUSupportedFeatures): null {
 return webgpu.symbols.wgpuSupportedFeaturesFreeMembers(value) 
}
export function wgpuSurfaceCapabilitiesFreeMembers(value:WGPUSurfaceCapabilities): null {
 return webgpu.symbols.wgpuSurfaceCapabilitiesFreeMembers(value) 
}
export function wgpuAdapterCreateDevice(adapter:WGPUAdapter, descriptor:WGPUDeviceDescriptor | null): WGPUDevice {
 return webgpu.symbols.wgpuAdapterCreateDevice(adapter, descriptor) 
}
export function wgpuAdapterGetFeatures(adapter:WGPUAdapter, features:WGPUSupportedFeatures | null): null {
 return webgpu.symbols.wgpuAdapterGetFeatures(adapter, features) 
}
export function wgpuAdapterGetFormatCapabilities(adapter:WGPUAdapter, format:WGPUTextureFormat, capabilities:WGPUFormatCapabilities | null): WGPUStatus {
 return webgpu.symbols.wgpuAdapterGetFormatCapabilities(adapter, format, capabilities) 
}
export function wgpuAdapterGetInfo(adapter:WGPUAdapter, info:WGPUAdapterInfo | null): WGPUStatus {
 return webgpu.symbols.wgpuAdapterGetInfo(adapter, info) 
}
export function wgpuAdapterGetInstance(adapter:WGPUAdapter): WGPUInstance {
 return webgpu.symbols.wgpuAdapterGetInstance(adapter) 
}
export function wgpuAdapterGetLimits(adapter:WGPUAdapter, limits:WGPUSupportedLimits | null): WGPUStatus {
 return webgpu.symbols.wgpuAdapterGetLimits(adapter, limits) 
}
export function wgpuAdapterHasFeature(adapter:WGPUAdapter, feature:WGPUFeatureName): boolean {
 return webgpu.symbols.wgpuAdapterHasFeature(adapter, feature) 
}
export function wgpuAdapterRequestDevice(adapter:WGPUAdapter, descriptor:WGPUDeviceDescriptor | null, callback:WGPURequestDeviceCallback, userdata:null): null {
 return webgpu.symbols.wgpuAdapterRequestDevice(adapter, descriptor, callback, userdata) 
}
export function wgpuAdapterRequestDevice2(adapter:WGPUAdapter, options:WGPUDeviceDescriptor | null, callbackInfo:WGPURequestDeviceCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuAdapterRequestDevice2(adapter, options, callbackInfo) 
}
export function wgpuAdapterRequestDeviceF(adapter:WGPUAdapter, options:WGPUDeviceDescriptor | null, callbackInfo:WGPURequestDeviceCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuAdapterRequestDeviceF(adapter, options, callbackInfo) 
}
export function wgpuAdapterAddRef(adapter:WGPUAdapter): null {
 return webgpu.symbols.wgpuAdapterAddRef(adapter) 
}
export function wgpuAdapterRelease(adapter:WGPUAdapter): null {
 return webgpu.symbols.wgpuAdapterRelease(adapter) 
}
export function wgpuBindGroupSetLabel(bindGroup:WGPUBindGroup, label:WGPUStringView): null {
 return webgpu.symbols.wgpuBindGroupSetLabel(bindGroup, label) 
}
export function wgpuBindGroupAddRef(bindGroup:WGPUBindGroup): null {
 return webgpu.symbols.wgpuBindGroupAddRef(bindGroup) 
}
export function wgpuBindGroupRelease(bindGroup:WGPUBindGroup): null {
 return webgpu.symbols.wgpuBindGroupRelease(bindGroup) 
}
export function wgpuBindGroupLayoutSetLabel(bindGroupLayout:WGPUBindGroupLayout, label:WGPUStringView): null {
 return webgpu.symbols.wgpuBindGroupLayoutSetLabel(bindGroupLayout, label) 
}
export function wgpuBindGroupLayoutAddRef(bindGroupLayout:WGPUBindGroupLayout): null {
 return webgpu.symbols.wgpuBindGroupLayoutAddRef(bindGroupLayout) 
}
export function wgpuBindGroupLayoutRelease(bindGroupLayout:WGPUBindGroupLayout): null {
 return webgpu.symbols.wgpuBindGroupLayoutRelease(bindGroupLayout) 
}
export function wgpuBufferDestroy(buffer:WGPUBuffer): null {
 return webgpu.symbols.wgpuBufferDestroy(buffer) 
}
export function wgpuBufferGetConstMappedRange(buffer:WGPUBuffer, offset:number, size:number): null {
 return webgpu.symbols.wgpuBufferGetConstMappedRange(buffer, offset, size) 
}
export function wgpuBufferGetMapState(buffer:WGPUBuffer): WGPUBufferMapState {
 return webgpu.symbols.wgpuBufferGetMapState(buffer) 
}
export function wgpuBufferGetMappedRange(buffer:WGPUBuffer, offset:number, size:number): null {
 return webgpu.symbols.wgpuBufferGetMappedRange(buffer, offset, size) 
}
export function wgpuBufferGetSize(buffer:WGPUBuffer): bigint {
 return webgpu.symbols.wgpuBufferGetSize(buffer) 
}
export function wgpuBufferGetUsage(buffer:WGPUBuffer): WGPUBufferUsage {
 return webgpu.symbols.wgpuBufferGetUsage(buffer) 
}
export function wgpuBufferMapAsync(buffer:WGPUBuffer, mode:WGPUMapMode, offset:number, size:number, callback:WGPUBufferMapCallback, userdata:null): null {
 return webgpu.symbols.wgpuBufferMapAsync(buffer, mode, offset, size, callback, userdata) 
}
export function wgpuBufferMapAsync2(buffer:WGPUBuffer, mode:WGPUMapMode, offset:number, size:number, callbackInfo:WGPUBufferMapCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuBufferMapAsync2(buffer, mode, offset, size, callbackInfo) 
}
export function wgpuBufferMapAsyncF(buffer:WGPUBuffer, mode:WGPUMapMode, offset:number, size:number, callbackInfo:WGPUBufferMapCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuBufferMapAsyncF(buffer, mode, offset, size, callbackInfo) 
}
export function wgpuBufferSetLabel(buffer:WGPUBuffer, label:WGPUStringView): null {
 return webgpu.symbols.wgpuBufferSetLabel(buffer, label) 
}
export function wgpuBufferUnmap(buffer:WGPUBuffer): null {
 return webgpu.symbols.wgpuBufferUnmap(buffer) 
}
export function wgpuBufferAddRef(buffer:WGPUBuffer): null {
 return webgpu.symbols.wgpuBufferAddRef(buffer) 
}
export function wgpuBufferRelease(buffer:WGPUBuffer): null {
 return webgpu.symbols.wgpuBufferRelease(buffer) 
}
export function wgpuCommandBufferSetLabel(commandBuffer:WGPUCommandBuffer, label:WGPUStringView): null {
 return webgpu.symbols.wgpuCommandBufferSetLabel(commandBuffer, label) 
}
export function wgpuCommandBufferAddRef(commandBuffer:WGPUCommandBuffer): null {
 return webgpu.symbols.wgpuCommandBufferAddRef(commandBuffer) 
}
export function wgpuCommandBufferRelease(commandBuffer:WGPUCommandBuffer): null {
 return webgpu.symbols.wgpuCommandBufferRelease(commandBuffer) 
}
export function wgpuCommandEncoderBeginComputePass(commandEncoder:WGPUCommandEncoder, descriptor:WGPUComputePassDescriptor | null): WGPUComputePassEncoder {
 return webgpu.symbols.wgpuCommandEncoderBeginComputePass(commandEncoder, descriptor) 
}
export function wgpuCommandEncoderBeginRenderPass(commandEncoder:WGPUCommandEncoder, descriptor:WGPURenderPassDescriptor | null): WGPURenderPassEncoder {
 return webgpu.symbols.wgpuCommandEncoderBeginRenderPass(commandEncoder, descriptor) 
}
export function wgpuCommandEncoderClearBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, offset:bigint, size:bigint): null {
 return webgpu.symbols.wgpuCommandEncoderClearBuffer(commandEncoder, buffer, offset, size) 
}
export function wgpuCommandEncoderCopyBufferToBuffer(commandEncoder:WGPUCommandEncoder, source:WGPUBuffer, sourceOffset:bigint, destination:WGPUBuffer, destinationOffset:bigint, size:bigint): null {
 return webgpu.symbols.wgpuCommandEncoderCopyBufferToBuffer(commandEncoder, source, sourceOffset, destination, destinationOffset, size) 
}
export function wgpuCommandEncoderCopyBufferToTexture(commandEncoder:WGPUCommandEncoder, source:WGPUImageCopyBuffer | null, destination:WGPUImageCopyTexture | null, copySize:WGPUExtent3D | null): null {
 return webgpu.symbols.wgpuCommandEncoderCopyBufferToTexture(commandEncoder, source, destination, copySize) 
}
export function wgpuCommandEncoderCopyTextureToBuffer(commandEncoder:WGPUCommandEncoder, source:WGPUImageCopyTexture | null, destination:WGPUImageCopyBuffer | null, copySize:WGPUExtent3D | null): null {
 return webgpu.symbols.wgpuCommandEncoderCopyTextureToBuffer(commandEncoder, source, destination, copySize) 
}
export function wgpuCommandEncoderCopyTextureToTexture(commandEncoder:WGPUCommandEncoder, source:WGPUImageCopyTexture | null, destination:WGPUImageCopyTexture | null, copySize:WGPUExtent3D | null): null {
 return webgpu.symbols.wgpuCommandEncoderCopyTextureToTexture(commandEncoder, source, destination, copySize) 
}
export function wgpuCommandEncoderFinish(commandEncoder:WGPUCommandEncoder, descriptor:WGPUCommandBufferDescriptor | null): WGPUCommandBuffer {
 return webgpu.symbols.wgpuCommandEncoderFinish(commandEncoder, descriptor) 
}
export function wgpuCommandEncoderInjectValidationError(commandEncoder:WGPUCommandEncoder, message:WGPUStringView): null {
 return webgpu.symbols.wgpuCommandEncoderInjectValidationError(commandEncoder, message) 
}
export function wgpuCommandEncoderInsertDebugMarker(commandEncoder:WGPUCommandEncoder, markerLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuCommandEncoderInsertDebugMarker(commandEncoder, markerLabel) 
}
export function wgpuCommandEncoderPopDebugGroup(commandEncoder:WGPUCommandEncoder): null {
 return webgpu.symbols.wgpuCommandEncoderPopDebugGroup(commandEncoder) 
}
export function wgpuCommandEncoderPushDebugGroup(commandEncoder:WGPUCommandEncoder, groupLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuCommandEncoderPushDebugGroup(commandEncoder, groupLabel) 
}
export function wgpuCommandEncoderResolveQuerySet(commandEncoder:WGPUCommandEncoder, querySet:WGPUQuerySet, firstQuery:number, queryCount:number, destination:WGPUBuffer, destinationOffset:bigint): null {
 return webgpu.symbols.wgpuCommandEncoderResolveQuerySet(commandEncoder, querySet, firstQuery, queryCount, destination, destinationOffset) 
}
export function wgpuCommandEncoderSetLabel(commandEncoder:WGPUCommandEncoder, label:WGPUStringView): null {
 return webgpu.symbols.wgpuCommandEncoderSetLabel(commandEncoder, label) 
}
export function wgpuCommandEncoderWriteBuffer(commandEncoder:WGPUCommandEncoder, buffer:WGPUBuffer, bufferOffset:bigint, data:uint8_t | null, size:bigint): null {
 return webgpu.symbols.wgpuCommandEncoderWriteBuffer(commandEncoder, buffer, bufferOffset, data, size) 
}
export function wgpuCommandEncoderWriteTimestamp(commandEncoder:WGPUCommandEncoder, querySet:WGPUQuerySet, queryIndex:number): null {
 return webgpu.symbols.wgpuCommandEncoderWriteTimestamp(commandEncoder, querySet, queryIndex) 
}
export function wgpuCommandEncoderAddRef(commandEncoder:WGPUCommandEncoder): null {
 return webgpu.symbols.wgpuCommandEncoderAddRef(commandEncoder) 
}
export function wgpuCommandEncoderRelease(commandEncoder:WGPUCommandEncoder): null {
 return webgpu.symbols.wgpuCommandEncoderRelease(commandEncoder) 
}
export function wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder:WGPUComputePassEncoder, workgroupCountX:number, workgroupCountY:number, workgroupCountZ:number): null {
 return webgpu.symbols.wgpuComputePassEncoderDispatchWorkgroups(computePassEncoder, workgroupCountX, workgroupCountY, workgroupCountZ) 
}
export function wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder:WGPUComputePassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint): null {
 return webgpu.symbols.wgpuComputePassEncoderDispatchWorkgroupsIndirect(computePassEncoder, indirectBuffer, indirectOffset) 
}
export function wgpuComputePassEncoderEnd(computePassEncoder:WGPUComputePassEncoder): null {
 return webgpu.symbols.wgpuComputePassEncoderEnd(computePassEncoder) 
}
export function wgpuComputePassEncoderInsertDebugMarker(computePassEncoder:WGPUComputePassEncoder, markerLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuComputePassEncoderInsertDebugMarker(computePassEncoder, markerLabel) 
}
export function wgpuComputePassEncoderPopDebugGroup(computePassEncoder:WGPUComputePassEncoder): null {
 return webgpu.symbols.wgpuComputePassEncoderPopDebugGroup(computePassEncoder) 
}
export function wgpuComputePassEncoderPushDebugGroup(computePassEncoder:WGPUComputePassEncoder, groupLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuComputePassEncoderPushDebugGroup(computePassEncoder, groupLabel) 
}
export function wgpuComputePassEncoderSetBindGroup(computePassEncoder:WGPUComputePassEncoder, groupIndex:number, group:WGPUBindGroup | null, dynamicOffsetCount:number, dynamicOffsets:number | null): null {
 return webgpu.symbols.wgpuComputePassEncoderSetBindGroup(computePassEncoder, groupIndex, group, dynamicOffsetCount, dynamicOffsets) 
}
export function wgpuComputePassEncoderSetLabel(computePassEncoder:WGPUComputePassEncoder, label:WGPUStringView): null {
 return webgpu.symbols.wgpuComputePassEncoderSetLabel(computePassEncoder, label) 
}
export function wgpuComputePassEncoderSetPipeline(computePassEncoder:WGPUComputePassEncoder, pipeline:WGPUComputePipeline): null {
 return webgpu.symbols.wgpuComputePassEncoderSetPipeline(computePassEncoder, pipeline) 
}
export function wgpuComputePassEncoderWriteTimestamp(computePassEncoder:WGPUComputePassEncoder, querySet:WGPUQuerySet, queryIndex:number): null {
 return webgpu.symbols.wgpuComputePassEncoderWriteTimestamp(computePassEncoder, querySet, queryIndex) 
}
export function wgpuComputePassEncoderAddRef(computePassEncoder:WGPUComputePassEncoder): null {
 return webgpu.symbols.wgpuComputePassEncoderAddRef(computePassEncoder) 
}
export function wgpuComputePassEncoderRelease(computePassEncoder:WGPUComputePassEncoder): null {
 return webgpu.symbols.wgpuComputePassEncoderRelease(computePassEncoder) 
}
export function wgpuComputePipelineGetBindGroupLayout(computePipeline:WGPUComputePipeline, groupIndex:number): WGPUBindGroupLayout {
 return webgpu.symbols.wgpuComputePipelineGetBindGroupLayout(computePipeline, groupIndex) 
}
export function wgpuComputePipelineSetLabel(computePipeline:WGPUComputePipeline, label:WGPUStringView): null {
 return webgpu.symbols.wgpuComputePipelineSetLabel(computePipeline, label) 
}
export function wgpuComputePipelineAddRef(computePipeline:WGPUComputePipeline): null {
 return webgpu.symbols.wgpuComputePipelineAddRef(computePipeline) 
}
export function wgpuComputePipelineRelease(computePipeline:WGPUComputePipeline): null {
 return webgpu.symbols.wgpuComputePipelineRelease(computePipeline) 
}
export function wgpuDeviceCreateBindGroup(device:WGPUDevice, descriptor:WGPUBindGroupDescriptor | null): WGPUBindGroup {
 return webgpu.symbols.wgpuDeviceCreateBindGroup(device, descriptor) 
}
export function wgpuDeviceCreateBindGroupLayout(device:WGPUDevice, descriptor:WGPUBindGroupLayoutDescriptor | null): WGPUBindGroupLayout {
 return webgpu.symbols.wgpuDeviceCreateBindGroupLayout(device, descriptor) 
}
export function wgpuDeviceCreateBuffer(device:WGPUDevice, descriptor:WGPUBufferDescriptor | null): WGPUBuffer {
 return webgpu.symbols.wgpuDeviceCreateBuffer(device, descriptor) 
}
export function wgpuDeviceCreateCommandEncoder(device:WGPUDevice, descriptor:WGPUCommandEncoderDescriptor | null): WGPUCommandEncoder {
 return webgpu.symbols.wgpuDeviceCreateCommandEncoder(device, descriptor) 
}
export function wgpuDeviceCreateComputePipeline(device:WGPUDevice, descriptor:WGPUComputePipelineDescriptor | null): WGPUComputePipeline {
 return webgpu.symbols.wgpuDeviceCreateComputePipeline(device, descriptor) 
}
export function wgpuDeviceCreateComputePipelineAsync(device:WGPUDevice, descriptor:WGPUComputePipelineDescriptor | null, callback:WGPUCreateComputePipelineAsyncCallback, userdata:null): null {
 return webgpu.symbols.wgpuDeviceCreateComputePipelineAsync(device, descriptor, callback, userdata) 
}
export function wgpuDeviceCreateComputePipelineAsync2(device:WGPUDevice, descriptor:WGPUComputePipelineDescriptor | null, callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuDeviceCreateComputePipelineAsync2(device, descriptor, callbackInfo) 
}
export function wgpuDeviceCreateComputePipelineAsyncF(device:WGPUDevice, descriptor:WGPUComputePipelineDescriptor | null, callbackInfo:WGPUCreateComputePipelineAsyncCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuDeviceCreateComputePipelineAsyncF(device, descriptor, callbackInfo) 
}
export function wgpuDeviceCreateErrorBuffer(device:WGPUDevice, descriptor:WGPUBufferDescriptor | null): WGPUBuffer {
 return webgpu.symbols.wgpuDeviceCreateErrorBuffer(device, descriptor) 
}
export function wgpuDeviceCreateErrorExternalTexture(device:WGPUDevice): WGPUExternalTexture {
 return webgpu.symbols.wgpuDeviceCreateErrorExternalTexture(device) 
}
export function wgpuDeviceCreateErrorShaderModule(device:WGPUDevice, descriptor:WGPUShaderModuleDescriptor | null, errorMessage:WGPUStringView): WGPUShaderModule {
 return webgpu.symbols.wgpuDeviceCreateErrorShaderModule(device, descriptor, errorMessage) 
}
export function wgpuDeviceCreateErrorTexture(device:WGPUDevice, descriptor:WGPUTextureDescriptor | null): WGPUTexture {
 return webgpu.symbols.wgpuDeviceCreateErrorTexture(device, descriptor) 
}
export function wgpuDeviceCreateExternalTexture(device:WGPUDevice, externalTextureDescriptor:WGPUExternalTextureDescriptor | null): WGPUExternalTexture {
 return webgpu.symbols.wgpuDeviceCreateExternalTexture(device, externalTextureDescriptor) 
}
export function wgpuDeviceCreatePipelineLayout(device:WGPUDevice, descriptor:WGPUPipelineLayoutDescriptor | null): WGPUPipelineLayout {
 return webgpu.symbols.wgpuDeviceCreatePipelineLayout(device, descriptor) 
}
export function wgpuDeviceCreateQuerySet(device:WGPUDevice, descriptor:WGPUQuerySetDescriptor | null): WGPUQuerySet {
 return webgpu.symbols.wgpuDeviceCreateQuerySet(device, descriptor) 
}
export function wgpuDeviceCreateRenderBundleEncoder(device:WGPUDevice, descriptor:WGPURenderBundleEncoderDescriptor | null): WGPURenderBundleEncoder {
 return webgpu.symbols.wgpuDeviceCreateRenderBundleEncoder(device, descriptor) 
}
export function wgpuDeviceCreateRenderPipeline(device:WGPUDevice, descriptor:WGPURenderPipelineDescriptor | null): WGPURenderPipeline {
 return webgpu.symbols.wgpuDeviceCreateRenderPipeline(device, descriptor) 
}
export function wgpuDeviceCreateRenderPipelineAsync(device:WGPUDevice, descriptor:WGPURenderPipelineDescriptor | null, callback:WGPUCreateRenderPipelineAsyncCallback, userdata:null): null {
 return webgpu.symbols.wgpuDeviceCreateRenderPipelineAsync(device, descriptor, callback, userdata) 
}
export function wgpuDeviceCreateRenderPipelineAsync2(device:WGPUDevice, descriptor:WGPURenderPipelineDescriptor | null, callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuDeviceCreateRenderPipelineAsync2(device, descriptor, callbackInfo) 
}
export function wgpuDeviceCreateRenderPipelineAsyncF(device:WGPUDevice, descriptor:WGPURenderPipelineDescriptor | null, callbackInfo:WGPUCreateRenderPipelineAsyncCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuDeviceCreateRenderPipelineAsyncF(device, descriptor, callbackInfo) 
}
export function wgpuDeviceCreateSampler(device:WGPUDevice, descriptor:WGPUSamplerDescriptor | null): WGPUSampler {
 return webgpu.symbols.wgpuDeviceCreateSampler(device, descriptor) 
}
export function wgpuDeviceCreateShaderModule(device:WGPUDevice, descriptor:WGPUShaderModuleDescriptor | null): WGPUShaderModule {
 return webgpu.symbols.wgpuDeviceCreateShaderModule(device, descriptor) 
}
export function wgpuDeviceCreateTexture(device:WGPUDevice, descriptor:WGPUTextureDescriptor | null): WGPUTexture {
 return webgpu.symbols.wgpuDeviceCreateTexture(device, descriptor) 
}
export function wgpuDeviceDestroy(device:WGPUDevice): null {
 return webgpu.symbols.wgpuDeviceDestroy(device) 
}
export function wgpuDeviceForceLoss(device:WGPUDevice, type:WGPUDeviceLostReason, message:WGPUStringView): null {
 return webgpu.symbols.wgpuDeviceForceLoss(device, type, message) 
}
export function wgpuDeviceGetAHardwareBufferProperties(device:WGPUDevice, handle:null, properties:WGPUAHardwareBufferProperties | null): WGPUStatus {
 return webgpu.symbols.wgpuDeviceGetAHardwareBufferProperties(device, handle, properties) 
}
export function wgpuDeviceGetAdapter(device:WGPUDevice): WGPUAdapter {
 return webgpu.symbols.wgpuDeviceGetAdapter(device) 
}
export function wgpuDeviceGetAdapterInfo(device:WGPUDevice, adapterInfo:WGPUAdapterInfo | null): WGPUStatus {
 return webgpu.symbols.wgpuDeviceGetAdapterInfo(device, adapterInfo) 
}
export function wgpuDeviceGetFeatures(device:WGPUDevice, features:WGPUSupportedFeatures | null): null {
 return webgpu.symbols.wgpuDeviceGetFeatures(device, features) 
}
export function wgpuDeviceGetLimits(device:WGPUDevice, limits:WGPUSupportedLimits | null): WGPUStatus {
 return webgpu.symbols.wgpuDeviceGetLimits(device, limits) 
}
export function wgpuDeviceGetLostFuture(device:WGPUDevice): WGPUFuture {
 return webgpu.symbols.wgpuDeviceGetLostFuture(device) 
}
export function wgpuDeviceGetQueue(device:WGPUDevice): WGPUQueue {
 return webgpu.symbols.wgpuDeviceGetQueue(device) 
}
export function wgpuDeviceHasFeature(device:WGPUDevice, feature:WGPUFeatureName): boolean {
 return webgpu.symbols.wgpuDeviceHasFeature(device, feature) 
}
export function wgpuDeviceImportSharedBufferMemory(device:WGPUDevice, descriptor:WGPUSharedBufferMemoryDescriptor | null): WGPUSharedBufferMemory {
 return webgpu.symbols.wgpuDeviceImportSharedBufferMemory(device, descriptor) 
}
export function wgpuDeviceImportSharedFence(device:WGPUDevice, descriptor:WGPUSharedFenceDescriptor | null): WGPUSharedFence {
 return webgpu.symbols.wgpuDeviceImportSharedFence(device, descriptor) 
}
export function wgpuDeviceImportSharedTextureMemory(device:WGPUDevice, descriptor:WGPUSharedTextureMemoryDescriptor | null): WGPUSharedTextureMemory {
 return webgpu.symbols.wgpuDeviceImportSharedTextureMemory(device, descriptor) 
}
export function wgpuDeviceInjectError(device:WGPUDevice, type:WGPUErrorType, message:WGPUStringView): null {
 return webgpu.symbols.wgpuDeviceInjectError(device, type, message) 
}
export function wgpuDevicePopErrorScope(device:WGPUDevice, oldCallback:WGPUErrorCallback, userdata:null): null {
 return webgpu.symbols.wgpuDevicePopErrorScope(device, oldCallback, userdata) 
}
export function wgpuDevicePopErrorScope2(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuDevicePopErrorScope2(device, callbackInfo) 
}
export function wgpuDevicePopErrorScopeF(device:WGPUDevice, callbackInfo:WGPUPopErrorScopeCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuDevicePopErrorScopeF(device, callbackInfo) 
}
export function wgpuDevicePushErrorScope(device:WGPUDevice, filter:WGPUErrorFilter): null {
 return webgpu.symbols.wgpuDevicePushErrorScope(device, filter) 
}
export function wgpuDeviceSetLabel(device:WGPUDevice, label:WGPUStringView): null {
 return webgpu.symbols.wgpuDeviceSetLabel(device, label) 
}
export function wgpuDeviceSetLoggingCallback(device:WGPUDevice, callback:WGPULoggingCallback, userdata:null): null {
 return webgpu.symbols.wgpuDeviceSetLoggingCallback(device, callback, userdata) 
}
export function wgpuDeviceTick(device:WGPUDevice): null {
 return webgpu.symbols.wgpuDeviceTick(device) 
}
export function wgpuDeviceValidateTextureDescriptor(device:WGPUDevice, descriptor:WGPUTextureDescriptor | null): null {
 return webgpu.symbols.wgpuDeviceValidateTextureDescriptor(device, descriptor) 
}
export function wgpuDeviceAddRef(device:WGPUDevice): null {
 return webgpu.symbols.wgpuDeviceAddRef(device) 
}
export function wgpuDeviceRelease(device:WGPUDevice): null {
 return webgpu.symbols.wgpuDeviceRelease(device) 
}
export function wgpuExternalTextureDestroy(externalTexture:WGPUExternalTexture): null {
 return webgpu.symbols.wgpuExternalTextureDestroy(externalTexture) 
}
export function wgpuExternalTextureExpire(externalTexture:WGPUExternalTexture): null {
 return webgpu.symbols.wgpuExternalTextureExpire(externalTexture) 
}
export function wgpuExternalTextureRefresh(externalTexture:WGPUExternalTexture): null {
 return webgpu.symbols.wgpuExternalTextureRefresh(externalTexture) 
}
export function wgpuExternalTextureSetLabel(externalTexture:WGPUExternalTexture, label:WGPUStringView): null {
 return webgpu.symbols.wgpuExternalTextureSetLabel(externalTexture, label) 
}
export function wgpuExternalTextureAddRef(externalTexture:WGPUExternalTexture): null {
 return webgpu.symbols.wgpuExternalTextureAddRef(externalTexture) 
}
export function wgpuExternalTextureRelease(externalTexture:WGPUExternalTexture): null {
 return webgpu.symbols.wgpuExternalTextureRelease(externalTexture) 
}
export function wgpuInstanceCreateSurface(instance:WGPUInstance, descriptor:WGPUSurfaceDescriptor | null): WGPUSurface {
 return webgpu.symbols.wgpuInstanceCreateSurface(instance, descriptor) 
}
export function wgpuInstanceEnumerateWGSLLanguageFeatures(instance:WGPUInstance, features:WGPUWGSLFeatureName | null): number {
 return webgpu.symbols.wgpuInstanceEnumerateWGSLLanguageFeatures(instance, features) 
}
export function wgpuInstanceHasWGSLLanguageFeature(instance:WGPUInstance, feature:WGPUWGSLFeatureName): boolean {
 return webgpu.symbols.wgpuInstanceHasWGSLLanguageFeature(instance, feature) 
}
export function wgpuInstanceProcessEvents(instance:WGPUInstance): null {
 return webgpu.symbols.wgpuInstanceProcessEvents(instance) 
}
export function wgpuInstanceRequestAdapter(instance:WGPUInstance, options:WGPURequestAdapterOptions | null, callback:WGPURequestAdapterCallback, userdata:null): null {
 return webgpu.symbols.wgpuInstanceRequestAdapter(instance, options, callback, userdata) 
}
export function wgpuInstanceRequestAdapter2(instance:WGPUInstance, options:WGPURequestAdapterOptions | null, callbackInfo:WGPURequestAdapterCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuInstanceRequestAdapter2(instance, options, callbackInfo) 
}
export function wgpuInstanceRequestAdapterF(instance:WGPUInstance, options:WGPURequestAdapterOptions | null, callbackInfo:WGPURequestAdapterCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuInstanceRequestAdapterF(instance, options, callbackInfo) 
}
export function wgpuInstanceWaitAny(instance:WGPUInstance, futureCount:number, futures:WGPUFutureWaitInfo | null, timeoutNS:bigint): WGPUWaitStatus {
 return webgpu.symbols.wgpuInstanceWaitAny(instance, futureCount, futures, timeoutNS) 
}
export function wgpuInstanceAddRef(instance:WGPUInstance): null {
 return webgpu.symbols.wgpuInstanceAddRef(instance) 
}
export function wgpuInstanceRelease(instance:WGPUInstance): null {
 return webgpu.symbols.wgpuInstanceRelease(instance) 
}
export function wgpuPipelineLayoutSetLabel(pipelineLayout:WGPUPipelineLayout, label:WGPUStringView): null {
 return webgpu.symbols.wgpuPipelineLayoutSetLabel(pipelineLayout, label) 
}
export function wgpuPipelineLayoutAddRef(pipelineLayout:WGPUPipelineLayout): null {
 return webgpu.symbols.wgpuPipelineLayoutAddRef(pipelineLayout) 
}
export function wgpuPipelineLayoutRelease(pipelineLayout:WGPUPipelineLayout): null {
 return webgpu.symbols.wgpuPipelineLayoutRelease(pipelineLayout) 
}
export function wgpuQuerySetDestroy(querySet:WGPUQuerySet): null {
 return webgpu.symbols.wgpuQuerySetDestroy(querySet) 
}
export function wgpuQuerySetGetCount(querySet:WGPUQuerySet): number {
 return webgpu.symbols.wgpuQuerySetGetCount(querySet) 
}
export function wgpuQuerySetGetType(querySet:WGPUQuerySet): WGPUQueryType {
 return webgpu.symbols.wgpuQuerySetGetType(querySet) 
}
export function wgpuQuerySetSetLabel(querySet:WGPUQuerySet, label:WGPUStringView): null {
 return webgpu.symbols.wgpuQuerySetSetLabel(querySet, label) 
}
export function wgpuQuerySetAddRef(querySet:WGPUQuerySet): null {
 return webgpu.symbols.wgpuQuerySetAddRef(querySet) 
}
export function wgpuQuerySetRelease(querySet:WGPUQuerySet): null {
 return webgpu.symbols.wgpuQuerySetRelease(querySet) 
}
export function wgpuQueueCopyExternalTextureForBrowser(queue:WGPUQueue, source:WGPUImageCopyExternalTexture | null, destination:WGPUImageCopyTexture | null, copySize:WGPUExtent3D | null, options:WGPUCopyTextureForBrowserOptions | null): null {
 return webgpu.symbols.wgpuQueueCopyExternalTextureForBrowser(queue, source, destination, copySize, options) 
}
export function wgpuQueueCopyTextureForBrowser(queue:WGPUQueue, source:WGPUImageCopyTexture | null, destination:WGPUImageCopyTexture | null, copySize:WGPUExtent3D | null, options:WGPUCopyTextureForBrowserOptions | null): null {
 return webgpu.symbols.wgpuQueueCopyTextureForBrowser(queue, source, destination, copySize, options) 
}
export function wgpuQueueOnSubmittedWorkDone(queue:WGPUQueue, callback:WGPUQueueWorkDoneCallback, userdata:null): null {
 return webgpu.symbols.wgpuQueueOnSubmittedWorkDone(queue, callback, userdata) 
}
export function wgpuQueueOnSubmittedWorkDone2(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuQueueOnSubmittedWorkDone2(queue, callbackInfo) 
}
export function wgpuQueueOnSubmittedWorkDoneF(queue:WGPUQueue, callbackInfo:WGPUQueueWorkDoneCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuQueueOnSubmittedWorkDoneF(queue, callbackInfo) 
}
export function wgpuQueueSetLabel(queue:WGPUQueue, label:WGPUStringView): null {
 return webgpu.symbols.wgpuQueueSetLabel(queue, label) 
}
export function wgpuQueueSubmit(queue:WGPUQueue, commandCount:number, commands:WGPUCommandBuffer | null): null {
 return webgpu.symbols.wgpuQueueSubmit(queue, commandCount, commands) 
}
export function wgpuQueueWriteBuffer(queue:WGPUQueue, buffer:WGPUBuffer, bufferOffset:bigint, data:null, size:number): null {
 return webgpu.symbols.wgpuQueueWriteBuffer(queue, buffer, bufferOffset, data, size) 
}
export function wgpuQueueWriteTexture(queue:WGPUQueue, destination:WGPUImageCopyTexture | null, data:null, dataSize:number, dataLayout:WGPUTextureDataLayout | null, writeSize:WGPUExtent3D | null): null {
 return webgpu.symbols.wgpuQueueWriteTexture(queue, destination, data, dataSize, dataLayout, writeSize) 
}
export function wgpuQueueAddRef(queue:WGPUQueue): null {
 return webgpu.symbols.wgpuQueueAddRef(queue) 
}
export function wgpuQueueRelease(queue:WGPUQueue): null {
 return webgpu.symbols.wgpuQueueRelease(queue) 
}
export function wgpuRenderBundleSetLabel(renderBundle:WGPURenderBundle, label:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderBundleSetLabel(renderBundle, label) 
}
export function wgpuRenderBundleAddRef(renderBundle:WGPURenderBundle): null {
 return webgpu.symbols.wgpuRenderBundleAddRef(renderBundle) 
}
export function wgpuRenderBundleRelease(renderBundle:WGPURenderBundle): null {
 return webgpu.symbols.wgpuRenderBundleRelease(renderBundle) 
}
export function wgpuRenderBundleEncoderDraw(renderBundleEncoder:WGPURenderBundleEncoder, vertexCount:number, instanceCount:number, firstVertex:number, firstInstance:number): null {
 return webgpu.symbols.wgpuRenderBundleEncoderDraw(renderBundleEncoder, vertexCount, instanceCount, firstVertex, firstInstance) 
}
export function wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder:WGPURenderBundleEncoder, indexCount:number, instanceCount:number, firstIndex:number, baseVertex:number, firstInstance:number): null {
 return webgpu.symbols.wgpuRenderBundleEncoderDrawIndexed(renderBundleEncoder, indexCount, instanceCount, firstIndex, baseVertex, firstInstance) 
}
export function wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder:WGPURenderBundleEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint): null {
 return webgpu.symbols.wgpuRenderBundleEncoderDrawIndexedIndirect(renderBundleEncoder, indirectBuffer, indirectOffset) 
}
export function wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder:WGPURenderBundleEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint): null {
 return webgpu.symbols.wgpuRenderBundleEncoderDrawIndirect(renderBundleEncoder, indirectBuffer, indirectOffset) 
}
export function wgpuRenderBundleEncoderFinish(renderBundleEncoder:WGPURenderBundleEncoder, descriptor:WGPURenderBundleDescriptor | null): WGPURenderBundle {
 return webgpu.symbols.wgpuRenderBundleEncoderFinish(renderBundleEncoder, descriptor) 
}
export function wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder:WGPURenderBundleEncoder, markerLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderBundleEncoderInsertDebugMarker(renderBundleEncoder, markerLabel) 
}
export function wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder): null {
 return webgpu.symbols.wgpuRenderBundleEncoderPopDebugGroup(renderBundleEncoder) 
}
export function wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderBundleEncoderPushDebugGroup(renderBundleEncoder, groupLabel) 
}
export function wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder:WGPURenderBundleEncoder, groupIndex:number, group:WGPUBindGroup | null, dynamicOffsetCount:number, dynamicOffsets:number | null): null {
 return webgpu.symbols.wgpuRenderBundleEncoderSetBindGroup(renderBundleEncoder, groupIndex, group, dynamicOffsetCount, dynamicOffsets) 
}
export function wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder:WGPURenderBundleEncoder, buffer:WGPUBuffer, format:WGPUIndexFormat, offset:bigint, size:bigint): null {
 return webgpu.symbols.wgpuRenderBundleEncoderSetIndexBuffer(renderBundleEncoder, buffer, format, offset, size) 
}
export function wgpuRenderBundleEncoderSetLabel(renderBundleEncoder:WGPURenderBundleEncoder, label:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderBundleEncoderSetLabel(renderBundleEncoder, label) 
}
export function wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder:WGPURenderBundleEncoder, pipeline:WGPURenderPipeline): null {
 return webgpu.symbols.wgpuRenderBundleEncoderSetPipeline(renderBundleEncoder, pipeline) 
}
export function wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder:WGPURenderBundleEncoder, slot:number, buffer:WGPUBuffer | null, offset:bigint, size:bigint): null {
 return webgpu.symbols.wgpuRenderBundleEncoderSetVertexBuffer(renderBundleEncoder, slot, buffer, offset, size) 
}
export function wgpuRenderBundleEncoderAddRef(renderBundleEncoder:WGPURenderBundleEncoder): null {
 return webgpu.symbols.wgpuRenderBundleEncoderAddRef(renderBundleEncoder) 
}
export function wgpuRenderBundleEncoderRelease(renderBundleEncoder:WGPURenderBundleEncoder): null {
 return webgpu.symbols.wgpuRenderBundleEncoderRelease(renderBundleEncoder) 
}
export function wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder:WGPURenderPassEncoder, queryIndex:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderBeginOcclusionQuery(renderPassEncoder, queryIndex) 
}
export function wgpuRenderPassEncoderDraw(renderPassEncoder:WGPURenderPassEncoder, vertexCount:number, instanceCount:number, firstVertex:number, firstInstance:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderDraw(renderPassEncoder, vertexCount, instanceCount, firstVertex, firstInstance) 
}
export function wgpuRenderPassEncoderDrawIndexed(renderPassEncoder:WGPURenderPassEncoder, indexCount:number, instanceCount:number, firstIndex:number, baseVertex:number, firstInstance:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderDrawIndexed(renderPassEncoder, indexCount, instanceCount, firstIndex, baseVertex, firstInstance) 
}
export function wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint): null {
 return webgpu.symbols.wgpuRenderPassEncoderDrawIndexedIndirect(renderPassEncoder, indirectBuffer, indirectOffset) 
}
export function wgpuRenderPassEncoderDrawIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint): null {
 return webgpu.symbols.wgpuRenderPassEncoderDrawIndirect(renderPassEncoder, indirectBuffer, indirectOffset) 
}
export function wgpuRenderPassEncoderEnd(renderPassEncoder:WGPURenderPassEncoder): null {
 return webgpu.symbols.wgpuRenderPassEncoderEnd(renderPassEncoder) 
}
export function wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder:WGPURenderPassEncoder): null {
 return webgpu.symbols.wgpuRenderPassEncoderEndOcclusionQuery(renderPassEncoder) 
}
export function wgpuRenderPassEncoderExecuteBundles(renderPassEncoder:WGPURenderPassEncoder, bundleCount:number, bundles:WGPURenderBundle | null): null {
 return webgpu.symbols.wgpuRenderPassEncoderExecuteBundles(renderPassEncoder, bundleCount, bundles) 
}
export function wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder:WGPURenderPassEncoder, markerLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderPassEncoderInsertDebugMarker(renderPassEncoder, markerLabel) 
}
export function wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint, maxDrawCount:number, drawCountBuffer:WGPUBuffer | null, drawCountBufferOffset:bigint): null {
 return webgpu.symbols.wgpuRenderPassEncoderMultiDrawIndexedIndirect(renderPassEncoder, indirectBuffer, indirectOffset, maxDrawCount, drawCountBuffer, drawCountBufferOffset) 
}
export function wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder:WGPURenderPassEncoder, indirectBuffer:WGPUBuffer, indirectOffset:bigint, maxDrawCount:number, drawCountBuffer:WGPUBuffer | null, drawCountBufferOffset:bigint): null {
 return webgpu.symbols.wgpuRenderPassEncoderMultiDrawIndirect(renderPassEncoder, indirectBuffer, indirectOffset, maxDrawCount, drawCountBuffer, drawCountBufferOffset) 
}
export function wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder:WGPURenderPassEncoder): null {
 return webgpu.symbols.wgpuRenderPassEncoderPixelLocalStorageBarrier(renderPassEncoder) 
}
export function wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder:WGPURenderPassEncoder): null {
 return webgpu.symbols.wgpuRenderPassEncoderPopDebugGroup(renderPassEncoder) 
}
export function wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder:WGPURenderPassEncoder, groupLabel:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderPassEncoderPushDebugGroup(renderPassEncoder, groupLabel) 
}
export function wgpuRenderPassEncoderSetBindGroup(renderPassEncoder:WGPURenderPassEncoder, groupIndex:number, group:WGPUBindGroup | null, dynamicOffsetCount:number, dynamicOffsets:number | null): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetBindGroup(renderPassEncoder, groupIndex, group, dynamicOffsetCount, dynamicOffsets) 
}
export function wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder:WGPURenderPassEncoder, color:WGPUColor | null): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetBlendConstant(renderPassEncoder, color) 
}
export function wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder:WGPURenderPassEncoder, buffer:WGPUBuffer, format:WGPUIndexFormat, offset:bigint, size:bigint): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetIndexBuffer(renderPassEncoder, buffer, format, offset, size) 
}
export function wgpuRenderPassEncoderSetLabel(renderPassEncoder:WGPURenderPassEncoder, label:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetLabel(renderPassEncoder, label) 
}
export function wgpuRenderPassEncoderSetPipeline(renderPassEncoder:WGPURenderPassEncoder, pipeline:WGPURenderPipeline): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetPipeline(renderPassEncoder, pipeline) 
}
export function wgpuRenderPassEncoderSetScissorRect(renderPassEncoder:WGPURenderPassEncoder, x:number, y:number, width:number, height:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetScissorRect(renderPassEncoder, x, y, width, height) 
}
export function wgpuRenderPassEncoderSetStencilReference(renderPassEncoder:WGPURenderPassEncoder, reference:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetStencilReference(renderPassEncoder, reference) 
}
export function wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder:WGPURenderPassEncoder, slot:number, buffer:WGPUBuffer | null, offset:bigint, size:bigint): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetVertexBuffer(renderPassEncoder, slot, buffer, offset, size) 
}
export function wgpuRenderPassEncoderSetViewport(renderPassEncoder:WGPURenderPassEncoder, x:number, y:number, width:number, height:number, minDepth:number, maxDepth:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderSetViewport(renderPassEncoder, x, y, width, height, minDepth, maxDepth) 
}
export function wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder:WGPURenderPassEncoder, querySet:WGPUQuerySet, queryIndex:number): null {
 return webgpu.symbols.wgpuRenderPassEncoderWriteTimestamp(renderPassEncoder, querySet, queryIndex) 
}
export function wgpuRenderPassEncoderAddRef(renderPassEncoder:WGPURenderPassEncoder): null {
 return webgpu.symbols.wgpuRenderPassEncoderAddRef(renderPassEncoder) 
}
export function wgpuRenderPassEncoderRelease(renderPassEncoder:WGPURenderPassEncoder): null {
 return webgpu.symbols.wgpuRenderPassEncoderRelease(renderPassEncoder) 
}
export function wgpuRenderPipelineGetBindGroupLayout(renderPipeline:WGPURenderPipeline, groupIndex:number): WGPUBindGroupLayout {
 return webgpu.symbols.wgpuRenderPipelineGetBindGroupLayout(renderPipeline, groupIndex) 
}
export function wgpuRenderPipelineSetLabel(renderPipeline:WGPURenderPipeline, label:WGPUStringView): null {
 return webgpu.symbols.wgpuRenderPipelineSetLabel(renderPipeline, label) 
}
export function wgpuRenderPipelineAddRef(renderPipeline:WGPURenderPipeline): null {
 return webgpu.symbols.wgpuRenderPipelineAddRef(renderPipeline) 
}
export function wgpuRenderPipelineRelease(renderPipeline:WGPURenderPipeline): null {
 return webgpu.symbols.wgpuRenderPipelineRelease(renderPipeline) 
}
export function wgpuSamplerSetLabel(sampler:WGPUSampler, label:WGPUStringView): null {
 return webgpu.symbols.wgpuSamplerSetLabel(sampler, label) 
}
export function wgpuSamplerAddRef(sampler:WGPUSampler): null {
 return webgpu.symbols.wgpuSamplerAddRef(sampler) 
}
export function wgpuSamplerRelease(sampler:WGPUSampler): null {
 return webgpu.symbols.wgpuSamplerRelease(sampler) 
}
export function wgpuShaderModuleGetCompilationInfo(shaderModule:WGPUShaderModule, callback:WGPUCompilationInfoCallback, userdata:null): null {
 return webgpu.symbols.wgpuShaderModuleGetCompilationInfo(shaderModule, callback, userdata) 
}
export function wgpuShaderModuleGetCompilationInfo2(shaderModule:WGPUShaderModule, callbackInfo:WGPUCompilationInfoCallbackInfo2): WGPUFuture {
 return webgpu.symbols.wgpuShaderModuleGetCompilationInfo2(shaderModule, callbackInfo) 
}
export function wgpuShaderModuleGetCompilationInfoF(shaderModule:WGPUShaderModule, callbackInfo:WGPUCompilationInfoCallbackInfo): WGPUFuture {
 return webgpu.symbols.wgpuShaderModuleGetCompilationInfoF(shaderModule, callbackInfo) 
}
export function wgpuShaderModuleSetLabel(shaderModule:WGPUShaderModule, label:WGPUStringView): null {
 return webgpu.symbols.wgpuShaderModuleSetLabel(shaderModule, label) 
}
export function wgpuShaderModuleAddRef(shaderModule:WGPUShaderModule): null {
 return webgpu.symbols.wgpuShaderModuleAddRef(shaderModule) 
}
export function wgpuShaderModuleRelease(shaderModule:WGPUShaderModule): null {
 return webgpu.symbols.wgpuShaderModuleRelease(shaderModule) 
}
export function wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:WGPUSharedBufferMemoryBeginAccessDescriptor | null): WGPUStatus {
 return webgpu.symbols.wgpuSharedBufferMemoryBeginAccess(sharedBufferMemory, buffer, descriptor) 
}
export function wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory:WGPUSharedBufferMemory, descriptor:WGPUBufferDescriptor | null): WGPUBuffer {
 return webgpu.symbols.wgpuSharedBufferMemoryCreateBuffer(sharedBufferMemory, descriptor) 
}
export function wgpuSharedBufferMemoryEndAccess(sharedBufferMemory:WGPUSharedBufferMemory, buffer:WGPUBuffer, descriptor:WGPUSharedBufferMemoryEndAccessState | null): WGPUStatus {
 return webgpu.symbols.wgpuSharedBufferMemoryEndAccess(sharedBufferMemory, buffer, descriptor) 
}
export function wgpuSharedBufferMemoryGetProperties(sharedBufferMemory:WGPUSharedBufferMemory, properties:WGPUSharedBufferMemoryProperties | null): WGPUStatus {
 return webgpu.symbols.wgpuSharedBufferMemoryGetProperties(sharedBufferMemory, properties) 
}
export function wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory:WGPUSharedBufferMemory): boolean {
 return webgpu.symbols.wgpuSharedBufferMemoryIsDeviceLost(sharedBufferMemory) 
}
export function wgpuSharedBufferMemorySetLabel(sharedBufferMemory:WGPUSharedBufferMemory, label:WGPUStringView): null {
 return webgpu.symbols.wgpuSharedBufferMemorySetLabel(sharedBufferMemory, label) 
}
export function wgpuSharedBufferMemoryAddRef(sharedBufferMemory:WGPUSharedBufferMemory): null {
 return webgpu.symbols.wgpuSharedBufferMemoryAddRef(sharedBufferMemory) 
}
export function wgpuSharedBufferMemoryRelease(sharedBufferMemory:WGPUSharedBufferMemory): null {
 return webgpu.symbols.wgpuSharedBufferMemoryRelease(sharedBufferMemory) 
}
export function wgpuSharedFenceExportInfo(sharedFence:WGPUSharedFence, info:WGPUSharedFenceExportInfo | null): null {
 return webgpu.symbols.wgpuSharedFenceExportInfo(sharedFence, info) 
}
export function wgpuSharedFenceAddRef(sharedFence:WGPUSharedFence): null {
 return webgpu.symbols.wgpuSharedFenceAddRef(sharedFence) 
}
export function wgpuSharedFenceRelease(sharedFence:WGPUSharedFence): null {
 return webgpu.symbols.wgpuSharedFenceRelease(sharedFence) 
}
export function wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:WGPUSharedTextureMemoryBeginAccessDescriptor | null): WGPUStatus {
 return webgpu.symbols.wgpuSharedTextureMemoryBeginAccess(sharedTextureMemory, texture, descriptor) 
}
export function wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory:WGPUSharedTextureMemory, descriptor:WGPUTextureDescriptor | null): WGPUTexture {
 return webgpu.symbols.wgpuSharedTextureMemoryCreateTexture(sharedTextureMemory, descriptor) 
}
export function wgpuSharedTextureMemoryEndAccess(sharedTextureMemory:WGPUSharedTextureMemory, texture:WGPUTexture, descriptor:WGPUSharedTextureMemoryEndAccessState | null): WGPUStatus {
 return webgpu.symbols.wgpuSharedTextureMemoryEndAccess(sharedTextureMemory, texture, descriptor) 
}
export function wgpuSharedTextureMemoryGetProperties(sharedTextureMemory:WGPUSharedTextureMemory, properties:WGPUSharedTextureMemoryProperties | null): WGPUStatus {
 return webgpu.symbols.wgpuSharedTextureMemoryGetProperties(sharedTextureMemory, properties) 
}
export function wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory:WGPUSharedTextureMemory): boolean {
 return webgpu.symbols.wgpuSharedTextureMemoryIsDeviceLost(sharedTextureMemory) 
}
export function wgpuSharedTextureMemorySetLabel(sharedTextureMemory:WGPUSharedTextureMemory, label:WGPUStringView): null {
 return webgpu.symbols.wgpuSharedTextureMemorySetLabel(sharedTextureMemory, label) 
}
export function wgpuSharedTextureMemoryAddRef(sharedTextureMemory:WGPUSharedTextureMemory): null {
 return webgpu.symbols.wgpuSharedTextureMemoryAddRef(sharedTextureMemory) 
}
export function wgpuSharedTextureMemoryRelease(sharedTextureMemory:WGPUSharedTextureMemory): null {
 return webgpu.symbols.wgpuSharedTextureMemoryRelease(sharedTextureMemory) 
}
export function wgpuSurfaceConfigure(surface:WGPUSurface, config:WGPUSurfaceConfiguration | null): null {
 return webgpu.symbols.wgpuSurfaceConfigure(surface, config) 
}
export function wgpuSurfaceGetCapabilities(surface:WGPUSurface, adapter:WGPUAdapter, capabilities:WGPUSurfaceCapabilities | null): WGPUStatus {
 return webgpu.symbols.wgpuSurfaceGetCapabilities(surface, adapter, capabilities) 
}
export function wgpuSurfaceGetCurrentTexture(surface:WGPUSurface, surfaceTexture:WGPUSurfaceTexture | null): null {
 return webgpu.symbols.wgpuSurfaceGetCurrentTexture(surface, surfaceTexture) 
}
export function wgpuSurfacePresent(surface:WGPUSurface): null {
 return webgpu.symbols.wgpuSurfacePresent(surface) 
}
export function wgpuSurfaceSetLabel(surface:WGPUSurface, label:WGPUStringView): null {
 return webgpu.symbols.wgpuSurfaceSetLabel(surface, label) 
}
export function wgpuSurfaceUnconfigure(surface:WGPUSurface): null {
 return webgpu.symbols.wgpuSurfaceUnconfigure(surface) 
}
export function wgpuSurfaceAddRef(surface:WGPUSurface): null {
 return webgpu.symbols.wgpuSurfaceAddRef(surface) 
}
export function wgpuSurfaceRelease(surface:WGPUSurface): null {
 return webgpu.symbols.wgpuSurfaceRelease(surface) 
}
export function wgpuTextureCreateErrorView(texture:WGPUTexture, descriptor:WGPUTextureViewDescriptor | null): WGPUTextureView {
 return webgpu.symbols.wgpuTextureCreateErrorView(texture, descriptor) 
}
export function wgpuTextureCreateView(texture:WGPUTexture, descriptor:WGPUTextureViewDescriptor | null): WGPUTextureView {
 return webgpu.symbols.wgpuTextureCreateView(texture, descriptor) 
}
export function wgpuTextureDestroy(texture:WGPUTexture): null {
 return webgpu.symbols.wgpuTextureDestroy(texture) 
}
export function wgpuTextureGetDepthOrArrayLayers(texture:WGPUTexture): number {
 return webgpu.symbols.wgpuTextureGetDepthOrArrayLayers(texture) 
}
export function wgpuTextureGetDimension(texture:WGPUTexture): WGPUTextureDimension {
 return webgpu.symbols.wgpuTextureGetDimension(texture) 
}
export function wgpuTextureGetFormat(texture:WGPUTexture): WGPUTextureFormat {
 return webgpu.symbols.wgpuTextureGetFormat(texture) 
}
export function wgpuTextureGetHeight(texture:WGPUTexture): number {
 return webgpu.symbols.wgpuTextureGetHeight(texture) 
}
export function wgpuTextureGetMipLevelCount(texture:WGPUTexture): number {
 return webgpu.symbols.wgpuTextureGetMipLevelCount(texture) 
}
export function wgpuTextureGetSampleCount(texture:WGPUTexture): number {
 return webgpu.symbols.wgpuTextureGetSampleCount(texture) 
}
export function wgpuTextureGetUsage(texture:WGPUTexture): WGPUTextureUsage {
 return webgpu.symbols.wgpuTextureGetUsage(texture) 
}
export function wgpuTextureGetWidth(texture:WGPUTexture): number {
 return webgpu.symbols.wgpuTextureGetWidth(texture) 
}
export function wgpuTextureSetLabel(texture:WGPUTexture, label:WGPUStringView): null {
 return webgpu.symbols.wgpuTextureSetLabel(texture, label) 
}
export function wgpuTextureAddRef(texture:WGPUTexture): null {
 return webgpu.symbols.wgpuTextureAddRef(texture) 
}
export function wgpuTextureRelease(texture:WGPUTexture): null {
 return webgpu.symbols.wgpuTextureRelease(texture) 
}
export function wgpuTextureViewSetLabel(textureView:WGPUTextureView, label:WGPUStringView): null {
 return webgpu.symbols.wgpuTextureViewSetLabel(textureView, label) 
}
export function wgpuTextureViewAddRef(textureView:WGPUTextureView): null {
 return webgpu.symbols.wgpuTextureViewAddRef(textureView) 
}
export function wgpuTextureViewRelease(textureView:WGPUTextureView): null {
 return webgpu.symbols.wgpuTextureViewRelease(textureView) 
}