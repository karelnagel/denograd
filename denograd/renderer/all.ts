import { AMDRenderer, ClangRenderer, HIPRenderer, IntelRenderer, MetalRenderer, OpenCLRenderer, QCOMRenderer } from './cstyle.ts'
import type { Renderer } from './index.ts'
import { WATRenderer } from './wat.ts'
import { WGSLRenderer } from './wgsl.ts'

// needed for cloud
export const RENDERERS: Record<string, typeof Renderer> = {
  ClangRenderer,
  WGSLRenderer,
  WATRenderer,
  MetalRenderer,
  AMDRenderer,
  IntelRenderer,
  //   CUDARenderer,
  HIPRenderer,
  //   NVRenderer,
  OpenCLRenderer,
  QCOMRenderer,
}
