import { AMDRenderer, ClangRenderer, CUDARenderer, HIPRenderer, IntelRenderer, MetalRenderer, NVRenderer, OpenCLRenderer, QCOMRenderer } from './cstyle.ts'
import { WATRenderer } from './wat.ts'
import { WGSLRenderer } from './wgsl.ts'

// needed for cloud
export const RENDERERS = {
  ClangRenderer,
  WGSLRenderer,
  WATRenderer,
  MetalRenderer,
  AMDRenderer,
  IntelRenderer,
  CUDARenderer,
  HIPRenderer,
  NVRenderer,
  OpenCLRenderer,
  QCOMRenderer,
}
