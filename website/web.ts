import * as Denograd from '../denograd/mod.ts'

Object.assign(globalThis, { Tensor: Denograd.Tensor, nn: Denograd.nn })

export * from '../denograd/mod.ts'
