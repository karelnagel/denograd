import { nn, Tensor } from '../denograd/mod.ts'

Object.assign(globalThis, { Tensor, nn })

export * from '../denograd/mod.ts'
