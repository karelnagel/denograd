import type { Tensor } from '../tensor.ts'
export * as optim from './optim.ts'
export * as state from './state.ts'
export * as datasets from './datasets.ts'

export const Conv2d = (x: number, y: number, z: number) => (x: Tensor) => x
export const BatchNorm = (x: number) => (x: Tensor) => x
export const Linear = (x: number, y: number) => (x: Tensor) => x
