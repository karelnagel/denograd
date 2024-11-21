export type Layer = (t: Tensor) => Tensor

export class Tensor {
  static relu = (x: Tensor) => x
  static max_pool2d = (x: Tensor) => x
  static randint = (s: { shape: number[]; low?: number; high?: number }) => new Tensor()
  get shape(): number[] {
    return []
  }
  flatten = () => this
  slice = (t: Tensor) => this
  sparseCategoricalCrossentropy = (t: Tensor) => this
  backward = () => this
  argmax = (p: { axis: number }) => this
  equals = (t: Tensor) => this
  mean = () => this
  mul = (t: number | Tensor) => this
  item = () => 0
  sequential = (t: Layer[]) => this
}
