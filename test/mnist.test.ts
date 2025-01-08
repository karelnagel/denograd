import { mnist } from '../src/mod.ts'
import { Tensor } from '../src/tensor.ts'
import { compare } from './helpers.ts'

Deno.test(
  'mnist.get.number',
  compare(
    [[]],
    async () => {
      const [X_train] = await mnist()
      return X_train.get(1)
    },
    [
      'from tinygrad.nn.datasets import mnist',
      'x_train, _, _, _ = mnist()',
      'out(x_train[1])',
    ],
  ),
)
// TODO: loads forever
Deno.test.ignore(
  'mnist.get.tensor',
  compare(
    [[]],
    async () => {
      const [X_train] = await mnist()
      const sample = new Tensor([1])
      return X_train.get(sample)
    },
    [
      'from tinygrad.nn.datasets import mnist',
      'x_train, _, _, _ = mnist()',
      'sample = tiny.Tensor([1])',
      'out(x_train[sample])',
    ],
  ),
)
