import { MNIST } from '../beautiful_mnist.ts'
import { mnist } from '../src/mod.ts'
import { Tensor } from '../src/tensor.ts'
import { compare } from './helpers.ts'

Deno.test(
  'mnist.get',
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

Deno.test(
  'mnist.call',
  compare(
    [[]],
    async () => {
      Tensor.manual_seed(333)
      const [x_train] = await mnist()
      const model = new MNIST()
      // TODO:randint doesn't return the same value
      // const samples = Tensor.randint([1], undefined, x_train.shape[0] as number)
      const samples = new Tensor([1])
      return model.call(x_train.get(samples))
    },
    [
      'from tinygrad.nn.datasets import mnist',
      'from examples.beautiful_mnist import Model',
      'tiny.Tensor.manual_seed(333)',
      'x_train, _, _, _ = mnist()',
      // 'samples = tiny.Tensor.randint(1, high=x_train.shape[0])',
      'samples = tiny.Tensor([1])',
      'model = Model()',
      'out(model(x_train[samples]))',
    ],
  ),
)
