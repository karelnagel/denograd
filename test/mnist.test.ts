import { expect } from 'expect/expect'
import { MNIST } from '../beautiful_mnist.ts'
import { load_state_dict, mnist, safe_load } from '../src/mod.ts'
import { Tensor } from '../src/tensor.ts'
import { asdict, compare, python } from './helpers.ts'

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
      // TODO: randint doesn't return the same value
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

Deno.test(
  'mnist.load',
  async () => {
    Tensor.manual_seed(333)
    const model = new MNIST()
    await model.load('./model.safetensors')
    const ts = [(model.layers[0] as any).weight, (model.layers.at(-1)! as any).weight]
    const py = await python([
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn',
      'tiny.Tensor.manual_seed(333)',
      'model = Model()',
      "nn.state.load_state_dict(model, nn.state.safe_load('./model.safetensors'))",
      'out([model.layers[0].weight, model.layers[-1].weight])',
    ])
    expect(await asdict(ts)).toEqual(await asdict(py))
  },
)
