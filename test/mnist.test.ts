import { expect } from 'expect/expect'
import { MNIST } from '../models/mnist.ts'
import { Adam, get_parameters, mnist } from '../denograd/mod.ts'
import { Tensor } from '../denograd/tensor.ts'
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
    await model.load('./mnist.safetensors')
    const ts = [(model.layers[0] as any).weight, (model.layers.at(-1)! as any).weight]
    const py = await python([
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn',
      'tiny.Tensor.manual_seed(333)',
      'model = Model()',
      "nn.state.load_state_dict(model, nn.state.safe_load('./mnist.safetensors'))",
      'out([model.layers[0].weight, model.layers[-1].weight])',
    ])
    expect(await asdict(ts)).toEqual(await asdict(py))
  },
)

Deno.test.ignore(
  'mnist.train',
  compare(
    [[]],
    async () => {
      Tensor.manual_seed(333)
      const [x_train, y_train] = await mnist()
      const model = new MNIST()
      const opt = Adam(get_parameters(model))

      Tensor.training = true
      opt.zero_grad()
      const samples = new Tensor([1])
      const loss = model.call(x_train.get(samples)).sparse_categorical_crossentropy(y_train.get(samples)).backward()
      await opt.step()
      Tensor.training = false

      return loss
    },
    [
      'from tinygrad import nn, Tensor',
      'from examples.beautiful_mnist import Model',

      'Tensor.manual_seed(333)',
      'x_train, y_train, _, _ = nn.datasets.mnist()',
      'model = Model()',
      'opt = nn.optim.Adam(nn.state.get_parameters(model))',

      'Tensor.training = True',
      'opt.zero_grad()',
      'samples = tiny.Tensor([1])',
      'loss = model(x_train[samples]).sparse_categorical_crossentropy(y_train[samples]).backward()',
      'opt.step()',
      'Tensor.training=False',

      'out(loss)',
    ],
  ),
)
