import { expect } from 'expect/expect'
import { MNIST } from '../beautiful_mnist.ts'
import { mnist } from '../denograd/nn/datasets.ts'
import { Tensor } from '../denograd/tensor.ts'
import { py_bench } from '../test/helpers.ts'

Deno.bench({
  name: 'python baseline speed',
  fn: async (b) => void await py_bench(b, []),
})

// Loading mnist datasets
Deno.bench({
  name: 'mnist.dataset.ts',
  group: 'mnist.dataset',
  baseline: true,
  fn: async () => void await mnist(),
})
Deno.bench({
  name: 'mnist.dataset.py',
  group: 'mnist.dataset',
  fn: async (b) => void await py_bench(b, ['from tinygrad import nn', 'nn.datasets.mnist()']),
})

// Load weights from file
Deno.bench({
  name: 'mnist.load.ts',
  group: 'mnist.load',
  baseline: true,
  fn: async () => {
    const model = new MNIST()
    await model.load('./model.safetensors')
  },
})
Deno.bench({
  name: 'mnist.load.py',
  group: 'mnist.load',
  fn: async (b) => {
    await py_bench(b, [
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn',
      'model = Model()',
      "nn.state.load_state_dict(model, nn.state.safe_load('./model.safetensors'))",
    ])
  },
})

// Inference
Deno.bench({
  name: 'mnist.inference.ts',
  group: 'mnist.inference',
  baseline: true,
  fn: async () => {
    Tensor.manual_seed(3)
    const model = new MNIST()
    const input = Tensor.randint([28, 28], 0, 255).reshape([1, 1, 28, 28])
    const res = await model.call(input).tolist()
    expect(res.length).toBe(1)
    expect(res[0].length).toBe(10)
  },
})
Deno.bench({
  name: 'mnist.inference.py',
  group: 'mnist.inference',
  fn: async (b) => {
    await py_bench(b, [
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn, Tensor',
      'Tensor.manual_seed(3)',
      'model = Model()',

      'input = Tensor.randint(28, 28, low=0, high=255).reshape(1, 1, 28, 28)',
      'res = model(input).tolist()',
      'assert len(res) == 1',
      'assert len(res[0]) == 10',
    ])
  },
})

// Test
const batch = 1
Deno.bench({
  name: 'mnist.test.ts',
  group: 'mnist.test',
  baseline: true,
  fn: async () => {
    Tensor.manual_seed(3)
    const model = new MNIST()
    const [_, _1, x_test, y_test] = await mnist()
    await model.call(x_test.get({ stop: batch })).argmax(1).eq(y_test.get({ stop: batch })).mean().mul(100).tolist()
  },
})
Deno.bench({
  name: 'mnist.test.py',
  group: 'mnist.test',
  fn: async (b) =>
    await py_bench(b, [
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn, Tensor',
      'Tensor.manual_seed(3)',
      'model = Model()',
      '_,_,x_test,y_test = nn.datasets.mnist()',
      `model(x_test[:${batch}]).argmax(1).eq(y_test[:${batch}]).mean().mul(100).tolist()`,
    ]),
})

// Train
Deno.bench({
  name: 'mnist.train.ts',
  group: 'mnist.train',
  baseline: true,
  fn: async () => {
    Tensor.manual_seed(3)
    const model = new MNIST()
    const [x_train, y_train] = await mnist()
    const samples = Tensor.randint([batch], undefined, x_train.shape[0])
    await model.call(x_train.get(samples)).sparse_categorical_crossentropy(y_train.get(samples)).tolist()
  },
})
Deno.bench({
  name: 'mnist.train.py',
  group: 'mnist.train',
  fn: async (b) =>
    await py_bench(b, [
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn, Tensor',
      'Tensor.manual_seed(3)',
      'model = Model()',
      'x_train,y_train,_,_ = nn.datasets.mnist()',
      `samples = Tensor.randint(${batch}, high=x_train.shape[0])`,
      `model(x_train[samples]).sparse_categorical_crossentropy(y_train[samples]).tolist()`,
    ]),
})
