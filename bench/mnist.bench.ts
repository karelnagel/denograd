import { expect } from 'expect/expect'
import { MNIST } from '../beautiful_mnist.ts'
import { mnist } from '../src/nn/datasets.ts'
import { Tensor } from '../src/tensor.ts'
import { py_bench } from '../test/helpers.ts'

Deno.bench({
  name: 'python baseline speed',
  fn: async (b) => void await py_bench(b, []),
})

// Loading mnist datasets
Deno.bench({
  name: 'mnist.dataset.py',
  group: 'mnist.dataset',
  baseline: true,
  fn: async (b) => void await py_bench(b, ['from tinygrad import nn', 'nn.datasets.mnist()']),
})
Deno.bench({
  name: 'mnist.dataset.ts',
  group: 'mnist.dataset',
  fn: async () => void await mnist(),
})

// Load weights from file
Deno.bench({
  name: 'mnist.load.py',
  group: 'mnist.load',
  baseline: true,
  fn: async (b) => {
    await py_bench(b, [
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn',
      'model = Model()',
      "nn.state.load_state_dict(model, nn.state.safe_load('./model.safetensors'))",
    ])
  },
})
Deno.bench({
  name: 'mnist.load.ts',
  group: 'mnist.load',
  fn: async () => {
    const model = new MNIST()
    await model.load('./model.safetensors')
  },
})

// Inference
Deno.bench({
  name: 'mnist.inference.py',
  group: 'mnist.inference',
  baseline: true,
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
Deno.bench({
  name: 'mnist.inference.ts',
  group: 'mnist.inference',
  fn: async () => {
    Tensor.manual_seed(3)
    const model = new MNIST()
    const input = Tensor.randint([28, 28], 0, 255).reshape([1, 1, 28, 28])
    const res = await model.call(input).tolist()
    expect(res.length).toBe(1)
    expect(res[0].length).toBe(10)
  },
})

// Test
const batch = 512
Deno.bench({
  name: 'mnist.test.py',
  group: 'mnist.test',
  baseline: true,
  fn: async (b) =>
    await py_bench(b, [
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn, Tensor',
      'Tensor.manual_seed(3)',
      'model = Model()',
      '_,_,x_test,y_test = nn.datasets.mnist()',
      `res = model(x_test[:${batch}]).argmax(1).eq(y_test[:${batch}]).mean().mul(100).tolist()`,
      'assert res>13 and res<15',
    ]),
})
Deno.bench({
  name: 'mnist.test.ts',
  group: 'mnist.test',
  fn: async () => {
    Tensor.manual_seed(3)
    const model = new MNIST()
    const [_, _1, x_test, y_test] = await mnist()
    const res = await model.call(x_test.get({ stop: batch })).argmax(1).eq(y_test.get({ stop: batch })).mean().mul(100).tolist()
    expect(res > 13 && res < 15).toBeTruthy()
  },
})

// Train
Deno.bench({
  name: 'mnist.train.py',
  group: 'mnist.train',
  baseline: true,
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
Deno.bench({
  name: 'mnist.train.ts',
  group: 'mnist.train',
  fn: async () => {
    Tensor.manual_seed(3)
    const model = new MNIST()
    const [x_train, y_train] = await mnist()
    const samples = Tensor.randint([batch], undefined, x_train.shape[0])
    await model.call(x_train.get(samples)).sparse_categorical_crossentropy(y_train.get(samples)).tolist()
  },
})
