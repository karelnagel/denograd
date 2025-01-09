import { MNIST } from '../beautiful_mnist.ts'
import { mnist } from '../src/nn/datasets.ts'
import { py_bench } from '../test/helpers.ts'

Deno.bench({
  name: 'python baseline speed',
  fn: async (b) => void await py_bench([], b),
})

// Loading mnist datasets
Deno.bench({
  name: 'mnist.dataset.py',
  group: 'mnist.dataset',
  baseline: true,
  fn: async (b) => void await py_bench(['from tinygrad import nn', 'nn.datasets.mnist()'], b),
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
    await py_bench([
      'from examples.beautiful_mnist import Model',
      'from tinygrad import nn',
      'model = Model()',
      "nn.state.load_state_dict(model, nn.state.safe_load('./model.safetensors'))",
    ], b)
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
