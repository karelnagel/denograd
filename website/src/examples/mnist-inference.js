import { BatchNorm, Conv2d, Linear, Model, Tensor } from '@jsgrad/jsgrad'

class MNIST extends Model {
  DEFAULT_LOAD = 'https://jsgrad.org/mnist.safetensors'
  layers = [
    new Conv2d(1, 32, 5),
    Tensor.relu,
    new Conv2d(32, 32, 5),
    Tensor.relu,
    new BatchNorm(32),
    Tensor.max_pool2d,
    new Conv2d(32, 64, 3),
    Tensor.relu,
    new Conv2d(64, 64, 3),
    Tensor.relu,
    new BatchNorm(64),
    Tensor.max_pool2d,
    (x) => x.flatten(1),
    new Linear(576, 10),
  ]
}

const model = await new MNIST().load()
const input = Tensor.rand([1, 1, 28, 28])
console.log(await model.call(input).tolist())
