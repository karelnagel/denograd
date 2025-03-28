import { BatchNorm, Conv2d, Linear, Model, Tensor } from '@jsgrad/jsgrad'

class MNIST extends Model {
  layers = [
    new Conv2d(1, 32, 5), Tensor.relu,
    new Conv2d(32, 32, 5), Tensor.relu,
    new BatchNorm(32), Tensor.max_pool2d,
    new Conv2d(32, 64, 3), Tensor.relu,
    new Conv2d(64, 64, 3), Tensor.relu,
    new BatchNorm(64), Tensor.max_pool2d,
    (x) => x.flatten(1), new Linear(576, 10),
  ]
}
const onProgress = (p) => document.querySelector("#out").textContent = `${p.label} - ${p.i}/${p.size}`
const model = await new MNIST().load("https://jsgrad.org/mnist.safetensors", onProgress)
const input = Tensor.randint([1, 1, 28, 28], 0, 255)

alert(await model.call(input).argmax(1).tolist())
