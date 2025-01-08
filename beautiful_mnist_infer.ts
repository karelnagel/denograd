// model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import { Layer, Tensor } from './src/tensor.ts'
import * as nn from './src/nn/index.ts'
import { mnist } from './src/nn/datasets.ts'
import { get_env } from './src/helpers.ts'

export class MNIST {
  layers: Layer[] = [
    new nn.Conv2d(1, 32, 5),
    Tensor.relu,
    new nn.Conv2d(32, 32, 5),
    Tensor.relu,
    new nn.BatchNorm(32),
    Tensor.max_pool2d,
    new nn.Conv2d(32, 64, 3),
    Tensor.relu,
    new nn.Conv2d(64, 64, 3),
    Tensor.relu,
    new nn.BatchNorm(64),
    Tensor.max_pool2d,
    (x) => x.flatten(1),
    new nn.Linear(576, 10),
  ]
  call = (x: Tensor): Tensor => x.sequential(this.layers)
}

if (import.meta.main) {
  const [X_train, Y_train, X_test, Y_test] = await mnist(undefined, !!get_env('FASHION'))

  const model = new MNIST()
  const samples = Tensor.randint([1], undefined, X_train.shape[0] as number)
  const res = await model.call(X_train.get(samples)).tolist()
  console.log(res)
}
