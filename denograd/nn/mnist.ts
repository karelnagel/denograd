// model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import { type Layer, Tensor } from '../tensor.ts'
import { BatchNorm, Conv2d, Linear, Model } from './index.ts'

export class MNIST extends Model {
  layers: Layer[] = [
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
