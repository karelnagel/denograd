// model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import { Tensor } from './src/tensor.ts'
import * as nn from './src/nn/index.ts'
import { mnist } from './src/nn/datasets.ts'
import { get_env, get_number_env, GlobalCounters, range } from './src/helpers.ts'
import { tqdm } from './src/tqdm.ts'

class Model {
  layers: ((x: Tensor) => Tensor)[]
  constructor() {
    this.layers = [
      new nn.Conv2d(1, 32, 5).call,
      (x) => x.relu(),
      new nn.Conv2d(32, 32, 5).call,
      (x) => x.relu(),
      new nn.BatchNorm(32).call,
      (x) => x.max_pool2d(),
      new nn.Conv2d(32, 64, 3).call,
      (x) => x.relu(),
      new nn.Conv2d(64, 64, 3).call,
      (x) => x.relu(),
      new nn.BatchNorm(64).call,
      (x) => x.max_pool2d(),
      (x) => x.flatten(1),
      new nn.Linear(576, 10).call,
    ]
  }
  call = (x: Tensor): Tensor => x.sequential(this.layers)
}

const [X_train, Y_train, X_test, Y_test] = await mnist(undefined, !!get_env('FASHION'))

const model = new Model()
const opt = nn.optim.Adam(nn.state.get_parameters(model))

const train_step = (): Tensor => {
  opt.zero_grad()
  const samples = Tensor.randint([get_number_env('BS', 512)], undefined, X_train.shape[0] as number)
  // TODO: this "gather" of samples === very slow. will be under 5s when this === fixed
  const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
  opt.step()
  return loss
}

const get_test_acc = (): Tensor => (model.call(X_test).argmax(1).eq(Y_test)).mean().mul(100)

let test_acc = NaN
const t = tqdm(range(get_number_env('STEPS', 14)))
Tensor.training = true
for await (const i of t) {
  GlobalCounters.reset() // NOTE: this makes it nice for DEBUG=2 timing
  const loss = train_step()
  if (i % 10 === 9) test_acc = get_test_acc().item() as number
  console.log(`loss: ${loss.item()} test_accuracy: ${test_acc}%`)
}
Tensor.training = false
