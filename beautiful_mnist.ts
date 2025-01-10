// model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import { Layer, Tensor } from './src/tensor.ts'
import { mnist } from './src/nn/datasets.ts'
import { get_env, get_number_env, range } from './src/helpers.ts'
import { Tqdm } from './src/tqdm.ts'
import { BatchNorm, Conv2d, Linear, Model } from './src/nn/index.ts'
import { get_parameters } from './src/nn/state.ts'
import { Adam } from './src/nn/optim.ts'

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

if (import.meta.main) {
  const [X_train, Y_train, X_test, Y_test] = await mnist(undefined, !!get_env('FASHION'))

  const model = new MNIST()

  await model.load('./mnist.safetensors').catch((x) => x)
  const opt = Adam(get_parameters(model))

  const train_step = async (): Promise<Tensor> => {
    opt.zero_grad()
    const samples = Tensor.randint([get_number_env('BS', 512)], undefined, X_train.shape[0])
    const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
    await opt.step()
    return loss
  }

  const get_test_acc = (): Tensor => model.call(X_test).argmax(1).eq(Y_test).mean().mul(100)

  Tensor.training = true
  let test_acc = NaN
  const t = new Tqdm(range(get_number_env('STEPS', 10)))
  for await (const i of t) {
    const loss = await train_step()
    if (i % 10 === 9) test_acc = await get_test_acc().item()
    t.set_description(`loss: ${(await loss.item()).toFixed(2)}, test_accuracy: ${test_acc}`)
  }
  Tensor.training = false
  await model.save('./mnist.safetensors')
}
