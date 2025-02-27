// model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import { get_number_env, range } from '../helpers.ts'
import { type Layer, Tensor } from '../tensor.ts'
import { Tqdm } from '../tqdm.ts'
import { mnist } from '../nn/datasets.ts'
import { BatchNorm, Conv2d, Linear, Model } from '../nn/index.ts'
import { Adam } from '../nn/optim.ts'
import { get_parameters } from '../nn/state.ts'
import { TinyJit } from '../engine/jit.ts'

export class MNIST extends Model {
  override DEFAULT_LOAD = 'https://denograd.com/mnist.safetensors'
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

const main = async () => {
  const [X_train, Y_train, X_test, Y_test] = await mnist(undefined)
  const BS = get_number_env('BS', 512)

  const model = new MNIST()

  // await model.load('./mnist.safetensors').catch((x) => x)
  const opt = Adam(get_parameters(model))

  const train_step = new TinyJit(async () => {
    Tensor.training = true
    opt.zero_grad()
    const samples = Tensor.randint([BS], undefined, X_train.shape[0])
    const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
    await opt.step()
    Tensor.training = false
    return loss
  })

  const get_test_acc = new TinyJit(async () => model.call(X_test).argmax(1).eq(Y_test).mean().mul(100))

  let test_acc = NaN
  const t = new Tqdm(range(get_number_env('STEPS', 70)))
  for await (const i of t) {
    const loss = await train_step.call()
    if (i % 10 === 9) test_acc = await get_test_acc.call().then((x) => x.item())
    t.set_description(`loss: ${await loss.item().then((x) => x.toFixed(2))}, test_accuracy: ${test_acc.toFixed(2)}`)
  }
  // await model.save('./mnist.safetensors')
}

if (import.meta.main) await main()
