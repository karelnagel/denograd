// model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392

import { get_number_env, range } from '../denograd/helpers.ts'
import { mnist } from '../denograd/nn/datasets.ts'
import { Adam } from '../denograd/nn/optim.ts'
import { get_parameters } from '../denograd/nn/state.ts'
import { Tensor } from '../denograd/tensor.ts'
import { Tqdm } from '../denograd/tqdm.ts'
import { MNIST } from './mod.ts'

if (import.meta.main) {
  const [X_train, Y_train, X_test, Y_test] = await mnist(undefined)
  const BS = get_number_env('BS', 512)

  const model = new MNIST()

  // await model.load('./mnist.safetensors').catch((x) => x)
  const opt = Adam(get_parameters(model))

  const train_step = async (): Promise<Tensor> => {
    Tensor.training = true
    opt.zero_grad()
    const samples = Tensor.randint([BS], undefined, X_train.shape[0])
    const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
    await opt.step()
    Tensor.training = false
    return loss
  }

  const get_test_acc = (): Tensor => model.call(X_test).argmax(1).eq(Y_test).mean().mul(100)

  let test_acc = NaN
  const t = new Tqdm(range(get_number_env('STEPS', 70)))
  for await (const i of t) {
    const loss = await (await train_step()).item()
    if (i % 10 === 9) test_acc = await get_test_acc().item()
    t.set_description(`loss: ${loss.toFixed(2)}, test_accuracy: ${test_acc.toFixed(2)}`)
  }
  await model.save('./mnist.safetensors')
}
