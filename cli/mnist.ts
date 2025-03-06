import { Adam, env, get_parameters, MNIST, mnist, range, Tensor, TinyJit, Tqdm } from '../denograd/mod.ts'

const [X_train, Y_train, X_test, Y_test] = await mnist(undefined)
const BS = env.get_num('BS', 512)

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

const get_test_acc = new TinyJit(() => model.call(X_test).argmax(1).eq(Y_test).mean().mul(100))

let test_acc = NaN
const t = new Tqdm(range(env.get_num('STEPS', 70)))
for (const i of t) {
  const loss = await train_step.call()
  if (i % 10 === 9) test_acc = await get_test_acc.call().then((x) => x.item())
  t.set_description(`loss: ${await loss.item().then((x) => x.toFixed(2))}, test_accuracy: ${test_acc.toFixed(2)}`)
}
// await model.save('./mnist.safetensors')
