import { Adam, get_parameters, GlobalCounters, MNIST, mnist, Tensor, TinyJit, Tqdm } from '../jsgrad/node.ts'
import { parseArgs, z } from './parse.ts'

const args = parseArgs({
  steps: z.number().default(70).describe('Steps'),
  bs: z.number().default(512).describe('Batch size'),
})
const [X_train, Y_train, X_test, Y_test] = await mnist(undefined)

const model = new MNIST()

// await model.load('./mnist.safetensors').catch((x) => x)
const opt = Adam(get_parameters(model))

const train_step = new TinyJit(async () => {
  Tensor.training = true
  opt.zero_grad()
  const samples = Tensor.randint([args.bs], undefined, X_train.shape_num[0])
  const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(
    Y_train.get(samples),
  ).backward()
  await opt.step()
  Tensor.training = false
  return loss
})

const get_test_acc = new TinyJit(() => model.call(X_test).argmax(1).eq(Y_test).mean().mul(100))

let test_acc = NaN
const t = new Tqdm<number>(args.steps)
for (const i of t) {
  GlobalCounters.reset()
  const loss = await train_step.call()
  if (i % 10 === 9) test_acc = await get_test_acc.call().then((x) => x.item())
  t.set_description(
    `loss: ${await loss.item().then((x) => x.toFixed(2))}, test_accuracy: ${test_acc.toFixed(2)}`,
  )
}
// await model.save('./mnist.safetensors')
