import { Adam, get_parameters, MNIST, mnist, Tensor, TinyJit } from '@jsgrad/jsgrad'

const [X_train, Y_train, X_test, Y_test] = await mnist(undefined, '/')
const model = new MNIST()
const opt = Adam(get_parameters(model))

const train_step = new TinyJit(async () => {
  Tensor.training = true
  opt.zero_grad()
  const samples = Tensor.randint([512], undefined, X_train.shape_num[0])
  const loss = model.call(X_train.get(samples))
    .sparse_categorical_crossentropy(Y_train.get(samples)).backward()
  await opt.step()
  Tensor.training = false
  return await loss.item()
})
const get_test_acc = new TinyJit(async () => {
  return await model.call(X_test).argmax(1).eq(Y_test)
    .mean().mul(100).item()
})

let test_acc = NaN
for (let i = 0; i < 70; i++) {
  const loss = await train_step.call()
  if (i % 10 === 9) test_acc = await get_test_acc.call()
  document.querySelector('#out').textContent = `step: ${i} loss: ${loss.toFixed(2)}, test_accuracy: ${test_acc.toFixed(2)}`
}
alert(`Finished training with acc ${test_acc.toFixed(2)}`)
