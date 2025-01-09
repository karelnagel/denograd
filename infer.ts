import { mnist } from './src/nn/datasets.ts'
import { get_env } from './src/helpers.ts'
import { MNIST } from './beautiful_mnist.ts'

if (import.meta.main) {
  const [X_train, Y_train, X_test, Y_test] = await mnist(undefined, !!get_env('FASHION'))
  const model = new MNIST()
  const res = (model.call(X_test).argmax(1).eq(Y_test)).mean().mul(100)

  const s = performance.now()
  const list = await res.tolist()
  console.log(`Took ${performance.now() - s}, ${list}`)
}
