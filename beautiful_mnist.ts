// import process from 'node:process'
// import * as nn from './src/nn/index.ts'
// import { type Layer, Tensor } from './src/tensor.ts'
// import { GlobalCounters } from './src/helpers.ts'

// export class Model {
//   layers: Layer[]
//   constructor() {
//     // deno-fmt-ignore
//     this.layers = [
//       nn.Conv2d(1, 32, 5), Tensor.relu,
//       nn.Conv2d(32, 32, 5), Tensor.relu,
//       nn.BatchNorm(32), Tensor.max_pool2d,
//       nn.Conv2d(32, 64, 3), Tensor.relu,
//       nn.Conv2d(64, 64, 3), Tensor.relu,
//       nn.BatchNorm(64), Tensor.max_pool2d,
//       (x) => x.flatten(), nn.Linear(576, 10),
//     ]
//   }
//   call = (x: Tensor) => x.sequential(this.layers)
// }

// const [XTrain, YTrain, XTest, YTest] = nn.datasets.mnist(null, !!process.env.FASHION)

// const model = new Model()
// const opt = new nn.optim.Adam(nn.state.getParameters(model))

// // @TinyJit
// // @Tensor.train()
// const trainStep = () => {
//   opt.zeroGrad()
//   const samples = Tensor.randint({ shape: [Number(process.env.BS || 512)], high: XTrain.shape[0] })
//   const loss = model.call(XTrain.slice(samples)).sparseCategoricalCrossentropy(YTrain.slice(samples)).backward()
//   opt.step()
//   return loss
// }

// // @TinyJit
// // @Tensor.test()
// const getTestAcc = () => model.call(XTest).argmax({ axis: 1 }).equals(YTest).mean().mul(100)

// let testAcc = NaN
// for (let i = 0; i < Number(process.env.STEPS || 70); i++) {
//   GlobalCounters.reset() // NOTE: this makes it nice for DEBUG=2 timing
//   const loss = trainStep()
//   if (i % 10 === 9) testAcc = getTestAcc().item()
//   console.log(`loss: ${loss.item().toFixed(2)} test_accuracy: ${testAcc.toFixed(2)}%`)
// }

// // verify eval acc
// const target = Number(process.env.TARGET_EVAL_ACC_PCT || 0.0)
// if (testAcc >= target && testAcc !== 100.0) console.log(`testAcc=${testAcc} >= ${target}`)
// else throw new Error(`test_acc=${testAcc} < ${target}`)
