# Denograd - best ML library in Typescript

Tinygrad rewritten in typescript

# Why?
1. I wanted to learn more about ML and tinygrad
2. I don't like python that much
3. There are 17.5M JS/TS devs (source ChatGPT) and no good ML lib
4. Creating python bindings would have been too easy and I would have worked only on the easy parts, and then JS/TS would still be kind of second class citizen + running this inside browser would be kind of hard(maybe) or just worse than native library. Downsides: a lot of work initially + keeping up with tinygrad updates is much harder, but doable.

# Goal - The easiest and fastest way to run and train models in JS/TS.

It should always use the fastest available runtime on your machine, so it would use WebGPU in browser and your GPU or CPU when using node/deno/bun. 

With the upcoming tinygrad CLOUD, you should be able to just run one docker image on any hardware, point your program to that server and run your model there. There shouldn't be any setup required for each model, just a general CLOUD program that will work with every model and cache your model for fast inference. This way there is no need to pay for someone for hosting your model, you would just find the cheapest/best GPU server you can and you can run all your models there.

Most popular models will be available as a package: 
```ts
import { Llama } from "@denograd/models"

// run with the fastest available runtime
const llama = new Llama({ model: "3.1-3B" })
const res = llama.run({ prompt: "Hello how are you?" })

// run on tinygrad CLOUD or self hosted cloud
const llama = new Llama({ device: "CLOUD", host: process.env.CLOUD_HOST })
```

Create, run and train your own models:
```ts
import { Tensor, nn } from '@denograd/core'
import { range, tqdm } from '@denograd/helpers'
import { mnist } from '@denograd/datasets'

export class MNIST {
  layers: ((x: Tensor) => Tensor)[]
  constructor() {
    this.layers = [
      new nn.Conv2d(1, 32, 5).call, (x) => x.relu(),
      new nn.Conv2d(32, 32, 5).call, (x) => x.relu(),
      new nn.BatchNorm(32).call, (x) => x.max_pool2d(),
      new nn.Conv2d(32, 64, 3).call, (x) => x.relu(),
      new nn.Conv2d(64, 64, 3).call, (x) => x.relu(),
      new nn.BatchNorm(64).call, (x) => x.max_pool2d(),
      (x) => x.flatten(1), new nn.Linear(576, 10).call,
    ]
  }
  call = (x: Tensor): Tensor => x.sequential(this.layers)
}

const [X_train, Y_train, X_test, Y_test] = await mnist()

const model = new MNIST()
const opt = nn.optim.Adam(nn.state.get_parameters(model))

const train_step = (): Tensor => {
  opt.zero_grad()
  const samples = Tensor.randint([512], undefined, X_train.shape[0])
  const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
  opt.step()
  return loss
}

const get_test_acc = (): Tensor => (model.call(X_test).argmax(1).eq(Y_test)).mean().mul(100)
let test_acc = NaN

Tensor.training = true
for await (const i of tqdm(range(100))) {
  const loss = train_step()
  if (i % 10 === 9) test_acc = get_test_acc().item()
  console.log(`loss: ${loss.item()} test_accuracy: ${test_acc}%`)
}
```

# Roadmap
- [x] rewrite all the necesary parts of tinygrad for MNIST, with 'PYTHON' runtime, with tests comparing the python and TS implemenations
- [x] Github CI
- [x] CLANG runtime (WIP)
- [ ] get MNIST training
- [ ] delete lazy + other tinygrad updates
- [ ] get it working inside browser with PYTHON runtime
- [ ] WebGPU runtime
- [ ] publish to npm and jsr
- [ ] AMD runtime 
- [ ] CLOUD runtime
- [ ] add all the missing parts of Tensor and other code that were left out in the beginning.
- [ ] other runtimes 
- [ ] have popular models as a package, maybe even as prebuilt binaries with `deno compile`
- [ ] make it fast