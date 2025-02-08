# Denograd - A Modern ML Library for JavaScript and TypeScript

Denograd is a rewrite of [tinygrad](https://tinygrad.org/) in TypeScript. JS ecosystem is very large, but it didn't have a good ML library for model inference and **training**. Since tinygrad doesn't use any external python libraries, has potential to be the fastest way to run models, is quite simple compared to others and supports many runtimes, I decided to rewrite it in TS to get the same experience in browser and in deno/node/bun.

Why you should use Denograd?
- 0 dependencies
- will be fast (not yet)
- Multiple runtime backends (WebGPU, CLANG, + others coming soon)
- Clean, modern API inspired by tinygrad's elegant design
- Works in browser and in Deno (Node and Bun support coming soon)

See MNIST inference and training example with WebGPU on [denograd.com](https://denograd.com)

# Usage

There are multiple ways to use denograd:

## Hosted script in HTML
```html
<html>
  <head>
    <!-- Makes `Denograd`, `Tensor` and `nn` available globally -->
    <script src="https://denograd.com/denograd.js"></script>
    <script>
      const run = async () => {
        const mnist = await new Denograd.MNIST().load()
        console.log(await mnist.call(Tensor.ones([1, 1, 28, 28])).tolist())
      }
      run()
    </script>
  </head>
  <body></body>
</html>

```

## Hosted esm script in JS
```js
import { Tensor, MNIST } from "https://denograd.com/denograd.mjs"

const mnist = await new MNIST().load()
console.log(await mnist.call(Tensor.ones([1, 1, 28, 28])).tolist()) 
```

## Install package from [jsr.io](https://jsr.io/@denograd/denograd)
```bash
# with deno
deno add jsr:@denograd/denograd
# with npm
npx jsr add @denograd/denograd
# with yarn
yarn dlx jsr add @denograd/denograd
# with pnpm
pnpm dlx jsr add @denograd/denograd
# with bun
bunx jsr add @denograd/denograd
```

and then import with 
```ts
import { Tensor, MNIST } from "@denograd/denograd"

const mnist = await new MNIST().load()
console.log(await mnist.call(Tensor.ones([1, 1, 28, 28])).tolist()) 
```


# Goal - The easiest and fastest way to run and train models in JS/TS.

Soon everything should work like this in browser and server with no install step, while still being fast:

### Uses the fastest local runtime for Llama
```ts
const llama = await new Llama({ model: '3.1-3B' }).load()
const res = await llama.run({ prompt: 'Hello how are you?' })
```

### Offload the computation to CLOUD
```ts
const llama = await new Llama({ model: '3.1-3B', device: 'CLOUD', host: process.env.CLOUD_HOST }).load()
const res = await llama.run({ prompt: 'Hello how are you?' })
```

### Whisper
```ts
const whisper = await new Whisper({ model: "large-v2" }).load()
const listening = whisper.startListening()
// after some time 
const text = await listening.stop()
```

### Text to speech
```ts
const tts = await new TTS()
const audio = await tts.run({ text: "Hello how are you?" })
audio.play()
```

### Training new models:
```ts
class MNIST extends Model {
  layers: Layer[] = [
    new nn.Conv2d(1, 32, 5),
    Tensor.relu,
    new nn.Conv2d(32, 32, 5),
    Tensor.relu,
    new nn.BatchNorm(32),
    Tensor.max_pool2d,
    new nn.Conv2d(32, 64, 3),
    Tensor.relu,
    new nn.Conv2d(64, 64, 3),
    Tensor.relu,
    new nn.BatchNorm(64),
    Tensor.max_pool2d,
    (x) => x.flatten(1),
    new nn.Linear(576, 10),
  ]
}

const [X_train, Y_train, X_test, Y_test] = await mnist()

const model = new MNIST()
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
```

# Roadmap

- [x] rewrite all the necesary parts of tinygrad for MNIST, with 'JS' runtime, with tests comparing the python and TS implemenations
- [x] Github CI
- [x] CLANG runtime
- [x] get MNIST training
- [x] get working inside browser with JS runtime
- [x] WebGPU runtime (MNIST inference running in WebGPU: https://karelnagel.github.io/denograd/, training coming soon)
- [x] delete lazy + other tinygrad updates
- [x] add all the missing parts of Tensor and other code that were left out in the beginning.
- [x] get hand_coded_optimisations working correctly
- [x] WEBGPU training MNIST in browser
- [ ] docs
- [ ] some LLM
- [ ] whisper
- [ ] bun support
- [ ] node support
- [ ] WASM runtime
- [ ] have popular models as a package, maybe even as prebuilt binaries with `deno compile`
- [ ] CLOUD runtime
- [ ] METAL runtime
- [ ] AMD runtime
- [ ] Nvidia runtime
- [ ] JIT
- [ ] Multi
