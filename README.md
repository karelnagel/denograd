# jsgrad - A Modern ML Library for JavaScript and TypeScript

jsgrad is a rewrite of [tinygrad](https://tinygrad.org/) in TypeScript. JS ecosystem is very large, but it didn't have a good ML library for model inference and **training**. Since tinygrad doesn't use any external python libraries, has potential to be the fastest way to run models, is quite simple compared to others and supports many runtimes, I decided to rewrite it in TS to get the same experience in browser and in deno/node/bun.

Why you should use jsgrad?

- 0 dependencies
- will be fast (not yet)
- Multiple runtime backends (WebGPU, WASM, CLANG, + others coming soon)
- Clean, modern API inspired by tinygrad's elegant design
- Works in browser and in Deno (Node and Bun support coming soon)

See MNIST inference and training example on [jsgrad.org](https://jsgrad.org)

# Usage

There are multiple ways to use jsgrad:

## Hosted esm script in JS ([minimal Llama HTLM example](/llama.html))

```js
import { MNIST, Tensor } from 'https://esm.sh/jsr/@jsgrad/jsgrad'

const mnist = await new MNIST().load()
console.log(await mnist.call(Tensor.ones([1, 1, 28, 28])).tolist())
```

## Install package from [jsr.io](https://jsr.io/@jsgrad/jsgrad)

```bash
# with deno
deno add jsr:@jsgrad/jsgrad
# with npm
npx jsr add @jsgrad/jsgrad
# with yarn
yarn dlx jsr add @jsgrad/jsgrad
# with pnpm
pnpm dlx jsr add @jsgrad/jsgrad
# with bun
bunx jsr add @jsgrad/jsgrad
```

and then import with

```ts
import { MNIST, Tensor } from '@jsgrad/jsgrad'

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
const whisper = await new Whisper({ model: 'large-v2' }).load()
const listening = whisper.startListening()
// after some time
const text = await listening.stop()
```

### Text to speech

```ts
const tts = await new TTS()
const audio = await tts.run({ text: 'Hello how are you?' })
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
