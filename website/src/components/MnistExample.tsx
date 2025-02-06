import { useEffect, useState } from 'preact/hooks'
import { Adam, Device, get_parameters, is_eq, mnist, range, Tensor } from '../../../denograd/mod.ts'
import { MNIST } from '../../../models/mod.ts'
import { Canvas } from './Canvas.tsx'
import * as Plot from './Plot.tsx'

const toast = (msg: string) => alert(msg)

console.log(`Using ${Device.DEFAULT} device`)

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))

Tensor.manual_seed(5)
const model = new MNIST()
const opt = Adam(get_parameters(model))

export const MnistExample = () => {
  // Inference
  const [image, setImage] = useState(EMPTY)
  const [res, setRes] = useState([])

  useEffect(() => {
    const timer = setTimeout(async () => {
      if (!model || is_eq(image, EMPTY)) return
      const res = await model.call(new Tensor([[image]])).reshape([10]).tolist()
      setRes(res)
    }, 200)
    return () => clearTimeout(timer)
  }, [image])

  // Training
  const [data, setData] = useState<Tensor[]>()
  const getData = async () => {
    if (data) return data
    const newData = await mnist(undefined, '/')
    setData(newData)
    return newData
  }
  const [BS, setBS] = useState(512)
  const [steps, setSteps] = useState(10)
  const [currentStep, setCurrentStep] = useState(0)
  const [acc, setAcc] = useState(NaN)
  const [loss, setLoss] = useState(NaN)

  const train = async () => {
    const [X_train, Y_train] = await getData()
    opt.zero_grad()
    setCurrentStep(0)
    for (const step of range(steps)) {
      Tensor.training = true
      const samples = Tensor.randint([BS], undefined, X_train.shape[0])
      const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
      await opt.step()
      Tensor.training = false
      setLoss(await loss.item())
      if (step % 10 === 9) await test()
      setCurrentStep(step + 1)
    }
  }
  const test = async () => {
    const [_, __, X_test, Y_test] = await getData()
    const acc = await model.call(X_test).argmax(1).eq(Y_test).mean().mul(100).item()
    setAcc(acc)
  }
  return (
    <div className='flex flex-col items-center gap-2'>
      <label className='flex flex-col text-center'>
        Seed
        <input type='text' className='bg-transparent outline rounded-md p-1' value={Tensor._seed} onChange={(e) => Tensor.manual_seed(Number(e.target.value))} />
      </label>
      <label className='flex flex-col text-center'>
        Batch size
        <input type='text' className='bg-transparent outline rounded-md p-1' value={BS.toString()} onChange={(e) => setBS(Number(e.target.value))} />
      </label>
      <label className='flex flex-col text-center'>
        Steps
        <input type='text' className='bg-transparent outline rounded-md p-1' value={steps.toString()} onChange={(e) => setSteps(Number(e.target.value))} />
      </label>
      <button
        onClick={async () => {
          await model.load(await Tensor.from_url('/mnist.safetensors', { device: 'PYTHON' }))
          toast('Pretrained model loaded')
        }}
      >
        Load pretrained model
      </button>
      <button onClick={train}>
        Start training
      </button>
      <button onClick={test}>
        Test
      </button>
      <div>
        loss:{loss.toFixed(2)}, accuracy: {acc.toFixed(2)}, step: {currentStep}/{steps}
      </div>
      <div className='flex flex-col md:flex-row gap-20 items-center'>
        <div className=' flex flex-col items-center gap-2'>
          <p className='text-xl font-bold'>Draw a digit here</p>
          <Canvas image={image} setImage={setImage} className='rounded-md' />
          <button
            onClick={() => {
              setImage(EMPTY)
              setRes([])
            }}
            className='p-1 outline outline-white rounded-lg px-3'
          >
            Clear
          </button>
        </div>
        <div>
          <Plot.Plot
            className='bg-white text-black rounded-lg'
            options={{
              x: { label: 'number' },
              y: { label: 'probability', grid: true },
              height: 280,
              width: 600,
              marks: [Plot.barY(res, {
                x: (_, i) => i,
                y: (d) => d,
                fill: (d) => (d === Math.max(...res) ? 'red' : 'black'),
              })],
            }}
          />
        </div>
      </div>
    </div>
  )
}
