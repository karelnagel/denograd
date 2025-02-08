import { useEffect, useState } from 'react'
import { Adam, Device, get_parameters, is_eq, MNIST, mnist, range, Tensor } from '../../../denograd/mod.ts'
import { Canvas } from './Canvas.tsx'
import * as Plot from './Plot.tsx'

const toast = (msg: string) => alert(msg)

console.log(`Using ${Device.DEFAULT} device`)

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))

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
  const [BS, setBS] = useState(256)
  const [steps, setSteps] = useState(30)
  const [currentStep, setCurrentStep] = useState(0)
  const [acc, setAcc] = useState(NaN)
  const [loss, setLoss] = useState(NaN)

  const train = async () => {
    const [X_train, Y_train] = await getData()
    opt.zero_grad()
    setCurrentStep(0)

    const step = async () => {
      Tensor.training = true
      opt.zero_grad()
      const samples = Tensor.randint([BS], undefined, X_train.shape[0])
      const loss = model.call(X_train.get(samples)).sparse_categorical_crossentropy(Y_train.get(samples)).backward()
      await opt.step()
      Tensor.training = false
      return await loss.item()
    }

    for (const i of range(steps)) {
      setLoss(await step())
      if (i % 10 === 9) await test()
      setCurrentStep(i + 1)
    }
  }
  const test = async () => {
    const [_, __, X_test, Y_test] = await getData()
    const acc = await model.call(X_test).argmax(1).eq(Y_test).mean().mul(100).item()
    setAcc(acc)
  }
  return (
    <div className='flex flex-col items-center gap-2'>
      <button
        className='rounded-lg bg-blue-500 p-2 '
        onClick={async () => {
          await model.load()
          toast('Pretrained model loaded')
        }}
      >
        Load pretrained model
      </button>

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
      <p className='text-2xl font-bold'>Training</p>
      <p>Still experimental, WebGPUT can use so much RAM that your PC crashes, mine crashes with BS 512 and STEPS 70</p>
      <label className='flex flex-col text-center'>
        Batch size
        <input type='text' className='bg-transparent outline rounded-md p-1' value={BS.toString()} onChange={(e) => setBS(Number(e.target.value))} />
      </label>
      <label className='flex flex-col text-center'>
        Steps
        <input type='text' className='bg-transparent outline rounded-md p-1' value={steps.toString()} onChange={(e) => setSteps(Number(e.target.value))} />
      </label>
      <button className='rounded-lg bg-blue-500 p-2' onClick={train}>
        Start training for {steps} steps
      </button>
      <button className='rounded-lg bg-blue-500 p-2' onClick={test}>
        Test accuracy
      </button>
      <div>
        loss:{loss.toFixed(2)}, accuracy: {acc.toFixed(2)}, step: {currentStep}/{steps}
      </div>
    </div>
  )
}
