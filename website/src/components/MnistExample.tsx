// @deno-types="npm:@types/react"
import { useEffect, useState } from 'react'
import { Adam, Device, env, get_parameters, is_eq, MNIST, mnist, perf, round, Tensor, TinyJit } from '../../../jsgrad/web.ts'
import { Canvas } from './Canvas.tsx'
import * as Plot from './Plot.tsx'

const toast = (msg: string) => alert(msg)

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))

export const MnistExample = () => {
  // Inference
  const [image, setImage] = useState(EMPTY)
  const [res, setRes] = useState<number[]>([])
  const [model, setModel] = useState(() => new MNIST())
  const [opt, setOpt] = useState(() => Adam(get_parameters(model)))
  const [device, _setDevice] = useState(Device.DEFAULT)
  const setDevice = (device: string) => {
    Device.setDefault(device)
    _setDevice(device)
    const mnist = new MNIST()
    setModel(mnist)
    setOpt(Adam(get_parameters(mnist)))
  }
  useEffect(() => {
    const timer = setTimeout(async () => {
      if (!model || is_eq(image, EMPTY)) return
      const res = await model.call(new Tensor([[image]])).reshape([10])
        .tolist()
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
  const [maxSteps, setMaxSteps] = useState(70)
  const [steps, setSteps] = useState<
    { acc?: number; duration?: number; step: number; loss?: number }[]
  >([])

  const train = async ([X_train, Y_train, X_test, Y_test]: Tensor[]) => {
    setSteps([])

    const trainStep = new TinyJit(async () => {
      Tensor.training = true
      opt.zero_grad()
      const samples = Tensor.randint([BS], undefined, X_train.shape_num[0])
      const loss = model.call(X_train.get(samples))
        .sparse_categorical_crossentropy(Y_train.get(samples)).backward()
      await opt.step()
      Tensor.training = false
      return loss
    })
    const test = async () => {
      const res = await new TinyJit(async () => model.call(X_test).argmax(1).eq(Y_test).mean().mul(100)).call()
      return await res.item()
    }
    const acc = await test()
    setSteps((steps) => [...steps, { acc, step: 0, duration: undefined }])
    for (let step = 0; step < maxSteps; step++) {
      let st = performance.now()
      const loss = await trainStep.call().then((x) => x.item())
      const duration = perf(st)
      const acc = (step % 10 === 9) ? await test() : undefined
      setSteps((steps) => [...steps, { loss, acc, step: step + 1, duration }])
    }
  }

  const maxDuration = steps.length ? Math.max(...steps.map((x) => x.duration!).filter(Boolean)) : 0
  const currentStep = steps.at(-1)
  return (
    <div className='flex flex-col items-center gap-6'>
      <p>Choose Device</p>
      <div className='flex gap-3'>
        <select
          defaultValue={device}
          onChange={(e) => setDevice(e.target.value)}
        >
          {Object.keys(env.DEVICES).map((x) => <option key={x} value={x}>{x}</option>)}
        </select>
        {device.startsWith('CLOUD') && (
          <input
            value={device.replace(/CLOUD:?/, '')}
            onChange={(e) => setDevice(`CLOUD:${e.target.value}`)}
            placeholder='CLOUD URL'
            className=' w-96'
          />
        )}
      </div>
      <label className='flex flex-col text-center'>
        Batch size
        <input
          type='text'
          className=''
          value={BS.toString()}
          onChange={(e) => setBS(Number(e.target.value))}
        />
      </label>
      <label className='flex flex-col text-center'>
        Steps
        <input
          type='text'
          className=''
          value={maxSteps.toString()}
          onChange={(e) => setMaxSteps(Number(e.target.value))}
        />
      </label>
      <button
        type='button'
        className='btn'
        onClick={async () => {
          const data = await getData()
          await train(data)
        }}
      >
        Start training for {maxSteps} steps
      </button>
      <Plot.Plot
        className='bg-white text-black rounded-lg'
        options={{
          x: { label: 'step', grid: true, domain: [0, maxSteps] },
          y: { label: 'accuracy', grid: true, domain: [0, 100] },
          height: 400,
          width: 1000,
          marks: [
            Plot.axisY({
              color: 'red',
              anchor: 'left',
              label: 'loss',
              tickFormat: (x) => (x / 33.33).toFixed(1),
            }),
            Plot.axisY({
              color: 'blue',
              anchor: 'right',
              label: 'accuracy (%)',
            }),
            Plot.line(steps, {
              x: (d) => d.step,
              y: (d) => d.loss * 33.33,
              stroke: 'red',
              strokeWidth: 2,
            }),
            Plot.line(steps.filter(({ acc }) => acc), {
              x: (d) => d.step,
              y: (d) => d.acc,
              stroke: 'blue',
              strokeWidth: 2,
            }),
            Plot.line(steps, {
              x: (d) => d.step,
              y: (d) => d.duration / maxDuration * 100,
              strokeWidth: 2,
              stroke: 'green',
            }),
          ],
          color: {
            legend: true,
            domain: ['Accuracy', 'Loss', `Step time (max: ${maxDuration}s)`],
            range: ['blue', 'red', 'green'],
          },
        }}
      />
      {currentStep && (
        <div>
          loss:{currentStep.loss?.toFixed(2)}, accuracy: {currentStep.acc?.toFixed(2)}, step: {currentStep.step}/{maxSteps}, step time: {round(currentStep.duration || 0)}s
        </div>
      )}

      <br />
      <button
        type='button'
        className='btn'
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
            type='button'
            onClick={() => {
              setImage(EMPTY)
              setRes([])
            }}
            className='btn'
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

      <br />
      <a
        className='text-blue-400'
        href='https://github.com/karelnagel/jsgrad/blob/main/website/src/components/MnistExample.tsx'
      >
        See this page's code here
      </a>
    </div>
  )
}
