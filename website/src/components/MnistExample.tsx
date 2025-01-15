import { useEffect, useState } from 'react'
import { is_eq, Tensor } from '../../../denograd/mod.ts'
import { MNIST } from '../../../models/mod.ts'
import { Canvas } from './Canvas.tsx'
import { barY, Plot } from './Plot.tsx'

const mnist = new MNIST()

const res = await fetch('https://api.github.com/repos/karelnagel/denograd/contents/mnist.safetensors').then((x) => x.json())
const data = new Uint8Array(atob(res.content).split('').map((c) => c.charCodeAt(0)))

await mnist.load(new Tensor(data, { device: 'PYTHON' }))

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))

export const MnistExample = () => {
  const [image, setImage] = useState(EMPTY)
  const [res, setRes] = useState([])
  useEffect(() => {
    const timer = setTimeout(() => {
      if (is_eq(image, EMPTY)) return
      mnist.call(new Tensor([[image]])).reshape([10]).tolist().then(setRes)
    }, 200)
    return () => clearTimeout(timer)
  }, [image])
  const clear = () => {
    setImage(EMPTY)
    setRes([])
  }
  return (
    <div className='flex flex-col md:flex-row gap-20 items-center'>
      <div className=' flex flex-col items-center gap-2'>
        <p className='text-xl font-bold'>Draw a digit here</p>
        <Canvas image={image} setImage={setImage} className='rounded-md' />
        <button onClick={clear} className='p-1 outline outline-white rounded-lg px-3'>
          Clear
        </button>
      </div>
      <div>
        <Plot
          className='bg-white rounded-lg'
          height={280}
          width={600}
          marks={[barY(res, {
            x: (_, i) => i,
            y: (d) => d,
            fill: (d) => (d === Math.max(...res) ? 'red' : 'steelblue'),
          })]}
        />
      </div>
    </div>
  )
}
