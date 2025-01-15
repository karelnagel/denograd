import { useEffect, useState } from 'preact/hooks'
import { is_eq, Tensor } from '../../../denograd/mod.ts'
import { MNIST } from '../../../models/mod.ts'
import { Canvas } from './Canvas.tsx'
import * as Plot from './Plot.tsx'

const loadModel = async () => {
  const mnist = new MNIST()

  let b64 = localStorage.getItem('mnist_model')!
  if (!b64) {
    b64 = await fetch('https://api.github.com/repos/karelnagel/denograd/contents/mnist.safetensors').then((x) => x.json()).then((x) => x.content)
    localStorage.setItem('mnist_model', b64)
  }
  const data = new Uint8Array(atob(b64).split('').map((c) => c.charCodeAt(0)))

  await mnist.load(new Tensor(data, { device: 'PYTHON' }))
  return mnist
}

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))

export const MnistExample = () => {
  const [mnist, setMnist] = useState<MNIST>()
  const [image, setImage] = useState(EMPTY)
  const [res, setRes] = useState([])
  useEffect(() => {
    loadModel().then(setMnist)
  }, [])
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!mnist || is_eq(image, EMPTY)) return
      mnist.call(new Tensor([[image]])).reshape([10]).tolist().then(setRes)
    }, 200)
    return () => clearTimeout(timer)
  }, [image])
  const clear = () => {
    setImage(EMPTY)
    setRes([])
  }
  if (!mnist) return <p>Loading...</p>
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
  )
}
