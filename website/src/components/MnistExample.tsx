import { useEffect, useState } from 'react'
import { is_eq, Tensor } from '../../../denograd/mod.ts'
import { MNIST } from '../../../models/mod.ts'
import { barY, plot } from '@observablehq/plot'
import { Canvas } from './Canvas.tsx'

const mnist = new MNIST()

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))

export const MnistExample = () => {
  const [image, setImage] = useState(EMPTY)
  const [res, setRes] = useState<number[]>()
  useEffect(() => {
    const timer = setTimeout(() => {
      if (is_eq(image, EMPTY)) return
      mnist.call(new Tensor([[image]])).reshape([10]).tolist().then(setRes)
    }, 200)
    return () => clearTimeout(timer)
  }, [image])

  useEffect(() => {
    if (!res) return
    const pred = res.map((value: number, i: number) => ({ i, value }))
    const graph = plot({
      title: 'Predictions',
      height: 400,
      marks: [
        barY(pred, {
          x: 'i',
          y: 'value',
          fill: (d) => (d.value === Math.max(...res) ? 'red' : 'steelblue'),
        }),
      ],
      document,
    })
    document.querySelector('#plot')!.replaceChildren(graph)
  }, [res])
  return (
    <>
      <Canvas image={image} setImage={setImage} />
      <button onClick={() => setImage(EMPTY)}>Clear</button>
      <div id='plot'></div>
    </>
  )
}
