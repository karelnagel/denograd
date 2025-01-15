import { StrictMode, useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { is_eq, Tensor } from '@denograd/denograd'
import { MNIST } from '@denograd/models'

const mnist = new MNIST()

const EMPTY = Array(28).fill(0).map(() => Array(28).fill(0))
type Image = number[][]
const Canvas = ({ image, setImage }: { image: Image; setImage: (img: Image) => void }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)

  useEffect(() => {
    const ctx = canvasRef.current!.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, 280, 280)
    ctx.lineCap = 'round'
    ctx.lineWidth = 20
    ctx.strokeStyle = 'black'
  }, [])

  const updateImage = () => {
    const big = canvasRef.current!
    const small = document.createElement('canvas')
    small.width = small.height = 28
    const scx = small.getContext('2d')!
    scx.drawImage(big, 0, 0, 28, 28)
    const d = scx.getImageData(0, 0, 28, 28).data
    const newImg = Array(28).fill(0).map(() => Array(28).fill(0))
    for (let i = 0; i < d.length; i += 4) {
      const x = (i / 4) % 28, y = Math.floor(i / 4 / 28)
      newImg[y][x] = 255 - d[i]
    }
    setImage(newImg)
  }

  const start = (e: React.MouseEvent) => {
    setIsDrawing(true)
    const r = canvasRef.current!.getBoundingClientRect()
    const ctx = canvasRef.current?.getContext('2d')
    ctx?.beginPath()
    ctx?.moveTo(e.clientX - r.left, e.clientY - r.top)
  }

  const draw = (e: React.MouseEvent) => {
    if (!isDrawing) return
    const r = canvasRef.current!.getBoundingClientRect()
    const ctx = canvasRef.current?.getContext('2d')
    ctx?.lineTo(e.clientX - r.left, e.clientY - r.top)
    ctx?.stroke()
    updateImage()
  }

  const end = () => {
    setIsDrawing(false)
    updateImage()
  }

  const clear = () => {
    const ctx = canvasRef.current!.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, 280, 280)
    setImage(EMPTY)
  }

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: '1px solid black', imageRendering: 'pixelated' }}
        onMouseDown={start}
        onMouseMove={draw}
        onMouseUp={end}
        onMouseLeave={end}
      />
      <button onClick={clear}>clear</button>
    </div>
  )
}

const App = () => {
  const [image, setImage] = useState(EMPTY)
  const [res, setRes] = useState<number[]>([])
  useEffect(() => {
    if (is_eq(image, EMPTY)) return
    const img = new Tensor([[Array(28).fill(0).map(() => Array(28).fill(0))]])
    mnist.call(img).tolist().then((res) => setRes(res))
  }, [image])
  return (
    <>
      <Canvas image={image} setImage={setImage} />
      <div>{JSON.stringify(res)}</div>
    </>
  )
}

createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
