import { useEffect, useRef, useState } from 'react'
import { is_eq } from '../../../denograd/helpers.ts'

type Image = number[][]
const H = 280
export const Canvas = ({ image, setImage, className }: { className?: string; image: Image; setImage: (img: Image) => void }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)

  useEffect(() => {
    const ctx = canvasRef.current!.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, H, H)
    ctx.lineCap = 'round'
    ctx.lineWidth = 20
    ctx.strokeStyle = 'black'
  }, [])

  // Used for clearing the canvas only
  useEffect(() => {
    if (!is_eq(image, Array(28).fill(0).map(() => Array(28).fill(0)))) return
    const ctx = canvasRef.current!.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, H, H)
  }, [image])

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

  return (
    <canvas
      className={className}
      ref={canvasRef}
      width={H}
      height={H}
      style={{ imageRendering: 'pixelated' }}
      onMouseDown={start}
      onMouseMove={draw}
      onMouseUp={end}
      onMouseLeave={end}
    />
  )
}
