import { useEffect, useRef, useState } from 'react'
import { is_eq } from '../../../jsgrad/helpers.ts'

type Image = number[][]
const H = 280
export const Canvas = (
  { image, setImage, className }: {
    className?: string
    image: Image
    setImage: (img: Image) => void
  },
) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [lastPos, setLastPos] = useState<{ x: number; y: number }>()

  useEffect(() => {
    const ctx = canvasRef.current!.getContext('2d')!
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, H, H)
    ctx.lineCap = 'round'
    ctx.lineWidth = 20
    ctx.strokeStyle = 'black'
  }, [])

  // Used for clearing the canvas only
  useEffect(() => {
    if (!is_eq(image, Array(28).fill(0).map(() => Array(28).fill(0)))) return
    const ctx = canvasRef.current!.getContext('2d')!
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

  const getCoordinates = (e: any) => {
    const r = canvasRef.current!.getBoundingClientRect()
    if ('touches' in e) {
      return {
        x: e.touches[0].clientX - r.left,
        y: e.touches[0].clientY - r.top,
      }
    }
    return {
      x: e.clientX - r.left,
      y: e.clientY - r.top,
    }
  }

  const start = (e: any) => {
    setIsDrawing(true)
    const pos = getCoordinates(e)
    setLastPos(pos)
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    ctx.beginPath()
    ctx.moveTo(pos.x, pos.y)
    ctx.lineTo(pos.x, pos.y)
    ctx.stroke()
  }

  const draw = (e: any) => {
    if (!isDrawing || !lastPos) return
    const ctx = canvasRef.current?.getContext('2d')
    if (!ctx) return
    const newPos = getCoordinates(e)
    ctx.beginPath()
    ctx.moveTo(lastPos.x, lastPos.y)
    ctx.lineTo(newPos.x, newPos.y)
    ctx.stroke()
    setLastPos(newPos)
    updateImage()
  }

  const end = () => {
    setIsDrawing(false)
    setLastPos(undefined)
    updateImage()
  }

  return (
    <canvas
      className={className}
      ref={canvasRef}
      width={H}
      height={H}
      style={{ imageRendering: 'pixelated', touchAction: 'none' }}
      onMouseDown={start}
      onMouseMove={draw}
      onMouseUp={end}
      onMouseLeave={end}
      onTouchStart={start}
      onTouchMove={draw}
      onTouchEnd={end}
      onTouchCancel={end}
    />
  )
}
