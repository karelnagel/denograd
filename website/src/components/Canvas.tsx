import { useEffect, useRef, useState } from 'react'

type Image = number[][]
export const Canvas = ({ image, setImage }: { image: Image; setImage: (img: Image) => void }) => {
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

  useEffect(() => {
    const ctx = canvasRef.current!.getContext('2d')
    ctx.fillStyle = 'white'
    ctx.fillRect(0, 0, 280, 280)
    
    // Scale up the 28x28 image to 280x280
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = tempCanvas.height = 28
    const tempCtx = tempCanvas.getContext('2d')!
    const imageData = tempCtx.createImageData(28, 28)
    
    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const i = (y * 28 + x) * 4
        const value = 255 - image[y][x]
        imageData.data[i] = value
        imageData.data[i + 1] = value
        imageData.data[i + 2] = value
        imageData.data[i + 3] = 255
      }
    }
    
    tempCtx.putImageData(imageData, 0, 0)
    ctx.drawImage(tempCanvas, 0, 0, 280, 280)
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
      ref={canvasRef}
      width={280}
      height={280}
      style={{ border: '1px solid black', imageRendering: 'pixelated' }}
      onMouseDown={start}
      onMouseMove={draw}
      onMouseUp={end}
      onMouseLeave={end}
    />
  )
}
