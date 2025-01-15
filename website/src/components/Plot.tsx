import { CSSProperties, useEffect, useRef } from 'react'
import { plot, PlotOptions } from '@observablehq/plot'
export * from '@observablehq/plot'

export const Plot = ({ style, className, ...options }: PlotOptions & { style?: CSSProperties; className?: string }) => {
  const ref = useRef(null)
  useEffect(() => {
    if (!ref.current) return
    const graph = plot(options)
    ref.current?.replaceChildren(graph)
  }, [options])
  return <div ref={ref} style={style} className={className}></div>
}
