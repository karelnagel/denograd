import { useEffect, useRef } from 'preact/hooks'
import { plot, PlotOptions } from '@observablehq/plot'
export * from '@observablehq/plot'

export const Plot = ({ style, className, options }: { options: PlotOptions; style?: any; className?: string }) => {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    const graph = plot(options)
    ref.current?.replaceChildren(graph)
  }, [options])
  return <div ref={ref} style={style} className={className}></div>
}
