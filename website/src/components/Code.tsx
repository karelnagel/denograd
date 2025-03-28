import { useEffect, useState } from 'react'
import { Loader2, Play, SquareArrowOutUpRight } from 'lucide-react'
import * as jsg from '../../../jsgrad/web.ts'

if (typeof window !== 'undefined') (window as any).jsg = jsg

export const Code = ({ examples: input }: { examples: Record<string, string> }) => {
  let [examples, setExamples] = useState(input)
  const [selected, setSelected] = useState(Object.keys(input)[0])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    document.querySelector('#code')!.textContent = examples[selected]
  }, [selected])
  useEffect(() => (window as any).Prism?.highlightAll(), [selected])

  const run = async () => {
    setLoading(true)
    document.querySelector('#out')!.textContent = ''
    try {
      let [imp, ...code] = examples[selected].split('\n')
      const imports = imp.split('import {')[1].split("} from '@jsgrad/jsgrad'")[0].split(',').map((x) => x.trim())
      await new Function(`{ ${imports.join(', ')} }`, `return (async () => {\n${code.join('\n')}\n})()`)(Object.fromEntries(imports.map((k) => [k, (jsg as any)[k]])))
    } catch (e) {
      alert(`Error: ${e}`)
    }
    setLoading(false)
  }
  return (
    <div className='relative !bg-[#121212] w-full border-4 border-white/5 h-[400px] text-sm p-3 rounded-xl flex flex-col gap-3'>
      <div className='flex justify-between items-center'>
        <div className='flex gap-1 text-xs'>
          {Object.keys(examples).map((x) => <p key={x} onClick={() => setSelected(x)} className={`cursor-pointer bg-white/5 p-1 rounded-md ${selected === x ? 'text-blue-400' : ''}`}>{x}</p>)}
        </div>
        <a href={`/playground/${selected.replace('.js', '')}`}>
          <SquareArrowOutUpRight className='h-5' />
        </a>
      </div>
      <div className='h-full border border-white/5 h-full p-2 relative flex overflow-hidden'>
        <code
          id='code'
          className='language-js !bg-transparent text-xs h-full w-full overflow-auto'
          onInput={(e) => {
            const code = e.currentTarget.textContent!
            setExamples((x) => ({ ...x, [selected]: code }))
          }}
          contentEditable
        />
        <button
          type='button'
          className='absolute bottom-4 right-4 bg-green-500 p-2 rounded-full hover:bg-green-800 duration-150'
          onClick={run}
        >
          {!loading ? <Play /> : <Loader2 className='animate-spin' />}
        </button>
      </div>
      <div id='out' className='absolute top-[100%] mt-3 w-full text-center'></div>
    </div>
  )
}
