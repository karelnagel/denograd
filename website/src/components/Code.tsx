import { useEffect, useState } from 'react'
import { Loader2, Play, SquareArrowOutUpRight } from 'lucide-react'
import * as jsg from '../../../jsgrad/web.ts'

export const Code = ({ examples }: { examples: { file: string; code: string }[] }) => {
  const [selected, setSelected] = useState(0)
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    ;(window as any).Prism?.highlightAll()
  }, [selected])
  const run = async () => {
    setLoading(true)
    document.querySelector('#out')!.textContent = ''
    let [imp, ...code] = examples[selected].code.split('\n')
    const imports = imp.split('import {')[1].split("} from '@jsgrad/jsgrad'")[0].split(',').map((x) => x.trim())
    await new Function(`{ ${imports.join(', ')} }`, `return (async () => {\n${code.join('\n')}\n})()`)(Object.fromEntries(imports.map((k) => [k, (jsg as any)[k]])))
    setLoading(false)
  }
  return (
    <div className='relative !bg-[#121212] w-full border-4 border-white/5 h-[400px] text-sm p-3 rounded-xl flex flex-col gap-3'>
      <div className='flex justify-between items-center'>
        <div className='flex gap-1 text-xs'>
          {examples.map((x, i) => <p key={i} onClick={() => setSelected(i)} className={`cursor-pointer bg-white/5 p-1 rounded-md ${selected === i ? 'text-blue-400' : ''}`}>{x.file}</p>)}
        </div>
        <a href={`/playground/${examples[0].file.replace('.js', '')}`}>
          <SquareArrowOutUpRight className='h-5' />
        </a>
      </div>
      <div className='h-full border border-white/5 h-full p-2 relative flex overflow-auto'>
        <code className='language-js !bg-transparent text-xs h-full'>
          {examples[selected].code}
        </code>
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
