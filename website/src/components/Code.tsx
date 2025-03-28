import { useEffect, useState } from 'react'
import { Play } from 'lucide-react'

export const Code = ({ examples }: { examples: { file: string; code: string }[] }) => {
  const [selected, setSelected] = useState(0)
  useEffect(() => {
    ;(window as any).Prism?.highlightAll()
  }, [selected])
  return (
    <div className='relative !bg-[#121212] w-full border-4 border-white/5 h-[400px] text-sm p-3 rounded-xl flex flex-col gap-3'>
      <div className='flex gap-1 text-xs'>
        {examples.map((x, i) => <p key={i} onClick={() => setSelected(i)} className={`cursor-pointer bg-white/5 p-1 rounded-md ${selected === i ? 'text-blue-400' : ''}`}>{x.file}</p>)}
      </div>
      <div className='h-full border border-white/5 h-full p-2 relative flex overflow-auto'>
        <code className='language-js !bg-transparent text-xs h-full'>
          {examples[selected].code}
        </code>
        <button type='button' className='absolute bottom-4 right-4 bg-green-500 p-2 rounded-full hover:bg-green-800 duration-150'>
          <Play />
        </button>
      </div>
    </div>
  )
}
