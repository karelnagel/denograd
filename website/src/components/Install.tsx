import { useEffect, useState } from 'react'
import { CopyIcon } from 'lucide-react'
import { toast } from 'sonner'

const usage = `import { Tensor } from '@jsgrad/jsgrad'\n\nconsole.log(await new Tensor([2, 2, 2]).add(5).tolist())`
const options = {
  npm: {
    install: `npm install @jsgrad/jsgrad`,
    usage,
  },
  html: {
    install: undefined,
    usage: `<script type='module'>
    import { Tensor } from 'https://esm.sh/@jsgrad/jsgrad'

    console.log(await new Tensor([2, 2, 2]).add(5).tolist())
</script>
`,
  },
  deno: {
    install: `deno install npm:@jsgrad/jsgrad`,
    usage,
  },
  bun: {
    install: `bun install @jsgrad/jsgrad`,
    usage,
  },
}

const Code = ({ code, lang = 'js' }: { lang?: string; code: string }) => {
  useEffect(() => (window as any).Prism?.highlightAll(), [code])
  return (
    <div
      className='relative bg-[#121212] border rounded-md p-2 px-4 group shadow-sm shadow-white'
      onClick={() => {
        navigator.clipboard.writeText(code)
        toast('Copied!')
      }}
    >
      <code className={`!bg-transparent language-${lang}`}>{code}</code>
      <div className='absolute hidden group-hover:flex right-2 top-2 bg-white text-black rounded-full p-[5px] cursor-pointer'>
        <CopyIcon className='h-4 w-4' />
      </div>
    </div>
  )
}

export const Install = () => {
  const [selected, setSelected] = useState<keyof typeof options>('npm')
  return (
    <div className='flex flex-col gap-2 w-full max-w-screen-sm'>
      <div className='flex gap-2 justify-center'>
        {Object.keys(options).map((k) => (
          <button type='button' className={`flex gap-2 p-2 py-1 items-center rounded-lg  text-black ${selected === k ? 'bg-blue-300 ext-white' : 'bg-white'}`} key={k} onClick={() => setSelected(k as any)}>
            <img className='h-4' src={`/${k}.png`} />
            <span>{k}</span>
          </button>
        ))}
      </div>
      {options[selected].install && (
        <>
          <p>Install with</p>
          <Code lang='shell' code={options[selected].install} />
        </>
      )}
      <p>Usage</p>
      <Code lang={selected === 'html' ? 'html' : 'js'} code={options[selected].usage} />
    </div>
  )
}
