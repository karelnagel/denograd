import { Settings2 } from 'lucide-react'
import { vars } from '../../../jsgrad/helpers/helpers.ts'
import { Device, env } from '../../../jsgrad/web.ts'
import * as jsg from '../../../jsgrad/web.ts'
import { useState } from 'react'

const allVars: (keyof typeof vars)[] = ['DEBUG', 'BEAM', 'JIT', 'NOOPT', 'CACHELEVEL']

if (typeof window !== 'undefined') (window as any).jsg = jsg

export const Header = () => {
  const [dev, _setDev] = useState(Device.DEFAULT)
  const setDev = (dev: string) => {
    _setDev(dev)
    Device.setDefault(dev)
  }
  return (
    <div className='fixed h-16 pt-3 px-6 w-full z-50'>
      <header className='max-w-[1200px] backdrop-blur-md px-6 rounded-full mx-auto w-full h-full flex justify-between items-center border border-white/10'>
        <div className='flex items-center gap-6 text-sm'>
          <a href='/' className='font-secondary text-2xl mr-2'>jsgrad</a>
          <a href='/docs'>Docs</a>
          <a href='/blog'>Blog</a>
          <a href='/chat'>Chat</a>
        </div>
        <div className='flex gap-2 items-center'>
          <a href='https://x.com/jsgrad_org' target='_blank' className='p-1 hover:bg-white/10 rounded-full duration-150'>
            <svg className='h-5 fill-white ' role='img' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
              <title>X</title>
              <path d='M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z' />
            </svg>
          </a>
          <a href='https://github.com/jsgrad-org/jsgrad' target='_blank' className='p-1 hover:bg-white/10 rounded-full duration-150'>
            <svg className='h-5 fill-white' role='img' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
              <title>GitHub</title>
              <path d='M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12' />
            </svg>
          </a>
          <div className='p-1 group hover:bg-white/10 rounded-full duration-150 relative'>
            <Settings2 className='' />
            <div className='absolute hidden group-hover:flex top-[100%] right-0 pt-3'>
              <div className='flex flex-col gap-2 bg-[#121212] p-2 border border-white/10 rounded-md w-80'>
                <div className='flex flex-col gap-2'>
                  <div className='grid grid-cols-2 items-center'>
                    <p>DEVICE</p>
                    <select className='' defaultValue={Device.DEFAULT} onChange={(e) => setDev(e.target.value)}>
                      {Object.keys(env.DEVICES).map((x) => <option key={x}>{x}</option>)}
                    </select>
                  </div>
                  {Device.DEFAULT.startsWith('CLOUD') && <input className='w-full' placeholder='CLOUD host' value={dev.split(':')[1] ?? ''} onChange={(e) => setDev(`CLOUD:${e.target.value}`)} />}
                </div>
                {allVars.map((k) => (
                  <div key={k} className='grid grid-cols-2'>
                    <span>{k}</span>
                    <input type='number' defaultValue={vars[k] as string} onChange={(e) => vars.set(k, e.currentTarget.value)} />
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </header>
    </div>
  )
}
