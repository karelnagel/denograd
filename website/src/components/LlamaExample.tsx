// @deno-types="npm:@types/react"
import { useEffect, useState } from 'react'
import { Llama3, type Llama3Message, type Llama3Usage, type TqdmProgress } from '../../../denograd/web.ts'

export const LlamaExample = ({ initOnLoad }: { initOnLoad?: boolean }) => {
  initOnLoad = false
  const [model, setModel] = useState<Llama3>()
  const [progress, onProgress] = useState<TqdmProgress>()
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<Llama3Message[]>([])
  const [usage, setUsage] = useState<Llama3Usage>()

  useEffect(() => void (initOnLoad ? init() : undefined), [])

  const init = async () => {
    if (model || progress) return
    const _model = await Llama3.load({ size: '1B', quantize: 'float16', system: 'You are an helpful assistant.', onProgress })
    onProgress(undefined)
    setModel(_model)
  }

  const chat = async () => {
    if (!model || progress) return

    setMessages((x) => [...x, { role: 'user', content: message }, { role: 'assistant', content: '' }])
    setMessage('')
    await model.chat({
      messages: [{ role: 'user', content: message }],
      onToken: (res) => {
        setMessages((x) => [...x.slice(0, -1), res.message])
        setUsage(res.usage)
      },
      onProgress,
    })
    onProgress(undefined)
  }

  return (
    <div className='flex flex-col h-screen w-screen justify-between'>
      <div className='h-full w-full flex flex-col'>
        {messages.length === 0
          ? (
            <div className='h-full w-full flex items-center justify-center'>
              <p className='text-7xl'>denochat</p>
            </div>
          )
          : (
            <div>
              {messages.map((msg, i) => (
                <div key={i}>
                  <p>{msg.content}</p>
                </div>
              ))}
            </div>
          )}
      </div>
      <p id='progress'>{JSON.stringify(progress)}</p>
      {!!usage && (
        <div className='flex gap-3'>
          <p>{usage.time_to_first_token.toFixed(2) || 0} sec to first token</p>
          <p>{usage.tokens_per_second?.toFixed(2) || 0} tokens/sec</p>
          <p>{usage.output_tokens} tokens</p>
        </div>
      )}
      <form
        onClick={init}
        onSubmit={(e) => {
          e.preventDefault()
          chat()
        }}
      >
        <textarea value={message} onChange={(e) => setMessage(e.target.value)} />
        <button type='submit'>Send</button>
      </form>
    </div>
  )
}
