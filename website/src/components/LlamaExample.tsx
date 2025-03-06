// @deno-types="npm:@types/react"
import { useEffect, useState } from 'react'
import { Device, Llama3, Tensor, Tokenizer, Tqdm, type TqdmProgress } from '../../web.ts'

export const LlamaExample = ({ initOnLoad }: { initOnLoad?: boolean }) => {
  initOnLoad = false
  const [model, setModel] = useState<Llama3>()
  const [tokenizer, setTokenizer] = useState<Tokenizer>()
  const [startPos, setStartPos] = useState<number>()
  const [progress, _setProgress] = useState<TqdmProgress>()
  const setProgress = (progress: TqdmProgress | undefined) => {
    document.querySelector('#progress')!.textContent = JSON.stringify(progress)
    _setProgress(progress)
  }
  const [message, setMessage] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string; tokens: number; tps?: number; ttft?: number }[]>([])

  useEffect(() => void (initOnLoad ? init() : undefined), [])

  const init = async () => {
    if (model || progress) return
    Tensor.no_grad = true
    const _model = new Llama3('1B', 'int8')
    const weights = await _model.download(undefined, (p) => setProgress(p))
    await _model.load(weights, undefined, Device.DEFAULT, (p) => setProgress(p))
    setProgress({ label: 'Loading tokenizer' })
    const _tokenizer = await Tokenizer.init(`${weights.split('/').slice(0, -1).join('/')}/tokenizer.model`)
    const system = [_tokenizer.bos_id, ..._tokenizer.encode_message('system', 'You are an helpful assistant.')]
    setStartPos(await _model.prefill(system, undefined, Device.DEFAULT, (p) => setProgress(p)))
    setProgress(undefined)
    setModel(_model)
    setTokenizer(_tokenizer)
  }

  const chat = async () => {
    if (!tokenizer || !model) throw new Error('Tokenizer or model not loaded')
    if (progress) return

    let st = performance.now()
    const toks = [...tokenizer.encode_message('user', message), ...tokenizer.encode_role('assistant')]
    setMessages((x) => [...x, { role: 'user', content: message, tokens: toks.length }, { role: 'assistant', content: '', tokens: 0 }])
    setMessage('')
    let start_pos = await model.prefill(toks.slice(0, -1), startPos, Device.DEFAULT, (p) => setProgress(p))
    const ttft = (performance.now() - st) / 1000
    st = performance.now()
    let last_tok = toks.at(-1)
    for (const _ of new Tqdm(undefined, { label: 'Generating', onProgress: setProgress })) {
      const tok = await model.call(new Tensor([[last_tok]]), start_pos)
      start_pos += 1
      last_tok = tok
      if (tokenizer.stop_tokens.includes(tok)) break
      setMessages((x) => {
        const old = x.slice(0, -1), last = x.at(-1)!, tokens = last.tokens + 1
        return [
          ...old,
          { ...last, content: last.content + tokenizer.decode([tok]), tokens, tps: tokens / ((performance.now() - st) / 1000), ttft },
        ]
      })
    }
    setProgress(undefined)
    setStartPos(start_pos)
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
      <div className='flex gap-3'>
        <p>{messages.at(-1)?.ttft?.toFixed(2) || 0} sec to first token</p>
        <p>{messages.at(-1)?.tps?.toFixed(2) || 0} tokens/sec</p>
        <p>{messages.at(-1)?.tokens} tokens</p>
      </div>
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
