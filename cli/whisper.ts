import { env, init_whisper, transcribe_file } from '../denograd/mod.ts'

const [model, enc] = await init_whisper('tiny.en', 1)
const res = await transcribe_file(model, enc, env.args()[0])
console.log(res)
