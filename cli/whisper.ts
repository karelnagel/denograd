import { init_whisper, LANGUAGES, MODELS, transcribe_file, type WhisperModel } from '../jsgrad/mod.ts'
import { parseArgs, z } from './parse.ts'

const args = parseArgs({
  input: z.string().describe('Audio path or url'),
  model: z.enum(Object.keys(MODELS) as WhisperModel[]).default('tiny.en')
    .describe('Whisper model'),
  batch_size: z.number().default(1).describe('Batch size'),
  language: z.enum(Object.keys(LANGUAGES)).optional().describe(
    'Language, only works with multilingual models',
  ),
})

const [model, enc] = await init_whisper(args.model, args.batch_size)
const res = await transcribe_file(model, enc, args.input, args.language)
console.log(res)
