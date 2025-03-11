import z from 'zod'
import { env, Llama3, string_to_bytes, Tensor } from '../denograd/mod.ts'
import { parseArgs } from './zod-cli.ts'

const args = parseArgs(
  z.object({
    // model: z.string().optional().describe('Model path'),
    size: z.enum(['1B', '8B', '70B']).default('1B').describe('Model size'),
    // shard: z.number().int().default(1).describe('Shard the model across multiple devices'),
    quantize: z.enum(['int8', 'nf4', 'float16']).optional().describe('Quantization method'),
    seed: z.number().optional().describe('Random seed'),
    temperature: z.number().default(0.85).describe('Temperature'),
    query: z.string().optional().describe('Query'),
  }).describe('Run Llama 3 locally, supported sizes: 1B, 8B and 70B'),
)

// download_model is the default without a model passed in
if (args.seed !== undefined) Tensor.manual_seed(args.seed)
console.log(`seed = ${Tensor._seed}`)
const model = await Llama3.load({
  size: '1B',
  temperature: args.temperature,
  quantize: args.quantize,
})

if (args.query) {
  const res = await model.chat({ messages: [{ role: 'user', content: args.query }] })
  console.log(res.message.content)
} else {
  while (true) {
    const content = prompt('Q: ')!
    await model.chat({
      messages: [{ role: 'user', content }],
      onToken: (res) => {
        env.writeStdout(string_to_bytes(res.token))
      },
    })
    env.writeStdout(string_to_bytes('\n'))
  }
}
