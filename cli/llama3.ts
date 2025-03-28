import { env, Llama3, Tensor } from '../jsgrad/node.ts'
import { parseArgs, z } from './parse.ts'

const args = parseArgs({
  // model: z.string().optional().describe('Model path'),
  size: z.enum(['1B', '8B', '70B']).default('1B').describe('Model size'),
  // shard: z.number().int().default(1).describe('Shard the model across multiple devices'),
  quantize: z.enum(['int8', 'nf4', 'float16']).optional().describe(
    'Quantization method',
  ),
  seed: z.number().optional().describe('Random seed'),
  temperature: z.number().default(0.85).describe('Temperature'),
  query: z.string().optional().describe('Query'),
})

// download_model is the default without a model passed in
if (args.seed !== undefined) Tensor.manual_seed(args.seed)
console.log(`seed = ${Tensor._seed}`)
const model = await Llama3.load({
  size: args.size,
  temperature: args.temperature,
  quantize: args.quantize,
})

if (args.query) {
  const res = await model.chat({
    messages: [{ role: 'user', content: args.query }],
  })
  console.log(
    `${res.message.content} (${res.usage.tokens_per_second.toFixed(1)} tokens/s)`,
  )
} else {
  while (true) {
    const content = (await env.prompt('Q: '))!
    await model.chat({
      messages: [{ role: 'user', content }],
      onToken: (res) => {
        env.writeStdout(res.token)
      },
    })
    env.writeStdout('\n')
  }
}
