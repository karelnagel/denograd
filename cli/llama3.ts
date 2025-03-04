import z from 'zod'
import { Device, env, GlobalCounters, Llama3, range, string_to_bytes, Tensor, Tokenizer } from '../denograd/mod.ts'
import { parseArgs } from './zod-cli.ts'

const args = parseArgs(
  z.object({
    model: z.string().optional().describe('Model path'),
    size: z.enum(['1B', '8B', '70B']).default('1B').describe('Model size'),
    shard: z.number().int().default(1).describe('Shard the model across multiple devices'),
    quantize: z.enum(['int8', 'nf4', 'float16']).optional().describe('Quantization method'),
    seed: z.number().optional().describe('Random seed'),
    temperature: z.number().default(0.85).describe('Temperature'),
  }).describe('Run Llama 3 locally, supported sizes: 1B, 8B and 70B'),
)
Tensor.no_grad = true

// download_model is the default without a model passed in
const model = new Llama3(args.size, args.quantize)
model.TEMPERATURE = args.temperature
if (args.seed !== undefined) Tensor.manual_seed(args.seed)
console.log(`seed = ${Tensor._seed}`)

if (!args.model) args.model = await model.download()

const device = Number(args.shard) > 1 ? range(Number(args.shard)).map((i) => `${Device.DEFAULT}:${i}` satisfies string) : Device.DEFAULT
await model.load(args.model, undefined, device)

const tokenizer = await Tokenizer.init(`${args.model.split('/').slice(0, -1).join('/')}/tokenizer.model`)

const system = [tokenizer.bos_id, ...tokenizer.encode_message('system', 'You are an helpful assistant.')]
let start_pos = await model.prefill(system, undefined, device)
while (true) {
  const query = prompt('Q: ')!
  const toks = [...tokenizer.encode_message('user', query), ...tokenizer.encode_role('assistant')]

  start_pos = await model.prefill(toks.slice(0, -1), start_pos, device)
  let last_tok = toks.at(-1)
  const st = performance.now()
  let tokens = 0
  while (true) {
    GlobalCounters.reset()
    const tok = await model.call(new Tensor([[last_tok]], { device: device }), start_pos)
    tokens++
    start_pos += 1
    last_tok = tok
    if (tokenizer.stop_tokens.includes(tok)) break
    env.writeStdout(string_to_bytes(tokenizer.decode([tok])))
  }
  env.writeStdout(string_to_bytes(` (${(tokens / ((performance.now() - st) / 1000)).toFixed(2)} tokens/s)\n`))
}
