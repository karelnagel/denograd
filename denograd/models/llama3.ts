// deno-lint-ignore-file no-invalid-regexp
import { encoding_for_model, get_encoding, type Tiktoken } from 'npm:tiktoken'
import { GlobalCounters, range, sum, zip } from '../helpers.ts'
import { Device, type DeviceType } from '../device.ts'
import { Tensor } from '../tensor.ts'
import { get_parameters, get_state_dict, gguf_load, load_state_dict, safe_load } from '../nn/state.ts'
import { dtypes } from '../dtype.ts'
import { convert_from_gguf, convert_from_huggingface, fix_bf16, Transformer } from './llama.ts'
import { Linear } from '../nn/index.ts'
import { parseArgs } from 'jsr:@std/cli'

class Tokenizer {
  pat_str = new RegExp("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")
  num_base_tokens: number
  special_tokens: Record<string, number>
  model: Tiktoken
  constructor(model_path: string) {
    // should be model_path instead of gpt-4
    const mergeable_ranks = encoding_for_model('gpt-4').token_byte_values()
    this.num_base_tokens = mergeable_ranks.length
    const special_tokens = [
      '<|begin_of_text|>',
      '<|end_of_text|>',
      '<|reserved_special_token_0|>',
      '<|reserved_special_token_1|>',
      '<|reserved_special_token_2|>',
      '<|reserved_special_token_3|>',
      '<|start_header_id|>',
      '<|end_header_id|>',
      '<|reserved_special_token_4|>',
      '<|eot_id|>',
      ...range(5, 256 - 5).map((i) => `<|reserved_special_token_{i}|>`),
    ]
    this.special_tokens = Object.fromEntries(special_tokens.map((token, i) => [token, mergeable_ranks.length + i]))

    this.model = get_encoding('cl100k_base', this.special_tokens)
  }

  get bos_id() {
    return this.special_tokens['<|begin_of_text|>']
  }
  get stop_tokens() {
    return [this.special_tokens['<|end_of_text|>'], this.special_tokens['<|eot_id|>']]
  }

  decode = (toks: number[]) => this.model.decode(new Uint32Array(toks.filter((t) => t < this.num_base_tokens)))
  encode = (text: string, allow_special = false) => [...this.model.encode(text, allow_special ? 'all' : [], [])]
}

// **** helper functions ****
const concat_weights = (models: Record<string, Tensor>[], device?: DeviceType): Record<string, Tensor> => {
  const convert = (name: string): Tensor => {
    const disk_tensors: Tensor[] = models.map((model) => model[name])
    if (disk_tensors.length === 1 || disk_tensors[0].shape.length === 1) {
      return disk_tensors[0].to(device)
    }
    const axis = name.endsWith('.attention.wo.weight') || name.endsWith('.feed_forward.w2.weight') ? 1 : 0
    const lazy_tensors = disk_tensors.map((data) => data.to(device))
    return lazy_tensors[0].cat(lazy_tensors.slice(1), axis)
  }
  return Object.fromEntries(models.flatMap((model) => Object.keys(model).map((name) => [name, convert(name)])))
}
const load = async (fn: string): Promise<Record<string, Tensor>> => {
  if (fn.endsWith('.index.json')) {
    const fp = Deno.readTextFileSync(fn)
    const weight_map = JSON.parse(fp)['weight_map']
    const parts = [...new Set(weight_map.values())].map((n) => [n, load(fn + n)])
    return Object.fromEntries(weight_map.entries().map(([k, n]: any) => [k, parts[n][k]]))
  } else if (fn.endsWith('.gguf')) {
    const gguf_tensor = Tensor.empty([Deno.statSync(fn).size], { dtype: dtypes.uint8, device: `DISK:${fn}` }).to(Device.DEFAULT)
    return gguf_load(gguf_tensor)[1]
  } else if (fn.endsWith('.safetensors')) return await safe_load(fn)
  throw new Error('invalid file')
}
// **** quantized linears ****
class Int8Linear {
  weight: Tensor
  scale: Tensor
  constructor(in_features: number, out_features: number, bias = false) {
    if (bias !== false) throw new Error(`Int8Linear bias has to be false`)
    this.weight = Tensor.ones([out_features, in_features], { dtype: dtypes.int8 })
    this.scale = Tensor.ones([out_features], { dtype: dtypes.half })
  }
  call = (x: Tensor) => x.dot(this.weight.cast(dtypes.half).T.mul(this.scale))

  static quantize = (tensors: Record<string, Tensor>, device?: DeviceType | DeviceType[]) => {
    const new_tensors: Record<string, Tensor> = {}
    for (const [name, v] of Object.entries(tensors)) {
      if (name.includes('feed_forward') || name.includes('attention.w')) {
        if (!name.includes('weight')) throw new Error()
        const scale = v.abs().max(1).div(127.0)
        const int8_weight = (v.T.div(scale)).T.cast(dtypes.int8)
        new_tensors[name] = int8_weight
        new_tensors[name.replace('weight', 'scale')] = scale
        if (Array.isArray(device)) {
          new_tensors[name].shard_(device, -1)
          new_tensors[name.replace('weight', 'scale')].shard_(device, undefined)
        }
      } else {
        new_tensors[name] = v
      }
    }
    return new_tensors
  }
}
const NF4Linear = (block_size: number) => {
  const _CODE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]
  const CODE = Tensor.stack(_CODE.map((c) => new Tensor(c, { dtype: dtypes.float16 })))

  class _NF4Linear {
    weight: Tensor
    scale: Tensor
    constructor(public in_features: number, public out_features: number, bias = false) {
      if (bias) throw new Error('bias not supported')
      this.weight = Tensor.empty([Math.trunc(out_features * in_features / 2)], { dtype: dtypes.uint8 })
      this.scale = Tensor.empty([Math.trunc(out_features * in_features / block_size), 1], { dtype: dtypes.float16 })
    }
    call = (x: Tensor): Tensor => {
      const high_bits = this.weight
      const low_bits = (this.weight.mul(2 ** 4)).contiguous()
      const unpacked = Tensor.stack([high_bits, low_bits], -1).idiv(2 ** 4)
      const unscaled = CODE.get(unpacked).to(x.device).reshape([-1, block_size]).mul(this.scale)
      return x.linear(unscaled.reshape([this.out_features, this.in_features]).T)
    }
    static quantize = (state_dict: Record<string, Tensor>, device: DeviceType) => {
      const new_state_dict: Record<string, Tensor> = {}
      for (const [k, v] of Object.entries(state_dict)) {
        if (k.includes('feed_forward') || k.includes('attention.w')) {
          const grouped = v.reshape([-1, block_size])
          const scale = grouped.abs().max(1, true)
          const coded = ((grouped.div(scale)).unsqueeze(-1).sub(CODE.to(v.device))).abs().argmin(-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded.get({ step: 2 }).mul(2 ** 4).add(coded.get({ start: 1, step: 2 }))
          new_state_dict[k.replace('.weight', '.scale')] = scale.cast(dtypes.float16)
          if (Array.isArray(device)) {
            new_state_dict[k].shard_(device, -1)
            new_state_dict[k.replace('weight', 'scale')].shard_(device)
          }
        } else new_state_dict[k] = v
      }
      return new_state_dict
    }
  }
  return _NF4Linear
}
const MODEL_PARAMS = {
  '1B': {
    'args': { 'dim': 2048, 'n_heads': 32, 'n_kv_heads': 8, 'n_layers': 16, 'norm_eps': 1e-5, 'rope_theta': 500000, 'vocab_size': 128256, 'hidden_dim': 8192 },
    'files': 1,
  },
  '8B': {
    'args': { 'dim': 4096, 'n_heads': 32, 'n_kv_heads': 8, 'n_layers': 32, 'norm_eps': 1e-5, 'rope_theta': 500000, 'vocab_size': 128256, 'hidden_dim': 14336 },
    'files': 1,
  },
  '70B': {
    'args': { 'dim': 8192, 'n_heads': 64, 'n_kv_heads': 8, 'n_layers': 80, 'norm_eps': 1e-5, 'rope_theta': 500000, 'vocab_size': 128256, 'hidden_dim': 28672 },
    'files': 8,
  },
}
const build_transformer = async (model_path: string, model_size: keyof typeof MODEL_PARAMS = '8B', quantize?: 'int8' | 'nf4' | 'float16', device?: DeviceType | DeviceType[]) => {
  // build model
  let linear
  if (quantize === 'int8') linear = Int8Linear
  else if (quantize === 'nf4') linear = NF4Linear(64)
  else linear = Linear
  const params = MODEL_PARAMS[model_size].args
  const model = new Transformer(params.dim, params.hidden_dim, params.n_heads, params.n_layers, params.norm_eps, params.vocab_size, linear, undefined, undefined, 8192, true)

  // load weights
  let weights: Record<string, Tensor>
  if (Deno.statSync(model_path).isDirectory) {
    if (Deno.statSync(`${model_path}/model.safetensors.index.json`).isFile) weights = await load(`${model_path}/model.safetensors.index.json`)
    else if (Deno.statSync(`${model_path}/model.safetensors`).isFile) weights = await load(`${model_path}/model.safetensors`)
    else weights = concat_weights(await Promise.all(range(MODEL_PARAMS[model_size].files).map((i) => load(`${model_path}/consolidated.${i.toString().padStart(2, '0')}.pth`))), Array.isArray(device) ? device[0] : device)
  } else weights = await load(model_path)
  if ('model.embed_tokens.weight' in weights) {
    weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]['args']['n_heads'], MODEL_PARAMS[model_size]['args']['n_kv_heads'])
  } else if ('token_embd.weight' in weights) {
    weights = convert_from_gguf(weights, model)
  }
  weights = fix_bf16(weights)

  // TODO: with Context(BEAM=0):
  // quantize
  if (quantize === 'float16') weights = Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.cast(quantize).contiguous()]))
  else if (quantize !== undefined) {
    weights = (linear as typeof Int8Linear).quantize(weights, device)
    for (const v of Object.values(weights)) v.realize()
  }
  //     # shard
  if (Array.isArray(device)) {
    for (const [k, v] of Object.entries(get_state_dict(model))) {
      if (k.includes('scale')) v.shard_(device) // from quantized
      else if (k.includes('.attention.')) v.shard_(device, -1)
      else if (k.includes('.feed_forward.w1.')) v.shard_(device, 0)
      else if (k.includes('.feed_forward.w3.')) v.shard_(device, 0)
      else if (k.includes('.feed_forward.')) v.shard_(device, -1)
      else if (k.includes('tok_embeddings.weight')) v.shard_(device, 0)
      else if (k.includes('output.weight')) v.shard_(device, 0)
      else v.shard_(device)
    }
  }
  // replace weights in model
  await load_state_dict(model, weights, false, undefined, true)
  return model
}
// # default settings
let TEMPERATURE = 0.95
const TOP_K = 0
const TOP_P = 0.0
const ALPHA_F = 0.0
const ALPHA_P = 0.0

let last_seen_toks: number[] = []
const prefill = async (model: Transformer, toks: number[], start_pos = 0, device: DeviceType | DeviceType[]) => {
  // we can skip part of the prompt if it is the same as last and start_pos=0
  if (start_pos === 0) {
    const i = zip(toks, last_seen_toks).every(([a, b]) => a === b) ? Math.min(toks.length, last_seen_toks.length) : 0
    start_pos += i
    last_seen_toks = toks
    toks = toks.slice(i)
  }

  // prefill the model
  for (const tok of toks) {
    GlobalCounters.reset()
    ;(await model.call(new Tensor([[tok]], { device: device }), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)).realize()
    start_pos += 1
  }
  return start_pos
}

if (import.meta.main) {
  Tensor.no_grad = true
  const args = parseArgs(Deno.args, {
    boolean: ['download-model'],
    string: ['model', 'size', 'shard', 'quantize', 'seed', 'temperature'],
    default: { size: '1B', shard: '1', temperature: '0.85' },
  })

  // download_model is the default without a model passed in
  const fetchSave = async (url: string, path: string, dir: string) => await fetch(url).then((x) => x.arrayBuffer()).then((x) => Deno.writeFile(`${dir}/${path}`, new Uint8Array(x))).then((x) => `${dir}/${path}`)
  if (args.download_model || !args.model) {
    if (args.size === '1B') {
      await fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', 'llama3-1b-instruct')
      args.model = await fetchSave('https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf', 'Llama-3.2-1B-Instruct-Q6_K.gguf', 'llama3-1b-instruct')
    } else if (args.size === '8B') {
      await fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', 'llama3-8b-sfr')
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors', 'model-00001-of-00004.safetensors', 'llama3-8b-sfr')
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'llama3-8b-sfr')
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'llama3-8b-sfr')
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors', 'model-00004-of-00004.safetensors', 'llama3-8b-sfr')
      args.model = await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json', 'model.safetensors.index.json', 'llama3-8b-sfr')
    } else if (args.size === '70B') {
      const subdir = 'DeepSeek-R1-Distill-Llama-70B'
      args.model = await fetchSave('https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json?download=true', 'model.safetensors.index.json', subdir)
      await fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', subdir)
      for (const i of range(17)) {
        await fetchSave(`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-${(i + 1).toString().padStart(5, '0')}-of-000017.safetensors?download=true`, `model-${(i + 1).toString().padStart(5, '0')}-of-000017.safetensors`, subdir)
      }
    }
  }
  if (args.model === undefined) throw new Error('please provide --model option')

  if (args.seed !== undefined) Tensor.manual_seed(Number(args.seed))
  console.log(`seed = ${Tensor._seed}`)
  TEMPERATURE = Number(args.temperature)

  const tokenizer = new Tokenizer(`${Deno.statSync(args.model).isDirectory ? args.model : args.model.split('/').slice(0, -1).join('/')}/tokenizer.model`)
  const encode_role = (role: string) => {
    return [tokenizer.special_tokens['<|start_header_id|>'], ...tokenizer.encode(role), tokenizer.special_tokens['<|end_header_id|>'], ...tokenizer.encode('\n\n')]
  }
  const encode_message = (role: string, content: string) => {
    return [...encode_role(role), ...tokenizer.encode(content.trim()), tokenizer.special_tokens['<|eot_id|>']]
  }

  const device = Number(args.shard) > 1 ? range(Number(args.shard)).map((i) => `${Device.DEFAULT}:${i}` as DeviceType) : Device.DEFAULT
  const model = await build_transformer(args.model, args.size as any, args.quantize as any, device)
  const param_bytes = sum(get_parameters(model).map((x) => x.lazydata.size * x.dtype.itemsize))

  const system = [tokenizer.bos_id, ...encode_message('system', 'You are an helpful assistant.')]

  let start_pos = await prefill(model, system, undefined, device)
  while (true) {
    const toks = [...encode_message('user', prompt('Q: ')!), ...encode_role('assistant')]

    start_pos = await prefill(model, toks.slice(0, -1), start_pos, device)
    let last_tok = toks.at(-1)
    while (true) {
      GlobalCounters.reset()
      if (args.timing || args.profile) console.log('')
      const st = GlobalCounters.time_sum_s

      const out = await model.call(new Tensor([[last_tok]], { device: device }), start_pos, TEMPERATURE, TOP_K, TOP_P, ALPHA_F, ALPHA_P)
      const tok = await out.item()
      start_pos += 1
      last_tok = tok
      if (tokenizer.stop_tokens.includes(tok)) break
      console.log(tokenizer.decode([tok]))
      console.log()
    }
  }
}
