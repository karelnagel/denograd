import { GlobalCounters, perf, range, zip } from '../helpers.ts'
import { Tensor } from '../tensor.ts'
import { get_state_dict, gguf_load, load_state_dict, safe_load } from '../nn/state.ts'
import { dtypes } from '../dtype.ts'
import { convert_from_gguf, convert_from_huggingface, fix_bf16, Transformer } from './llama.ts'
import { Embedding, Linear } from '../nn/index.ts'
import { env, withEnvAsync } from '../env/index.ts'
import { Tqdm, type TqdmOnProgress } from '../tqdm.ts'
import { Device } from '../device.ts'
import { Tokenizer } from './tokenizer.ts'

// **** helper functions ****
const concat_weights = (models: Record<string, Tensor>[], device?: string): Record<string, Tensor> => {
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

// **** quantized linears ****
class Int8Linear {
  weight: Tensor
  scale: Tensor
  constructor(in_features: number, out_features: number, bias = false) {
    if (bias !== false) throw new Error(`Int8Linear bias has to be false`)
    this.weight = Tensor.ones([out_features, in_features], { dtype: dtypes.int8 })
    this.scale = Tensor.ones([out_features], { dtype: dtypes.half })
  }
  call = (x: Tensor) => x.dot(this.weight.cast(this.scale.dtype).T.mul(this.scale))

  static quantize = (tensors: Record<string, Tensor>, device?: string | string[], scale_dtype = dtypes.float16, quantize_embeds = false) => {
    const new_tensors: Record<string, Tensor> = {}
    for (let [name, v] of Object.entries(tensors)) {
      if (name.includes('feed_forward') || name.includes('attention.w') || (quantize_embeds && name.includes('tok_embeddings.weight'))) {
        if (!name.includes('weight')) throw new Error()
        v = v.cast(scale_dtype)
        const scale = v.abs().max(1).div(127.0)
        const int8_weight = (v.T.div(scale)).T.round().cast(dtypes.int8) // without round(), cast truncates -34.9 to -34
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
    if (quantize_embeds) {
      new_tensors['output.weight'] = new_tensors['tok_embeddings.weight']
      new_tensors['output.scale'] = new_tensors['tok_embeddings.scale']
    }
    return new_tensors
  }
}

class Int8Embedding {
  weight: Tensor
  scale: Tensor
  arange?: Tensor
  constructor(public vocab_size: number, public embed_size: number) {
    this.weight = Tensor.ones([vocab_size, embed_size], { dtype: dtypes.int8 })
    this.scale = Tensor.ones([vocab_size], { dtype: dtypes.half })
  }
  call = (idx: Tensor) => {
    if (!this.arange) this.arange = Tensor.arange(this.vocab_size, undefined, undefined, { requires_grad: false, device: this.weight.device }).unsqueeze(-1)
    const big_shp = [...idx.shape, this.vocab_size, this.embed_size]
    idx = idx.reshape([...idx.shape, 1, 1]).expand(big_shp)
    const arange = this.arange.expand(big_shp), vals = this.weight.cast(this.scale.dtype).T.mul(this.scale).T
    return arange.eq(idx).mul(vals).sum(-2, undefined, vals.dtype)
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
    static quantize = (state_dict: Record<string, Tensor>, device: string, scale_dtype = dtypes.float16) => {
      const new_state_dict: Record<string, Tensor> = {}
      for (const [k, v] of Object.entries(state_dict)) {
        if (k.includes('feed_forward') || k.includes('attention.w')) {
          const grouped = v.reshape([-1, block_size])
          const scale = grouped.abs().max(1, true)
          const coded = ((grouped.div(scale)).unsqueeze(-1).sub(CODE.to(v.device))).abs().argmin(-1).cast(dtypes.uint8).flatten()
          new_state_dict[k] = coded.get({ step: 2 }).mul(2 ** 4).add(coded.get({ start: 1, step: 2 }))
          new_state_dict[k.replace('.weight', '.scale')] = scale.cast(scale_dtype)
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

export type Llama3Size = keyof typeof MODEL_PARAMS
export type Llama3Quantize = 'int8' | 'nf4' | 'float16'
export type Llama3Constructor = {
  size: Llama3Size
  quantize?: Llama3Quantize
  device?: string | string[]
  max_context?: number
  top_k?: number
  top_p?: number
  temperature?: number
  alpha_f?: number
  alpha_p?: number
}
export type Llama3Load = {
  system?: string
  onProgress?: TqdmOnProgress
}
export type Llama3StaticLoad = Llama3Constructor & Llama3Load
export type Llama3Message = { role: 'user' | 'assistant'; content: string }
export type Llama3Chat = {
  messages: Llama3Message[]
  onProgress?: TqdmOnProgress
  onToken?: (res: Llama3Response & { token: string }) => void
}
export type Llama3StopReason = 'end_turn'
export type Llama3Usage = {
  input_tokens: number
  output_tokens: number
  time_to_first_token: number
  tokens_per_second: number
}
export type Llama3Response = {
  message: Llama3Message
  stop_reason?: Llama3StopReason
  usage: Llama3Usage
}
export class Llama3 implements Llama3Constructor {
  // args
  size: Llama3Size
  quantize?: Llama3Quantize
  device: string | string[]
  max_context: number
  temperature: number
  top_k: number
  top_p: number
  alpha_f: number
  alpha_p: number

  private model: Transformer
  private tokenizer?: Tokenizer
  private start_pos = 0
  private last_seen_toks: number[] = []

  private quantize_embeds: boolean
  private linear: typeof Linear
  private embedding: typeof Embedding
  constructor(args: Llama3Constructor) {
    this.size = args.size, this.quantize = args.quantize, this.max_context = args.max_context ?? 8192, this.device = args.device ?? Device.DEFAULT
    this.temperature = args.temperature ?? 0.95, this.top_k = args.top_k ?? 0, this.top_p = args.top_p ?? 0, this.alpha_f = args.alpha_f ?? 0, this.alpha_p = args.alpha_p ?? 0
    if (this.quantize === 'int8') this.linear = Int8Linear, this.embedding = Int8Embedding as typeof Embedding, this.quantize_embeds = true
    else if (this.quantize === 'nf4') this.linear = NF4Linear(64), this.embedding = Embedding, this.quantize_embeds = false
    else this.linear = Linear, this.embedding = Embedding, this.quantize_embeds = false
    Tensor.no_grad = true

    const params = MODEL_PARAMS[this.size].args
    this.model = new Transformer(params.dim, params.hidden_dim, params.n_heads, params.n_layers, params.norm_eps, params.vocab_size, this.linear, this.embedding, params.n_kv_heads, params.rope_theta, this.max_context, true)
  }
  _load = async (fn: string, onProgress?: TqdmOnProgress): Promise<Record<string, Tensor>> => {
    if (fn.endsWith('.index.json')) {
      const fp = await env.readTextFile(fn)
      const weight_map: Record<string, string> = JSON.parse(fp)['weight_map']
      const parts = Object.fromEntries(await Promise.all([...new Set(Object.values(weight_map))].map(async (n) => [n, await this._load(`${fn.split('/').slice(0, -1).join('/')}/${n}`)])))
      return Object.fromEntries(Object.entries(weight_map).map(([k, n]) => [k, parts[n][k]]))
    } else if (fn.endsWith('.gguf')) {
      const data = await env.readFile(fn)
      return (await gguf_load(data, onProgress))[1]
    } else if (fn.endsWith('.safetensors')) return await safe_load(fn)
    throw new Error('invalid file')
  }

  load = async ({ onProgress, system }: Llama3Load) => {
    const model_path = await this._download(undefined, onProgress)
    const scale_dtype = dtypes.float16
    // load weights
    let weights: Record<string, Tensor>

    // Deno WEBGPU doesn't support f16 yet, so we load weights in CLANG
    await withEnvAsync({ BEAM: 0, DEVICE: Device.DEFAULT }, async () => {
      weights = await this._load(model_path, onProgress)
      if ('model.embed_tokens.weight' in weights) {
        weights = convert_from_huggingface(weights, this.model, MODEL_PARAMS[this.size].args.n_heads, MODEL_PARAMS[this.size].args.n_kv_heads)
      } else if ('token_embd.weight' in weights) {
        weights = convert_from_gguf(weights, this.model)
      }
      weights = fix_bf16(weights)

      // quantize
      if (this.quantize === 'float16') weights = Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.cast(dtypes.float16).contiguous()]))
      else if (this.quantize !== undefined) {
        weights = (this.linear as typeof Int8Linear).quantize(weights, this.device, scale_dtype, this.quantize_embeds)
        for (const v of new Tqdm(Object.values(weights), { onProgress, label: `Quantizing to ${this.quantize}` })) await v.realize()
      }
      // shard
      if (Array.isArray(this.device)) {
        for (const [k, v] of Object.entries(get_state_dict(this.model))) {
          if (k.includes('scale')) v.shard_(this.device) // from quantized
          else if (k.includes('.attention.')) v.shard_(this.device, -1)
          else if (k.includes('.feed_forward.w1.')) v.shard_(this.device, 0)
          else if (k.includes('.feed_forward.w3.')) v.shard_(this.device, 0)
          else if (k.includes('.feed_forward.')) v.shard_(this.device, -1)
          else if (k.includes('tok_embeddings.weight')) v.shard_(this.device, 0)
          else if (k.includes('output.weight')) v.shard_(this.device, 0)
          else v.shard_(this.device)
        }
      }
    })

    await withEnvAsync({ BEAM: 0 }, async () => {
      // replace weights in model
      await load_state_dict(this.model, weights, false, undefined, true, onProgress)
    })
    this.tokenizer = await Tokenizer.init(`${model_path.split('/').slice(0, -1).join('/')}/tokenizer.model`)
    await this._system(system, onProgress)
    return this
  }
  static load = async ({ onProgress, system, ...args }: Llama3StaticLoad) => {
    const llama = new Llama3(args)
    return await llama.load({ onProgress, system })
  }
  _system = async (msg: string = 'You are an helpful assistant.', onProgress?: TqdmOnProgress) => {
    const system = [this.tokenizer!.bos_id, ...this.tokenizer!.encode_message('system', msg)]
    await this._prefill(system, onProgress)
  }
  chat = async ({ messages, onProgress, onToken }: Llama3Chat): Promise<Llama3Response> => {
    const toks = [
      ...messages.flatMap((x) => this.tokenizer!.encode_message(x.role, x.content)),
      ...this.tokenizer!.encode_role('assistant'),
    ]

    let st = performance.now()
    await this._prefill(toks.slice(0, -1), onProgress)
    const time_to_first_token = perf(st)

    let last_tok = toks.at(-1), message: Llama3Message = { role: 'assistant', content: '' }
    let usage = { input_tokens: toks.length - 1, time_to_first_token, output_tokens: 0, tokens_per_second: 0 }
    st = performance.now()

    while (true) {
      const tok = await this._call(new Tensor([[last_tok]], { device: this.device }))
      this.start_pos += 1
      usage.output_tokens++
      usage.tokens_per_second = usage.output_tokens / perf(st)
      last_tok = tok
      if (this.tokenizer!.stop_tokens.includes(tok)) return { stop_reason: 'end_turn', message, usage }

      const token = this.tokenizer!.decode([tok])
      message.content += token
      if (onToken) onToken({ message, usage, token })
    }
  }
  _prefill = async (toks: number[], onProgress?: TqdmOnProgress) => {
    // we can skip part of the prompt if it is the same as last and start_pos=0
    if (this.start_pos === 0) {
      const i = zip(toks, this.last_seen_toks).every(([a, b]) => a === b) ? Math.min(toks.length, this.last_seen_toks.length) : 0
      this.start_pos += i
      this.last_seen_toks = toks
      toks = toks.slice(i)
    }

    // prefill the model
    for (const tok of new Tqdm(toks, { onProgress, label: `Prefilling ${toks.length} tokens` })) {
      GlobalCounters.reset()
      await this._call(new Tensor([[tok]], { device: this.device }))
      this.start_pos += 1
    }
  }

  _download = async (dir = `weights/llama3-${this.size}`, onProgress?: TqdmOnProgress) => {
    if (this.size === '1B') {
      await env.fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', dir, onProgress)
      return await env.fetchSave('https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf', 'Llama-3.2-1B-Instruct-Q6_K.gguf', dir, onProgress)
    } else if (this.size === '8B') {
      await env.fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', dir, onProgress)
      await env.fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors', 'model-00001-of-00004.safetensors', dir, onProgress)
      await env.fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors', 'model-00002-of-00004.safetensors', dir, onProgress)
      await env.fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors', 'model-00003-of-00004.safetensors', dir, onProgress)
      await env.fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors', 'model-00004-of-00004.safetensors', dir, onProgress)
      return await env.fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json', 'model.safetensors.index.json', dir, onProgress)
    } else if (this.size === '70B') {
      const model = await env.fetchSave('https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json?download=true', 'model.safetensors.index.json', dir, onProgress)
      await env.fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', dir, onProgress)
      for (const i of range(17)) {
        await env.fetchSave(`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-${(i + 1).toString().padStart(5, '0')}-of-000017.safetensors?download=true`, `model-${(i + 1).toString().padStart(5, '0')}-of-000017.safetensors`, dir, onProgress)
      }
      return model
    } else throw new Error(`Invalid model size ${this.size}`)
  }
  _call = async (input: Tensor) => {
    const res = await this.model.call(input, this.start_pos, this.temperature, this.top_k, this.top_p, this.alpha_f, this.alpha_p)
    return await res.item()
  }
}
