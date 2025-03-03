// deno-lint-ignore-file no-invalid-regexp
import { bytes_to_string, GlobalCounters, range, string_to_bytes, zip } from '../helpers.ts'
import { Tensor } from '../tensor.ts'
import { get_state_dict, gguf_load, load_state_dict, safe_load } from '../nn/state.ts'
import { dtypes } from '../dtype.ts'
import { convert_from_gguf, convert_from_huggingface, fix_bf16, Transformer } from './llama.ts'
import { Embedding, Linear } from '../nn/index.ts'
import { env } from '../env/index.ts'
import { Tqdm } from '../tqdm.ts'

export class Tokenizer {
  pat = new RegExp("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+", 'gu')

  special_tokens: Map<string, number>
  decode_map: Map<number, string>

  static init = async (path: string) => {
    const ranks = new Map((await env.readTextFile(path)).split('\n').filter(Boolean).map((x) => x.split(' ')).map(([k, v]) => [atob(k), Number(v)]))
    return new Tokenizer(ranks)
  }
  constructor(public mergeable_ranks: Map<string, number>) {
    const specialTokensList = [
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
      ...range(5, 256 - 5).map((_, i) => `<|reserved_special_token_${i}|>`),
    ]
    this.special_tokens = new Map(specialTokensList.map((token, i) => [token, this.mergeable_ranks.size + i]))
    this.decode_map = new Map(Array.from(this.mergeable_ranks.entries()).map(([token, id]) => [id, token]))
  }

  get bos_id(): number {
    return this.special_tokens.get('<|begin_of_text|>')!
  }

  get stop_tokens() {
    return [this.special_tokens.get('<|end_of_text|>')!, this.special_tokens.get('<|eot_id|>')!]
  }

  decode(toks: number[]): string {
    const byteArrays = toks.filter((t) => t < this.mergeable_ranks.size).map((t) => Uint8Array.from(this.decode_map.get(t)!, (c) => c.charCodeAt(0)))

    let allBytes = new Uint8Array()
    for (const curr of byteArrays) {
      const combined = new Uint8Array(allBytes.length + curr.length)
      combined.set(allBytes)
      combined.set(curr, allBytes.length)
      allBytes = combined
    }

    return bytes_to_string(allBytes)
  }

  encode(text: string, allowSpecial: boolean = false): number[] {
    const pieces = [...text.matchAll(this.pat)].map((match) => match[0])

    const tokens: number[] = []
    for (const piece of pieces) {
      if (allowSpecial && this.special_tokens.has(piece)) tokens.push(this.special_tokens.get(piece)!)
      else tokens.push(...this.bpe_encode(string_to_bytes(piece)))
    }

    return tokens
  }

  private bpe_encode(bytes: Uint8Array): number[] {
    let tokens: string[] = Array.from(bytes, (b) => String.fromCharCode(b))

    while (true) {
      let minRank: number = Infinity
      let mergePair: [string, string] | undefined

      for (let i = 0; i < tokens.length - 1; i++) {
        const pair = tokens[i] + tokens[i + 1]
        const rank = this.mergeable_ranks.get(pair)
        if (rank !== undefined && rank < minRank) {
          minRank = rank
          mergePair = [tokens[i], tokens[i + 1]]
        }
      }

      if (mergePair === undefined) break

      const newTokens: string[] = []
      let i = 0
      while (i < tokens.length) {
        if (
          i < tokens.length - 1 &&
          tokens[i] === mergePair[0] &&
          tokens[i + 1] === mergePair[1]
        ) {
          newTokens.push(tokens[i] + tokens[i + 1])
          i += 2
        } else {
          newTokens.push(tokens[i])
          i += 1
        }
      }
      tokens = newTokens
    }

    return tokens.map((token) => this.mergeable_ranks.get(token)!)
  }
  encode_role = (role: string) => {
    return [this.special_tokens.get('<|start_header_id|>')!, ...this.encode(role), this.special_tokens.get('<|end_header_id|>')!, ...this.encode('\n\n')]
  }
  encode_message = (role: string, content: string) => {
    return [...this.encode_role(role), ...this.encode(content.trim()), this.special_tokens.get('<|eot_id|>')!]
  }
}

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

const fetchSave = async (url: string, path: string, dir: string) => {
  path = `${dir}/${path}`
  await env.mkdir(dir)
  if (await env.stat(path).then((x) => x.isFile).catch(() => false)) {
    console.log(`File ${path} already exists, skipping`)
    return path
  }
  console.log(`Downloading into ${path}`)
  const data = await fetch(url).then((x) => x.arrayBuffer())
  await env.writeFile(path, new Uint8Array(data))
  return path
}

type Size = keyof typeof MODEL_PARAMS
type Quantize = 'int8' | 'nf4' | 'float16'
export class Llama3 {
  TEMPERATURE = 0.95
  TOP_K = 0
  TOP_P = 0.0
  ALPHA_F = 0.0
  ALPHA_P = 0.0
  model: Transformer
  quantize_embeds: boolean
  linear: typeof Linear
  embedding: typeof Embedding
  constructor(public size: Size = '8B', public quantize?: Quantize, public max_context = 8192) {
    if (quantize === 'int8') this.linear = Int8Linear, this.embedding = Int8Embedding as typeof Embedding, this.quantize_embeds = true
    else if (quantize === 'nf4') this.linear = NF4Linear(64), this.embedding = Embedding, this.quantize_embeds = false
    else this.linear = Linear, this.embedding = Embedding, this.quantize_embeds = false

    const params = MODEL_PARAMS[size].args
    this.model = new Transformer(params.dim, params.hidden_dim, params.n_heads, params.n_layers, params.norm_eps, params.vocab_size, this.linear, this.embedding, params.n_kv_heads, params.rope_theta, max_context, true)
  }
  _load = async (fn: string): Promise<Record<string, Tensor>> => {
    if (fn.endsWith('.index.json')) {
      const fp = await env.readTextFile(fn)
      const weight_map: Record<string, string> = JSON.parse(fp)['weight_map']
      const parts = Object.fromEntries(await Promise.all([...new Set(Object.values(weight_map))].map(async (n) => [n, await this._load(`${fn.split('/').slice(0, -1).join('/')}/${n}`)])))
      return Object.fromEntries(Object.entries(weight_map).map(([k, n]) => [k, parts[n][k]]))
    } else if (fn.endsWith('.gguf')) {
      const data = await env.readFile(fn)
      return (await gguf_load(data))[1]
    } else if (fn.endsWith('.safetensors')) return await safe_load(fn)
    throw new Error('invalid file')
  }
  load = async (model_path: string, scale_dtype = dtypes.float16, device?: string | string[]) => {
    // load weights
    let weights: Record<string, Tensor>
    weights = await this._load(model_path)
    if ('model.embed_tokens.weight' in weights) {
      weights = convert_from_huggingface(weights, this.model, MODEL_PARAMS[this.size]['args']['n_heads'], MODEL_PARAMS[this.size]['args']['n_kv_heads'])
    } else if ('token_embd.weight' in weights) {
      weights = convert_from_gguf(weights, this.model)
    }
    weights = fix_bf16(weights)

    // TODO: with Context(BEAM=0):
    // quantize
    if (this.quantize === 'float16') weights = Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.cast(dtypes.float16).contiguous()]))
    else if (this.quantize !== undefined) {
      weights = (this.linear as typeof Int8Linear).quantize(weights, device, scale_dtype, this.quantize_embeds)
      for (const v of Object.values(weights)) await v.realize()
    }
    //     # shard
    if (Array.isArray(device)) {
      for (const [k, v] of Object.entries(get_state_dict(this.model))) {
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
    await load_state_dict(this.model, weights, false, undefined, true)
    console.log('loaded')
    return this
  }
  last_seen_toks: number[] = []
  prefill = async (toks: number[], start_pos = 0, device: string | string[]) => {
    // we can skip part of the prompt if it is the same as last and start_pos=0
    if (start_pos === 0) {
      const i = zip(toks, this.last_seen_toks).every(([a, b]) => a === b) ? Math.min(toks.length, this.last_seen_toks.length) : 0
      start_pos += i
      this.last_seen_toks = toks
      toks = toks.slice(i)
    }

    // prefill the model
    for await (const tok of new Tqdm(toks)) {
      GlobalCounters.reset()
      await (await this.model.call(new Tensor([[tok]], { device: device }), start_pos, this.TEMPERATURE, this.TOP_K, this.TOP_P, this.ALPHA_F, this.ALPHA_P)).realize()
      start_pos += 1
    }
    return start_pos
  }

  download = async (dir = `weights/llama3-${this.size}`) => {
    if (this.size === '1B') {
      await fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', dir)
      return await fetchSave('https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf', 'Llama-3.2-1B-Instruct-Q6_K.gguf', dir)
    } else if (this.size === '8B') {
      await fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', dir)
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00001-of-00004.safetensors', 'model-00001-of-00004.safetensors', dir)
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00002-of-00004.safetensors', 'model-00002-of-00004.safetensors', dir)
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00003-of-00004.safetensors', 'model-00003-of-00004.safetensors', dir)
      await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/resolve/main/model-00004-of-00004.safetensors', 'model-00004-of-00004.safetensors', dir)
      return await fetchSave('https://huggingface.co/TriAiExperiments/SFR-Iterative-DPO-LLaMA-3-8B-R/raw/main/model.safetensors.index.json', 'model.safetensors.index.json', dir)
    } else if (this.size === '70B') {
      const model = await fetchSave('https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model.safetensors.index.json?download=true', 'model.safetensors.index.json', dir)
      await fetchSave('https://huggingface.co/bofenghuang/Meta-Llama-3-8B/resolve/main/original/tokenizer.model', 'tokenizer.model', dir)
      for (const i of range(17)) {
        await fetchSave(`https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/resolve/main/model-${(i + 1).toString().padStart(5, '0')}-of-000017.safetensors?download=true`, `model-${(i + 1).toString().padStart(5, '0')}-of-000017.safetensors`, dir)
      }
      return model
    } else throw new Error(`Invalid model size ${this.size}`)
  }
  call = async (input: Tensor, start_pos: number) => {
    return await this.model.call(input, start_pos, this.TEMPERATURE, this.TOP_K, this.TOP_P, this.ALPHA_F, this.ALPHA_P).then((x) => x.item())
  }
}
