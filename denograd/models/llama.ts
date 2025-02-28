import { Device } from '../device.ts'
import { dtypes } from '../dtype.ts'
import { TinyJit } from '../engine/jit.ts'
import { add, assert, get_env, get_number_env, idiv, is_eq, range } from '../helpers.ts'
import { Embedding, Linear, RMSNorm } from '../nn/index.ts'
import { UOp, type Variable } from '../ops.ts'
import { Tensor } from '../tensor.ts'

// https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
export const precompute_freqs_cis = (dim: number, end: number, theta = 10000.0): Tensor => {
  let freqs = Tensor.arange(0, dim, 2).get({ stop: idiv(dim, 2) }).div(dim).pow(theta, true).div(1.0, true)
  freqs = Tensor.arange(end).unsqueeze(1).mul(freqs.unsqueeze(0))
  return Tensor.stack([freqs.cos(), freqs.sin()], -1).reshape([1, end, 1, idiv(dim, 2), 2])
}

// (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
export const complex_mult = (A: Tensor, c: Tensor, d: Tensor) => {
  const a = A.get('...', { start: 0, stop: 1 }), b = A.get('...', { start: 1, stop: 2 })
  const ro = a.mul(c).sub(b.mul(d))
  const co = a.mul(d).add(b.mul(c))
  return ro.cat([co], -1)
}
export const apply_rotary_emb = (xq: Tensor, xk: Tensor, freqs_cis: Tensor): [Tensor, Tensor] => {
  if (freqs_cis.shape[1] !== xq.shape[1] || xq.shape[1] !== xk.shape[1]) throw new Error(`freqs_cis shape mismatch ${freqs_cis.shape} xq:${xq.shape} xk:${xk.shape}`)
  xq = xq.reshape([...xq.shape.slice(0, -1), -1, 2])
  xk = xk.reshape([...xk.shape.slice(0, -1), -1, 2])
  assert(xq.shape.length === xk.shape.length && xk.shape.length === freqs_cis.shape.length && freqs_cis.shape.length === 5)
  const c = freqs_cis.get('...', { start: 0, stop: 1 }), d = freqs_cis.get('...', { start: 1, stop: 2 })
  const xq_out = complex_mult(xq, c, d)
  const xk_out = complex_mult(xk, c, d)
  return [xq_out.flatten(3), xk_out.flatten(3)]
}

export const repeat_kv = (x: Tensor, n_rep: number): Tensor => {
  const [bs, seqlen, n_kv_heads, head_dim] = x.shape
  if (n_rep === 1) return x
  // NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  return x.repeat([1, 1, 1, n_rep]).reshape([bs, seqlen, n_kv_heads * n_rep, head_dim])
}
export class Attention {
  n_kv_heads: number
  head_dim: number
  n_rep: number
  wq: Linear
  wk: Linear
  wv: Linear
  wo: Linear

  wqkv?: Tensor
  cache_kv?: Tensor
  constructor(dim: number, public n_heads: number, n_kv_heads: number | undefined, public max_context: number, linear = Linear) {
    this.n_kv_heads = n_kv_heads !== undefined ? n_kv_heads : n_heads // n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    this.head_dim = idiv(dim, n_heads)
    this.n_rep = idiv(this.n_heads, this.n_kv_heads)

    this.wq = new linear(dim, this.n_heads * this.head_dim, false)
    this.wk = new linear(dim, this.n_kv_heads * this.head_dim, false)
    this.wv = new linear(dim, this.n_kv_heads * this.head_dim, false)
    this.wo = new linear(this.n_heads * this.head_dim, dim, false)
  }
  call = async (x: Tensor, start_pos: number | Variable, freqs_cis: Tensor, mask?: Tensor): Promise<Tensor> => {
    let xq, xk, xv
    if (get_env('WQKV')) {
      if (this.wqkv === undefined) this.wqkv = Tensor.cat([this.wq.weight, this.wk.weight, this.wv.weight])
      const xqkv = x.matmul(this.wqkv.T)
      ;[xq, xk, xv] = xqkv.split([this.wq.weight.shape[0], this.wk.weight.shape[0], this.wv.weight.shape[0]], 2)
    } else {
      xq = this.wq.call(x), xk = this.wk.call(x), xv = this.wv.call(x)
    }

    xq = xq.reshape([xq.shape[0], xq.shape[1] as number, this.n_heads, this.head_dim])
    xk = xk.reshape([xk.shape[0], xk.shape[1], this.n_kv_heads, this.head_dim])
    xv = xv.reshape([xv.shape[0], xv.shape[1], this.n_kv_heads, this.head_dim])
    ;[xq, xk] = apply_rotary_emb(xq, xk, freqs_cis)
    const [bsz, seqlen] = xq.shape

    // create kv cache
    if (this.cache_kv === undefined) {
      this.cache_kv = await Tensor.zeros([2, bsz, this.max_context, this.n_kv_heads, this.head_dim], { dtype: x.dtype }).contiguous().realize()
      if (Array.isArray(x.device)) {
        // TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
        await this.cache_kv.shard_(x.device, get_env('SHARD_KVCACHE') ? 3 : undefined).realize()
      }
    }
    // update the cache
    if (xk.dtype !== xv.dtype || xv.dtype !== this.cache_kv.dtype) throw new Error(`${xk.dtype}, ${xv.dtype}, ${this.cache_kv.dtype}`)
    await this.cache_kv.shrink([undefined, undefined, [start_pos, add(start_pos, seqlen)], undefined, undefined]).assign(Tensor.stack([xk, xv])).realize()

    let keys = this.cache_kv.get(0).shrink([undefined, [0, add(start_pos, seqlen)], undefined, undefined])
    let values = this.cache_kv.get(1).shrink([undefined, [0, add(start_pos, seqlen)], undefined, undefined])

    keys = repeat_kv(keys, this.n_rep), values = repeat_kv(values, this.n_rep)
    xq = xq.transpose(1, 2), keys = keys.transpose(1, 2), values = values.transpose(1, 2)
    let attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    attn = attn.reshape([bsz, seqlen, -1])
    return this.wo.call(attn)
  }
}
export class FeedForward {
  w1: Linear
  w2: Linear
  w3: Linear
  constructor(dim: number, hidden_dim: number, linear = Linear) {
    this.w1 = new linear(dim, hidden_dim, false)
    this.w2 = new linear(hidden_dim, dim, false)
    this.w3 = new linear(dim, hidden_dim, false) // the gate in Gated Linear Unit
  }
  call = (x: Tensor): Tensor => {
    return this.w2.call(this.w1.call(x).silu().mul(this.w3.call(x))) // SwiGLU [arxiv/2002.05202, eq (5)]
  }
}
export class TransformerBlock {
  attention: Attention
  feed_forward: FeedForward
  attention_norm: RMSNorm
  ffn_norm: RMSNorm
  constructor(dim: number, hidden_dim: number, n_heads: number, n_kv_heads: number | undefined, norm_eps: number, max_context: number, linear = Linear, feed_forward = FeedForward) {
    this.attention = new Attention(dim, n_heads, n_kv_heads, max_context, linear)
    this.feed_forward = new feed_forward(dim, hidden_dim, linear)
    this.attention_norm = new RMSNorm(dim, norm_eps)
    this.ffn_norm = new RMSNorm(dim, norm_eps)
  }

  call = async (x: Tensor, start_pos: number | Variable, freqs_cis: Tensor, mask?: Tensor) => {
    const h = x.add(await this.attention.call(this.attention_norm.call(x), start_pos, freqs_cis, mask))
    return h.add(this.feed_forward.call(this.ffn_norm.call(h))).contiguous()
  }
}
// standard openai sampling
let alpha_counter: Tensor | undefined = undefined
export const sample = (logits: Tensor, temp: number, k: number, p: number, af: number, ap: number) => {
  if (logits.ndim !== 1) throw new Error('only works on 1d tensors')
  if (p < 0 || p > 1) throw new Error('p must be between 0 and 1')
  if (k < 0 || k > (logits.numel() as number)) throw new Error('k must be between 0 and numel')

  // if temperature is very low just use argmax
  if (temp < 1e-6) return logits.argmax()

  logits = logits.to(Device.DEFAULT)

  // alpha sampling
  if (af || ap) {
    if (alpha_counter === undefined) {
      alpha_counter = logits.zeros_like({ dtype: dtypes.int32 }).contiguous()
    }
    logits = logits.sub(alpha_counter.mul(af).add(alpha_counter.ge(0).mul(ap)))
  }

  // replace NaNs with -inf
  logits = logits.ne(logits).where(-Infinity, logits)

  // softmax
  let t = logits.div(temp).softmax()

  const counter = Tensor.arange(t.numel() as number, undefined, undefined, { device: logits.device }).contiguous()
  const counter2 = Tensor.arange(t.numel() as number - 1, -1, -1, { device: logits.device }).contiguous()
  // top k
  let output, output_indices, output_token
  if (k) {
    output = Tensor.zeros([k], { device: logits.device }).contiguous()
    output_indices = Tensor.zeros([k], { device: logits.device, dtype: dtypes.int32 }).contiguous()
    for (const i of range(k)) {
      const t_max = t.max()
      const t_argmax = t.eq(t_max).mul(counter2).max().sub(t.numel() as number, true).sub(1).cast(dtypes.default_int)
      output = output.add(t_max.unsqueeze(0).pad([i, k - i - 1]))
      output_indices = output_indices.add(t_argmax.unsqueeze(0).pad([[i, k - i - 1]]))
      t = counter.eq(t_argmax).where(0, t)
    }

    // approximate top p
    // because we are already limited to top k elements we can do top p "without sorting"
    const output_cumsum = output.get({ step: -1 }).cumsum().get({ step: -1 }).add(t.sum())
    output = output_cumsum.ge(1 - p).mul(output)
    output_indices = output_cumsum.ge(1 - p).mul(output_indices)

    // sample
    const output_idx = output.multinomial()
    output_token = output_indices.get(output_idx)
  } else {
    output_token = t.multinomial()
  }
  // increase alpha counter
  if (af || ap) alpha_counter = counter.eq(output_token).where(alpha_counter!.add(1), alpha_counter!)
  return output_token
}
export class Transformer {
  layers: TransformerBlock[]
  norm: RMSNorm
  tok_embeddings: Embedding
  output: Linear
  freqs_cis: Tensor
  forward_jit?: TinyJit<[tokens: Tensor, start_pos: number | Variable, temperature: number, top_k: number, top_p: number, alpha_f: number, alpha_p: number], Tensor> = undefined
  constructor(dim: number, hidden_dim: number, n_heads: number, n_layers: number, norm_eps: number, vocab_size: number, linear = Linear, embedding = Embedding, n_kv_heads?: number, rope_theta = 10000, public max_context = 1024, jit = true, feed_forward = FeedForward) {
    this.layers = range(n_layers).map(() => new TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear, feed_forward))
    this.norm = new RMSNorm(dim, norm_eps)
    this.tok_embeddings = new embedding(vocab_size, dim)
    this.output = embedding === Embedding ? new Linear(dim, vocab_size, false) : new linear(dim, vocab_size, false)
    this.freqs_cis = precompute_freqs_cis(idiv(dim, n_heads), this.max_context * 2, rope_theta).contiguous()
    this.forward_jit = jit ? new TinyJit(this.forward) : undefined
  }
  forward = async (tokens: Tensor, start_pos: number | Variable, temperature: number, top_k: number, top_p: number, alpha_f: number, alpha_p: number) => {
    const [_bsz, seqlen] = tokens.shape
    let h = this.tok_embeddings.call(tokens)

    this.freqs_cis = await this.freqs_cis.cast(h.dtype).realize()
    const freqs_cis = this.freqs_cis.shrink([undefined, [start_pos, add(start_pos, seqlen)], undefined, undefined, undefined])

    const mask = seqlen > 1 ? await Tensor.full([1, 1, seqlen, add(start_pos, seqlen)], -Infinity, { dtype: h.dtype, device: h.device }).triu((start_pos as number) + 1).realize() : undefined
    for (const layer of this.layers) h = await layer.call(h, start_pos, freqs_cis, mask)
    const logits = this.output.call(this.norm.call(h)).float().get({}, -1, {})

    return await sample(logits.flatten(), temperature, top_k, top_p, alpha_f, alpha_p).realize()
  }
  call = async (tokens: Tensor, start_pos: number, temperature = 0.0, top_k: number = 0, top_p: number = 0.8, alpha_f: number = 0.0, alpha_p: number = 0.0) => {
    // TODO: better way to handle the first call v.s. the rest?
    if (is_eq(tokens.shape.slice(0, 2), [1, 1]) && this.forward_jit !== undefined && start_pos !== 0) {
      return await this.forward_jit.call(tokens, UOp.variable('start_pos', 1, this.max_context).bind(start_pos), temperature, top_k, top_p, alpha_f, alpha_p)
    }
    return await this.forward(tokens, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)
  }
}
// # *** helpers ***

export const convert_from_huggingface = (weights: Record<string, Tensor>, model: Transformer, n_heads: number, n_kv_heads: number, permute_layers = true) => {
  const permute = (v: Tensor, n_heads: number) => {
    return v.reshape([n_heads, 2, idiv(idiv(v.shape[0], n_heads), 2), v.shape[1]]).transpose(1, 2).reshape(v.shape.slice(0, 2))
  }
  const items = range(model.layers.length)
  const keymap = Object.fromEntries([
    ['model.embed_tokens.weight', 'tok_embeddings.weight'],
    ...items.map((l) => [`model.layers.${l}.input_layernorm.weight`, `layers.${l}.attention_norm.weight`] as const),
    ...items.flatMap((l) => ['q', 'k', 'v', 'o'].map((x) => [`model.layers.${l}.self_attn.${x}_proj.weight`, `layers.${l}.attention.w${x}.weight`] as const)),
    ...items.flatMap((l) => ['q', 'k', 'v', 'o'].map((x) => [`model.layers.${l}.self_attn.{x}_proj.bias`, `layers.{l}.attention.w{x}.bias`] as const)),
    ...items.map((l) => [`model.layers.{l}.post_attention_layernorm.weight`, `layers.{l}.ffn_norm.weight`] as const),
    ...items.flatMap((l) => [['gate', '1'], ['down', '2'], ['up', '3']].map(([x, y]) => [`model.layers.${l}.mlp.${x}_proj.weight`, `layers.${l}.feed_forward.w${y}.weight`] as const)),
    ['model.norm.weight', 'norm.weight'],
    ['lm_head.weight', 'output.weight'],
  ])
  const sd: Record<string, Tensor> = {}
  for (let [k, v] of Object.entries(weights)) {
    if (k.includes('.rotary_emb.')) continue
    v = v.to(Device.DEFAULT)
    if (k.includes('model.layers')) {
      if (k.includes('q_proj') && permute_layers) {
        v = permute(v, n_heads)
      } else if (k.includes('k_proj') && permute_layers) {
        v = permute(v, n_kv_heads)
      }
    }
    sd[keymap[k]] = v
  }
  return sd
}

export const convert_from_gguf = (weights: Record<string, Tensor>, model: Transformer) => {
  const items = range(model.layers.length)
  const keymap = Object.fromEntries([
    ['token_embd.weight', 'tok_embeddings.weight'],
    ...items.map((l) => [`blk.${l}.attn_norm.weight`, `layers.${l}.attention_norm.weight`] as const),
    ...items.flatMap((l) => ['q', 'k', 'v'].map((x) => [`blk.${l}.attn_${x}.weight`, `layers.${l}.attention.w${x}.weight`] as const)),
    ...items.map((l) => [`blk.${l}.attn_output.weight`, `layers.${l}.attention.wo.weight`] as const),
    ...items.map((l) => [`blk.${l}.ffn_norm.weight`, `layers.${l}.ffn_norm.weight`] as const),
    ...items.flatMap((l) => [['gate', '1'], ['down', '2'], ['up', '3']].map(([x, y]) => [`blk.${l}.ffn_${x}.weight`, `layers.${l}.feed_forward.w${y}.weight`] as const)),
    ['output_norm.weight', 'norm.weight'],
    ['rope_freqs.weight', 'rope_freqs.weight'],
  ])
  const sd = Object.fromEntries(Object.entries(weights).map(([k, v]) => [keymap[k], v]))
  sd['output.weight'] = weights['token_embd.weight']
  return sd
}

export const fix_bf16 = (weights: Record<string, Tensor>) => {
  if (get_number_env('SUPPORT_BF16', 1)) {
    // TODO: without casting to float16, 70B llama OOM on tinybox.
    return Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.dtype === dtypes.bfloat16 ? v.cast(dtypes.float32).cast(dtypes.float16) : v]))
  }
  // TODO: check if device supports bf16
  return Object.fromEntries(Object.entries(weights).map(([k, v]) => [k, v.dtype === dtypes.bfloat16 ? v.llvm_bf16_cast(dtypes.half).to(v.device) : v]))
}
