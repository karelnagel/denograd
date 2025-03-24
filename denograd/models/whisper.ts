import { add, ArrayMap, Conv1d, type Conv2d, dtypes, Embedding, env, get_key, idiv, type Layer, LayerNorm, Linear, load_state_dict, range, replace_state_dict, safe_load, type sint, sub, Tensor, TinyJit, Tokenizer, UOp, type Variable, zip } from '../mod.ts'
import wav from 'node-wav'

export class MultiHeadAttention {
  query: Linear
  key: Linear
  value: Linear
  out: Linear
  cache_v?: Tensor
  cache_k?: Tensor
  constructor(n_state: number, public n_head: number, public kv_caching?: 'cross' | 'self', public max_self_attn_cache_len?: number) {
    this.query = new Linear(n_state, n_state)
    this.key = new Linear(n_state, n_state, false)
    this.value = new Linear(n_state, n_state)
    this.out = new Linear(n_state, n_state)
  }
  call = async (x: Tensor, xa?: Tensor, mask?: Tensor, len?: Variable | number) => {
    let k: Tensor, v: Tensor
    if (this.kv_caching === 'cross') {
      if (xa !== undefined) {
        k = this.key.call(xa), v = this.value.call(xa)
        if (!this.cache_v) {
          this.cache_k = k, this.cache_v = v
        } else {
          this.cache_k!.assign(k).realize()
          this.cache_v.assign(v).realize()
        }
      } else {
        k = this.cache_k!, v = this.cache_v!
      }
    } else {
      k = this.key.call(x), v = this.value.call(x)
      if (this.kv_caching === 'self') {
        if (len === undefined || this.max_self_attn_cache_len === undefined) throw new Error()
        if (!this.cache_k) {
          this.cache_k = Tensor.zeros([x.shape[0], this.max_self_attn_cache_len, x.shape[2]])
          this.cache_v = Tensor.zeros([x.shape[0], this.max_self_attn_cache_len, x.shape[2]])
        }
        k = this.cache_k.shrink([undefined, [0, len], undefined]).cat([k], 1)
        v = this.cache_v!.shrink([undefined, [0, len], undefined]).cat([v], 1)
        const padding = sub(sub(this.max_self_attn_cache_len, len), x.shape[1])
        await this.cache_k.assign(k.pad([undefined, [0, padding], undefined]).contiguous()).realize()
        await this.cache_v!.assign(v.pad([undefined, [0, padding], undefined]).contiguous()).realize()
      }
    }
    let q = this.query.call(x)
    const n_ctx = q.shape_num[1]
    if (q.shape.at(-1) !== k.shape.at(-1) || k.shape.at(-1) !== v.shape.at(-1)) throw new Error()
    const head_dim = idiv(q.shape.at(-1)!, this.n_head)
    q = q.reshape([...q.shape.slice(0, 2), this.n_head, head_dim]).permute(0, 2, 1, 3)
    k = k.reshape([...k.shape.slice(0, 2), this.n_head, head_dim]).permute(0, 2, 1, 3)
    v = v.reshape([...v.shape.slice(0, 2), this.n_head, head_dim]).permute(0, 2, 1, 3)
    const attn = q.scaled_dot_product_attention(k, v, mask !== undefined ? mask.get({ stop: n_ctx }, { stop: n_ctx }) : undefined)
    const wv = attn.permute(0, 2, 1, 3).flatten(2)
    return this.out.call(wv)
  }
}
export class ResidualAttentionBlock {
  attn: MultiHeadAttention
  attn_ln: LayerNorm
  cross_attn?: MultiHeadAttention
  cross_attn_ln?: LayerNorm
  mlp: Layer[]
  mlp_ln: LayerNorm
  constructor(n_state: number, n_head: number, is_decoder_block = false, max_self_attn_cache_len?: number) {
    this.attn = new MultiHeadAttention(n_state, n_head, is_decoder_block ? 'self' : undefined, max_self_attn_cache_len)
    this.attn_ln = new LayerNorm(n_state)

    this.cross_attn = is_decoder_block ? new MultiHeadAttention(n_state, n_head, 'cross') : undefined
    this.cross_attn_ln = is_decoder_block ? new LayerNorm(n_state) : undefined

    this.mlp = [new Linear(n_state, n_state * 4), Tensor.gelu, new Linear(n_state * 4, n_state)]
    this.mlp_ln = new LayerNorm(n_state)
  }
  call = async (x: Tensor, xa?: Tensor, mask?: Tensor, len?: Variable | number) => {
    x = x.add(await this.attn.call(this.attn_ln.call(x), undefined, mask, len))
    if (this.cross_attn) x = x.add(await this.cross_attn.call(this.cross_attn_ln!.call(x), xa))
    x = x.add(this.mlp_ln.call(x).sequential(this.mlp))
    return await x.realize()
  }
}
export class AudioEncoder {
  conv1: Conv2d
  conv2: Conv2d
  blocks: ResidualAttentionBlock[]
  ln_post: LayerNorm
  positional_embedding: Tensor
  encode: TinyJit<[x: Tensor], Tensor>
  constructor({ n_mels, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer }: Dims) {
    this.conv1 = Conv1d(n_mels, n_audio_state, 3, undefined, 1)
    this.conv2 = Conv1d(n_audio_state, n_audio_state, 3, 2, 1)
    this.blocks = range(n_audio_layer).map((x) => new ResidualAttentionBlock(n_audio_state, n_audio_head))
    this.ln_post = new LayerNorm(n_audio_state)
    this.positional_embedding = Tensor.empty([n_audio_ctx, n_audio_state])
    this.encode = new TinyJit(this.call)
  }
  call = async (x: Tensor) => {
    x = this.conv1.call(x).gelu()
    x = this.conv2.call(x).gelu()
    x = x.permute(0, 2, 1)
    x = x.add(this.positional_embedding.get({ stop: x.shape_num[1] }))
    x = await x.sequentialAsync(this.blocks)
    x = this.ln_post.call(x)
    return await x.realize()
  }
}
class TextDecoder {
  max_tokens_to_sample!: number
  max_self_attn_cache_len!: number
  token_embedding!: Embedding
  positional_embedding!: Tensor
  blocks!: ResidualAttentionBlock[]
  ln!: LayerNorm
  mask!: Tensor
  getjitted = new ArrayMap<number[], TinyJit<[x: Tensor, pos: number | UOp, encoded_audio: Tensor], Tensor>>()
  static init = async ({ n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer }: Dims) => {
    const ret = new TextDecoder()
    ret.max_tokens_to_sample = idiv(n_text_ctx, 2)
    ret.max_self_attn_cache_len = ret.max_tokens_to_sample * 2 + 5 // roughly prompt + start toks + max_tokens_to_sample

    ret.token_embedding = new Embedding(n_vocab, n_text_state)
    ret.positional_embedding = Tensor.empty([n_text_ctx, n_text_state])
    ret.blocks = range(n_text_layer).map((x) => new ResidualAttentionBlock(n_text_state, n_text_head, true, ret.max_self_attn_cache_len))
    ret.ln = new LayerNorm(n_text_state)
    ret.mask = await Tensor.full([n_text_ctx, n_text_ctx], -Infinity).triu(1).realize()
    return ret
  }
  call = async (x: Tensor, pos: number, encoded_audio: Tensor) => {
    let jit = this.getjitted.get(x.shape_num)
    if (!jit) {
      jit = new TinyJit(this.forward)
      this.getjitted.set(x.shape_num, jit)
    }
    return jit.call(x, pos ? UOp.variable('self_attn_cache_len', 1, this.max_self_attn_cache_len).bind(pos) : 0, encoded_audio)
  }
  forward = async (x: Tensor, pos: Variable | number, encoded_audio: Tensor) => {
    const seqlen = x.shape.at(-1)!
    x = this.token_embedding.call(x).add(this.positional_embedding.shrink([[pos, add(pos, seqlen)], undefined, undefined]))
    for (const block of this.blocks) x = await block.call(x, encoded_audio, this.mask, pos)
    return await this.output_tok(x)
  }
  output_tok = async (x: Tensor) => await this.ln.call(x).matmul(this.token_embedding.weight.T).realize()
}

class Whisper {
  encoder!: AudioEncoder
  decoder!: TextDecoder
  is_multilingual!: boolean
  batch_size!: number
  static init = async (dims: Dims, batch_size = 1) => {
    const res = new Whisper()
    res.encoder = new AudioEncoder(dims)
    res.decoder = await TextDecoder.init(dims)
    res.is_multilingual = dims.n_vocab === 51865
    res.batch_size = batch_size
    return res
  }
}
const RATE = 16000
const SEGMENT_SECONDS = 30
const SAMPLES_PER_SEGMENT = RATE * SEGMENT_SECONDS // 480000
const N_FFT = 400
const HOP_LENGTH = 160
const N_MELS = 80
const FRAMES_PER_SEGMENT = SAMPLES_PER_SEGMENT / HOP_LENGTH // 3000

const padWaveforms = (waveforms: Float32Array[], batchSize: number) => {
  const currRows = waveforms.length
  if (currRows >= batchSize) return waveforms

  const rowsToPad = batchSize - currRows
  const colLength = waveforms[0].length
  const zeroRow = new Float32Array(colLength)

  const padding = Array(rowsToPad).fill(0).map(() => new Float32Array(zeroRow))
  return waveforms.concat(padding)
}
/**
 * param waveforms: A list of possibly variable length 16000Hz audio samples
 * param batch_size: The batch_size associated with the Whisper model being used to transcribe the audio. Used to prevent JIT mismatch errors since the encoder does not accept symbolic shapes
 * param truncate: If true, truncates (or pads) audio to exactly 30s for a single encoder pass
 * return: mel spectrogram of the given waveforms
 */
export const prep_audio = async (waveforms: Float32Array[], batch_size: number, truncate = false) => {
  const data = await env.readFile('log_spec.bin')
  return await new Tensor(data).bitcast(dtypes.float32).reshape([1, 80, 3000]).realize()

  //   const pad_or_trim = (arr: Float32Array, target_len: number): Float32Array => {
  //     const curr_len = arr.length
  //     if (curr_len === target_len) return arr
  //     else if (curr_len < target_len) {
  //       const res = new Float32Array(target_len)
  //       res.set(arr)
  //       return res
  //     } else return arr.slice(0, target_len)
  //   }

  //   let max_len = truncate ? SAMPLES_PER_SEGMENT : Math.max(...waveforms.map((wav) => wav.length))
  //   const r = mod(max_len, SAMPLES_PER_SEGMENT)
  //   if (r > 0) max_len += SAMPLES_PER_SEGMENT - r
  //   waveforms = waveforms.map((w) => pad_or_trim(w, max_len))
  //   if (waveforms[0].length > batch_size) throw new Error()
  //   if (waveforms[0].length < batch_size) {
  //     // we could have a symbolic batch_size dim instead of manually padding here if conv/layernorm supported symbolic shapes
  //     waveforms = padWaveforms(waveforms, batch_size)
  //   }
  // stft = librosa.stft(waveforms, n_fft=N_FFT, hop_length=HOP_LENGTH, window='hann', dtype=np.csingle)
  // magnitudes = np.absolute(stft[..., :-1]) ** 2
  // mel_spec = librosa.filters.mel(sr=RATE, n_fft=N_FFT, n_mels=N_MELS) @ magnitudes

  //   log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
  //   log_spec = np.maximum(log_spec, log_spec.max((1,2), keepdims=True) - 8.0)
  //   log_spec = (log_spec + 4.0) / 4.0

  //   return log_spec
}

// deno-fmt-ignore
const LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish", "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese", "pt": "portuguese", "tr": "turkish",
    "pl": "polish", "ca": "catalan", "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian", "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay", "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian", "ta": "tamil", "no": "norwegian",
    "th": "thai", "ur": "urdu", "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin", "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak", "te": "telugu",
    "fa": "persian", "lv": "latvian", "bn": "bengali", "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada", "et": "estonian", "mk": "macedonian",
    "br": "breton", "eu": "basque", "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian", "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala", "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali", "af": "afrikaans", "oc": "occitan", "ka": "georgian",
    "be": "belarusian", "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic", "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese", "ht": "haitian creole",
    "ps": "pashto", "tk": "turkmen", "nn": "nynorsk", "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy",
    "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese",
}

const get_encoding = async (encoding_name: string) => {
  const url = `https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/${encoding_name}.tiktoken`
  const path = await env.fetchSave(url, get_key(url), env.CACHE_DIR)
  const data = await env.readTextFile(path)
  const ranks = data.split('\n').filter(Boolean).map((line) => line.split(' ')).map(([token, rank]) => [atob(token), Number(rank)])
  let n_vocab = ranks.length
  const specials = [
    '<|endoftext|>',
    '<|startoftranscript|>',
    ...Object.keys(LANGUAGES).map((lang) => `<|${lang}|>`),
    '<|translate|>',
    '<|transcribe|>',
    '<|startoflm|>',
    '<|startofprev|>',
    '<|nospeech|>',
    '<|notimestamps|>',
    ...range(1501).map((i) => `<|${(i * 0.02).toFixed(2)}|>`),
  ]
  n_vocab += specials.length
  const pat = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu
  return new Tokenizer(pat, Object.fromEntries(ranks), Object.fromEntries(specials.map((x, i) => [x, ranks.length + i])))
}

const MODEL_URLS = {
  'tiny.en': 'https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors?download=true',
  'tiny': 'https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors?download=true',
  'base.en': 'https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt',
  'base': 'https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt',
  'small.en': 'https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt',
  'small': 'https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt',
  'medium.en': 'https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt',
  'medium': 'https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt',
  'large-v1': 'https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt',
  'large-v2': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt',
  'large': 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt',
}
const DIMS = {
  'n_mels': 80,
  'n_vocab': 51864,
  'n_audio_ctx': 1500,
  'n_audio_state': 384,
  'n_audio_head': 6,
  'n_audio_layer': 4,
  'n_text_ctx': 448,
  'n_text_state': 384,
  'n_text_head': 6,
  'n_text_layer': 4,
}
type Dims = typeof DIMS
type Model = keyof typeof MODEL_URLS

const state_map = {
  'model.encoder.conv1': 'encoder.conv1',
  'model.encoder.conv2': 'encoder.conv2',
  'model.encoder.embed_positions.weight': 'encoder.positional_embedding',
  'model.encoder.layer_norm': 'encoder.ln_post',
  'model.encoder.layers.([0-3]).self_attn.q_proj': 'encoder.blocks.$1.attn.query',
  'model.encoder.layers.([0-3]).self_attn.k_proj': 'encoder.blocks.$1.attn.key',
  'model.encoder.layers.([0-3]).self_attn.v_proj': 'encoder.blocks.$1.attn.value',
  'model.encoder.layers.([0-3]).self_attn.out_proj': 'encoder.blocks.$1.attn.out',
  'model.encoder.layers.([0-3]).self_attn_layer_norm': 'encoder.blocks.$1.attn_ln',
  'model.encoder.layers.([0-3]).fc1': 'encoder.blocks.$1.mlp.0',
  'model.encoder.layers.([0-3]).fc2': 'encoder.blocks.$1.mlp.2',
  'model.encoder.layers.([0-3]).final_layer_norm': 'encoder.blocks.$1.mlp_ln',
  'model.decoder.embed_tokens.weight': 'decoder.token_embedding.weight',
  'model.decoder.embed_positions.weight': 'decoder.positional_embedding',
  'model.decoder.layer_norm': 'decoder.ln',
  'model.decoder.layers.([0-3]).self_attn.q_proj': 'decoder.blocks.$1.attn.query',
  'model.decoder.layers.([0-3]).self_attn.k_proj': 'decoder.blocks.$1.attn.key',
  'model.decoder.layers.([0-3]).self_attn.v_proj': 'decoder.blocks.$1.attn.value',
  'model.decoder.layers.([0-3]).self_attn.out_proj': 'decoder.blocks.$1.attn.out',
  'model.decoder.layers.([0-3]).self_attn_layer_norm': 'decoder.blocks.$1.attn_ln',
  'model.decoder.layers.([0-3]).encoder_attn.q_proj': 'decoder.blocks.$1.cross_attn.query',
  'model.decoder.layers.([0-3]).encoder_attn.k_proj': 'decoder.blocks.$1.cross_attn.key',
  'model.decoder.layers.([0-3]).encoder_attn.v_proj': 'decoder.blocks.$1.cross_attn.value',
  'model.decoder.layers.([0-3]).encoder_attn.out_proj': 'decoder.blocks.$1.cross_attn.out',
  'model.decoder.layers.([0-3]).encoder_attn_layer_norm': 'decoder.blocks.$1.cross_attn_ln',
  'model.decoder.layers.([0-3]).fc1': 'decoder.blocks.$1.mlp.0',
  'model.decoder.layers.([0-3]).fc2': 'decoder.blocks.$1.mlp.2',
  'model.decoder.layers.([0-3]).final_layer_norm': 'decoder.blocks.$1.mlp_ln',
}

export const init_whisper = async (model_name: Model, batch_size = 1): Promise<[Whisper, Tokenizer]> => {
  if (!MODEL_URLS[model_name]) throw new Error()

  const filename = await env.fetchSave(MODEL_URLS[model_name], model_name, env.CACHE_DIR)
  let state = await safe_load(filename)
  state = replace_state_dict(state, state_map)
  const model = await Whisper.init(DIMS, batch_size)
  await load_state_dict(model, state, false)
  const enc = await get_encoding(model.is_multilingual ? 'multilingual' : 'gpt2')
  return [model, enc]
}
export const load_file_waveform = async (filename: string): Promise<Float32Array[]> => {
  const data = await env.readFile(filename)
  const res = wav.decode(data)
  if (res.sampleRate !== RATE) throw new Error()
  return res.channelData
}

const transcribe_file = async (model: any, enc: Tokenizer, filename: string) => {
  if (filename.startsWith('http')) filename = await env.fetchSave(filename, get_key(filename), env.CACHE_DIR)
  const waveforms = await load_file_waveform(filename)
  return await transcribe_waveform(model, enc, waveforms)
}
/**
 * Expects an array of shape (N,S) where N is the number waveforms to transcribe in parallel and S is number of 16000Hz samples
 * Returns the transcribed text if a single waveform is provided, or an array of transcriptions if multiple are provided
 */
const transcribe_waveform = async (model: Whisper, enc: Tokenizer, waveforms: Float32Array[], truncate = false) => {
  const log_spec = await prep_audio(waveforms, model.batch_size, truncate)
  const nsample = model.decoder.max_tokens_to_sample

  const inferloop = async (ctx: Tensor, encoded_audio: Tensor) => {
    let pos = 0, next_tokens = ctx
    for (const i of range((nsample - start_tokens.length) * 2)) {
      next_tokens = (await model.decoder.call(next_tokens, pos, encoded_audio)).get({}, -1).argmax(-1).cast(dtypes.int32).reshape([-1, 1])
      next_tokens = (ctx.get({}, -1).eq(eot)).reshape([-1, 1]).where(next_tokens.full_like(eot), next_tokens)
      ctx = Tensor.cat([ctx, next_tokens], 1)
      pos = ctx.shape_num.at(-1)! - 1
      if (await (next_tokens.eq(eot)).all().item()) break
    }
    return ctx
  }

  const gettexttoks = (line: number[]) => line.filter((tok) => tok < eot || tok > enc.special_tokens['<|notimestamps|>']).slice(-nsample + start_tokens.length)
  let start_tokens = [enc.special_tokens['<|startoftranscript|>']]
  if (model.is_multilingual) {
    // TODO detect language
    const language_token = enc.special_tokens['<|startoftranscript|>'] + 1 + Object.keys(LANGUAGES).indexOf('en')
    start_tokens.push(language_token)
    start_tokens.push(enc.special_tokens['<|transcribe|>'])
  }
  start_tokens.push(enc.special_tokens['<|notimestamps|>'])
  const eot = enc.special_tokens['<|endoftext|>']

  let ctx = new Tensor(start_tokens).reshape([1, -1]).expand([model.batch_size, start_tokens.length])
  let transcriptions: number[][] = waveforms.map(() => [])

  for (const curr_frame of range(0, log_spec.shape_num.at(-1), FRAMES_PER_SEGMENT)) {
    const encoded_audio = await model.encoder.encode.call(await log_spec.get({}, {}, { start: curr_frame, stop: curr_frame + FRAMES_PER_SEGMENT }).realize())

    if ((await ctx.tolist<number[][]>()).every((c) => c.length === ctx.get(0).length)) ctx = await inferloop(ctx, encoded_audio)
    else {
      throw new Error()
      // ctx = await Promise.all((await ctx.tolist()).map(async (c, i) => (await inferloop(new Tensor(range(model.batch_size).map(() => c)), encoded_audio)).get(i)))
    }

    for (const [i, [res, arr]] of zip(transcriptions, await ctx.tolist<number[][]>()).entries()) {
      if (curr_frame * HOP_LENGTH <= waveforms[i].length) {
        const start = arr.indexOf(start_tokens.at(-1)!) + 1
        res.push(...arr.slice(start, arr.indexOf(eot, start)))
      }
    }
    ctx = new Tensor((await ctx.tolist<number[][]>()).map((cs) => [enc.special_tokens['<|startofprev|>'], ...gettexttoks(cs), ...start_tokens]))
  }
  const out = transcriptions.map((tokens) => enc.decode(tokens).trim())
  return out.length > 1 ? out : out[0]
}

if (import.meta.main) {
  Tensor.manual_seed(3)
  const [model, enc] = await init_whisper('tiny.en', 1)
  console.log(await transcribe_file(model, enc, env.args()[0]))
}
