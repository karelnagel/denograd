import { add, ArrayMap, concat_bytes, Conv1d, type Conv2d, Device, dtypes, Embedding, env, get_key, idiv, type Layer, LayerNorm, Linear, load_state_dict, mod, num, range, replace_state_dict, safe_load, type sint, sub, Tensor, TinyJit, Tokenizer, UOp, type Variable, withEnvAsync, zip } from '../web.ts'

// deno-fmt-ignore
export const LANGUAGES = {
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
export const MODELS = {
  'tiny.en': {
    'url': 'https://huggingface.co/openai/whisper-tiny.en/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4 },
  },
  'tiny': {
    'url': 'https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4 },
  },
  'base.en': {
    'url': 'https://huggingface.co/openai/whisper-base.en/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6 },
  },
  'base': {
    'url': 'https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6 },
  },
  'small.en': {
    'url': 'https://huggingface.co/openai/whisper-small.en/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12 },
  },
  'small': {
    'url': 'https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12 },
  },
  'medium.en': {
    'url': 'https://huggingface.co/openai/whisper-medium.en/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24 },
  },
  'medium': {
    'url': 'https://huggingface.co/openai/whisper-medium/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24 },
  },
  'large-v2': {
    'url': 'https://huggingface.co/openai/whisper-large-v2/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1280, 'n_audio_head': 20, 'n_audio_layer': 32, 'n_text_ctx': 448, 'n_text_state': 1280, 'n_text_head': 20, 'n_text_layer': 32 },
  },
  'large': {
    'url': 'https://huggingface.co/openai/whisper-large/resolve/main/model.safetensors',
    'dims': { 'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1280, 'n_audio_head': 20, 'n_audio_layer': 32, 'n_text_ctx': 448, 'n_text_state': 1280, 'n_text_head': 20, 'n_text_layer': 32 },
  },
}
export type WhisperModel = keyof typeof MODELS
type Dims = typeof MODELS['tiny.en']['dims']

const get_window = (window_type: string, length: number) => {
  if (window_type !== 'hann') throw new Error("Only 'hann' window is implemented")
  const n = Tensor.arange(0, length, 1.0)
  return n.mul(2 * Math.PI, true).div(length - 1).cos().sub(1, true).mul(0.5, true)
}

const create_fourier_kernels = (n_fft: number, win_length?: number, freq_bins?: number, fmin = 50, fmax = 6000, sr = 16000, window = 'hann') => {
  if (freq_bins === undefined) freq_bins = idiv(n_fft, 2) + 1
  if (win_length === undefined) win_length = n_fft

  const s = Tensor.arange(0, n_fft, 1.0)
  const wsin = Tensor.empty([freq_bins, 1, n_fft])
  const wcos = Tensor.empty([freq_bins, 1, n_fft])
  const bins2freq: number[] = []
  const binslist: number[] = []

  let window_mask = get_window(window, win_length)
  const n = window_mask.shape.at(-1)!
  const lpad = idiv(sub(n_fft, n), 2)
  const lengths = range(window_mask.ndim).map(() => [0, 0] as [sint, sint])
  lengths[lengths.length - 1] = [lpad, sub(sub(n_fft, n), lpad)]
  if (num(lpad) <= 0) console.log('Warning: positive lpad implies n_fft higher than window length')
  window_mask = window_mask.pad(lengths)

  for (const k of range(freq_bins)) {
    bins2freq.push(k * sr / n_fft)
    binslist.push(k)
    wsin.set([k, 0, {}], s.mul(2 * Math.PI * k, true).div(n_fft).sin())
    wcos.set([k, 0, {}], s.mul(2 * Math.PI * k, true).div(n_fft).cos())
  }
  return [wsin.cast(dtypes.float32), wcos.cast(dtypes.float32), bins2freq, binslist, window_mask.cast(dtypes.float32)] as const
}

const mel_frequencies = async (n_mels = 128, fmin = 0.0, fmax = 11025.0, htk = false) => {
  const min_mel = await hz_to_mel(fmin, htk)
  const max_mel = await hz_to_mel(fmax, htk)

  const mels = Tensor.linspace(min_mel, max_mel, n_mels)

  return mel_to_hz(mels, htk)
}
const hz_to_mel = async (_frequencies: number, htk = false) => {
  const frequencies = new Tensor(_frequencies)

  if (htk) return frequencies.div(700.0).add(1.0, true).log().div(Math.log(10)).mul(2595, true)

  const f_min = 0.0
  const f_sp = 200.0 / 3

  let mels = frequencies.sub(f_min).div(f_sp)

  const min_log_hz = 1000.0
  const min_log_mel = (min_log_hz - f_min) / f_sp
  const logstep = Math.log(6.4) / 27.0

  if (frequencies.ndim) {
    const log_t = frequencies.ge(min_log_hz)
    mels = log_t.where(frequencies.div(min_log_hz).log().div(logstep).add(min_log_mel, true), mels)
  } else if (await frequencies.ge(min_log_hz).item()) {
    mels = frequencies.div(min_log_hz).log().div(logstep).add(min_log_mel, true)
  }
  return mels
}

const mel_to_hz = async (mels: Tensor, htk = false) => {
  if (htk) return mels.div(2595.0).pow(10, true).sub(1.0).mul(700, true)

  const f_min = 0.0
  const f_sp = 200.0 / 3
  let freqs = mels.mul(f_sp, true).add(f_min, true)

  const min_log_hz = 1000.0
  const min_log_mel = (min_log_hz - f_min) / f_sp
  const logstep = Math.log(6.4) / 27.0

  if (mels.ndim) {
    const log_t = mels.ge(min_log_mel)
    freqs = log_t.where(mels.sub(min_log_mel).mul(logstep).exp().mul(min_log_hz, true), freqs)
  } else if (await mels.ge(min_log_mel).item()) {
    freqs = (mels.sub(min_log_mel).mul(logstep, true)).exp().mul(min_log_hz)
  }
  return freqs
}

const get_mel = async (sr: number, n_fft: number, n_mels = 128, fmin = 0.0, fmax?: number, htk = false, norm = 2, dtype = dtypes.float32) => {
  if (fmax === undefined) fmax = sr / 2

  if (norm === undefined || norm === 1 && norm === Infinity) throw new Error(`Unsupported norm: ${norm}`)

  const fftfreqs = Tensor.linspace(0, sr / 2, 1 + idiv(n_fft, 2))

  const mel_f = await mel_frequencies(n_mels + 2, fmin, fmax, htk)

  const fdiff = mel_f.get({ start: 1 }).sub(mel_f.get({ stop: -1 }))
  const ramps = mel_f.reshape([-1, 1]).sub(fftfreqs.reshape([1, -1]))

  const t = []
  for (const i of range(n_mels)) {
    const lower = ramps.get(i).neg().div(fdiff.get(i))
    const upper = ramps.get(i + 2).div(fdiff.get(i + 1))
    t.push(lower.minimum(upper).maximum(0).unsqueeze(0))
  }
  let weights = await Tensor.cat(t, 0).realize()

  if (norm === 1) {
    const enorm = mel_f.get({ start: 2, stop: n_mels + 2 }).sub(mel_f.get({ stop: n_mels })).div(2, true)
    weights = weights.mul(enorm.unsqueeze(-1))
  }
  if (!mel_f.get({ stop: -2 }).eq(0).bitwise_or(weights.max(1).gt(0)).all().item()) {
    console.log(`Empty filters detected in mel frequency basis. Some channels will produce empty responses. Try increasing your sampling rate (and fmax) or reducing n_mels.`)
  }
  return weights
}
class STFT {
  pad_amount: number
  wsin: Tensor
  wcos: Tensor
  stride: number

  constructor(public n_fft = 128, public win_length = 128, public freq_bins?: number, hop_length = 64, public window = 'hann', public center = true, fmin = 50, fmax = 6000, sr = 16000, public trainable = false, public eps = 1e-10) {
    if (hop_length === undefined) hop_length = idiv(win_length, 4)

    this.pad_amount = idiv(this.n_fft, 2)
    this.stride = hop_length

    const [kernel_sin, kernel_cos, _, __, window_mask] = create_fourier_kernels(n_fft, win_length, freq_bins, fmin, fmax, sr, window)

    this.wsin = kernel_sin.mul(window_mask)
    this.wcos = kernel_cos.mul(window_mask)
    this.wsin.requires_grad = this.trainable
    this.wcos.requires_grad = this.trainable
  }
  call = (x: Tensor, return_spec = false) => {
    if (x.shape.length !== 2) throw new Error(`Input shape must be (batch, len), but is ${x.shape}`)
    if (this.center) x = x.pad([[0, 0], [this.pad_amount, this.pad_amount]])
    x = x.get({}, undefined, {})

    const spec_imag = x.conv2d(this.wsin, undefined, undefined, this.stride).get({}, { stop: this.freq_bins }, {})
    const spec_real = x.conv2d(this.wcos, undefined, undefined, this.stride).get({}, { stop: this.freq_bins }, {})
    if (return_spec) {
      let spec = spec_real.pow(2).add(spec_imag.pow(2)).sqrt()
      return this.trainable ? spec.add(this.eps) : spec
    } else {
      return Tensor.stack([spec_real, spec_imag.neg()], -1)
    }
  }
}
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

/**
 * param waveforms: A list of possibly variable length 16000Hz audio samples
 * param batch_size: The batch_size associated with the Whisper model being used to transcribe the audio. Used to prevent JIT mismatch errors since the encoder does not accept symbolic shapes
 * param truncate: If true, truncates (or pads) audio to exactly 30s for a single encoder pass
 * return: mel spectrogram of the given waveforms
 */
export const prep_audio = async (_waveforms: Float32Array[], batch_size: number, truncate = false) => {
  const pad_or_trim = (arr: Float32Array, target_len: number) => {
    const curr_len = arr.length
    if (curr_len === target_len) return arr
    else if (curr_len < target_len) {
      const res = new Float32Array(target_len)
      res.set(arr)
      return res
    } else return arr.slice(0, target_len)
  }

  let max_len = truncate ? SAMPLES_PER_SEGMENT : Math.max(..._waveforms.map((x) => x.length))
  const r = mod(max_len, SAMPLES_PER_SEGMENT)
  if (r > 0) max_len += SAMPLES_PER_SEGMENT - r
  let waveforms = new Tensor(concat_bytes(..._waveforms.map((w) => new Uint8Array(pad_or_trim(w, max_len).buffer))), { dtype: dtypes.float32 }).reshape([_waveforms.length, max_len])
  if (num(waveforms.shape[0]) > batch_size) throw new Error()
  if (num(waveforms.shape[0]) < batch_size) {
    // we could have a symbolic batch_size dim instead of manually padding here if conv/layernorm supported symbolic shapes
    waveforms = waveforms.pad([[0, batch_size - num(waveforms.shape[0])], [0, 0]])
  }
  const stft = new STFT(N_FFT, undefined, undefined, HOP_LENGTH).call(waveforms)
  const magnitudes = stft.get('...', 0).pow(2).add(stft.get('...', 1).pow(2)).get('...', { stop: -1 })
  const mel_spec = (await get_mel(RATE, N_FFT, N_MELS)).matmul(magnitudes)

  let log_spec = mel_spec.clip(1e-10).log().div(Math.log(10))
  log_spec = log_spec.maximum(log_spec.max([1, 2], true).sub(8.0))
  log_spec = log_spec.add(4.0).div(4.0)

  return log_spec.realize()
}

const get_encoding = async (is_multilingual: boolean) => {
  const url = `https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/${is_multilingual ? 'multilingual' : 'gpt2'}.tiktoken`
  const path = await env.fetchSave(url, get_key(url), env.CACHE_DIR)
  const data = await env.readTextFile(path)
  const ranks = data.split('\n').filter((line) => line).map((line) => line.split(' ')).map(([token, rank]) => [token === '=' ? '<THIS_IS_EXTRA_TOKEN_FOR_MULTILANG_MODELS>' : atob(token), Number(rank)])
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

const state_map = {
  'model.encoder.conv1': 'encoder.conv1',
  'model.encoder.conv2': 'encoder.conv2',
  'model.encoder.embed_positions.weight': 'encoder.positional_embedding',
  'model.encoder.layer_norm': 'encoder.ln_post',
  'model.encoder.layers.(\\d+).self_attn.q_proj': 'encoder.blocks.$1.attn.query',
  'model.encoder.layers.(\\d+).self_attn.k_proj': 'encoder.blocks.$1.attn.key',
  'model.encoder.layers.(\\d+).self_attn.v_proj': 'encoder.blocks.$1.attn.value',
  'model.encoder.layers.(\\d+).self_attn.out_proj': 'encoder.blocks.$1.attn.out',
  'model.encoder.layers.(\\d+).self_attn_layer_norm': 'encoder.blocks.$1.attn_ln',
  'model.encoder.layers.(\\d+).fc1': 'encoder.blocks.$1.mlp.0',
  'model.encoder.layers.(\\d+).fc2': 'encoder.blocks.$1.mlp.2',
  'model.encoder.layers.(\\d+).final_layer_norm': 'encoder.blocks.$1.mlp_ln',
  'model.decoder.embed_tokens.weight': 'decoder.token_embedding.weight',
  'model.decoder.embed_positions.weight': 'decoder.positional_embedding',
  'model.decoder.layer_norm': 'decoder.ln',
  'model.decoder.layers.(\\d+).self_attn.q_proj': 'decoder.blocks.$1.attn.query',
  'model.decoder.layers.(\\d+).self_attn.k_proj': 'decoder.blocks.$1.attn.key',
  'model.decoder.layers.(\\d+).self_attn.v_proj': 'decoder.blocks.$1.attn.value',
  'model.decoder.layers.(\\d+).self_attn.out_proj': 'decoder.blocks.$1.attn.out',
  'model.decoder.layers.(\\d+).self_attn_layer_norm': 'decoder.blocks.$1.attn_ln',
  'model.decoder.layers.(\\d+).encoder_attn.q_proj': 'decoder.blocks.$1.cross_attn.query',
  'model.decoder.layers.(\\d+).encoder_attn.k_proj': 'decoder.blocks.$1.cross_attn.key',
  'model.decoder.layers.(\\d+).encoder_attn.v_proj': 'decoder.blocks.$1.cross_attn.value',
  'model.decoder.layers.(\\d+).encoder_attn.out_proj': 'decoder.blocks.$1.cross_attn.out',
  'model.decoder.layers.(\\d+).encoder_attn_layer_norm': 'decoder.blocks.$1.cross_attn_ln',
  'model.decoder.layers.(\\d+).fc1': 'decoder.blocks.$1.mlp.0',
  'model.decoder.layers.(\\d+).fc2': 'decoder.blocks.$1.mlp.2',
  'model.decoder.layers.(\\d+).final_layer_norm': 'decoder.blocks.$1.mlp_ln',
}

export const init_whisper = async (model_name: WhisperModel, batch_size = 1): Promise<[Whisper, Tokenizer]> => {
  if (!MODELS[model_name]) throw new Error()
  const { dims, url } = MODELS[model_name]
  const filename = await env.fetchSave(url, model_name, env.CACHE_DIR)
  let state = await safe_load(filename)
  state = replace_state_dict(state, state_map)
  const model = await Whisper.init(dims, batch_size)
  await load_state_dict(model, state, false)
  const enc = await get_encoding(model.is_multilingual)
  return [model, enc]
}
const wav = (bytes: Uint8Array) => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)

  if (String.fromCharCode(...bytes.slice(0, 4)) !== 'RIFF' || String.fromCharCode(...bytes.slice(8, 12)) !== 'WAVE') {
    throw new Error('Not a valid WAV file')
  }

  let offset = 12, sampleRate = 0, numChannels = 0, bitsPerSample = 0, audioData = null

  while (offset < bytes.length) {
    const chunkId = String.fromCharCode(...bytes.slice(offset, offset + 4))
    const chunkSize = view.getUint32(offset + 4, true)
    offset += 8

    if (chunkId === 'fmt ') {
      const audioFormat = view.getUint16(offset, true)
      if (audioFormat !== 1) throw new Error('Only PCM format supported')
      numChannels = view.getUint16(offset + 2, true)
      sampleRate = view.getUint32(offset + 4, true)
      bitsPerSample = view.getUint16(offset + 14, true)
      offset += chunkSize
    } else if (chunkId === 'data') {
      audioData = bytes.slice(offset, offset + chunkSize)
      offset += chunkSize
    } else {
      offset += chunkSize
    }
  }
  if (!audioData) throw new Error('No data chunk found')

  const bytesPerSample = bitsPerSample / 8
  const sampleCount = audioData.length / (bytesPerSample * numChannels)

  const channelData = Array.from({ length: numChannels }, () => new Float32Array(sampleCount))

  for (let i = 0; i < sampleCount; i++) {
    const sampleOffset = i * bytesPerSample * numChannels
    for (let ch = 0; ch < numChannels; ch++) {
      const chOffset = sampleOffset + ch * bytesPerSample
      let sample
      if (bitsPerSample === 8) sample = (view.getUint8(chOffset) / 255 - 0.5) * 2
      else if (bitsPerSample === 16) sample = view.getInt16(chOffset, true) / 32768
      else throw new Error('Unsupported bits per sample: ' + bitsPerSample)
      channelData[ch][i] = sample
    }
  }

  return { sampleRate, numChannels, bitsPerSample, channelData }
}

export const load_file_waveform = async (filename: string): Promise<Float32Array[]> => {
  const data = await env.readFile(filename)
  const res = wav(data)
  if (res.sampleRate !== RATE) throw new Error()
  return res.channelData
}

export const transcribe_file = async (model: any, enc: Tokenizer, filename: string, language?: string) => {
  if (filename.startsWith('http')) filename = await env.fetchSave(filename, get_key(filename), env.CACHE_DIR)
  const waveforms = await load_file_waveform(filename)
  return await transcribe_waveform(model, enc, waveforms, false, language)
}

/**
 * Expects an array of shape (N,S) where N is the number waveforms to transcribe in parallel and S is number of 16000Hz samples
 * Returns the transcribed text if a single waveform is provided, or an array of transcriptions if multiple are provided
 */
const transcribe_waveform = async (model: Whisper, enc: Tokenizer, waveforms: Float32Array[], truncate = false, language?: string) => {
  let log_spec = await withEnvAsync({ DEVICE: env.CPU_DEVICE }, async () => await prep_audio(waveforms, model.batch_size, truncate))
  log_spec = log_spec.to(Device.DEFAULT)

  const nsample = model.decoder.max_tokens_to_sample

  const inferloop = async (ctx: Tensor, encoded_audio: Tensor) => {
    let pos = 0, next_tokens = ctx
    for (const i of range((nsample - start_tokens.length) * 2)) {
      next_tokens = (await model.decoder.call(next_tokens, pos, encoded_audio)).get({}, -1).argmax(-1).cast(dtypes.int32).reshape([-1, 1])
      if (env.DEBUG >= 1) console.log(enc.decode(await next_tokens.tolist()))
      next_tokens = ctx.get({}, -1).eq(eot).reshape([-1, 1]).where(next_tokens.full_like(eot), next_tokens)
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
    const language_token = enc.special_tokens['<|startoftranscript|>'] + 1 + Object.keys(LANGUAGES).indexOf(language || 'en')
    start_tokens.push(language_token)
    start_tokens.push(enc.special_tokens['<|transcribe|>'])
  }
  start_tokens.push(enc.special_tokens['<|notimestamps|>'])
  const eot = enc.special_tokens['<|endoftext|>']

  let ctx = new Tensor(start_tokens).reshape([1, -1]).expand([model.batch_size, start_tokens.length])
  let transcriptions: number[][] = waveforms.map(() => [])

  for (const curr_frame of range(0, log_spec.shape_num.at(-1), FRAMES_PER_SEGMENT)) {
    const encoded_audio = await model.encoder.encode.call(log_spec.get({}, {}, { start: curr_frame, stop: curr_frame + FRAMES_PER_SEGMENT }))

    if ((await ctx.tolist<number[][]>()).every((c) => c.length === ctx.get(0).length)) ctx = await inferloop(ctx, encoded_audio)
    else {
      throw new Error('Not tested')
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
