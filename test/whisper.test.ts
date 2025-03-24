import { expect, test } from 'vitest'
import { env, init_whisper, load_file_waveform, prep_audio } from '../denograd/mod.ts'
import { python } from './helpers.ts'

test(
  'Whisper',
  async () => {
    const [model, enc] = await init_whisper('tiny.en', 1)
    const url = 'https://huggingface.co/datasets/FL33TW00D-HF/ratchet-util/resolve/0c9601ad0d235e2e193b016b151aec991f497bf7/jfk.wav?download=true'
    const path = await env.fetchSave(url, 'jfk.wav', env.CACHE_DIR)

    // Getting log_spec
    const waveforms = await load_file_waveform(path)
    const waveforms_py = await python([
      'from examples.whisper import load_file_waveform',
      `out(load_file_waveform('${path}').shape)`,
    ])
    expect([waveforms[0].length]).toEqual(waveforms_py)

    // Prep audio
    const log_spec = await prep_audio(waveforms, 1)
    const log_spec_py = await python([
      'from examples.whisper import load_file_waveform, prep_audio',
      `waveforms = load_file_waveform('${path}')`,
      `out(prep_audio([waveforms], 1).shape)`,
    ])
    expect(log_spec.shape).toEqual(log_spec_py)

    // encoder
    const curr_frame = 0
    const encoded_audio = await model.encoder.encode.call(await log_spec.get({}, {}, { start: curr_frame, stop: curr_frame + 3000 }).realize())
    const encoded_audio_py = await python([
      'from tinygrad import Tensor',
      'from examples.whisper import load_file_waveform, prep_audio, init_whisper',
      `model, enc = init_whisper("tiny.en", batch_size=1)`,
      `waveforms = load_file_waveform('${path}')`,
      `log_spec = prep_audio([waveforms], 1)`,
      `out(model.encoder.encode(Tensor(log_spec[:, :, ${curr_frame}:${curr_frame} + ${3000}])).shape)`,
    ])
    expect(encoded_audio.shape).toEqual(encoded_audio_py)

    // decoder
    
  },
)
