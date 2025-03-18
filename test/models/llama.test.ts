import { Tensor } from '../../denograd/tensor.ts'
import { dtypes } from '../../denograd/dtype.ts'
import { Device } from '../../denograd/device.ts'
import { compare } from '../helpers.ts'
import { precompute_freqs_cis, Transformer } from '../../denograd/models/llama.ts'
import { describe as describe } from 'vitest'

describe(
  'precompute_freqs_cis',
  compare(
    [
      [4, 164, 500],
    ],
    precompute_freqs_cis,
    [
      'from extra.models.llama import precompute_freqs_cis',
      'out(precompute_freqs_cis(*data))',
    ],
  { skip: Device.DEFAULT === 'WASM' || Device.DEFAULT === 'DAWN' },
),
)

describe(
  'Transformer.forward',
  compare(
    () =>
      [
        [
          [2048, 8192, 32, 16, 1e-05, 128256, undefined, undefined, 8, 500000, 8192, false, undefined],
          [new Tensor([[459]], { requires_grad: undefined, dtype: dtypes.int }), 7, 0.85, 0, 0.0, 0.0, 0.0],
        ],
        [
          [2048, 8192, 32, 16, 1e-05, 128256, undefined, undefined, 8, 500000, 8192, false, undefined],
          [new Tensor([[271]], { requires_grad: undefined, dtype: dtypes.int }), 4, 0.85, 0, 0.0, 0.0, 0.0],
        ],
      ] as any,
    (init: any, forward: any) => {
      Tensor.manual_seed(4)
      return new (Transformer as any)(...init).forward(...forward)
    },
    [
      'from extra.models.llama import Transformer',
      'import tinygrad.nn as nn',
      'tiny.Tensor.manual_seed(4)',
      'd = data[0]',
      'out(Transformer(d[0],d[1],d[2],d[3],d[4],d[5],nn.Linear,nn.Embedding,d[8],d[9],d[10],d[11]).forward(*data[1]))',
    ],
  { skip: Device.DEFAULT === 'WEBGPU' || Device.DEFAULT === 'WASM' },
),
)
