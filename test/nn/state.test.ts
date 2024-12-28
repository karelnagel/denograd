import { expect } from 'expect/expect'
import { Tensor } from '../../src/tensor.ts'
import { get_state_dict, safe_load } from '../../src/nn/state.ts'
import { zip } from '../../src/helpers.ts'
import { dtypes } from '../../src/dtype.ts'
import { safe_save } from '../../src/nn/state.ts'
import { python } from '../helpers.ts'

Deno.test('get_state_dict', () => {
  class MNIST {
    idk = new Tensor([2])
    layers: ((x: Tensor) => Tensor)[]
    constructor() {
      this.layers = [
        // new nn.Conv2d(1, 32, 5).call,
        (x) => x.relu(),
        // new nn.Conv2d(32, 32, 5).call,
        (x) => x.relu(),
        // new nn.BatchNorm(32).call,
        (x) => x.max_pool2d(),
        // new nn.Conv2d(32, 64, 3).call,
        (x) => x.relu(),
        // new nn.Conv2d(64, 64, 3).call,
        (x) => x.relu(),
        // new nn.BatchNorm(64).call,
        (x) => x.max_pool2d(),
        (x) => x.flatten(1),
        // new nn.Linear(576, 10).call,
      ]
    }
    call = (x: Tensor): Tensor => x.sequential(this.layers)
  }
  const model = new MNIST()
  const dict = get_state_dict(model)
  for (const [entry, expected] of zip(Object.entries(dict), ['idk'])) {
    expect(entry[0]).toEqual(expected)
    expect(entry[1]).toBeInstanceOf(Tensor)
  }
})

Deno.test('safe_save', async () => {
  const dict = {
    idk: new Tensor([3, 3, 4], { dtype: dtypes.int }),
    shaped: new Tensor([3, 3, 4, 2], { dtype: dtypes.int }).reshape([2, 2]),
  }
  const path = '/tmp/safe_save_test.safetensor'
  // Saving in TS
  safe_save(dict, path)

  // Loading in PY
  const res = await python<Map<string, Tensor>>([
    'from tinygrad.nn.state import safe_load',
    'out(safe_load(data[0]))',
  ], [path])
  expect(res).toBe(false)
})

Deno.test('safe_load', async () => {
  const path = '/tmp/safe_load_test.safetensor'
  const dict = {
    'idk': [2, 3, 4, 4],
    'idk33': [2, 3, 4, 4, 4, 4],
  }

  // Saving in PY
  await python([
    'from tinygrad.nn.state import safe_save',
    'dict = { k: tiny.Tensor(v, device="PYTHON") for k,v in data[1].items() }',
    'safe_save(dict, data[0])',
  ], [path, dict])

  // Reading in TS
  const res = safe_load(path)
  for (const [entry, expected] of zip(Object.entries(res), Object.entries(dict))) {
    expect(entry[0]).toBe(expected[0])
    expect(entry[1].tolist()).toBe(expected[1])
  }
})
