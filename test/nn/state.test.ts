import { Tensor } from '../../jsgrad/tensor.ts'
import { get_parameters, get_state_dict, gguf_load, safe_load } from '../../jsgrad/nn/state.ts'
import { zip } from '../../jsgrad/helpers/helpers.ts'
import { safe_save } from '../../jsgrad/nn/state.ts'
import { python } from '../helpers.ts'
import { MNIST } from '../../jsgrad/models/mnist.ts'
import { expect, test } from 'vitest'

test('get_state_dict', () => {
  const model = new MNIST()
  const dict = get_state_dict(model)
  expect(Object.entries(dict).length).toBe(20)
  for (const [key, tensor] of Object.entries(dict)) {
    const [layers, num, name] = key.split('.')
    expect(layers).toBe('layers')
    expect(Number(num)).toBeGreaterThanOrEqual(0)
    expect(Number(num)).toBeLessThanOrEqual(13)
    expect([
      'weight',
      'bias',
      'num_batches_tracked',
      'running_mean',
      'running_var',
      'num_batches_tracked',
    ]).toContain(name)
    expect(tensor).toBeInstanceOf(Tensor)
  }
})
test('get_parameters', () => {
  const model = new MNIST()
  const params = get_parameters(model)
  expect(params.length).toBe(20)
  for (const tensor of params) {
    expect(tensor).toBeInstanceOf(Tensor)
  }
})

test('safe_save', { skip: true }, async () => {
  const dict = {
    idk: new Tensor([2, 3, 4, 4]),
    idk33: new Tensor([2, 3, 4, 4, 4, 4]),
  }
  const path = '/tmp/safe_save_test.safetensor'
  // Saving in TS
  await safe_save(dict, path)

  // Loading in PY
  const res = await python<Map<string, Tensor>>([
    'from tinygrad.nn.state import safe_load',
    'out(safe_load(data[0]))',
  ], [path])
  for (const [entry, expected] of zip(res.entries(), Object.entries(dict))) {
    expect(entry[0]).toBe(expected[0])
    expect(await entry[1].tolist()).toEqual(await expected[1].tolist())
  }
})

test('safe_load', async () => {
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
  const res = await safe_load(path)
  for (
    const [entry, expected] of zip(Object.entries(res), Object.entries(dict))
  ) {
    expect(entry[0]).toBe(expected[0])
    expect(await entry[1].tolist()).toEqual(expected[1])
  }
})

test('gguf_load', { skip: true }, async () => {
  const path = 'weights/llama3-1b-instruct/Llama-3.2-1B-Instruct-Q6_K.gguf'
  const [py1, py2] = await python([
    'from tinygrad.nn.state import gguf_load',
    `res = gguf_load("${path}")`,
    `out([[key for key in res[0]], [key for key in res[1]]])`,
  ])
  const data = await Deno.readFile(path)
  const [ts1, ts2] = await gguf_load(data)
  expect(Object.keys(ts1)).toEqual(py1)
  expect(Object.keys(ts2)).toEqual(py2)
})
