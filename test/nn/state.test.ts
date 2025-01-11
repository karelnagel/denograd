import { expect } from 'expect/expect'
import { Tensor } from '../../denograd/tensor.ts'
import { get_parameters, get_state_dict, safe_load } from '../../denograd/nn/state.ts'
import { zip } from '../../denograd/helpers.ts'
import { safe_save } from '../../denograd/nn/state.ts'
import { python } from '../helpers.ts'
import { MNIST } from '../../models/mnist.ts'

Deno.test('get_state_dict', () => {
  const model = new MNIST()
  const dict = get_state_dict(model)
  expect(Object.entries(dict).length).toBe(20)
  for (const [key, tensor] of Object.entries(dict)) {
    const [layers, num, name] = key.split('.')
    expect(layers).toBe('layers')
    expect(Number(num)).toBeGreaterThanOrEqual(0)
    expect(Number(num)).toBeLessThanOrEqual(13)
    expect(['weight', 'bias', 'num_batches_tracked', 'running_mean', 'running_var', 'num_batches_tracked']).toContain(name)
    expect(tensor).toBeInstanceOf(Tensor)
  }
})
Deno.test('get_parameters', () => {
  const model = new MNIST()
  const params = get_parameters(model)
  expect(params.length).toBe(20)
  for (const tensor of params) {
    expect(tensor).toBeInstanceOf(Tensor)
  }
})

Deno.test('safe_save', async () => {
  const dict = {
    idk: new Tensor([2, 3, 4, 4]),
    idk33: new Tensor([2, 3, 4, 4, 4, 4]),
  }
  const path = '/tmp/safe_save_test.safetensor'
  // Saving in TS
  safe_save(dict, path)

  // Loading in PY
  const res = await python<Map<string, Tensor>>([
    'from tinygrad.nn.state import safe_load',
    'out(safe_load(data[0]))',
  ], [path])
  for (const [entry, expected] of zip(res.entries(), Object.entries(dict))) {
    expect(entry[0]).toBe(expected[0])
    expect(entry[1].tolist()).toEqual(expected[1].tolist())
  }
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
    expect(entry[1].tolist()).toEqual(expected[1])
  }
})
