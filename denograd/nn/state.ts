import { dtypes } from '../dtype.ts'
import { Env } from '../env/index.ts'
import { bytes_to_string, DEBUG, is_eq, isinstance, NotImplemented, round_up, string_to_bytes } from '../helpers.ts'
import { Tensor } from '../tensor.ts'

export const safe_dtypes = {
  'BOOL': dtypes.bool,
  'I8': dtypes.int8,
  'U8': dtypes.uint8,
  'I16': dtypes.int16,
  'U16': dtypes.uint16,
  'I32': dtypes.int,
  'U32': dtypes.uint,
  'I64': dtypes.int64,
  'U64': dtypes.uint64,
  'F16': dtypes.float16,
  'BF16': dtypes.bfloat16,
  'F32': dtypes.float32,
  'F64': dtypes.float64,
}
type SafeDType = keyof typeof safe_dtypes
export const inverse_safe_dtypes = new Map(Object.entries(safe_dtypes).map(([k, v]) => [v, k]))

/**
 * Loads a .safetensor file from disk, returning the data, metadata length, && metadata.
 */
export const safe_load_metadata = async (t: Tensor | string): Promise<[Tensor, number, Record<string, any>]> => {
  if (typeof t === 'string') t = new Tensor(t)
  const data_start = (await t.get({ start: 0, stop: 8 }).data()).cast('i').getValue(0) + 8
  return [t, data_start, JSON.parse(bytes_to_string((await t.get({ start: 8, stop: data_start }).data()).toBytes()))]
}
/**
 * Loads a .safetensor file from disk, returning the state_dict.
 *
 * ```python
 * state_dict = nn.state.safe_load("test.safetensor")
 * ```
 */
export const safe_load = async (fn: Tensor | string): Promise<Record<string, Tensor>> => {
  if (typeof fn === 'string') {
    fn = (fn.startsWith('http://') || fn.startsWith('https://')) ? await Tensor.from_url(fn, { device: Env.CPU_DEVICE }) : new Tensor(fn)
  }
  const [t, data_start, metadata] = await safe_load_metadata(fn)
  const data = t.get({ start: data_start })
  return Object.fromEntries(
    Object.entries(metadata)
      .filter(([k]) => k !== '__metadata__')
      .map(([k, v]) => [k, data.get({ start: v.data_offsets[0], stop: v.data_offsets[1] }).bitcast(safe_dtypes[v.dtype as SafeDType]).reshape(v.shape)]),
  )
}

/**
 * Saves a state_dict to disk in a .safetensor file with optional metadata.
 *
 * ```python
 * t = Tensor([1, 2, 3])
 * nn.state.safe_save({'t':t}, "test.safetensor")
 * ```
 */
export const safe_save = async (tensors: Record<string, Tensor>, fn: string, metadata?: Record<string, any>) => {
  const headers: Record<string, any> = {}
  let offset = 0
  if (metadata) headers.__metadata__ = metadata
  for (const [k, v] of Object.entries(tensors)) {
    headers[k] = { 'dtype': inverse_safe_dtypes.get(v.dtype), 'shape': v.shape, 'data_offsets': [offset, offset + v.nbytes()] }
    offset += v.nbytes()
  }
  let j = JSON.stringify(headers)
  j += '\x20'.repeat(round_up(j.length, 8) - j.length)
  // pathlib.Path(fn).unlink(missing_ok=true)
  const t = Tensor.empty([8 + j.length + offset], { dtype: dtypes.uint8, device: `DISK:${fn}` })
  await t.get({ start: 0, stop: 8 }).bitcast(dtypes.int64).assign_disk([j.length])
  await t.get({ start: 8, stop: 8 + j.length }).assign_disk(string_to_bytes(j))
  for (const [k, v] of Object.entries(await safe_load(t))) await v.assign_disk(tensors[k])
}
// state dict

/**
 * Returns a state_dict of the object, with optional prefix.
 *
 * ```python exec="true" source="above" session="tensor" result="python"
 * class Net {
 * const __init__ = () => {
 * this.l1 = nn.Linear(4, 5)
 * this.l2 = nn.Linear(5, 6)
 *
 * net = Net()
 * console.log(nn.state.get_state_dict(net).keys())
 * ```
 */
export const get_state_dict = (obj: any, prefix = ''): Record<string, Tensor> => {
  if (isinstance(obj, Tensor)) return { [prefix.replace(/^\.+|\.+$/g, '')]: obj }
  if (Array.isArray(obj)) return Object.fromEntries(obj.flatMap((x, i) => Object.entries(get_state_dict(x, `${prefix}${i}.`))))
  if (typeof obj === 'object') return Object.fromEntries(Object.entries(obj).flatMap(([k, v]) => Object.entries(get_state_dict(v, `${prefix}${k}.`))))
  return {}
}

/**
 * ```python exec="true" source="above" session="tensor" result="python"
 * class Net:
 * const __init__ = () => {
 * this.l1 = nn.Linear(4, 5)
 * this.l2 = nn.Linear(5, 6)
 *
 * net = Net()
 * console.log(len(nn.state.get_parameters(net)))
 * ```
 */
export const get_parameters = (obj: any): Tensor[] => {
  return Object.values(get_state_dict(obj))
}

/**
 * Loads a state_dict into a model.
 *
 * ```python
 * class Net:
 * const __init__ = () => {
 * this.l1 = nn.Linear(4, 5)
 * this.l2 = nn.Linear(5, 6)
 *
 * net = Net()
 * state_dict = nn.state.get_state_dict(net)
 * nn.state.load_state_dict(net, state_dict)
 * ```
 */
export const load_state_dict = async (model: any, state_dict: Record<string, Tensor>, strict = true, verbose = true, consume = false) => {
  const model_state_dict = get_state_dict(model)
  if (DEBUG >= 1 && Object.keys(state_dict).length > Object.keys(model_state_dict).length) {
    console.log('WARNING: unused weights in state_dict', Object.keys(state_dict).filter((x) => !Object.keys(model_state_dict).includes(x)).toSorted())
  }
  const t = Object.entries(model_state_dict)
  for await (const [k, v] of t) {
    if (state_dict[k] === undefined && !strict) {
      if (DEBUG >= 1) console.log(`WARNING: !loading ${k}`)
      continue
    }
    if (!is_eq(v.shape, state_dict[k].shape)) throw new Error(`Shape mismatch in layer ${k}: Expected shape ${v.shape}, but found ${state_dict[k].shape} in state dict.`)
    if (Array.isArray(v.device)) {
      if (Array.isArray(state_dict[k].device)) await v.replace(state_dict[k]).realize()
      else await v.replace(state_dict[k].shard(v.device, v.lazydata.axis)).realize()
    } else {
      await v.replace(state_dict[k].to(v.device)).realize()
    }
    if (consume) delete state_dict[k]
  }
}

export const tar_extract = (t: Tensor): Record<string, Tensor> => {
  throw new NotImplemented()
}

export const ggml_data_to_tensor = (t: Tensor, n: number, ggml_type: number): Tensor => {
  throw new NotImplemented()
}

export const gguf_load = (tensor: Tensor): [Record<string, any>, Record<string, Tensor>] => {
  throw new NotImplemented()
}
