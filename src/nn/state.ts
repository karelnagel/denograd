import { dtypes } from '../dtype.ts'
import { bytesToString, DEBUG, isinstance, stringToBytes } from '../helpers.ts'
import { Tensor } from '../tensor.ts'
import { tqdm } from '../tqdm.ts'

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
export const safe_load_metadata = (t: Tensor | string): [Tensor, number, Record<string, any>] => {
  if (typeof t === 'string') t = new Tensor(t)
  const data_start = t.get({ start: 0, stop: 8 }).data().cast('b').getValue(0) + 8
  return [t, data_start, JSON.parse(bytesToString(t.get({ start: 8, stop: data_start }).data().toBytes()))]
}
/**
 * Loads a .safetensor file from disk, returning the state_dict.
 * ```python
 * state_dict = nn.state.safe_load("test.safetensor")
 * ```
 */
export const safe_load = (fn: Tensor | string): Record<string, Tensor> => {
  if (typeof fn === 'string') fn = new Tensor(fn)
  const [t, data_start, metadata] = safe_load_metadata(fn)
  const data = t.get({ start: data_start })
  return Object.fromEntries(
    Object.entries(metadata)
      .filter(([k]) => k !== '__metadata__')
      .map(([k, v]) => [k, data.get({ start: v.data_offsets[0], stop: v.data_offsets[1] }).bitcast(safe_dtypes[v.dtype as SafeDType]).reshape(v.shape)]),
  )
}

/**
 * Saves a state_dict to disk in a .safetensor file with optional metadata.
 * ```python
 * t = Tensor([1, 2, 3])
 * nn.state.safe_save({'t':t}, "test.safetensor")
 * ```
 */
export const safe_save = (tensors: Record<string, Tensor>, fn: string, metadata?: Record<string, any>) => {
  const headers: Record<string, any> = {}
  let offset = 0
  if (metadata) headers.__metadata__ = metadata
  for (const [k, v] of Object.entries(tensors)) {
    headers[k] = { 'dtype': inverse_safe_dtypes.get(v.dtype), 'shape': v.shape, 'data_offsets': [offset, offset + v.nbytes()] }
    offset += v.nbytes()
  }
  let j = JSON.stringify(headers)
  j += '\x20'.repeat((8 - j.length % 8) % 8)
  // pathlib.Path(fn).unlink(missing_ok=true)
  const t = Tensor.empty([8 + j.length + offset], { dtype: dtypes.uint8, device: `DISK:${fn}` })
  t.get({ start: 0, stop: 8 }).bitcast(dtypes.int64).assign([j.length])
  t.get({ start: 8, stop: 8 + j.length }).assign(stringToBytes(j))
  for (const [k, v] of Object.entries(safe_load(t))) v.assign(tensors[k])
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
 * ```python
 * class Net:
 * const __init__ = () => {
 * this.l1 = nn.Linear(4, 5)
 * this.l2 = nn.Linear(5, 6)
 * net = Net()
 * state_dict = nn.state.get_state_dict(net)
 * nn.state.load_state_dict(net, state_dict)
 * ```
 */
export const load_state_dict = async (model: any, state_dict: Record<string, Tensor>, strict = true, verbose = true, consume = false) => {
  // const start_mem_used = GlobalCounters.mem_used
  // with Timing("loaded weights in ", lambda et_ns: `, ${(B:=(GlobalCounters.mem_used-start_mem_used))/1e9:.2f} GB loaded at ${B/et_ns:.2f} GB/s`):
  const model_state_dict = get_state_dict(model)
  if (DEBUG >= 1 && Object.keys(state_dict).length > Object.keys(model_state_dict).length) {
    console.log('WARNING: unused weights in state_dict', Object.keys(state_dict).filter((x) => !Object.keys(model_state_dict).includes(x)).toSorted())
  }
  const t = tqdm(Object.entries(model_state_dict), { label: 'Downloading' })
  for await (const [k, v] of t) {
    // t.desc = `ram used: ${GlobalCounters.mem_used/1e9:5.2f} GB, ${k:50s}: `
    if (state_dict[k] === undefined && !strict) {
      if (DEBUG >= 1) console.log(`WARNING: !loading ${k}`)
      continue
    }
    if (v.shape !== state_dict[k].shape) throw new Error(`Shape mismatch in layer ${k}: Expected shape ${v.shape}, but found ${state_dict[k].shape} in state dict.`)
    //     // if isinstance((mlb:=v.lazydata), MultiLazyBuffer):
    //     //   if isinstance(state_dict[k].lazydata, MultiLazyBuffer): v.replace(state_dict[k]).realize()
    //     //   else: v.replace(state_dict[k].shard(mlb.device, mlb.axis)).realize()
    else v.replace(state_dict[k].to(v.device)).realize()
    if (consume) delete state_dict[k]
  }
}

export const tar_extract = (t: Tensor): Record<string, Tensor> => {
  throw new Error('Not implemented')
}

export const ggml_data_to_tensor = (t: Tensor, n: number, ggml_type: number): Tensor => {
  throw new Error('Not implemented')
}

export const gguf_load = (tensor: Tensor): [Record<string, any>, Record<string, Tensor>] => {
  throw new Error('Not implemented')
}
