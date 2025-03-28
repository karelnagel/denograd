import { dtypes, type FmtStr } from '../dtype.ts'
import { env } from '../env/index.ts'
import { bytes_to_string, idiv, is_eq, NotImplemented, prod, range, round_up, string_to_bytes, vars } from '../helpers/helpers.ts'
import { MemoryView } from '../helpers/memoryview.ts'
import { Tensor } from '../tensor.ts'
import { Tqdm, type TqdmOnProgress } from '../helpers/tqdm.ts'

class TensorIO {
  // TODO: if mmap working for disk device, then it should use tensor
  constructor(public _data: Uint8Array, public _position = 0) {
    // if (_tensor.ndim !== 1 || _tensor.dtype !== dtypes.uint8) throw new Error('Tensor must be 1d and of dtype uint8!')
  }
  readable = () => true
  read = async (size: number | bigint): Promise<Uint8Array> => {
    if (typeof size === 'bigint') size = Number(size)
    const data = this._data.slice(this._position, this._position + size)
    this._position += size
    return data
  }
}

const safe_dtypes = {
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
  return [t, data_start, JSON.parse(bytes_to_string((await t.get({ start: 8, stop: data_start }).data()).bytes))]
}

const accept_filename = async (fn: Tensor | string): Promise<Tensor> => {
  if (typeof fn === 'string') {
    fn = (fn.startsWith('http://') || fn.startsWith('https://')) ? await Tensor.from_url(fn, { device: env.CPU_DEVICE }) : await Tensor.from_file(fn)
  }
  return fn
}
/**
 * Loads a .safetensor file from disk, returning the state_dict.
 *
 * ```python
 * state_dict = nn.state.safe_load("test.safetensor")
 * ```
 */
export const safe_load = async (fn: Tensor | string): Promise<Record<string, Tensor>> => {
  fn = await accept_filename(fn)
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
  if (obj instanceof Tensor) return { [prefix.replace(/^\.+|\.+$/g, '')]: obj }
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

export const replace_state_dict = (state: Record<string, Tensor>, replace: Record<string, string>, strict = true) => {
  const out: Record<string, Tensor> = {}
  for (let [key, value] of Object.entries(state)) {
    for (const [k, v] of Object.entries(replace)) {
      key = key.replace(RegExp(k), v)
    }
    out[key] = value
  }

  return out
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
export const load_state_dict = async (model: any, state_dict: Record<string, Tensor>, strict = true, verbose = true, consume = false, onProgress?: TqdmOnProgress) => {
  const model_state_dict = get_state_dict(model)
  if (vars.DEBUG >= 1 && Object.keys(state_dict).length > Object.keys(model_state_dict).length) {
    console.log('WARNING: unused weights in state_dict', Object.keys(state_dict).filter((x) => !Object.keys(model_state_dict).includes(x)).toSorted())
  }
  const t = Object.entries(model_state_dict)
  for (const [k, v] of new Tqdm(t, { label: `Loading state dict`, onProgress })) {
    if (state_dict[k] === undefined && !strict) {
      if (vars.DEBUG >= 1) console.warn(`WARNING: not loading ${k}`)
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

/**
 * Converts ggml tensor data to a tinygrad tensor.
 *
 * Supported native types: float32 (id: 0), float16 (id: 1), int8 (id: 16), int16 (id: 17), int32 (id: 18)
 * Supported quantized types: Q4_0 (id: 2), Q4_1 (id: 3), Q8_0 (id: 8), Q6_K (id: 14)
 */
// https://github.com/ggerganov/ggml/blob/6dccc647264f5429df2624f36138f601e7ce23e5/include/ggml.h#L356
export const ggml_data_to_tensor = (t: Uint8Array, n: number, ggml_type: number): Tensor => {
  // native types
  const dtype = { 0: dtypes.float32, 1: dtypes.float16, 16: dtypes.int8, 17: dtypes.int16, 18: dtypes.int32 }[ggml_type]
  if (dtype !== undefined) {
    return new Tensor(t.slice(0, dtype.itemsize * n)).bitcast(dtype)
  }

  const q_to_uint8 = (t: Tensor, b: number): Tensor => {
    // TODO: rewrite with arange?
    const shift_tensor = Tensor.stack(range(idiv(8, b)).map((i) => new Tensor(2 ** (i * b), { device: t.device, dtype: t.dtype }))), bitmask = 0xff >> (8 - b)
    return t.unsqueeze(-1).expand([...t.shape, idiv(8, b)]).idiv(shift_tensor).bitwise_and(bitmask).transpose(-1, -2).flatten(-2)
  }

  // map to (number of elements, number of bytes)
  const nelements_nbytes = { 2: [32, 18], 3: [32, 20], 14: [256, 210], 8: [32, 34] }[ggml_type]
  if (nelements_nbytes !== undefined) {
    const blocks = new Tensor(t.slice(0, idiv(n, nelements_nbytes[0]) * nelements_nbytes[1])).reshape([-1, nelements_nbytes[1]])
    if (ggml_type === 2) return (q_to_uint8(blocks.get({}, { start: 2 }), 4).bitcast(dtypes.int8).sub(8)).mul(blocks.get({}, { stop: 2 }).bitcast(dtypes.float16).cast(dtypes.float32))
    if (ggml_type === 3) {
      const [d, m] = [0, 2].map((s) => blocks.get({}, { start: s, stop: s + 2 }).bitcast(dtypes.float16).cast(dtypes.float32))
      return q_to_uint8(blocks.get({}, { start: 4 }), 4).bitcast(dtypes.int8).mul(d).add(m)
    }
    if (ggml_type === 8) return blocks.get({}, { stop: 2 }).bitcast(dtypes.float16).cast(dtypes.float32).mul(blocks.get({}, { start: 2 }).bitcast(dtypes.int8))
    if (ggml_type === 14) {
      const xl = q_to_uint8(blocks.get({}, { stop: 128 }).reshape([-1, 2, 64]), 4), xh = q_to_uint8(blocks.get({}, { start: 128, stop: 192 }).reshape([-1, 2, 32]), 2).lshift(4)
      const scales = blocks.get({}, { start: 192, stop: 208 }).bitcast(dtypes.int8).unsqueeze(-1).expand([-1, 16, 16]).reshape([-1, 256])
      const d = blocks.get({}, { start: -2 }).bitcast(dtypes.float16).cast(dtypes.float32).expand([-1, 256])
      return d.mul(xl.bitwise_or(xh).bitcast(dtypes.int8).sub(32).flatten(-2)).mul(scales)
    }
  }
  throw new Error(`GGML type '${ggml_type}' is not supported!`)
}

const TYPES: [number, FmtStr, number][] = [[0, 'B', 1], [1, 'b', 1], [2, 'H', 2], [3, 'h', 2], [4, 'I', 4], [5, 'i', 4], [6, 'f', 4], [7, '?', 1], [10, 'Q', 8], [11, 'q', 8], [12, 'd', 8]]
/**
 * Loads a gguf file from a tensor.
 *
 * ```python
 * fn = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
 * gguf_tensor = Tensor.empty(os.stat(fn).st_size, dtype=dtypes.uint8, device=f"disk:{fn}").to(Device.DEFAULT)
 * kv_data, state_dict = gguf_load(gguf_tensor)
 * ```
 */
export const gguf_load = async (data: Uint8Array, onProgress?: TqdmOnProgress): Promise<[Record<string, any>, Record<string, Tensor>]> => {
  // tensor = await accept_filename(tensor)
  const reader = new TensorIO(data)
  const kv_data: Record<string, any> = {}, state_dict: Record<string, any> = {}
  const read_unpack = async (fmt: FmtStr, n: number) => {
    const res = new MemoryView(await reader.read(n)).cast(fmt)
    if (res.length !== 1) throw new Error(`Res lenght should be 1, but was ${res.length}`)
    return res.getValue(0)
  }
  const read_str = async () => bytes_to_string(await reader.read(await read_uint64()))
  const read_arr = async (): Promise<any[]> => {
    const reader = readers[await read_int32()], n = await read_uint64()
    const res = []
    for (let i = 0; i < n; i++) {
      res.push(await reader())
    }
    return res
  }
  const readers: Record<number, () => Promise<any>> = {
    8: read_str,
    9: read_arr,
    ...Object.fromEntries(TYPES.map(([t, f, nb]) => [t, async () => await read_unpack(f, nb)])),
  }
  const read_uint32: () => Promise<number> = readers[4], read_int32: () => Promise<number> = readers[5], read_uint64: () => Promise<bigint> = readers[10], read_int64: () => Promise<bigint> = readers[11]

  const magic = await reader.read(4), version = await read_int32(), n_tensors = await read_int64(), n_kv = await read_int64()
  if (bytes_to_string(magic) !== 'GGUF' || ![2, 3].includes(version)) throw new Error(`Invalid GGUF format, magic= ${bytes_to_string(magic)}`)
  for (const i of new Tqdm(Number(n_kv), { onProgress, label: 'Loading gguf kv data' })) {
    const k = await read_str(), typ = await read_int32()
    kv_data[k] = await readers[typ]()
  }
  const t_infos: [string, number[], number, number][] = []
  for (const i of new Tqdm(Number(n_tensors), { onProgress, label: `Loading gguf tensors` })) {
    const first = await read_str()
    const second = []
    for (const _ of range(await read_uint32())) second.push(Number(await read_uint64()))
    t_infos.push([first, second, await read_int32(), Number(await read_uint64())])
  }
  const data_start = round_up(reader._position, kv_data['general.alignment'] || 32)

  for (const [name, dims, typ, off] of new Tqdm(t_infos, { label: 'ggml data to tensor' })) state_dict[name] = ggml_data_to_tensor(data.slice(data_start + off), prod(dims), typ).reshape(dims.toReversed())
  return [kv_data, state_dict]
}
