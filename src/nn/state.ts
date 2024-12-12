import { dtypes } from '../dtype.ts'
import { isinstance } from '../helpers.ts'
import { Tensor } from '../tensor.ts'

export class TensorIO {}
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
export const inverse_safe_dtypes = new Map(Object.entries(safe_dtypes).map(([k, v]) => [v, k]))

export const safe_load_metadata = (t: Tensor): [Tensor, number, Record<string, any>] => {
  throw new Error()
}
export const safe_load = (fn: Tensor | string): Map<string, Tensor> => {
  throw new Error()
}
export const safe_save = (tensors: Map<string, Tensor>, fn: string, metadata?: Map<string, any>) => {
  throw new Error()
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
export const get_state_dict = (obj: any, prefix = '', tensor_type = Tensor): Record<string, Tensor> => {
  if (isinstance(obj, tensor_type)) return { [prefix.replace(/^\.+|\.+$/g, '')]: obj }
  // if (hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  // namedtuple
  // if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
  // if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
  if (Array.isArray(obj)) return obj.reduce((acc, x, i) => ({ ...acc, ...get_state_dict(x, `${prefix}${i}`, tensor_type) }))
  if (typeof obj === 'object') return Object.entries(obj).reduce((acc, [k, v]) => ({ ...acc, ...get_state_dict(v, `${prefix}${k}`, tensor_type) }), {})
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
export const load_state_dict = (model: any, state_dict: Map<string, Tensor>, strict = true, verbose = true, consume = false): undefined => {
  throw new Error()
}
export const tar_extract = (t: Tensor): Record<string, Tensor> => {
  throw new Error()
}
