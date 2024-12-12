// import os, json, pathlib, zipfile, pickle, tarfile, struct, functools, io
// from typing import Dict, Union, List, Optional, any, Tuple, Callable, BinaryIO, Iterable, TypeVar
// from tinygrad.tensor import Tensor
// from tinygrad.dtype import dtypes
// from tinygrad.helpers import prod, argsort, DEBUG, Timing, CI, unwrap, GlobalCounters, tqdm
// from tinygrad.shape.view import strides_for_shape

import { Tensor } from '../tensor.ts'

// class TensorIO extends io.RawIOBase, BinaryIO {
//   pass
// safe_dtypes = {"BOOL":dtypes.boolean, "I8":dtypes.int8, "U8":dtypes.uint8, "I16":dtypes.int16, "U16":dtypes.uint16, "I32":dtypes.number, "U32":dtypes.uint,
//                "I64":dtypes.int64, "U64":dtypes.uint64, "F16":dtypes.float16, "BF16":dtypes.bfloat16, "F32":dtypes.float32, "F64":dtypes.float64}
// inverse_safe_dtypes = {v:k for k,v in safe_dtypes.items()}

// R = TypeVar('R')
// const accept_filename = (func: Callable[[Tensor], R]): Callable[[Union[Tensor, string, pathlib.Path]], R] => {
//   @functools.wraps(func)
//   const wrapper = (fn: Union[Tensor, string, pathlib.Path]): R =>  func(Tensor(pathlib.Path(fn)) if !isinstance(fn, Tensor) else fn)
//   return wrapper

// @accept_filename
// const safe_load_metadata = (t:Tensor): Tensor, number, Map<string[>] => {
//   pass
// const safe_load = (fn:Union[Tensor, string, pathlib.Path]): Map<string, Tensor> => {
//   pass
// const safe_save = (tensors:Map<string, Tensor>, fn:string, metadata?: Map<string, any>) => {
//   pass
// // state dict

// from collections import OrderedDict
// const get_state_dict = (obj, prefix:string='', tensor_type=Tensor): Map<string, Tensor> => {
// /**
//  * Returns a state_dict of the object, with optional prefix.
//  *
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * class Net {
//  * const __init__ = () => {
//  * this.l1 = nn.Linear(4, 5)
//  * this.l2 = nn.Linear(5, 6)
//  *
//  * net = Net()
//  * console.log(nn.state.get_state_dict(net).keys())
//  * ```
//  */
//   if isinstance(obj, tensor_type): return {prefix.strip('.'):obj}
//   if hasattr(obj, '_asdict'): return get_state_dict(obj._asdict(), prefix, tensor_type)  // namedtuple
//   if isinstance(obj, OrderedDict): return get_state_dict(dict(obj), prefix, tensor_type)
//   if hasattr(obj, '__dict__'): return get_state_dict(obj.__dict__, prefix, tensor_type)
//   state_dict = {}
//   if isinstance(obj, (list, tuple)):
//     for (const i,x of enumerate(obj)){ state_dict.update(get_state_dict(x, `${prefix}${string(i)}.`, tensor_type))
//   elif isinstance(obj, dict):
//     for (const k,v of obj.items()){ state_dict.update(get_state_dict(v, `${prefix}${string(k)}.`, tensor_type))
//   return state_dict
// const get_parameters = (obj): Tensor[] => {
// /**
//  * ```python exec="true" source="above" session="tensor" result="python"
//  * class Net:
//  * const __init__ = () => {
//  * this.l1 = nn.Linear(4, 5)
//  * this.l2 = nn.Linear(5, 6)
//  *
//  * net = Net()
//  * console.log(len(nn.state.get_parameters(net)))
//  * ```
//  */
//   return list(get_state_dict(obj).values())

// const load_state_dict = (model, state_dict:Map<string, Tensor>, strict=true, verbose=true, consume=false): undefined => {
//   pass
export const tar_extract = (t: Tensor): Record<string, Tensor> => {
  throw new Error()
}
// // torch support!

// const torch_load = (fn:string): Map<string, Tensor> => {
//   pass
// const ggml_data_to_tensor = (t: Tensor, n:number, ggml_type: number): Tensor => {
//   pass
// @accept_filename
// const gguf_load = (tensor: Tensor): [Dict, Map<string, Tensor>] => {
//   pass
