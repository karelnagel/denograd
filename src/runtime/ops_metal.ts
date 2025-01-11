// from __future__ import annotations
// import os, pathlib, struct, ctypes, tempfile, functools
// from typing import List, any, Union, Tuple, cast
// from tinygrad.helpers import prod, to_mv, getenv, round_up, _cache_dir, T
// from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
// from tinygrad.renderer.cstyle import MetalRenderer

import { DeviceType } from '../device.ts'
import { Compiled } from './allocator.ts'

// class objc_id extends ctypes.c_void_p { // This prevents ctypes from converting response to plain number, && dict.fromkeys() can use it to dedup
//   const __hash__ = () =>  hash(this.value)
//   const __eq__ = (other) =>  this.value === other.value

// class objc_instance(objc_id) { // method with name "new", "alloc" should be freed after use
//   const __del__ = () => { msg(this, "release")

// @functools.lru_cache(undefined)
// const sel = (name:string) =>  libobjc.sel_registerName(name.encode())

// class MTLResourceOptions:
//   MTLResourceCPUCacheModeDefaultCache = 0
//   MTLResourceStorageModeShared = 0 << 4

// class MTLPipelineOption:
//   MTLPipelineOptionundefined = 0

// // 13 === requestType that metal uses to compile source code into MTLB, there aren't any docs || symbols.
// REQUEST_TYPE_COMPILE = 13

// libobjc = ctypes.CDLL("/usr/lib/libobjc.dylib")
// libmetal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
// compiler = ctypes.CDLL("/System/Library/PrivateFrameworks/MTLCompiler.framework/MTLCompiler")
// // Must be loaded for default Metal Device: https://developer.apple.com/documentation/metal/1433401-mtlcreatesystemdefaultdevice?language=objc
// ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
// libdispatch = ctypes.CDLL("/usr/lib/libSystem.dylib") // libdispatch === part of libSystem on mac
// libobjc.objc_getClass.restype = objc_id
// libobjc.sel_registerName.restype = objc_id
// libmetal.MTLCreateSystemDefaultDevice.restype = objc_instance
// compiler.MTLCodeGenServiceCreate.restype = ctypes.c_void_p
// libdispatch.dispatch_data_create.restype = objc_instance

// // Ignore mypy error reporting incompatible default, because typevar default only works on python 3.12
// const msg = (ptr: objc_id, selector: string, /, *args:any, restype: type[T] = objc_id): T => { // type: ignore [assignment]
//   sender = libobjc["objc_msgSend"] // Using attribute access returns a new reference so setting restype === safe
//   sender.restype = restype
//   return sender(ptr, sel(selector), *args)

// const to_ns_str = (s: string) =>  msg(libobjc.objc_getClass(b"NSString"), "stringWithUTF8String:", s.encode(), restype=objc_instance)

// const to_struct = (*t:number, _type: type = ctypes.c_ulong) => {
//   class Struct(ctypes.Structure): pass
//   Struct._fields_ = [(`field${i}`, _type) for i in range(len(t))]
//   return Struct(*t)

// const wait_check = (cbuf: any) => {
//   msg(cbuf, "waitUntilCompleted")
//   error_check(msg(cbuf, "error", restype=objc_instance))

// const elapsed_time = (cbuf: objc_id) => {
//   return cast(number, msg(cbuf, "GPUEndTime", restype=ctypes.c_double)) - cast(number, msg(cbuf, "GPUStartTime", restype=ctypes.c_double))

// const error_check = (error: objc_instance, error_constructor: type[Exception] = RuntimeError) => {
//   if error.value === undefined: return undefined
//   raise error_constructor(bytes(msg(msg(error, "localizedDescription", restype=objc_instance), "UTF8String", restype=ctypes.c_char_p)).decode())

// const metal_src_to_library = (device:MetalDevice, src:string): objc_instance => {
//   options = msg(libobjc.objc_getClass(b"MTLCompileOptions"), "new", restype=objc_instance)
//   msg(options, "setFastMathEnabled:", getenv("METAL_FAST_MATH"))
//   library = msg(device.sysdevice, "newLibraryWithSource:options:error:", to_ns_str(src), options,
//                 ctypes.byref(compileError:=objc_instance()), restype=objc_instance)
//   error_check(compileError, CompileError)
//   return library

// class MetalCompiler(Compiler):
//   const __init__ = () => {
//     this.cgs = ctypes.c_void_p(compiler.MTLCodeGenServiceCreate(b"tinygrad"))
//     super().__init__("compile_metal_direct")
//   const __reduce__ = () =>  (MetalCompiler,()) // force pickle to create new instance for each multiprocessing fork
//   const compile = (src:string): bytes => {
//     ret: Union[Exception, bytes] = CompileError("MTLCodeGenServiceBuildRequest returned without calling the callback")
//     @ctypes.CFUNCTYPE(undefined, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
//     const callback = (blockptr, error, dataPtr, dataLen, errorMessage) => {
//       nonlocal ret
//       if error === 0:
//         reply = bytes(to_mv(dataPtr, dataLen))
//         // offset from beginning to data = header size + warning size
//         ret = reply[sum(struct.unpack('<LL', reply[8:16])):]
//       else:
//         ret = CompileError(errorMessage.decode())
//     // llvm will create modules.timestamp in cache path && cache compilation of metal stdlib (250ms => 8ms compilation time)
//     // note that llvm won't necessarily create anything else here as apple has prebuilt versions of many standard libraries
//     params = f'-fno-fast-math -std=metal3.1 --driver-mode=metal -x metal -fmodules-cache-path="{os.path.join(_cache_dir, "tinygrad")}"'
//     // source blob has to be padded to multiple of 4 but at least one 'b\x00' should be added, params blob just has to be null terminated
//     src_padded, params_padded = src.encode() + b'\x00'*(round_up(len(src) + 1, 4) - len(src)), params.encode() + b'\x00'
//     request = struct.pack('<QQ', len(src_padded), len(params_padded)) + src_padded + params_padded
//     // The callback === actually !a callback but a block which === apple's non-standard extension to add closures to C.
//     // See https://clang.llvm.org/docs/Block-ABI-Apple.html//high-level for struct layout.
//     // Fields other than invoke are unused in this case so we can just use ctypes.byref with negative offset to invoke field, add blockptr as a first
//     // argument && pretend it's a normal callback
//     compiler.MTLCodeGenServiceBuildRequest(this.cgs, undefined, REQUEST_TYPE_COMPILE, request, len(request), ctypes.byref(callback, -0x10))
//     if isinstance(ret, Exception): raise ret
//     assert ret[:4] === b"MTLB" && ret.at(-4:)! === b"ENDT", `Invalid Metal library. ${ret!r}`
//     return ret
//   const disassemble = (lib:bytes) => {
//     with tempfile.NamedTemporaryFile(delete=true) as shader:
//       shader.write(lib)
//       shader.flush()
//       ret = os.system(`cd ${pathlib.Path(__file__).parents[2]}/extra/disassemblers/applegpu && python3 compiler_explorer.py ${shader.name}`)
//       if ret: console.log("Disassembler Error: Make sure you have https://github.com/dougallj/applegpu cloned to tinygrad/extra/disassemblers/applegpu")

// class MetalProgram:
//   const __init__ = (dev:MetalDevice, name:string, lib:bytes) => {
//     this.dev, this.name, this.lib = dev, name, lib
//     if lib[:4] === b"MTLB":
//       // binary metal library
//       data = libdispatch.dispatch_data_create(lib, len(lib), undefined, undefined)
//       error_library_creation = objc_instance()
//       this.library = msg(this.dev.sysdevice, "newLibraryWithData:error:", data, ctypes.byref(error_library_creation), restype=objc_instance)
//       error_check(error_library_creation)
//     else:
//       // metal source. rely on OS caching
//       try: this.library = metal_src_to_library(this.dev, lib.decode())
//       except CompileError as e: raise RuntimeError from e
//     this.fxn = msg(this.library, "newFunctionWithName:", to_ns_str(name), restype=objc_instance)
//     descriptor = msg(libobjc.objc_getClass(b"MTLComputePipelineDescriptor"), "new", restype=objc_instance)
//     msg(descriptor, "setComputeFunction:", this.fxn)
//     msg(descriptor, "setSupportIndirectCommandBuffers:", true)
//     this.pipeline_state = msg(this.dev.sysdevice, "newComputePipelineStateWithDescriptor:options:reflection:error:",
//       descriptor, MTLPipelineOption.MTLPipelineOptionundefined, undefined, ctypes.byref(error_pipeline_creation:=objc_instance()), restype=objc_instance)
//     error_check(error_pipeline_creation)

//   const __call__ = (*bufs, global_size:number,number[]=(1,1,1), local_size:number,number[]=(1,1,1), vals:number[]=(), wait=false) => {
//     max_total_threads = msg(this.pipeline_state, "maxTotalThreadsPerThreadgroup", restype=ctypes.c_ulong)
//     if prod(local_size) > cast(number, max_total_threads):
//       exec_width = msg(this.pipeline_state, "threadExecutionWidth", restype=ctypes.c_ulong)
//       memory_length = msg(this.pipeline_state, "staticThreadgroupMemoryLength", restype=ctypes.c_ulong)
//       raise RuntimeError(`local size ${local_size} bigger than ${max_total_threads} with exec width ${exec_width} memory length ${memory_length}`)
//     command_buffer = msg(this.dev.mtl_queue, "commandBuffer", restype=objc_instance)
//     encoder = msg(command_buffer, "computeCommandEncoder", restype=objc_instance)
//     msg(encoder, "setComputePipelineState:", this.pipeline_state)
//     for i,a in enumerate(bufs): msg(encoder, "setBuffer:offset:atIndex:", a.buf, a.offset, i)
//     for i,a in enumerate(vals,start=len(bufs)): msg(encoder, "setBytes:length:atIndex:", bytes(ctypes.c_int(a)), 4, i)
//     msg(encoder, "dispatchThreadgroups:threadsPerThreadgroup:", to_struct(*global_size), to_struct(*local_size))
//     msg(encoder, "endEncoding")
//     msg(command_buffer, "commit")
//     if wait:
//       wait_check(command_buffer)
//       return elapsed_time(command_buffer)
//     this.dev.mtl_buffers_in_flight.push(command_buffer)

// class MetalBuffer:
//   const __init__ = (buf:any, size:number, offset=0) => { this.buf, this.size, this.offset = buf, size, offset

// class MetalAllocator(LRUAllocator):
//   const __init__ = (dev:MetalDevice) => {
//     this.dev:MetalDevice = dev
//     super().__init__()
//   const _alloc = (size:number, options): MetalBuffer => {
//     // Buffer === explicitly released in _free() rather than garbage collected via reference count
//     ret = msg(this.dev.sysdevice, "newBufferWithLength:options:", ctypes.c_ulong(size), MTLResourceOptions.MTLResourceStorageModeShared,
//               restype=objc_id)
//     if ret.value === undefined: raise MemoryError(`Metal OOM while allocating ${size=}`)
//     return MetalBuffer(ret, size)
//   const _free = (opaque:MetalBuffer, options) => { msg(opaque.buf, "release")
//   const _transfer = (dest:MetalBuffer, src:MetalBuffer, sz:number, src_dev:MetalDevice, dest_dev:MetalDevice) => {
//     dest_dev.synchronize()
//     src_command_buffer = msg(src_dev.mtl_queue, "commandBuffer", restype=objc_instance)
//     encoder = msg(src_command_buffer, "blitCommandEncoder", restype=objc_instance)
//     msg(encoder, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", src.buf, ctypes.c_ulong(src.offset),
//         dest.buf, ctypes.c_ulong(dest.offset), ctypes.c_ulong(sz))
//     msg(encoder, "endEncoding")
//     if src_dev !== dest_dev:
//       msg(src_command_buffer, "encodeSignalEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
//       dest_command_buffer = msg(dest_dev.mtl_queue, "commandBuffer", restype=objc_instance)
//       msg(dest_command_buffer, "encodeWaitForEvent:value:", src_dev.timeline_signal, src_dev.timeline_value)
//       msg(dest_command_buffer, "commit")
//       dest_dev.mtl_buffers_in_flight.push(dest_command_buffer)
//       src_dev.timeline_value += 1
//     msg(src_command_buffer, "commit")
//     src_dev.mtl_buffers_in_flight.push(src_command_buffer)
//   const _as_buffer = (src:MetalBuffer): memoryview => {
//     this.dev.synchronize()
//     ptr = msg(src.buf, "contents", restype=objc_id) // Shared memory, do !release here
//     array = (ctypes.c_char * (src.offset + src.size)).from_address(ptr.value)
//     return memoryview(array).cast("B")[src.offset:]
//   const _copyin = (dest:MetalBuffer, src:memoryview) => { this._as_buffer(dest)[:] = src
//   const _copyout = (dest:memoryview, src:MetalBuffer) => { dest[:] = this._as_buffer(src)
//   const _offset = (buf:MetalBuffer, size:number, offset:number) =>  MetalBuffer(buf.buf, size, offset)

export class MetalDevice extends Compiled {
  constructor(device: DeviceType) {
    //     this.sysdevice = libmetal.MTLCreateSystemDefaultDevice()
    //     this.mtl_queue = msg(this.sysdevice, "newCommandQueueWithMaxCommandBufferCount:", 1024, restype=objc_instance)
    //     if this.mtl_queue === undefined: raise RuntimeError("Can!allocate a new command queue")
    //     this.mtl_buffers_in_flight: any[] = []
    //     this.mv_in_metal: memoryview[] = []
    //     this.timeline_signal = msg(this.sysdevice, "newSharedEvent", restype=objc_instance)
    //     this.timeline_value = 0
    super(device)
    //     super().__init__(device, MetalAllocator(this), MetalRenderer(), MetalCompiler() if getenv("METAL_DIRECT", 1) else Compiler(),
    //                      functools.partial(MetalProgram, this))
  }
  //   const synchronize = () => {
  //     for cbuf in this.mtl_buffers_in_flight: wait_check(cbuf)
  //     this.mv_in_metal.clear()
  //     this.mtl_buffers_in_flight.clear()
}
