import type * as _webgpu from 'npm:@webgpu/types@0.1.54'
import { bytes_to_string, isInt, range, round_up } from '../helpers.ts'
import { Allocator, type BufferSpec, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import type { MemoryView } from '../memoryview.ts'
import * as c from '../../ctypes/dawn.ts'

const desc = new c.InstanceDescriptor()
desc.$features.$timedWaitAnyEnable.set(1)

const instance = c.createInstance(desc.ptr())
if (!instance.value) throw new Error(`Failed creating instance!`)
const wgpu_wait = (future: c.Future) => {
  const res = c.instanceWaitAny(instance, c.Size.new(1n), c.FutureWaitInfo.new({ future }).ptr(), c.U64.new(2n ** 64n - 1n))
  if (res.value !== c.WaitStatus.Success.value) throw new Error('Future failed')
}
const from_wgpu_str = (_str: c.StringView): string => {
  if (_str.$length.value <= 1) return ''
  const buf = Deno.UnsafePointerView.getArrayBuffer(_str.$data.native as Deno.PointerObject, Number(_str.$length.value))
  console.log(buf)
  return new TextDecoder().decode(buf)
}
const to_str = (str: string) => {
  const data = new TextEncoder().encode(str)
  return new c.Type(data.buffer as ArrayBuffer, 0, data.length, 8)
}

const write_buffer = (device: c.Device, buf: c.Buffer, offset: number, src: Uint8Array) => {
  c.queueWriteBuffer(c.deviceGetQueue(device), buf, c.U64.new(BigInt(offset)), new c.Pointer().set(Deno.UnsafePointer.value(Deno.UnsafePointer.of(src))), c.Size.new(BigInt(src.length)))
}

const map_buffer = (buf: c.Buffer, size: c.U64) => {
  let result: any[] | undefined

  const cb_info = new c.BufferMapCallbackInfo2()
  cb_info.$mode.set(c.CallbackMode.WaitAnyOnly.value)
  cb_info.$callback.set((status, msg, u1, u2) => {
    result = [status, from_wgpu_str(msg)]
  })
  wgpu_wait(c.bufferMapAsync2(buf, c.MapMode.new(BigInt(c.MapMode_Read.value)), c.Size.new(0n), size, cb_info))

  if (!result || result[0].value !== c.BufferMapAsyncStatus.Success.value) {
    throw new Error(`Failed to map buffer: ${result![0]} ${result![1]}`)
  }
}

const copy_buffer_to_buffer = (dev: c.Device, src: c.Buffer, src_offset: number, dst: c.Buffer, dst_offset: number, size: c.U64) => {
  const encoder = c.deviceCreateCommandEncoder(dev, new c.CommandEncoderDescriptor().ptr())
  c.commandEncoderCopyBufferToBuffer(encoder, src, c.U64.new(BigInt(src_offset)), dst, c.U64.new(BigInt(dst_offset)), size)
  const cb = c.commandEncoderFinish(encoder, new c.CommandBufferDescriptor().ptr())
  c.queueSubmit(c.deviceGetQueue(dev), c.Size.new(BigInt(1)), cb)
  c.commandBufferRelease(cb)
  c.commandEncoderRelease(encoder)
}
const read_buffer = (dev: c.Device, buf: c.Buffer) => {
  const size = c.bufferGetSize(buf)
  const desc = c.BufferDescriptor.new({
    size: c.U64.new(size.value),
    usage: c.BufferUsage.new(BigInt(c.BufferUsage_CopyDst.value) | BigInt(c.BufferUsage_MapRead.value)),
    mappedAtCreation: c.Bool.new(0),
  })
  const tmp_buffer = c.deviceCreateBuffer(dev, desc.ptr())
  copy_buffer_to_buffer(dev, buf, 0, tmp_buffer, 0, size)
  map_buffer(tmp_buffer, size)
  const void_ptr = c.bufferGetConstMappedRange(tmp_buffer, c.Size.new(0n), size)
  const buf_copy = new Uint8Array(Deno.UnsafePointerView.getArrayBuffer(void_ptr.native as Deno.PointerObject, Number(size.value)))
  c.bufferUnmap(tmp_buffer)
  c.bufferDestroy(tmp_buffer)
  return buf_copy
}

const pop_error = (device: c.Device) => {
  let result = ''

  const cb_info = new c.PopErrorScopeCallbackInfo()
  cb_info.$mode.set(c.CallbackMode.WaitAnyOnly.value)
  cb_info.$callback.set((status, err_type, msg, i2) => {
    console.log(status, err_type, msg, i2)
    result = from_wgpu_str(msg)
  })
  wgpu_wait(c.devicePopErrorScopeF(device, cb_info))
  return result
}
const create_uniform = (wgpu_device: c.Device, val: number) => {
  const desc = c.BufferDescriptor.new({ size: c.U64.new(4n), usage: c.BufferUsage.new(BigInt(c.BufferUsage_Uniform.value) | BigInt(c.BufferUsage_CopyDst.value)) })
  const buf = c.deviceCreateBuffer(wgpu_device, desc.ptr())
  const bytes = new Uint8Array(4)
  if (isInt(val)) new DataView(bytes.buffer).setInt32(0, val, true)
  else new DataView(bytes.buffer).setFloat32(0, val, true)
  write_buffer(wgpu_device, buf, 0, bytes)
  return buf
}
class WebGPUProgram extends Program {
  prg!: c.ShaderModule
  static override init = (name: string, lib: Uint8Array) => {
    const res = new WebGPUProgram(name, lib)
    const code = bytes_to_string(res.lib)

    // Creating shader module
    const str = to_str(code)
    const shader = c.ShaderModuleWGSLDescriptor.new({
      code: c.StringView.new({ data: str.ptr(), length: c.Size.new(BigInt(str.byteLength)) }),
      chain: c.ChainedStruct.new({ sType: c.SType.ShaderSourceWGSL }),
    })
    const module = new c.ShaderModuleDescriptor()
    module.$nextInChain.set(shader.ptr().value)
    // Check compiler error
    c.devicePushErrorScope(WEBGPU.device, c.ErrorFilter.Validation)
    const shader_module = c.deviceCreateShaderModule(WEBGPU.device, module.ptr())
    const err = pop_error(WEBGPU.device)
    if (err) throw new Error(`Shader compilation failed: ${err}`)
      console.log(shader_module)
    res.prg = shader_module
    return res
  }
  override call = async (bufs: c.Buffer[], { global_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    let tmp_bufs = [...bufs]
    let buf_patch = false

    //   # WebGPU does not allow using the same buffer for input and output
    for (const i of range(1, bufs.length)) {
      if (bufs[i].value === bufs[0].value) {
        tmp_bufs[0] = c.deviceCreateBuffer(
          WEBGPU.device,
          c.BufferDescriptor.new({
            size: c.bufferGetSize(bufs[0]),
            usage: c.bufferGetUsage(bufs[0]),
          }).ptr(),
        )
        buf_patch = true
      }
    }

    // Creating bind group layout
    let binding_layouts = [c.BindGroupLayoutEntry.new({
      binding: c.U32.new(0),
      visibility: c.ShaderStage.new(BigInt(c.ShaderStage_Compute.value)),
      buffer: c.BufferBindingLayout.new({ type: c.BufferBindingType.Uniform }),
    })]
    for (const i of range(tmp_bufs.length + vals.length)) {
      binding_layouts.push(
        c.BindGroupLayoutEntry.new({
          binding: c.U32.new(i + 1),
          visibility: c.ShaderStage.new(BigInt(c.ShaderStage_Compute.value)),
          buffer: c.BufferBindingLayout.new({ type: i >= tmp_bufs.length ? c.BufferBindingType.Uniform : c.BufferBindingType.Storage }),
        }),
      )
    }

    const bl_arr = c.createArray(binding_layouts)
    c.devicePushErrorScope(WEBGPU.device, c.ErrorFilter.Validation)
    const bind_group_layouts = [c.deviceCreateBindGroupLayout(
      WEBGPU.device,
      c.BindGroupLayoutDescriptor.new({
        entryCount: c.Size.new(BigInt(binding_layouts.length)),
        entries: bl_arr.ptr(),
      }).ptr(),
    )]
    const bg_layout_err = pop_error(WEBGPU.device)
    if (bg_layout_err) throw new Error(`Error creating bind group layout: ${bg_layout_err}`)

    // Creating pipeline layout
    const bindGroupLayouts = c.createArray(bind_group_layouts)
    const pipeline_layout_desc = c.PipelineLayoutDescriptor.new({
      bindGroupLayoutCount: c.Size.new(BigInt(bind_group_layouts.length)),
      bindGroupLayouts: bindGroupLayouts.ptr(),
    })

    c.devicePushErrorScope(WEBGPU.device, c.ErrorFilter.Validation)
    const pipeline_layout = c.deviceCreatePipelineLayout(WEBGPU.device, pipeline_layout_desc.ptr())

    const pipe_err = pop_error(WEBGPU.device)
    if (pipe_err) throw new Error(`Error creating pipeline layout: ${pipe_err}`)

    // Creating bind group
    const bindings = [c.BindGroupEntry.new({ binding: c.U32.new(0), buffer: create_uniform(WEBGPU.device, Infinity), offset: c.U64.new(0n), size: c.U64.new(4n) })]
    for (const [i, x] of [...tmp_bufs, ...vals].entries()) {
      bindings.push(
        c.BindGroupEntry.new({
          binding: c.U32.new(i + 1),
          buffer: i >= tmp_bufs.length ? create_uniform(WEBGPU.device, x as number) : x as c.Buffer,
          offset: c.U64.new(0n),
          size: i >= tmp_bufs.length ? c.U64.new(4n) : c.bufferGetSize(x as c.Buffer),
        }),
      )
    }

    const bg_arr = c.createArray(bindings)
    const bind_group_desc = c.BindGroupDescriptor.new({ layout: bind_group_layouts[0], entryCount: c.Size.new(BigInt(bindings.length)), entries: bg_arr.ptr() })
    c.devicePushErrorScope(WEBGPU.device, c.ErrorFilter.Validation)
    const bind_group = c.deviceCreateBindGroup(WEBGPU.device, bind_group_desc.ptr())

    const bind_err = pop_error(WEBGPU.device)
    if (bind_err) throw new Error(`Error creating bind group: ${bind_err}`)

    // Creating compute pipeline
    const str = to_str(this.name)
    const compute_desc = c.ComputePipelineDescriptor.new({
      layout: pipeline_layout,
      compute: c.ComputeState.new({ module: this.prg, entryPoint: c.StringView.new({ data: str.ptr(), length: c.Size.new(BigInt(str.byteLength)) }) }),
    })
    let pipeline_result: [c.CreatePipelineAsyncStatus, c.ComputePipeline, string] | undefined

    // def cb(status, compute_pipeline_impl, msg, u1, u2): pipeline_result[:] = status, compute_pipeline_impl, from_wgpu_str(msg)
    const cb_info = new c.CreateComputePipelineAsyncCallbackInfo2()
    cb_info.$mode.set(c.CallbackMode.WaitAnyOnly.value)
    cb_info.$callback.set((status, compute_pipline_impl, msg, u1, u2) => {
      pipeline_result = [status, compute_pipline_impl, from_wgpu_str(msg)]
    })
    c.devicePushErrorScope(WEBGPU.device, c.ErrorFilter.Validation)
    wgpu_wait(c.deviceCreateComputePipelineAsync2(WEBGPU.device, compute_desc.ptr(), cb_info))

    if (!pipeline_result || pipeline_result[0].value !== c.CreatePipelineAsyncStatus.Success.value) {
      throw new Error(`${pipeline_result![0]}: ${pipeline_result![2]}, ${pop_error(WEBGPU.device)}`)
    }
    const command_encoder = c.deviceCreateCommandEncoder(WEBGPU.device, new c.CommandEncoderDescriptor().ptr())
    const comp_pass_desc = new c.ComputePassDescriptor()

    let query_set: c.QuerySet, query_buf: c.Buffer
    if (wait) {
      query_set = c.deviceCreateQuerySet(WEBGPU.device, c.QuerySetDescriptor.new({ type: c.QueryType.Timestamp, count: c.U32.new(2) }).ptr())
      query_buf = c.deviceCreateBuffer(
        WEBGPU.device,
        c.BufferDescriptor.new({
          size: c.U64.new(16n),
          usage: c.BufferUsage.new(BigInt(c.BufferUsage_QueryResolve.value) | BigInt(c.BufferUsage_CopySrc.value)),
        }).ptr(),
      )
      comp_pass_desc.$timestampWrites.set(
        c.ComputePassTimestampWrites.new({
          querySet: query_set,
          beginningOfPassWriteIndex: c.U32.new(0),
          endOfPassWriteIndex: c.U32.new(1),
        }).ptr().value,
      )
    }
    // Begin compute pass
    const compute_pass = c.commandEncoderBeginComputePass(command_encoder, comp_pass_desc.ptr())
    c.computePassEncoderSetPipeline(compute_pass, pipeline_result[1])
    c.computePassEncoderSetBindGroup(compute_pass, c.U32.new(0), bind_group, c.Size.new(0n), new c.Pointer())
    c.computePassEncoderDispatchWorkgroups(compute_pass, c.U32.new(global_size[0]), c.U32.new(global_size[1]), c.U32.new(global_size[2]))
    c.computePassEncoderEnd(compute_pass)

    if (wait) c.commandEncoderResolveQuerySet(command_encoder, query_set!, c.U32.new(0), c.U32.new(2), query_buf!, c.U64.new(0n))

    const cmd_buf = c.commandEncoderFinish(command_encoder, new c.CommandBufferDescriptor().ptr())
    c.queueSubmit(c.deviceGetQueue(WEBGPU.device), c.Size.new(1n), cmd_buf.ptr())

    if (buf_patch) {
      copy_buffer_to_buffer(WEBGPU.device, tmp_bufs[0], 0, bufs[0], 0, c.bufferGetSize(bufs[0]))
    }
    c.bufferDestroy(tmp_bufs[0])

    if (wait) {
      const timestamps = new BigUint64Array(read_buffer(WEBGPU.device, query_buf!).buffer)
      const time = (timestamps[1] - timestamps[0]) / BigInt(1e9)
      c.bufferDestroy(query_buf!)
      c.querySetDestroy(query_set!)
      return Number(time)
    }
  }
}

class WebGpuAllocator extends Allocator<c.Buffer> {
  _alloc = (size: number, options?: BufferSpec) => {
    // WebGPU buffers have to be 4-byte aligned
    const desc = c.BufferDescriptor.new({
      size: c.U64.new(BigInt(round_up(size, 4))),
      usage: c.BufferUsage.new(BigInt(c.BufferUsage_Storage.value) | BigInt(c.BufferUsage_CopyDst.value) | BigInt(c.BufferUsage_CopySrc.value)),
    })
    return c.deviceCreateBuffer(WEBGPU.device, desc.ptr())
  }
  _copyin = (dest: c.Buffer, src: MemoryView) => {
    if (src.byteLength % 4) {
      const padded_src = new Uint8Array(round_up(src.byteLength, 4))
      padded_src.set(src.bytes)
      write_buffer(WEBGPU.device, dest, 0, padded_src)
    } else write_buffer(WEBGPU.device, dest, 0, src.bytes)
  }
  _copyout = async (dest: MemoryView, src: c.Buffer) => {
    const buffer_data = read_buffer(WEBGPU.device, src)
    dest.set(buffer_data.slice(0, dest.byteLength))
  }
  _free = (opaque: c.Buffer, options?: BufferSpec) => {
    c.bufferDestroy(opaque)
  }
}

export class WEBGPU extends Compiled {
  static device: c.Device
  constructor(device: string) {
    super(device, new WebGpuAllocator(), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
  override init = async () => {
    if (WEBGPU.device) return
    let adapter_result: [c.RequestAdapterStatus, c.Adapter, string] | undefined

    const cb_info = new c.RequestAdapterCallbackInfo()
    cb_info.$mode.set(c.CallbackMode.WaitAnyOnly.value)
    cb_info.$callback.set((status, adapter, msg) => {
      adapter_result = [status, adapter, from_wgpu_str(msg)]
    })
    wgpu_wait(c.instanceRequestAdapterF(instance, c.RequestAdapterOptions.new({ powerPreference: c.PowerPreference.HighPerformance }).ptr(), cb_info))

    if (!adapter_result || adapter_result![0].value !== c.RequestAdapterStatus.Success.value) {
      throw new Error(`Error requesting adapter: [${adapter_result![0].value}] ${adapter_result![2]}`)
    }
    console.log(`Adapter: ${adapter_result[1]}`)

    const supported_features = new c.SupportedFeatures()
    c.adapterGetFeatures(adapter_result![1], supported_features.ptr())
    const supported: c.FeatureName[] = []
    for (let i = 0n; i < supported_features.$featureCount.value; i++) {
      supported.push(new c.FeatureName().loadFromPtr(c.Pointer.new(supported_features.$features.value + i)))
    }
    const features = [c.FeatureName.TimestampQuery, c.FeatureName.ShaderF16].filter((feat) => supported.some((s) => s.value === feat.value))

    // TODO: do array correcly
    const featArr = c.createArray(features)
    const dev_desc = c.DeviceDescriptor.new({ requiredFeatureCount: c.Size.new(BigInt(features.length)), requiredFeatures: featArr.ptr() })

    const supported_limits = new c.SupportedLimits()
    c.adapterGetLimits(adapter_result![1], supported_limits.ptr())
    const limits = c.RequiredLimits.new({ limits: supported_limits.$limits })
    dev_desc.$requiredLimits.set(limits.ptr().value)

    // Requesting a device
    let device_result: [c.RequestDeviceStatus, c.Device, string] | undefined

    const cb_info2 = new c.RequestDeviceCallbackInfo()
    cb_info2.$mode.set(c.CallbackMode.WaitAnyOnly.value)
    cb_info2.$callback.set((status, device_impl, msg, _) => {
      device_result = [status, device_impl, from_wgpu_str(msg)]
    })
    wgpu_wait(c.adapterRequestDeviceF(adapter_result![1], dev_desc.ptr(), cb_info2))

    if (!device_result || device_result[0].value !== c.RequestDeviceStatus.Success.value) {
      throw new Error(`Failed to request device: [${device_result![0].value}] ${device_result![2]}`)
    }
    WEBGPU.device = device_result[1]
    console.log(`Device: ${WEBGPU.device}`)
  }
  override synchronize = () => {
    // result: List[Any] = []
    // def cb(status, u1, u2): result[:] = [status]
    // cb_info = create_cb_info(webgpu.WGPUQueueWorkDoneCallbackInfo2, webgpu.WGPUQueueWorkDoneCallback2, cb)
    // wgpu_wait(webgpu.wgpuQueueOnSubmittedWorkDone2(webgpu.wgpuDeviceGetQueue(this.runtime.args[0][0]), cb_info))
    // if result[0] != webgpu.WGPUQueueWorkDoneStatus_Success: raise RuntimeError(webgpu.WGPUQueueWorkDoneStatus__enumvalues[result[0]])
    throw new Error()
  }
}
