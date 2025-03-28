import type * as _webgpu from '@webgpu/types'
import { bytes_to_string, isInt, range, round_up } from '../helpers/helpers.ts'
import { Allocator, type BufferSpec, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import type { MemoryView } from '../helpers/memoryview.ts'
import * as c from '../../dawn/bindings.ts'
import { env } from '../env/index.ts'

const _wait = (future: c.Future) => {
  const res = c.instanceWaitAny(DAWN.instance, c.Size.new(1n), c.FutureWaitInfo.new({ future }).ptr(), c.U64.new(2n ** 64n - 1n))
  if (res.value !== c.WaitStatus.Success.value) throw new Error('Future failed')
}
const from_wgpu_str = (_str: c.StringView): string => {
  if (_str.$length.value <= 1) return ''
  const buf = env.getArrayBuffer(_str.$data.native, Number(_str.$length.value))
  return new TextDecoder().decode(buf)
}
const to_wgpu_str = (str: string) => {
  const data = new TextEncoder().encode(str)
  const _str = new c.Type(data.buffer as ArrayBuffer, 0, data.length, 8)
  return c.StringView.new({ data: _str.ptr(), length: c.Size.new(BigInt(_str.byteLength)) })
}

const write_buffer = (device: c.Device, buf: c.Buffer, offset: number, src: Uint8Array) => {
  c.queueWriteBuffer(c.deviceGetQueue(device), buf, c.U64.new(BigInt(offset)), new c.Pointer().setNative(env.ptr(src.buffer as ArrayBuffer)), c.Size.new(BigInt(src.length)))
}

type CallBack = typeof c.BufferMapCallbackInfo2 | typeof c.PopErrorScopeCallbackInfo | typeof c.CreateComputePipelineAsyncCallbackInfo2 | typeof c.RequestAdapterCallbackInfo | typeof c.RequestDeviceCallbackInfo | typeof c.QueueWorkDoneCallbackInfo2
const _run = async <T extends CallBack>(cb_class: T, async_fn: (cb: InstanceType<T>) => c.Future): Promise<Parameters<Parameters<InstanceType<T>['$callback']['set']>[0]>> => {
  return await new Promise((resolve) => {
    const cb = new cb_class()
    cb.$mode.set(c.CallbackMode.WaitAnyOnly.value)
    cb.$callback.set((...args) => resolve(args as any))
    _wait(async_fn(cb as any))
  })
}

const copy_buffer_to_buffer = (dev: c.Device, src: c.Buffer, src_offset: number, dst: c.Buffer, dst_offset: number, size: c.U64) => {
  const encoder = c.deviceCreateCommandEncoder(dev, new c.CommandEncoderDescriptor().ptr())
  c.commandEncoderCopyBufferToBuffer(encoder, src, c.U64.new(BigInt(src_offset)), dst, c.U64.new(BigInt(dst_offset)), size)
  const cb = c.commandEncoderFinish(encoder, new c.CommandBufferDescriptor().ptr())
  c.queueSubmit(c.deviceGetQueue(dev), c.Size.new(1n), cb.ptr())
  c.commandBufferRelease(cb)
  c.commandEncoderRelease(encoder)
}
const read_buffer = async (dev: c.Device, buf: c.Buffer) => {
  const size = c.bufferGetSize(buf)
  const desc = c.BufferDescriptor.new({
    size: c.U64.new(size.value),
    usage: c.BufferUsage.new(c.BufferUsage_CopyDst.value | c.BufferUsage_MapRead.value),
    mappedAtCreation: c.Bool.new(0),
  })
  const tmp_buffer = c.deviceCreateBuffer(dev, desc.ptr())
  copy_buffer_to_buffer(dev, buf, 0, tmp_buffer, 0, size)

  const [status, msg] = await _run(c.BufferMapCallbackInfo2, (cb) => c.bufferMapAsync2(tmp_buffer, c.MapMode.new(c.MapMode_Read.value), c.Size.new(0n), size, cb))
  if (status.value !== c.BufferMapAsyncStatus.Success.value) throw new Error(`Async failed: ${from_wgpu_str(msg)}`)

  const void_ptr = c.bufferGetConstMappedRange(tmp_buffer, c.Size.new(0n), size)
  const buf_copy = new c.Type(new ArrayBuffer(Number(size.value)), 0, Number(size.value)).replaceWithPtr(void_ptr)
  c.bufferUnmap(tmp_buffer)
  c.bufferDestroy(tmp_buffer)
  return buf_copy.bytes
}

const pop_error = async (device: c.Device) => {
  const [_, __, msg] = await _run(c.PopErrorScopeCallbackInfo, (cb) => c.devicePopErrorScopeF(device, cb))
  return from_wgpu_str(msg)
}

const create_uniform = (wgpu_device: c.Device, val: number) => {
  const desc = c.BufferDescriptor.new({ size: c.U64.new(4n), usage: c.BufferUsage.new(c.BufferUsage_Uniform.value | c.BufferUsage_CopyDst.value) })
  const buf = c.deviceCreateBuffer(wgpu_device, desc.ptr())
  const bytes = new Uint8Array(4)
  if (isInt(val)) new DataView(bytes.buffer).setInt32(0, val, true)
  else new DataView(bytes.buffer).setFloat32(0, val, true)
  write_buffer(wgpu_device, buf, 0, bytes)
  return buf
}
class WebGPUProgram extends Program {
  prg!: c.ShaderModule
  static override init = async (name: string, lib: Uint8Array) => {
    const res = new WebGPUProgram(name, lib)
    const code = bytes_to_string(res.lib)

    // Creating shader module
    const shader = c.ShaderModuleWGSLDescriptor.new({
      code: to_wgpu_str(code),
      chain: c.ChainedStruct.new({ sType: c.SType.ShaderSourceWGSL }),
    })
    const module = new c.ShaderModuleDescriptor()
    module.$nextInChain.set(shader.ptr().value)
    // Check compiler error
    c.devicePushErrorScope(DAWN.device, c.ErrorFilter.Validation)
    const shader_module = c.deviceCreateShaderModule(DAWN.device, module.ptr())
    const err = await pop_error(DAWN.device)
    if (err) throw new Error(`Shader compilation failed: ${err}`)
    res.prg = shader_module
    return res
  }
  override call = async (bufs: c.Buffer[], { global_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    let tmp_bufs = [...bufs]
    let buf_patch = false

    // WebGPU does not allow using the same buffer for input and output
    for (const i of range(1, bufs.length)) {
      if (bufs[i].value === bufs[0].value) {
        tmp_bufs[0] = c.deviceCreateBuffer(
          DAWN.device,
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
      visibility: c.ShaderStage.new(c.ShaderStage_Compute.value),
      buffer: c.BufferBindingLayout.new({ type: c.BufferBindingType.Uniform }),
    })]
    for (const i of range(tmp_bufs.length + vals.length)) {
      binding_layouts.push(
        c.BindGroupLayoutEntry.new({
          binding: c.U32.new(i + 1),
          visibility: c.ShaderStage.new(c.ShaderStage_Compute.value),
          buffer: c.BufferBindingLayout.new({ type: i >= tmp_bufs.length ? c.BufferBindingType.Uniform : c.BufferBindingType.Storage }),
        }),
      )
    }

    c.devicePushErrorScope(DAWN.device, c.ErrorFilter.Validation)
    const bind_group_layouts = [c.deviceCreateBindGroupLayout(
      DAWN.device,
      c.BindGroupLayoutDescriptor.new({
        entryCount: c.Size.new(BigInt(binding_layouts.length)),
        entries: c.createArray(binding_layouts).ptr(),
      }).ptr(),
    )]
    const bg_layout_err = await pop_error(DAWN.device)
    if (bg_layout_err) throw new Error(`Error creating bind group layout: ${bg_layout_err}`)

    // Creating pipeline layout
    const pipeline_layout_desc = c.PipelineLayoutDescriptor.new({
      bindGroupLayoutCount: c.Size.new(BigInt(bind_group_layouts.length)),
      bindGroupLayouts: c.createArray(bind_group_layouts).ptr(),
    })

    c.devicePushErrorScope(DAWN.device, c.ErrorFilter.Validation)
    const pipeline_layout = c.deviceCreatePipelineLayout(DAWN.device, pipeline_layout_desc.ptr())
    const pipe_err = await pop_error(DAWN.device)
    if (pipe_err) throw new Error(`Error creating pipeline layout: ${pipe_err}`)

    // Creating bind group
    const bindings = [c.BindGroupEntry.new({ binding: c.U32.new(0), buffer: create_uniform(DAWN.device, Infinity), offset: c.U64.new(0n), size: c.U64.new(4n) })]
    for (const [i, x] of [...tmp_bufs, ...vals].entries()) {
      bindings.push(
        c.BindGroupEntry.new({
          binding: c.U32.new(i + 1),
          buffer: typeof x === 'number' ? create_uniform(DAWN.device, x) : x,
          offset: c.U64.new(0n),
          size: typeof x === 'number' ? c.U64.new(4n) : c.bufferGetSize(x),
        }),
      )
    }

    const bind_group_desc = c.BindGroupDescriptor.new({ layout: bind_group_layouts[0], entryCount: c.Size.new(BigInt(bindings.length)), entries: c.createArray(bindings).ptr() })
    c.devicePushErrorScope(DAWN.device, c.ErrorFilter.Validation)
    const bind_group = c.deviceCreateBindGroup(DAWN.device, bind_group_desc.ptr())

    const bind_err = await pop_error(DAWN.device)
    if (bind_err) throw new Error(`Error creating bind group: ${bind_err}`)

    // Creating compute pipeline
    const compute_desc = c.ComputePipelineDescriptor.new({
      layout: pipeline_layout,
      compute: c.ComputeState.new({ module: this.prg, entryPoint: to_wgpu_str(this.name) }),
    })
    c.devicePushErrorScope(DAWN.device, c.ErrorFilter.Validation)
    const [status, compute_pipeline, msg] = await _run(c.CreateComputePipelineAsyncCallbackInfo2, (cb) => c.deviceCreateComputePipelineAsync2(DAWN.device, compute_desc.ptr(), cb))
    if (status.value !== c.CreatePipelineAsyncStatus.Success.value) throw new Error(`${status}: ${from_wgpu_str(msg)}, ${await pop_error(DAWN.device)}`)

    const command_encoder = c.deviceCreateCommandEncoder(DAWN.device, new c.CommandEncoderDescriptor().ptr())
    const comp_pass_desc = new c.ComputePassDescriptor()

    let query_set: c.QuerySet, query_buf: c.Buffer
    if (wait) {
      query_set = c.deviceCreateQuerySet(DAWN.device, c.QuerySetDescriptor.new({ type: c.QueryType.Timestamp, count: c.U32.new(2) }).ptr())
      query_buf = c.deviceCreateBuffer(
        DAWN.device,
        c.BufferDescriptor.new({
          size: c.U64.new(16n),
          usage: c.BufferUsage.new(c.BufferUsage_QueryResolve.value | c.BufferUsage_CopySrc.value),
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
    c.computePassEncoderSetPipeline(compute_pass, compute_pipeline)
    c.computePassEncoderSetBindGroup(compute_pass, c.U32.new(0), bind_group, c.Size.new(0n), new c.Pointer())
    c.computePassEncoderDispatchWorkgroups(compute_pass, c.U32.new(global_size[0]), c.U32.new(global_size[1]), c.U32.new(global_size[2]))
    c.computePassEncoderEnd(compute_pass)

    if (wait) c.commandEncoderResolveQuerySet(command_encoder, query_set!, c.U32.new(0), c.U32.new(2), query_buf!, c.U64.new(0n))

    const cmd_buf = c.commandEncoderFinish(command_encoder, new c.CommandBufferDescriptor().ptr())
    c.queueSubmit(c.deviceGetQueue(DAWN.device), c.Size.new(1n), cmd_buf.ptr())

    if (buf_patch) {
      copy_buffer_to_buffer(DAWN.device, tmp_bufs[0], 0, bufs[0], 0, c.bufferGetSize(bufs[0]))
      c.bufferDestroy(tmp_bufs[0])
    }

    if (wait) {
      const timestamps = new BigUint64Array((await read_buffer(DAWN.device, query_buf!)).buffer)
      const time = Number(timestamps[1] - timestamps[0]) / 1e9
      c.bufferDestroy(query_buf!)
      c.querySetDestroy(query_set!)
      return time
    }
  }
}

class WebGpuAllocator extends Allocator<c.Buffer> {
  _alloc = (size: number, options?: BufferSpec) => {
    // WebGPU buffers have to be 4-byte aligned
    const desc = c.BufferDescriptor.new({
      size: c.U64.new(BigInt(round_up(size, 4))),
      usage: c.BufferUsage.new(c.BufferUsage_Storage.value | c.BufferUsage_CopyDst.value | c.BufferUsage_CopySrc.value),
    })
    return c.deviceCreateBuffer(DAWN.device, desc.ptr())
  }
  _copyin = (dest: c.Buffer, src: MemoryView) => {
    if (src.byteLength % 4) {
      const padded_src = new Uint8Array(round_up(src.byteLength, 4))
      padded_src.set(src.bytes)
      write_buffer(DAWN.device, dest, 0, padded_src)
    } else write_buffer(DAWN.device, dest, 0, src.bytes)
  }
  _copyout = async (dest: MemoryView, src: c.Buffer) => {
    const buffer_data = await read_buffer(DAWN.device, src)
    dest.set(buffer_data.slice(0, dest.byteLength))
  }
  _free = (opaque: c.Buffer, options?: BufferSpec) => {
    c.bufferDestroy(opaque)
  }
}

export class DAWN extends Compiled {
  static device: c.Device
  static instance: c.Instance
  constructor(device: string) {
    super(device, new WebGpuAllocator(), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
  override init = async () => {
    if (DAWN.device) return
    const FILE = env.OSX ? 'libwebgpu_dawn.dylib' : 'libwebgpu_dawn.so'
    const URL = `https://github.com/wpmed92/pydawn/releases/download/v0.1.6/${FILE}`
    const PATH = `${env.CACHE_DIR}/${FILE}`

    await env.fetchSave(URL, PATH)
    await c.init(PATH)

    const desc = new c.InstanceDescriptor()
    desc.$features.$timedWaitAnyEnable.set(1)

    DAWN.instance = c.createInstance(desc.ptr())
    if (!DAWN.instance.value) throw new Error(`Failed creating instance!`)

    const [status, adapter, msg] = await _run(c.RequestAdapterCallbackInfo, (cb) => c.instanceRequestAdapterF(DAWN.instance, c.RequestAdapterOptions.new({ powerPreference: c.PowerPreference.HighPerformance }).ptr(), cb))
    if (status.value !== c.RequestAdapterStatus.Success.value) throw new Error(`Error requesting adapter: ${status} ${from_wgpu_str(msg)}`)

    const supported_features = new c.SupportedFeatures()
    c.adapterGetFeatures(adapter, supported_features.ptr())
    const supported: c.FeatureName[] = []
    for (let i = 0n; i < supported_features.$featureCount.value; i++) {
      supported.push(new c.FeatureName().loadFromPtr(c.Pointer.new(supported_features.$features.value + i)))
    }
    const features = [c.FeatureName.TimestampQuery, c.FeatureName.ShaderF16].filter((feat) => supported.some((s) => s.value === feat.value))
    const dev_desc = c.DeviceDescriptor.new({ requiredFeatureCount: c.Size.new(BigInt(features.length)), requiredFeatures: c.createArray(features).ptr() })

    const supported_limits = new c.SupportedLimits()
    c.adapterGetLimits(adapter, supported_limits.ptr())
    const limits = c.RequiredLimits.new({ limits: supported_limits.$limits })
    dev_desc.$requiredLimits.set(limits.ptr().value)

    // Requesting a device
    const [dev_status, device, dev_msg] = await _run(c.RequestDeviceCallbackInfo, (cb) => c.adapterRequestDeviceF(adapter, dev_desc.ptr(), cb))
    if (dev_status.value !== c.RequestDeviceStatus.Success.value) throw new Error(`Failed to request device: ${dev_status}] ${from_wgpu_str(dev_msg)}`)

    DAWN.device = device
  }
  override synchronize = async () => {
    const [status] = await _run(c.QueueWorkDoneCallbackInfo2, (cb) => c.queueOnSubmittedWorkDone2(c.deviceGetQueue(DAWN.device), cb))
    if (status.value !== c.QueueWorkDoneStatus.Success.value) throw new Error(`Failed to synchronize: ${status}`)
  }
}
