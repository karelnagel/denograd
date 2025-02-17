import type * as _webgpu from 'npm:@webgpu/types@0.1.54'
import { bytes_to_string, cpu_time_execution, isInt, memsize_to_str, round_up } from '../helpers.ts'
import { Allocator, type BufferSpec, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import type { DeviceType } from '../device.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import type { MemoryView } from '../memoryview.ts'

const uniforms: { [key: number]: GPUBuffer } = {}
const create_uniform = (wgpu_device: GPUDevice, val: number): GPUBuffer => {
  if (uniforms[val]) return uniforms[val]
  const buf = wgpu_device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const bytes = new Uint8Array(4)
  if (isInt(val)) new DataView(bytes.buffer).setInt32(0, val, true)
  else new DataView(bytes.buffer).setFloat32(0, val, true)
  wgpu_device.queue.writeBuffer(buf, 0, bytes)
  uniforms[val] = buf
  return buf
}

let device: GPUDevice
let adapter: GPUAdapter | null
const getDevice = async () => {
  adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('No adapter')
  const timestamp_supported = adapter.features.has('timestamp-query')
  const { maxStorageBufferBindingSize, maxBufferSize, maxUniformBufferBindingSize } = adapter.limits
  device = await adapter.requestDevice({
    requiredFeatures: timestamp_supported ? ['timestamp-query'] : [],
    requiredLimits: { maxStorageBufferBindingSize, maxBufferSize, maxUniformBufferBindingSize },
  })
}

getDevice()

class WebGPUProgram extends Program {
  prg: GPUShaderModule
  bind_group_layout?: GPUBindGroupLayout
  compute_pipeline?: GPUComputePipeline
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    this.prg = device.createShaderModule({ code: bytes_to_string(this.lib) })
  }
  override call = cpu_time_execution(async (bufs: GPUBuffer[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    const isStorage = (i: number) => i < bufs.length && bytes_to_string(this.lib).split('\n').find((x) => x.includes(`binding(${i + 1})`))?.includes('var<storage,read_write>')
    if (!this.bind_group_layout || !this.compute_pipeline) {
      const binding_layouts: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[...bufs, ...vals].map<GPUBindGroupLayoutEntry>((_, i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: isStorage(i) ? 'storage' : 'uniform' } })),
      ]

      this.bind_group_layout = device.createBindGroupLayout({ entries: binding_layouts })
      const pipeline_layout = device.createPipelineLayout({ bindGroupLayouts: [this.bind_group_layout] })
      this.compute_pipeline = await device.createComputePipelineAsync({ layout: pipeline_layout, compute: { module: this.prg, entryPoint: this.name } })
    }
    const bindings: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: create_uniform(device, Infinity), offset: 0, size: 4 } },
      ...[...bufs, ...vals].map<GPUBindGroupEntry>((x, i) => (
        typeof x === 'number' ? { binding: i + 1, resource: { buffer: create_uniform(device, x), offset: 0, size: 4 } } : { binding: i + 1, resource: { buffer: x, offset: 0, size: x.size } }
      )),
    ]
    const bind_group = device.createBindGroup({ layout: this.bind_group_layout, entries: bindings })
    const encoder = device.createCommandEncoder()
    const compute_pass = encoder.beginComputePass()
    compute_pass.setPipeline(this.compute_pipeline)
    compute_pass.setBindGroup(0, bind_group)
    compute_pass.dispatchWorkgroups(global_size[0], global_size[1], global_size[2]) // x y z
    compute_pass.end()

    device.queue.submit([encoder.finish()])
    await device.queue.onSubmittedWorkDone()
  })
}
let allocated = 0, freed = 0
// WebGPU buffers have to be 4-byte aligned
class WebGpuAllocator extends Allocator<GPUBuffer> {
  _alloc = (size: number, options?: BufferSpec) => {
    const buf = device.createBuffer({ size: round_up(size, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    allocated += buf.size
    return buf
  }
  _copyin = (dest: GPUBuffer, src: MemoryView) => device.queue.writeBuffer(dest, 0, src.bytes)
  _copyout = async (dest: MemoryView, src: GPUBuffer) => {
    const staging = device.createBuffer({ size: src.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(src, 0, staging, 0, src.size)
    device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    dest.set(new Uint8Array(staging.getMappedRange()).slice(0, dest.length))
    staging.destroy()
  }
  _free = (opaque: GPUBuffer, options?: BufferSpec) => {
    freed += opaque.size
    opaque.destroy()
    console.log({ allocated: memsize_to_str(allocated), freed: memsize_to_str(freed) })
  }
}

export class WEBGPU extends Compiled {
  constructor(device: DeviceType) {
    super(device, new WebGpuAllocator(), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
}
