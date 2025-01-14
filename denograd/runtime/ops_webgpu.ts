import * as _webgpu from 'https://esm.sh/@webgpu/types@0.1.52'
import { bytes_to_string, cpu_time_execution, isInt, range, round_up } from '../helpers.ts'
import { Allocator, BufferSpec, Compiled, Compiler, Program, ProgramCallArgs } from './allocator.ts'
import type { DeviceType } from '../device.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import { MemoryView } from '../memoryview.ts'

// init webgpu
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
if (!adapter) throw new Error('No adapter')
const timestamp_supported = adapter.features.has('timestamp-query')
const wgpu_device = await adapter.requestDevice({ requiredFeatures: timestamp_supported ? ['timestamp-query'] : [] })

const create_uniform = (wgpu_device: GPUDevice, val: number): GPUBuffer => {
  const buf = wgpu_device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const bytes = new Uint8Array(4)
  if (isInt(val)) new DataView(bytes.buffer).setInt32(0, val, true)
  else new DataView(bytes.buffer).setFloat32(0, val, true)
  wgpu_device.queue.writeBuffer(buf, 0, bytes)
  return buf
}

class WebGPUProgram extends Program {
  prg: GPUShaderModule
  dev: GPUDevice
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    this.dev = wgpu_device
    this.prg = this.dev.createShaderModule({ code: bytes_to_string(lib) })
  }
  // TODO handle wait
  override call = cpu_time_execution(async (bufs: GPUBuffer[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    const binding_layouts: GPUBindGroupLayoutEntry[] = [
      { 'binding': 0, 'visibility': GPUShaderStage.COMPUTE, 'buffer': { 'type': 'uniform' } },
      ...range(bufs.length + vals.length).map((i) => ({ 'binding': i + 1, 'visibility': GPUShaderStage.COMPUTE, 'buffer': { 'type': i >= bufs.length ? 'uniform' : 'storage' } } satisfies GPUBindGroupLayoutEntry)),
    ]
    const bindings: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: create_uniform(this.dev, Infinity), offset: 0, size: 4 } },
      ...bufs.map((x, i) => ({ binding: i + 1, resource: { buffer: x, offset: 0, size: x.size } })),
      ...vals.map((x, i) => ({ binding: i + 1, resource: { buffer: create_uniform(this.dev, x), offset: 0, size: 4 } })),
    ]
    const bind_group_layout = this.dev.createBindGroupLayout({ entries: binding_layouts })
    const pipeline_layout = this.dev.createPipelineLayout({ bindGroupLayouts: [bind_group_layout] })
    const bind_group = this.dev.createBindGroup({ layout: bind_group_layout, entries: bindings })
    const compute_pipeline = this.dev.createComputePipeline({ layout: pipeline_layout, compute: { 'module': this.prg, entryPoint: this.name } })
    const command_encoder = this.dev.createCommandEncoder()
    const compute_pass = command_encoder.beginComputePass({})
    compute_pass.setPipeline(compute_pipeline)
    compute_pass.setBindGroup(0, bind_group)
    compute_pass.dispatchWorkgroups(global_size[0], global_size[1], global_size[2]) // x y z
    compute_pass.end()

    this.dev.queue.submit([command_encoder.finish()])
  })
}
// # WebGPU buffers have to be 4-byte aligned
class WebGpuAllocator extends Allocator<GPUBuffer> {
  constructor(public dev: GPUDevice) {
    super()
  }
  _alloc = (size: number, options: BufferSpec) => this.dev.createBuffer({ size: round_up(size, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
  _copyin = (dest: GPUBuffer, src: MemoryView) => {
    let padded_src
    if (src.byteLength % 4) {
      padded_src = new Uint8Array(round_up(src.byteLength, 4))
      padded_src.set(src.toBytes())
    }
    this.dev.queue.writeBuffer(dest, 0, src.byteLength % 4 ? padded_src! : src.toBytes())
  }
  _copyout = async (dest: MemoryView, src: GPUBuffer) => {
    const size = round_up(dest.byteLength, 4)

    const staging = this.dev.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

    const encoder = this.dev.createCommandEncoder()
    encoder.copyBufferToBuffer(src, 0, staging, 0, size)
    this.dev.queue.submit([encoder.finish()])

    await staging.mapAsync(GPUMapMode.READ)
    dest.set(new Uint8Array(staging.getMappedRange()).subarray(0, dest.byteLength))
    staging.unmap()
  }
  _free = (opaque: GPUBuffer, options: BufferSpec) => {
    throw new Error('not implemented')
  }
}

export class WebGpuDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, new WebGpuAllocator(wgpu_device), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
}
