import type * as _webgpu from 'https://esm.sh/@webgpu/types@0.1.52'
import { bytes_to_string, cpu_time_execution, isInt, mod, range, round_up } from '../helpers.ts'
import { Allocator, type BufferSpec, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import type { DeviceType } from '../device.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import type { MemoryView } from '../memoryview.ts'

const create_uniform = (wgpu_device: GPUDevice, val: number): GPUBuffer => {
  const buf = wgpu_device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const bytes = new Uint8Array(4)
  if (isInt(val)) new DataView(bytes.buffer).setInt32(0, val, true)
  else new DataView(bytes.buffer).setFloat32(0, val, true)
  wgpu_device.queue.writeBuffer(buf, 0, bytes)
  return buf
}
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
if (!adapter) throw new Error('No adapter')
const timestamp_supported = adapter.features.has('timestamp-query')
const device = await adapter.requestDevice({ requiredFeatures: timestamp_supported ? ['timestamp-query'] : [] })

class WebGPUProgram extends Program {
  prg: GPUShaderModule
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    this.prg = device.createShaderModule({ code: bytes_to_string(this.lib) })
  }
  override call = cpu_time_execution(async (bufs: GPUBuffer[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    const binding_layouts: GPUBindGroupLayoutEntry[] = [
      { 'binding': 0, 'visibility': GPUShaderStage.COMPUTE, 'buffer': { 'type': 'uniform' } },
      ...range(bufs.length + vals.length).map((i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: i >= bufs.length ? 'uniform' : 'storage' } } satisfies GPUBindGroupLayoutEntry)),
    ]

    const bindings: GPUBindGroupEntry[] = [
      { 'binding': 0, 'resource': { 'buffer': create_uniform(device, Infinity), 'offset': 0, 'size': 4 } },
      ...[...bufs, ...vals].entries().map(([i, x]) => ({
        'binding': i + 1,
        'resource': {
          'buffer': i >= bufs.length ? create_uniform(device, x as number) : x as GPUBuffer,
          'offset': 0,
          'size': i >= bufs.length ? 4 : (x as GPUBuffer).size,
        },
      } satisfies GPUBindGroupEntry)),
    ]

    const bind_group_layout = device.createBindGroupLayout({ entries: binding_layouts })
    const pipeline_layout = device.createPipelineLayout({ bindGroupLayouts: [bind_group_layout] })
    const bind_group = device.createBindGroup({ layout: bind_group_layout, entries: bindings })
    const compute_pipeline = device.createComputePipeline({ layout: pipeline_layout, compute: { module: this.prg, entryPoint: this.name } })
    const command_encoder = device.createCommandEncoder()
    const compute_pass = command_encoder.beginComputePass({})
    compute_pass.setPipeline(compute_pipeline)
    compute_pass.setBindGroup(0, bind_group, [])
    compute_pass.dispatchWorkgroups(global_size[0], global_size[1], global_size[2]) // x y z
    compute_pass.end()

    device.queue.submit([command_encoder.finish()])
    await device.queue.onSubmittedWorkDone()
  })
}
// WebGPU buffers have to be 4-byte aligned
class WebGpuAllocator extends Allocator<GPUBuffer> {
  _alloc = (size: number, options?: BufferSpec) => device.createBuffer({ size: round_up(size, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
  _copyin = (dest: GPUBuffer, src: MemoryView) => {
    let padded_src
    if (mod(src.byteLength, 4)) {
      padded_src = new Uint8Array(round_up(src.byteLength, 4))
      padded_src.set(src.toBytes())
    }
    device.queue.writeBuffer(dest, 0, mod(src.byteLength, 4) ? padded_src! : src.toBytes())
  }
  _copyout = async (dest: MemoryView, src: GPUBuffer) => {
    const size = round_up(dest.byteLength, 4)

    const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(src, 0, staging, 0, size)
    device.queue.submit([encoder.finish()])

    await staging.mapAsync(GPUMapMode.READ)
    dest.set(new Uint8Array(staging.getMappedRange()).subarray(0, dest.byteLength))
    staging.unmap()
  }
  _free = (opaque: GPUBuffer, options?: BufferSpec) => opaque.destroy()
}

export class WEBGPU extends Compiled {
  constructor(device: DeviceType) {
    super(device, new WebGpuAllocator(), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
}
