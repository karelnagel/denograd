import * as _webgpu from 'https://esm.sh/@webgpu/types@0.1.52'
import { bytes_to_string, isInt, range, round_up } from '../helpers.ts'
import { Allocator, BufferSpec, Compiled, Compiler, Program, ProgramCallArgs } from './allocator.ts'
import { DeviceType } from '../device.ts'
import { WGSLRenderer } from '../renderer/wgsl.ts'
import { MemoryView } from '../memoryview.ts'

// init webgpu
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
if (!adapter) throw new Error('No adapter')
const timestamp_supported = adapter!.features.has('timestamp-query')
const wgpu_device = await adapter.requestDevice({ requiredFeatures: timestamp_supported ? ['timestamp-query'] : [] })

const create_uniform = (wgpu_device: GPUDevice, val: number): GPUBuffer => {
  const buf = wgpu_device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  if (isInt(val)) {
    const bytes = new Uint8Array(4)
    new DataView(bytes.buffer).setInt32(0, val, true)
    wgpu_device.queue.writeBuffer(buf, 0, bytes)
  } else {
    const bytes = new Uint8Array(4)
    new DataView(bytes.buffer).setFloat32(0, val, true)
    wgpu_device.queue.writeBuffer(buf, 0, bytes)
  }
  return buf
}

class WebGPUProgram extends Program {
  prg: GPUShaderModule
  timestamp_supported: boolean
  dev: GPUDevice
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    this.timestamp_supported = timestamp_supported
    this.dev = wgpu_device
    this.prg = this.dev.createShaderModule({ code: bytes_to_string(lib) })
  }
  override call = async (bufs: GPUBuffer[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    wait = wait && this.timestamp_supported
    let binding_layouts: GPUBindGroupLayoutEntry[] = [
      { 'binding': 0, 'visibility': GPUShaderStage.COMPUTE, 'buffer': { 'type': 'uniform' } },
      ...range(bufs.length + vals.length).map((i) => ({ 'binding': i + 1, 'visibility': GPUShaderStage.COMPUTE, 'buffer': { 'type': i >= bufs.length ? 'uniform' : 'storage' } } satisfies GPUBindGroupLayoutEntry)),
    ]
    let bindings: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: create_uniform(this.dev, Infinity), offset: 0, size: 4 } },
      ...bufs.map((x, i) => ({ binding: i + 1, resource: { buffer: x, offset: 0, size: x.size } })),
      ...vals.map((x, i) => ({ binding: i + 1, resource: { buffer: create_uniform(this.dev, x), offset: 0, size: 4 } })),
    ]
    const bind_group_layout = this.dev.createBindGroupLayout({ entries: binding_layouts })
    const pipeline_layout = this.dev.createPipelineLayout({ bindGroupLayouts: [bind_group_layout] })
    const bind_group = this.dev.createBindGroup({ layout: bind_group_layout, entries: bindings })
    const compute_pipeline = this.dev.createComputePipeline({ layout: pipeline_layout, compute: { 'module': this.prg, entryPoint: this.name } })
    const command_encoder = this.dev.createCommandEncoder()
    let query_set: undefined | GPUQuerySet, query_buf: undefined | GPUBuffer, timestampWrites: undefined | GPUComputePassTimestampWrites
    if (wait) {
      query_set = this.dev.createQuerySet({ type: 'timestamp', count: 2 })
      query_buf = this.dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC })
      timestampWrites = { querySet: query_set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
    }
    const compute_pass = command_encoder.beginComputePass({ timestampWrites: wait ? timestampWrites! : undefined })
    compute_pass.setPipeline(compute_pipeline)
    compute_pass.setBindGroup(0, bind_group, new Uint32Array(999999), 0, 999999) // last 2 not used
    compute_pass.dispatchWorkgroups(global_size[0], global_size[1], global_size[2]) // x y z
    compute_pass.end()
    if (wait) command_encoder.resolveQuerySet(query_set!, 0, 2, query_buf!, 0)

    this.dev.queue.submit([command_encoder.finish()])
    if (!wait) return 0

    await query_buf!.mapAsync(GPUMapMode.READ)
    const timestamps = new BigInt64Array(query_buf!.getMappedRange())
    query_buf!.unmap()
    return Number((timestamps[1] - timestamps[0]) / 1_000_000_000n)
  }
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
  _free = (opaque: MemoryView, options: BufferSpec) => {
    throw new Error('not implemented')
  }
}

export class WebGpuDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, new WebGpuAllocator(wgpu_device), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
}
