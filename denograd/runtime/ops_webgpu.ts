import type * as _webgpu from 'npm:@webgpu/types@0.1.54'
import { bytes_to_string, cpu_time_execution, DEBUG, get_number_env, isInt, memsize_to_str, round_up } from '../helpers.ts'
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

class WebGPUProgram extends Program {
  prg!: GPUShaderModule
  bind_group_layout?: GPUBindGroupLayout
  compute_pipeline?: GPUComputePipeline
  static override init = (name: string, lib: Uint8Array) => {
    const res = new WebGPUProgram(name, lib)
    res.prg = WEBGPU.device.createShaderModule({ code: bytes_to_string(res.lib) })
    return res
  }
  override call = cpu_time_execution(async (bufs: GPUBuffer[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    const isStorage = (i: number) => i < bufs.length && bytes_to_string(this.lib).split('\n').find((x) => x.includes(`binding(${i + 1})`))?.includes('var<storage,read_write>')
    if (!this.bind_group_layout || !this.compute_pipeline) {
      const binding_layouts: GPUBindGroupLayoutEntry[] = [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...[...bufs, ...vals].map<GPUBindGroupLayoutEntry>((_, i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: isStorage(i) ? 'storage' : 'uniform' } })),
      ]

      this.bind_group_layout = WEBGPU.device.createBindGroupLayout({ entries: binding_layouts })
      const pipeline_layout = WEBGPU.device.createPipelineLayout({ bindGroupLayouts: [this.bind_group_layout] })
      this.compute_pipeline = WEBGPU.device.createComputePipeline({ layout: pipeline_layout, compute: { module: this.prg, entryPoint: this.name } })
    }
    const bindings: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: create_uniform(WEBGPU.device, Infinity), offset: 0, size: 4 } },
      ...[...bufs, ...vals].map<GPUBindGroupEntry>((x, i) => (
        typeof x === 'number' ? { binding: i + 1, resource: { buffer: create_uniform(WEBGPU.device, x), offset: 0, size: 4 } } : { binding: i + 1, resource: { buffer: x, offset: 0, size: x.size } }
      )),
    ]
    const bind_group = WEBGPU.device.createBindGroup({ layout: this.bind_group_layout, entries: bindings })
    const encoder = WEBGPU.device.createCommandEncoder()
    const compute_pass = encoder.beginComputePass()
    compute_pass.setPipeline(this.compute_pipeline)
    compute_pass.setBindGroup(0, bind_group)
    compute_pass.dispatchWorkgroups(global_size[0], global_size[1], global_size[2]) // x y z
    compute_pass.end()

    WEBGPU.device.queue.submit([encoder.finish()])
    // TODO: remove 'if' when Deno supports this
    if (typeof Deno === 'undefined') await WEBGPU.device.queue.onSubmittedWorkDone()
  })
}

let allocated = 0, freed = 0
class WebGpuAllocator extends Allocator<GPUBuffer> {
  _alloc = (size: number, options?: BufferSpec) => {
    // WebGPU buffers have to be 4-byte aligned
    const buf = WEBGPU.device.createBuffer({ size: round_up(size, 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })
    allocated += buf.size
    if (DEBUG === 1) console.log({ allocated: memsize_to_str(allocated), freed: memsize_to_str(freed) })
    return buf
  }
  _copyin = (dest: GPUBuffer, src: MemoryView) => WEBGPU.device.queue.writeBuffer(dest, 0, src.bytes)
  _copyout = async (dest: MemoryView, src: GPUBuffer) => {
    const staging = WEBGPU.device.createBuffer({ size: src.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    const encoder = WEBGPU.device.createCommandEncoder()
    encoder.copyBufferToBuffer(src, 0, staging, 0, src.size)
    WEBGPU.device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    dest.set(new Uint8Array(staging.getMappedRange()).slice(0, dest.length))
    staging.destroy()
  }
  _free = (opaque: GPUBuffer, options?: BufferSpec) => {
    freed += opaque.size
    opaque.destroy()
    if (DEBUG === 1) console.log({ allocated: memsize_to_str(allocated), freed: memsize_to_str(freed) })
  }
}

export class WEBGPU extends Compiled {
  static device: GPUDevice
  static adapter: GPUAdapter | null
  constructor(device: DeviceType) {
    super(device, new WebGpuAllocator(), new WGSLRenderer(), new Compiler(), WebGPUProgram)
  }
  static override init = async () => {
    if (!get_number_env('WEBGPU')) return
    WEBGPU.adapter = await navigator.gpu.requestAdapter()
    if (!WEBGPU.adapter) throw new Error('No adapter')
    WEBGPU.device = await WEBGPU.adapter.requestDevice({ requiredFeatures: ['shader-f16'] })
  }
}
