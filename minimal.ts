import { ProgramCallArgs } from './denograd/device.ts'
import { isInt, range, round_up } from './denograd/helpers.ts'

const create_uniform = (wgpu_device: GPUDevice, val: number): GPUBuffer => {
  const buf = wgpu_device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const bytes = new Uint8Array(4)
  if (isInt(val)) new DataView(bytes.buffer).setInt32(0, val, true)
  else new DataView(bytes.buffer).setFloat32(0, val, true)
  wgpu_device.queue.writeBuffer(buf, 0, bytes)
  return buf
}

const code = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn is_nan(v:f32) -> bool { return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0; }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0:array<i32>;
@group(0) @binding(2)var<storage,read_write>data1:array<i32>;
@group(0) @binding(3)var<storage,read_write>data2:array<i32>;
@compute @workgroup_size(1) fn E_(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var val0 = data1[0];
  var val1 = data2[0];
  data0[0] = (val0+val1);
}`
const name = 'E_'

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
if (!adapter) throw new Error('No adapter')
const timestamp_supported = adapter.features.has('timestamp-query')
const dev = await adapter.requestDevice({ requiredFeatures: timestamp_supported ? ['timestamp-query'] : [] })
const prg = dev.createShaderModule({ code })

const call = async (bufs: GPUBuffer[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
  wait = wait && timestamp_supported
  const binding_layouts: GPUBindGroupLayoutEntry[] = [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ...range(bufs.length + vals.length).map((i) => ({ binding: i + 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: i >= bufs.length ? 'uniform' : 'storage' } } satisfies GPUBindGroupLayoutEntry)),
  ]
  const bindings: GPUBindGroupEntry[] = [
    { binding: 0, resource: { buffer: create_uniform(dev, Infinity), offset: 0, size: 4 } },
    ...bufs.map((x, i) => ({ binding: i + 1, resource: { buffer: x } } satisfies GPUBindGroupEntry)),
    ...vals.map((x, i) => ({ binding: bufs.length + i + 1, resource: { buffer: create_uniform(dev, x) } } satisfies GPUBindGroupEntry)),
  ]
  const bind_group_layout = dev.createBindGroupLayout({ entries: binding_layouts })
  const pipeline_layout = dev.createPipelineLayout({ bindGroupLayouts: [bind_group_layout] })
  const bind_group = dev.createBindGroup({ layout: bind_group_layout, entries: bindings })
  const compute_pipeline = dev.createComputePipeline({ layout: pipeline_layout, compute: { module: prg, entryPoint: name } })
  const command_encoder = dev.createCommandEncoder()
  let query_set: GPUQuerySet | undefined, query_buf: GPUBuffer | undefined, timestampWrites: GPUComputePassTimestampWrites | undefined

  if (wait) {
    query_set = dev.createQuerySet({ type: 'timestamp', count: 2 })
    query_buf = dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    timestampWrites = { querySet: query_set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 }
  }
  const compute_pass = command_encoder.beginComputePass({ timestampWrites: wait ? timestampWrites : undefined })
  compute_pass.setPipeline(compute_pipeline)
  compute_pass.setBindGroup(0, bind_group)
  compute_pass.dispatchWorkgroups(global_size[0], global_size[1], global_size[2])
  compute_pass.end()
  if (wait) command_encoder.resolveQuerySet(query_set!, 0, 2, query_buf!, 0)
  dev.queue.submit([command_encoder.finish()])
  if (!wait) return 0

  await query_buf!.mapAsync(GPUMapMode.READ)
  const timestamps = new BigInt64Array(query_buf!.getMappedRange())
  query_buf!.unmap()
  return Number((timestamps[1] - timestamps[0]) / 1_000_000_000n)
}

const alloc = (size: number) => dev.createBuffer({ size: round_up(size, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC })

const copyin = (dest: GPUBuffer, src: Uint8Array) => {
  let padded_src
  if (src.byteLength % 4) {
    padded_src = new Uint8Array(round_up(src.byteLength, 4))
    padded_src.set(src)
  }
  dev.queue.writeBuffer(dest, 0, src.byteLength % 4 ? padded_src! : src)
}

const copyout = async (dest: Uint8Array, src: GPUBuffer) => {
  const size = round_up(dest.byteLength, 4)
  const staging = dev.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

  const encoder = dev.createCommandEncoder()
  encoder.copyBufferToBuffer(src, 0, staging, 0, size)
  dev.queue.submit([encoder.finish()])

  await staging.mapAsync(GPUMapMode.READ)
  dest.set(new Uint8Array(staging.getMappedRange()).subarray(0, dest.byteLength))
  staging.unmap()
}

const in1 = new Uint8Array([2, 0, 0, 0])
const in2 = new Uint8Array([6, 0, 0, 0])
const out = new Uint8Array(4)

const [buf_out, buf_in1, buf_in2] = [out, in1, in2].map((x) => alloc(x.byteLength))

copyin(buf_in1, in1)
copyin(buf_in2, in2)

const res = await call([buf_out, buf_in1, buf_in2], { global_size: [1, 1, 1], local_size: [1, 1, 1], vals: [] }, false)

console.log('Timing result (ns):', res)

await copyout(out, buf_out)
console.log('Out buffer:', out)
