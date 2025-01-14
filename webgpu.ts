import * as _webgpu from 'https://esm.sh/@webgpu/types@0.1.52'

const arrA = new Int32Array([10, -20, 30, 40])
const arrB = new Int32Array([1, 2, -3, 255])

const adapter = await navigator.gpu.requestAdapter()
if (!adapter) throw new Error('Failed to request WebGPU adapter.')

const device = await adapter.requestDevice()

const shaderCode = `
@group(0) @binding(0) var<storage, read> inA : array<i32>;
@group(0) @binding(1) var<storage, read> inB : array<i32>;
@group(0) @binding(2) var<storage, read_write> outC : array<i32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let i = global_id.x;
  outC[i] = inA[i] + inB[i];
}
`

const module = device.createShaderModule({ code: shaderCode })
const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } })

const length = Math.min(arrA.length, arrB.length)

const BUFFER_USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function createBufferForArray(arr: Int32Array): GPUBuffer {
  const buffer = device.createBuffer({
    size: arr.byteLength,
    usage: BUFFER_USAGE,
    mappedAtCreation: true,
  })
  new Int32Array(buffer.getMappedRange()).set(arr)
  buffer.unmap()
  return buffer
}

const bufferA = createBufferForArray(arrA)
const bufferB = createBufferForArray(arrB)

const bufferOut = device.createBuffer({ size: arrA.byteLength, usage: BUFFER_USAGE, mappedAtCreation: false })

const bindGroup = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: bufferA } },
    { binding: 1, resource: { buffer: bufferB } },
    { binding: 2, resource: { buffer: bufferOut } },
  ],
})

const commandEncoder = device.createCommandEncoder()
const passEncoder = commandEncoder.beginComputePass()
passEncoder.setPipeline(pipeline)
passEncoder.setBindGroup(0, bindGroup)
passEncoder.dispatchWorkgroups(Math.ceil(length / 64))
passEncoder.end()

device.queue.submit([commandEncoder.finish()])

const readBuffer = device.createBuffer({ size: arrA.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })

const copyEncoder = device.createCommandEncoder()
copyEncoder.copyBufferToBuffer(bufferOut, 0, readBuffer, 0, arrA.byteLength)
device.queue.submit([copyEncoder.finish()])

await readBuffer.mapAsync(GPUMapMode.READ)
const resultArray = new Int32Array(readBuffer.getMappedRange())

console.log('Result array:', resultArray)
