import { cpu_time_execution, round_up, zip } from '../helpers.ts'
import { Allocator, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import type { BufferSpec } from '../device.ts'
import type { MemoryView } from '../memoryview.ts'
import { WabtModule } from './autogen/wabt.js'
import { WATRenderer } from '../renderer/wat.ts'

const workerScript = `
self.onmessage = ({ data }) => {
    const mod = new WebAssembly.Module(data.lib)
    const memory = new WebAssembly.Memory({ initial: Math.min(65536, Math.ceil((data.mem.length + 128) / 65536)) })
    const instance = new WebAssembly.Instance(mod, { env: { memory } })

    const wasmMem = new Uint8Array(memory.buffer)
    wasmMem.set(data.mem)

    instance.exports[data.name](...data.offsets)
    self.postMessage({ mem: wasmMem });
}
`
const url = URL.createObjectURL(new Blob([workerScript], { type: 'application/javascript' }))

class WASMProgram extends Program {
  static override init = (name: string, lib: Uint8Array) => new WASMProgram(name, lib)
  override call = cpu_time_execution(async (bufs: Uint8Array[], { global_size, local_size, vals }: ProgramCallArgs, wait = false) => {
    if (global_size || local_size || vals?.length) throw new Error(`I don't know what to do with these: ${global_size}, ${local_size}, ${vals}`)
    const offsets = bufs.reduce((acc, x) => [...acc, acc.at(-1)! + round_up(x.length, 128)], [0]) // rounding up to 128 because otherwise sometimes v128.store will override other buffer's value

    let mem = new Uint8Array(offsets.pop()!)
    for (const [buf, offset] of zip(bufs, offsets)) mem.set(buf, offset)

    const worker = new Worker(url, { type: 'module' })
    worker.postMessage({ mem, offsets, name: this.name, lib: this.lib })
    mem = await new Promise((res) => worker.onmessage = ({ data: { mem } }) => res(mem))
    worker.terminate()

    for (const [buf, offset] of zip(bufs, offsets)) buf.set(mem.slice(offset, offset + buf.length))
  })
}

class WASMCompiler extends Compiler {
  override compile = (src: string) => {
    try {
      const parsedModule = WASM.wabt.parseWat('inline.wat', src)
      parsedModule.validate()
      const { buffer } = parsedModule.toBinary({})
      parsedModule.destroy()
      return buffer
    } catch (e) {
      throw new Error(`Failed to compile src: \n${src}\n${e}`)
    }
  }
}
export class WASMAllocator extends Allocator<Uint8Array> {
  _alloc = (size: number, options: BufferSpec) => new Uint8Array(size)
  _copyin = (dest: Uint8Array, src: MemoryView) => void dest.set(src.bytes)
  _copyout = (dest: MemoryView, src: Uint8Array) => void dest.set(src)
  _free = (opaque: Uint8Array, options: BufferSpec) => {
  }
}

export class WASM extends Compiled {
  static wabt: any
  constructor(device: string) {
    super(device, new WASMAllocator(), new WATRenderer(), new WASMCompiler('wasm'), WASMProgram)
  }
  override init = async () => {
    WASM.wabt = await WabtModule()
  }
}
