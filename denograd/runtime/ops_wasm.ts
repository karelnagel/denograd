import { cpu_time_execution } from '../helpers.ts'
import { Allocator, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import type { BufferSpec, DeviceType } from '../device.ts'
import type { MemoryView } from '../memoryview.ts'
import wabtInit from 'npm:wabt'
import { WASMRenderer } from '../renderer/wasm.ts'

const wabt = await wabtInit()

class WASMProgram extends Program {
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
  }
  override call = cpu_time_execution(async (bufs: Uint8Array[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    const memory = new WebAssembly.Memory({ initial: 10 })

    const wasmModule = new WebAssembly.Module(this.lib)
    const wasmInstance = new WebAssembly.Instance(wasmModule, { env: { memory } })

    const { add_arrays } = wasmInstance.exports as any

    const mem = new Uint8Array(memory.buffer)

    const res = bufs[0]
    const a = bufs[1]
    const b = bufs[2]

    const RES_OFFSET_BYTES = 0
    const A_OFFSET_BYTES = res.length
    const B_OFFSET_BYTES = res.length + a.length

    mem.set(a, A_OFFSET_BYTES)
    mem.set(b, B_OFFSET_BYTES)

    add_arrays(A_OFFSET_BYTES, B_OFFSET_BYTES, RES_OFFSET_BYTES)

    const result = mem.slice(RES_OFFSET_BYTES, RES_OFFSET_BYTES + res.length)
    res.set(result)
  })
}

class WASMCompiler extends Compiler {
  override compile = (src: string) => {
    const parsedModule = wabt.parseWat('inline.wat', src)
    parsedModule.validate()
    const { buffer } = parsedModule.toBinary({ log: false, write_debug_names: true })
    parsedModule.destroy()
    return buffer
  }
}
export class WASMAllocator extends Allocator<Uint8Array> {
  _alloc = (size: number, options: BufferSpec) => {
    return new Uint8Array(size)
  }
  _copyin = (dest: Uint8Array, src: MemoryView) => void dest.set(src.toBytes())
  _copyout = (dest: MemoryView, src: Uint8Array): void => void dest.set(src)
  _free = (opaque: Uint8Array, options: BufferSpec) => {
  }
}

export class WASM extends Compiled {
  constructor(device: DeviceType) {
    super(device, new WASMAllocator(), new WASMRenderer(), new WASMCompiler(), WASMProgram)
  }
}
