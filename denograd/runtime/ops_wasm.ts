import { cpu_time_execution, round_up, zip } from '../helpers.ts'
import { Allocator, Compiled, Compiler, Program, type ProgramCallArgs } from './allocator.ts'
import type { BufferSpec, DeviceType } from '../device.ts'
import type { MemoryView } from '../memoryview.ts'
import { WabtModule } from './autogen/wabt.js'
import { WASMRenderer } from '../renderer/wat.ts'

let wabt: any = undefined
WabtModule().then((x) => wabt = x)

class WASMProgram extends Program {
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
  }
  override call = cpu_time_execution(async (bufs: Uint8Array[], { global_size, local_size, vals }: ProgramCallArgs, wait = false) => {
    if (global_size || local_size || vals?.length) throw new Error(`I don't know what to do with these: ${global_size}, ${local_size}, ${vals}`)
    const offsets = bufs.reduce((acc, x) => [...acc, acc.at(-1)! + round_up(x.length, 128)], [0]) // adding 128 because otherwise sometimes v128.store will override other buffer's value
    const memory = new WebAssembly.Memory({ initial: Math.min(65536, Math.ceil((offsets.pop()! + 128) / 65536)) }) // it shouldn't need +128, but otheriwise it will sometimes throw memory access out of bounds
    const mem = new Uint8Array(memory.buffer)

    const wasmModule = new WebAssembly.Module(this.lib)
    const wasmInstance = new WebAssembly.Instance(wasmModule, { env: { memory } })
    const fn = wasmInstance.exports[this.name] as any

    for (const [buf, offset] of zip(bufs, offsets)) mem.set(buf, offset)
    fn(...offsets)
    for (const [buf, offset] of zip(bufs, offsets)) buf.set(mem.slice(offset, offset + buf.length))
  })
}

class WASMCompiler extends Compiler {
  override compile = (src: string) => {
    try {
      const parsedModule = wabt.parseWat('inline.wat', src)
      parsedModule.validate()
      const { buffer } = parsedModule.toBinary()
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
  constructor(device: DeviceType) {
    super(device, new WASMAllocator(), new WASMRenderer(), new WASMCompiler(), WASMProgram)
  }
}
