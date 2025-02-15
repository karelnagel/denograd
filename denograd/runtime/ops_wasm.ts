import { cpu_time_execution, zip } from '../helpers.ts'
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
  override call = cpu_time_execution(async (bufs: Uint8Array[], { global_size = [1, 1, 1], local_size = [1, 1, 1], vals = [] }: ProgramCallArgs, wait = false) => {
    const bytes = bufs.reduce((acc, x) => acc + x.length, 0)
    const memory = new WebAssembly.Memory({ initial: Math.min(65536, Math.ceil(bytes / (64 * 1024)) * 4) })
    const wasmModule = new WebAssembly.Module(this.lib)
    const wasmInstance = new WebAssembly.Instance(wasmModule, { env: { memory } })

    const fn = wasmInstance.exports[this.name] as any

    const mem = new Uint8Array(memory.buffer)

    const offsets: number[] = [0]
    for (const buf of bufs.slice(0, -1)) offsets.push(offsets.at(-1)! + buf.length)
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
      const { buffer } = parsedModule.toBinary({ log: false, write_debug_names: true })
      parsedModule.destroy()
      return buffer
    } catch (e) {
      const name = src.split('(func (export "')[1].split('")')[0]
      Deno.writeTextFileSync(`${name}.ts`, `export const ${name}=\`\n${src}\`\nconst err=\`${e}\``)
      throw new Error(`Failed to compile src: \n${src}\n${e}`)
    }
  }
}
export class WASMAllocator extends Allocator<Uint8Array> {
  _alloc = (size: number, options: BufferSpec) => {
    return new Uint8Array(size)
  }
  _copyin = (dest: Uint8Array, src: MemoryView) => void dest.set(src.bytes)
  _copyout = (dest: MemoryView, src: Uint8Array): void => void dest.set(src.slice(0, dest.byteLength))
  _free = (opaque: Uint8Array, options: BufferSpec) => {
  }
}

export class WASM extends Compiled {
  constructor(device: DeviceType) {
    super(device, new WASMAllocator(), new WASMRenderer(), new WASMCompiler(), WASMProgram)
  }
}
