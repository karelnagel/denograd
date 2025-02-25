import { Compiled, Compiler, MallocAllocator, Program, type ProgramCallArgs } from './allocator.ts'
import { bytes_to_string, cpu_objdump, cpu_time_execution, range, temp } from '../helpers.ts'
import { ClangRenderer } from '../renderer/cstyle.ts'
import type { DeviceType } from '../device.ts'
import type { MemoryView } from '../memoryview.ts'

export class ClangCompiler extends Compiler {
  constructor(cachekey = 'compile_clang', public args: string[] = ['-march=native'], public objdumpTool = 'objdump') {
    super(cachekey)
  }

  override compile = async (src: string): Promise<Uint8Array> => {
    // KAREL: TODO: try without files
    const code = temp()
    const bin = temp()
    await Deno.writeTextFile(code, src)

    const args = ['-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', code, '-o', bin]
    const res = await new Deno.Command('clang', { args }).output()
    if (!res.success) throw new Error(`Clang compiling failed, error: ${bytes_to_string(res.stderr)}`)

    const data = await Deno.readFile(bin)
    await Deno.remove(code)
    await Deno.remove(bin)
    return data
  }
  override disassemble = (lib: Uint8Array) => cpu_objdump(lib, this.objdumpTool)
}

export class ClangProgram extends Program {
  file!: string
  fxn?: Deno.DynamicLibrary<{ readonly call: { readonly parameters: 'buffer'[]; readonly result: 'void'; readonly name: string; readonly nonblocking: true } }>
  static override init = async (name: string, lib: Uint8Array) => {
    const res = new ClangProgram(name, lib)
    if (!lib?.length) throw new Error('Lib is empty')
    if (!name) throw new Error("Name can't be undefined")
    res.file = temp()
    await Deno.writeFile(res.file, res.lib)
    return res
  }
  override call = cpu_time_execution(async (bufs: MemoryView[], vals: ProgramCallArgs, wait = false) => {
    if (!this.fxn) {
      this.fxn = Deno.dlopen(this.file, {
        call: {
          parameters: range(bufs.length).map(() => 'buffer'),
          result: 'void',
          name: this.name,
          nonblocking: true,
        },
      })
    }
    await this.fxn.symbols.call(...bufs.map((b) => b.buffer))
  })
}

export class CLANG extends Compiled {
  constructor(device: DeviceType) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
