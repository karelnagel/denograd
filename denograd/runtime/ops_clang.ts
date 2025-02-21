import { Compiled, Compiler, MallocAllocator, Program, type ProgramCallArgs } from './allocator.ts'
import { cpu_objdump, cpu_time_execution, range, temp } from '../helpers.ts'
import { ClangRenderer } from '../renderer/cstyle.ts'
import type { DeviceType } from '../device.ts'
import type { MemoryView } from '../memoryview.ts'
import { Env } from '../env/index.ts'

export class ClangCompiler extends Compiler {
  constructor(cachekey = 'compile_clang', public args: string[] = ['-march=native'], public objdumpTool = 'objdump') {
    super(cachekey)
  }

  override compile = (src: string): Uint8Array => {
    // KAREL: TODO: try without files
    const code = temp()
    const bin = temp()
    Env.writeTextFileSync(code, src)

    const args = ['-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', code, '-o', bin]
    const res = Env.execSync('clang', { args })
    if (!res.success) throw new Error(`Clang compiling failed, error: ${new TextDecoder().decode(res.stderr)}`)

    const data = Env.readFileSync(bin)
    Env.removeSync(code), Env.removeSync(bin)
    return data
  }
  override disassemble = (lib: Uint8Array) => cpu_objdump(lib, this.objdumpTool)
}

export class ClangProgram extends Program {
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    if (!lib?.length) throw new Error('Lib is empty')
    if (!name) throw new Error("Name can't be undefined")
  }
  override call = cpu_time_execution(async (bufs: MemoryView[], vals: ProgramCallArgs, wait = false) => {
    const file = temp()
    Env.writeFileSync(file, this.lib)
    const fxn = Deno.dlopen(file, {
      call: {
        parameters: range(bufs.length).map(() => 'buffer'),
        result: 'void',
        name: this.name,
        nonblocking: true,
      },
    })
    Env.removeSync(file)
    await fxn.symbols.call(...bufs.map((b) => b.buffer))
    fxn.close()
  })
}

export class CLANG extends Compiled {
  constructor(device: DeviceType) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
