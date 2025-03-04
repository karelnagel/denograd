import { Compiled, Compiler, MallocAllocator, Program, type ProgramCallArgs } from './allocator.ts'
import { bytes_to_string, cpu_objdump } from '../helpers.ts'
import { ClangRenderer } from '../renderer/cstyle.ts'
import type { MemoryView } from '../memoryview.ts'
import { env } from '../env/index.ts'

export class ClangCompiler extends Compiler {
  constructor(cachekey = 'compile_clang', public args: string[] = ['-march=native'], public objdumpTool = 'objdump') {
    super(cachekey)
  }

  override compile = async (src: string): Promise<Uint8Array> => {
    // KAREL: TODO: do without files
    const code = await env.tempFile()
    const bin = await env.tempFile()
    await env.writeTextFile(code, src)

    const args = ['-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', code, '-o', bin]
    const res = await new Deno.Command('clang', { args }).output()
    if (!res.success) throw new Error(`Clang compiling failed, error: ${bytes_to_string(res.stderr)}`)

    const data = await env.readFile(bin)
    await env.remove(code)
    await env.remove(bin)
    return data
  }
  override disassemble = (lib: Uint8Array) => cpu_objdump(lib, this.objdumpTool)
}

export class ClangProgram extends Program {
  file!: string
  fxn?: any
  static override init = async (name: string, lib: Uint8Array) => {
    const res = new ClangProgram(name, lib)
    if (!lib?.length) throw new Error('Lib is empty')
    if (!name) throw new Error("Name can't be undefined")
    res.file = await env.tempFile()
    await env.writeFile(res.file, res.lib)
    return res
  }
  override call = async (bufs: MemoryView[], args: ProgramCallArgs, wait = false) => {
    const vals = args.vals || []
    if (vals.some((x) => x === undefined)) throw new Error(`${vals}`)
    if (!this.fxn) {
      this.fxn = Deno.dlopen(this.file, {
        call: {
          parameters: [...bufs.map(() => 'buffer' as Deno.NativeType), ...vals.map(() => 'i32' as Deno.NativeType)],
          result: 'void',
          name: this.name,
          nonblocking: true,
        },
      })
    }
    const st = performance.now()
    await this.fxn.symbols.call(...bufs.map((x) => x.buffer), ...vals.map((x) => x))
    if (wait) return performance.now() - st
  }
}

export class CLANG extends Compiled {
  constructor(device: string) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
