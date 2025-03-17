import { Compiled, Compiler, MallocAllocator, Program, type ProgramCallArgs } from './allocator.ts'
import { cpu_objdump, perf } from '../helpers.ts'
import { ClangRenderer } from '../renderer/cstyle.ts'
import type { MemoryView } from '../memoryview.ts'
import { env } from '../env/index.ts'
import { dlopen, FFIType } from 'bun:ffi'

export class ClangCompiler extends Compiler {
  constructor(cachekey = 'compile_clang_bun', public args: string[] = ['-march=native'], public objdumpTool = 'objdump') {
    super(cachekey)
  }

  override compile = async (src: string): Promise<Uint8Array> => {
    const code = await env.tempFile()
    const bin = await env.tempFile()
    await env.writeTextFile(code, src)

    const args = ['-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', code, '-o', bin]
    const proc = Bun.spawn(['clang', ...args])
    const exitCode = await proc.exited
    if (exitCode !== 0) throw new Error(`Clang compiling failed, error: ${await new Response(proc.stderr).text()}`)

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
    res.file = await env.tempFile()
    await env.writeFile(res.file, res.lib)
    return res
  }

  override call = async (bufs: MemoryView[], args: ProgramCallArgs, wait = false) => {
    const vals = args.vals || []
    this.fxn = dlopen(this.file, {
      [this.name]: {
        args: [...bufs.map(() => FFIType.ptr), ...vals.map(() => FFIType.i32)],
        returns: FFIType.void,
      },
    })
    const st = performance.now()
    this.fxn.symbols[this.name](...bufs.map((x) => x.buffer), ...vals)
    if (wait) return perf(st)
  }
}

export class CLANG extends Compiled {
  constructor(device: string) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
