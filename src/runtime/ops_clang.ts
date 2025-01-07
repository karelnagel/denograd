import { Compiled, Compiler, MallocAllocator, Program } from './allocator.ts'
import { cpuObjdump, cpuTimeExecution, isNone, range, temp } from '../helpers.ts'
import { execSync } from 'node:child_process'
import { ClangRenderer } from '../renderer/cstyle.ts'
import type { DeviceType } from '../device.ts'
import { MemoryView } from '../memoryview.ts'

export class ClangCompiler extends Compiler {
  args
  objdumpTool
  constructor(cachekey = 'compile_clang', args?: string[], objdumpTool = 'objdump') {
    super(cachekey)
    this.args = isNone(args) ? ['-march=native'] : args
    this.objdumpTool = objdumpTool
  }

  override compile = (src: string): Uint8Array => {
    console.log(src)
    // KAREL: TODO: try without files
    const code = Deno.makeTempFileSync()
    const bin = Deno.makeTempFileSync()
    Deno.writeTextFileSync(code, src)

    const args = ['-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', code, '-o', bin]
    new Deno.Command('clang', { args }).outputSync()

    const data = Deno.readFileSync(bin)
    Deno.removeSync(code), Deno.removeSync(bin)
    return data
  }
  override disassemble = (lib: Uint8Array) => cpuObjdump(lib, this.objdumpTool)
}

export class ClangProgram extends Program {
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    if (!lib?.length) throw new Error('Lib is empty')
    if (!name) throw new Error("Name can't be undefined")
  }
  override call = cpuTimeExecution(async (bufs: MemoryView[], vals: any, wait = false) => {
    console.log({ vals })
    const file = await Deno.makeTempFile()
    await Deno.writeFile(file, this.lib)
    console.log(`Wrote code to ${file}, ${this.name}, ${this.lib}`)
    const fxn = Deno.dlopen(file, {
      call: {
        parameters: range(bufs.length).map(() => 'buffer'),
        result: 'void',
        name: this.name,
        nonblocking: true,
      },
    })
    await Deno.remove(file)
    await fxn.symbols.call(...bufs.map((b) => b.buffer))
    fxn.close()
  })
}

export class ClangDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
