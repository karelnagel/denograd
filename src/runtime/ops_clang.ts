import { Compiled, Compiler, MallocAllocator, Program } from './allocator.ts'
import { cpuObjdump, cpuTimeExecution, isNone, temp } from '../helpers.ts'
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
    // TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    const outputFile = temp('temp_output.so')
    const args = ['clang', '-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', '-', '-o', outputFile]
    execSync(args.join(' '), { input: src, stdio: 'pipe' })
    const data = Deno.readFileSync(outputFile)
    Deno.removeSync(outputFile)
    return data
  }
  override disassemble = (lib: Uint8Array) => cpuObjdump(lib, this.objdumpTool)
}

type Fxn = Deno.DynamicLibrary<{ readonly call: { readonly parameters: readonly ['buffer', 'buffer']; readonly result: 'buffer'; readonly name: string } }>
export class ClangProgram extends Program {
  fxn: Fxn
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    // write to disk so we can load it
    const cachedFile = temp('cachedFile')
    Deno.writeTextFileSync(cachedFile, lib.toString())
    console.log(`wrote ${cachedFile} fn name: ${name}`)
    this.fxn = Deno.dlopen(cachedFile, {
      'call': {
        parameters: ['buffer', 'buffer'],
        result: 'buffer',
        name,
      } as const,
    })
  }
  override call = (bufs: MemoryView[], vals: any, wait = false) => cpuTimeExecution(() => this.fxn.symbols.call!(bufs[0], bufs[1]), wait)
}

export class ClangDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
