import { Compiled, Compiler, MallocAllocator, Program } from './allocator.ts'
import { cpuObjdump, cpuTimeExecution, ctypes, isNone, temp } from '../helpers.ts'
import { execSync } from 'node:child_process'
import { ClangRenderer } from '../renderer/cstyle.ts'
import type { DeviceType } from '../device.ts'

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

export class ClangProgram extends Program {
  fxn
  constructor(name: string, lib: Uint8Array) {
    super(name, lib)
    // write to disk so we can load it
    const cachedFile = temp('cachedFile')
    Deno.writeTextFileSync(cachedFile, lib.toString())
    this.fxn = ctypes.CDLL(cachedFile).get(name)
  }
  override call = (bufs: any[], vals: any, wait = false) => cpuTimeExecution(() => this.fxn(...bufs, ...vals), wait)
}

export class ClangDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
