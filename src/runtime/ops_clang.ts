import { Compiled, Compiler, DeviceType, MallocAllocator } from '../device.ts'
import { cpuObjdump, cpuTimeExecution, ctypes, isNone, temp } from '../helpers.ts'
import { execSync } from 'node:child_process'
import { readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { ClangRenderer } from '../renderer/cstyle.ts'

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
    const data = readFileSync(outputFile)
    unlinkSync(outputFile)
    return data
  }
  override disassemble = (lib: Uint8Array) => cpuObjdump(lib, this.objdumpTool)
}
export class ClangProgram {
  name
  lib
  fxn
  constructor(name: string, lib: Uint8Array) {
    this.name = name
    this.lib = lib
    // write to disk so we can load it
    const cachedFile = temp('cachedFile')
    writeFileSync(cachedFile, lib.toString())
    this.fxn = ctypes.CDLL(cachedFile).get(name)
  }
  __call__ = (bufs: any[], vals: any[], wait = false) => cpuTimeExecution(() => this.fxn(...bufs, ...vals), wait)
}

export class ClangDevice extends Compiled {
  constructor(device: DeviceType) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram)
  }
}
