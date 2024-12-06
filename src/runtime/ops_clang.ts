import { Compiled, Compiler, MallocAllocator } from '../device.ts'
import { type bytes, cpuObjdump, cpuTimeExecution, ctypes, isNone, temp } from '../helpers.ts'
import { execSync } from 'node:child_process'
import { readFileSync, unlinkSync, writeFileSync } from 'node:fs'
import { ClangRenderer } from '../renderer/cstyle.ts'
import { ClangGraph } from './graph/clang.ts'

export class ClangCompiler extends Compiler {
  args
  objdumpTool
  constructor(cachekey = 'compile_clang', args?: string[], objdumpTool = 'objdump') {
    super(cachekey)
    this.args = isNone(args) ? ['-march=native'] : args
    this.objdumpTool = objdumpTool
  }

  override compile = (src: string): bytes => {
    // TODO: remove file write. sadly clang doesn't like the use of /dev/stdout here
    const outputFile = temp('temp_output.so')
    const args = ['clang', '-shared', ...this.args, '-O2', '-Wall', '-Werror', '-x', 'c', '-fPIC', '-ffreestanding', '-nostdlib', '-', '-o', outputFile]
    execSync(args.join(' '), { input: src, stdio: 'pipe' })
    const data = readFileSync(outputFile)
    unlinkSync(outputFile)
    return data
  }
  override disassemble = (lib: bytes) => cpuObjdump(lib, this.objdumpTool)
}
export class ClangProgram {
  name
  lib
  fxn
  constructor(name: string, lib: bytes) {
    this.name = name
    this.lib = lib
    // write to disk so we can load it
    const cachedFile = temp('cachedFile')
    writeFileSync(cachedFile, lib)
    this.fxn = ctypes.CDLL(cachedFile).get(name)
  }
  __call__ = (bufs: any[], vals: any[], wait = false) => cpuTimeExecution(() => this.fxn(...bufs, ...vals), wait)
}

export class ClangDevice extends Compiled {
  constructor(device: string) {
    super(device, MallocAllocator, new ClangRenderer(), new ClangCompiler(), ClangProgram, ClangGraph)
  }
}
