import process from 'node:process'
import { DISK } from '../runtime/ops_disk.ts'
import { JS } from '../runtime/ops_js.ts'
import { WASM } from '../runtime/ops_wasm.ts'
import { WEBGPU } from '../runtime/ops_webgpu.ts'
import { CLANG } from '../runtime/ops_clang.ts'
import { CLOUD } from '../runtime/ops_cloud.ts'
import { NodeEnv } from './node.ts'

export class DenoEnv extends NodeEnv {
  override NAME = 'deno'
  override CPU_DEVICE = 'CLANG'
  override PLATFORM = process.platform
  override DEVICES = { CLANG, WEBGPU, WASM, JS, DISK, CLOUD }
  override dlopen = Deno.dlopen
  override ptr = (buffer: ArrayBuffer) => Deno.UnsafePointer.of(buffer)
}
