import type { UOp } from '../ops.ts'
import { Renderer } from './index.ts'

export class WASMRenderer extends Renderer {
  override render = (name: string, uops: UOp[]) => {
    return `
(module
  (import "env" "memory" (memory 1))

  (func (export "add_arrays")
    (param $ptrA i32)
    (param $ptrB i32)
    (param $ptrRes i32)
    (local $i i32)

    block $break
      loop $loop
        local.get $i
        i32.const 4
        i32.eq
        br_if $break

        local.get $ptrRes
        local.get $i
        i32.const 2
        i32.shl
        i32.add

        local.get $ptrA
        local.get $i
        i32.const 2
        i32.shl
        i32.add
        i32.load

        local.get $ptrB
        local.get $i
        i32.const 2
        i32.shl
        i32.add
        i32.load

        i32.add
        i32.store

        local.get $i
        i32.const 1
        i32.add
        local.set $i

        br $loop
      end
    end
  )
)
`
  }
}
